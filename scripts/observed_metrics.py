"""Registry-driven production of observed professional player-season metrics.

The module deliberately separates exact estimator completion from
metric-defined missingness.  A partial table contains only estimators that have
actually completed; the final table is emitted only when every metric supported
for that league has a validated artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Iterable

import numpy as np
import pandas as pd

from advanced_aev import calculate_aev_from_foundation
from advanced_apm import build_apm_point_table, calculate_season_apm
from advanced_psf import calculate_season_psf
from canonical_data import CanonicalSeason, build_canonical_season
from evollve_metrics import (
    calculate_boxscore_evollve_candidates,
    calculate_evollve_candidates,
)
from metric_registry import (
    ADVANCED_METRICS,
    CONVENTIONAL_METRICS,
    EVOLLVE_METRICS,
    METRIC_METADATA,
    expected_player_seasons,
    metrics_for_league,
    seasons_for_league,
)
from metrics import calculate_tier1_metrics
from participation import build_player_actual_sets


DEFAULT_OUTPUT_ROOT = Path('generated') / 'observed_metrics'
KEYS = ['season', 'player_id']
AU_EXACT_EVOLLVE = (
    'Blocked_Rate', 'Srv_Avg', 'Attack_Error_Rate_No_Blocks',
)
CODE_INPUTS = (
    'observed_metrics.py',
    'metric_registry.py', 'metrics.py', 'evollve_metrics.py', 'participation.py',
    'canonical_data.py', 'canonical_sequence.py', 'mlv_adapter.py',
    'lovb_adapter.py', 'au_adapter.py', 'advanced_psf.py',
    'advanced_aev.py', 'advanced_aev.R', 'advanced_apm.py',
    'advanced_apm.R', 'r_bridge.py',
)
SOURCE_RELEASE_BASE = 'https://github.com/awosoga/volleydata/releases/download'


@dataclass(frozen=True)
class ObservedRunResult:
    league: str
    season: int
    output_directory: Path
    partial: pd.DataFrame
    final: pd.DataFrame | None
    completed_metrics: tuple[str, ...]
    pending_metrics: tuple[str, ...]
    validation: pd.DataFrame


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash an adapter-scoped frame without claiming access to raw bytes."""
    header = json.dumps(
        [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()],
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode()
    body = frame.to_csv(
        index=False, na_rep='<NA>', lineterminator='\n', date_format='%Y-%m-%dT%H:%M:%S.%f'
    ).encode()
    return _sha256_bytes(header + b'\n' + body)


def _code_fingerprint(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in CODE_INPUTS:
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _canonical_source_hashes(canonical: CanonicalSeason) -> dict[str, str]:
    frames = {
        'schedule': canonical.schedule,
        'source_actions': canonical.source_actions,
        'source_events': canonical.source_events,
        'source_player_info': canonical.source_player_info,
        'source_player_boxscore': canonical.source_player_boxscore,
    }
    return {name: dataframe_sha256(frame) for name, frame in frames.items()}


def _source_fingerprint(hashes: dict[str, str]) -> str:
    return _sha256_bytes(json.dumps(hashes, sort_keys=True).encode())


def _canonical_name(values: pd.Series):
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return pd.NA
    return sorted(counts[counts.eq(counts.max())].index)[0]


def build_player_season_backbone(
    canonical: CanonicalSeason,
    conventional: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the audited played-player population before metric missingness."""
    conventional = (
        calculate_tier1_metrics(canonical.player_sets, round_digits=None)
        if conventional is None else conventional.copy()
    )
    actual_sets = build_player_actual_sets(
        canonical,
        conventional[['player_id', 'sets_played']].assign(
            season=int(canonical.season)
        ),
    )
    names = pd.concat([
        canonical.player_sets[['player_id', 'player_name']],
        canonical.player_metadata[['player_id', 'player_name']],
        conventional[['player_id', 'player_name']],
    ], ignore_index=True).dropna(subset=['player_id'])
    names['player_id'] = names['player_id'].astype('string')
    names = names.groupby('player_id', as_index=False, sort=True).agg(
        player_name=('player_name', _canonical_name)
    )
    backbone = actual_sets.merge(
        names, on='player_id', how='left', validate='many_to_one'
    )
    backbone.insert(0, 'league', canonical.league)
    backbone['season'] = backbone['season'].astype(int)
    backbone['player_id'] = backbone['player_id'].astype('string')
    backbone = backbone.rename(columns={'actual_sets_played': 'sets_played'})
    backbone = backbone[[
        'league', 'season', 'player_id', 'player_name', 'sets_played',
        'actual_sets_source',
    ]].sort_values(KEYS, kind='stable').reset_index(drop=True)
    if backbone.duplicated(KEYS).any() or backbone['player_id'].isna().any():
        raise ValueError('Observed player-season backbone has invalid identity keys')
    expected = expected_player_seasons(canonical.league, canonical.season)
    if len(backbone) != expected:
        raise ValueError(
            f'{canonical.league.upper()} {canonical.season} has {len(backbone)} '
            f'player-seasons; expected {expected}'
        )
    return backbone


def _normalize_metric_frame(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    *,
    season: int,
) -> pd.DataFrame:
    metrics = tuple(metrics)
    result = frame.copy()
    if 'season' not in result:
        result.insert(0, 'season', int(season))
    if 'player_id' not in result:
        raise ValueError('Metric artifact lacks player_id')
    result['season'] = result['season'].astype(int)
    result['player_id'] = result['player_id'].astype('string')
    missing = set(metrics) - set(result)
    if missing:
        raise ValueError(f'Metric artifact lacks columns: {sorted(missing)}')
    if result.duplicated(KEYS).any():
        raise ValueError('Metric artifact is not unique by player-season')
    return result[[*KEYS, *metrics]].sort_values(KEYS, kind='stable').reset_index(drop=True)


def _artifact_paths(directory: Path, component: str) -> tuple[Path, Path]:
    path = directory / 'metrics' / f'{component}.csv'
    return path, path.with_suffix('.meta.json')


def _write_artifact(
    directory: Path,
    component: str,
    frame: pd.DataFrame,
    metrics: Iterable[str],
    context: dict,
) -> None:
    path, meta_path = _artifact_paths(directory, component)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    metadata = {
        **context,
        'component': component,
        'metrics': list(metrics),
        'artifact_sha256': _sha256_bytes(path.read_bytes()),
        'rows': len(frame),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')


def _read_artifact(
    directory: Path,
    component: str,
    context: dict,
) -> tuple[pd.DataFrame, tuple[str, ...]] | None:
    path, meta_path = _artifact_paths(directory, component)
    if not path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text())
    for key in ('league', 'season', 'source_fingerprint', 'code_fingerprint'):
        if metadata.get(key) != context.get(key):
            return None
    if metadata.get('artifact_sha256') != _sha256_bytes(path.read_bytes()):
        raise ValueError(f'Cached artifact checksum failed: {path}')
    return pd.read_csv(path, dtype={'player_id': 'string'}), tuple(metadata['metrics'])


def _save_supporting_frame(directory: Path, name: str, frame: pd.DataFrame) -> None:
    path = directory / 'sufficient_statistics' / f'{name}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)


def _merge_completed(
    backbone: pd.DataFrame,
    artifacts: list[tuple[pd.DataFrame, tuple[str, ...]]],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    result = backbone.copy()
    completed = []
    for frame, metrics in artifacts:
        normalized = _normalize_metric_frame(
            frame, metrics, season=int(backbone['season'].iloc[0])
        )
        overlap = set(metrics) & set(result)
        if overlap:
            raise ValueError(f'Duplicate metric artifacts: {sorted(overlap)}')
        result = result.merge(normalized, on=KEYS, how='left', validate='one_to_one')
        completed.extend(metrics)
    return result, tuple(completed)


def _validation_table(frame: pd.DataFrame, supported: Iterable[str]) -> pd.DataFrame:
    rows = []
    for metric in supported:
        if metric not in frame:
            continue
        values = pd.to_numeric(frame[metric], errors='coerce')
        finite = values[np.isfinite(values)]
        rows.append({
            'metric': metric,
            'rows': len(frame),
            'finite': int(len(finite)),
            'missing': int(values.isna().sum()),
            'min': float(finite.min()) if len(finite) else np.nan,
            'max': float(finite.max()) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def _r_versions() -> dict[str, str | None]:
    expression = (
        "cat(as.character(getRversion()), '\\n'); "
        "for (p in c('ovlytics','glmnet')) "
        "cat(p, if (requireNamespace(p, quietly=TRUE)) "
        "as.character(packageVersion(p)) else 'NOT_INSTALLED', '\\n')"
    )
    environment = os.environ.copy()
    library = Path('.r-lib/advanced-source-fidelity').resolve()
    environment['R_LIBS_USER'] = str(library)
    try:
        completed = subprocess.run(
            ['Rscript', '-e', expression], check=True, capture_output=True,
            text=True, env=environment, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return {'R': None, 'ovlytics': None, 'glmnet': None}
    lines = completed.stdout.strip().splitlines()
    versions = {'R': lines[0] if lines else None}
    for line in lines[1:]:
        name, _, version = line.partition(' ')
        versions[name] = version.strip() or None
    return versions


def _git_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ['git', 'status', '--short'], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {'base_commit': head, 'working_tree_changes': status}


def _source_urls(league: str, season: int) -> dict[str, str]:
    internal = {'mlv': 'pvf', 'lovb': 'lovb', 'au': 'aupvb'}[league]
    return {
        'pbp': f'{SOURCE_RELEASE_BASE}/{internal}-pbp/{internal}_pbp_{season}.csv',
        'events': f'{SOURCE_RELEASE_BASE}/{internal}-events-log/{internal}_events_log_{season}.csv',
        'player_info': f'{SOURCE_RELEASE_BASE}/{internal}-player-info/{internal}_player_info.csv',
        'player_boxscore': f'{SOURCE_RELEASE_BASE}/{internal}-player-boxscore/{internal}_player_boxscore.csv',
        'schedule': f'{SOURCE_RELEASE_BASE}/{internal}-schedule/{internal}_schedule.csv',
    }


def _write_provenance(
    directory: Path,
    canonical: CanonicalSeason,
    source_hashes: dict[str, str],
    timings: dict[str, float],
) -> None:
    import pyvolleydata

    previous_timings = {}
    existing_path = directory / 'provenance.json'
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if (
            existing.get('source_fingerprint') == _source_fingerprint(source_hashes)
            and existing.get('code', {}).get('code_fingerprint') == _code_fingerprint()
        ):
            previous_timings = existing.get('timings_seconds', {})
    manifest = {
        'league': canonical.league,
        'season': canonical.season,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_urls': _source_urls(canonical.league, canonical.season),
        'source_hash_kind': 'adapter_scoped_dataframe_sha256_not_raw_asset_bytes',
        'source_hashes': source_hashes,
        'source_fingerprint': _source_fingerprint(source_hashes),
        'pyvolleydata_version': getattr(pyvolleydata, '__version__', 'unknown'),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'r_versions': _r_versions(),
        'author_implementation_pins': {
            'ovlytics': {'version': '0.3.3', 'commit': '96d0670d0f9fcc856bffb4b6182314279f4c4a6b'},
            'glmnet': {'version': '2.0-16'},
            'sbgcop_preserved_not_run': {'version': '0.975'},
        },
        'code': {**_git_provenance(), 'code_fingerprint': _code_fingerprint()},
        'timings_seconds': {**previous_timings, **timings},
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'provenance.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    )


def run_observed_metrics(
    league: str,
    season: int,
    *,
    metrics: Iterable[str] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
    canonical: CanonicalSeason | None = None,
) -> ObservedRunResult:
    """Build/cache requested observed metrics for one frozen league-season."""
    league = str(league).lower()
    season = int(season)
    if season not in seasons_for_league(league):
        raise ValueError(f'{league} {season} is outside the frozen analysis scope')
    supported = metrics_for_league(league)
    requested = tuple(supported if metrics is None else dict.fromkeys(metrics))
    unknown = set(requested) - set(supported)
    if unknown:
        raise ValueError(
            f'{league} does not support requested analysis metrics: {sorted(unknown)}'
        )
    timings = {}
    start = time.perf_counter()
    canonical = canonical or build_canonical_season(league, season)
    timings['canonical_build'] = time.perf_counter() - start
    conventional = calculate_tier1_metrics(canonical.player_sets, round_digits=None)
    backbone = build_player_season_backbone(canonical, conventional)
    directory = Path(output_root) / league / str(season)
    directory.mkdir(parents=True, exist_ok=True)
    backbone.to_csv(directory / 'player_season_backbone.csv', index=False)
    source_hashes = _canonical_source_hashes(canonical)
    context = {
        'league': league,
        'season': season,
        'source_fingerprint': _source_fingerprint(source_hashes),
        'code_fingerprint': _code_fingerprint(),
    }

    components = {
        'conventional': tuple(CONVENTIONAL_METRICS),
        'psf': ('PSF',),
        'evollve': tuple(
            AU_EXACT_EVOLLVE if league == 'au' else EVOLLVE_METRICS
        ),
        'aev': ('AEV',),
        'apm': ('APM',),
    }
    artifacts: list[tuple[pd.DataFrame, tuple[str, ...]]] = []
    for component, component_metrics in components.items():
        if not set(component_metrics) & set(supported):
            continue
        cached = None if force else _read_artifact(directory, component, context)
        if cached is not None:
            artifacts.append(cached)
            continue
        if not set(component_metrics) & set(requested):
            continue
        start = time.perf_counter()
        if component == 'conventional':
            values = conventional.assign(season=season)
            _save_supporting_frame(directory, 'conventional_player_sets', canonical.player_sets)
        elif component == 'psf':
            values = calculate_season_psf(canonical)
            _save_supporting_frame(directory, 'psf', values)
        elif component == 'evollve':
            if league == 'au':
                values, statistics = calculate_boxscore_evollve_candidates(canonical)
            else:
                values, statistics, sequence = calculate_evollve_candidates(canonical)
                _save_supporting_frame(directory, 'evollve_sequence_actions', sequence.actions)
                _save_supporting_frame(directory, 'evollve_dig_ancestry', sequence.dig_ancestry)
                _save_supporting_frame(
                    directory, 'evollve_exposure',
                    sequence.player_point_exposure,
                )
            _save_supporting_frame(directory, 'evollve', statistics)
        elif component == 'aev':
            values, attacks, statistics, cells, set_counts = calculate_aev_from_foundation(canonical)
            for name, frame in (
                ('aev_attacks', attacks), ('aev_statistics', statistics),
                ('aev_cells', cells), ('aev_set_counts', set_counts),
            ):
                _save_supporting_frame(directory, name, frame)
        elif component == 'apm':
            points = build_apm_point_table(canonical)
            _save_supporting_frame(directory, 'apm_points', points)
            values = calculate_season_apm(canonical)
            (directory / 'apm_diagnostics.json').write_text(json.dumps({
                key: value for key, value in values.attrs.items()
                if isinstance(value, (str, int, float, bool))
            }, indent=2, sort_keys=True) + '\n')
        else:  # pragma: no cover - exhaustive internal mapping
            raise AssertionError(component)
        metrics_here = components[component]
        normalized = _normalize_metric_frame(values, metrics_here, season=season)
        _write_artifact(directory, component, normalized, metrics_here, context)
        artifacts.append((normalized, metrics_here))
        timings[f'{component}_calculation'] = time.perf_counter() - start

    # Load every valid completed artifact, including work from earlier invocations.
    artifacts = []
    for component in components:
        cached = _read_artifact(directory, component, context)
        if cached is not None:
            artifacts.append(cached)
    partial, completed = _merge_completed(backbone, artifacts)
    completed = tuple(metric for metric in supported if metric in set(completed))
    pending = tuple(metric for metric in supported if metric not in set(completed))
    partial = partial[[
        'league', 'season', 'player_id', 'player_name', 'sets_played',
        'actual_sets_source', *completed,
    ]]
    partial.to_csv(directory / 'observed_metrics_partial.csv', index=False)
    final = None
    if not pending:
        final = partial.copy()
        final.to_csv(directory / 'observed_metrics.csv', index=False)
    validation = _validation_table(partial, completed)
    validation.to_csv(directory / 'validation.csv', index=False)
    status = {
        **context,
        'supported_metrics': list(supported),
        'completed_metrics': list(completed),
        'pending_metrics': list(pending),
        'final_product_complete': not pending,
        'player_season_rows': len(partial),
        'duplicate_player_seasons': int(partial.duplicated(KEYS).sum()),
        'missing_player_ids': int(partial['player_id'].isna().sum()),
    }
    (directory / 'status.json').write_text(
        json.dumps(status, indent=2, sort_keys=True) + '\n'
    )
    _write_provenance(directory, canonical, source_hashes, timings)
    return ObservedRunResult(
        league=league, season=season, output_directory=directory,
        partial=partial, final=final, completed_metrics=completed,
        pending_metrics=pending, validation=validation,
    )


def run_scope(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> list[ObservedRunResult]:
    """Run every frozen league-season through the registry-driven pipeline."""
    return [
        run_observed_metrics(
            league, season, output_root=output_root,
            force=force,
        )
        for league in ('mlv', 'lovb', 'au')
        for season in seasons_for_league(league)
    ]
