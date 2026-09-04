"""Resumable Franks reliability production for published metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Iterable

import numpy as np
import pandas as pd

from advanced_aev import (
    build_ovlytics_aev_plays,
    qualifying_aev_sufficient_statistics,
    run_ovlytics_aev,
)
from advanced_apm import fit_apm
from advanced_psf import (
    SUFFICIENT_COUNT_COLUMNS as PSF_COUNT_COLUMNS,
    build_psf_sufficient_statistics,
    build_serve_events,
    calculate_psf,
)
from au_adapter import build_au_canonical_season_from_frames
from bootstrap import (
    build_team_match_draw_plan,
    draw_multiplicities,
    physical_match_weights,
    sample_additive_blocks,
    sample_player_set_rows,
)
from canonical_data import CanonicalSeason, build_canonical_season
from evollve_metrics import (
    COUNT_COLUMNS as EVOLLVE_COUNT_COLUMNS,
    calculate_boxscore_evollve_player_seasons,
    calculate_evollve_player_seasons,
)
from lovb_adapter import build_lovb_canonical_season_from_frames
from meta_reliability import MetaReliabilityEngine, require_unique_rows
from metric_registry import (
    CONVENTIONAL_METRICS,
    EVOLLVE_METRICS,
    analysis_ready_metrics_for_league,
    expected_player_seasons,
    seasons_for_league,
)
from metrics import (
    apply_reliability_attempt_eligibility,
    calculate_tier1_metrics,
)
from mlv_adapter import build_mlv_canonical_season_from_frames


KEYS = ['season', 'player_id']
DEFAULT_OBSERVED_ROOT = Path('generated/observed_metrics')
DEFAULT_OUTPUT_ROOT = Path('generated/production_reliability')
DEFAULT_SOURCE_CACHE = Path('/private/tmp/phase4_sources')
SOURCE_COMPONENTS = {
    'mlv': ('conventional', 'psf', 'evollve', 'aev', 'apm'),
    'lovb': ('conventional', 'psf', 'evollve', 'aev', 'apm'),
    'au': ('conventional', 'evollve'),
}
CODE_INPUTS = (
    'production_reliability.py', 'bootstrap.py', 'meta_reliability.py',
    'metric_registry.py', 'metrics.py', 'evollve_metrics.py',
    'advanced_psf.py', 'advanced_aev.py', 'advanced_apm.py',
    'advanced_apm.R',
)


@dataclass(frozen=True)
class ReliabilitySeasonAssets:
    league: str
    season: int
    canonical: CanonicalSeason
    schedule: pd.DataFrame
    draw_schedule: pd.DataFrame
    draw_team_map: pd.DataFrame
    observed: pd.DataFrame
    observed_reliability: pd.DataFrame
    psf_statistics: pd.DataFrame
    evollve_statistics: pd.DataFrame
    aev_qualifying_statistics: pd.DataFrame
    aev_set_counts: pd.DataFrame
    aev_plays: pd.DataFrame
    apm_points: pd.DataFrame
    source_fingerprint: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _code_fingerprint(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in CODE_INPUTS:
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_observed_artifacts(
    league: str,
    season: int,
    *,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
) -> tuple[pd.DataFrame, dict]:
    """Validate Phase 6 checksums and return its analysis-ready partial table."""
    league = str(league).lower()
    directory = Path(observed_root) / league / str(int(season))
    status = _load_json(directory / 'status.json')
    ready = analysis_ready_metrics_for_league(league)
    if status.get('completed_metrics') != list(ready):
        raise ValueError(f'{league} {season} completed-metric status is not analysis-ready')
    pending_active = set(status.get('pending_metrics', ())).intersection(ready)
    if pending_active:
        raise ValueError(f'{league} {season} has incomplete active metrics')
    if status.get('player_season_rows') != expected_player_seasons(league, season):
        raise ValueError(f'{league} {season} observed backbone count changed')
    if status.get('duplicate_player_seasons') or status.get('missing_player_ids'):
        raise ValueError(f'{league} {season} observed identity status is invalid')

    for component in SOURCE_COMPONENTS[league]:
        artifact = directory / 'metrics' / f'{component}.csv'
        metadata = _load_json(artifact.with_suffix('.meta.json'))
        if metadata.get('artifact_sha256') != _sha256(artifact):
            raise ValueError(f'Observed artifact checksum failed: {artifact}')
        if metadata.get('source_fingerprint') != status.get('source_fingerprint'):
            raise ValueError(f'Observed source provenance differs: {artifact}')

    observed = pd.read_csv(
        directory / 'observed_metrics_partial.csv', dtype={'player_id': 'string'}
    )
    required = ['league', *KEYS, 'sets_played', *ready]
    if observed.columns.intersection(required).tolist() != required:
        missing = set(required) - set(observed)
        raise ValueError(f'Observed partial table has wrong schema; missing={sorted(missing)}')
    if observed.duplicated(KEYS).any() or observed['player_id'].isna().any():
        raise ValueError('Observed partial table has invalid player-season keys')
    if len(observed) != expected_player_seasons(league, season):
        raise ValueError('Observed partial table has the wrong row count')
    return observed[required].copy(), status


def build_canonical_from_source_cache(
    league: str,
    season: int,
    source_cache: str | Path = DEFAULT_SOURCE_CACHE,
) -> CanonicalSeason:
    """Build through the audited adapter from an injected public-source cache."""
    league = str(league).lower()
    path = Path(source_cache) / f'{league}_{int(season)}.pkl'
    if not path.exists():
        return build_canonical_season(league, int(season))
    with path.open('rb') as handle:
        source = pickle.load(handle)
    arguments = (
        int(season), source['schedule'], source['pbp'], source['events'],
        source['player_info'], source['player_boxscore'],
    )
    builders = {
        'mlv': build_mlv_canonical_season_from_frames,
        'lovb': build_lovb_canonical_season_from_frames,
        'au': build_au_canonical_season_from_frames,
    }
    try:
        return builders[league](*arguments)
    except KeyError as exc:
        raise ValueError(f'Unknown reliability league {league!r}') from exc


def _draw_products(canonical: CanonicalSeason) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use contextual AU squads while retaining source team labels in blocks."""
    schedule = canonical.schedule.copy()
    has_assignments = {
        'home_team_assignment_id', 'away_team_assignment_id'
    }.issubset(schedule)
    home_block = (
        schedule['home_team_assignment_id'] if has_assignments
        else schedule['home_team']
    )
    away_block = (
        schedule['away_team_assignment_id'] if has_assignments
        else schedule['away_team']
    )
    draw_schedule = schedule[['season', 'match_id']].copy()
    draw_schedule['home_team'] = home_block
    draw_schedule['away_team'] = away_block
    home = pd.DataFrame({
        'season': schedule['season'], 'match_id': schedule['match_id'],
        'block_team': home_block, 'team': schedule['home_team'],
    })
    away = pd.DataFrame({
        'season': schedule['season'], 'match_id': schedule['match_id'],
        'block_team': away_block, 'team': schedule['away_team'],
    })
    mapping = pd.concat([home, away], ignore_index=True)
    if mapping.duplicated(['season', 'match_id', 'block_team']).any():
        raise ValueError('Bootstrap team-assignment mapping is ambiguous')
    return draw_schedule, mapping


def _source_team_multiplicities(
    block_multiplicities: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    result = block_multiplicities.rename(columns={'team': 'block_team'}).merge(
        mapping,
        on=['season', 'match_id', 'block_team'],
        how='left',
        validate='one_to_one',
    )
    if result['team'].isna().any():
        raise ValueError('Bootstrap multiplicity lacks a source-team mapping')
    return result[['season', 'match_id', 'team', 'multiplicity']]


def _load_supporting(directory: Path, name: str) -> pd.DataFrame:
    path = directory / 'sufficient_statistics' / f'{name}.pkl'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_pickle(path)


def build_reliability_season_assets(
    league: str,
    season: int,
    *,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
    source_cache: str | Path = DEFAULT_SOURCE_CACHE,
    canonical: CanonicalSeason | None = None,
) -> ReliabilitySeasonAssets:
    """Prepare exact observed values and reconstructable team-match inputs."""
    league = str(league).lower()
    season = int(season)
    observed, status = validate_observed_artifacts(
        league, season, observed_root=observed_root
    )
    canonical = canonical or build_canonical_from_source_cache(
        league, season, source_cache
    )
    if canonical.league != league or canonical.season != season:
        raise ValueError('Canonical source does not match requested league-season')

    observed_reliability = observed.copy()
    conventional = calculate_tier1_metrics(
        canonical.player_sets, round_digits=None
    )
    conventional.insert(0, 'season', season)
    conventional = apply_reliability_attempt_eligibility(
        conventional, canonical.player_sets
    )
    masked = conventional[[*KEYS, *CONVENTIONAL_METRICS]]
    observed_reliability = observed_reliability.drop(
        columns=list(CONVENTIONAL_METRICS)
    ).merge(masked, on=KEYS, how='left', validate='one_to_one')
    observed_reliability = observed_reliability[[
        *observed.columns.drop(list(CONVENTIONAL_METRICS)),
        *CONVENTIONAL_METRICS,
    ]]

    directory = Path(observed_root) / league / str(season)
    psf_statistics = (
        build_psf_sufficient_statistics(build_serve_events(canonical))
        if league in {'mlv', 'lovb'} else pd.DataFrame()
    )
    evollve_statistics = _load_supporting(directory, 'evollve')
    if evollve_statistics.empty:
        raise ValueError(f'{league} {season} lacks Evollve sufficient statistics')

    aev_statistics = _load_supporting(directory, 'aev_statistics')
    aev_cells = _load_supporting(directory, 'aev_cells')
    aev_qualifying = (
        qualifying_aev_sufficient_statistics(aev_statistics, aev_cells)
        if league in {'mlv', 'lovb'} else pd.DataFrame()
    )
    aev_set_counts = _load_supporting(directory, 'aev_set_counts')
    aev_plays = (
        build_ovlytics_aev_plays(canonical)
        if league in {'mlv', 'lovb'} else pd.DataFrame()
    )
    apm_points = _load_supporting(directory, 'apm_points')
    if league in {'mlv', 'lovb'} and (
        aev_statistics.empty or aev_set_counts.empty or apm_points.empty
    ):
        raise ValueError(f'{league} {season} lacks advanced sufficient statistics')

    draw_schedule, draw_team_map = _draw_products(canonical)
    return ReliabilitySeasonAssets(
        league=league,
        season=season,
        canonical=canonical,
        schedule=canonical.schedule.copy(),
        draw_schedule=draw_schedule,
        draw_team_map=draw_team_map,
        observed=observed,
        observed_reliability=observed_reliability,
        psf_statistics=psf_statistics,
        evollve_statistics=evollve_statistics,
        aev_qualifying_statistics=aev_qualifying,
        aev_set_counts=aev_set_counts,
        aev_plays=aev_plays,
        apm_points=apm_points,
        source_fingerprint=str(status['source_fingerprint']),
    )


def _merge_metric_values(
    backbone: pd.DataFrame,
    frames: Iterable[tuple[pd.DataFrame, Iterable[str]]],
    ready: tuple[str, ...],
) -> pd.DataFrame:
    result = backbone[[*KEYS, 'sets_played']].copy()
    result['player_id'] = result['player_id'].astype('string')
    for frame, metrics in frames:
        metrics = tuple(metric for metric in metrics if metric in ready)
        if not metrics:
            continue
        values = frame.copy()
        if 'season' not in values:
            values.insert(0, 'season', int(backbone['season'].iloc[0]))
        values['player_id'] = values['player_id'].astype('string')
        values = values[[*KEYS, *metrics]]
        require_unique_rows(values, KEYS, f"{'/'.join(metrics)} reconstruction")
        result = result.merge(values, on=KEYS, how='left', validate='one_to_one')
    missing = set(ready) - set(result)
    if missing:
        raise ValueError(f'Reconstruction omitted metrics: {sorted(missing)}')
    return result[[*KEYS, 'sets_played', *ready]]


def sample_aev_occurrence_plays(
    assets: ReliabilitySeasonAssets,
    draw_occurrences: pd.DataFrame,
) -> pd.DataFrame:
    """Duplicate author-package attack rows with distinct occurrence IDs."""
    occurrences = draw_occurrences.rename(
        columns={'team': 'block_team', 'source_match_id': 'match_id'}
    ).merge(
        assets.draw_team_map,
        on=['season', 'match_id', 'block_team'],
        how='left',
        validate='many_to_one',
    )
    sampled = assets.aev_plays.merge(
        occurrences[['match_id', 'team', 'occurrence_id']],
        on=['match_id', 'team'],
        how='inner',
        validate='many_to_many',
    )
    sampled['match_id'] = sampled['occurrence_id']
    return sampled.drop(columns='occurrence_id')


def calculate_sampled_aev(
    assets: ReliabilitySeasonAssets,
    sampled_plays: pd.DataFrame,
    sampled_set_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Call pinned ovlytics, then reapply the sampled source setter gate."""
    source_aev = run_ovlytics_aev(sampled_plays)
    if source_aev.empty:
        return pd.DataFrame(columns=['season', 'player_id', 'AEV'])
    source_aev['setter_id'] = source_aev['setter_id'].astype('string')
    source_aev['weighted_aev'] = source_aev['aev'] * source_aev['N_attacks']
    aev = source_aev.groupby('setter_id', as_index=False).agg(
        weighted_aev=('weighted_aev', 'sum'),
        N_attacks=('N_attacks', 'sum'),
    )
    aev['AEV'] = aev['weighted_aev'] / aev['N_attacks']
    gates = sampled_set_counts.groupby('setter_id', as_index=False).agg(
        qualifying_sets=('qualifying_sets', 'sum')
    )
    aev = aev.merge(gates, on='setter_id', how='left')
    aev['qualifying_sets'] = aev['qualifying_sets'].fillna(0)
    aev = aev[aev['qualifying_sets'].ge(100)].rename(
        columns={'setter_id': 'player_id'}
    )
    aev.insert(0, 'season', assets.season)
    return aev


def reconstruct_metrics(
    assets: ReliabilitySeasonAssets,
    block_multiplicities: pd.DataFrame,
    draw_occurrences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Recalculate every analysis-ready metric from one shared season draw."""
    ready = analysis_ready_metrics_for_league(assets.league)
    multiplicities = _source_team_multiplicities(
        block_multiplicities, assets.draw_team_map
    )
    sampled_player_sets = sample_player_set_rows(
        assets.canonical.player_sets, multiplicities, team_column='team_name'
    )
    conventional = calculate_tier1_metrics(sampled_player_sets, round_digits=None)
    conventional.insert(0, 'season', assets.season)
    sampled_sets = conventional[[*KEYS, 'sets_played']]
    backbone = assets.observed[KEYS].merge(
        sampled_sets, on=KEYS, how='left', validate='one_to_one'
    )
    backbone['sets_played'] = backbone['sets_played'].fillna(0)

    frames: list[tuple[pd.DataFrame, Iterable[str]]] = [
        (conventional, CONVENTIONAL_METRICS),
    ]
    if assets.league in {'mlv', 'lovb'}:
        psf = calculate_psf(sample_additive_blocks(
            assets.psf_statistics, multiplicities, PSF_COUNT_COLUMNS
        ))
        frames.append((psf, ('PSF',)))

    evollve_counts = (
        [
            'attack_attempts', 'unblocked_attack_errors', 'blocked_attacks',
            'serves', 'serve_aces', 'service_errors',
        ]
        if assets.league == 'au' else EVOLLVE_COUNT_COLUMNS
    )
    evollve_stats = sample_additive_blocks(
        assets.evollve_statistics, multiplicities, evollve_counts
    )
    evollve = (
        calculate_boxscore_evollve_player_seasons(evollve_stats)
        if assets.league == 'au'
        else calculate_evollve_player_seasons(evollve_stats)
    )
    frames.append((evollve, EVOLLVE_METRICS))

    if assets.league in {'mlv', 'lovb'}:
        aev_sets = sample_additive_blocks(
            assets.aev_set_counts, multiplicities, ['qualifying_sets']
        )
        if draw_occurrences is None:
            raise ValueError('AEV reconstruction requires sampled occurrences')
        sampled_plays = sample_aev_occurrence_plays(assets, draw_occurrences)
        aev = calculate_sampled_aev(assets, sampled_plays, aev_sets)
        frames.append((aev, ('AEV',)))

        physical = physical_match_weights(assets.schedule, multiplicities)
        apm_points = assets.apm_points.merge(
            physical[['season', 'match_id', 'physical_match_weight']],
            on=['season', 'match_id'],
            how='left',
            validate='many_to_one',
        )
        apm = fit_apm(apm_points, sample_weight='physical_match_weight')
        apm.insert(0, 'season', assets.season)
        frames.append((apm, ('APM',)))
    return _merge_metric_values(backbone, frames, ready)


def identity_max_errors(assets: ReliabilitySeasonAssets) -> dict[str, float]:
    """Require identity reconstruction to reproduce Phase 6 observed values."""
    plan = build_team_match_draw_plan(
        assets.draw_schedule, n_boot=1, seed=42, identity=True
    )
    rebuilt = reconstruct_metrics(
        assets,
        draw_multiplicities(plan, 0),
        plan[plan['bootstrap_id'].eq(0)],
    )
    compared = assets.observed.merge(
        rebuilt,
        on=KEYS,
        how='outer',
        suffixes=('_observed', '_identity'),
        validate='one_to_one',
    )
    errors = {}
    for metric in analysis_ready_metrics_for_league(assets.league):
        observed = pd.to_numeric(compared[f'{metric}_observed'], errors='coerce')
        identity = pd.to_numeric(compared[f'{metric}_identity'], errors='coerce')
        if not observed.isna().equals(identity.isna()):
            raise AssertionError(
                f'{assets.league} {assets.season} {metric} identity missingness differs'
            )
        finite = observed.notna()
        error = (
            float(np.max(np.abs(observed[finite] - identity[finite])))
            if finite.any() else 0.0
        )
        tolerance = 1e-8 if metric == 'APM' else 1e-12
        if error > tolerance:
            raise AssertionError(
                f'{assets.league} {assets.season} {metric} identity error '
                f'{error:.3g} exceeds {tolerance:g}'
            )
        errors[metric] = error
    return errors


def _replicate_paths(directory: Path, bootstrap_id: int) -> tuple[Path, Path]:
    path = directory / 'replicates' / f'{bootstrap_id:03d}.pkl'
    return path, path.with_suffix('.meta.json')


def _read_replicate(
    directory: Path,
    bootstrap_id: int,
    context: dict,
) -> pd.DataFrame | None:
    path, metadata_path = _replicate_paths(directory, bootstrap_id)
    if not path.exists() or not metadata_path.exists():
        return None
    metadata = _load_json(metadata_path)
    for key, value in context.items():
        if metadata.get(key) != value:
            return None
    if metadata.get('artifact_sha256') != _sha256(path):
        raise ValueError(f'Reliability replicate checksum failed: {path}')
    return pd.read_pickle(path)


def _write_replicate(
    directory: Path,
    bootstrap_id: int,
    frame: pd.DataFrame,
    context: dict,
    elapsed_seconds: float,
) -> None:
    path, metadata_path = _replicate_paths(directory, bootstrap_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    metadata_path.write_text(json.dumps({
        **context,
        'bootstrap_id': bootstrap_id,
        'artifact_sha256': _sha256(path),
        'elapsed_seconds': elapsed_seconds,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True) + '\n')


def run_season_bootstrap(
    assets: ReliabilitySeasonAssets,
    *,
    n_boot: int = 100,
    seed: int = 42,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run or resume exact metric reconstruction for one league-season."""
    directory = Path(output_root) / assets.league / str(assets.season)
    directory.mkdir(parents=True, exist_ok=True)
    plan = build_team_match_draw_plan(
        assets.draw_schedule, n_boot=n_boot, seed=seed
    )
    plan_path = directory / 'draw_plan.csv'
    if plan_path.exists():
        cached_plan = pd.read_csv(plan_path, dtype={'occurrence_id': 'string'})
        pd.testing.assert_frame_equal(
            cached_plan, plan, check_dtype=False, check_exact=True,
            obj='cached production draw plan',
        )
    else:
        plan.to_csv(plan_path, index=False)

    context = {
        'league': assets.league,
        'season': assets.season,
        'n_boot': int(n_boot),
        'seed': int(seed),
        'source_fingerprint': assets.source_fingerprint,
        'code_fingerprint': _code_fingerprint(),
        'metrics': list(analysis_ready_metrics_for_league(assets.league)),
    }
    outputs = []
    for bootstrap_id in range(n_boot):
        cached = _read_replicate(directory, bootstrap_id, context)
        if cached is not None:
            outputs.append(cached)
            continue
        start = time.perf_counter()
        values = reconstruct_metrics(
            assets,
            draw_multiplicities(plan, bootstrap_id),
            plan[plan['bootstrap_id'].eq(bootstrap_id)],
        )
        values['bootstrap_id'] = bootstrap_id
        elapsed = time.perf_counter() - start
        _write_replicate(
            directory, bootstrap_id, values, context, elapsed
        )
        outputs.append(values)
    errors = identity_max_errors(assets)
    pd.DataFrame([
        {'metric': metric, 'max_absolute_error': error}
        for metric, error in errors.items()
    ]).to_csv(directory / 'identity_errors.csv', index=False)
    return pd.concat(outputs, ignore_index=True), plan, errors


def finalize_league(
    assets_by_season: dict[int, ReliabilitySeasonAssets],
    replicates_by_season: dict[int, pd.DataFrame],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, pd.DataFrame]:
    """Apply the frozen Franks estimators within one league."""
    if not assets_by_season:
        raise ValueError('No league seasons were supplied')
    league = next(iter(assets_by_season.values())).league
    metrics = list(analysis_ready_metrics_for_league(league))
    observed = pd.concat(
        [assets_by_season[s].observed_reliability for s in sorted(assets_by_season)],
        ignore_index=True,
    )
    replicates = pd.concat(
        [replicates_by_season[s] for s in sorted(replicates_by_season)],
        ignore_index=True,
    )
    require_unique_rows(observed, KEYS, f'{league} observed reliability table')
    require_unique_rows(
        replicates, [*KEYS, 'bootstrap_id'], f'{league} reliability replicates'
    )
    engine = MetaReliabilityEngine(metrics, min_sets=20, seed=42)
    bv, coverage = engine.player_season_bootstrap_variance(replicates, observed)
    results = engine.extract_meta_metrics(observed, bv)
    drift = engine.league_drift_check(observed)

    bv_long = bv.reset_index().melt(
        id_vars=KEYS, value_vars=metrics,
        var_name='metric', value_name='bootstrap_variance',
    )
    diagnostics = coverage.merge(
        bv_long, on=[*KEYS, 'metric'], how='left', validate='one_to_one'
    )
    sets = observed[[*KEYS, 'sets_played']]
    diagnostics = diagnostics.merge(sets, on=KEYS, how='left', validate='many_to_one')
    diagnostics['observed_set_eligible'] = diagnostics['sets_played'].ge(20)

    directory = Path(output_root) / league
    directory.mkdir(parents=True, exist_ok=True)
    observed.to_csv(directory / 'observed_reliability_population.csv', index=False)
    results.to_csv(directory / 'reliability_results.csv', index=False)
    drift.to_csv(directory / 'league_drift.csv', index=False)
    coverage.to_csv(directory / 'bootstrap_coverage.csv', index=False)
    bv.reset_index().to_csv(directory / 'bootstrap_variance.csv', index=False)
    diagnostics.to_csv(directory / 'player_metric_diagnostics.csv', index=False)
    return {
        'observed': observed,
        'replicates': replicates,
        'bootstrap_variance': bv.reset_index(),
        'coverage': coverage,
        'results': results,
        'drift': drift,
        'diagnostics': diagnostics,
    }


def run_league(
    league: str,
    *,
    n_boot: int = 100,
    seed: int = 42,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    source_cache: str | Path = DEFAULT_SOURCE_CACHE,
) -> dict[str, pd.DataFrame]:
    """Prepare, resume, and finalize every frozen season in one league."""
    assets = {}
    replicates = {}
    for season in seasons_for_league(league):
        print(f'Preparing {league.upper()} {season} reliability assets...', flush=True)
        asset = build_reliability_season_assets(
            league, season, observed_root=observed_root,
            source_cache=source_cache,
        )
        assets[season] = asset
        values, _, _ = run_season_bootstrap(
            asset, n_boot=n_boot, seed=seed, output_root=output_root
        )
        replicates[season] = values
    return finalize_league(assets, replicates, output_root=output_root)
