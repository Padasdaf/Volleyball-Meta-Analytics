"""Registry-driven production independence analysis for professional leagues.

This module prepares the unfiltered Phase 6 observed player-season population
and delegates all statistical calculations to the already-audited
``meta_independence.R`` implementation.  It deliberately does not consume any
reliability-population artifact or apply reliability participation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from metric_registry import (
    ANALYSIS_SEASONS_BY_LEAGUE,
    analysis_ready_metrics_for_league,
    expected_player_seasons,
    seasons_for_league,
)
from production_reliability import (
    DEFAULT_OBSERVED_ROOT,
    validate_observed_artifacts,
)
from setup_r_environment import ensure_sbgcop


DEFAULT_OUTPUT_ROOT = Path('generated') / 'production_independence'
PRIMARY_SEED = 42
PRIMARY_NSAMP = 5000
PRIMARY_ODENS = 5
REQUIRED_R_OUTPUTS = (
    'input_population.csv',
    'input_missingness.csv',
    'latent_correlation.csv',
    'latent_correlation_validation.csv',
    'C_posterior_samples.rds',
    'mcmc_diagnostics.csv',
    'mcmc_startup_sensitivity.csv',
    'independence_scores.csv',
    'independence_curves.csv',
    'pca_variance.csv',
    'pca_loadings.csv',
    'analysis_metadata.csv',
)


@dataclass(frozen=True)
class IndependenceRunResult:
    league: str
    output_directory: Path
    player_seasons: int
    metric_count: int
    validation: dict[str, float | int | str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_setting(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype='string')
    return dict(zip(frame['setting'], frame['value']))


def build_league_independence_input(
    league: str,
    *,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> pd.DataFrame:
    """Write one unfiltered, registry-exact pooled matrix for ``league``."""
    league = str(league).lower()
    seasons = seasons_for_league(league)
    metrics = analysis_ready_metrics_for_league(league)
    frames = []
    for season in seasons:
        observed, _ = validate_observed_artifacts(
            league, season, observed_root=observed_root
        )
        if observed['season'].nunique() != 1 or int(observed['season'].iloc[0]) != season:
            raise ValueError(f'{league} {season} observed artifact has wrong season')
        if observed['league'].nunique() != 1 or observed['league'].iloc[0] != league:
            raise ValueError(f'{league} {season} observed artifact has wrong league')
        metric_columns = tuple(
            column for column in observed.columns
            if column not in {
                'league', 'season', 'player_id', 'player_name', 'sets_played',
                'actual_sets_source',
            }
        )
        if metric_columns != metrics:
            raise ValueError(
                f'{league} {season} metric columns differ from analysis-ready registry'
            )
        if len(observed) != expected_player_seasons(league, season):
            raise ValueError(f'{league} {season} player-season count changed')
        frames.append(observed)

    pooled = pd.concat(frames, ignore_index=True)
    keys = ['season', 'player_id']
    if pooled.duplicated(keys).any():
        raise ValueError(f'{league} independence input has duplicate player-seasons')
    if pooled[keys].isna().any().any():
        raise ValueError(f'{league} independence input has missing identity keys')
    expected_rows = sum(expected_player_seasons(league, season) for season in seasons)
    if len(pooled) != expected_rows:
        raise ValueError(f'{league} pooled population is {len(pooled)}; expected {expected_rows}')

    analysis = pooled[keys + list(metrics)].copy()
    analysis['player_id'] = analysis['player_id'].astype('string')
    analysis = analysis.sort_values(keys, kind='stable').reset_index(drop=True)
    output_directory = Path(output_root) / league
    output_directory.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(
        output_directory / 'independence_input.csv', index=False, na_rep='NA'
    )
    return analysis


def _run_authoritative_r(
    input_path: Path,
    output_directory: Path,
    *,
    seed: int = PRIMARY_SEED,
    nsamp: int = PRIMARY_NSAMP,
    odens: int = PRIMARY_ODENS,
) -> None:
    if (seed, nsamp, odens) != (PRIMARY_SEED, PRIMARY_NSAMP, PRIMARY_ODENS):
        raise ValueError('Production independence settings are frozen at 42/5000/5')
    rscript = shutil.which('Rscript')
    if rscript is None:
        raise RuntimeError('Rscript is required for production independence')
    r_program = Path(__file__).resolve().with_name('meta_independence.R')
    library = ensure_sbgcop('0.975')
    env = os.environ.copy()
    env.update({
        'SBGCOP_EXPECTED_VERSION': '0.975',
        'SBGCOP_LIBRARY': str(library),
        'SBGCOP_SEED': str(seed),
        'SBGCOP_NSAMP': str(nsamp),
        'SBGCOP_ODENS': str(odens),
        'SBGCOP_PLUGIN_MODE': 'default',
        'INDEPENDENCE_MAKE_FIGURES': 'false',
        'INDEPENDENCE_FIGURE_DIR': str(output_directory.resolve()),
    })
    subprocess.run(
        [rscript, str(r_program), str(input_path.resolve()), str(output_directory.resolve())],
        check=True,
        env=env,
    )


def validate_independence_outputs(
    league: str,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, float | int | str]:
    """Validate every numerical artifact emitted by the authoritative R run."""
    league = str(league).lower()
    output_directory = Path(output_root) / league
    input_path = output_directory / 'independence_input.csv'
    missing_files = [
        name for name in REQUIRED_R_OUTPUTS
        if not (output_directory / name).is_file()
    ]
    if missing_files:
        raise ValueError(f'{league} R outputs are incomplete: {missing_files}')

    metrics = analysis_ready_metrics_for_league(league)
    seasons = seasons_for_league(league)
    expected_rows = sum(expected_player_seasons(league, season) for season in seasons)
    analysis = pd.read_csv(input_path, dtype={'player_id': 'string'})
    if analysis.columns.tolist() != ['season', 'player_id', *metrics]:
        raise ValueError(f'{league} independence input schema changed')
    if len(analysis) != expected_rows or analysis.duplicated(['season', 'player_id']).any():
        raise ValueError(f'{league} independence input population changed')
    if analysis['player_id'].isna().any():
        raise ValueError(f'{league} independence input has missing player IDs')

    missingness = pd.read_csv(output_directory / 'input_missingness.csv')
    if missingness['metric'].tolist() != list(metrics):
        raise ValueError(f'{league} missingness metric order changed')
    expected_observed = analysis[list(metrics)].notna().sum().to_numpy()
    expected_unique = analysis[list(metrics)].nunique(dropna=True).to_numpy()
    if not np.array_equal(missingness['observed'].to_numpy(), expected_observed):
        raise ValueError(f'{league} missingness observed counts disagree with input')
    if not np.array_equal(missingness['unique_observed'].to_numpy(), expected_unique):
        raise ValueError(f'{league} missingness unique counts disagree with input')

    latent = pd.read_csv(output_directory / 'latent_correlation.csv')
    if latent.columns.tolist() != ['metric', *metrics] or latent['metric'].tolist() != list(metrics):
        raise ValueError(f'{league} latent C labels changed')
    C = latent[list(metrics)].to_numpy(dtype=float)
    if C.shape != (len(metrics), len(metrics)) or not np.isfinite(C).all():
        raise ValueError(f'{league} latent C is malformed')
    symmetry_error = float(np.max(np.abs(C - C.T)))
    diagonal_error = float(np.max(np.abs(np.diag(C) - 1)))
    eigenvalues = np.linalg.eigvalsh(C)
    if symmetry_error > 1e-10 or diagonal_error > 1e-10:
        raise ValueError(f'{league} latent C violates correlation-matrix contracts')
    if float(eigenvalues.min()) < -1e-10:
        raise ValueError(f'{league} latent C is materially non-PSD')

    scores = pd.read_csv(output_directory / 'independence_scores.csv')
    if set(scores['metric']) != set(metrics) or len(scores) != len(metrics):
        raise ValueError(f'{league} independence scores do not cover registry metrics')
    if not np.isfinite(scores['independence']).all():
        raise ValueError(f'{league} independence scores are non-finite')

    curves = pd.read_csv(output_directory / 'independence_curves.csv')
    for metric in metrics:
        curve = curves.loc[curves['target_metric'].eq(metric)]
        expected_sizes = list(range(len(metrics) - 1, -1, -1))
        if curve['conditioning_size'].tolist() != expected_sizes:
            raise ValueError(f'{league} {metric} greedy curve has wrong sizes')
        values = curve['independence'].to_numpy(dtype=float)
        if abs(values[-1] - 1) > 1e-12 or np.any(np.diff(values) < -1e-8):
            raise ValueError(f'{league} {metric} greedy curve violates contracts')

    pca = pd.read_csv(output_directory / 'pca_variance.csv')
    if len(pca) != len(metrics):
        raise ValueError(f'{league} PCA has wrong dimension')
    if abs(float(pca['eigenvalue'].sum()) - len(metrics)) > 1e-8:
        raise ValueError(f'{league} PCA eigenvalues do not sum to metric count')
    if abs(float(pca['cumulative_variance'].iloc[-1]) - 1) > 1e-10:
        raise ValueError(f'{league} PCA cumulative variance does not end at one')

    settings = _read_setting(output_directory / 'analysis_metadata.csv')
    expected_settings = {
        'sbgcop_version': '0.975', 'seed': '42', 'nsamp': '5000',
        'odens': '5', 'saved_samples': '1000',
        'player_seasons': str(expected_rows), 'metrics': str(len(metrics)),
        'plugin_mode': 'default',
    }
    for key, value in expected_settings.items():
        if settings.get(key) != value:
            raise ValueError(f'{league} metadata {key}={settings.get(key)!r}; expected {value!r}')

    diagnostics = pd.read_csv(output_directory / 'mcmc_diagnostics.csv')
    diag = dict(zip(diagnostics['statistic'], diagnostics['value']))
    if int(diag['saved_samples']) != 1000:
        raise ValueError(f'{league} did not save 1000 posterior samples')
    sensitivity = pd.read_csv(output_directory / 'mcmc_startup_sensitivity.csv')
    if sensitivity['discarded_fraction'].tolist() != [0.0, 0.1, 0.2]:
        raise ValueError(f'{league} startup sensitivity grid changed')
    if not np.isfinite(sensitivity.select_dtypes(include='number')).all().all():
        raise ValueError(f'{league} startup sensitivity is non-finite')

    thresholds = {
        threshold: int(pca.loc[pca['cumulative_variance'].ge(threshold), 'component'].iloc[0])
        for threshold in (0.8, 0.9, 0.95)
    }
    validation = {
        'player_seasons': expected_rows,
        'metrics': len(metrics),
        'saved_samples': int(diag['saved_samples']),
        'min_effective_sample_size': float(diag['min_effective_sample_size']),
        'median_effective_sample_size': float(diag['median_effective_sample_size']),
        'max_effective_sample_size': float(diag['max_effective_sample_size']),
        'max_first_vs_second_half_C_difference': float(
            diag['max_first_vs_second_half_C_difference']
        ),
        'symmetry_error': symmetry_error,
        'diagonal_error': diagonal_error,
        'min_eigenvalue': float(eigenvalues.min()),
        'max_eigenvalue': float(eigenvalues.max()),
        'condition_number': float(np.linalg.cond(C)),
        'pcs_80': thresholds[0.8],
        'pcs_90': thresholds[0.9],
        'pcs_95': thresholds[0.95],
        'input_sha256': _sha256(input_path),
        'posterior_sha256': _sha256(output_directory / 'C_posterior_samples.rds'),
    }
    (output_directory / 'production_validation.json').write_text(
        json.dumps(validation, indent=2, sort_keys=True) + '\n'
    )
    return validation


def run_league_independence(
    league: str,
    *,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    prepare_only: bool = False,
) -> IndependenceRunResult:
    league = str(league).lower()
    analysis = build_league_independence_input(
        league, observed_root=observed_root, output_root=output_root
    )
    output_directory = Path(output_root) / league
    if prepare_only:
        validation: dict[str, float | int | str] = {
            'status': 'INPUT_PREPARED',
            'input_sha256': _sha256(output_directory / 'independence_input.csv'),
        }
    else:
        _run_authoritative_r(
            output_directory / 'independence_input.csv', output_directory
        )
        validation = validate_independence_outputs(
            league, output_root=output_root
        )
    return IndependenceRunResult(
        league=league,
        output_directory=output_directory,
        player_seasons=len(analysis),
        metric_count=len(analysis.columns) - 2,
        validation=validation,
    )


def run_scope(
    *,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    prepare_only: bool = False,
) -> list[IndependenceRunResult]:
    return [
        run_league_independence(
            league,
            observed_root=observed_root,
            output_root=output_root,
            prepare_only=prepare_only,
        )
        for league in ANALYSIS_SEASONS_BY_LEAGUE
    ]
