"""GLMNET volleyball Adjusted Plus-Minus from complete public lineups.

Hass and Craig (2018) model point outcomes with penalized logistic player
effects and report the effect on an adjusted plus/minus per 50-play scale:

    APM_j = 50 * logistic(beta_j) - 25.

Hass and Craig's presence matrix is binary: a focal-team player is 1 when on
court, not a signed league-wide home-minus-away column.  Public MLV therefore
contributes two team-perspective observations per retained physical point, with
the focal six players marked 1 and an explicit +/- home-context column.  This
is the narrowest league-wide extension of their single-team application.  The
ridge fit must be repeated after resampling; coefficients are not sufficient
statistics.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from canonical_data import (
    CanonicalSeason,
    RALLY_KEY,
    require_columns as _require_columns,
)
from mlv_adapter import build_mlv_canonical_season
from r_bridge import advanced_r_library, run_r_csv_exchange


REPOSITORY_ROOT = Path(__file__).resolve().parent
APM_R_SCRIPT = REPOSITORY_ROOT / 'advanced_apm.R'
DEFAULT_ADVANCED_R_LIBRARY = (
    REPOSITORY_ROOT / '.r-lib' / 'advanced-source-fidelity'
)


def build_apm_point_table(foundation: CanonicalSeason) -> pd.DataFrame:
    """Return one retained official point with two complete six-player lineups."""
    lineup_columns = [
        *(f'home_p{i}_player_id' for i in range(1, 7)),
        *(f'away_p{i}_player_id' for i in range(1, 7)),
    ]
    _require_columns(
        foundation.lineups,
        [*RALLY_KEY, 'lineup_complete', *lineup_columns],
        'lineup table',
    )
    _require_columns(
        foundation.rallies,
        [*RALLY_KEY, 'home_team', 'away_team', 'point_winner_team'],
        'rally table',
    )
    points = foundation.rallies[
        [*RALLY_KEY, 'home_team', 'away_team', 'point_winner_team']
    ].merge(
        foundation.lineups[
            [*RALLY_KEY, 'lineup_complete', *lineup_columns]
        ],
        on=RALLY_KEY,
        validate='one_to_one',
    )
    points['retained_for_apm'] = points['lineup_complete'].eq(True)
    points['apm_exclusion_reason'] = pd.Series(
        pd.NA, index=points.index, dtype='string'
    )
    points.loc[
        ~points['retained_for_apm'], 'apm_exclusion_reason'
    ] = 'incomplete_lineup'

    home_lineups = []
    away_lineups = []
    for row in points.itertuples(index=False):
        home = tuple(str(getattr(row, f'home_p{i}_player_id')) for i in range(1, 7))
        away = tuple(str(getattr(row, f'away_p{i}_player_id')) for i in range(1, 7))
        if row.retained_for_apm:
            if any(value in ('<NA>', 'nan', 'None') for value in (*home, *away)):
                raise ValueError("A complete APM lineup contains a missing player")
            if len(set(home)) != 6 or len(set(away)) != 6:
                raise ValueError("A complete APM lineup does not contain six unique players")
            if set(home) & set(away):
                raise ValueError("A player appears for both teams on one point")
        home_lineups.append(home)
        away_lineups.append(away)
    points['home_lineup'] = home_lineups
    points['away_lineup'] = away_lineups
    winner_valid = (
        points['point_winner_team'].eq(points['home_team'])
        | points['point_winner_team'].eq(points['away_team'])
    )
    if not winner_valid.all():
        raise ValueError("APM point winner is not one of the scheduled teams")
    points['home_win'] = points['point_winner_team'].eq(
        points['home_team']
    ).astype(int)
    return points[[
        *RALLY_KEY, 'home_team', 'away_team', 'point_winner_team',
        'home_win', 'home_lineup', 'away_lineup', 'retained_for_apm',
        'apm_exclusion_reason',
    ]]


def build_apm_design(
    point_table: pd.DataFrame,
) -> tuple[csr_matrix, np.ndarray, list[str], np.ndarray]:
    """Construct the paper's binary focal-team presence design.

    Each physical point yields complementary home- and away-team observations.
    This retains the paper's response (whether the focal team won the play),
    0/1 player-presence convention, and home-court context while extending its
    single-team dataset to a closed professional league.
    """
    _require_columns(
        point_table,
        ['home_lineup', 'away_lineup', 'home_win', 'retained_for_apm'],
        'APM point table',
    )
    retained = point_table[point_table['retained_for_apm'].eq(True)].copy()
    if retained.empty:
        raise ValueError("No complete-lineup points are available for APM")
    players = sorted({
        str(player)
        for lineup in pd.concat([
            retained['home_lineup'], retained['away_lineup']
        ])
        for player in lineup
    })
    player_index = {player: index for index, player in enumerate(players)}
    rows, columns, values = [], [], []
    outcome = []
    home_context = []
    for row_index, row in enumerate(retained.itertuples(index=False)):
        home_row = 2 * row_index
        away_row = home_row + 1
        for player in row.home_lineup:
            rows.append(home_row)
            columns.append(player_index[str(player)])
            values.append(1.0)
        for player in row.away_lineup:
            rows.append(away_row)
            columns.append(player_index[str(player)])
            values.append(1.0)
        outcome.extend([float(row.home_win), float(1 - row.home_win)])
        home_context.extend([1.0, -1.0])
    design = csr_matrix(
        (values, (rows, columns)), shape=(2 * len(retained), len(players))
    )
    return (
        design,
        np.asarray(outcome, dtype=float),
        players,
        np.asarray(home_context, dtype=float),
    )


def build_apm_cv_fold_ids(
    physical_point_count: int,
    *,
    n_folds: int = 10,
) -> np.ndarray:
    """Assign both focal-team views of one physical point to the same fold."""
    if physical_point_count < 1:
        raise ValueError("APM CV requires at least one physical point")
    if n_folds < 2:
        raise ValueError("APM CV requires at least two folds")
    physical_folds = np.arange(physical_point_count, dtype=int) % n_folds + 1
    return np.repeat(physical_folds, 2)


def fit_apm(
    point_table: pd.DataFrame,
    *,
    sample_weight: Iterable[float] | str | None = None,
    r_library: str | Path | None = None,
) -> pd.DataFrame:
    """Fit Hass-Craig APM with pinned R GLMNET Ridge.

    The paper does not publish the precise GLMNET call or lambda selection.
    This adapter therefore uses contemporaneous GLMNET 2.0-16 and its
    10-fold deviance CV, standardization, and ``lambda.min``.  Home context is
    an unpenalized covariate; player columns receive the Ridge penalty.  Every
    one of those tuning choices is recorded as an unavoidable specification
    assumption, not represented as unpublished source behavior.
    """
    weighted = point_table.copy()
    if sample_weight is None:
        weighted['_apm_sample_weight'] = 1.0
    elif isinstance(sample_weight, str):
        if sample_weight not in weighted:
            raise ValueError(f"APM sample-weight column is missing: {sample_weight}")
        weighted['_apm_sample_weight'] = pd.to_numeric(
            weighted[sample_weight], errors='coerce'
        )
    else:
        values = np.asarray(list(sample_weight), dtype=float)
        if len(values) != len(weighted):
            raise ValueError("APM sample weights must match point-table length")
        weighted['_apm_sample_weight'] = values
    weights = weighted['_apm_sample_weight']
    if weights.isna().any() or (~np.isfinite(weights)).any() or weights.lt(0).any():
        raise ValueError("APM sample weights must be finite and nonnegative")
    weighted['retained_for_apm'] = (
        weighted['retained_for_apm'].eq(True) & weights.gt(0)
    )
    design, outcome, players, home_context = build_apm_design(weighted)
    retained = weighted[weighted['retained_for_apm'].eq(True)]
    physical_weights = retained['_apm_sample_weight'].to_numpy(dtype=float)
    fit_weights = physical_weights.repeat(2)

    input_frame = pd.DataFrame(design.toarray(), columns=players)
    input_frame.insert(
        0, 'cv_fold_id', build_apm_cv_fold_ids(len(retained), n_folds=10)
    )
    input_frame.insert(0, 'home_context', home_context)
    input_frame.insert(0, 'sample_weight', fit_weights)
    input_frame.insert(0, 'outcome', outcome)
    result = run_r_csv_exchange(
        APM_R_SCRIPT,
        inputs={'input': input_frame},
        outputs={'output': {'player_id': 'string'}},
        argument_order=('input', 'output'),
        prefix='apm-glmnet-',
        error_label='Pinned GLMNET APM fit',
        library=advanced_r_library(DEFAULT_ADVANCED_R_LIBRARY, r_library),
    )['output']

    result['APM'] = result['APM_per_50']
    point_counts = {}
    for row, weight in zip(
        retained.itertuples(index=False), physical_weights, strict=True
    ):
        for player in (*row.home_lineup, *row.away_lineup):
            point_counts[str(player)] = point_counts.get(str(player), 0.0) + weight
    result['retained_points'] = result['player_id'].map(point_counts)
    result.attrs['model_intercept'] = float(result['model_intercept'].iloc[0])
    result.attrs['lambda_min'] = float(result['lambda_min'].iloc[0])
    result.attrs['lambda_1se'] = float(result['lambda_1se'].iloc[0])
    result.attrs['home_context_effect'] = float(
        result['home_context_effect'].iloc[0]
    )
    result.attrs['glmnet_version'] = '2.0-16'
    result.attrs['retained_complete_lineup_points'] = len(retained)
    result.attrs['effective_weighted_points'] = float(physical_weights.sum())
    result.attrs['total_points'] = len(point_table)
    result.attrs['coverage'] = len(retained) / len(point_table)
    return result.sort_values('player_id', kind='stable').reset_index(drop=True)


def _player_names(foundation: CanonicalSeason) -> pd.DataFrame:
    info = foundation.player_metadata.dropna(subset=['player_id']).copy()
    info['player_id'] = info['player_id'].astype('string')
    names = (
        info.groupby('player_id')['player_name']
        .agg(lambda x: x.dropna().astype(str).value_counts().index[0])
        .rename('player_name')
        .reset_index()
    )
    return names


def calculate_season_apm(
    foundation_or_season: CanonicalSeason | int,
    *,
    r_library: str | Path | None = None,
) -> pd.DataFrame:
    """Calculate one complete-lineup APM value per persistent player-season."""
    foundation = (
        build_mlv_canonical_season(int(foundation_or_season))
        if isinstance(foundation_or_season, (int, np.integer))
        else foundation_or_season
    )
    points = build_apm_point_table(foundation)
    result = fit_apm(points, r_library=r_library)
    result.insert(0, 'season', int(points['season'].iloc[0]))
    result = result.merge(_player_names(foundation), on='player_id', how='left')
    result.attrs.update({
        'retained_complete_lineup_points': int(
            points['retained_for_apm'].sum()
        ),
        'total_points': len(points),
        'coverage': float(points['retained_for_apm'].mean()),
    })
    return result
