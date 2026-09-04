"""Faithfully specified Evollve player metrics.

Phase 5 promotes these formulas into the final analysis registry with
league-specific inclusion. All nonlinear rates are reconstructed from additive
team-match sufficient statistics; zero denominators remain missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canonical_data import CanonicalSeason, require_columns
from canonical_sequence import (
    ATTACK_ERROR_OUTCOMES,
    ATTACK_NONTERMINAL_OUTCOMES,
    DIG_SUCCESS_OUTCOMES,
    CanonicalSequenceProducts,
    build_canonical_sequence_products,
)
from mlv_adapter import build_mlv_canonical_season
from metric_registry import EVOLLVE_METRICS


# Historical public name retained for callers written before Phase 5. The
# authoritative list now lives in metric_registry.py.
EVOLLVE_CANDIDATE_METRICS = EVOLLVE_METRICS

EVOLLVE_CANDIDATE_FORMULAS = {
    'SR_Hit_Pct': '(sr_kills - sr_errors) / sr_attacks',
    'Transition_Hit_Pct': (
        '(transition_kills - transition_errors) / transition_attacks'
    ),
    'SR_Hit_Avg': (
        '(sr_kills + 0.433 * sr_nonterminal_attacks) / sr_attacks'
    ),
    'Transition_Hit_Avg': (
        '(transition_kills + 0.433 * transition_nonterminal_attacks) '
        '/ transition_attacks'
    ),
    'Dig_Att_Cnv_Pct': 'digs_converted_to_attack / modeled_digs',
    'Dig_Kill_Pct': 'digs_converted_to_kill / modeled_digs',
    'Dig_Rate': 'player_digs_on_court / opponent_attacks_on_court',
    'Reception_Rate': (
        'player_receptions_on_court / opponent_serves_on_court'
    ),
    'Pct_Points_Won_On': 'team_points_won_on_court / on_court_points',
    'Blocked_Rate': 'blocked_attacks / attack_attempts',
    'Srv_Avg': (
        '(serve_aces + 0.435 * '
        '(serves - serve_aces - service_errors)) / serves'
    ),
    'Attack_Error_Rate_No_Blocks': (
        'unblocked_attack_errors / attack_attempts'
    ),
    'Kill_Rate_Rally_Point': (
        'kills_on_court_rally_points / on_court_rally_points'
    ),
}

COUNT_COLUMNS = (
    'sr_attacks', 'sr_kills', 'sr_errors', 'sr_nonterminal_attacks',
    'transition_attacks', 'transition_kills', 'transition_errors',
    'transition_nonterminal_attacks', 'modeled_digs',
    'digs_converted_to_attack', 'digs_converted_to_kill',
    'player_digs_on_court', 'opponent_attacks_on_court',
    'player_receptions_on_court', 'opponent_serves_on_court',
    'team_points_won_on_court', 'on_court_points',
    'kills_on_court_rally_points', 'on_court_rally_points',
    'attack_attempts', 'unblocked_attack_errors', 'blocked_attacks',
    'serves', 'serve_aces', 'service_errors',
)


def _canonical_name(values: pd.Series):
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return pd.NA
    return sorted(counts[counts.eq(counts.max())].index)[0]


def _name_lookup(canonical: CanonicalSeason) -> pd.Series:
    names = pd.concat([
        canonical.player_sets[['player_id', 'player_name']],
        canonical.player_metadata[['player_id', 'player_name']],
    ], ignore_index=True).dropna(subset=['player_id'])
    names['player_id'] = names['player_id'].astype('string')
    return names.groupby('player_id')['player_name'].agg(_canonical_name)


def _phase_attack_statistics(actions: pd.DataFrame) -> pd.DataFrame:
    attacks = actions[
        actions['source_action'].eq('A')
        & actions['attack_metric_eligible'].eq(True)
    ].copy()
    if attacks.empty:
        return pd.DataFrame(columns=[
            'season', 'match_id', 'team', 'player_id',
            'sr_attacks', 'sr_kills', 'sr_errors',
            'sr_nonterminal_attacks', 'transition_attacks',
            'transition_kills', 'transition_errors',
            'transition_nonterminal_attacks',
        ])
    attacks['team'] = attacks['acting_team']
    attacks['player_id'] = attacks['player_id'].astype('string')
    rows = []
    for phase, prefix in (
        ('serve_receive', 'sr'), ('transition', 'transition')
    ):
        selected = attacks[attacks['attack_phase'].eq(phase)].copy()
        selected[f'{prefix}_attacks'] = 1
        selected[f'{prefix}_kills'] = selected['attack_result'].eq('kill').astype(int)
        selected[f'{prefix}_errors'] = selected['source_outcome'].isin(
            ATTACK_ERROR_OUTCOMES
        ).astype(int)
        selected[f'{prefix}_nonterminal_attacks'] = selected[
            'source_outcome'
        ].isin(ATTACK_NONTERMINAL_OUTCOMES).astype(int)
        columns = [
            f'{prefix}_attacks', f'{prefix}_kills', f'{prefix}_errors',
            f'{prefix}_nonterminal_attacks',
        ]
        rows.append(
            selected.groupby(
                ['season', 'match_id', 'team', 'player_id'],
                as_index=False,
                sort=True,
            )[columns].sum()
        )
    return rows[0].merge(
        rows[1],
        on=['season', 'match_id', 'team', 'player_id'],
        how='outer',
        validate='one_to_one',
    ).fillna(0)


def _dig_statistics(dig_ancestry: pd.DataFrame) -> pd.DataFrame:
    digs = dig_ancestry[dig_ancestry['dig_model_eligible'].eq(True)].copy()
    if digs.empty:
        return pd.DataFrame(columns=[
            'season', 'match_id', 'team', 'player_id', 'modeled_digs',
            'digs_converted_to_attack', 'digs_converted_to_kill',
        ])
    digs['modeled_digs'] = 1
    digs['digs_converted_to_attack'] = digs[
        'dig_converted_to_attack'
    ].astype(int)
    digs['digs_converted_to_kill'] = digs[
        'dig_ensuing_attack_kill'
    ].astype(int)
    return digs.groupby(
        ['season', 'match_id', 'team', 'player_id'],
        as_index=False,
        sort=True,
    )[[
        'modeled_digs', 'digs_converted_to_attack',
        'digs_converted_to_kill',
    ]].sum()


def _exposure_statistics(
    sequence: CanonicalSequenceProducts,
) -> pd.DataFrame:
    exposure = sequence.player_point_exposure.copy()
    exposure_counts = exposure.groupby(
        ['season', 'match_id', 'team', 'player_id'],
        as_index=False,
        sort=True,
    ).agg(
        opponent_attacks_on_court=(
            'opponent_attack_opportunities', 'sum'
        ),
        opponent_serves_on_court=('opponent_serve_opportunities', 'sum'),
        team_points_won_on_court=('team_point_wins', 'sum'),
        on_court_points=('on_court_points', 'sum'),
        on_court_rally_points=('on_court_rally_points', 'sum'),
    )

    actions = sequence.actions[
        sequence.actions['sequence_retained']
        & sequence.actions['lineup_complete']
        & sequence.actions['actor_on_court']
        & sequence.actions['player_id'].notna()
    ].copy()
    actions['team'] = actions['acting_team']
    actions['player_id'] = actions['player_id'].astype('string')
    actions['player_digs_on_court'] = (
        actions['source_action'].eq('D')
        & actions['source_outcome'].isin(DIG_SUCCESS_OUTCOMES)
    ).astype(int)
    actions['player_receptions_on_court'] = actions[
        'source_action'
    ].eq('R').astype(int)
    actions['kills_on_court_rally_points'] = (
        actions['source_action'].eq('A')
        & actions['attack_result'].eq('kill')
        & actions['attack_semantics_valid'].eq(True)
        & actions['rally_point_status'].eq('rally_point')
    ).astype(int)
    action_counts = actions.groupby(
        ['season', 'match_id', 'team', 'player_id'],
        as_index=False,
        sort=True,
    )[[
        'player_digs_on_court', 'player_receptions_on_court',
        'kills_on_court_rally_points',
    ]].sum()
    return exposure_counts.merge(
        action_counts,
        on=['season', 'match_id', 'team', 'player_id'],
        how='outer',
        validate='one_to_one',
    ).fillna(0)


def _boxscore_statistics(canonical: CanonicalSeason) -> pd.DataFrame:
    required = [
        'season', 'match_id', 'team_name', 'player_id', 'attack_attempts',
        'attack_errors', 'spike_hp', 'serves', 'serve_aces', 'serve_errors',
    ]
    require_columns(canonical.player_sets, required, 'canonical player sets')
    boxscore = canonical.player_sets.copy()
    boxscore['player_id'] = boxscore['player_id'].astype('string')
    result = boxscore.groupby(
        ['season', 'match_id', 'team_name', 'player_id'],
        as_index=False,
        sort=True,
    )[[
        'attack_attempts', 'attack_errors', 'spike_hp', 'serves',
        'serve_aces', 'serve_errors',
    ]].sum()
    return result.rename(columns={
        'team_name': 'team',
        'attack_errors': 'unblocked_attack_errors',
        'spike_hp': 'blocked_attacks',
        'serve_errors': 'service_errors',
    })


def build_boxscore_evollve_sufficient_statistics(
    canonical: CanonicalSeason,
) -> pd.DataFrame:
    """Return only the source-exact boxscore inputs supported for AU.

    This intentionally does not build sequence or lineup products and does not
    expose numeric diagnostic versions of unsupported AU metrics.
    """
    result = _boxscore_statistics(canonical)
    names = _name_lookup(canonical)
    result['player_name'] = result['player_id'].map(names)
    return result[[
        'season', 'match_id', 'team', 'player_id', 'player_name',
        'attack_attempts', 'unblocked_attack_errors', 'blocked_attacks',
        'serves', 'serve_aces', 'service_errors',
    ]].sort_values(
        ['season', 'match_id', 'team', 'player_id'], kind='stable'
    ).reset_index(drop=True)


def calculate_boxscore_evollve_player_seasons(
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the three exact glossary formulas supported for AU."""
    supported = (
        'Blocked_Rate', 'Srv_Avg', 'Attack_Error_Rate_No_Blocks',
    )
    required_counts = (
        'attack_attempts', 'unblocked_attack_errors', 'blocked_attacks',
        'serves', 'serve_aces', 'service_errors',
    )
    require_columns(
        statistics,
        ['season', 'player_id', 'player_name', *required_counts],
        'boxscore Evollve sufficient statistics',
    )
    expanded = statistics.copy()
    for column in COUNT_COLUMNS:
        if column not in expanded:
            expanded[column] = 0
    values = calculate_evollve_player_seasons(expanded)
    return values[['season', 'player_id', 'player_name', *supported]]


def calculate_boxscore_evollve_candidates(
    canonical: CanonicalSeason,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return AU-supported Evollve values and their additive inputs."""
    statistics = build_boxscore_evollve_sufficient_statistics(canonical)
    return calculate_boxscore_evollve_player_seasons(statistics), statistics


def build_evollve_sufficient_statistics(
    canonical: CanonicalSeason,
    sequence: CanonicalSequenceProducts | None = None,
) -> pd.DataFrame:
    """Return additive team-match/player inputs for all 13 candidates."""
    sequence = sequence or build_canonical_sequence_products(canonical)
    keys = ['season', 'match_id', 'team', 'player_id']
    components = [
        _phase_attack_statistics(sequence.actions),
        _dig_statistics(sequence.dig_ancestry),
        _exposure_statistics(sequence),
        _boxscore_statistics(canonical),
    ]
    result = components[0]
    for component in components[1:]:
        result = result.merge(
            component, on=keys, how='outer', validate='one_to_one'
        )
    for column in COUNT_COLUMNS:
        if column not in result:
            result[column] = 0
        result[column] = pd.to_numeric(
            result[column], errors='coerce'
        ).fillna(0)
        if (result[column] < 0).any() or not np.allclose(
            result[column], np.round(result[column]), rtol=0, atol=0
        ):
            raise ValueError(f'Invalid Evollve sufficient count: {column}')
        result[column] = result[column].astype(np.int64)
    names = _name_lookup(canonical)
    result['player_name'] = result['player_id'].map(names)
    if result[keys].isna().any().any() or result.duplicated(keys).any():
        raise ValueError('Evollve sufficient-statistic keys are invalid')
    return result[[*keys, 'player_name', *COUNT_COLUMNS]].sort_values(
        keys, kind='stable'
    ).reset_index(drop=True)


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    return numerator.astype(float).div(denominator.where(denominator.gt(0)))


def calculate_evollve_player_seasons(
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate persistent player-seasons and apply published formulas."""
    require_columns(
        statistics,
        ['season', 'player_id', 'player_name', *COUNT_COLUMNS],
        'Evollve sufficient statistics',
    )
    grouped = statistics.groupby(
        ['season', 'player_id'], as_index=False, sort=True
    ).agg({
        'player_name': _canonical_name,
        **{column: 'sum' for column in COUNT_COLUMNS},
    })
    grouped['SR_Hit_Pct'] = _rate(
        grouped['sr_kills'] - grouped['sr_errors'], grouped['sr_attacks']
    )
    grouped['Transition_Hit_Pct'] = _rate(
        grouped['transition_kills'] - grouped['transition_errors'],
        grouped['transition_attacks'],
    )
    grouped['SR_Hit_Avg'] = _rate(
        grouped['sr_kills'] + 0.433 * grouped['sr_nonterminal_attacks'],
        grouped['sr_attacks'],
    )
    grouped['Transition_Hit_Avg'] = _rate(
        grouped['transition_kills']
        + 0.433 * grouped['transition_nonterminal_attacks'],
        grouped['transition_attacks'],
    )
    grouped['Dig_Att_Cnv_Pct'] = _rate(
        grouped['digs_converted_to_attack'], grouped['modeled_digs']
    )
    grouped['Dig_Kill_Pct'] = _rate(
        grouped['digs_converted_to_kill'], grouped['modeled_digs']
    )
    grouped['Dig_Rate'] = _rate(
        grouped['player_digs_on_court'], grouped['opponent_attacks_on_court']
    )
    grouped['Reception_Rate'] = _rate(
        grouped['player_receptions_on_court'],
        grouped['opponent_serves_on_court'],
    )
    grouped['Pct_Points_Won_On'] = _rate(
        grouped['team_points_won_on_court'], grouped['on_court_points']
    )
    grouped['Blocked_Rate'] = _rate(
        grouped['blocked_attacks'], grouped['attack_attempts']
    )
    grouped['Srv_Avg'] = _rate(
        grouped['serve_aces']
        + 0.435 * (
            grouped['serves'] - grouped['serve_aces']
            - grouped['service_errors']
        ),
        grouped['serves'],
    )
    grouped['Attack_Error_Rate_No_Blocks'] = _rate(
        grouped['unblocked_attack_errors'], grouped['attack_attempts']
    )
    grouped['Kill_Rate_Rally_Point'] = _rate(
        grouped['kills_on_court_rally_points'],
        grouped['on_court_rally_points'],
    )
    return grouped[[
        'season', 'player_id', 'player_name', *EVOLLVE_CANDIDATE_METRICS,
        *COUNT_COLUMNS,
    ]]


def calculate_evollve_candidates(
    canonical: CanonicalSeason,
) -> tuple[pd.DataFrame, pd.DataFrame, CanonicalSequenceProducts]:
    sequence = build_canonical_sequence_products(canonical)
    statistics = build_evollve_sufficient_statistics(canonical, sequence)
    values = calculate_evollve_player_seasons(statistics)
    return values, statistics, sequence


def calculate_mlv_evollve_candidates(
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, CanonicalSequenceProducts]:
    return calculate_evollve_candidates(build_mlv_canonical_season(int(season)))
