"""Frozen conventional volleyball metric definitions.

Raw-source loading, identity reconciliation, and played-set reconstruction live
at the league-adapter boundary. This module accepts canonical player-set rows
and computes only the fourteen approved player-season statistics.
"""

from __future__ import annotations

import numpy as np


RELIABILITY_ATTEMPT_REQUIREMENTS = {
    'K%': 'attack_attempts',
    'Eff': 'attack_attempts',
    'Pass_Eff': 'receptions',
    'Srv_Eff': 'serves',
}
RELIABILITY_MIN_ATTEMPTS = 20


def calculate_tier1_metrics(df, round_digits=None):
    """Calculate the fourteen approved metrics from canonical player sets."""
    if 'played_set' not in df.columns:
        raise ValueError(
            "Raw metric input must contain the audited 'played_set' indicator"
        )
    working = df.copy()
    working['perfect_passes'] = np.round(
        working['receptions'] * working['perfect_reception_ratio']
    ).fillna(0)

    def canonical_player_name(names):
        counts = names.dropna().value_counts()
        if counts.empty:
            return np.nan
        candidates = counts[counts == counts.max()].index
        return min(candidates, key=str)

    stats = working.groupby('player_id').agg(
        player_name=('player_name', canonical_player_name),
        sets_played=('played_set', 'sum'),
        K=('attack_kills', 'sum'),
        Attack_Error=('attack_errors', 'sum'),
        Blocked_Attack_Error=('spike_hp', 'sum'),
        TA=('attack_attempts', 'sum'),
        A=('assists', 'sum'),
        SA=('serve_aces', 'sum'),
        SE=('serve_errors', 'sum'),
        TSrv=('serves', 'sum'),
        Perfect_Pass=('perfect_passes', 'sum'),
        RE=('reception_errors', 'sum'),
        TPass=('receptions', 'sum'),
        D=('successful_digs', 'sum'),
        B=('block_points', 'sum'),
        BT=('block_touches', 'sum'),
    ).reset_index()

    stats['E'] = stats['Attack_Error'] + stats['Blocked_Attack_Error']
    stats['K%'] = np.where(stats['TA'] > 0, stats['K'] / stats['TA'], np.nan)
    stats['Eff'] = np.where(
        stats['TA'] > 0, (stats['K'] - stats['E']) / stats['TA'], np.nan
    )
    stats['Pass_Eff'] = np.where(
        stats['TPass'] > 0,
        (stats['Perfect_Pass'] - stats['RE']) / stats['TPass'],
        np.nan,
    )
    stats['Srv_Eff'] = np.where(
        stats['TSrv'] > 0,
        (stats['SA'] - stats['SE']) / stats['TSrv'],
        np.nan,
    )
    raw_points = stats['K'] + stats['SA'] + stats['B']
    rates = {
        'K_S': stats['K'], 'E_S': stats['E'], 'AST_S': stats['A'],
        'SA_S': stats['SA'], 'SE_S': stats['SE'], 'RE_S': stats['RE'],
        'D_S': stats['D'], 'BT_S': stats['BT'], 'B_S': stats['B'],
        'PTS_S': raw_points,
    }
    for metric, numerator in rates.items():
        stats[metric] = np.where(
            stats['sets_played'] > 0,
            numerator / stats['sets_played'],
            np.nan,
        )
    columns = [
        'player_id', 'player_name', 'sets_played',
        'K_S', 'E_S', 'AST_S', 'SA_S', 'SE_S', 'RE_S', 'D_S', 'BT_S',
        'B_S', 'K%', 'Eff', 'Pass_Eff', 'Srv_Eff', 'PTS_S',
    ]
    result = stats[columns]
    return result.round(round_digits) if round_digits is not None else result


def apply_reliability_attempt_eligibility(
    metric_df,
    raw_df,
    min_attempts=RELIABILITY_MIN_ATTEMPTS,
):
    """Mask attempt metrics below the reliability-only observed cutoff."""
    if not isinstance(min_attempts, (int, np.integer)) or min_attempts < 1:
        raise ValueError("min_attempts must be a positive integer")
    keys = ['player_id']
    if 'season' in metric_df.columns and 'season' in raw_df.columns:
        keys.insert(0, 'season')
    required_metrics = [*keys, *RELIABILITY_ATTEMPT_REQUIREMENTS]
    required_raw = [
        *keys, *sorted(set(RELIABILITY_ATTEMPT_REQUIREMENTS.values()))
    ]
    missing_metrics = [c for c in required_metrics if c not in metric_df]
    missing_raw = [c for c in required_raw if c not in raw_df]
    if missing_metrics or missing_raw:
        raise ValueError(
            "Reliability attempt eligibility is missing columns: "
            f"metric={missing_metrics}, raw={missing_raw}"
        )
    if metric_df.duplicated(keys).any():
        raise ValueError(f"Metric rows must be unique on {keys}")
    attempt_columns = sorted(set(RELIABILITY_ATTEMPT_REQUIREMENTS.values()))
    attempts = (
        raw_df.groupby(keys, as_index=False, dropna=False)[attempt_columns]
        .sum(min_count=1)
        .fillna({column: 0 for column in attempt_columns})
    )
    eligibility = metric_df[keys].merge(
        attempts, on=keys, how='left', validate='one_to_one'
    )
    if eligibility[attempt_columns].isna().any().any():
        raise ValueError("A reliability metric row lacks raw attempts")
    result = metric_df.copy()
    for metric, attempt_column in RELIABILITY_ATTEMPT_REQUIREMENTS.items():
        result.loc[
            eligibility[attempt_column].lt(min_attempts).to_numpy(), metric
        ] = np.nan
    return result
