"""League-independent rally sequence, ancestry, and exposure products.

The functions here calculate no player metric.  They conservatively annotate
canonical actions with replay-safe order, possession/phase ancestry, terminal
attack semantics, and complete-lineup opportunity exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from canonical_data import CanonicalSeason, RALLY_KEY, require_columns


ATTACK_KILL_OUTCOMES = frozenset({'#'})
ATTACK_ERROR_OUTCOMES = frozenset({'=', '/'})
ATTACK_NONTERMINAL_OUTCOMES = frozenset({'+', '!', '-'})
DIG_SUCCESS_OUTCOMES = frozenset({'#', '+', '!', '-', '/'})


@dataclass(frozen=True)
class CanonicalSequenceProducts:
    actions: pd.DataFrame
    dig_ancestry: pd.DataFrame
    player_point_exposure: pd.DataFrame


def build_sequence_action_table(actions: pd.DataFrame) -> pd.DataFrame:
    """Annotate ordered actions without guessing unresolved phase semantics."""
    require_columns(
        actions,
        [
            *RALLY_KEY, 'rally_id', 'action_order', 'acting_team',
            'point_winner_team', 'player_id', 'jersey_number',
            'source_action', 'source_outcome',
        ],
        'canonical actions',
    )
    result = actions.copy()
    result['sequence_action_id'] = (
        result['rally_id'].astype(str) + ':'
        + result['action_order'].astype(str)
    )
    if 'sequence_source_valid' in result:
        result['sequence_retained'] = result[
            'sequence_source_valid'
        ].fillna(False).astype(bool)
    else:
        result['sequence_retained'] = True
    result['sequence_exclusion_reason'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result.loc[
        ~result['sequence_retained'], 'sequence_exclusion_reason'
    ] = 'source_score_winner_inconsistency'
    result['possession_index'] = pd.Series(
        pd.NA, index=result.index, dtype='Int64'
    )
    result['possession_id'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['attack_phase'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['attack_phase_reason'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['preceding_reception_action_id'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['preceding_dig_action_id'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['attack_result'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['attack_semantics_valid'] = pd.Series(
        pd.NA, index=result.index, dtype='boolean'
    )
    result['attack_semantics_reason'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['attack_metric_eligible'] = False
    result['attack_exclusion_reason'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )
    result['point_bearing_serve'] = False
    result['serve_opportunity_valid'] = False
    result['rally_point_status'] = pd.Series(
        pd.NA, index=result.index, dtype='string'
    )

    for _, rally in result.groupby(RALLY_KEY, sort=False, dropna=False):
        ordered = rally.sort_values('action_order', kind='stable')
        serve_rows = ordered[ordered['source_action'].eq('S')]
        if len(serve_rows) > 1:
            final_serve_order = int(serve_rows['action_order'].max())
            replay_prefix = ordered['action_order'].lt(final_serve_order)
            prefix_index = ordered.index[replay_prefix]
            result.loc[prefix_index, 'sequence_retained'] = False
            result.loc[prefix_index, 'sequence_exclusion_reason'] = np.where(
                result.loc[prefix_index, 'sequence_exclusion_reason'].isna(),
                'replay_prefix',
                result.loc[prefix_index, 'sequence_exclusion_reason'],
            )

        retained = result.loc[
            ordered.index[result.loc[ordered.index, 'sequence_retained']]
        ].sort_values('action_order', kind='stable')
        retained_serves = retained[retained['source_action'].eq('S')]
        if retained_serves.empty:
            rally_point_status = 'no_source_serve'
        else:
            final_serve_index = retained_serves.index[-1]
            result.loc[final_serve_index, 'point_bearing_serve'] = True
            if 'serve_opportunity_supported' in result:
                supported = result.loc[
                    final_serve_index, 'serve_opportunity_supported'
                ]
                valid_serve = bool(supported) if not pd.isna(supported) else False
            else:
                jersey = pd.to_numeric(
                    pd.Series([result.loc[final_serve_index, 'jersey_number']]),
                    errors='coerce',
                ).iloc[0]
                valid_serve = pd.notna(jersey) and float(jersey) > 0
            result.loc[
                final_serve_index, 'serve_opportunity_valid'
            ] = valid_serve
            later = retained['action_order'].gt(
                result.loc[final_serve_index, 'action_order']
            ).any()
            if not valid_serve:
                rally_point_status = 'anonymous_or_synthetic_serve'
            elif later:
                rally_point_status = 'rally_point'
            elif result.loc[final_serve_index, 'source_outcome'] in {'#', '='}:
                rally_point_status = 'terminal_serve'
            else:
                rally_point_status = 'unresolved_terminal_sequence'
        result.loc[ordered.index, 'rally_point_status'] = rally_point_status

        possession_index = 0
        prior_team = None
        control_action = None
        possession_first_action = None
        retained_indices = retained.index.tolist()
        for position, index in enumerate(retained_indices):
            row = result.loc[index]
            if row['acting_team'] != prior_team:
                possession_index += 1
                prior_team = row['acting_team']
                control_action = None
                possession_first_action = row['source_action']
            result.loc[index, 'possession_index'] = possession_index
            result.loc[index, 'possession_id'] = (
                f"{row['rally_id']}:{possession_index}"
            )
            action = row['source_action']
            if action in {'R', 'D'}:
                # An error cannot establish controlled possession ancestry.
                control_action = row if row['source_outcome'] != '=' else None
            elif action == 'F':
                # A freeball sends possession to the opponent.  It cannot be
                # used as a reception/dig ancestor for a later same-team attack.
                control_action = None
            if action != 'A':
                continue

            if control_action is not None:
                control_id = control_action['sequence_action_id']
                if control_action['source_action'] == 'R':
                    phase = 'serve_receive'
                    phase_reason = 'preceding_reception'
                    result.loc[
                        index, 'preceding_reception_action_id'
                    ] = control_id
                else:
                    phase = 'transition'
                    phase_reason = 'preceding_dig'
                    result.loc[index, 'preceding_dig_action_id'] = control_id
            else:
                phase = 'unresolved'
                phase_reason = (
                    'no_reception_or_dig_ancestor:'
                    f"first_{possession_first_action or 'attack'}"
                )
            result.loc[index, 'attack_phase'] = phase
            result.loc[index, 'attack_phase_reason'] = phase_reason

            outcome = row['source_outcome']
            winner_is_actor = row['point_winner_team'] == row['acting_team']
            has_following_action = position < len(retained_indices) - 1
            if outcome in ATTACK_KILL_OUTCOMES:
                attack_result = 'kill'
                valid = bool(winner_is_actor)
                semantics_reason = (
                    'source_kill' if valid else 'kill_winner_contradiction'
                )
            elif outcome == '=':
                attack_result = 'unblocked_error'
                valid = not bool(winner_is_actor)
                semantics_reason = (
                    'source_unblocked_error'
                    if valid else 'error_winner_contradiction'
                )
            elif outcome == '/':
                attack_result = 'blocked_error'
                valid = not bool(winner_is_actor)
                semantics_reason = (
                    'source_blocked_error'
                    if valid else 'blocked_winner_contradiction'
                )
            elif outcome in ATTACK_NONTERMINAL_OUTCOMES:
                attack_result = 'nonterminal'
                valid = has_following_action
                semantics_reason = (
                    'source_nonterminal'
                    if valid else 'nonterminal_without_following_action'
                )
            else:  # canonical action construction rejects unknown outcomes
                raise AssertionError(f'Unexpected attack outcome {outcome!r}')
            result.loc[index, 'attack_result'] = attack_result
            result.loc[index, 'attack_semantics_valid'] = valid
            result.loc[index, 'attack_semantics_reason'] = semantics_reason

            exclusion = None
            if pd.isna(row['player_id']):
                exclusion = 'attacker_identity_missing'
            elif phase == 'unresolved':
                exclusion = phase_reason
            elif not valid:
                exclusion = semantics_reason
            result.loc[index, 'attack_metric_eligible'] = exclusion is None
            if exclusion is not None:
                result.loc[index, 'attack_exclusion_reason'] = exclusion
            # A later same-team attack must have its own reception/dig ancestor.
            control_action = None

    return result.sort_values(
        [*RALLY_KEY, 'action_order'], kind='stable'
    ).reset_index(drop=True)


def build_dig_ancestry(sequence_actions: pd.DataFrame) -> pd.DataFrame:
    """Link each safely modeled dig to its ensuing same-possession attack."""
    require_columns(
        sequence_actions,
        [
            *RALLY_KEY, 'rally_id', 'action_order', 'sequence_action_id',
            'sequence_retained',
            'sequence_exclusion_reason',
            'possession_id', 'acting_team', 'player_id', 'source_action',
            'source_outcome', 'attack_result', 'attack_semantics_valid',
            'preceding_dig_action_id',
        ],
        'canonical sequence actions',
    )
    digs = sequence_actions[
        sequence_actions['source_action'].eq('D')
    ].copy()
    attacks_by_dig = sequence_actions[
        sequence_actions['sequence_retained']
        & sequence_actions['source_action'].eq('A')
        & sequence_actions['preceding_dig_action_id'].notna()
    ].copy()
    if attacks_by_dig['preceding_dig_action_id'].duplicated().any():
        raise AssertionError('One dig is the ancestor of multiple attacks')
    attacks_by_dig = (
        attacks_by_dig.set_index('preceding_dig_action_id').to_dict('index')
    )
    rows = []
    for dig in digs.itertuples(index=False):
        exclusion = None
        if not dig.sequence_retained:
            exclusion = dig.sequence_exclusion_reason
        elif pd.isna(dig.player_id):
            exclusion = 'digger_identity_missing'
        elif dig.source_outcome not in DIG_SUCCESS_OUTCOMES:
            exclusion = 'unsuccessful_dig'

        resulting_attack = None
        if exclusion is None:
            if dig.sequence_action_id in attacks_by_dig:
                resulting_attack = attacks_by_dig[dig.sequence_action_id]
                if not bool(resulting_attack['attack_semantics_valid']):
                    exclusion = 'ensuing_attack_semantics_invalid'

        eligible = exclusion is None
        converted = bool(eligible and resulting_attack is not None)
        ensuing_kill = bool(
            converted and resulting_attack['attack_result'] == 'kill'
        )
        rows.append({
            **{column: getattr(dig, column) for column in RALLY_KEY},
            'rally_id': dig.rally_id,
            'dig_action_id': dig.sequence_action_id,
            'action_order': dig.action_order,
            'team': dig.acting_team,
            'player_id': dig.player_id,
            'source_outcome': dig.source_outcome,
            'dig_model_eligible': eligible,
            'dig_exclusion_reason': exclusion,
            'resulting_attack_action_id': (
                resulting_attack['sequence_action_id']
                if resulting_attack is not None else pd.NA
            ),
            'dig_converted_to_attack': converted,
            'dig_ensuing_attack_kill': ensuing_kill,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result['player_id'] = result['player_id'].astype('string')
        result['dig_exclusion_reason'] = result[
            'dig_exclusion_reason'
        ].astype('string')
        result['resulting_attack_action_id'] = result[
            'resulting_attack_action_id'
        ].astype('string')
    return result


def build_player_point_exposure(
    canonical: CanonicalSeason,
    sequence_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Return complete-lineup player exposure and opponent opportunities."""
    lineups = canonical.lineups.merge(
        canonical.rallies[[
            *RALLY_KEY, 'home_team', 'away_team', 'point_winner_team'
        ]],
        on=RALLY_KEY,
        how='left',
        validate='one_to_one',
    )
    complete = lineups[lineups['lineup_complete'].fillna(False)].copy()
    rows = []
    for side in ('home', 'away'):
        for position in range(1, 7):
            part = complete[[
                *RALLY_KEY, 'rally_id', 'home_team', 'away_team',
                'point_winner_team', 'lineup_status',
                f'{side}_p{position}_player_id',
            ]].rename(columns={
                f'{side}_p{position}_player_id': 'player_id'
            })
            part['team'] = part[f'{side}_team']
            part['opponent_team'] = part[
                'away_team' if side == 'home' else 'home_team'
            ]
            part['court_position'] = position
            rows.append(part)
    exposure = pd.concat(rows, ignore_index=True)
    exposure['player_id'] = exposure['player_id'].astype('string')
    if exposure['player_id'].isna().any():
        raise AssertionError('Complete lineup contains a missing player identity')
    if exposure.duplicated([*RALLY_KEY, 'team', 'player_id']).any():
        raise AssertionError('Complete lineup repeats a player for one team')

    retained = sequence_actions[sequence_actions['sequence_retained']].copy()
    attack_counts = (
        retained[retained['source_action'].eq('A')]
        .groupby([*RALLY_KEY, 'acting_team'])
        .size()
        .to_dict()
    )
    serve_rows = retained[
        retained['point_bearing_serve']
        & retained['serve_opportunity_valid']
    ]
    if serve_rows.duplicated(RALLY_KEY).any():
        raise AssertionError('Rally has multiple point-bearing serves')
    serve_team = serve_rows.set_index(RALLY_KEY)['acting_team'].to_dict()
    rally_status = (
        sequence_actions.groupby(RALLY_KEY, sort=False)[
            'rally_point_status'
        ].first().to_dict()
    )
    exposure['on_court_points'] = 1
    exposure['team_point_wins'] = exposure['point_winner_team'].eq(
        exposure['team']
    ).astype(int)
    exposure['opponent_attack_opportunities'] = [
        attack_counts.get((*key, opponent), 0)
        for (*key, opponent) in exposure[[
            *RALLY_KEY, 'opponent_team'
        ]].itertuples(index=False, name=None)
    ]
    exposure['opponent_serve_opportunities'] = [
        int(serve_team.get(tuple(key)) == opponent)
        for (*key, opponent) in exposure[[
            *RALLY_KEY, 'opponent_team'
        ]].itertuples(index=False, name=None)
    ]
    exposure['rally_point_status'] = [
        rally_status[tuple(key)]
        for key in exposure[RALLY_KEY].itertuples(index=False, name=None)
    ]
    exposure['on_court_rally_points'] = exposure[
        'rally_point_status'
    ].eq('rally_point').astype(int)
    columns = [
        *RALLY_KEY, 'rally_id', 'team', 'opponent_team', 'player_id',
        'court_position', 'lineup_status', 'point_winner_team',
        'rally_point_status', 'on_court_points', 'team_point_wins',
        'opponent_attack_opportunities', 'opponent_serve_opportunities',
        'on_court_rally_points',
    ]
    return exposure[columns].sort_values(
        [*RALLY_KEY, 'team', 'court_position'], kind='stable'
    ).reset_index(drop=True)


def attach_action_lineup_scope(
    sequence_actions: pd.DataFrame,
    player_point_exposure: pd.DataFrame,
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    """Mark whether each attributed action belongs to its complete lineup."""
    result = sequence_actions.copy()
    complete_keys = lineups.loc[
        lineups['lineup_complete'].fillna(False), RALLY_KEY
    ].drop_duplicates()
    complete_keys['_lineup_complete'] = True
    result = result.merge(
        complete_keys,
        on=RALLY_KEY,
        how='left',
        validate='many_to_one',
    )
    result['lineup_complete'] = result['_lineup_complete'].eq(True)
    result = result.drop(columns='_lineup_complete')
    court = player_point_exposure[[
        *RALLY_KEY, 'team', 'player_id'
    ]].drop_duplicates()
    court['_actor_on_court'] = True
    result = result.merge(
        court,
        left_on=[*RALLY_KEY, 'acting_team', 'player_id'],
        right_on=[*RALLY_KEY, 'team', 'player_id'],
        how='left',
        validate='many_to_one',
    ).drop(columns='team')
    result['actor_on_court'] = result['_actor_on_court'].eq(True)
    return result.drop(columns='_actor_on_court')


def build_canonical_sequence_products(
    canonical: CanonicalSeason,
) -> CanonicalSequenceProducts:
    actions = build_sequence_action_table(canonical.actions)
    exposure = build_player_point_exposure(canonical, actions)
    actions = attach_action_lineup_scope(actions, exposure, canonical.lineups)
    digs = build_dig_ancestry(actions)
    return CanonicalSequenceProducts(
        actions=actions,
        dig_ancestry=digs,
        player_point_exposure=exposure,
    )
