"""League-independent canonical play-by-play reconstruction.

This module computes no metric and loads no league. It turns adapter-corrected
PBP and event tables into deterministic action, rally, and lineup products.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from canonical_data import (
    IDENTITY_KEY,
    RALLY_KEY,
    normalize_jersey_numbers,
    require_columns as _require_columns,
)

# DataVolley-style action codes supplied by pyvolleydata.  Source codes are
# retained beside these labels so downstream metrics can use action-specific
# outcome semantics rather than treating the generic grade as a statistic.
ACTION_MAP = {
    'S': 'serve',
    'R': 'reception',
    'E': 'set',
    'A': 'attack',
    'B': 'block',
    'D': 'dig',
    'F': 'freeball',
}
OUTCOME_MAP = {
    '#': 'excellent_or_winning',
    '+': 'positive',
    '!': 'neutral',
    '-': 'negative',
    '/': 'poor',
    '=': 'error',
}

def _identity_candidates(frame, value_column):
    valid = frame.dropna(subset=[*IDENTITY_KEY, value_column])
    return (
        valid.groupby(IDENTITY_KEY, dropna=False)[value_column]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
        .to_dict()
    )


def _apply_audited_pbp_identity_overrides(
    identity_map: pd.DataFrame,
    mapped_boxscore: pd.DataFrame,
    mapped_info: pd.DataFrame,
    season: int,
    audited_overrides,
) -> pd.DataFrame:
    """Apply adapter-supplied aliases while enforcing generic evidence guards."""
    result = identity_map.copy()
    applicable = [
        (key, evidence)
        for key, evidence in audited_overrides.items()
        if key[0] == int(season)
    ]
    alias_keys = [(key[1], key[2], key[3]) for key, _ in applicable]
    if len(alias_keys) != len(set(alias_keys)):
        raise ValueError("Audited PBP identity overrides contain duplicate keys")

    additions = []
    for (
        override_season, match_id, team_name, source_jersey
    ), evidence in applicable:
        player_id = evidence['player_id']
        canonical_jersey = evidence['canonical_jersey']
        normalized_name = evidence['normalized_name']
        key_mask = (
            result['match_id'].eq(match_id)
            & result['team_name'].eq(team_name)
            & result['_jersey_number'].eq(source_jersey)
        )
        if key_mask.any():
            existing = result.loc[
                key_mask, ['player_id', 'identity_status']
            ].to_dict('records')
            raise ValueError(
                "Audited malformed PBP jersey now has source identity evidence: "
                f"{(override_season, match_id, team_name, source_jersey)} -> "
                f"{existing}"
            )

        source_jersey_box = mapped_boxscore[
            mapped_boxscore['match_id'].eq(match_id)
            & mapped_boxscore['team_name'].eq(team_name)
            & mapped_boxscore['_jersey_number'].eq(source_jersey)
        ]
        source_jersey_info = mapped_info[
            mapped_info['match_id'].eq(match_id)
            & mapped_info['team_name'].eq(team_name)
            & mapped_info['_jersey_number'].eq(source_jersey)
        ]
        if not source_jersey_box.empty or not source_jersey_info.empty:
            raise ValueError(
                "Audited PBP alias collides with a legitimate source jersey: "
                f"{(override_season, match_id, team_name, source_jersey)}"
            )

        target_box = mapped_boxscore[
            mapped_boxscore['match_id'].eq(match_id)
            & mapped_boxscore['team_name'].eq(team_name)
            & mapped_boxscore['player_id'].eq(player_id)
        ]
        target_info = mapped_info[
            mapped_info['match_id'].eq(match_id)
            & mapped_info['team_name'].eq(team_name)
            & mapped_info['_canonical_player_id'].eq(player_id)
        ]
        if target_box.empty or target_info.empty:
            raise ValueError(
                "Audited PBP identity target lacks same-match boxscore/roster "
                f"evidence: {(override_season, match_id, team_name, player_id)}"
            )
        for label, target in (
            ('boxscore', target_box),
            ('player info', target_info),
        ):
            jerseys = set(target['_jersey_number'].dropna().astype(int))
            names = set(target['_normalized_name'].dropna())
            if jerseys != {canonical_jersey} or names != {normalized_name}:
                raise ValueError(
                    f"Audited PBP target {label} evidence changed for "
                    f"{(override_season, match_id, team_name, player_id)}: "
                    f"jerseys={sorted(jerseys)}, names={sorted(names)}"
                )

        canonical_mask = (
            result['match_id'].eq(match_id)
            & result['team_name'].eq(team_name)
            & result['_jersey_number'].eq(canonical_jersey)
        )
        canonical = result.loc[canonical_mask]
        if (
            len(canonical) != 1
            or pd.isna(canonical.iloc[0]['player_id'])
            or str(canonical.iloc[0]['player_id']) != player_id
        ):
            raise ValueError(
                "Audited PBP target is not uniquely resolved at its canonical "
                f"jersey: {(override_season, match_id, team_name, canonical_jersey)}"
            )

        additions.append({
            'match_id': match_id,
            'team_name': team_name,
            '_jersey_number': source_jersey,
            'player_id': player_id,
            'identity_status': 'audited_pbp_override',
            'boxscore_candidate_ids': np.nan,
            'roster_candidate_ids': np.nan,
        })

    if additions:
        result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True)
    if result.duplicated(IDENTITY_KEY).any():
        raise AssertionError("Audited PBP override created a duplicate identity key")
    return result


def build_player_identity_map(
    player_boxscore: pd.DataFrame,
    player_info: pd.DataFrame,
    season: int,
    audited_overrides=None,
) -> pd.DataFrame:
    """Build a match/team/jersey map, prioritizing actual boxscore evidence.

    A roster can contain two players assigned the same jersey in one match.  If
    only one has a set boxscore, that observed player is authoritative.  A
    roster-only collision remains explicitly ambiguous.
    """
    audited_overrides = audited_overrides or {}
    mapped_boxscore = player_boxscore.copy()
    mapped_info = player_info.copy()
    _require_columns(
        mapped_boxscore,
        [*IDENTITY_KEY, 'player_id', '_normalized_name'],
        'canonical player boxscore',
    )
    _require_columns(
        mapped_info,
        [*IDENTITY_KEY, '_canonical_player_id', '_normalized_name'],
        'canonical player info',
    )
    box_candidates = _identity_candidates(mapped_boxscore, 'player_id')
    info_candidates = _identity_candidates(mapped_info, '_canonical_player_id')
    keys = sorted(
        set(box_candidates) | set(info_candidates),
        key=lambda key: (int(key[0]), str(key[1]), int(key[2])),
    )
    rows = []
    for match_id, team_name, jersey_number in keys:
        box_ids = box_candidates.get((match_id, team_name, jersey_number), ())
        info_ids = info_candidates.get((match_id, team_name, jersey_number), ())
        if len(box_ids) == 1:
            player_id = box_ids[0]
            status = 'boxscore'
        elif len(box_ids) > 1:
            player_id = pd.NA
            status = 'ambiguous_boxscore'
        elif len(info_ids) == 1:
            player_id = info_ids[0]
            status = 'roster'
        elif len(info_ids) > 1:
            player_id = pd.NA
            status = 'ambiguous_roster'
        else:  # pragma: no cover - keys originate in one of these mappings
            player_id = pd.NA
            status = 'unmapped'
        rows.append({
            'match_id': match_id,
            'team_name': team_name,
            '_jersey_number': int(jersey_number),
            'player_id': player_id,
            'identity_status': status,
            'boxscore_candidate_ids': '|'.join(box_ids),
            'roster_candidate_ids': '|'.join(info_ids),
        })
    identity_map = pd.DataFrame(rows)
    if identity_map.duplicated(IDENTITY_KEY).any():
        raise AssertionError("Identity map is not unique by match/team/jersey")
    identity_map = _apply_audited_pbp_identity_overrides(
        identity_map, mapped_boxscore, mapped_info, season, audited_overrides
    )
    if not identity_map.empty:
        identity_map['player_id'] = identity_map['player_id'].astype('string')
        identity_map['_jersey_number'] = identity_map[
            '_jersey_number'
        ].astype('Int64')
    return identity_map.sort_values(IDENTITY_KEY, kind='stable').reset_index(
        drop=True
    )


def build_action_table(
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    identity_map: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Construct one deterministic row per public PBP action."""
    required = [
        'match_id', 'season', 'home_team_name', 'away_team_name',
        'team_involved', 'jersey_number', 'action', 'outcome', 'set',
        'point_number', 'point_winner', 'home_score', 'away_score',
    ]
    _require_columns(pbp, required, 'PBP')
    _require_columns(
        schedule, ['match_id', 'home_team', 'away_team'], 'schedule'
    )
    if pbp.empty:
        raise ValueError("PBP input is empty")
    unknown_actions = sorted(set(pbp['action'].dropna()) - set(ACTION_MAP))
    unknown_outcomes = sorted(set(pbp['outcome'].dropna()) - set(OUTCOME_MAP))
    if unknown_actions or unknown_outcomes:
        raise ValueError(
            f"Unknown PBP codes: actions={unknown_actions}, "
            f"outcomes={unknown_outcomes}"
        )
    if pbp[['action', 'outcome']].isna().any().any():
        raise ValueError("PBP contains missing action or outcome codes")

    actions = pbp.copy().reset_index(drop=True)
    actions['source_row_number'] = np.arange(len(actions), dtype=np.int64)
    if not pd.to_numeric(actions['season'], errors='coerce').eq(int(season)).all():
        raise ValueError(f"PBP contains rows outside requested season {season}")
    actions = actions.merge(
        schedule[['match_id', 'home_team', 'away_team']],
        on='match_id', how='left', validate='many_to_one', sort=False,
    )
    if actions[['home_team', 'away_team']].isna().any().any():
        raise ValueError("PBP contains match IDs absent from regular schedule")
    source_teams_match = (
        actions['home_team_name'].eq(actions['home_team'])
        & actions['away_team_name'].eq(actions['away_team'])
    )
    if not source_teams_match.all():
        examples = actions.loc[
            ~source_teams_match,
            ['match_id', 'home_team_name', 'away_team_name',
             'home_team', 'away_team'],
        ].drop_duplicates().head(10).to_dict('records')
        raise ValueError(f"PBP and schedule team names disagree: {examples}")
    acting_valid = actions['team_involved'].isin(['home', 'away'])
    winner_valid = actions['point_winner'].isin(['home', 'away'])
    if not acting_valid.all() or not winner_valid.all():
        raise ValueError("PBP contains an acting team or point winner outside the match")

    actions = actions.rename(columns={
        'set': 'set_number',
        'team_involved': 'acting_side',
        'point_winner': 'point_winner_side',
        'home_score': 'post_home_score',
        'away_score': 'post_away_score',
        'action': 'source_action',
        'outcome': 'source_outcome',
    })
    actions['acting_team'] = np.where(
        actions['acting_side'].eq('home'),
        actions['home_team'], actions['away_team'],
    )
    actions['point_winner_team'] = np.where(
        actions['point_winner_side'].eq('home'),
        actions['home_team'], actions['away_team'],
    )
    actions['opponent_team'] = np.where(
        actions['acting_team'].eq(actions['home_team']),
        actions['away_team'], actions['home_team'],
    )
    actions['team_name'] = actions['acting_team']
    actions['canonical_action'] = actions['source_action'].map(ACTION_MAP)
    actions['canonical_outcome'] = actions['source_outcome'].map(OUTCOME_MAP)
    actions['_jersey_number'] = normalize_jersey_numbers(
        actions['jersey_number']
    )
    actions['rally_id'] = (
        actions['season'].astype(str) + ':'
        + actions['match_id'].astype(str) + ':'
        + actions['set_number'].astype(str) + ':'
        + actions['point_number'].astype(str)
    )
    # The release has no event-sequence field.  Its positional CSV order is
    # semantically meaningful and is captured before any merge or sort.
    actions['action_order'] = (
        actions.groupby(RALLY_KEY, sort=False).cumcount() + 1
    ).astype(np.int64)

    identity_columns = [
        *IDENTITY_KEY, 'player_id', 'identity_status',
        'boxscore_candidate_ids', 'roster_candidate_ids',
    ]
    actions = actions.merge(
        identity_map[identity_columns],
        on=IDENTITY_KEY, how='left', validate='many_to_one', sort=False,
    )
    missing_jersey = actions['_jersey_number'].isna()
    actions.loc[
        missing_jersey & actions['identity_status'].isna(), 'identity_status'
    ] = 'anonymous_source_jersey'
    actions.loc[
        ~missing_jersey & actions['identity_status'].isna(), 'identity_status'
    ] = 'unmapped'
    actions['player_id'] = actions['player_id'].astype('string')
    actions = actions.sort_values('source_row_number', kind='stable').reset_index(
        drop=True
    )
    expected_order = actions.groupby(RALLY_KEY, sort=False).cumcount() + 1
    if not actions['action_order'].eq(expected_order).all():
        raise AssertionError("PBP action order changed during canonical joins")

    output_columns = [
        *RALLY_KEY, 'rally_id', 'source_row_number', 'action_order',
        'home_team', 'away_team', 'acting_side', 'acting_team',
        'opponent_team', 'point_winner_side', 'point_winner_team',
        'jersey_number', 'player_id', 'identity_status',
        'boxscore_candidate_ids', 'roster_candidate_ids',
        'source_action', 'source_outcome', 'canonical_action',
        'canonical_outcome', 'post_home_score', 'post_away_score',
    ]
    if 'serve_opportunity_supported' in actions:
        output_columns.append('serve_opportunity_supported')
    return actions[output_columns].copy()


def _unique_group_value(group, column):
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Rally has inconsistent {column}: {values.head(10).tolist()}"
        )
    return values.iloc[0]


def build_rally_table(actions: pd.DataFrame) -> pd.DataFrame:
    """Collapse canonical actions to one score-state row per rally."""
    _require_columns(
        actions,
        [*RALLY_KEY, 'rally_id', 'action_order', 'home_team', 'away_team',
         'acting_team', 'point_winner_team', 'canonical_action',
         'jersey_number', 'post_home_score', 'post_away_score'],
        'action table',
    )
    rows = []
    for key, group in actions.groupby(RALLY_KEY, sort=False, dropna=False):
        ordered = group.sort_values('action_order', kind='stable')
        if not ordered['action_order'].tolist() == list(
            range(1, len(ordered) + 1)
        ):
            raise ValueError(f"Non-consecutive action order for rally {key}")
        serve_rows = ordered[ordered['canonical_action'].eq('serve')]
        if len(serve_rows) > 1 and (
            serve_rows['acting_team'].nunique() != 1
            or serve_rows['jersey_number'].nunique() != 1
        ):
            raise ValueError(
                f"Official point {key} contains conflicting source servers"
            )
        rows.append({
            **dict(zip(RALLY_KEY, key)),
            'rally_id': _unique_group_value(ordered, 'rally_id'),
            'home_team': _unique_group_value(ordered, 'home_team'),
            'away_team': _unique_group_value(ordered, 'away_team'),
            'point_winner_team': _unique_group_value(
                ordered, 'point_winner_team'
            ),
            'post_home_score': int(_unique_group_value(
                ordered, 'post_home_score'
            )),
            'post_away_score': int(_unique_group_value(
                ordered, 'post_away_score'
            )),
            'n_actions': len(ordered),
            'source_serve_present': len(serve_rows) >= 1,
            'source_serve_count': len(serve_rows),
            'replayed_source_sequence': len(serve_rows) > 1,
            'source_serving_team': (
                serve_rows.iloc[0]['acting_team'] if len(serve_rows) else pd.NA
            ),
            'server_jersey_number': (
                serve_rows.iloc[0]['jersey_number'] if len(serve_rows) else pd.NA
            ),
            'server_player_id': (
                serve_rows.iloc[0]['player_id'] if len(serve_rows) else pd.NA
            ),
        })
    rallies = pd.DataFrame(rows)
    if rallies.duplicated(RALLY_KEY).any():
        raise AssertionError("Canonical rally key is not unique")
    if rallies[RALLY_KEY].isna().any().any():
        raise ValueError("Canonical rally key contains missing values")

    output = []
    for _, set_rallies in rallies.groupby(
        ['season', 'match_id', 'set_number'], sort=False
    ):
        set_rallies = set_rallies.sort_values('point_number', kind='stable').copy()
        point_numbers = set_rallies['point_number'].astype(int).tolist()
        if point_numbers != list(range(1, len(point_numbers) + 1)):
            raise ValueError(
                "PBP point numbers are not complete and consecutive for set "
                f"{tuple(set_rallies.iloc[0][['season','match_id','set_number']])}"
            )
        set_rallies['pre_home_score'] = (
            set_rallies['post_home_score'].shift(fill_value=0).astype(int)
        )
        set_rallies['pre_away_score'] = (
            set_rallies['post_away_score'].shift(fill_value=0).astype(int)
        )
        previous_winner = set_rallies['point_winner_team'].shift()
        set_rallies['serving_team'] = set_rallies[
            'source_serving_team'
        ].fillna(previous_winner)
        set_rallies['service_succession_valid'] = (
            set_rallies['serving_team'].eq(previous_winner)
        ).astype('boolean')
        set_rallies.loc[
            previous_winner.isna(), 'service_succession_valid'
        ] = pd.NA
        set_rallies['serving_team_source'] = np.select(
            [
                set_rallies['source_serving_team'].notna(),
                set_rallies['source_serving_team'].isna()
                & previous_winner.notna(),
            ],
            ['pbp_serve', 'previous_point_winner'],
            default='missing',
        )
        set_rallies['receiving_team'] = np.where(
            set_rallies['serving_team'].eq(set_rallies['home_team']),
            set_rallies['away_team'],
            np.where(
                set_rallies['serving_team'].eq(set_rallies['away_team']),
                set_rallies['home_team'], pd.NA,
            ),
        )
        home_won = set_rallies['point_winner_team'].eq(set_rallies['home_team'])
        away_won = set_rallies['point_winner_team'].eq(set_rallies['away_team'])
        set_rallies['score_transition_valid'] = (
            (
                set_rallies['post_home_score']
                == set_rallies['pre_home_score'] + home_won.astype(int)
            )
            & (
                set_rallies['post_away_score']
                == set_rallies['pre_away_score'] + away_won.astype(int)
            )
        )
        output.append(set_rallies)
    rallies = pd.concat(output, ignore_index=True)
    rallies['server_player_id'] = rallies['server_player_id'].astype('string')
    columns = [
        *RALLY_KEY, 'rally_id', 'home_team', 'away_team', 'serving_team',
        'receiving_team', 'point_winner_team', 'pre_home_score',
        'pre_away_score', 'post_home_score', 'post_away_score',
        'score_transition_valid', 'n_actions', 'source_serve_present',
        'source_serve_count', 'replayed_source_sequence',
        'serving_team_source', 'service_succession_valid',
        'server_jersey_number', 'server_player_id',
    ]
    return rallies[columns].copy()


def _event_timestamp(events: pd.DataFrame) -> pd.Series:
    event_time = pd.to_datetime(events['event_time'], utc=True, errors='coerce')
    rally_time = pd.to_datetime(
        events['rally_start_time'], utc=True, errors='coerce'
    )
    return event_time.fillna(rally_time)


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {'true', 'false'}:
        return value.strip().lower() == 'true'
    raise ValueError(f"Unrecognized source boolean: {value!r}")


def _replace_jersey(lineup, outgoing, incoming):
    outgoing = int(outgoing)
    incoming = int(incoming)
    positions = [index for index, value in enumerate(lineup) if value == outgoing]
    if len(positions) != 1 or incoming in lineup:
        raise ValueError(
            f"Cannot replace jersey {outgoing} with {incoming} in {lineup}"
        )
    updated = list(lineup)
    updated[positions[0]] = incoming
    return updated


def _rotate(lineup):
    return [*lineup[1:], lineup[0]]


def _lineup_identity_lookup(identity_map):
    return {
        (match_id, team_name, int(jersey_number)): (player_id, status)
        for match_id, team_name, jersey_number, player_id, status in identity_map[
            ['match_id', 'team_name', '_jersey_number',
             'player_id', 'identity_status']
        ].itertuples(index=False, name=None)
    }


def build_lineup_table(
    rallies: pd.DataFrame,
    events: pd.DataFrame,
    identity_map: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct conservative pre-rally court lineups from public events.

    The event feed does not provide a lineup snapshot on every rally.  We use
    audited set starters plus substitutions/libero replacements and rotate on
    final PBP point winners.  Sets with mismatched event/PBP rally counts or a
    timestamp-free manual correction are intentionally left incomplete.
    """
    starter_columns = [
        *(f'home_team_starter_position_{position}' for position in range(1, 7)),
        *(f'away_team_starter_position_{position}' for position in range(1, 7)),
    ]
    required = [
        'match_id', 'season', 'set', 'event_type', 'event_time',
        'rally_start_time', 'rally_point_winner', 'team_involved',
        'substitute_in_jersey_number', 'substitute_out_jersey_number',
        'libero_enters', 'libero_jersey_number',
        'libero_substitute_jersey_number', *starter_columns,
    ]
    _require_columns(events, required, 'events log')
    if rallies.empty:
        raise ValueError("Rally table is empty")
    events = events.copy().reset_index(drop=True)
    events['_source_event_row'] = np.arange(len(events), dtype=np.int64)
    events['_event_timestamp'] = _event_timestamp(events)
    known_event_types = {
        'rally', 'libero', 'substitution', 'timeout', 'technicalTimeout',
        'videoChallenge', 'manualChange', 'sanction', 'delay',
        'improperRequest', 'injury', 'newLibero', 'captain',
        'faultAdmission',
    }
    unknown_event_types = sorted(set(events['event_type'].dropna()) - known_event_types)
    if unknown_event_types:
        raise ValueError(f"Unknown event-log types: {unknown_event_types}")
    nonmanual_missing_time = (
        events['_event_timestamp'].isna()
        & ~events['event_type'].eq('manualChange')
    )
    if nonmanual_missing_time.any():
        raise ValueError("Non-manual event rows lack an ordering timestamp")

    identity_lookup = _lineup_identity_lookup(identity_map)
    output = []
    event_groups = {
        key: group for key, group in events.groupby(
            ['season', 'match_id', 'set'], sort=False
        )
    }
    rally_set_keys = set(
        rallies[['season', 'match_id', 'set_number']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    extra_event_sets = set(event_groups) - rally_set_keys
    if extra_event_sets:
        raise ValueError(
            "Event log contains sets with no PBP official points: "
            f"{sorted(extra_event_sets)[:10]}"
        )
    for key, set_rallies in rallies.groupby(
        ['season', 'match_id', 'set_number'], sort=False
    ):
        event_group = event_groups.get(key)
        set_rallies = set_rallies.sort_values('point_number', kind='stable')
        failure = None
        if event_group is None:
            failure = 'missing_event_set'
        elif event_group['event_type'].eq('manualChange').any():
            # These rare source corrections carry no timestamp, so their exact
            # position cannot be merged into the rally stream without guessing.
            failure = 'manual_change_present'
        elif event_group['event_type'].eq('rally').sum() != len(set_rallies):
            failure = 'event_rally_count_mismatch'

        def incomplete_rows(reason):
            for rally in set_rallies.itertuples(index=False):
                row = {column: getattr(rally, column) for column in RALLY_KEY}
                row.update({
                    'rally_id': rally.rally_id,
                    'lineup_status': reason,
                    'lineup_state_valid': False,
                    'lineup_identity_complete': False,
                    'lineup_complete': False,
                    'server_on_court': pd.NA,
                    'event_rally_winner_matches': pd.NA,
                })
                for side in ('home', 'away'):
                    for position in range(1, 7):
                        row[f'{side}_p{position}_jersey'] = pd.NA
                        row[f'{side}_p{position}_player_id'] = pd.NA
                output.append(row)

        if failure is not None:
            incomplete_rows(failure)
            continue

        starters = {}
        starter_failure = None
        for side in ('home', 'away'):
            values = []
            for position in range(1, 7):
                column = f'{side}_team_starter_position_{position}'
                unique = event_group[column].dropna().unique()
                if len(unique) != 1:
                    starter_failure = 'ambiguous_set_starters'
                    break
                values.append(int(unique[0]))
            if len(set(values)) != 6:
                starter_failure = 'invalid_set_starters'
            starters[side] = values
        if starter_failure:
            incomplete_rows(starter_failure)
            continue

        lineups = {side: list(values) for side, values in starters.items()}
        ordered_events = event_group.sort_values(
            ['_event_timestamp', '_source_event_row'], kind='stable'
        )
        rally_iterator = iter(set_rallies.itertuples(index=False))
        current_rally = next(rally_iterator, None)
        state_failure = None
        for event in ordered_events.itertuples(index=False):
            event_type = event.event_type
            if event_type in {'substitution', 'libero'}:
                if state_failure:
                    continue
                side = event.team_involved
                if side not in {'home', 'away'}:
                    state_failure = 'unknown_event_team'
                    continue
                try:
                    if event_type == 'substitution':
                        outgoing = event.substitute_out_jersey_number
                        incoming = event.substitute_in_jersey_number
                    elif _as_bool(event.libero_enters):
                        outgoing = event.libero_substitute_jersey_number
                        incoming = event.libero_jersey_number
                    else:
                        outgoing = event.libero_jersey_number
                        incoming = event.libero_substitute_jersey_number
                    if pd.isna(outgoing) or pd.isna(incoming):
                        raise ValueError("missing replacement jersey")
                    lineups[side] = _replace_jersey(
                        lineups[side], outgoing, incoming
                    )
                except ValueError:
                    state_failure = 'invalid_lineup_operation'
                continue
            if event_type != 'rally':
                continue
            if current_rally is None:
                raise AssertionError("Event feed contains surplus rally rows")

            row = {column: getattr(current_rally, column) for column in RALLY_KEY}
            row['rally_id'] = current_rally.rally_id
            row['lineup_status'] = state_failure or 'complete'
            row['lineup_state_valid'] = state_failure is None
            row['event_rally_winner_matches'] = (
                event.rally_point_winner
                == ('home' if current_rally.point_winner_team
                    == current_rally.home_team else 'away')
            )
            serving_side = (
                'home' if current_rally.serving_team == current_rally.home_team
                else 'away' if current_rally.serving_team == current_rally.away_team
                else None
            )
            server_on_court = pd.NA
            if state_failure is None and not pd.isna(
                current_rally.server_jersey_number
            ):
                server_on_court = int(
                    current_rally.server_jersey_number
                ) in lineups[serving_side]
                if not server_on_court:
                    state_failure = 'server_not_on_court'
                    row['lineup_status'] = state_failure
                    row['lineup_state_valid'] = False
            row['server_on_court'] = server_on_court

            identity_complete = state_failure is None
            seen_player_ids = {'home': set(), 'away': set()}
            for side in ('home', 'away'):
                team_name = getattr(current_rally, f'{side}_team')
                for position, jersey in enumerate(lineups[side], start=1):
                    row[f'{side}_p{position}_jersey'] = jersey
                    player_id, status = identity_lookup.get(
                        (current_rally.match_id, team_name, int(jersey)),
                        (pd.NA, 'unmapped'),
                    )
                    if pd.isna(player_id) or status.startswith('ambiguous'):
                        identity_complete = False
                    else:
                        seen_player_ids[side].add(str(player_id))
                    row[f'{side}_p{position}_player_id'] = player_id
                if len(seen_player_ids[side]) != 6:
                    identity_complete = False
            if seen_player_ids['home'] & seen_player_ids['away']:
                raise ValueError("A player appears simultaneously for both teams")
            row['lineup_identity_complete'] = identity_complete
            row['lineup_complete'] = (
                row['lineup_state_valid'] and identity_complete
            )
            if row['lineup_state_valid'] and not identity_complete:
                row['lineup_status'] = 'identity_incomplete'
            output.append(row)

            # Rotation is determined from the final PBP point winner rather than
            # the occasionally pre-correction event-log rally winner.
            if state_failure is None and (
                current_rally.point_winner_team != current_rally.serving_team
            ):
                receiving_side = 'away' if serving_side == 'home' else 'home'
                lineups[receiving_side] = _rotate(lineups[receiving_side])
            current_rally = next(rally_iterator, None)
        if current_rally is not None:
            raise AssertionError("Event feed ended before all rallies were emitted")

    lineups = pd.DataFrame(output)
    if len(lineups) != len(rallies):
        raise AssertionError("Lineup table does not contain one row per rally")
    if lineups.duplicated(RALLY_KEY).any():
        raise AssertionError("Lineup table key is not unique")
    player_columns = [
        f'{side}_p{position}_player_id'
        for side in ('home', 'away') for position in range(1, 7)
    ]
    for column in player_columns:
        lineups[column] = lineups[column].astype('string')
    return lineups.sort_values(RALLY_KEY, kind='stable').reset_index(drop=True)


def action_outcome_mapping() -> pd.DataFrame:
    """Return the complete explicit source-to-canonical vocabulary."""
    return pd.DataFrame(
        [
            {
                'source_action': source_action,
                'canonical_action': canonical_action,
                'source_outcome': source_outcome,
                'canonical_outcome': canonical_outcome,
            }
            for source_action, canonical_action in ACTION_MAP.items()
            for source_outcome, canonical_outcome in OUTCOME_MAP.items()
        ]
    )
