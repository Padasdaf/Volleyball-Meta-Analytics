"""Shared mechanics for audited pyvolleydata league adapters.

Competition scope, aliases, and source corrections remain owned by each league
adapter.  This module only reconciles the common public table shapes after an
adapter has made those league-specific decisions.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from canonical_data import normalize_jersey_numbers


PARTICIPATION_STAT_COLUMNS = (
    'serves', 'serve_errors', 'serve_aces', 'attack_attempts',
    'attack_errors', 'attack_kills', 'receptions', 'reception_errors',
    'block_points', 'block_touches', 'earned_points', 'net_points',
    'assists', 'successful_digs', 'spike_hp', 'points',
)


def normalized_name(value) -> str:
    if pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKD', str(value)).encode(
        'ascii', 'ignore'
    ).decode('ascii')
    return ''.join(character for character in text.lower() if character.isalnum())


def clean_source_player_id(value):
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    if cleaned.endswith('.0') and cleaned[:-2].isdigit():
        cleaned = cleaned[:-2]
    if cleaned.lower() in {'', '-', '0', 'nan', 'none'}:
        return None
    return cleaned


def _unique_lookup(frame, keys, value):
    grouped = (
        frame.dropna(subset=[*keys, value])
        .groupby(keys, dropna=False)[value]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
    )
    return {
        key if isinstance(key, tuple) else (key,): values[0]
        for key, values in grouped.items() if len(values) == 1
    }


def map_public_player_ids(
    boxscore_df: pd.DataFrame,
    info_df: pd.DataFrame,
    *,
    league: str,
    name_aliases=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map set boxscores through persistent roster IDs without guessing.

    Source player IDs are league-namespaced because pyvolleydata does not state
    that the identifiers share one global namespace across providers.
    """
    league = str(league).lower()
    boxscore = boxscore_df.copy()
    info = info_df.copy()
    name_aliases = name_aliases or {}
    boxscore['_normalized_name'] = boxscore['player_name'].map(
        normalized_name
    ).replace(name_aliases)
    info['_normalized_name'] = info['player_name'].map(
        normalized_name
    ).replace(name_aliases)
    boxscore['_jersey_number'] = normalize_jersey_numbers(
        boxscore['player_number']
    )
    info['_jersey_number'] = normalize_jersey_numbers(info['jersey_number'])
    info['_source_player_id'] = info['player_id'].map(clean_source_player_id)
    info['_canonical_player_id'] = info['_source_player_id'].map(
        lambda value: f'{league}:{value}' if value is not None else pd.NA
    ).astype('string')

    source = info[info['_canonical_player_id'].notna()].copy()
    exact = _unique_lookup(
        source,
        ['match_id', 'team_name', '_jersey_number', '_normalized_name'],
        '_canonical_player_id',
    )
    match_name = _unique_lookup(
        source,
        ['match_id', 'team_name', '_normalized_name'],
        '_canonical_player_id',
    )
    match_jersey = _unique_lookup(
        source,
        ['match_id', 'team_name', '_jersey_number'],
        '_canonical_player_id',
    )
    season_name = _unique_lookup(
        source, ['_normalized_name'], '_canonical_player_id'
    )

    player_ids = []
    methods = []
    for match_id, team, name, jersey in boxscore[[
        'match_id', 'team_name', '_normalized_name', '_jersey_number'
    ]].itertuples(index=False, name=None):
        candidates = []
        if not pd.isna(jersey):
            candidates.extend([
                ('match_team_jersey_name', exact.get(
                    (match_id, team, jersey, name)
                )),
                ('match_team_jersey', match_jersey.get(
                    (match_id, team, jersey)
                )),
            ])
        candidates.extend([
            ('match_team_name', match_name.get((match_id, team, name))),
            ('season_name_fallback', season_name.get((name,))),
        ])
        resolved = [(method, value) for method, value in candidates if value]
        unique = {value for _, value in resolved}
        if len(unique) > 1:
            raise ValueError(
                'Conflicting public identity evidence for '
                f'{(match_id, team, name, jersey)}: {resolved}'
            )
        if resolved:
            method, player_id = resolved[0]
        else:
            method, player_id = 'unresolved', pd.NA
        player_ids.append(player_id)
        methods.append(method)
    boxscore['player_id'] = pd.Series(
        player_ids, index=boxscore.index, dtype='string'
    )
    boxscore['player_id_mapping_method'] = methods
    unresolved = boxscore[boxscore['player_id'].isna()]
    if not unresolved.empty:
        examples = unresolved[[
            'match_id', 'team_name', 'player_name', 'player_number'
        ]].drop_duplicates().head(20).to_dict('records')
        raise ValueError(f'Public boxscore rows lack persistent IDs: {examples}')
    return boxscore, info


def _starter_participation_keys(info: pd.DataFrame) -> set[tuple]:
    keys = set()
    for set_number in range(1, 6):
        column = f'set_{set_number}_is_starter'
        if column not in info:
            continue
        selected = info[info[column].eq(True)].dropna(
            subset=['_canonical_player_id']
        )
        keys.update(
            (match_id, team, set_number, str(player_id))
            for match_id, team, player_id in selected[[
                'match_id', 'team_name', '_canonical_player_id'
            ]].itertuples(index=False, name=None)
        )
    return keys


def _event_participation_jerseys(
    events_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> set[tuple]:
    teams = schedule_df[['match_id', 'home_team', 'away_team']].drop_duplicates()
    events = events_df.merge(teams, on='match_id', validate='many_to_one')
    keys = set()
    for side in ('home', 'away'):
        team_column = f'{side}_team'
        columns = [
            *(f'{side}_team_starter_position_{i}' for i in range(1, 7)),
            *(f'{side}_team_p{i}' for i in range(1, 7)),
        ]
        for column in columns:
            if column not in events:
                continue
            jerseys = normalize_jersey_numbers(events[column])
            valid = jerseys.notna() & events['set'].notna()
            keys.update(
                (match_id, team, int(set_number), int(jersey))
                for match_id, team, set_number, jersey in zip(
                    events.loc[valid, 'match_id'],
                    events.loc[valid, team_column],
                    events.loc[valid, 'set'],
                    jerseys.loc[valid],
                )
            )
    event_columns = {
        'substitution': (
            'substitute_in_jersey_number', 'substitute_out_jersey_number'
        ),
        'libero': ('libero_jersey_number', 'libero_substitute_jersey_number'),
    }
    for event_type, columns in event_columns.items():
        selected = events[events['event_type'].eq(event_type)].copy()
        if selected.empty:
            continue
        selected['_team_name'] = np.where(
            selected['team_involved'].eq('home'),
            selected['home_team'], selected['away_team'],
        )
        for column in columns:
            jerseys = normalize_jersey_numbers(selected[column])
            valid = jerseys.notna() & selected['set'].notna()
            keys.update(
                (match_id, team, int(set_number), int(jersey))
                for match_id, team, set_number, jersey in zip(
                    selected.loc[valid, 'match_id'],
                    selected.loc[valid, '_team_name'],
                    selected.loc[valid, 'set'],
                    jerseys.loc[valid],
                )
            )
    return keys


def mark_public_played_sets(
    boxscore_df: pd.DataFrame,
    info_df: pd.DataFrame,
    events_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the audited evidence-union played-set contract."""
    boxscore = boxscore_df.copy()
    missing = [
        column for column in PARTICIPATION_STAT_COLUMNS
        if column not in boxscore
    ]
    if missing:
        raise ValueError(f'Boxscore is missing participation fields: {missing}')
    statistic = boxscore[list(PARTICIPATION_STAT_COLUMNS)].notna().any(axis=1)
    position = boxscore['set_starting_position'].notna()
    starter_keys = _starter_participation_keys(info_df)
    event_keys = _event_participation_jerseys(events_df, schedule_df)
    counts = (
        info_df.dropna(subset=['_jersey_number', '_canonical_player_id'])
        .groupby(['match_id', 'team_name', '_jersey_number'])[
            '_canonical_player_id'
        ].nunique()
    )
    ambiguous = {key for key, count in counts.items() if count > 1}
    starter = pd.Series([
        (match_id, team, int(set_number), str(player_id)) in starter_keys
        for match_id, team, set_number, player_id in boxscore[[
            'match_id', 'team_name', 'set_number', 'player_id'
        ]].itertuples(index=False, name=None)
    ], index=boxscore.index)
    event = pd.Series([
        not pd.isna(jersey)
        and (match_id, team, jersey) not in ambiguous
        and (match_id, team, int(set_number), int(jersey)) in event_keys
        for match_id, team, set_number, jersey in boxscore[[
            'match_id', 'team_name', 'set_number', '_jersey_number'
        ]].itertuples(index=False, name=None)
    ], index=boxscore.index)
    boxscore['played_set'] = statistic | position | starter | event
    boxscore['played_set_statistic_record'] = statistic
    boxscore['played_set_boxscore_position'] = position
    boxscore['played_set_starter_record'] = starter
    boxscore['played_set_event_record'] = event
    return boxscore


def build_public_player_metadata(
    mapped_info: pd.DataFrame,
    player_sets: pd.DataFrame,
) -> pd.DataFrame:
    """Return canonical match-local metadata from source-ID evidence."""
    rows = mapped_info.copy()
    rows['source_player_id'] = rows['_source_player_id'].astype('string')
    rows['player_id'] = rows['_canonical_player_id'].astype('string')
    rows['jersey'] = rows['_jersey_number']
    rows['position'] = pd.to_numeric(rows['primary_position'], errors='coerce')
    rows['identity_provenance'] = np.where(
        rows['_source_player_id'].notna(), 'source_player_id',
        'unresolved_source_identity',
    )
    columns = [
        'match_id', 'team_name', 'player_id', 'source_player_id',
        'player_name', 'jersey', 'position', 'primary_position', 'is_libero',
        'identity_provenance',
    ]
    rows = rows[columns]
    additions = player_sets[
        ~player_sets['player_id'].isin(rows['player_id'])
    ][[
        'match_id', 'team_name', 'player_id', 'player_name',
        '_jersey_number', 'player_id_mapping_method',
    ]].drop_duplicates()
    if not additions.empty:
        additions = additions.rename(columns={'_jersey_number': 'jersey'})
        additions['source_player_id'] = additions['player_id'].str.split(
            ':', n=1
        ).str[-1]
        additions['position'] = np.nan
        additions['primary_position'] = np.nan
        additions['is_libero'] = False
        additions['identity_provenance'] = additions[
            'player_id_mapping_method'
        ]
        rows = pd.concat([rows, additions[columns]], ignore_index=True)

    def canonical_name(values):
        counts = values.dropna().astype(str).value_counts()
        if counts.empty:
            return pd.NA
        return sorted(counts[counts.eq(counts.max())].index)[0]

    key = ['match_id', 'team_name', 'player_id']
    return rows.groupby(key, as_index=False, dropna=False).agg(
        source_player_id=(
            'source_player_id',
            lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NA,
        ),
        player_name=('player_name', canonical_name),
        jersey=(
            'jersey', lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NA
        ),
        position=(
            'position', lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
        ),
        primary_position=(
            'primary_position',
            lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
        ),
        is_libero=(
            'is_libero',
            lambda x: bool(x.dropna().iloc[0]) if x.notna().any() else False,
        ),
        identity_provenance=(
            'identity_provenance',
            lambda x: sorted(x.dropna().astype(str).unique())[0],
        ),
    ).sort_values(key, kind='stable').reset_index(drop=True)


def attach_team_assignment_ids(
    schedule: pd.DataFrame,
    player_sets: pd.DataFrame,
    metadata: pd.DataFrame,
    actions: pd.DataFrame,
    rallies: pd.DataFrame,
    lineups: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Expose contextual team assignments without changing display teams."""
    required = ['home_team_assignment_id', 'away_team_assignment_id']
    if any(column not in schedule for column in required):
        raise ValueError('Canonical schedule lacks team-assignment identifiers')
    assignments = []
    for side in ('home', 'away'):
        assignments.append(schedule[[
            'match_id', f'{side}_team', f'{side}_team_assignment_id'
        ]].rename(columns={
            f'{side}_team': 'team_name',
            f'{side}_team_assignment_id': 'team_assignment_id',
        }))
    assignment_map = pd.concat(assignments, ignore_index=True)
    if assignment_map.duplicated(['match_id', 'team_name']).any():
        raise ValueError('Team assignment is not unique within match/team')
    player_sets = player_sets.merge(
        assignment_map, on=['match_id', 'team_name'], how='left',
        validate='many_to_one',
    )
    metadata = metadata.merge(
        assignment_map, on=['match_id', 'team_name'], how='left',
        validate='many_to_one',
    )
    actions = actions.merge(
        assignment_map.rename(columns={
            'team_name': 'acting_team',
            'team_assignment_id': 'acting_team_assignment_id',
        }),
        on=['match_id', 'acting_team'], how='left', validate='many_to_one',
    )
    actions = actions.merge(
        assignment_map.rename(columns={
            'team_name': 'opponent_team',
            'team_assignment_id': 'opponent_team_assignment_id',
        }),
        on=['match_id', 'opponent_team'], how='left', validate='many_to_one',
    )
    rally_assignments = schedule[[
        'match_id', 'home_team_assignment_id', 'away_team_assignment_id'
    ]]
    rallies = rallies.merge(
        rally_assignments, on='match_id', how='left', validate='many_to_one'
    )
    for role in ('serving', 'receiving', 'point_winner'):
        rallies = rallies.merge(
            assignment_map.rename(columns={
                'team_name': f'{role}_team',
                'team_assignment_id': f'{role}_team_assignment_id',
            }),
            on=['match_id', f'{role}_team'], how='left', validate='many_to_one',
        )
    lineups = lineups.merge(
        rally_assignments, on='match_id', how='left', validate='many_to_one'
    )
    for label, frame, column in (
        ('player sets', player_sets, 'team_assignment_id'),
        ('player metadata', metadata, 'team_assignment_id'),
        ('actions', actions, 'acting_team_assignment_id'),
        ('rallies', rallies, 'home_team_assignment_id'),
        ('lineups', lineups, 'home_team_assignment_id'),
    ):
        if frame[column].isna().any():
            raise ValueError(f'{label} contains a missing team assignment')
    return player_sets, metadata, actions, rallies, lineups


def build_public_canonical_from_filtered_frames(
    *,
    league: str,
    season: int,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    events: pd.DataFrame,
    player_info: pd.DataFrame,
    player_boxscore: pd.DataFrame,
    name_aliases=None,
):
    """Build a CanonicalSeason after league-specific filtering/normalization."""
    from canonical_data import CanonicalSeason
    from pbp_foundation import (
        build_action_table,
        build_lineup_table,
        build_player_identity_map,
        build_rally_table,
    )

    player_set_source = player_boxscore.merge(
        schedule[[
            'match_id', 'phase', 'competition_scope',
            'home_team', 'away_team',
        ]],
        on='match_id', how='left', validate='many_to_one',
    )
    valid_team = (
        player_set_source['team_name'].eq(player_set_source['home_team'])
        | player_set_source['team_name'].eq(player_set_source['away_team'])
    )
    if not valid_team.all():
        examples = player_set_source.loc[
            ~valid_team, ['match_id', 'team_name', 'home_team', 'away_team']
        ].drop_duplicates().head(10).to_dict('records')
        raise ValueError(f'Boxscore teams disagree with schedule: {examples}')
    mapped_boxscore, mapped_info = map_public_player_ids(
        player_set_source, player_info, league=league,
        name_aliases=name_aliases,
    )
    player_sets = mark_public_played_sets(
        mapped_boxscore, mapped_info, events, schedule
    )
    identity_map = build_player_identity_map(
        mapped_boxscore, mapped_info, season=int(season)
    )
    actions = build_action_table(pbp, schedule, identity_map, int(season))
    rallies = build_rally_table(actions)
    lineups = build_lineup_table(rallies, events, identity_map)
    invalid_keys = rallies.loc[
        ~rallies['score_transition_valid'],
        ['season', 'match_id', 'set_number', 'point_number'],
    ]
    actions = actions.merge(
        invalid_keys.assign(_invalid_score_transition=True),
        on=['season', 'match_id', 'set_number', 'point_number'],
        how='left', validate='many_to_one',
    )
    actions['sequence_source_valid'] = ~actions[
        '_invalid_score_transition'
    ].eq(True)
    actions = actions.drop(columns='_invalid_score_transition')
    if not invalid_keys.empty:
        invalid_sets = invalid_keys[['season', 'match_id', 'set_number']].drop_duplicates()
        invalid_sets['_invalid_score_set'] = True
        lineups = lineups.merge(
            invalid_sets,
            on=['season', 'match_id', 'set_number'],
            how='left', validate='many_to_one',
        )
        invalid_lineup = lineups['_invalid_score_set'].eq(True)
        lineups.loc[invalid_lineup, 'lineup_status'] = (
            'score_winner_inconsistency'
        )
        lineups.loc[
            invalid_lineup,
            ['lineup_state_valid', 'lineup_identity_complete', 'lineup_complete'],
        ] = False
        lineups = lineups.drop(columns='_invalid_score_set')
    metadata = build_public_player_metadata(mapped_info, player_sets)
    player_sets, metadata, actions, rallies, lineups = attach_team_assignment_ids(
        schedule, player_sets, metadata, actions, rallies, lineups
    )
    return CanonicalSeason(
        league=league,
        season=int(season),
        schedule=schedule,
        player_sets=player_sets,
        player_metadata=metadata,
        actions=actions,
        rallies=rallies,
        lineups=lineups,
        identity_map=identity_map,
        source_actions=pbp,
        source_events=events,
        source_player_info=player_info,
        source_player_boxscore=player_boxscore,
    )
