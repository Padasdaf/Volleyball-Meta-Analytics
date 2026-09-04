"""Athletes Unlimited source scope and contextual-squad canonical adapter."""

from __future__ import annotations

import pandas as pd
from pyvolleydata.get_data import (
    load_events_log,
    load_pbp,
    load_player_boxscore,
    load_player_info,
    load_schedule,
)

from canonical_data import require_columns
from pyvolleydata_adapter import (
    build_public_canonical_from_filtered_frames,
    normalized_name,
)


SUPPORTED_AU_SEASONS = (2022, 2023, 2024, 2025)
EXPECTED_ANALYSIS_MATCH_COUNTS = {2022: 30, 2023: 30, 2024: 30, 2025: 24}
AU_SEPARATE_GOLDEN_SET_MATCH_IDS = {2025: frozenset({2249996})}
NORMAL_AU_SET_NUMBERS = frozenset({1, 2, 3})
AU_SEQUENCE_SOURCE_EXCLUSIONS = {
    # These opening-week match IDs contain color-team test PBP/events and
    # roster data under championship schedule/boxscore IDs.
    2023: frozenset({2122427, 2122428, 2122429, 2122430}),
}
AU_UNUSABLE_SEQUENCE_SETS = {
    # Each source PBP omits one complete official point, making lineup rotation
    # through the remainder of that set unknowable without fabrication.
    2022: frozenset({(2106114, 2)}),
    2023: frozenset({(2122441, 2)}),
}
AU_NAME_ALIASES = {
    'taylorreid': 'taylormorgan',
    'madikingdonrishel': 'madirishel',
}


def _squad_assignment_id(season, phase, team):
    return f'au:{season}:{normalized_name(phase)}:{normalized_name(team)}'


def filter_au_analysis_sources(
    season: int,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    events: pd.DataFrame,
    player_info: pd.DataFrame,
    player_boxscore: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Retain AU championship matches and the three normal sets only."""
    season = int(season)
    if season not in SUPPORTED_AU_SEASONS:
        raise ValueError(f'Supported AU seasons are {SUPPORTED_AU_SEASONS}')
    require_columns(
        schedule,
        ['match_id', 'season', 'phase', 'home_team', 'away_team'],
        'AU schedule',
    )
    schedule = schedule.copy()
    if schedule['match_id'].duplicated().any():
        raise ValueError('AU schedule match_id is not unique')
    phase_prefix = f'AU VB {season} '
    schedule = schedule[
        schedule['phase'].astype(str).str.startswith(phase_prefix)
    ].copy()
    golden_match_ids = AU_SEPARATE_GOLDEN_SET_MATCH_IDS.get(
        season, frozenset()
    )
    if not golden_match_ids.issubset(set(schedule['match_id'])):
        raise ValueError('Audited AU Golden Set schedule row disappeared')
    schedule = schedule[~schedule['match_id'].isin(golden_match_ids)].copy()
    expected = EXPECTED_ANALYSIS_MATCH_COUNTS[season]
    if len(schedule) != expected:
        raise ValueError(
            f'AU {season} analysis scope has {len(schedule)} matches; '
            f'expected {expected}'
        )
    schedule['competition_scope'] = 'championship_normal_sets'
    schedule['scope_provenance'] = 'audited_au_format'
    schedule['home_team_assignment_id'] = [
        _squad_assignment_id(season, phase, team)
        for phase, team in zip(schedule['phase'], schedule['home_team'])
    ]
    schedule['away_team_assignment_id'] = [
        _squad_assignment_id(season, phase, team)
        for phase, team in zip(schedule['phase'], schedule['away_team'])
    ]
    match_ids = set(schedule['match_id'])

    pbp = pbp[pbp['match_id'].isin(match_ids)].copy()
    events = events[events['match_id'].isin(match_ids)].copy()
    player_info = player_info[player_info['match_id'].isin(match_ids)].copy()
    player_boxscore = player_boxscore[
        player_boxscore['match_id'].isin(match_ids)
    ].copy()
    for label, frame in (
        ('player boxscore', player_boxscore),
    ):
        if set(frame['match_id']) != match_ids:
            missing = sorted(match_ids - set(frame['match_id']))
            raise ValueError(f'AU {label} is missing matches: {missing[:10]}')

    sequence_exclusions = AU_SEQUENCE_SOURCE_EXCLUSIONS.get(
        season, frozenset()
    )
    pbp = pbp[~pbp['match_id'].isin(sequence_exclusions)].copy()
    events = events[~events['match_id'].isin(sequence_exclusions)].copy()
    sequence_match_ids = match_ids - set(sequence_exclusions)
    for label, frame in (('PBP', pbp), ('events', events)):
        if set(frame['match_id']) != sequence_match_ids:
            missing = sorted(sequence_match_ids - set(frame['match_id']))
            raise ValueError(f'AU {label} is missing usable matches: {missing[:10]}')
    unusable_sets = AU_UNUSABLE_SEQUENCE_SETS.get(season, frozenset())
    if unusable_sets:
        pbp_keys = pd.MultiIndex.from_frame(pbp[['match_id', 'set']])
        event_keys = pd.MultiIndex.from_frame(events[['match_id', 'set']])
        excluded_index = pd.MultiIndex.from_tuples(
            sorted(unusable_sets), names=['match_id', 'set']
        )
        pbp = pbp[~pbp_keys.isin(excluded_index)].copy()
        events = events[~event_keys.isin(excluded_index)].copy()

    # AU matches consist of three ordinary sets. Source set 4 is a Golden Set
    # tiebreak and does not contribute ordinary player-stat/performance totals.
    pbp = pbp[pd.to_numeric(pbp['set']).isin(NORMAL_AU_SET_NUMBERS)].copy()
    events = events[
        pd.to_numeric(events['set'], errors='coerce').isin(NORMAL_AU_SET_NUMBERS)
        | events['set'].isna()
    ].copy()
    player_boxscore = player_boxscore[
        pd.to_numeric(player_boxscore['set_number']).isin(NORMAL_AU_SET_NUMBERS)
    ].copy()

    valid_teams = pd.concat([
        schedule[['match_id', 'home_team']].rename(
            columns={'home_team': 'team_name'}
        ),
        schedule[['match_id', 'away_team']].rename(
            columns={'away_team': 'team_name'}
        ),
    ]).drop_duplicates()
    # Four 2023 opening-week match IDs carry stale color-team roster rows from
    # the test event. They contradict both schedule and boxscore and are not
    # source evidence for the championship squads.
    player_info = player_info.merge(
        valid_teams, on=['match_id', 'team_name'], how='inner',
        validate='many_to_one',
    )
    checked_box = player_boxscore.merge(
        valid_teams, on=['match_id', 'team_name'], how='left', indicator=True
    )
    if checked_box['_merge'].ne('both').any():
        raise ValueError('AU boxscore contains a non-match squad')

    # A zero actor jersey means the scorer did not identify the player. The
    # team/action is still observed, so an anonymous serve remains a valid
    # opponent serve opportunity while actor-attributed metrics remain missing.
    jerseys = pd.to_numeric(pbp['jersey_number'], errors='coerce')
    pbp['serve_opportunity_supported'] = pbp['action'].eq('S')
    pbp.loc[jerseys.le(0), 'jersey_number'] = pd.NA
    return schedule, pbp, events, player_info, player_boxscore


def load_au_analysis_sources(season: int) -> tuple[pd.DataFrame, ...]:
    season = int(season)
    return filter_au_analysis_sources(
        season,
        load_schedule(league='au', seasons=season),
        load_pbp(league='au', seasons=season),
        load_events_log(league='au', seasons=season),
        load_player_info(league='au', seasons=season),
        load_player_boxscore(league='au', seasons=season),
    )


def build_au_canonical_season_from_frames(
    season: int,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    events: pd.DataFrame,
    player_info: pd.DataFrame,
    player_boxscore: pd.DataFrame,
):
    filtered = filter_au_analysis_sources(
        season, schedule, pbp, events, player_info, player_boxscore
    )
    return build_public_canonical_from_filtered_frames(
        league='au', season=int(season),
        schedule=filtered[0], pbp=filtered[1], events=filtered[2],
        player_info=filtered[3], player_boxscore=filtered[4],
        name_aliases=AU_NAME_ALIASES,
    )


def build_au_canonical_season(season: int):
    return build_public_canonical_from_filtered_frames(
        league='au', season=int(season),
        **dict(zip(
            ('schedule', 'pbp', 'events', 'player_info', 'player_boxscore'),
            load_au_analysis_sources(int(season)),
        )),
        name_aliases=AU_NAME_ALIASES,
    )
