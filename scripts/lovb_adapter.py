"""LOVB source scope, aliases, and canonical-season adapter."""

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


SUPPORTED_LOVB_SEASONS = (2025, 2026)
EXPECTED_ANALYSIS_MATCH_COUNTS = {2025: 48, 2026: 60}

# 2025: the seven-match LOVB Classic and five-match championship tournament.
# 2026: the six postseason matches. The three 2026 Classic matches are
# explicitly regular-season matches and remain in scope.
LOVB_EXCLUDED_MATCH_IDS = {
    2025: frozenset({
        *range(2161311, 2161318),
        *range(2161391, 2161396),
    }),
    2026: frozenset(range(2251973, 2251979)),
}

LOVB_TEAM_ALIASES = {
    2025: {
        'Atlanta': 'LOVB Atlanta',
        'Austin': 'LOVB Austin',
        'Houston': 'LOVB Houston',
        'Madison': 'LOVB Madison',
        'Nebraska': 'LOVB Nebraska',
        'Salt Lake': 'LOVB Salt Lake',
        'LOVB Omaha': 'LOVB Nebraska',
    },
    2026: {
        'Atlanta': 'LOVB Atlanta',
        'Austin': 'LOVB Austin',
        'Houston': 'LOVB Houston',
        'Madison': 'LOVB Madison',
        'Nebraska': 'LOVB Nebraska',
        'Salt Lake': 'LOVB Salt Lake',
    },
}


def _normalize_teams(frame: pd.DataFrame, columns, aliases) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].replace(aliases)
    return result


def filter_lovb_analysis_sources(
    season: int,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    events: pd.DataFrame,
    player_info: pd.DataFrame,
    player_boxscore: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Apply the audited LOVB regular-season scope and club aliases."""
    season = int(season)
    if season not in SUPPORTED_LOVB_SEASONS:
        raise ValueError(f'Supported LOVB seasons are {SUPPORTED_LOVB_SEASONS}')
    require_columns(
        schedule,
        ['match_id', 'season', 'date', 'phase', 'home_team', 'away_team'],
        'LOVB schedule',
    )
    schedule = schedule.copy()
    if schedule['match_id'].duplicated().any():
        raise ValueError('LOVB schedule match_id is not unique')
    excluded = LOVB_EXCLUDED_MATCH_IDS[season]
    unknown_exclusions = excluded - set(schedule['match_id'])
    if unknown_exclusions:
        raise ValueError(
            f'LOVB {season} audited exclusions disappeared: '
            f'{sorted(unknown_exclusions)}'
        )
    schedule = schedule[~schedule['match_id'].isin(excluded)].copy()
    expected = EXPECTED_ANALYSIS_MATCH_COUNTS[season]
    if len(schedule) != expected:
        raise ValueError(
            f'LOVB {season} analysis scope has {len(schedule)} matches; '
            f'expected {expected}'
        )
    aliases = LOVB_TEAM_ALIASES[season]
    schedule = _normalize_teams(schedule, ['home_team', 'away_team'], aliases)
    schedule['competition_scope'] = 'regular_season'
    schedule['scope_provenance'] = 'audited_lovb_schedule'
    schedule['home_team_assignment_id'] = schedule['home_team'].map(
        lambda team: f'lovb:{season}:{normalized_name(team)}'
    )
    schedule['away_team_assignment_id'] = schedule['away_team'].map(
        lambda team: f'lovb:{season}:{normalized_name(team)}'
    )
    match_ids = set(schedule['match_id'])

    outputs = [schedule]
    for label, frame, team_columns in (
        ('PBP', pbp, ('home_team_name', 'away_team_name')),
        ('events', events, ()),
        ('player info', player_info, ('team_name',)),
        ('player boxscore', player_boxscore, ('team_name',)),
    ):
        selected = frame[frame['match_id'].isin(match_ids)].copy()
        if set(selected['match_id']) != match_ids:
            missing = sorted(match_ids - set(selected['match_id']))
            raise ValueError(f'LOVB {label} is missing matches: {missing[:10]}')
        selected = _normalize_teams(selected, team_columns, aliases)
        outputs.append(selected)

    schedule, pbp, events, player_info, player_boxscore = outputs
    valid_teams = pd.concat([
        schedule[['match_id', 'home_team']].rename(
            columns={'home_team': 'team_name'}
        ),
        schedule[['match_id', 'away_team']].rename(
            columns={'away_team': 'team_name'}
        ),
    ]).drop_duplicates()
    for label, frame in (
        ('player info', player_info), ('player boxscore', player_boxscore)
    ):
        checked = frame.merge(
            valid_teams, on=['match_id', 'team_name'], how='left', indicator=True
        )
        if checked['_merge'].ne('both').any():
            raise ValueError(f'LOVB {label} contains a non-match club')
    return schedule, pbp, events, player_info, player_boxscore


def load_lovb_analysis_sources(season: int) -> tuple[pd.DataFrame, ...]:
    season = int(season)
    return filter_lovb_analysis_sources(
        season,
        load_schedule(league='lovb', seasons=season),
        load_pbp(league='lovb', seasons=season),
        load_events_log(league='lovb', seasons=season),
        load_player_info(league='lovb', seasons=season),
        load_player_boxscore(league='lovb', seasons=season),
    )


def build_lovb_canonical_season_from_frames(
    season: int,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    events: pd.DataFrame,
    player_info: pd.DataFrame,
    player_boxscore: pd.DataFrame,
):
    filtered = filter_lovb_analysis_sources(
        season, schedule, pbp, events, player_info, player_boxscore
    )
    return build_public_canonical_from_filtered_frames(
        league='lovb', season=int(season),
        schedule=filtered[0], pbp=filtered[1], events=filtered[2],
        player_info=filtered[3], player_boxscore=filtered[4],
    )


def build_lovb_canonical_season(season: int):
    return build_public_canonical_from_filtered_frames(
        league='lovb', season=int(season),
        **dict(zip(
            ('schedule', 'pbp', 'events', 'player_info', 'player_boxscore'),
            load_lovb_analysis_sources(int(season)),
        )),
    )
