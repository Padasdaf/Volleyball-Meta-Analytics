"""MLV-specific source loading, repairs, and canonical-season construction."""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from pyvolleydata.get_data import (
    load_events_log,
    load_pbp,
    load_player_boxscore,
    load_player_info,
    load_schedule,
)

from canonical_data import (
    CanonicalSeason,
    normalize_jersey_numbers,
    require_columns,
)


SUPPORTED_MLV_SEASONS = (2024, 2025, 2026)
EXPECTED_REGULAR_MATCH_COUNTS = {2024: 84, 2025: 112, 2026: 112}
REGULAR_SEASON_PHASE = 'regular season'
NON_LEAGUE_MATCH_EXCLUSIONS = {2025: frozenset({2167801})}

SOURCE_PLACEHOLDER_PLAYER_IDS = frozenset({
    '5555555', '7777777', '8888888', '9999998', '9999999',
})
AUDITED_PLAYER_ID_OVERRIDES = {
    (2024, 'Orlando Valkyries', 'kalyahwilliams'): 'mlv:kalyah-williams',
}
AUDITED_NAME_ALIASES = {
    'michellbartschhackley': 'michellebartschhackley',
    'paigebriggsromine': 'paigebriggs',
}
AUDITED_PBP_IDENTITY_OVERRIDES = {
    (2024, 2125275, 'Grand Rapids Rise', 18): {
        'player_id': '2130288',
        'canonical_jersey': 10,
        'normalized_name': 'shannonscully',
    },
}

PARTICIPATION_STAT_COLUMNS = (
    'serves', 'serve_errors', 'serve_aces', 'attack_attempts',
    'attack_errors', 'attack_kills', 'receptions', 'reception_errors',
    'block_points', 'block_touches', 'earned_points', 'net_points',
    'assists', 'successful_digs', 'spike_hp', 'points',
)


def normalized_name(value) -> str:
    """Normalize an MLV display spelling for identity evidence."""
    if pd.isna(value):
        return ''
    ascii_value = unicodedata.normalize('NFKD', str(value)).encode(
        'ascii', 'ignore'
    ).decode('ascii')
    normalized = ''.join(
        character for character in ascii_value.lower() if character.isalnum()
    )
    return AUDITED_NAME_ALIASES.get(normalized, normalized)


def _clean_source_player_id(value):
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    if cleaned.endswith('.0') and cleaned[:-2].isdigit():
        cleaned = cleaned[:-2]
    if cleaned.lower() in {'', '-', 'nan', 'none', '0'}:
        return None
    return cleaned


def _unique_mapping(
    frame,
    keys,
    value,
    label,
    *,
    fail_on_ambiguity=False,
):
    grouped = (
        frame.dropna(subset=[*keys, value])
        .groupby(keys, dropna=False)[value]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
    )
    ambiguous = grouped[grouped.map(len).gt(1)]
    if not ambiguous.empty and fail_on_ambiguity:
        raise ValueError(
            f"{label} has ambiguous source mappings: "
            f"{ambiguous.head(10).to_dict()}"
        )
    return {
        key if isinstance(key, tuple) else (key,): values[0]
        for key, values in grouped.items()
        if len(values) == 1
    }


def map_mlv_player_ids(
    boxscore_df: pd.DataFrame,
    info_df: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map boxscore and roster rows through audited match-local evidence."""
    season = int(season)
    boxscore = boxscore_df.copy()
    info = info_df.copy()
    boxscore['_normalized_name'] = boxscore['player_name'].map(normalized_name)
    info['_normalized_name'] = info['player_name'].map(normalized_name)
    boxscore['_jersey_number'] = normalize_jersey_numbers(
        boxscore['player_number']
    )
    info['_jersey_number'] = normalize_jersey_numbers(info['jersey_number'])
    info['_source_player_id'] = info['player_id'].map(_clean_source_player_id)

    real_info = info[
        info['_source_player_id'].notna()
        & ~info['_source_player_id'].isin(SOURCE_PLACEHOLDER_PLAYER_IDS)
    ]
    ids_by_name = _unique_mapping(
        real_info,
        ['_normalized_name'],
        '_source_player_id',
        'season-wide normalized player name',
    )
    info['_canonical_player_id'] = info['_source_player_id']
    placeholder = info['_canonical_player_id'].isin(
        SOURCE_PLACEHOLDER_PLAYER_IDS
    )
    info.loc[placeholder, '_canonical_player_id'] = info.loc[
        placeholder, '_normalized_name'
    ].map(lambda name: ids_by_name.get((name,)))
    unresolved = info[
        info['_source_player_id'].isin(SOURCE_PLACEHOLDER_PLAYER_IDS)
        & info['_canonical_player_id'].isna()
    ]
    if not unresolved.empty:
        examples = unresolved[
            ['match_id', 'team_name', 'player_name', 'player_id']
        ].head(10).to_dict('records')
        raise ValueError(
            f"Could not resolve placeholder player IDs: {examples}"
        )

    exact = _unique_mapping(
        info,
        ['match_id', 'team_name', '_jersey_number', '_normalized_name'],
        '_canonical_player_id',
        'match/team/jersey/name identity',
        fail_on_ambiguity=True,
    )
    name = _unique_mapping(
        info,
        ['_normalized_name'],
        '_canonical_player_id',
        'season-wide normalized player name',
    )
    player_ids = []
    methods = []
    for match_id, team, player_name, jersey_number in boxscore[[
        'match_id', 'team_name', '_normalized_name', '_jersey_number'
    ]].itertuples(index=False, name=None):
        identity = None
        method = None
        if not pd.isna(jersey_number):
            identity = exact.get((match_id, team, jersey_number, player_name))
            method = 'match_team_jersey_name'
        if identity is None:
            identity = name.get((player_name,))
            method = 'season_name_fallback'
        if identity is None:
            identity = AUDITED_PLAYER_ID_OVERRIDES.get(
                (season, team, player_name)
            )
            method = 'audited_override'
        player_ids.append(identity)
        methods.append(method if identity is not None else 'unresolved')

    boxscore['player_id'] = pd.Series(
        player_ids, index=boxscore.index, dtype='string'
    )
    boxscore['player_id_mapping_method'] = methods
    unresolved = boxscore[boxscore['player_id'].isna()]
    if not unresolved.empty:
        examples = unresolved[
            ['match_id', 'team_name', 'player_name', 'player_number']
        ].drop_duplicates().head(20).to_dict('records')
        raise ValueError(
            "Regular-season boxscore rows lack persistent IDs: "
            f"{examples}"
        )
    collisions = set(AUDITED_PLAYER_ID_OVERRIDES.values()) & set(
        real_info['_source_player_id']
    )
    if collisions:
        raise ValueError(
            f"Audited player ID overrides collide with source IDs: {collisions}"
        )
    return boxscore, info


def _starter_participation_keys(info: pd.DataFrame) -> set[tuple]:
    keys = set()
    for set_number in range(1, 6):
        flag = f'set_{set_number}_is_starter'
        if flag not in info:
            continue
        for match_id, team, player_id in info[info[flag].eq(True)][[
            'match_id', 'team_name', '_canonical_player_id'
        ]].itertuples(index=False, name=None):
            if not pd.isna(player_id):
                keys.add((match_id, team, set_number, str(player_id)))
    return keys


def _event_participation_keys(
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
        if not selected['team_involved'].isin(['home', 'away']).all():
            raise ValueError(
                f"{event_type} events contain unknown team_involved values"
            )
        selected['_team_name'] = np.where(
            selected['team_involved'].eq('home'),
            selected['home_team'],
            selected['away_team'],
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


def mark_mlv_played_sets(
    boxscore_df: pd.DataFrame,
    info_df: pd.DataFrame,
    events_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the audited played-set indicator and its evidence columns."""
    boxscore = boxscore_df.copy()
    missing = [c for c in PARTICIPATION_STAT_COLUMNS if c not in boxscore]
    if missing:
        raise ValueError(f"Boxscore is missing participation fields: {missing}")
    statistic = boxscore[list(PARTICIPATION_STAT_COLUMNS)].notna().any(axis=1)
    position = boxscore['set_starting_position'].notna()
    starter_keys = _starter_participation_keys(info_df)
    event_keys = _event_participation_keys(events_df, schedule_df)
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


def filter_mlv_regular_season_sources(
    season: int,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    events: pd.DataFrame,
    player_info: pd.DataFrame,
    player_boxscore: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Validate and competition-filter current public MLV source tables."""
    season = int(season)
    require_columns(
        schedule,
        ['match_id', 'season', 'phase', 'home_team', 'away_team'],
        'schedule',
    )
    filtered = schedule.copy()
    if filtered['match_id'].duplicated().any():
        raise ValueError("Schedule match_id must be unique within a season")
    if filtered[['match_id', 'phase', 'home_team', 'away_team']].isna().any().any():
        raise ValueError("Schedule contains missing match, phase, or team values")
    source_seasons = pd.to_numeric(filtered['season'], errors='coerce')
    if source_seasons.isna().any() or not source_seasons.eq(season).all():
        raise ValueError(f"Schedule contains rows outside season {season}")
    filtered['_normalized_phase'] = (
        filtered['phase'].astype(str).str.strip().str.lower()
    )
    filtered = filtered[
        filtered['_normalized_phase'].eq(REGULAR_SEASON_PHASE)
    ].copy()
    excluded = NON_LEAGUE_MATCH_EXCLUSIONS.get(season, frozenset())
    filtered = filtered[~filtered['match_id'].isin(excluded)].copy()
    expected = EXPECTED_REGULAR_MATCH_COUNTS.get(season)
    if expected is not None and len(filtered) != expected:
        raise ValueError(
            f"MLV {season} schedule has {len(filtered)} regular matches; "
            f"expected {expected}"
        )
    regular_ids = set(filtered['match_id'])
    outputs = [filtered]
    for label, frame in (
        ('PBP', pbp), ('events', events), ('player info', player_info),
        ('player boxscore', player_boxscore),
    ):
        require_columns(frame, ['match_id'], label)
        if frame['match_id'].isna().any():
            raise ValueError(f"{label} contains missing match IDs")
        selected = frame[frame['match_id'].isin(regular_ids)].copy()
        selected = selected[~selected['match_id'].isin(excluded)].copy()
        if set(selected['match_id']) != regular_ids:
            missing = sorted(regular_ids - set(selected['match_id']))[:10]
            raise ValueError(f"{label} is missing regular matches: {missing}")
        if 'season' in selected:
            selected_seasons = pd.to_numeric(
                selected['season'], errors='coerce'
            )
            if selected_seasons.isna().any() or not selected_seasons.eq(
                season
            ).all():
                raise ValueError(f"{label} contains rows outside season {season}")
        outputs.append(selected)
    return tuple(outputs)


def load_mlv_regular_season_sources(season: int) -> tuple[pd.DataFrame, ...]:
    season = int(season)
    if season not in SUPPORTED_MLV_SEASONS:
        raise ValueError(f"Supported MLV seasons are {SUPPORTED_MLV_SEASONS}")
    return filter_mlv_regular_season_sources(
        season,
        load_schedule(league='mlv', seasons=season),
        load_pbp(league='mlv', seasons=season),
        load_events_log(league='mlv', seasons=season),
        load_player_info(league='mlv', seasons=season),
        load_player_boxscore(league='mlv', seasons=season),
    )


def build_mlv_player_metadata(
    mapped_info: pd.DataFrame,
    player_sets: pd.DataFrame,
) -> pd.DataFrame:
    """Return canonical match-local identity and generic roster metadata."""
    rows = mapped_info.copy()
    rows['source_player_id'] = rows['_source_player_id'].astype('string')
    rows['player_id'] = rows['_canonical_player_id'].astype('string')
    rows['jersey'] = rows['_jersey_number']
    rows['position'] = pd.to_numeric(rows['primary_position'], errors='coerce')
    rows['identity_provenance'] = np.select(
        [
            rows['_source_player_id'].isna(),
            rows['_source_player_id'].isin(SOURCE_PLACEHOLDER_PLAYER_IDS),
        ],
        [
            'unresolved_source_identity',
            'historical_placeholder_name_resolution',
        ],
        default='source_player_id',
    )
    columns = [
        'match_id', 'team_name', 'player_id', 'source_player_id',
        'player_name', 'jersey', 'position', 'primary_position', 'is_libero',
        'identity_provenance',
    ]
    rows = rows[columns]
    missing = player_sets[
        ~player_sets['player_id'].isin(rows['player_id'])
    ][['match_id', 'team_name', 'player_id', 'player_name', '_jersey_number',
       'player_id_mapping_method']].drop_duplicates()
    if not missing.empty:
        additions = pd.DataFrame({
            'match_id': missing['match_id'],
            'team_name': missing['team_name'],
            'player_id': missing['player_id'],
            'source_player_id': pd.NA,
            'player_name': missing['player_name'],
            'jersey': missing['_jersey_number'],
            'position': np.nan,
            'primary_position': np.nan,
            'is_libero': False,
            'identity_provenance': missing['player_id_mapping_method'],
        })
        additions = additions.dropna(axis=1, how='all')
        rows = pd.concat([rows, additions], ignore_index=True)
    key = ['match_id', 'team_name', 'player_id']
    checks = rows.groupby(key, dropna=False).agg(
        names=('player_name', lambda x: x.dropna().nunique()),
        positions=('primary_position', lambda x: x.dropna().nunique()),
        libero=('is_libero', lambda x: x.dropna().nunique()),
    )
    if checks.gt(1).any(axis=1).any():
        raise ValueError("MLV player metadata conflicts within match/team/player")

    def canonical_name(values):
        counts = values.dropna().astype(str).value_counts()
        if counts.empty:
            return pd.NA
        return sorted(counts[counts.eq(counts.max())].index)[0]

    # Historical placeholder rows did not join canonical PBP actors in the
    # approved advanced preparation. Preserve the resulting deterministic
    # season-name fallback while exposing the repaired persistent ID centrally.
    if rows.loc[
        rows['identity_provenance'].eq('source_player_id'),
        'source_player_id',
    ].isna().any():
        raise AssertionError("source_player_id provenance requires source evidence")

    stable_names = (
        rows[rows['identity_provenance'].eq('source_player_id')]
        .groupby('player_id')['player_name']
        .agg(canonical_name)
    )
    placeholder = rows['identity_provenance'].eq(
        'historical_placeholder_name_resolution'
    )
    rows.loc[placeholder, 'player_name'] = rows.loc[
        placeholder, 'player_id'
    ].map(stable_names)

    return rows.groupby(key, as_index=False, dropna=False).agg(
        source_player_id=(
            'source_player_id',
            lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NA,
        ),
        player_name=('player_name', canonical_name),
        jersey=('jersey', lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NA),
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


def build_mlv_canonical_season_from_frames(
    season: int,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    events: pd.DataFrame,
    player_info: pd.DataFrame,
    player_boxscore: pd.DataFrame,
) -> CanonicalSeason:
    """Build canonical products from injected current-source MLV frames."""
    from pbp_foundation import (
        build_action_table,
        build_lineup_table,
        build_player_identity_map,
        build_rally_table,
    )

    schedule, pbp, events, player_info, player_boxscore = (
        filter_mlv_regular_season_sources(
            season, schedule, pbp, events, player_info, player_boxscore
        )
    )
    player_set_source = player_boxscore.merge(
        schedule[[
            'match_id', 'phase', '_normalized_phase', 'home_team', 'away_team'
        ]],
        on='match_id',
        how='left',
        validate='many_to_one',
    )
    valid_team = (
        player_set_source['team_name'].eq(player_set_source['home_team'])
        | player_set_source['team_name'].eq(player_set_source['away_team'])
    )
    if not valid_team.all():
        examples = player_set_source.loc[
            ~valid_team,
            ['match_id', 'team_name', 'home_team', 'away_team'],
        ].drop_duplicates().head(10).to_dict('records')
        raise ValueError(f"Boxscore team names disagree with schedule: {examples}")
    mapped_boxscore, mapped_info = map_mlv_player_ids(
        player_set_source, player_info, season
    )
    player_sets = mark_mlv_played_sets(
        mapped_boxscore, mapped_info, events, schedule
    )
    identity_map = build_player_identity_map(
        mapped_boxscore,
        mapped_info,
        season=season,
        audited_overrides=AUDITED_PBP_IDENTITY_OVERRIDES,
    )
    actions = build_action_table(pbp, schedule, identity_map, season)
    rallies = build_rally_table(actions)
    lineups = build_lineup_table(rallies, events, identity_map)
    metadata = build_mlv_player_metadata(mapped_info, player_sets)
    return CanonicalSeason(
        league='mlv',
        season=season,
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


def build_mlv_canonical_season(season: int) -> CanonicalSeason:
    return build_mlv_canonical_season_from_frames(
        int(season), *load_mlv_regular_season_sources(int(season))
    )


def get_mlv_player_sets(season: int) -> pd.DataFrame:
    """Return canonical MLV set-level boxscore/participation rows."""
    return build_mlv_canonical_season(int(season)).player_sets.copy()
