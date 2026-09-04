"""Authoritative deterministic team-match bootstrap draw plans."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from canonical_data import require_columns


def build_team_match_draw_plan(
    schedule: pd.DataFrame,
    *,
    n_boot: int,
    seed: int = 42,
    identity: bool = False,
) -> pd.DataFrame:
    """Sample each team's matches independently using one shared plan."""
    require_columns(
        schedule,
        ['season', 'match_id', 'home_team', 'away_team'],
        'schedule',
    )
    seasons = schedule['season'].drop_duplicates().tolist()
    if len(seasons) != 1:
        raise ValueError('A draw plan must contain exactly one season')
    if not identity and (not isinstance(n_boot, (int, np.integer)) or n_boot < 1):
        raise ValueError('n_boot must be a positive integer')
    season = int(seasons[0])
    long = pd.concat([
        schedule[['match_id', 'home_team']].rename(
            columns={'home_team': 'team'}
        ),
        schedule[['match_id', 'away_team']].rename(
            columns={'away_team': 'team'}
        ),
    ], ignore_index=True).drop_duplicates()
    return _draw_plan_from_team_matches(
        long, season=season, n_boot=n_boot, seed=seed, identity=identity
    )


def build_team_match_draw_plan_for_blocks(
    blocks: pd.DataFrame,
    *,
    season: int,
    n_boot: int,
    seed: int = 42,
    identity: bool = False,
    team_column: str = 'team',
    match_column: str = 'match_id',
) -> pd.DataFrame:
    """Build the same plan from an already canonical team-match table."""
    require_columns(blocks, [team_column, match_column], 'team-match blocks')
    long = blocks[[team_column, match_column]].rename(columns={
        team_column: 'team', match_column: 'match_id'
    }).drop_duplicates()
    return _draw_plan_from_team_matches(
        long, season=int(season), n_boot=n_boot, seed=seed, identity=identity
    )


def _draw_plan_from_team_matches(
    long: pd.DataFrame,
    *,
    season: int,
    n_boot: int,
    seed: int,
    identity: bool,
) -> pd.DataFrame:
    if long.empty or long[['team', 'match_id']].isna().any().any():
        raise ValueError('Team-match draw-plan input must be nonempty and complete')
    if not identity and (not isinstance(n_boot, (int, np.integer)) or n_boot < 1):
        raise ValueError('n_boot must be a positive integer')
    team_matches = {
        team: np.array(sorted(group['match_id'].tolist(), key=str), dtype=object)
        for team, group in long.groupby('team', sort=True)
    }
    rng = np.random.default_rng(np.random.SeedSequence([seed, season, 2]))
    records = []
    for bootstrap_id in range(1 if identity else n_boot):
        for team in sorted(team_matches, key=str):
            matches = team_matches[team]
            indices = (
                np.arange(len(matches))
                if identity
                else rng.integers(0, len(matches), size=len(matches))
            )
            for draw_slot, index in enumerate(indices):
                records.append({
                    'season': season,
                    'bootstrap_id': bootstrap_id,
                    'team': team,
                    'draw_slot': draw_slot,
                    'occurrence_id': f'{bootstrap_id}:{team}:{draw_slot}',
                    'source_match_id': matches[index],
                })
    return pd.DataFrame(records)


def draw_multiplicities(plan: pd.DataFrame, bootstrap_id: int) -> pd.DataFrame:
    selected = plan[plan['bootstrap_id'].eq(bootstrap_id)]
    if selected.empty:
        raise ValueError(f'Draw plan lacks bootstrap {bootstrap_id}')
    return (
        selected.groupby(
            ['season', 'team', 'source_match_id'], as_index=False, sort=True
        )
        .size()
        .rename(columns={
            'source_match_id': 'match_id', 'size': 'multiplicity'
        })
    )


def physical_match_weights(
    schedule: pd.DataFrame,
    multiplicities: pd.DataFrame,
) -> pd.DataFrame:
    """Average independently drawn home/away multiplicities for joint fits."""
    matches = schedule[
        ['season', 'match_id', 'home_team', 'away_team']
    ].drop_duplicates()
    if matches.duplicated(['season', 'match_id']).any():
        raise ValueError('Schedule is not unique by season/match')
    for side in ('home', 'away'):
        selected = multiplicities.rename(columns={
            'team': f'{side}_team', 'multiplicity': f'{side}_multiplicity'
        })[['season', 'match_id', f'{side}_team', f'{side}_multiplicity']]
        matches = matches.merge(
            selected,
            on=['season', 'match_id', f'{side}_team'],
            how='left',
            validate='one_to_one',
        )
        matches[f'{side}_multiplicity'] = matches[
            f'{side}_multiplicity'
        ].fillna(0)
    matches['physical_match_weight'] = (
        matches['home_multiplicity'] + matches['away_multiplicity']
    ) / 2.0
    return matches[[
        'season', 'match_id', 'home_multiplicity', 'away_multiplicity',
        'physical_match_weight',
    ]]


def sample_additive_blocks(
    statistics: pd.DataFrame,
    multiplicities: pd.DataFrame,
    count_columns: Iterable[str],
) -> pd.DataFrame:
    require_columns(statistics, ['season', 'match_id', 'team'], 'statistics')
    result = statistics.merge(
        multiplicities,
        on=['season', 'match_id', 'team'],
        how='left',
        validate='many_to_one',
    )
    result['multiplicity'] = result['multiplicity'].fillna(0)
    result = result[result['multiplicity'].gt(0)].copy()
    for column in count_columns:
        result[column] = result[column] * result['multiplicity']
    return result.drop(columns='multiplicity')


def sample_occurrence_rows(
    rows: pd.DataFrame,
    multiplicities: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(rows, ['season', 'match_id', 'team'], 'occurrence rows')
    sampled = rows.merge(
        multiplicities,
        on=['season', 'match_id', 'team'],
        how='left',
        validate='many_to_one',
    )
    sampled['multiplicity'] = sampled['multiplicity'].fillna(0).astype(int)
    if sampled['multiplicity'].lt(0).any():
        raise ValueError('Occurrence multiplicity cannot be negative')
    sampled = sampled.loc[
        sampled.index.repeat(sampled['multiplicity'])
    ].drop(columns='multiplicity')
    return sampled.reset_index(drop=True)


def sample_player_set_rows(
    player_sets: pd.DataFrame,
    multiplicities: pd.DataFrame,
    *,
    team_column: str = 'team_name',
) -> pd.DataFrame:
    """Repeat canonical player-set team-match blocks from the shared plan."""
    working = player_sets.rename(columns={team_column: 'team'})
    had_season = 'season' in working.columns
    keys = ['match_id', 'team']
    if 'season' in working.columns:
        keys.insert(0, 'season')
    rows = working.merge(
        multiplicities,
        on=keys,
        how='left',
        validate='many_to_one',
    )
    rows['multiplicity'] = rows['multiplicity'].fillna(0).astype(int)
    sampled = rows.loc[
        rows.index.repeat(rows['multiplicity'])
    ].drop(columns='multiplicity')
    if not had_season:
        sampled = sampled.drop(columns='season')
    return sampled.rename(columns={'team': team_column}).reset_index(drop=True)
