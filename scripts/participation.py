"""League-independent player participation helpers."""

from __future__ import annotations

import pandas as pd

from canonical_data import CanonicalSeason, RALLY_KEY, require_columns


def build_player_actual_sets(
    foundation: CanonicalSeason,
    conventional_sets: pd.DataFrame,
) -> pd.DataFrame:
    """Complete audited set totals for public PBP actors absent from boxscores.

    The conventional reconstruction remains authoritative whenever a player has
    a regular-season boxscore row. A small number of legitimate 2024 PBP actors
    have no player-boxscore record; for them only, unique sets evidenced by a
    mapped action or canonical lineup provide an explicit fallback.
    """
    actions = foundation.actions.dropna(subset=['player_id'])[
        ['season', 'match_id', 'set_number', 'player_id']
    ].copy()
    actions['player_id'] = actions['player_id'].astype('string')
    evidence = [actions]
    for side in ('home', 'away'):
        player_columns = [
            f'{side}_p{position}_player_id' for position in range(1, 7)
        ]
        require_columns(
            foundation.lineups,
            [*RALLY_KEY, *player_columns],
            'canonical lineup table',
        )
        lineup = foundation.lineups[
            ['season', 'match_id', 'set_number', *player_columns]
        ].melt(
            id_vars=['season', 'match_id', 'set_number'],
            value_vars=player_columns,
            value_name='player_id',
        ).dropna(subset=['player_id'])
        lineup['player_id'] = lineup['player_id'].astype('string')
        evidence.append(lineup[
            ['season', 'match_id', 'set_number', 'player_id']
        ])
    fallback = (
        pd.concat(evidence, ignore_index=True)
        .drop_duplicates(['season', 'match_id', 'set_number', 'player_id'])
        .groupby(['season', 'player_id'], as_index=False, sort=True)
        .size()
        .rename(columns={'size': 'actual_sets_played'})
    )
    base = conventional_sets.copy()
    if 'actual_sets_played' not in base and 'sets_played' in base:
        base = base.rename(columns={'sets_played': 'actual_sets_played'})
    require_columns(
        base,
        ['season', 'player_id', 'actual_sets_played'],
        'conventional actual-sets table',
    )
    base['player_id'] = base['player_id'].astype('string')
    base['actual_sets_source'] = 'audited_conventional_reconstruction'
    missing = fallback.merge(
        base[['season', 'player_id']],
        on=['season', 'player_id'],
        how='left',
        indicator=True,
    )
    missing = missing[missing['_merge'].eq('left_only')].drop(columns='_merge')
    missing['actual_sets_source'] = 'canonical_pbp_participation_fallback'
    return pd.concat(
        [
            base[
                ['season', 'player_id', 'actual_sets_played',
                 'actual_sets_source']
            ],
            missing,
        ],
        ignore_index=True,
    )
