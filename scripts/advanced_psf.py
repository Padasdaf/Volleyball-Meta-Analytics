"""Point-Scoring Fraction (PSF) from public MLV play-by-play.

The metric follows Burton and Powers (2015):

    PSF = a + (1 - a - e) * (1 - s)

where ``a`` and ``e`` are the fractions of all serves that are aces and
service errors, and ``s`` is the opponent's modified sideout fraction among
nonterminal (in-play) serves.  This module contains PSF-specific logic only;
the shared canonical PBP tables come from :mod:`pbp_foundation`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canonical_data import (
    CanonicalSeason,
    RALLY_KEY,
    require_columns as _require_columns,
)
from mlv_adapter import build_mlv_canonical_season


ACE_OUTCOMES = frozenset({'#'})
SERVICE_ERROR_OUTCOMES = frozenset({'='})
IN_PLAY_OUTCOMES = frozenset({'!', '+', '-', '/'})
KNOWN_SERVE_OUTCOMES = (
    ACE_OUTCOMES | SERVICE_ERROR_OUTCOMES | IN_PLAY_OUTCOMES
)

# The source encodes a set-opening penalty as a synthetic S+ / F= sequence.
# The event log independently records the penalty at the same instant, the
# source jersey is 0, and the player boxscore contains no corresponding serve.
AUDITED_NON_SERVE_RALLY_KEYS = frozenset({(2025, 2161005, 2, 1)})

# At match/set/point 2256693/3/2 the source row is S= even though the PBP score
# and the event-log rally winner both give the point to Grand Rapids, and
# Columbus immediately records F=.  The combined evidence establishes an
# in-play serve followed by a receiving-team error, not a service error.
AUDITED_SERVE_OUTCOME_OVERRIDES = {
    (2026, 2256693, 3, 2): 'in_play',
}

SUFFICIENT_COUNT_COLUMNS = [
    'serves',
    'aces',
    'service_errors',
    'in_play_serves',
    'receiving_team_wins_after_in_play',
    'serving_team_wins_after_in_play',
]


def build_serve_events(foundation: CanonicalSeason) -> pd.DataFrame:
    """Return one source-supported, point-bearing serve per official rally.

    A replayed source sequence can contain more than one serve row for one
    official point.  Only its final serve is retained because only that serve
    has the official rally outcome.  Rallies with no source serve are not
    imputed.  The single audited penalty placeholder is explicitly excluded.
    """
    actions = foundation.actions.copy()
    rallies = foundation.rallies.copy()
    _require_columns(
        actions,
        [
            *RALLY_KEY,
            'rally_id',
            'action_order',
            'acting_team',
            'jersey_number',
            'player_id',
            'identity_status',
            'source_action',
            'source_outcome',
        ],
        'canonical action table',
    )
    _require_columns(
        rallies,
        [
            *RALLY_KEY,
            'rally_id',
            'serving_team',
            'receiving_team',
            'point_winner_team',
            'source_serve_count',
            'replayed_source_sequence',
        ],
        'canonical rally table',
    )

    serves = actions[actions['source_action'].eq('S')].copy()
    if serves.empty:
        raise ValueError("Canonical action table contains no serve actions")
    unknown_outcomes = sorted(
        set(serves['source_outcome'].dropna()) - KNOWN_SERVE_OUTCOMES
    )
    if unknown_outcomes or serves['source_outcome'].isna().any():
        raise ValueError(f"Unknown or missing serve outcomes: {unknown_outcomes}")

    key_tuples = pd.MultiIndex.from_frame(serves[RALLY_KEY])
    non_serve_mask = key_tuples.isin(AUDITED_NON_SERVE_RALLY_KEYS)
    audited_non_serves = serves.loc[non_serve_mask]
    if not audited_non_serves.empty:
        expected_placeholder = (
            audited_non_serves['source_outcome'].eq('+')
            & pd.to_numeric(
                audited_non_serves['jersey_number'], errors='coerce'
            ).eq(0)
            & audited_non_serves['player_id'].isna()
        )
        if not expected_placeholder.all():
            raise ValueError("Audited penalty placeholder changed upstream")
    serves = serves.loc[~non_serve_mask].copy()

    serve_counts = (
        serves.groupby(RALLY_KEY, sort=False)
        .size()
        .rename('_retained_serve_count')
        .reset_index()
    )
    rally_checks = rallies.merge(
        serve_counts, on=RALLY_KEY, how='inner', validate='one_to_one'
    )
    unexpected_duplicates = rally_checks[
        rally_checks['_retained_serve_count'].gt(1)
        & ~rally_checks['replayed_source_sequence'].eq(True)
    ]
    if not unexpected_duplicates.empty:
        raise ValueError(
            "Multiple serve rows occur outside an explicitly flagged replay: "
            f"{unexpected_duplicates[RALLY_KEY].head(10).to_dict('records')}"
        )

    # Stable source action order is preserved by the PBP foundation.  The last
    # serve in a replayed official point is the point-bearing serve.
    selected = (
        serves.sort_values([*RALLY_KEY, 'action_order'], kind='stable')
        .groupby(RALLY_KEY, sort=False)
        .tail(1)
        .copy()
    )
    selected = selected.merge(
        rallies[
            [
                *RALLY_KEY,
                'rally_id',
                'serving_team',
                'receiving_team',
                'point_winner_team',
                'source_serve_count',
                'replayed_source_sequence',
            ]
        ],
        on=RALLY_KEY,
        how='left',
        validate='one_to_one',
        suffixes=('', '_rally'),
        sort=False,
    )
    if selected[
        ['serving_team', 'receiving_team', 'point_winner_team']
    ].isna().any().any():
        raise ValueError("A serve action lacks its canonical rally state")
    if not selected['acting_team'].eq(selected['serving_team']).all():
        bad = selected.loc[
            ~selected['acting_team'].eq(selected['serving_team']),
            [*RALLY_KEY, 'acting_team', 'serving_team'],
        ]
        raise ValueError(
            "Serve acting team disagrees with canonical serving team: "
            f"{bad.head(10).to_dict('records')}"
        )

    selected['identity_resolution'] = selected['identity_status'].astype('string')
    if selected['player_id'].isna().any():
        bad = selected.loc[
            selected['player_id'].isna(),
            [*RALLY_KEY, 'serving_team', 'jersey_number', 'identity_status'],
        ]
        raise ValueError(
            "Legitimate serve actions lack persistent player IDs: "
            f"{bad.head(20).to_dict('records')}"
        )
    selected['player_id'] = selected['player_id'].astype('string')

    selected['serve_outcome'] = np.select(
        [
            selected['source_outcome'].isin(ACE_OUTCOMES),
            selected['source_outcome'].isin(SERVICE_ERROR_OUTCOMES),
            selected['source_outcome'].isin(IN_PLAY_OUTCOMES),
        ],
        ['ace', 'service_error', 'in_play'],
        default='unknown',
    )
    audited_outcome_masks = []
    for key, outcome in AUDITED_SERVE_OUTCOME_OVERRIDES.items():
        mask = pd.MultiIndex.from_frame(selected[RALLY_KEY]).isin([key])
        if mask.any():
            if mask.sum() != 1:
                raise ValueError(
                    f"Audited serve-outcome key is not unique: {key}"
                )
            row = selected.loc[mask].iloc[0]
            if not (
                row['source_outcome'] == '='
                and row['point_winner_team'] == row['serving_team']
            ):
                raise ValueError(
                    f"Audited serve-outcome evidence changed upstream: {key}"
                )
            selected.loc[mask, 'serve_outcome'] = outcome
            audited_outcome_masks.append(mask)
    if selected['serve_outcome'].eq('unknown').any():
        raise AssertionError("Serve outcome partition is not exhaustive")
    selected['is_ace'] = selected['serve_outcome'].eq('ace')
    selected['is_service_error'] = selected['serve_outcome'].eq('service_error')
    selected['is_in_play'] = selected['serve_outcome'].eq('in_play')
    partition = selected[
        ['is_ace', 'is_service_error', 'is_in_play']
    ].sum(axis=1)
    if not partition.eq(1).all():
        raise AssertionError("Serve categories are not mutually exclusive")

    ace_winner_valid = selected.loc[
        selected['is_ace'], 'point_winner_team'
    ].eq(selected.loc[selected['is_ace'], 'serving_team'])
    error_winner_valid = selected.loc[
        selected['is_service_error'], 'point_winner_team'
    ].eq(selected.loc[selected['is_service_error'], 'receiving_team'])
    if not ace_winner_valid.all() or not error_winner_valid.all():
        raise ValueError("Terminal serve outcome contradicts official point winner")
    winner_valid = (
        selected['point_winner_team'].eq(selected['serving_team'])
        | selected['point_winner_team'].eq(selected['receiving_team'])
    )
    if not winner_valid.all():
        raise ValueError("Official point winner is not one of the two rally teams")

    selected['serving_team_win_after_in_play'] = (
        selected['is_in_play']
        & selected['point_winner_team'].eq(selected['serving_team'])
    )
    selected['receiving_team_win_after_in_play'] = (
        selected['is_in_play']
        & selected['point_winner_team'].eq(selected['receiving_team'])
    )
    in_play_partition = (
        selected['serving_team_win_after_in_play']
        + selected['receiving_team_win_after_in_play']
    )
    if not in_play_partition.eq(selected['is_in_play'].astype(int)).all():
        raise AssertionError("In-play winner partition is invalid")

    selected['serve_source_status'] = np.where(
        selected['replayed_source_sequence'].eq(True),
        'replay_final_serve',
        'source_serve',
    )
    for mask in audited_outcome_masks:
        selected.loc[mask, 'serve_source_status'] = 'audited_outcome_override'
    output_columns = [
        *RALLY_KEY,
        'rally_id',
        'serving_team',
        'receiving_team',
        'point_winner_team',
        'player_id',
        'jersey_number',
        'identity_resolution',
        'source_outcome',
        'serve_outcome',
        'is_ace',
        'is_service_error',
        'is_in_play',
        'serving_team_win_after_in_play',
        'receiving_team_win_after_in_play',
        'source_serve_count',
        'replayed_source_sequence',
        'serve_source_status',
    ]
    output = selected[output_columns].sort_values(
        RALLY_KEY, kind='stable'
    ).reset_index(drop=True)
    if output.duplicated(RALLY_KEY).any():
        raise AssertionError("Serve-event table is not unique by official rally")
    return output


def _validate_sufficient_statistics(statistics: pd.DataFrame) -> None:
    _require_columns(
        statistics,
        ['season', 'match_id', 'team', 'player_id', *SUFFICIENT_COUNT_COLUMNS],
        'PSF sufficient statistics',
    )
    numeric = statistics[SUFFICIENT_COUNT_COLUMNS].apply(
        pd.to_numeric, errors='coerce'
    )
    if numeric.isna().any().any() or (numeric < 0).any().any():
        raise ValueError("PSF counts must be finite and nonnegative")
    if not np.isclose(numeric, np.round(numeric)).all():
        raise ValueError("PSF counts must be integers")
    serve_partition = (
        numeric['aces']
        + numeric['service_errors']
        + numeric['in_play_serves']
    )
    if not serve_partition.eq(numeric['serves']).all():
        raise ValueError("serves != aces + service errors + in-play serves")
    winner_partition = (
        numeric['receiving_team_wins_after_in_play']
        + numeric['serving_team_wins_after_in_play']
    )
    if not winner_partition.eq(numeric['in_play_serves']).all():
        raise ValueError("In-play serves do not partition by official point winner")


def build_psf_sufficient_statistics(serve_events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact team-match/server counts for later block resampling.

    Duplicate input rows are intentionally counted with multiplicity.  This is
    what makes concatenated, repeatedly sampled team-match blocks exact.
    """
    _require_columns(
        serve_events,
        [
            'season',
            'match_id',
            'serving_team',
            'player_id',
            'is_ace',
            'is_service_error',
            'is_in_play',
            'receiving_team_win_after_in_play',
            'serving_team_win_after_in_play',
        ],
        'serve-event table',
    )
    if serve_events.empty:
        return pd.DataFrame(
            columns=['season', 'match_id', 'team', 'player_id',
                     *SUFFICIENT_COUNT_COLUMNS]
        )
    if serve_events['player_id'].isna().any():
        raise ValueError("Serve-event table contains a missing player_id")
    events = serve_events.copy()
    for column in [
        'is_ace',
        'is_service_error',
        'is_in_play',
        'receiving_team_win_after_in_play',
        'serving_team_win_after_in_play',
    ]:
        events[column] = events[column].astype(int)
    events['serves'] = 1
    events = events.rename(columns={'serving_team': 'team'})
    statistics = (
        events.groupby(
            ['season', 'match_id', 'team', 'player_id'],
            sort=True,
            as_index=False,
        )
        .agg(
            serves=('serves', 'sum'),
            aces=('is_ace', 'sum'),
            service_errors=('is_service_error', 'sum'),
            in_play_serves=('is_in_play', 'sum'),
            receiving_team_wins_after_in_play=(
                'receiving_team_win_after_in_play', 'sum'
            ),
            serving_team_wins_after_in_play=(
                'serving_team_win_after_in_play', 'sum'
            ),
        )
    )
    _validate_sufficient_statistics(statistics)
    return statistics


def calculate_psf(statistics: pd.DataFrame) -> pd.DataFrame:
    """Calculate one PSF value per persistent player-season.

    Counts are combined across every team stint before the nonlinear fractions
    are evaluated.  The publication defines ``s`` conditionally on a
    nonterminal serve and does not define an all-terminal boundary extension.
    PSF is therefore missing when a server has no in-play serves.
    """
    _validate_sufficient_statistics(statistics)
    grouped = (
        statistics.groupby(['season', 'player_id'], sort=True, as_index=False)[
            SUFFICIENT_COUNT_COLUMNS
        ]
        .sum()
    )
    serves = grouped['serves'].astype(float)
    in_play = grouped['in_play_serves'].astype(float)
    grouped['ace_fraction'] = grouped['aces'].div(serves.where(serves.gt(0)))
    grouped['service_error_fraction'] = grouped['service_errors'].div(
        serves.where(serves.gt(0))
    )
    grouped['modified_sideout_fraction'] = grouped[
        'receiving_team_wins_after_in_play'
    ].div(in_play.where(in_play.gt(0)))

    direct = (
        grouped['aces'] + grouped['serving_team_wins_after_in_play']
    ).div(serves.where(serves.gt(0)))
    published = (
        grouped['ace_fraction']
        + (
            1
            - grouped['ace_fraction']
            - grouped['service_error_fraction']
        )
        * (1 - grouped['modified_sideout_fraction'])
    )
    comparable = serves.gt(0) & in_play.gt(0)
    if not np.allclose(
        direct[comparable], published[comparable], rtol=0, atol=1e-12
    ):
        raise AssertionError("Published PSF and direct scoring fraction disagree")
    # Do not use the algebraic direct-count identity to invent a value outside
    # the domain on which the source defines modified sideout.
    grouped['PSF'] = published.where(comparable)
    finite = grouped['PSF'].notna()
    if ((grouped.loc[finite, 'PSF'] < 0) | (grouped.loc[finite, 'PSF'] > 1)).any():
        raise AssertionError("PSF lies outside its probability range [0, 1]")
    return grouped


def calculate_season_psf(
    foundation_or_season: CanonicalSeason | int,
) -> pd.DataFrame:
    """Build public serve events and return player-season PSF values."""
    foundation = (
        build_mlv_canonical_season(int(foundation_or_season))
        if isinstance(foundation_or_season, (int, np.integer))
        else foundation_or_season
    )
    events = build_serve_events(foundation)
    statistics = build_psf_sufficient_statistics(events)
    return calculate_psf(statistics)
