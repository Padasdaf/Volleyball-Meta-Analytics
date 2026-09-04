"""Attack Evenness (AEV) from the audited public MLV foundation.

The metric follows Raymond et al. (2024).  Within each match/lineup cell,
the observed attack shares of the five non-setter players are compared with
rotation-adjusted reference shares using

    AEV = 1 - 0.5 * sum_i(abs(a_i - r_i)).

The public feed does not expose DataVolley's designated-setter-position field.
This implementation therefore uses a source-reported primary setter position
only when exactly one non-libero setter is present in a validated lineup; it
does not guess in zero- or two-setter cases.  All other role assignments follow
the SHM rotation assumption used in the source's across-leagues analysis.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_data import (
    CanonicalSeason,
    RALLY_KEY,
    require_columns as _require_columns,
)
from mlv_adapter import build_mlv_canonical_season
from pbp_foundation import build_lineup_table
from r_bridge import advanced_r_library, run_r_csv_exchange


MIN_ATTACKS_PER_CELL = 10
MIN_SETS_PER_SETTER = 100
SETTER_PRIMARY_POSITION = 5
REPOSITORY_ROOT = Path(__file__).resolve().parent
AEV_R_SCRIPT = REPOSITORY_ROOT / 'advanced_aev.R'
DEFAULT_ADVANCED_R_LIBRARY = (
    REPOSITORY_ROOT / '.r-lib' / 'advanced-source-fidelity'
)

CELL_KEY = [
    'season', 'match_id', 'team', 'setter_id', 'lineup_id',
]
OPTION_KEY = [*CELL_KEY, 'attack_option_id']

# Relative slots begin with the setter.  This is the SHM order used by the
# source implementation when the setter occupies rotational position 1.
SHM_RELATIVE_ROLES = (
    'setter', 'outside', 'middle', 'opposite', 'outside', 'middle',
)


def order_lineup_relative_to_setter(
    players: Iterable[object], setter_position: int
) -> tuple[str, ...]:
    """Return a six-player lineup in setter-relative rotational order."""
    players = list(players)
    if len(players) != 6:
        raise ValueError("An AEV lineup must contain exactly six players")
    if setter_position not in range(1, 7):
        raise ValueError("setter_position must be in 1..6")
    if any(pd.isna(player_id) for player_id in players):
        raise ValueError("An AEV lineup cannot contain missing player IDs")
    ordered = players[setter_position - 1:] + players[:setter_position - 1]
    ordered = tuple(str(player_id) for player_id in ordered)
    if len(set(ordered)) != 6:
        raise ValueError("An AEV lineup must contain six unique players")
    return ordered


def _source_bool(series: pd.Series, label: str) -> pd.Series:
    """Normalize source boolean values without truthiness guesses."""
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        'true': True,
        'false': False,
        '1': True,
        '0': False,
    }
    normalized = series.map(
        lambda value: mapping.get(
            value.strip().lower() if isinstance(value, str) else value,
            pd.NA,
        )
        if not pd.isna(value) else pd.NA
    )
    bad = series.notna() & normalized.isna()
    if bad.any():
        raise ValueError(
            f"{label} has unrecognized values: "
            f"{series.loc[bad].drop_duplicates().head(10).tolist()}"
        )
    return normalized.astype('boolean')


def attack_evenness(actual_shares, reference_shares) -> float:
    """Calculate the published total-variation similarity for five options."""
    actual = np.asarray(actual_shares, dtype=float)
    reference = np.asarray(reference_shares, dtype=float)
    if actual.shape != (5,) or reference.shape != (5,):
        raise ValueError("AEV requires exactly five attack options")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("AEV shares must be finite")
    if (actual < 0).any() or (reference < 0).any():
        raise ValueError("AEV shares must be non-negative")
    if actual.sum() <= 0 or reference.sum() <= 0:
        raise ValueError("AEV share vectors must have positive sums")
    actual = actual / actual.sum()
    reference = reference / reference.sum()
    value = 1.0 - 0.5 * np.abs(actual - reference).sum()
    if value < -1e-12 or value > 1 + 1e-12:
        raise AssertionError(f"AEV lies outside [0, 1]: {value}")
    return float(value)


def _player_metadata(foundation: CanonicalSeason) -> pd.DataFrame:
    required = [
        'match_id', 'team_name', 'player_id', 'player_name',
        'primary_position', 'is_libero',
    ]
    _require_columns(foundation.player_metadata, required, 'player metadata')
    metadata = foundation.player_metadata[required].copy()
    metadata['player_id'] = metadata['player_id'].astype('string')
    metadata['is_libero'] = _source_bool(
        metadata['is_libero'], 'player-info is_libero'
    )
    key = ['match_id', 'team_name', 'player_id']
    conflicts = (
        metadata.groupby(key, dropna=False)
        .agg(
            n_position=('primary_position', 'nunique'),
            n_libero=('is_libero', 'nunique'),
            n_name=('player_name', 'nunique'),
        )
        .reset_index()
    )
    bad = conflicts[
        conflicts[['n_position', 'n_libero', 'n_name']].gt(1).any(axis=1)
    ]
    if not bad.empty:
        raise ValueError(
            "Player metadata conflicts within match/team/player: "
            f"{bad.head(10).to_dict('records')}"
        )
    return metadata.drop_duplicates(key).reset_index(drop=True)


def build_non_libero_rotation_lineups(
    foundation: CanonicalSeason,
) -> pd.DataFrame:
    """Reconstruct the source's six-player rotation lineup, ignoring liberos.

    DataVolley AEV treats the back-row middle as part of the lineup with zero
    expected attack share rather than replacing that player with the libero.
    The public events normally supply non-libero set starters and explicit
    libero moves. Reusing the audited lineup builder after removing only libero
    events yields that rotation lineup. A libero server is masked solely from
    the builder's actual-court server check because the requested lineup
    intentionally omits liberos. If a feed dynamically redesignates a libero
    and then uses that player in an ordinary substitution without enough role
    history to recover SHM position, only that context is marked incomplete.
    The ordinary foundation's ``lineup_complete`` flag remains a mandatory
    eligibility condition downstream.
    """
    rallies = foundation.rallies.copy()
    _require_columns(
        rallies,
        [*RALLY_KEY, 'serving_team', 'server_player_id',
         'server_jersey_number'],
        'rally table',
    )
    # The ordinary foundation has already validated the actual server against
    # the actual on-court lineup.  That check is intentionally inapplicable to
    # this second view because a serving libero is absent from the underlying
    # non-libero rotation lineup.  Mask only the check input; no rally or actor
    # identity is changed in the canonical foundation.
    rallies['server_jersey_number'] = pd.NA

    events = foundation.source_events[
        ~foundation.source_events['event_type'].eq('libero')
    ].copy()
    rotation_lineups = build_lineup_table(
        rallies, events, foundation.identity_map
    )
    if len(rotation_lineups) != len(foundation.lineups):
        raise AssertionError("Rotation-lineup reconstruction changed rally count")
    if rotation_lineups.duplicated(RALLY_KEY).any():
        raise AssertionError("Rotation-lineup key is not unique")

    metadata = _player_metadata(foundation)
    metadata_key = {
        (row.match_id, row.team_name, str(row.player_id)): bool(row.is_libero)
        for row in metadata.itertuples(index=False)
    }
    rally_teams = foundation.rallies[
        [*RALLY_KEY, 'home_team', 'away_team']
    ]
    check = rotation_lineups.merge(
        rally_teams, on=RALLY_KEY, validate='one_to_one'
    )
    ambiguous_libero_role = pd.Series(False, index=check.index)
    for side in ('home', 'away'):
        team_column = f'{side}_team'
        for position in range(1, 7):
            player_column = f'{side}_p{position}_player_id'
            complete = check['lineup_complete'] & check[player_column].notna()
            is_libero = [
                metadata_key.get((match_id, team, str(player_id)), False)
                for match_id, team, player_id in check.loc[
                    complete, ['match_id', team_column, player_column]
                ].itertuples(index=False, name=None)
            ]
            ambiguous_libero_role.loc[complete] |= np.asarray(
                is_libero, dtype=bool
            )
    if ambiguous_libero_role.any():
        # Some non-MLV feeds allow a statically rostered libero to enter through
        # an ordinary substitution after a dynamic libero redesignation.  The
        # source records the court move but not enough role history to place the
        # player safely in AEV's SHM choice space.  Exclude only those contexts;
        # do not guess a hitter role or reject an otherwise supported season.
        affected_keys = check.loc[ambiguous_libero_role, RALLY_KEY]
        affected = rotation_lineups.merge(
            affected_keys.assign(_ambiguous_libero_role=True),
            on=RALLY_KEY,
            how='left',
            validate='one_to_one',
        )['_ambiguous_libero_role'].eq(True)
        rotation_lineups.loc[affected, 'lineup_complete'] = False
        rotation_lineups.loc[affected, 'lineup_state_valid'] = False
        rotation_lineups.loc[affected, 'lineup_status'] = (
            'ambiguous_dynamic_libero_role'
        )
    return rotation_lineups


def _team_rally_context(foundation: CanonicalSeason) -> pd.DataFrame:
    """Build one source-method lineup/setter context per team and rally."""
    metadata = _player_metadata(foundation)
    metadata_lookup = {
        (row.match_id, row.team_name, str(row.player_id)): (
            row.primary_position, bool(row.is_libero)
        )
        for row in metadata.itertuples(index=False)
    }
    rotation = build_non_libero_rotation_lineups(foundation)
    source_complete = foundation.lineups[
        [*RALLY_KEY, 'lineup_complete']
    ].rename(columns={'lineup_complete': 'source_lineup_complete'})
    rallies = foundation.rallies[
        [*RALLY_KEY, 'home_team', 'away_team']
    ]
    rotation = rotation.merge(
        source_complete, on=RALLY_KEY, validate='one_to_one'
    ).merge(rallies, on=RALLY_KEY, validate='one_to_one')

    contexts = []
    for side in ('home', 'away'):
        player_columns = [
            f'{side}_p{position}_player_id' for position in range(1, 7)
        ]
        selected = rotation[
            [*RALLY_KEY, f'{side}_team', 'lineup_complete',
             'source_lineup_complete', *player_columns]
        ].rename(columns={f'{side}_team': 'team'})
        for row in selected.itertuples(index=False):
            record = dict(zip(selected.columns, row))
            players = [record[column] for column in player_columns]
            candidates = []
            for position, player_id in enumerate(players, start=1):
                if pd.isna(player_id):
                    continue
                player_metadata = metadata_lookup.get((
                    record['match_id'], record['team'], str(player_id)
                ))
                if player_metadata is None:
                    continue
                primary_position, is_libero = player_metadata
                is_setter = (
                    primary_position == SETTER_PRIMARY_POSITION
                    and not is_libero
                )
                if is_setter:
                    candidates.append((position, str(player_id)))

            context = {
                column: record[column] for column in RALLY_KEY
            }
            context.update({
                'team': record['team'],
                'source_lineup_complete': bool(
                    record['source_lineup_complete']
                ),
                'rotation_lineup_complete': bool(record['lineup_complete']),
                'setter_candidate_count': len(candidates),
                'setter_id': pd.NA,
                'setter_position': pd.NA,
                'lineup_id': pd.NA,
            })
            for option in range(1, 6):
                context[f'option_{option}_id'] = pd.NA
                context[f'option_{option}_role'] = pd.NA
                context[f'option_{option}_expected'] = np.nan
            if len(candidates) == 1 and all(pd.notna(value) for value in players):
                setter_position, setter_id = candidates[0]
                ordered = list(order_lineup_relative_to_setter(
                    players, setter_position
                ))
                context['setter_id'] = setter_id
                context['setter_position'] = setter_position
                context['lineup_id'] = '|'.join(ordered)
                for option, (player_id, role) in enumerate(
                    zip(ordered[1:], SHM_RELATIVE_ROLES[1:]), start=1
                ):
                    court_position = players.index(player_id) + 1
                    expected = (
                        0.0 if role == 'middle' and court_position in (1, 5, 6)
                        else 0.25
                    )
                    context[f'option_{option}_id'] = player_id
                    context[f'option_{option}_role'] = role
                    context[f'option_{option}_expected'] = expected
                expected_sum = sum(
                    context[f'option_{option}_expected']
                    for option in range(1, 6)
                )
                if not np.isclose(expected_sum, 1.0):
                    raise AssertionError(
                        f"Rotation reference shares sum to {expected_sum}"
                    )
            contexts.append(context)
    result = pd.DataFrame(contexts)
    key = [*RALLY_KEY, 'team']
    if result.duplicated(key).any():
        raise AssertionError("Team-rally AEV context is not unique")
    result['setter_id'] = result['setter_id'].astype('string')
    for option in range(1, 6):
        result[f'option_{option}_id'] = result[
            f'option_{option}_id'
        ].astype('string')
    return result


def build_aev_attack_events(
    foundation: CanonicalSeason,
    contexts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach source-method lineup and designated-setter state to attacks.

    The source method counts every attack evaluation and attributes it to the
    designated setter on court.  It does not pair each attack with the actor of
    the immediately preceding set.  Setter attacks are excluded as dumps.
    Exclusion reasons remain explicit for the coverage audit.
    """
    _require_columns(
        foundation.actions,
        [*RALLY_KEY, 'action_order', 'acting_team', 'player_id',
         'source_action', 'source_outcome'],
        'action table',
    )
    attacks = foundation.actions[
        foundation.actions['source_action'].eq('A')
    ].copy()
    if contexts is None:
        contexts = _team_rally_context(foundation)
    attacks = attacks.merge(
        contexts,
        left_on=[*RALLY_KEY, 'acting_team'],
        right_on=[*RALLY_KEY, 'team'],
        how='left',
        validate='many_to_one',
        sort=False,
    )
    attacks['player_id'] = attacks['player_id'].astype('string')
    option_columns = [f'option_{option}_id' for option in range(1, 6)]
    option_known = pd.Series(False, index=attacks.index)
    for column in option_columns:
        option_known |= attacks['player_id'].eq(attacks[column])

    attacks['aev_exclusion_reason'] = pd.NA
    conditions = [
        (~attacks['source_lineup_complete'].fillna(False),
         'source_lineup_incomplete'),
        (~attacks['rotation_lineup_complete'].fillna(False),
         'rotation_lineup_incomplete'),
        (attacks['setter_candidate_count'].fillna(0).ne(1),
         'setter_unresolved'),
        (attacks['player_id'].isna(), 'attacker_identity_missing'),
        (attacks['player_id'].eq(attacks['setter_id']), 'setter_dump'),
        (~option_known, 'attacker_not_in_rotation_lineup'),
    ]
    for condition, reason in conditions:
        open_mask = attacks['aev_exclusion_reason'].isna()
        attacks.loc[open_mask & condition, 'aev_exclusion_reason'] = reason
    attacks['aev_eligible'] = attacks['aev_exclusion_reason'].isna()
    return attacks.sort_values(
        [*RALLY_KEY, 'action_order'], kind='stable'
    ).reset_index(drop=True)


def build_aev_sufficient_statistics(
    attack_events: pd.DataFrame,
) -> pd.DataFrame:
    """Return additive match/lineup/option counts for exact recomputation."""
    required = [
        *CELL_KEY, 'player_id', 'aev_eligible',
        *(f'option_{option}_id' for option in range(1, 6)),
        *(f'option_{option}_role' for option in range(1, 6)),
        *(f'option_{option}_expected' for option in range(1, 6)),
    ]
    _require_columns(attack_events, required, 'AEV attack events')
    eligible = attack_events[attack_events['aev_eligible'].eq(True)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=[
            *OPTION_KEY, 'attack_option_role', 'attack_count',
            'expected_attack_count', 'cell_attack_count',
        ])
    rows = []
    for option in range(1, 6):
        part = eligible[CELL_KEY].copy()
        part['attack_option_id'] = eligible[f'option_{option}_id'].values
        part['attack_option_role'] = eligible[f'option_{option}_role'].values
        part['attack_count'] = eligible['player_id'].eq(
            eligible[f'option_{option}_id']
        ).astype(int).values
        part['expected_attack_count'] = eligible[
            f'option_{option}_expected'
        ].astype(float).values
        rows.append(part)
    expanded = pd.concat(rows, ignore_index=True)
    result = (
        expanded.groupby(
            [*OPTION_KEY, 'attack_option_role'],
            sort=True,
            dropna=False,
            as_index=False,
        )[['attack_count', 'expected_attack_count']]
        .sum()
    )
    totals = (
        eligible.groupby(CELL_KEY, sort=True, dropna=False)
        .size()
        .rename('cell_attack_count')
        .reset_index()
    )
    result = result.merge(
        totals, on=CELL_KEY, how='left', validate='many_to_one'
    )
    checks = result.groupby(CELL_KEY, dropna=False).agg(
        n_options=('attack_option_id', 'nunique'),
        attacks=('attack_count', 'sum'),
        expected=('expected_attack_count', 'sum'),
        total=('cell_attack_count', 'first'),
    )
    if not checks['n_options'].eq(5).all():
        raise ValueError("An AEV cell does not contain five unique options")
    if not checks['attacks'].eq(checks['total']).all():
        raise AssertionError("AEV actual attack counts do not sum to cell total")
    if not np.allclose(checks['expected'], checks['total']):
        raise AssertionError("AEV expected attack counts do not sum to cell total")
    return result.sort_values(OPTION_KEY, kind='stable').reset_index(drop=True)


def build_ovlytics_aev_plays(
    foundation: CanonicalSeason,
    contexts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adapt public MLV attacks to the DataVolley columns used by ``ov_aev``.

    The adapter excludes only public-data contexts that cannot supply a
    complete underlying non-libero lineup and one designated setter. Setter
    attacks and malformed attacker assignments remain in the input so the
    author implementation performs its own exclusions.
    """
    if contexts is None:
        contexts = _team_rally_context(foundation)
    attacks = build_aev_attack_events(foundation, contexts=contexts)
    unusable = attacks['aev_exclusion_reason'].isin([
        'source_lineup_incomplete',
        'rotation_lineup_incomplete',
        'setter_unresolved',
        'attacker_identity_missing',
    ])
    attacks = attacks[~unusable].copy()

    rallies = foundation.rallies[
        [*RALLY_KEY, 'home_team', 'away_team']
    ].copy()
    rotation = build_non_libero_rotation_lineups(foundation)
    rotation_columns = [
        *RALLY_KEY,
        *(f'home_p{i}_player_id' for i in range(1, 7)),
        *(f'away_p{i}_player_id' for i in range(1, 7)),
    ]
    # Canonical actions may already carry these schedule columns. Reattach the
    # authoritative rally values without creating pandas ``_x``/``_y`` names.
    attacks = attacks.drop(
        columns=['home_team', 'away_team'], errors='ignore'
    ).merge(rallies, on=RALLY_KEY, validate='many_to_one')
    attacks = attacks.merge(
        rotation[rotation_columns], on=RALLY_KEY, validate='many_to_one'
    )
    for side, team_column in (('home', 'home_team'), ('visiting', 'away_team')):
        selected = contexts.merge(rallies, on=RALLY_KEY, validate='many_to_one')
        selected = selected[selected['team'].eq(selected[team_column])][
            [*RALLY_KEY, 'setter_position']
        ].rename(columns={'setter_position': f'{side}_setter_position'})
        attacks = attacks.merge(selected, on=RALLY_KEY, validate='many_to_one')

    output = pd.DataFrame({
        'match_id': attacks['match_id'],
        'team': attacks['acting_team'],
        'team_id': attacks['acting_team'],
        'home_team': attacks['home_team'],
        'home_team_id': attacks['home_team'],
        'visiting_team': attacks['away_team'],
        'visiting_team_id': attacks['away_team'],
        'player_id': attacks['player_id'],
        'player_name': attacks['player_id'],
        'skill': 'Attack',
        'evaluation': np.where(
            attacks['source_outcome'].eq('#'), 'Winning attack', 'Other attack'
        ),
        'home_setter_position': attacks['home_setter_position'],
        'visiting_setter_position': attacks['visiting_setter_position'],
    })
    for position in range(1, 7):
        output[f'home_player_id{position}'] = attacks[
            f'home_p{position}_player_id'
        ]
        output[f'visiting_player_id{position}'] = attacks[
            f'away_p{position}_player_id'
        ]
    return output.reset_index(drop=True)


def run_ovlytics_aev(
    plays: pd.DataFrame,
    *,
    r_library: str | Path | None = None,
) -> pd.DataFrame:
    """Run the pinned author ``ovlytics::ov_aev`` implementation in R."""
    if plays.empty:
        return pd.DataFrame(columns=['team', 'setter_id', 'aev', 'N_attacks'])
    result = run_r_csv_exchange(
        AEV_R_SCRIPT,
        inputs={'input': plays},
        outputs={'output': {'setter_id': 'string'}},
        argument_order=('input', 'output'),
        prefix='aev-ovlytics-',
        error_label='Pinned ovlytics AEV',
        library=advanced_r_library(DEFAULT_ADVANCED_R_LIBRARY, r_library),
    )
    return result['output']


def combine_aev_sufficient_statistics(
    blocks: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Add sampled team-match AEV blocks without averaging cell values.

    Callers must pass only source-qualifying (at-least-10-attack) cells or keep
    sampled occurrences distinct until that cell gate has been applied.  This
    helper then combines repeated qualifying occurrences exactly.
    """
    blocks = list(blocks)
    if not blocks:
        return pd.DataFrame(columns=[
            *OPTION_KEY, 'attack_option_role', 'attack_count',
            'expected_attack_count', 'cell_attack_count',
        ])
    required = [
        *OPTION_KEY, 'attack_option_role', 'attack_count',
        'expected_attack_count', 'cell_attack_count',
    ]
    for block in blocks:
        _require_columns(block, required, 'AEV sufficient-statistic block')
    combined = pd.concat(blocks, ignore_index=True)
    result = (
        combined.groupby(
            [*OPTION_KEY, 'attack_option_role'],
            sort=True,
            dropna=False,
            as_index=False,
        )[['attack_count', 'expected_attack_count', 'cell_attack_count']]
        .sum()
    )
    checks = result.groupby(CELL_KEY, dropna=False).agg(
        n_options=('attack_option_id', 'nunique'),
        attacks=('attack_count', 'sum'),
        expected=('expected_attack_count', 'sum'),
        total_min=('cell_attack_count', 'min'),
        total_max=('cell_attack_count', 'max'),
    )
    if not checks['n_options'].eq(5).all():
        raise ValueError("A combined AEV cell does not contain five options")
    if not checks['total_min'].eq(checks['total_max']).all():
        raise ValueError("A combined AEV cell has inconsistent totals")
    if not checks['attacks'].eq(checks['total_min']).all():
        raise AssertionError("Combined AEV attack-count identity failed")
    if not np.allclose(checks['expected'], checks['total_min']):
        raise AssertionError("Combined AEV expected-count identity failed")
    return result.sort_values(OPTION_KEY, kind='stable').reset_index(drop=True)


def qualifying_aev_sufficient_statistics(
    sufficient_statistics: pd.DataFrame,
    eligible_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict additive statistics to cells that passed the source gate."""
    _require_columns(sufficient_statistics, OPTION_KEY, 'AEV statistics')
    _require_columns(eligible_cells, CELL_KEY, 'eligible AEV cells')
    cells = eligible_cells[CELL_KEY].drop_duplicates()
    return sufficient_statistics.merge(
        cells, on=CELL_KEY, how='inner', validate='many_to_one'
    ).sort_values(OPTION_KEY, kind='stable').reset_index(drop=True)


def calculate_aev_cells(
    sufficient_statistics: pd.DataFrame,
    min_attacks: int = MIN_ATTACKS_PER_CELL,
) -> pd.DataFrame:
    """Calculate AEV for source-defined match/lineup cells."""
    _require_columns(
        sufficient_statistics,
        [*OPTION_KEY, 'attack_count', 'expected_attack_count',
         'cell_attack_count'],
        'AEV sufficient statistics',
    )
    if min_attacks < 1:
        raise ValueError("min_attacks must be positive")
    rows = []
    for key, group in sufficient_statistics.groupby(
        CELL_KEY, sort=True, dropna=False
    ):
        if len(group) != 5:
            raise ValueError(f"AEV cell {key} does not have five option rows")
        total = int(group['attack_count'].sum())
        if group['cell_attack_count'].nunique() != 1:
            raise ValueError(f"AEV cell {key} has inconsistent totals")
        if total != int(group['cell_attack_count'].iloc[0]):
            raise ValueError(f"AEV cell {key} count identity failed")
        if total < min_attacks:
            continue
        rows.append({
            **dict(zip(CELL_KEY, key)),
            'N_attacks': total,
            'AEV': attack_evenness(
                group['attack_count'].to_numpy(dtype=float),
                group['expected_attack_count'].to_numpy(dtype=float),
            ),
        })
    return pd.DataFrame(rows, columns=[*CELL_KEY, 'N_attacks', 'AEV'])


def build_qualifying_set_counts(
    foundation: CanonicalSeason,
    contexts: pd.DataFrame,
    eligible_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Count designated-setter actions in complete, eligible lineup cells.

    The source's setter-table gate is the number of Set actions made by the
    designated setter to match-lineup cells that survive the 10-attack gate.
    A set action on an individually incomplete or setter-ambiguous rally is not
    allowed to borrow eligibility merely because another rally has the same
    otherwise recoverable match-lineup key.
    """
    if eligible_cells.empty:
        return pd.DataFrame(columns=[*CELL_KEY, 'qualifying_sets'])
    context_columns = [
        *RALLY_KEY, 'team', 'setter_id', 'lineup_id',
        'source_lineup_complete', 'rotation_lineup_complete',
        'setter_candidate_count',
    ]
    _require_columns(contexts, context_columns, 'AEV team-rally contexts')
    contexts = contexts[context_columns].drop_duplicates()
    if contexts.duplicated([*RALLY_KEY, 'team']).any():
        raise AssertionError("AEV attack contexts are not team-rally unique")
    contexts = contexts[
        contexts['source_lineup_complete'].fillna(False)
        & contexts['rotation_lineup_complete'].fillna(False)
        & contexts['setter_candidate_count'].eq(1)
    ].copy()
    sets = foundation.actions[
        foundation.actions['source_action'].eq('E')
    ].copy()
    sets['player_id'] = sets['player_id'].astype('string')
    sets = sets.merge(
        contexts,
        left_on=[*RALLY_KEY, 'acting_team'],
        right_on=[*RALLY_KEY, 'team'],
        how='left',
        validate='many_to_one',
        sort=False,
    )
    sets = sets[sets['player_id'].eq(sets['setter_id'])]
    cells = eligible_cells[CELL_KEY].drop_duplicates()
    sets = sets.merge(cells, on=CELL_KEY, how='inner', validate='many_to_one')
    result = (
        sets.groupby(CELL_KEY, sort=True, as_index=False)
        .size()
        .rename(columns={'size': 'qualifying_sets'})
    )
    result['setter_id'] = result['setter_id'].astype('string')
    return result


def combine_qualifying_set_counts(
    blocks: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Add sampled team-match qualifying-set blocks with multiplicity."""
    blocks = list(blocks)
    if not blocks:
        return pd.DataFrame(columns=[*CELL_KEY, 'qualifying_sets'])
    required = [*CELL_KEY, 'qualifying_sets']
    for block in blocks:
        _require_columns(block, required, 'AEV qualifying-set block')
    combined = pd.concat(blocks, ignore_index=True)
    return (
        combined.groupby(CELL_KEY, sort=True, as_index=False)[
            'qualifying_sets'
        ]
        .sum()
    )


def calculate_season_aev(
    sufficient_statistics: pd.DataFrame,
    qualifying_set_counts: pd.DataFrame,
    min_attacks: int = MIN_ATTACKS_PER_CELL,
    min_sets: int = MIN_SETS_PER_SETTER,
) -> pd.DataFrame:
    """Aggregate cell AEV to one persistent setter-season value."""
    if min_sets < 0:
        raise ValueError("min_sets cannot be negative")
    cells = calculate_aev_cells(sufficient_statistics, min_attacks=min_attacks)
    if cells.empty:
        return pd.DataFrame(columns=[
            'season', 'player_id', 'AEV', 'N_attacks', 'qualifying_sets',
            'N_cells', 'N_teams',
        ])
    _require_columns(
        qualifying_set_counts, [*CELL_KEY, 'qualifying_sets'],
        'AEV qualifying set counts',
    )
    cells = cells.merge(
        qualifying_set_counts,
        on=CELL_KEY,
        how='left',
        validate='one_to_one',
    )
    cells['qualifying_sets'] = cells['qualifying_sets'].fillna(0).astype(int)
    rows = []
    for (season, setter_id), group in cells.groupby(
        ['season', 'setter_id'], sort=True
    ):
        n_attacks = int(group['N_attacks'].sum())
        qualifying_sets = int(group['qualifying_sets'].sum())
        if qualifying_sets < min_sets:
            continue
        rows.append({
            'season': int(season),
            'player_id': str(setter_id),
            'AEV': float(np.average(group['AEV'], weights=group['N_attacks'])),
            'N_attacks': n_attacks,
            'qualifying_sets': qualifying_sets,
            'N_cells': len(group),
            'N_teams': group['team'].nunique(),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[
            'season', 'player_id', 'AEV', 'N_attacks', 'qualifying_sets',
            'N_cells', 'N_teams',
        ])
    if result.duplicated(['season', 'player_id']).any():
        raise AssertionError("AEV output is not unique by player-season")
    if result['AEV'].lt(-1e-12).any() or result['AEV'].gt(1 + 1e-12).any():
        raise AssertionError("Season AEV lies outside [0, 1]")
    return result.sort_values(
        ['season', 'player_id'], kind='stable'
    ).reset_index(drop=True)


def calculate_aev_from_foundation(
    foundation: CanonicalSeason,
    min_attacks: int = MIN_ATTACKS_PER_CELL,
    min_sets: int = MIN_SETS_PER_SETTER,
    *,
    r_library: str | Path | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Return source-package AEV and auditable MLV adapter tables."""
    if min_attacks != MIN_ATTACKS_PER_CELL:
        raise ValueError('Source-package AEV fixes min_N_attacks at 10')
    contexts = _team_rally_context(foundation)
    attacks = build_aev_attack_events(foundation, contexts=contexts)
    statistics = build_aev_sufficient_statistics(attacks)
    cells = calculate_aev_cells(statistics, min_attacks=min_attacks)
    set_counts = build_qualifying_set_counts(
        foundation, contexts, cells
    )
    plays = build_ovlytics_aev_plays(foundation, contexts=contexts)
    source_values = run_ovlytics_aev(plays, r_library=r_library)
    if source_values.empty:
        values = pd.DataFrame(columns=[
            'season', 'player_id', 'AEV', 'N_attacks', 'qualifying_sets',
            'N_cells', 'N_teams',
        ])
        return values, attacks, statistics, cells, set_counts
    source_values['setter_id'] = source_values['setter_id'].astype('string')
    source_values['weighted_aev'] = (
        source_values['aev'] * source_values['N_attacks']
    )
    aggregated = source_values.groupby('setter_id', as_index=False).agg(
        weighted_aev=('weighted_aev', 'sum'),
        N_attacks=('N_attacks', 'sum'),
        N_teams=('team', 'nunique'),
    )
    aggregated['AEV'] = aggregated['weighted_aev'] / aggregated['N_attacks']
    gates = set_counts.groupby('setter_id', as_index=False).agg(
        qualifying_sets=('qualifying_sets', 'sum')
    )
    n_cells = cells.groupby('setter_id', as_index=False).size().rename(
        columns={'size': 'N_cells'}
    )
    values = aggregated.merge(gates, on='setter_id', how='left').merge(
        n_cells, on='setter_id', how='left'
    )
    values['qualifying_sets'] = values['qualifying_sets'].fillna(0).astype(int)
    values = values[values['qualifying_sets'].ge(min_sets)].copy()
    values.insert(0, 'season', int(foundation.actions['season'].iloc[0]))
    values = values.rename(columns={'setter_id': 'player_id'})[
        ['season', 'player_id', 'AEV', 'N_attacks', 'qualifying_sets',
         'N_cells', 'N_teams']
    ].sort_values(['season', 'player_id'], kind='stable').reset_index(drop=True)
    return values, attacks, statistics, cells, set_counts


def calculate_mlv_season_aev(
    season: int,
    *,
    r_library: str | Path | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load one audited MLV season and calculate pinned-ovlytics AEV."""
    return calculate_aev_from_foundation(
        build_mlv_canonical_season(season), r_library=r_library
    )
