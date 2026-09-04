"""League-independent canonical data contracts used by volleyball metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


RALLY_KEY = ['season', 'match_id', 'set_number', 'point_number']
IDENTITY_KEY = ['match_id', 'team_name', '_jersey_number']


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    """Fail with one consistent message when a canonical input is incomplete."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def normalize_jersey_numbers(series: pd.Series) -> pd.Series:
    """Return positive integral source jerseys with missing values explicit."""
    numeric = pd.to_numeric(series, errors='coerce')
    malformed = series.notna() & numeric.isna()
    nonintegral = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
    if malformed.any() or nonintegral.any():
        examples = series.loc[malformed | nonintegral].drop_duplicates().head(10)
        raise ValueError(f"Malformed source jersey numbers: {examples.tolist()}")
    return numeric.where(numeric > 0).round().astype('Int64')


@dataclass(frozen=True)
class CanonicalSeason:
    """Canonical products for one league-season.

    The first seven frames are the stable analytical contract. Source frames
    remain available for narrow metric adapters that must reproduce an author
    package's input schema; those adapters must not reinterpret source identity.
    """

    league: str
    season: int
    schedule: pd.DataFrame
    player_sets: pd.DataFrame
    player_metadata: pd.DataFrame
    actions: pd.DataFrame
    rallies: pd.DataFrame
    lineups: pd.DataFrame
    identity_map: pd.DataFrame
    source_actions: pd.DataFrame
    source_events: pd.DataFrame
    source_player_info: pd.DataFrame
    source_player_boxscore: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.league:
            raise ValueError("CanonicalSeason requires a league identifier")
        object.__setattr__(self, 'league', str(self.league).lower())
        object.__setattr__(self, 'season', int(self.season))
        if self.schedule['match_id'].duplicated().any():
            raise ValueError("Canonical schedule must be unique by match_id")
        if self.rallies.duplicated(RALLY_KEY).any():
            raise ValueError("Canonical rallies must be unique by rally key")
        if self.lineups.duplicated(RALLY_KEY).any():
            raise ValueError("Canonical lineups must be unique by rally key")


def build_canonical_season(league: str, season: int) -> CanonicalSeason:
    """Build a canonical season through the registered league adapter."""
    normalized = str(league).lower()
    if normalized == 'mlv':
        from mlv_adapter import build_mlv_canonical_season

        return build_mlv_canonical_season(int(season))
    if normalized == 'lovb':
        from lovb_adapter import build_lovb_canonical_season

        return build_lovb_canonical_season(int(season))
    if normalized == 'au':
        from au_adapter import build_au_canonical_season

        return build_au_canonical_season(int(season))
    raise NotImplementedError(
        f"No production canonical adapter is registered for league {league!r}"
    )
