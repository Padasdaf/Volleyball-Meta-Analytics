"""Authoritative league, season, and player-metric analysis registry."""

from __future__ import annotations

from dataclasses import dataclass


ADDITIVE_RECOMPUTATION = 'additive_sufficient_statistics'
DETERMINISTIC_RECOMPUTATION = 'deterministic_estimator'
FULL_MODEL_REFIT = 'full_model_refit'

ANALYSIS_SEASONS_BY_LEAGUE = {
    'mlv': (2024, 2025, 2026),
    'lovb': (2025, 2026),
    'au': (2022, 2023, 2024, 2025),
}

# Audited canonical player-season population before metric-defined missingness.
# Production runners validate this contract rather than deriving their backbone
# from whichever metric family happens to execute first.
EXPECTED_PLAYER_SEASONS = {
    ('mlv', 2024): 119,
    ('mlv', 2025): 138,
    ('mlv', 2026): 135,
    ('lovb', 2025): 94,
    ('lovb', 2026): 98,
    ('au', 2022): 44,
    ('au', 2023): 45,
    ('au', 2024): 44,
    ('au', 2025): 44,
}

CONVENTIONAL_METRICS = (
    'K_S', 'E_S', 'AST_S', 'SA_S', 'SE_S', 'RE_S', 'D_S', 'BT_S',
    'B_S', 'PTS_S', 'K%', 'Eff', 'Pass_Eff', 'Srv_Eff',
)
ADVANCED_METRICS = ('PSF', 'AEV', 'APM')
EVOLLVE_METRICS = (
    'SR_Hit_Pct',
    'Transition_Hit_Pct',
    'SR_Hit_Avg',
    'Transition_Hit_Avg',
    'Dig_Att_Cnv_Pct',
    'Dig_Kill_Pct',
    'Dig_Rate',
    'Reception_Rate',
    'Pct_Points_Won_On',
    'Blocked_Rate',
    'Srv_Avg',
    'Attack_Error_Rate_No_Blocks',
    'Kill_Rate_Rally_Point',
)

ALL_ANALYSIS_METRICS = (
    *CONVENTIONAL_METRICS,
    *ADVANCED_METRICS,
    *EVOLLVE_METRICS,
)
# Backwards-compatible name used by existing production validation code. As
# of Phase 5, production analysis includes the promoted Evollve metrics.
PRODUCTION_METRICS = ALL_ANALYSIS_METRICS

AU_ANALYSIS_METRICS = (
    *CONVENTIONAL_METRICS,
    'Blocked_Rate',
    'Srv_Avg',
    'Attack_Error_Rate_No_Blocks',
)
ANALYSIS_METRICS_BY_LEAGUE = {
    'mlv': ALL_ANALYSIS_METRICS,
    'lovb': ALL_ANALYSIS_METRICS,
    'au': AU_ANALYSIS_METRICS,
}

ADVANCED_SOURCE_FIDELITY = {
    'PSF': 'EXACT',
    'AEV': 'SOURCE-EXACT WITH DATA ADAPTER',
    'APM': 'PAPER-FAITHFUL RECONSTRUCTION WITH TUNING/SCOPE AMBIGUITY',
}

_SEQUENCE_EVOLLVE = {
    'SR_Hit_Pct', 'Transition_Hit_Pct', 'SR_Hit_Avg',
    'Transition_Hit_Avg', 'Dig_Att_Cnv_Pct', 'Dig_Kill_Pct',
}
_EXPOSURE_EVOLLVE = {
    'Dig_Rate', 'Reception_Rate', 'Pct_Points_Won_On',
    'Kill_Rate_Rally_Point',
}
_BOXSCORE_EVOLLVE = {
    'Blocked_Rate', 'Srv_Avg', 'Attack_Error_Rate_No_Blocks',
}


@dataclass(frozen=True)
class MetricSpec:
    """Machine-readable execution and source-support metadata for one metric."""

    name: str
    family: str
    source_fidelity: str
    input_requirements: tuple[str, ...]
    bootstrap_class: str
    analysis_leagues: tuple[str, ...]
    league_support: tuple[tuple[str, str], ...]
    expensive_to_fit: bool = False
    requires_model_refit: bool = False
    shared_fit_group: str | None = None

    def support_for(self, league: str) -> str:
        support = dict(self.league_support)
        try:
            return support[str(league).lower()]
        except KeyError as exc:
            raise ValueError(f'Unknown league {league!r}') from exc


def _analysis_leagues(metric: str) -> tuple[str, ...]:
    return tuple(
        league
        for league in ANALYSIS_SEASONS_BY_LEAGUE
        if metric in ANALYSIS_METRICS_BY_LEAGUE[league]
    )


def _support(metric: str, family: str) -> tuple[tuple[str, str], ...]:
    if family == 'conventional' or metric in _BOXSCORE_EVOLLVE:
        values = {'mlv': 'EXACT', 'lovb': 'EXACT', 'au': 'EXACT'}
    elif family == 'advanced':
        values = {
            'PSF': {
                'mlv': 'EXACT', 'lovb': 'SOURCE-EXACT WITH DATA ADAPTER',
                'au': 'DATA-CONSTRAINED',
            },
            'AEV': {
                'mlv': 'SOURCE-EXACT WITH DATA ADAPTER',
                'lovb': 'SOURCE-EXACT WITH DATA ADAPTER',
                'au': 'DATA-CONSTRAINED',
            },
            'APM': {
                'mlv': 'PAPER-FAITHFUL RECONSTRUCTION',
                'lovb': 'PAPER-FAITHFUL RECONSTRUCTION',
                'au': 'DATA-CONSTRAINED',
            },
        }[metric]
    else:
        mlv_lovb = (
            'FAITHFULLY RECONSTRUCTABLE'
            if metric in _SEQUENCE_EVOLLVE
            else 'SOURCE-EXACT WITH DATA ADAPTER'
        )
        values = {
            'mlv': mlv_lovb,
            'lovb': mlv_lovb,
            'au': 'DATA-CONSTRAINED',
        }
    return tuple((league, values[league]) for league in ANALYSIS_SEASONS_BY_LEAGUE)


def _spec(metric: str, family: str) -> MetricSpec:
    if family == 'conventional':
        requirements = ('boxscore', 'participation')
        fidelity = 'AUDITED SOURCE DEFINITION'
    elif family == 'advanced':
        requirements = {
            'PSF': ('sequence',),
            'AEV': ('sequence', 'lineup', 'player_metadata'),
            'APM': ('lineup',),
        }[metric]
        fidelity = ADVANCED_SOURCE_FIDELITY[metric]
    elif metric in _SEQUENCE_EVOLLVE:
        requirements = ('sequence',)
        fidelity = 'GLOSSARY-FAITHFUL RECONSTRUCTION'
    elif metric in _EXPOSURE_EVOLLVE:
        requirements = ('sequence', 'lineup')
        fidelity = 'GLOSSARY-FAITHFUL RECONSTRUCTION'
    else:
        requirements = ('boxscore',)
        fidelity = 'EXACT GLOSSARY FORMULA'

    bootstrap_class = {
        'AEV': DETERMINISTIC_RECOMPUTATION,
        'APM': FULL_MODEL_REFIT,
    }.get(metric, ADDITIVE_RECOMPUTATION)
    full_refit = bootstrap_class == FULL_MODEL_REFIT
    return MetricSpec(
        name=metric,
        family=family,
        source_fidelity=fidelity,
        input_requirements=requirements,
        bootstrap_class=bootstrap_class,
        analysis_leagues=_analysis_leagues(metric),
        league_support=_support(metric, family),
        expensive_to_fit=full_refit,
        requires_model_refit=full_refit,
        shared_fit_group=None,
    )


REGISTRY = tuple(
    _spec(metric, 'conventional') for metric in CONVENTIONAL_METRICS
) + tuple(
    _spec(metric, 'advanced') for metric in ADVANCED_METRICS
) + tuple(
    _spec(metric, 'evollve') for metric in EVOLLVE_METRICS
)
METRIC_METADATA = {spec.name: spec for spec in REGISTRY}


def metrics_for_league(league: str) -> tuple[str, ...]:
    try:
        return ANALYSIS_METRICS_BY_LEAGUE[str(league).lower()]
    except KeyError as exc:
        raise ValueError(f'Unknown analysis league {league!r}') from exc


def analysis_ready_metrics_for_league(league: str) -> tuple[str, ...]:
    """Return the active published metrics for one league."""
    return metrics_for_league(league)


def seasons_for_league(league: str) -> tuple[int, ...]:
    try:
        return ANALYSIS_SEASONS_BY_LEAGUE[str(league).lower()]
    except KeyError as exc:
        raise ValueError(f'Unknown analysis league {league!r}') from exc


def expected_player_seasons(league: str, season: int) -> int:
    key = (str(league).lower(), int(season))
    try:
        return EXPECTED_PLAYER_SEASONS[key]
    except KeyError as exc:
        raise ValueError(f'Unknown analysis league-season {key!r}') from exc


def validate_metric_columns(columns, *, league: str | None = None) -> None:
    required = (
        ALL_ANALYSIS_METRICS if league is None else metrics_for_league(league)
    )
    missing = set(required) - set(columns)
    if missing:
        scope = 'analysis' if league is None else f'{league} analysis'
        raise ValueError(f"{scope.title()} metric output is missing: {sorted(missing)}")


def _validate_registry() -> None:
    if len(ALL_ANALYSIS_METRICS) != 30:
        raise AssertionError('The published metric universe must contain 30 metrics')
    if len(set(ALL_ANALYSIS_METRICS)) != len(ALL_ANALYSIS_METRICS):
        raise AssertionError('Metric names must be unique')
    if set(METRIC_METADATA) != set(ALL_ANALYSIS_METRICS):
        raise AssertionError('Every analysis metric needs metadata')
    expected_scope = {
        (league, season)
        for league, seasons in ANALYSIS_SEASONS_BY_LEAGUE.items()
        for season in seasons
    }
    if set(EXPECTED_PLAYER_SEASONS) != expected_scope:
        raise AssertionError('Every frozen league-season needs an expected population')
    for league, metrics in ANALYSIS_METRICS_BY_LEAGUE.items():
        if len(metrics) != len(set(metrics)):
            raise AssertionError(f'{league} analysis metrics are not unique')
        unsupported = {
            metric for metric in metrics
            if METRIC_METADATA[metric].support_for(league) == 'DATA-CONSTRAINED'
        }
        if unsupported:
            raise AssertionError(
                f'{league} analysis set contains data-constrained metrics: '
                f'{sorted(unsupported)}'
            )
_validate_registry()
