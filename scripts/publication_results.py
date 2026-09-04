"""Publication figures and tables from frozen Phase 6--8 artifacts.

This module is intentionally a presentation layer.  It validates and reads
production outputs; it never estimates metrics, reruns bootstrap replicates,
or invokes sbgcop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from metric_registry import (
    ANALYSIS_SEASONS_BY_LEAGUE,
    METRIC_METADATA,
    analysis_ready_metrics_for_league,
    expected_player_seasons,
)
from production_reliability import validate_observed_artifacts


LEAGUES = ("mlv", "lovb", "au")
# Stable short labels remain part of filenames and machine-readable products.
LEAGUE_LABELS = {"mlv": "MLV", "lovb": "LOVB", "au": "AU"}
LEAGUE_DISPLAY_LABELS = {"mlv": "MLV", "lovb": "LOVB Pro", "au": "AU Pro"}
EXPECTED_RELIABILITY_POPULATIONS = {"mlv": 321, "lovb": 157, "au": 136}
EXPECTED_PCA_THRESHOLDS = {
    "mlv": (11, 16, 21),
    "lovb": (13, 19, 23),
    "au": (7, 10, 12),
}
PCA_LABEL_OFFSETS = {"mlv": (3, 4), "lovb": (3, -15), "au": (3, 4)}
DEFAULT_OBSERVED_ROOT = Path("generated") / "observed_metrics"
DEFAULT_RELIABILITY_ROOT = Path("generated") / "production_reliability"
DEFAULT_INDEPENDENCE_ROOT = Path("generated") / "production_independence"
DEFAULT_OUTPUT_ROOT = Path("generated") / "publication"

DISPLAY_LABELS = {
    "K_S": "Kills/set", "E_S": "Errors/set", "AST_S": "Assists/set",
    "SA_S": "Aces/set", "SE_S": "Serve errors/set",
    "RE_S": "Reception errors/set", "D_S": "Digs/set",
    "BT_S": "Block touches/set", "B_S": "Blocks/set",
    "PTS_S": "Points/set", "K%": "Kill %", "Eff": "Hitting efficiency",
    "Pass_Eff": "Pass efficiency", "Srv_Eff": "Serve efficiency",
    "PSF": "PSF", "AEV": "AEV", "APM": "APM",
    "SR_Hit_Pct": "SR hitting %",
    "Transition_Hit_Pct": "Transition hitting %",
    "SR_Hit_Avg": "SR hit average",
    "Transition_Hit_Avg": "Transition hit average",
    "Dig_Att_Cnv_Pct": "Dig-to-attack %", "Dig_Kill_Pct": "Dig-to-kill %",
    "Dig_Rate": "Dig rate", "Reception_Rate": "Reception rate",
    "Pct_Points_Won_On": "% points won on court",
    "Blocked_Rate": "Blocked rate", "Srv_Avg": "Serve average",
    "Attack_Error_Rate_No_Blocks": "Attack error rate (no blocks)",
    "Kill_Rate_Rally_Point": "Kills/rally point",
}
PLOT_LABELS = {
    "K_S": "K/S", "E_S": "E/S", "AST_S": "AST/S", "SA_S": "SA/S",
    "SE_S": "SE/S", "RE_S": "RE/S", "D_S": "D/S", "BT_S": "BT/S",
    "B_S": "B/S", "PTS_S": "PTS/S", "K%": "K%", "Eff": "Eff",
    "Pass_Eff": "Pass Eff", "Srv_Eff": "Srv Eff", "PSF": "PSF",
    "AEV": "AEV", "APM": "APM",
    "SR_Hit_Pct": "SR Hit %", "Transition_Hit_Pct": "Trans Hit %",
    "SR_Hit_Avg": "SR Hit Avg", "Transition_Hit_Avg": "Trans Hit Avg",
    "Dig_Att_Cnv_Pct": "Dig→Att %", "Dig_Kill_Pct": "Dig→Kill %",
    "Dig_Rate": "Dig Rate", "Reception_Rate": "Rec Rate",
    "Pct_Points_Won_On": "% Won On", "Blocked_Rate": "Blocked %",
    "Srv_Avg": "Srv Avg", "Attack_Error_Rate_No_Blocks": "Att Err %",
    "Kill_Rate_Rally_Point": "K/Rally",
}
FAMILY_LABELS = {
    "conventional": "Conventional", "advanced": "Advanced",
    "evollve": "Evollve",
}
FAMILY_COLORS = {
    "conventional": "#3569A8", "advanced": "#C85454", "evollve": "#398B68",
}
FAMILY_MARKERS = {"conventional": "o", "advanced": "D", "evollve": "^"}
RELIABILITY_FAMILY_GROUP = {
    "conventional": "conventional",
    "advanced": "advanced",
    "evollve": "advanced",
}
RELIABILITY_DISPLAY_FAMILIES = ("conventional", "advanced")


@dataclass(frozen=True)
class LeagueArtifacts:
    league: str
    metrics: tuple[str, ...]
    reliability: pd.DataFrame
    reliability_population: pd.DataFrame
    bootstrap_coverage: pd.DataFrame
    drift: pd.DataFrame
    independence_input: pd.DataFrame
    missingness: pd.DataFrame
    latent: pd.DataFrame
    latent_validation: pd.DataFrame
    scores: pd.DataFrame
    curves: pd.DataFrame
    pca: pd.DataFrame
    mcmc: pd.DataFrame
    startup: pd.DataFrame
    metadata: dict[str, str]


CURATED_FIGURES = (
    "Figure_1_Reliability_All.png", "Figure_1_Reliability_All.pdf",
    "Figure_2_Independence_Curves_All.png", "Figure_2_Independence_Curves_All.pdf",
    "Figure_3_PCA_Cumulative_Variance.png", "Figure_3_PCA_Cumulative_Variance.pdf",
    "Figure_4_Dendrogram_MLV.png", "Figure_4_Dendrogram_MLV.pdf",
    "Figure_4_Dendrogram_LOVB.png", "Figure_4_Dendrogram_LOVB.pdf",
    "Figure_4_Dendrogram_AU.png", "Figure_4_Dendrogram_AU.pdf",
)
CURATED_SUPPLEMENTAL_FIGURES = (
    "Figure_1_Reliability_MLV.png", "Figure_1_Reliability_MLV.pdf",
    "Figure_1_Reliability_LOVB.png", "Figure_1_Reliability_LOVB.pdf",
    "Figure_1_Reliability_AU.png", "Figure_1_Reliability_AU.pdf",
    "Figure_2_Independence_Curves_MLV.png",
    "Figure_2_Independence_Curves_MLV.pdf",
    "Figure_2_Independence_Curves_LOVB.png",
    "Figure_2_Independence_Curves_LOVB.pdf",
    "Figure_2_Independence_Curves_AU.png",
    "Figure_2_Independence_Curves_AU.pdf",
)
CURATED_TABLES = (
    "league_summary.csv", "league_summary.tex",
    "full_meta_metric_results.csv", "full_meta_metric_results.tex",
    "metric_rankings.csv", "metric_rankings.tex",
    "pca_summary.csv", "pca_summary.tex",
    "advanced_metric_coverage.csv", "advanced_metric_coverage.tex",
    "independence_diagnostics.csv", "independence_diagnostics.tex",
    "figure_captions.md",
)
CURATED_GENERATED_RESULTS = (
    "publication_results_long.csv", "frozen_artifact_manifest.csv",
    "publication_validation.csv", "metric_display_labels.csv",
    "dendrogram_merges_mlv.csv", "dendrogram_merges_lovb.csv",
    "dendrogram_merges_au.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_settings(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype="string")
    return dict(zip(frame["setting"], frame["value"]))


def pca_component_thresholds(pca: pd.DataFrame) -> tuple[int, int, int]:
    cumulative = pca["cumulative_variance"].to_numpy(float)
    if len(cumulative) == 0 or not np.isclose(cumulative[-1], 1.0, atol=1e-10):
        raise ValueError("PCA cumulative variance must end at one")
    return tuple(int(np.searchsorted(cumulative, level, side="left") + 1)
                 for level in (0.80, 0.90, 0.95))


def _validate_reliability_replicates(
    league: str, root: Path, metrics: tuple[str, ...]
) -> None:
    for season in ANALYSIS_SEASONS_BY_LEAGUE[league]:
        season_dir = root / league / str(season)
        replicate_dir = season_dir / "replicates"
        pkl_paths = sorted(replicate_dir.glob("*.pkl"))
        meta_paths = sorted(replicate_dir.glob("*.meta.json"))
        expected_names = [f"{index:03d}" for index in range(100)]
        if [path.stem for path in pkl_paths] != expected_names:
            raise ValueError(f"{league} {season} does not have 100 replicate artifacts")
        if [path.name.removesuffix(".meta.json") for path in meta_paths] != expected_names:
            raise ValueError(f"{league} {season} does not have 100 replicate manifests")
        for artifact, manifest_path in zip(pkl_paths, meta_paths):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("artifact_sha256") != _sha256(artifact):
                raise ValueError(f"Checksum mismatch: {artifact}")
            if (manifest.get("league"), int(manifest.get("season"))) != (league, season):
                raise ValueError(f"Wrong replicate provenance: {manifest_path}")
            if int(manifest.get("n_boot")) != 100 or int(manifest.get("seed")) != 42:
                raise ValueError(f"Wrong bootstrap configuration: {manifest_path}")
            if tuple(manifest.get("metrics", ())) != metrics:
                raise ValueError(f"Wrong replicate metric registry: {manifest_path}")
        draw_plan = pd.read_csv(season_dir / "draw_plan.csv")
        if set(draw_plan["bootstrap_id"].unique()) != set(range(100)):
            raise ValueError(f"{league} {season} draw plan is incomplete")
        identity = pd.read_csv(season_dir / "identity_errors.csv")
        if set(identity["metric"]) != set(metrics):
            raise ValueError(f"{league} {season} identity metrics changed")


def load_frozen_league(
    league: str,
    *,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
    reliability_root: str | Path = DEFAULT_RELIABILITY_ROOT,
    independence_root: str | Path = DEFAULT_INDEPENDENCE_ROOT,
) -> LeagueArtifacts:
    """Read and validate one league's frozen production artifacts without writes."""
    league = league.lower()
    metrics = analysis_ready_metrics_for_league(league)
    observed_root, reliability_root, independence_root = map(
        Path, (observed_root, reliability_root, independence_root)
    )

    for season in ANALYSIS_SEASONS_BY_LEAGUE[league]:
        observed, _ = validate_observed_artifacts(
            league, season, observed_root=observed_root
        )
        if len(observed) != expected_player_seasons(league, season):
            raise ValueError(f"{league} {season} observed population changed")

    rel_dir = reliability_root / league
    reliability = pd.read_csv(rel_dir / "reliability_results.csv")
    if reliability["Metric"].tolist() != list(metrics):
        raise ValueError(f"{league} reliability registry mismatch")
    latest = ANALYSIS_SEASONS_BY_LEAGUE[league][-1]
    latest_column = f"Discrimination_{latest}"
    if not np.allclose(
        reliability["Discrimination"], reliability[latest_column],
        equal_nan=True, rtol=0, atol=0,
    ):
        raise ValueError(f"{league} publication discrimination is not latest-season")
    reliability_population = pd.read_csv(
        rel_dir / "observed_reliability_population.csv", dtype={"player_id": "string"}
    )
    expected_backbone = sum(
        expected_player_seasons(league, season)
        for season in ANALYSIS_SEASONS_BY_LEAGUE[league]
    )
    if len(reliability_population) != expected_backbone:
        raise ValueError(f"{league} reliability backbone changed")
    eligible = int(reliability_population["sets_played"].ge(20).sum())
    if eligible != EXPECTED_RELIABILITY_POPULATIONS[league]:
        raise ValueError(f"{league} reliability population is {eligible}")
    bootstrap_coverage = pd.read_csv(rel_dir / "bootstrap_coverage.csv")
    if set(bootstrap_coverage["metric"]) != set(metrics):
        raise ValueError(f"{league} bootstrap coverage registry mismatch")
    finite_reps = bootstrap_coverage["finite_replicates"].to_numpy(float)
    if np.any(finite_reps < 0) or np.any(finite_reps > 100):
        raise ValueError(f"{league} invalid bootstrap coverage")
    drift = pd.read_csv(rel_dir / "league_drift.csv")
    if drift["Metric"].tolist() != list(metrics):
        raise ValueError(f"{league} drift registry mismatch")
    _validate_reliability_replicates(league, reliability_root, metrics)

    ind_dir = independence_root / league
    independence_input = pd.read_csv(
        ind_dir / "independence_input.csv", dtype={"player_id": "string"}
    )
    if independence_input.columns.tolist() != ["season", "player_id", *metrics]:
        raise ValueError(f"{league} independence schema mismatch")
    if len(independence_input) != expected_backbone:
        raise ValueError(f"{league} independence population changed")
    if independence_input.duplicated(["season", "player_id"]).any():
        raise ValueError(f"{league} duplicate independence player-seasons")
    missingness = pd.read_csv(ind_dir / "input_missingness.csv")
    if missingness["metric"].tolist() != list(metrics):
        raise ValueError(f"{league} missingness registry mismatch")
    if not np.array_equal(
        missingness["observed"].to_numpy(),
        independence_input[list(metrics)].notna().sum().to_numpy(),
    ):
        raise ValueError(f"{league} missingness does not match input")

    latent = pd.read_csv(ind_dir / "latent_correlation.csv")
    if latent.columns.tolist() != ["metric", *metrics] or latent["metric"].tolist() != list(metrics):
        raise ValueError(f"{league} latent C labels changed")
    C = latent[list(metrics)].to_numpy(float)
    if (C.shape != (len(metrics), len(metrics)) or not np.isfinite(C).all()
            or not np.allclose(C, C.T, atol=1e-10, rtol=0)
            or not np.allclose(np.diag(C), 1, atol=1e-10, rtol=0)):
        raise ValueError(f"{league} latent C is malformed")
    latent_validation = pd.read_csv(ind_dir / "latent_correlation_validation.csv")
    latent_stats = dict(zip(latent_validation["statistic"], latent_validation["value"]))
    eigenvalues = np.linalg.eigvalsh(C)
    calculated = {
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(np.linalg.cond(C)),
        "max_symmetry_error": float(np.max(np.abs(C - C.T))),
        "max_diagonal_error": float(np.max(np.abs(np.diag(C) - 1))),
    }
    for statistic, value in calculated.items():
        if not np.isclose(latent_stats[statistic], value, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{league} latent validation mismatch for {statistic}")
    scores = pd.read_csv(ind_dir / "independence_scores.csv")
    if set(scores["metric"]) != set(metrics) or len(scores) != len(metrics):
        raise ValueError(f"{league} independence scores changed")
    curves = pd.read_csv(ind_dir / "independence_curves.csv")
    for metric in metrics:
        curve = curves.loc[curves["target_metric"].eq(metric)]
        if curve["conditioning_size"].tolist() != list(range(len(metrics) - 1, -1, -1)):
            raise ValueError(f"{league} {metric} greedy curve sizes changed")
        values = curve["independence"].to_numpy(float)
        if not np.isclose(values[-1], 1, atol=1e-12) or np.any(np.diff(values) < -1e-8):
            raise ValueError(f"{league} {metric} greedy curve contract failed")
    pca = pd.read_csv(ind_dir / "pca_variance.csv")
    if len(pca) != len(metrics) or not np.isclose(pca["eigenvalue"].sum(), len(metrics), atol=1e-8):
        raise ValueError(f"{league} PCA dimension changed")
    if pca_component_thresholds(pca) != EXPECTED_PCA_THRESHOLDS[league]:
        raise ValueError(f"{league} PCA thresholds changed")
    mcmc = pd.read_csv(ind_dir / "mcmc_diagnostics.csv")
    startup = pd.read_csv(ind_dir / "mcmc_startup_sensitivity.csv")
    metadata = _read_settings(ind_dir / "analysis_metadata.csv")
    if metadata.get("sbgcop_version") != "0.975":
        raise ValueError(f"{league} did not use pinned sbgcop 0.975")
    if (metadata.get("seed"), metadata.get("nsamp"), metadata.get("odens")) != ("42", "5000", "5"):
        raise ValueError(f"{league} independence settings changed")
    validation = json.loads((ind_dir / "production_validation.json").read_text())
    if validation.get("input_sha256") != _sha256(ind_dir / "independence_input.csv"):
        raise ValueError(f"{league} frozen input checksum mismatch")
    if validation.get("posterior_sha256") != _sha256(ind_dir / "C_posterior_samples.rds"):
        raise ValueError(f"{league} frozen posterior checksum mismatch")

    return LeagueArtifacts(
        league, metrics, reliability, reliability_population, bootstrap_coverage,
        drift, independence_input, missingness, latent, latent_validation,
        scores, curves, pca, mcmc, startup, metadata,
    )


def deterministic_rank(frame: pd.DataFrame, value: str, *, ascending: bool) -> pd.Series:
    """Sequential rank with metric-name lexical tie breaking and NA preserved."""
    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    finite = frame.loc[pd.to_numeric(frame[value], errors="coerce").notna()].copy()
    finite = finite.sort_values([value, "metric"], ascending=[ascending, True], kind="stable")
    result.loc[finite.index] = np.arange(1, len(finite) + 1)
    return result


def build_long_results(artifacts: dict[str, LeagueArtifacts]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in LEAGUES:
        item = artifacts[league]
        rel = item.reliability.rename(columns={"Metric": "metric"}).copy()
        scores = item.scores.rename(columns={"independence": "Independence"})
        frame = rel.merge(scores[["metric", "Independence"]], on="metric", validate="one_to_one")
        missing = item.missingness.rename(columns={
            "observed": "independence_observed", "missing": "independence_missing",
            "unique_observed": "independence_unique_observed",
        })
        frame = frame.merge(missing, on="metric", validate="one_to_one")
        frame.insert(0, "league", league)
        frame.insert(2, "display_label", frame["metric"].map(DISPLAY_LABELS))
        frame.insert(3, "plot_label", frame["metric"].map(PLOT_LABELS))
        frame.insert(4, "family", frame["metric"].map(lambda name: METRIC_METADATA[name].family))
        latest = ANALYSIS_SEASONS_BY_LEAGUE[league][-1]
        latest_coverage = item.bootstrap_coverage.loc[
            item.bootstrap_coverage["season"].eq(latest)
        ]
        used_counts = latest_coverage.groupby("metric")["used_for_bv"].sum()
        replicate_min = latest_coverage.groupby("metric")["finite_replicates"].min()
        replicate_median = latest_coverage.groupby("metric")["finite_replicates"].median()
        frame["reliability_latest_eligible"] = frame["metric"].map(used_counts).astype("Int64")
        frame["bootstrap_finite_replicates_min"] = frame["metric"].map(replicate_min)
        frame["bootstrap_finite_replicates_median"] = frame["metric"].map(replicate_median)
        frame["independence_rank"] = deterministic_rank(frame, "Independence", ascending=False)
        frame["discrimination_rank"] = deterministic_rank(frame, "Discrimination", ascending=False)
        frame["stability_rank"] = deterministic_rank(frame, "Stability_corrected", ascending=False)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13.5,
        "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5, "figure.dpi": 140,
        "savefig.dpi": 320, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save_figure(
    fig: plt.Figure, directory: Path, stem: str, *, tight: bool = True
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bounding_box = "tight" if tight else None
    fig.savefig(
        directory / f"{stem}.png", bbox_inches=bounding_box, facecolor="white",
        metadata={"Software": "Volleyball Meta-Analytics publication_results.py"},
    )
    fig.savefig(
        directory / f"{stem}.pdf", bbox_inches=bounding_box, facecolor="white",
        metadata={
            "Creator": "Volleyball Meta-Analytics publication_results.py",
            "CreationDate": None, "ModDate": None,
        },
    )
    plt.close(fig)


def _canonicalize_pdf_timestamps(path: Path) -> None:
    """Replace base-R PDF date digits in place without changing xref offsets."""
    original = path.read_bytes()
    canonical = re.sub(
        rb"/(CreationDate|ModDate) \(D:\d{14}",
        lambda match: b"/" + match.group(1) + b" (D:20000101000000",
        original,
    )
    if len(canonical) != len(original):
        raise AssertionError("PDF timestamp canonicalization changed byte offsets")
    path.write_bytes(canonical)


def _label_points(
    ax: plt.Axes, frame: pd.DataFrame, x: str, y: str, *, fontsize: float
) -> list[plt.Text]:
    """Place deterministic, repelled labels and return them for validation."""
    finite = frame.dropna(subset=[x, y]).sort_values([x, y, "metric"])
    texts = [
        ax.text(
            getattr(row, x), getattr(row, y), row.plot_label,
            fontsize=fontsize, ha="center", va="center", color="#202020",
            zorder=4,
        )
        for row in finite.itertuples(index=False)
    ]
    # adjustText uses random jitter only to separate exactly coincident labels.
    # Save and restore NumPy's global state so the publication output is stable
    # without influencing any caller's random stream.
    random_state = np.random.get_state()
    np.random.seed(42)
    try:
        adjust_text(
            texts,
            x=finite[x].to_numpy(float), y=finite[y].to_numpy(float), ax=ax,
            prevent_crossings=True, ensure_inside_axes=True,
            expand=(1.35, 1.65), force_text=(0.70, 1.00),
            force_static=(0.30, 0.50), force_pull=(0.008, 0.008),
            max_move=(24, 28), explode_radius="auto", iter_lim=2000,
        )
    finally:
        np.random.set_state(random_state)
    _separate_remaining_label_overlaps(ax, texts)
    # Draw subtle leaders after the final deterministic placement so their
    # endpoints follow any small post-repulsion adjustment.
    for text, row in zip(texts, finite.itertuples(index=False)):
        point = np.asarray(ax.transData.transform((getattr(row, x), getattr(row, y))))
        label = np.asarray(ax.transData.transform(text.get_position()))
        if np.linalg.norm(point - label) >= 9:
            ax.annotate(
                "", xy=(getattr(row, x), getattr(row, y)),
                xytext=text.get_position(), textcoords="data",
                arrowprops={"arrowstyle": "-", "color": "#777777",
                            "linewidth": .45, "alpha": .65},
                zorder=2,
            )
    return texts


def _separate_remaining_label_overlaps(
    ax: plt.Axes, texts: list[plt.Text]
) -> None:
    """Deterministically clear any pixel-level overlaps left by adjustText."""
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    occupied = []
    offsets = [(0.0, 0.0)]
    step = 4.0
    for radius in range(1, 31):
        ring = [
            (dx * step, dy * step)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if max(abs(dx), abs(dy)) == radius
        ]
        ring.sort(key=lambda offset: (
            offset[0] ** 2 + offset[1] ** 2,
            abs(offset[0]), abs(offset[1]), offset[1] < 0, offset[0] < 0,
        ))
        offsets.extend(ring)

    for text in texts:
        base_box = text.get_window_extent(renderer).expanded(1.02, 1.10)
        base_position = np.asarray(text.get_transform().transform(text.get_position()))
        for dx, dy in offsets:
            candidate = base_box.translated(dx, dy)
            inside = (
                candidate.x0 >= axes_box.x0 and candidate.x1 <= axes_box.x1
                and candidate.y0 >= axes_box.y0 and candidate.y1 <= axes_box.y1
            )
            if inside and not any(candidate.overlaps(previous) for previous in occupied):
                if dx or dy:
                    text.set_position(
                        text.get_transform().inverted().transform(
                            base_position + np.asarray((dx, dy))
                        )
                    )
                occupied.append(candidate)
                break
        else:
            raise RuntimeError("Unable to place all reliability labels without overlap")


def _plot_reliability_panel(
    ax: plt.Axes, frame: pd.DataFrame, league: str, *, label_fontsize: float
) -> list[plt.Text]:
    reliability_group = frame["family"].map(RELIABILITY_FAMILY_GROUP)
    for family in RELIABILITY_DISPLAY_FAMILIES:
        subset = frame.loc[reliability_group.eq(family)].dropna(
            subset=["Discrimination", "Stability_corrected"]
        )
        ax.scatter(
            subset["Discrimination"], subset["Stability_corrected"],
            color=FAMILY_COLORS[family], marker=FAMILY_MARKERS[family], s=28,
            linewidth=.5, edgecolor="white", label=FAMILY_LABELS[family], zorder=3,
        )
    missing = frame.loc[frame["Stability_corrected"].isna(), "display_label"].tolist()
    if missing:
        ax.text(.02, .02, "Stability unavailable: " + ", ".join(missing),
                transform=ax.transAxes, fontsize=9, color="#555", va="bottom")
    ax.axhline(0, color="#9a9a9a", linewidth=.6, linestyle="--", zorder=0)
    ax.axvline(0, color="#9a9a9a", linewidth=.6, linestyle="--", zorder=0)
    ax.margins(x=.08, y=.08)
    ax.grid(color="#e6e6e6", linewidth=.5, zorder=0)
    ax.set_title(LEAGUE_DISPLAY_LABELS[league], fontweight="semibold")
    latest_year = ANALYSIS_SEASONS_BY_LEAGUE[league][-1]
    ax.set_xlabel(f"Discrimination ({latest_year})")
    ax.set_ylabel("Corrected stability")
    ax.set_box_aspect(1)
    labels = _label_points(
        ax, frame, "Discrimination", "Stability_corrected",
        fontsize=label_fontsize,
    )
    return labels


def create_reliability_figures(long: pd.DataFrame, directory: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.8), constrained_layout=False)
    fig.subplots_adjust(left=.065, right=.985, bottom=.11, top=.74, wspace=.34)
    for ax, league in zip(axes, LEAGUES):
        _plot_reliability_panel(
            ax, long.loc[long["league"].eq(league)], league,
            label_fontsize=9.0,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .905))
    fig.suptitle(
        "Metric discrimination and stability by professional league",
        y=.975, fontsize=17,
    )
    _save_figure(fig, directory, "Figure_1_Reliability_All")
    for league in LEAGUES:
        fig, ax = plt.subplots(figsize=(8.2, 8.2), constrained_layout=False)
        fig.subplots_adjust(left=.06, right=.985, bottom=.125, top=.935)
        _plot_reliability_panel(
            ax, long.loc[long["league"].eq(league)], league,
            label_fontsize=11.5,
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, frameon=False, ncol=2, loc="lower center",
            bbox_to_anchor=(.5, .005), borderaxespad=0,
        )
        _save_figure(
            fig, directory, f"Figure_1_Reliability_{LEAGUE_LABELS[league]}",
            tight=False,
        )


def _plot_curve_panel(
    ax: plt.Axes, item: LeagueArtifacts, *, legend_fontsize: float
) -> list[str]:
    ranking = item.scores.sort_values(["independence", "metric"], ascending=[False, True])
    highlighted = ranking.head(5)["metric"].tolist()
    for metric in item.metrics:
        curve = item.curves.loc[item.curves["target_metric"].eq(metric)]
        if metric in highlighted:
            family = METRIC_METADATA[metric].family
            ax.plot(curve["conditioning_size"], curve["independence"],
                    color=FAMILY_COLORS[family], linewidth=2.0, alpha=.95,
                    label=DISPLAY_LABELS[metric], zorder=3)
        else:
            ax.plot(curve["conditioning_size"], curve["independence"],
                    color="#9aa0a6", linewidth=.65, alpha=.35, zorder=1)
    ax.grid(color="#e6e6e6", linewidth=.5)
    ax.set_title(LEAGUE_DISPLAY_LABELS[item.league], fontweight="semibold")
    ax.set_xlabel("Metrics in conditioning set")
    ax.set_ylabel("Independence")
    ax.set_box_aspect(1)
    ax.legend(frameon=False, fontsize=legend_fontsize, loc="best")
    return highlighted


def create_independence_figures(
    artifacts: dict[str, LeagueArtifacts], directory: Path
) -> dict[str, list[str]]:
    _style()
    highlighted: dict[str, list[str]] = {}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=.065, right=.985, bottom=.11, top=.79, wspace=.34)
    for ax, league in zip(axes, LEAGUES):
        highlighted[league] = _plot_curve_panel(
            ax, artifacts[league], legend_fontsize=9.5,
        )
    fig.suptitle("Greedy conditional-independence curves", y=.975, fontsize=17)
    _save_figure(fig, directory, "Figure_2_Independence_Curves_All")
    for league in LEAGUES:
        fig, ax = plt.subplots(figsize=(8.2, 8.2), constrained_layout=True)
        _plot_curve_panel(ax, artifacts[league], legend_fontsize=11.5)
        _save_figure(
            fig, directory, f"Figure_2_Independence_Curves_{LEAGUE_LABELS[league]}",
            tight=False,
        )
    return highlighted


def create_pca_figure(artifacts: dict[str, LeagueArtifacts], directory: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(12.5, 4.9), constrained_layout=True)
    colors = {"mlv": "#3569A8", "lovb": "#C85454", "au": "#398B68"}
    for league in LEAGUES:
        pca = artifacts[league].pca
        ax.plot(pca["component"], pca["cumulative_variance"],
                marker="o", markersize=3.2, linewidth=2, color=colors[league],
                label=LEAGUE_DISPLAY_LABELS[league])
        for count, level in zip(pca_component_thresholds(pca), (.80, .90, .95)):
            observed_level = pca.loc[
                pca["component"].eq(count), "cumulative_variance"
            ].iloc[0]
            ax.scatter([count], [observed_level],
                       s=28, color=colors[league], edgecolor="white", linewidth=.4)
            label_offset = PCA_LABEL_OFFSETS[league]
            vertical_alignment = "top" if league == "lovb" else "bottom"
            ax.annotate(str(count), (count, observed_level), xytext=label_offset,
                        textcoords="offset points", fontsize=10.5,
                        va=vertical_alignment,
                        color=colors[league], fontweight="bold")
    for level in (.80, .90, .95):
        ax.axhline(level, color="#8c8c8c", linewidth=.7, linestyle="--")
        ax.text(.4, level + .006, f"{int(level * 100)}%", fontsize=10, color="#666")
    ax.set_xlim(left=.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative latent variance explained")
    ax.set_title(
        "PCA of posterior-mean latent metric correlations",
        fontweight="bold",
    )
    ax.grid(color="#e8e8e8", linewidth=.5)
    ax.legend(frameon=False)
    _save_figure(fig, directory, "Figure_3_PCA_Cumulative_Variance")


def create_dendrograms(
    artifacts: dict[str, LeagueArtifacts], directory: Path,
    *, r_script: str | Path | None = None,
    independence_root: str | Path = DEFAULT_INDEPENDENCE_ROOT,
) -> pd.DataFrame:
    """Render exact R hclust(dist(abs(C))) dendrograms from frozen C."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required for exact publication dendrograms")
    r_program = Path(r_script or Path(__file__).with_name("publication_dendrogram.R"))
    directory.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame({
        "metric": list(DISPLAY_LABELS),
        "display_label": list(DISPLAY_LABELS.values()),
        "plot_label": [PLOT_LABELS[name] for name in DISPLAY_LABELS],
        "family": [METRIC_METADATA[name].family for name in DISPLAY_LABELS],
    })
    mapping_path = directory.parent / "data" / "metric_display_labels.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_path, index=False)
    merge_frames = []
    for league in LEAGUES:
        latent_path = Path(independence_root) / league / "latent_correlation.csv"
        output_stem = directory / f"Figure_4_Dendrogram_{LEAGUE_LABELS[league]}"
        merge_path = directory.parent / "data" / f"dendrogram_merges_{league}.csv"
        subprocess.run([
            rscript, str(r_program), str(latent_path), str(mapping_path),
            str(output_stem.with_suffix(".png")), str(output_stem.with_suffix(".pdf")),
            LEAGUE_DISPLAY_LABELS[league], str(merge_path),
        ], check=True)
        _canonicalize_pdf_timestamps(output_stem.with_suffix(".pdf"))
        merges = pd.read_csv(merge_path)
        merges.insert(0, "league", league)
        merge_frames.append(merges)
    return pd.concat(merge_frames, ignore_index=True)


def _format_value(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _write_table(frame: pd.DataFrame, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{stem}.csv", index=False, na_rep="NA")
    (directory / f"{stem}.md").write_text(dataframe_to_markdown(frame))
    latex = frame.to_latex(index=False, na_rep="NA", float_format=lambda value: f"{value:.3f}")
    (directory / f"{stem}.tex").write_text(latex)


def build_rankings(long: pd.DataFrame, number: int = 5) -> pd.DataFrame:
    records = []
    statistics = {
        "Discrimination": "Discrimination",
        "Stability_corrected": "Corrected stability",
        "Independence": "Independence",
    }
    for league in LEAGUES:
        league_frame = long.loc[long["league"].eq(league)]
        for column, label in statistics.items():
            finite = league_frame.dropna(subset=[column])
            for direction, ascending in (("highest", False), ("lowest", True)):
                ordered = finite.sort_values([column, "metric"], ascending=[ascending, True], kind="stable").head(number)
                for rank, row in enumerate(ordered.itertuples(), 1):
                    records.append({"league": league, "statistic": label,
                                    "direction": direction, "rank": rank,
                                    "metric": row.metric, "display_label": row.display_label,
                                    "value": getattr(row, column)})
    return pd.DataFrame(records)


def _advanced_coverage(artifacts: dict[str, LeagueArtifacts]) -> pd.DataFrame:
    records = []
    for league in LEAGUES:
        item = artifacts[league]
        coverage = item.bootstrap_coverage
        for metric in ("PSF", "AEV", "APM"):
            supported = metric in item.metrics
            status = "ANALYZED" if supported else "NOT_IN_AU_SCOPE"
            missing_row = item.missingness.loc[item.missingness["metric"].eq(metric)]
            metric_coverage = coverage.loc[coverage["metric"].eq(metric)]
            records.append({
                "league": league, "metric": metric, "status": status,
                "independence_observed": (int(missing_row["observed"].iloc[0]) if len(missing_row) else pd.NA),
                "independence_missing": (int(missing_row["missing"].iloc[0]) if len(missing_row) else pd.NA),
                "bootstrap_finite_replicates_min": (int(metric_coverage["finite_replicates"].min()) if len(metric_coverage) else pd.NA),
                "bootstrap_finite_replicates_median": (float(metric_coverage["finite_replicates"].median()) if len(metric_coverage) else pd.NA),
                "note": "",
            })
    return pd.DataFrame(records)


def create_tables(
    artifacts: dict[str, LeagueArtifacts], long: pd.DataFrame,
    directory: Path,
) -> dict[str, pd.DataFrame]:
    summaries = []
    pca_rows = []
    for league in LEAGUES:
        item = artifacts[league]
        thresholds = pca_component_thresholds(item.pca)
        observed = len(item.independence_input)
        summaries.append({
            "league": LEAGUE_LABELS[league],
            "seasons": ", ".join(map(str, ANALYSIS_SEASONS_BY_LEAGUE[league])),
            "observed_player_seasons": observed,
            "reliability_player_seasons": EXPECTED_RELIABILITY_POPULATIONS[league],
            "analyzed_metrics": len(item.metrics),
            "PCs_80": thresholds[0], "PCs_90": thresholds[1], "PCs_95": thresholds[2],
        })
        pca_rows.append({"league": LEAGUE_LABELS[league], "metric_count": len(item.metrics),
                         "PCs_80": thresholds[0], "PCs_90": thresholds[1], "PCs_95": thresholds[2]})
    league_summary = pd.DataFrame(summaries)
    pca_summary = pd.DataFrame(pca_rows)
    full_columns = [
        "league", "metric", "display_label", "family", "Discrimination",
        "Discrimination_avg", "Stability_corrected", "Stab_CI_low",
        "Stab_CI_high", "Stability_paper", "n_multi_season", "Independence",
        "independence_rank", "discrimination_rank", "stability_rank",
        "independence_observed", "independence_missing",
        "reliability_latest_eligible", "bootstrap_finite_replicates_min",
        "bootstrap_finite_replicates_median",
    ]
    full = long[full_columns].copy()
    full.insert(
        full.columns.get_loc("Stab_CI_low"),
        "Stability_95pct_CI",
        [
            ("NA" if pd.isna(low) or pd.isna(high) else f"[{low:.3f}, {high:.3f}]")
            for low, high in zip(full["Stab_CI_low"], full["Stab_CI_high"])
        ],
    )
    full["stability_note"] = ""
    full.loc[
        full["league"].eq("lovb") & full["metric"].eq("Dig_Kill_Pct"),
        "stability_note",
    ] = "Unavailable: noise-adjusted denominator is nonpositive"
    full["league"] = full["league"].map(LEAGUE_LABELS)
    full["family"] = full["family"].map(FAMILY_LABELS)
    rankings = build_rankings(long)
    rankings["league"] = rankings["league"].map(LEAGUE_LABELS)
    advanced = _advanced_coverage(artifacts)
    advanced["league"] = advanced["league"].map(LEAGUE_LABELS)
    diagnostic_rows = []
    for league in LEAGUES:
        item = artifacts[league]
        mcmc = dict(zip(item.mcmc["statistic"], item.mcmc["value"]))
        latent_stats = dict(zip(
            item.latent_validation["statistic"], item.latent_validation["value"]
        ))
        startup20 = item.startup.loc[np.isclose(item.startup["discarded_fraction"], .2)].iloc[0]
        diagnostic_rows.append({
            "league": LEAGUE_LABELS[league],
            "saved_posterior_samples": int(mcmc["saved_samples"]),
            "ESS_min": mcmc["min_effective_sample_size"],
            "ESS_median": mcmc["median_effective_sample_size"],
            "ESS_max": mcmc["max_effective_sample_size"],
            "max_first_vs_second_half_C_difference": mcmc["max_first_vs_second_half_C_difference"],
            "startup_20pct_max_C_difference": startup20["max_C_difference_from_all"],
            "startup_20pct_max_score_difference": startup20["max_score_difference_from_all"],
            "min_eigenvalue": latent_stats["min_eigenvalue"],
            "condition_number": latent_stats["condition_number"],
        })
    diagnostics = pd.DataFrame(diagnostic_rows)
    tables = {
        "league_summary": league_summary,
        "full_meta_metric_results": full,
        "metric_rankings": rankings,
        "pca_summary": pca_summary,
        "advanced_metric_coverage": advanced,
        "independence_diagnostics": diagnostics,
    }
    for stem, frame in tables.items():
        _write_table(frame, directory, stem)
    return tables


def _artifact_manifest(
    reliability_root: Path, independence_root: Path
) -> pd.DataFrame:
    records = []
    reliability_files = (
        "reliability_results.csv", "observed_reliability_population.csv",
        "bootstrap_coverage.csv", "bootstrap_variance.csv", "league_drift.csv",
    )
    independence_files = (
        "independence_input.csv", "input_missingness.csv", "latent_correlation.csv",
        "C_posterior_samples.rds", "mcmc_diagnostics.csv",
        "mcmc_startup_sensitivity.csv", "independence_scores.csv",
        "independence_curves.csv", "pca_variance.csv", "pca_loadings.csv",
    )
    for league in LEAGUES:
        for root, files, phase in ((reliability_root, reliability_files, "reliability"),
                                   (independence_root, independence_files, "independence")):
            for name in files:
                path = root / league / name
                records.append({"league": league, "phase": phase,
                                "path": str(path), "sha256": _sha256(path)})
    return pd.DataFrame(records)


def export_curated_publication_bundle(
    artifacts: dict[str, LeagueArtifacts],
    *,
    generated_root: str | Path = DEFAULT_OUTPUT_ROOT,
    tracked_root: str | Path = ".",
    independence_root: str | Path = DEFAULT_INDEPENDENCE_ROOT,
) -> pd.DataFrame:
    """Export the small durable manuscript bundle from internal generated output.

    Large replicate caches, posterior samples, input populations, and run logs
    deliberately remain under ``generated/``.  This function copies only the
    approved figures/tables and writes compact canonical numerical results.
    """
    generated_root, tracked_root, independence_root = map(
        Path, (generated_root, tracked_root, independence_root)
    )
    figure_dir = tracked_root / "figures"
    supplemental_dir = figure_dir / "supplemental"
    table_dir = tracked_root / "tables"
    result_dir = tracked_root / "results"
    for directory in (figure_dir, supplemental_dir, table_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []

    def copy_curated(source: Path, destination: Path, category: str, purpose: str) -> None:
        if not source.is_file():
            raise ValueError(f"Missing generated publication artifact: {source}")
        shutil.copyfile(source, destination)
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"Curated copy differs from generated source: {destination}")
        records.append({
            "category": category, "path": str(destination),
            "generated_source": str(source), "sha256": _sha256(destination),
            "purpose": purpose,
        })

    for name in CURATED_FIGURES:
        purpose = (
            "Main reliability comparison" if name.startswith("Figure_1")
            else "Main independence-curve comparison" if name.startswith("Figure_2")
            else "Main latent-PCA summary" if name.startswith("Figure_3")
            else "League-specific Franks dendrogram"
        )
        copy_curated(generated_root / "figures" / name, figure_dir / name,
                     "figure", purpose)
    for name in CURATED_SUPPLEMENTAL_FIGURES:
        purpose = (
            "Readable league-specific reliability panel"
            if name.startswith("Figure_1")
            else "Readable league-specific independence curves"
        )
        copy_curated(generated_root / "figures" / name, supplemental_dir / name,
                     "supplemental_figure", purpose)
    for name in CURATED_TABLES:
        copy_curated(generated_root / "tables" / name, table_dir / name,
                     "table", "Manuscript or supplement table")
    for name in CURATED_GENERATED_RESULTS:
        copy_curated(generated_root / "data" / name, result_dir / name,
                     "result", "Compact publication provenance or numerical result")

    combined_results = {
        "pca_variance_all.csv": pd.concat([
            artifacts[league].pca.assign(league=league).loc[:, [
                "league", *artifacts[league].pca.columns
            ]]
            for league in LEAGUES
        ], ignore_index=True),
        "independence_curves_all.csv": pd.concat([
            artifacts[league].curves.assign(league=league).loc[:, [
                "league", *artifacts[league].curves.columns
            ]]
            for league in LEAGUES
        ], ignore_index=True),
        "league_drift_all.csv": pd.concat([
            artifacts[league].drift.assign(league=league).loc[:, [
                "league", *artifacts[league].drift.columns
            ]]
            for league in LEAGUES
        ], ignore_index=True),
        "pca_loadings_all.csv": pd.concat([
            pd.read_csv(independence_root / league / "pca_loadings.csv")
            .assign(league=league)
            for league in LEAGUES
        ], ignore_index=True),
    }
    purposes = {
        "pca_variance_all.csv": "Exact PCA eigenvalue and variance curves",
        "independence_curves_all.csv": "Exact frozen greedy independence curves",
        "league_drift_all.csv": "Final reliability league-drift diagnostics",
        "pca_loadings_all.csv": "Posterior-mean latent PCA loadings",
    }
    for name, frame in combined_results.items():
        path = result_dir / name
        frame.to_csv(path, index=False, na_rep="NA")
        records.append({
            "category": "result", "path": str(path),
            "generated_source": "combined frozen Phase 7/8 CSV artifacts",
            "sha256": _sha256(path), "purpose": purposes[name],
        })
    for league in LEAGUES:
        path = result_dir / f"latent_correlation_{league}.csv"
        artifacts[league].latent.to_csv(path, index=False)
        records.append({
            "category": "result", "path": str(path),
            "generated_source": str(independence_root / league / "latent_correlation.csv"),
            "sha256": _sha256(path), "purpose": "Frozen posterior-mean latent C",
        })

    manifest = pd.DataFrame(records).sort_values(["category", "path"], kind="stable")
    manifest.to_csv(result_dir / "PUBLICATION_BUNDLE_MANIFEST.csv", index=False)
    expected = set(manifest["path"]) | {str(result_dir / "PUBLICATION_BUNDLE_MANIFEST.csv")}
    actual = {
        str(path) for directory in (figure_dir, table_dir, result_dir)
        for path in directory.rglob("*") if path.is_file()
    }
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(f"Curated bundle inventory mismatch; unexpected={unexpected}, missing={missing}")
    return manifest


def _top_sentence(rankings: pd.DataFrame, league: str, statistic: str, direction: str) -> str:
    rows = rankings.loc[
        rankings["league"].eq(LEAGUE_LABELS[league])
        & rankings["statistic"].eq(statistic)
        & rankings["direction"].eq(direction)
    ]
    return ", ".join(f"{row.display_label} ({row.value:.3f})" for row in rows.itertuples())


def _cluster_pairs(merges: pd.DataFrame, league: str, number: int = 4) -> str:
    subset = merges.loc[(merges["league"].eq(league)) & (merges["size"].eq(2))]
    subset = subset.sort_values(["height", "members"], kind="stable").head(number)
    return "; ".join(f"{row.members} (height {row.height:.3f})" for row in subset.itertuples())


def _drift_summary(item: LeagueArtifacts, number: int = 5) -> str:
    ordered = item.drift.sort_values(
        ["drift_share_of_SV", "Metric"], ascending=[False, True], kind="stable"
    ).head(number)
    return ", ".join(
        f"{DISPLAY_LABELS[row.Metric]} ({row.drift_share_of_SV:.3f})"
        for row in ordered.itertuples()
    )


def figure_captions(artifacts: dict[str, LeagueArtifacts]) -> str:
    observed = {league: len(artifacts[league].independence_input) for league in LEAGUES}
    metric_counts = {league: len(artifacts[league].metrics) for league in LEAGUES}
    return "\n".join([
        "# Draft Figure Captions", "",
        "**Figure 1.** Latest-season discrimination versus corrected stability for the reliability-eligible "
        f"MLV ({EXPECTED_RELIABILITY_POPULATIONS['mlv']} player-seasons; {metric_counts['mlv']} metrics), "
        f"LOVB Pro ({EXPECTED_RELIABILITY_POPULATIONS['lovb']}; {metric_counts['lovb']}), and "
        f"AU Pro ({EXPECTED_RELIABILITY_POPULATIONS['au']}; {metric_counts['au']}) populations. Points identify "
        "conventional versus advanced metrics; Evollve-sourced metrics are included in the Advanced category. "
        "Estimates are shown without clipping; LOVB Dig-to-kill % lacks stability because its "
        "noise-adjusted denominator is nonpositive.", "",
        "**Figure 2.** Greedy conditional-independence curves for the unfiltered pooled observed populations "
        f"(MLV {observed['mlv']} player-seasons, LOVB Pro {observed['lovb']}, "
        f"AU Pro {observed['au']}). "
        "The horizontal axis is the number of conditioning metrics "
        "and the vertical axis is stored independence. All curves are shown; the five metrics with the highest "
        "full-conditioning independence are highlighted deterministically.", "",
        "**Figure 3.** Cumulative variance explained by principal components of each league's frozen posterior-mean "
        "sbgcop latent correlation matrix. Dashed lines mark 80%, 90%, and 95% variance.", "",
        "**Figure 4.** League-specific complete-linkage hierarchical clustering from "
        "`hclust(dist(abs(C)))`, where C is the frozen posterior-mean latent correlation matrix. Smaller height "
        "denotes more similar rows of absolute latent correlations, not necessarily positive association.", "",
    ])


def write_results_memo(
    artifacts: dict[str, LeagueArtifacts], tables: dict[str, pd.DataFrame],
    merges: pd.DataFrame, highlighted: dict[str, list[str]], path: Path,
) -> None:
    rankings = tables["metric_rankings"]
    aev_observed = {
        league: int(
            artifacts[league].missingness.loc[
                artifacts[league].missingness["metric"].eq("AEV"), "observed"
            ].iloc[0]
        )
        for league in ("mlv", "lovb")
    }
    lines = [
        "# Publication Results", "",
        "This memo is generated only from the frozen Phase 6–8 production artifacts. "
        "It reports statistical properties of metrics—not volleyball importance, causal value, "
        "or predictive relevance to winning.", "",
        "## Analysis populations", "",
        dataframe_to_markdown(tables["league_summary"]).strip(), "",
        f"MLV and LOVB analyze {len(artifacts['mlv'].metrics)} metrics; AU analyzes its "
        f"{len(artifacts['au'].metrics)} source-supported metrics.", "",
        "## Discrimination and stability", "",
    ]
    for league in LEAGUES:
        rel = artifacts[league].reliability
        outside = rel.loc[
            rel["Stability_corrected"].notna()
            & ((rel["Stability_corrected"] < 0) | (rel["Stability_corrected"] > 1)),
            ["Metric", "Stability_corrected"],
        ]
        outside_text = (", ".join(
            f"{DISPLAY_LABELS[row.Metric]} ({row.Stability_corrected:.3f})"
            for row in outside.itertuples()
        ) or "none")
        lines += [
            f"### {LEAGUE_LABELS[league]}", "",
            f"Highest latest-season discrimination: {_top_sentence(rankings, league, 'Discrimination', 'highest')}.",
            f"Lowest latest-season discrimination: {_top_sentence(rankings, league, 'Discrimination', 'lowest')}.",
            f"Highest corrected stability: {_top_sentence(rankings, league, 'Corrected stability', 'highest')}.",
            f"Lowest corrected stability: {_top_sentence(rankings, league, 'Corrected stability', 'lowest')}.",
            f"Finite corrected estimates outside [0, 1]: {outside_text}. Such values are preserved because "
            "finite-sample noise-corrected estimators are not probabilities and were not clamped.", "",
        ]
    lines += [
        "LOVB Dig-to-kill % has no corrected stability estimate because its noise-adjusted denominator "
        "is nonpositive; the missing value is retained rather than replaced.", "",
        "## Independence", "",
    ]
    for league in LEAGUES:
        lines += [
            f"**{LEAGUE_LABELS[league]}:** highest full-conditioning independence: "
            f"{_top_sentence(rankings, league, 'Independence', 'highest')}. Lowest: "
            f"{_top_sentence(rankings, league, 'Independence', 'lowest')}. "
            f"Figure 2 highlights exactly the five highest values ({', '.join(DISPLAY_LABELS[m] for m in highlighted[league])}); "
            "all other curves remain visible in the background.", "",
        ]
    lines += [
        "High independence means that a metric contributes latent information not represented by the "
        "conditioning metrics; it does not establish sporting quality or importance.", "",
        "## PCA and clustering", "",
        dataframe_to_markdown(tables["pca_summary"]).strip(), "",
    ]
    for league in LEAGUES:
        lines += [f"Early two-metric merges for {LEAGUE_LABELS[league]}: {_cluster_pairs(merges, league)}."]
    lines += [
        "", "Dendrograms use exactly `hclust(dist(abs(C)))` with R's default complete linkage. "
        "The use of absolute latent correlations means proximity can reflect either positive or negative association.", "",
        "## Missingness and diagnostic context", "",
        f"AEV has a limited finite independence population ({aev_observed['mlv']} MLV and "
        f"{aev_observed['lovb']} LOVB player-seasons), reflecting "
        "its source eligibility gates. Phase 8 also reported weaker MCMC mixing for some AEV-related latent "
        "correlations, especially the AEV–assists pairing; these diagnostics warrant caution without changing "
        "the frozen posterior. Metric-specific missingness was passed to sbgcop without complete-case deletion.", "",
        f"The five largest league-drift shares were {_drift_summary(artifacts['mlv'])} in MLV; "
        f"{_drift_summary(artifacts['lovb'])} in LOVB; and {_drift_summary(artifacts['au'])} in AU. "
        "Drift is descriptive of season-to-season mean movement and is not a causal league comparison. "
        "The source files are `generated/production_reliability/<league>/league_drift.csv`.", "",
        f"Phase 8 saved {int(dict(zip(artifacts['mlv'].mcmc['statistic'], artifacts['mlv'].mcmc['value']))['saved_samples']):,} "
        "posterior C samples per league. MLV/LOVB/AU minimum ESS values were "
        f"{dict(zip(artifacts['mlv'].mcmc['statistic'], artifacts['mlv'].mcmc['value']))['min_effective_sample_size']:.2f}, "
        f"{dict(zip(artifacts['lovb'].mcmc['statistic'], artifacts['lovb'].mcmc['value']))['min_effective_sample_size']:.2f}, and "
        f"{dict(zip(artifacts['au'].mcmc['statistic'], artifacts['au'].mcmc['value']))['min_effective_sample_size']:.2f}; "
        "the maximum first-half versus second-half C differences were "
        f"{dict(zip(artifacts['mlv'].mcmc['statistic'], artifacts['mlv'].mcmc['value']))['max_first_vs_second_half_C_difference']:.3f}, "
        f"{dict(zip(artifacts['lovb'].mcmc['statistic'], artifacts['lovb'].mcmc['value']))['max_first_vs_second_half_C_difference']:.3f}, and "
        f"{dict(zip(artifacts['au'].mcmc['statistic'], artifacts['au'].mcmc['value']))['max_first_vs_second_half_C_difference']:.3f}. "
        "These are diagnostics, not post-hoc acceptance thresholds.", "",
        "## Numerical provenance", "",
        "Primary reliability numbers come from each league's `reliability_results.csv`; independence rankings "
        "from `independence_scores.csv`; curves from `independence_curves.csv`; PCA from `pca_variance.csv`; "
        "and clustering from the frozen `latent_correlation.csv`. Their exact generated paths and SHA-256 "
        "hashes are recorded in `results/frozen_artifact_manifest.csv`; the joined values used by the figures "
        "and tables are retained in `results/publication_results_long.csv`.", "",
        "## Figure captions", "", figure_captions(artifacts).replace("# Draft Figure Captions\n\n", "").strip(), "",
        "## Suggested Results Narrative", "",
    ]
    narrative = _suggested_narrative(artifacts, tables, merges)
    word_count = len(narrative.split())
    if not 800 <= word_count <= 1200:
        raise AssertionError(f"Suggested Results Narrative has {word_count} words")
    lines += [narrative, "", f"*Narrative word count: {word_count}.*", ""]
    path.write_text("\n".join(lines))


def _suggested_narrative(
    artifacts: dict[str, LeagueArtifacts], tables: dict[str, pd.DataFrame], merges: pd.DataFrame
) -> str:
    rankings = tables["metric_rankings"]
    observed = {league: len(artifacts[league].independence_input) for league in LEAGUES}
    counts = {league: len(artifacts[league].metrics) for league in LEAGUES}
    family_counts = {
        league: pd.Series(
            [METRIC_METADATA[metric].family for metric in artifacts[league].metrics]
        ).value_counts().to_dict()
        for league in LEAGUES
    }
    aev_observed = {
        league: int(
            artifacts[league].missingness.loc[
                artifacts[league].missingness["metric"].eq("AEV"), "observed"
            ].iloc[0]
        )
        for league in ("mlv", "lovb")
    }
    pca_thresholds = {
        league: pca_component_thresholds(artifacts[league].pca)
        for league in LEAGUES
    }
    return f"""The production analysis comprised {observed['mlv']} MLV player-seasons from 2024–2026, {observed['lovb']} LOVB player-seasons from 2025–2026, and {observed['au']} AU player-seasons from 2022–2025. Reliability analyses used the previously specified observed-population rule, yielding {EXPECTED_RELIABILITY_POPULATIONS['mlv']} MLV, {EXPECTED_RELIABILITY_POPULATIONS['lovb']} LOVB, and {EXPECTED_RELIABILITY_POPULATIONS['au']} AU player-seasons. Independence retained the full observed backbones and metric-defined missingness rather than importing the reliability eligibility rule. MLV and LOVB each contributed {counts['mlv']} analysis-ready metrics: {family_counts['mlv']['conventional']} conventional measures, PSF, AEV, APM, and {family_counts['mlv']['evollve']} Evollve measures. AU contributed the {counts['au']} metrics supported by its source data: the {family_counts['au']['conventional']} conventional measures and {family_counts['au']['evollve']} boxscore-derived Evollve rates.

Latest-season discrimination varied substantially across metrics. In MLV, the five highest estimates were {_top_sentence(rankings, 'mlv', 'Discrimination', 'highest')}, whereas the five lowest were {_top_sentence(rankings, 'mlv', 'Discrimination', 'lowest')}. LOVB showed highest values for {_top_sentence(rankings, 'lovb', 'Discrimination', 'highest')} and lowest values for {_top_sentence(rankings, 'lovb', 'Discrimination', 'lowest')}. In AU, the corresponding high group was {_top_sentence(rankings, 'au', 'Discrimination', 'highest')}, and the low group was {_top_sentence(rankings, 'au', 'Discrimination', 'lowest')}. These quantities characterize whether observed between-player differences exceed estimated resampling noise. They do not establish which metrics are more consequential to match outcomes.

Corrected stability likewise differed by league and metric. MLV's highest estimates were {_top_sentence(rankings, 'mlv', 'Corrected stability', 'highest')}; its lowest were {_top_sentence(rankings, 'mlv', 'Corrected stability', 'lowest')}. LOVB's highest were {_top_sentence(rankings, 'lovb', 'Corrected stability', 'highest')}, while its lowest finite estimates were {_top_sentence(rankings, 'lovb', 'Corrected stability', 'lowest')}. Dig-to-kill percentage had no LOVB stability estimate because the required noise-adjusted denominator was nonpositive. AU's highest stability values were {_top_sentence(rankings, 'au', 'Corrected stability', 'highest')}, and its lowest were {_top_sentence(rankings, 'au', 'Corrected stability', 'lowest')}. Several MLV and LOVB estimates fell below zero or above one. Those values were retained because the corrected finite-sample estimator is not constrained to the unit interval. Confidence intervals, including unusually wide intervals for sparse or noisy metrics, were also preserved without clipping.

Full-conditioning independence identified different sources of nonredundant latent information. MLV's five highest values were {_top_sentence(rankings, 'mlv', 'Independence', 'highest')}; the lowest were {_top_sentence(rankings, 'mlv', 'Independence', 'lowest')}. LOVB's five highest were {_top_sentence(rankings, 'lovb', 'Independence', 'highest')}, and its lowest were {_top_sentence(rankings, 'lovb', 'Independence', 'lowest')}. AU's five highest were {_top_sentence(rankings, 'au', 'Independence', 'highest')}, compared with lows of {_top_sentence(rankings, 'au', 'Independence', 'lowest')}. These rankings were derived from each league's posterior-mean semiparametric Gaussian-copula correlation matrix. High independence indicates information not linearly accounted for in the latent Gaussian representation by the remaining metrics; it should not be described as better performance measurement without a separate criterion.

The PCA results indicated considerable redundancy but no extremely low-dimensional representation. MLV required {pca_thresholds['mlv'][0]}, {pca_thresholds['mlv'][1]}, and {pca_thresholds['mlv'][2]} components to explain 80%, 90%, and 95% of latent variance, respectively. LOVB required {pca_thresholds['lovb'][0]}, {pca_thresholds['lovb'][1]}, and {pca_thresholds['lovb'][2]} components, while AU required {pca_thresholds['au'][0]}, {pca_thresholds['au'][1]}, and {pca_thresholds['au'][2]}. The AU counts partly reflect its smaller {counts['au']}-metric universe, so direct component-count comparisons should consider different matrix dimensions. Hierarchical clustering used the audited Franks construction, complete-linkage clustering of Euclidean distances between rows of the absolute latent correlation matrix. Early pairings included {_cluster_pairs(merges, 'mlv', 3)} in MLV; {_cluster_pairs(merges, 'lovb', 3)} in LOVB; and {_cluster_pairs(merges, 'au', 3)} in AU. Because absolute correlations define the row profiles, cluster proximity can arise from either positive or negative relationships and should not be read as interchangeability.

Cross-league contrasts were descriptive rather than pooled. Assists per set was highly discriminative in all three leagues, but the relative stability and independence of other measures varied. Pass efficiency had high full-conditioning independence in each setting where it was analyzed, while several scoring-volume measures occupied the lower end of the independence rankings. League-drift diagnostics also differed: pass-efficiency drift was especially prominent in LOVB and AU, whereas MLV's largest shares included block touches, AEV, and dig-to-attack conversion. These patterns could reflect schedule, role, population, or measurement differences and do not by themselves identify mechanisms.

Missingness and sampling uncertainty remain important. AEV produced only {aev_observed['mlv']} finite MLV and {aev_observed['lovb']} finite LOVB observations for independence because its source-defined attack-cell and setter-season gates are restrictive. Phase 8 diagnostics found generally usable posterior sampling but weaker effective sample sizes and larger startup or half-chain differences for selected correlations, notably some involving AEV and assists. We therefore report the frozen values while flagging their diagnostic context. LOVB has only two analyzed seasons, limiting the information available for stability relative to MLV's three and AU's four. Finally, AU's source limitations prevented faithful inclusion of lineup-, sequence-, and model-dependent metrics; its smaller metric set should not be treated as though unsupported quantities were zero. Across all displays and tables, missing estimates, out-of-range corrected estimates, and league-specific support differences were retained exactly."""


def generate_publication_results(
    *,
    observed_root: str | Path = DEFAULT_OBSERVED_ROOT,
    reliability_root: str | Path = DEFAULT_RELIABILITY_ROOT,
    independence_root: str | Path = DEFAULT_INDEPENDENCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    memo_path: str | Path = "PUBLICATION_RESULTS.md",
    tracked_root: str | Path | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    figure_dir, table_dir, data_dir = (
        output_root / "figures", output_root / "tables", output_root / "data"
    )
    artifacts = {
        league: load_frozen_league(
            league, observed_root=observed_root,
            reliability_root=reliability_root,
            independence_root=independence_root,
        )
        for league in LEAGUES
    }
    long = build_long_results(artifacts)
    data_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(data_dir / "publication_results_long.csv", index=False, na_rep="NA")
    manifest = _artifact_manifest(Path(reliability_root), Path(independence_root))
    manifest.to_csv(data_dir / "frozen_artifact_manifest.csv", index=False)
    validation = pd.DataFrame([
        {
            "league": league,
            "observed_player_seasons": len(item.independence_input),
            "reliability_player_seasons": EXPECTED_RELIABILITY_POPULATIONS[league],
            "metrics": len(item.metrics),
            "bootstrap_replicates_per_season": 100,
            "latent_symmetry_error": float(np.max(np.abs(
                item.latent[list(item.metrics)].to_numpy(float)
                - item.latent[list(item.metrics)].to_numpy(float).T))),
            "pca_cumulative_final": float(item.pca["cumulative_variance"].iloc[-1]),
        }
        for league, item in artifacts.items()
    ])
    validation.to_csv(data_dir / "publication_validation.csv", index=False)
    create_reliability_figures(long, figure_dir)
    highlighted = create_independence_figures(artifacts, figure_dir)
    create_pca_figure(artifacts, figure_dir)
    merges = create_dendrograms(
        artifacts, figure_dir, independence_root=independence_root
    )
    tables = create_tables(artifacts, long, table_dir)
    (table_dir / "figure_captions.md").write_text(figure_captions(artifacts))
    write_results_memo(artifacts, tables, merges, highlighted, Path(memo_path))
    curated_manifest = None
    if tracked_root is not None:
        curated_manifest = export_curated_publication_bundle(
            artifacts, generated_root=output_root, tracked_root=tracked_root,
            independence_root=independence_root,
        )
    return {"artifacts": artifacts, "long": long, "tables": tables,
            "merges": merges, "highlighted": highlighted,
            "validation": validation, "curated_manifest": curated_manifest}
