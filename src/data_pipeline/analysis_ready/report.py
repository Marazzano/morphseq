"""analysis_ready report (TERMINAL) — the only report whose input surface is the whole joined DAG.

Three artifacts, all over the predicted_stage_hpf axis (analysis_ready is the one step that HAS
stage + genotype + embeddings together, so it is no longer constrained to time_index):

  1. latent_projection      — PCA→UMAP of the z_mu_b_* (biological, else flat z_mu_*) block, drawn
                              twice: colored by predicted_stage_hpf and by genotype.
  2. survival_over_stage    — recapitulates death_detection's alive_embryos_experiment survival
                              curve, but on the stage axis: total alive physical embryos vs stage.
  3. genotype_survival_panel— the same survival curve, overlaid across genotypes (header) then one
                              small-multiple per genotype (3-col gallery; 1 col if a single genotype).

Wiring only: derives report-only values (survival counts, latent selection) and hands columns to the
shared renderers in viz/reporting.py. Imports its own contract + viz/reporting only — never matplotlib
directly, and nothing imports this module (terminal leaf).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.viz.reporting import (
    plot_grouped_survival_panel,
    plot_latent_projection_dual,
    plot_line,
    plot_well_survival_over_time,
)

from .contract import select_latent_columns

# Genotype colors are an analyze-side convention; import lazily/defensively so a missing analyze
# tree does not break the whole report (colors are cosmetic — the renderer falls back to defaults).
try:
    from analyze.viz.styling.color_mapping_config import get_color_for_genotype  # type: ignore
except Exception:  # pragma: no cover - cosmetic fallback
    get_color_for_genotype = None  # type: ignore

_STAGE_COL = "predicted_stage_hpf"
_GENOTYPE_COL = "genotype"
_ANIMAL_COL = "physical_embryo_id"
_DEATH_STAGE_COL = "death_event_stage_hpf"
_WELL_COL = "well_id"
_QC_REASONS_COL = "qc_fail_reasons"
_PERSISTENCE_FLAG = "persistence_dead_flag"
# Stage bin width (hpf) for the per-well × stage survival heatmap: predicted_stage_hpf is continuous,
# so we bin it into a stable, legible column axis (mirrors the death_detection time_index heatmap).
_STAGE_BIN_HPF = 2.0


def _fraction_alive_per_well_stage(analysis_ready: pd.DataFrame) -> pd.DataFrame:
    """Per (well_id, stage_bin[, genotype]): FRACTION of snips alive on the predicted_stage_hpf axis.

    "Alive" reuses the persistence flag already folded into the snip_qc verdict: a snip is alive
    unless ``qc_fail_reasons`` names ``persistence_dead_flag`` (no new death_detection input needed —
    analysis_ready carries the SNIP_QC payload). Stage is binned to _STAGE_BIN_HPF for a stable axis.
    Returns long rows with the bin CENTER as ``stage_hpf`` for the shared heatmap helper.
    """
    df = analysis_ready.dropna(subset=[_STAGE_COL]).copy()
    reasons = df[_QC_REASONS_COL].fillna("").astype(str)
    df["alive"] = ~reasons.str.contains(_PERSISTENCE_FLAG, regex=False)
    # Bin stage to a fixed grid; report the bin CENTER so the axis reads in hpf, not bin index.
    df["stage_hpf"] = (np.floor(df[_STAGE_COL] / _STAGE_BIN_HPF) * _STAGE_BIN_HPF) + _STAGE_BIN_HPF / 2.0
    keys = [_WELL_COL, "stage_hpf"]
    if _GENOTYPE_COL in df.columns:
        keys.append(_GENOTYPE_COL)
    return df.groupby(keys)["alive"].mean().reset_index(name="frac_alive")


def _genotype_colors(genotypes: list[str]) -> dict[str, str] | None:
    if get_color_for_genotype is None:
        return None
    return {g: get_color_for_genotype(g) for g in genotypes}


def _survival_over_stage(
    analysis_ready: pd.DataFrame, death_event: pd.DataFrame, *, subset_mask: pd.Series | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Total alive physical embryos over a stage grid — the stage-axis recapitulation of
    death_detection's experiment survival curve.

    An animal is alive at stage s if its death_event_stage_hpf is null (never died) or > s. The
    stage grid is the sorted unique per-animal max stage observed (the stages at which the cohort is
    actually sampled). Zero-fill is inherent: as animals die, the count draws down and stays down.
    """
    ar = analysis_ready if subset_mask is None else analysis_ready[subset_mask]

    # One row per animal: the death stage (null = survived) restricted to animals in this subset.
    animals = ar[[_ANIMAL_COL]].drop_duplicates()
    death_by_animal = death_event.set_index(_ANIMAL_COL)[_DEATH_STAGE_COL]
    animals = animals.copy()
    animals["death_stage"] = animals[_ANIMAL_COL].map(death_by_animal)

    # Stage grid = the stages the cohort is observed at (per-snip predicted_stage_hpf, deduped).
    stages = np.sort(ar[_STAGE_COL].dropna().unique())
    if stages.size == 0:
        return np.array([]), np.array([])

    death = animals["death_stage"].to_numpy(dtype=float)
    n_total = len(animals)
    # alive at s = animals with NaN death OR death > s.
    alive = np.array([int(np.sum(np.isnan(death) | (death > s))) for s in stages])
    # Guard: never exceed cohort size.
    alive = np.minimum(alive, n_total)
    return stages, alive


def build_analysis_ready_report(
    *,
    analysis_ready_parquet: Path,
    death_event_csv: Path,
    output_latent_projection_png: Path,
    output_survival_over_stage_png: Path,
    output_genotype_survival_panel_png: Path,
    output_well_survival_over_stage_png: Path,
) -> None:
    ar = pd.read_parquet(analysis_ready_parquet)
    death_event = pd.read_csv(death_event_csv)

    # ── 1. Latent projection (dual-colored) ──────────────────────────────────────────────────
    latent_cols = select_latent_columns(ar.columns)
    if not latent_cols:
        raise ValueError(
            "analysis_ready report: no z_mu_* latent columns in the analysis_ready table — cannot "
            "build the latent projection panel."
        )
    genotypes_all = sorted(ar[_GENOTYPE_COL].dropna().astype(str).unique())
    plot_latent_projection_dual(
        ar,
        latent_cols,
        continuous_col=_STAGE_COL,
        categorical_col=_GENOTYPE_COL,
        title="analysis_ready — latent PCA→UMAP",
        output_path=Path(output_latent_projection_png),
        categorical_colors=_genotype_colors(genotypes_all),
    )

    # ── 2. Experiment survival over stage ────────────────────────────────────────────────────
    stages, alive = _survival_over_stage(ar, death_event)
    plot_line(
        pd.Series(stages),
        pd.Series(alive),
        title="analysis_ready — total alive embryos over predicted_stage_hpf",
        output_path=Path(output_survival_over_stage_png),
        xlabel=_STAGE_COL,
        ylabel="total alive embryos",
    )

    # ── 3. Per-genotype survival panel (overlay + gallery) ───────────────────────────────────
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for g in genotypes_all:
        mask = ar[_GENOTYPE_COL].astype(str) == g
        gx, gy = _survival_over_stage(ar, death_event, subset_mask=mask)
        if gx.size:
            curves[g] = (gx, gy)
    if curves:
        plot_grouped_survival_panel(
            curves,
            title="analysis_ready — genotype survival over predicted_stage_hpf",
            output_path=Path(output_genotype_survival_panel_png),
            xlabel=_STAGE_COL,
            ylabel="total alive embryos",
            group_colors=_genotype_colors(list(curves)),
        )

    # ── 4. Per-well fraction-alive over predicted_stage_hpf, GENOTYPE-faceted ─────────────────
    # The stage-axis, genotype-split twin of the death_detection well×time survival heatmap: same
    # shared renderer, global panel + one panel per genotype. Fraction color scale fixed 0-1 so the
    # facets are directly comparable.
    plot_well_survival_over_time(
        _fraction_alive_per_well_stage(ar),
        time_col="stage_hpf",
        value_col="frac_alive",
        value_label="fraction alive",
        title="analysis_ready — fraction alive per well over predicted_stage_hpf (by genotype)",
        output_path=Path(output_well_survival_over_stage_png),
        genotype_col=_GENOTYPE_COL,
        vmin=0.0,
        vmax=1.0,
    )
