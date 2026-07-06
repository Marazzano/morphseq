"""analysis_ready report (TERMINAL) — the only report whose input surface is the whole joined DAG.

Report artifacts over the predicted_stage_hpf axis (analysis_ready is the one step that HAS stage +
genotype + embeddings together, so it is no longer constrained to time_index):

  1. latent_pca_*           — PCA views of the z_mu_b_* (biological, else flat z_mu_*) block:
                              all snips by QC state, passing-QC snips by predicted_stage_hpf, and
                              passing-QC snips by genotype. Axis labels include explained variance.
  2. post_qc_*_gallery      — processed-snip galleries over passed-QC rows for the established
                              feature axes already used by feature reports: area_um2 and
                              baseline_deviation_normalized.
  3. survival_over_stage    — recapitulates death_detection's alive_embryos_experiment survival
                              curve, but on the stage axis: total alive physical embryos vs stage.
  4. genotype_survival_panel— the same survival curve, overlaid across genotypes (header) then one
                              small-multiple per genotype (3-col gallery; 1 col if a single genotype).
  5. well_survival_*        — embryo-count survival heatmaps over stage, split into separate
                              all-genotype and per-genotype artifacts.

Wiring only: derives report-only values (survival counts, latent selection) and hands columns to the
shared renderers in viz/reporting.py. Imports its own contract + viz/reporting only — never matplotlib
directly, and nothing imports this module (terminal leaf).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.viz.reporting import (
    plot_latent_pca_categorical,
    plot_latent_pca_continuous,
    plot_latent_pca_qc_state,
    plot_grouped_survival_panel,
    plot_line,
    plot_well_survival_over_time,
    render_value_quartile_gallery,
)
from data_pipeline.object_extraction.snip_processing.io import resolve_snip_inventory_image_paths

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
_QC_PASS_COL = "use_snip"
_PERSISTENCE_FLAG = "persistence_dead_flag"
# Stage bin width (hpf) for the per-well × stage survival heatmap: predicted_stage_hpf is continuous,
# so we bin it into a stable, legible column axis (mirrors the death_detection time_index heatmap).
_STAGE_BIN_HPF = 2.0


def _alive_embryos_per_well_stage(analysis_ready: pd.DataFrame) -> pd.DataFrame:
    """Per (well_id, stage_bin[, genotype]): alive physical-embryo COUNT over predicted_stage_hpf.

    "Alive" reuses the persistence flag already folded into the snip_qc verdict: an embryo is alive
    in a stage bin if at least one of its snips in that bin is not flagged by ``persistence_dead_flag``.
    This avoids using raw snip rows as the first/lead-column number when wells contain multiple
    physical embryos. Stage is binned to _STAGE_BIN_HPF for a stable axis.
    Returns long rows with the bin CENTER as ``stage_hpf`` for the shared heatmap helper.
    """
    df = analysis_ready.dropna(subset=[_STAGE_COL]).copy()
    reasons = df[_QC_REASONS_COL].fillna("").astype(str)
    df["alive"] = ~reasons.str.contains(_PERSISTENCE_FLAG, regex=False)
    # Bin stage to a fixed grid; report the bin CENTER so the axis reads in hpf, not bin index.
    df["stage_hpf"] = (np.floor(df[_STAGE_COL] / _STAGE_BIN_HPF) * _STAGE_BIN_HPF) + _STAGE_BIN_HPF / 2.0
    embryo_keys = [_WELL_COL, "stage_hpf", _ANIMAL_COL]
    if _GENOTYPE_COL in df.columns:
        embryo_keys.append(_GENOTYPE_COL)
    per_embryo = df.groupby(embryo_keys, dropna=False)["alive"].any().reset_index()
    count_keys = [_WELL_COL, "stage_hpf"]
    if _GENOTYPE_COL in per_embryo.columns:
        count_keys.append(_GENOTYPE_COL)
    return per_embryo.groupby(count_keys, dropna=False)["alive"].sum().reset_index(name="alive_embryos")


def _add_survival_fraction(
    well_stage: pd.DataFrame, *, group_cols: list[str]
) -> pd.DataFrame:
    """Add a 0-1 survival fraction using each row group's first observed alive count as baseline."""
    out = well_stage.sort_values(group_cols + ["stage_hpf"]).copy()
    starts = out.groupby(group_cols, dropna=False)["alive_embryos"].transform("first")
    out["survival_fraction"] = out["alive_embryos"] / starts.replace(0, np.nan)
    return out


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
    snip_inventory_csv: Path,
    output_root: Path,
    output_latent_pca_qc_state_png: Path,
    output_latent_pca_stage_png: Path,
    output_latent_pca_genotype_png: Path,
    output_post_qc_area_um2_gallery_png: Path,
    output_post_qc_baseline_deviation_gallery_png: Path,
    output_survival_over_stage_png: Path,
    output_genotype_survival_panel_png: Path,
    output_well_survival_over_stage_all_png: Path,
    output_well_survival_over_stage_by_genotype_png: Path,
) -> None:
    ar = pd.read_parquet(analysis_ready_parquet)
    death_event = pd.read_csv(death_event_csv)
    snip_inventory = pd.read_csv(snip_inventory_csv)

    # ── 1. Latent PCA views ──────────────────────────────────────────────────────────────────
    latent_cols = select_latent_columns(ar.columns)
    if not latent_cols:
        raise ValueError(
            "analysis_ready report: no z_mu_* latent columns in the analysis_ready table — cannot "
            "build PCA panels."
        )
    genotypes_all = sorted(ar[_GENOTYPE_COL].dropna().astype(str).unique())
    qc_pass = ar[_QC_PASS_COL].fillna(False).astype(bool)
    plot_latent_pca_qc_state(
        ar,
        latent_cols,
        qc_pass_col=_QC_PASS_COL,
        title="analysis_ready — latent PCA colored by QC state",
        output_path=Path(output_latent_pca_qc_state_png),
    )
    plot_latent_pca_continuous(
        ar[qc_pass],
        latent_cols,
        continuous_col=_STAGE_COL,
        title="analysis_ready — passing-QC latent PCA colored by predicted_stage_hpf",
        output_path=Path(output_latent_pca_stage_png),
    )
    plot_latent_pca_categorical(
        ar[qc_pass],
        latent_cols,
        category_col=_GENOTYPE_COL,
        title="analysis_ready — passing-QC latent PCA colored by genotype",
        output_path=Path(output_latent_pca_genotype_png),
        category_colors=_genotype_colors(genotypes_all),
    )

    # ── 2. Post-QC feature galleries ─────────────────────────────────────────────────────────
    gallery_df = ar[qc_pass].merge(
        resolve_snip_inventory_image_paths(snip_inventory, output_root=Path(output_root)), on="snip_id", how="left"
    )
    render_value_quartile_gallery(
        gallery_df,
        "area_um2",
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="analysis_ready — post-QC area_um2 value quartiles",
        output_path=Path(output_post_qc_area_um2_gallery_png),
    )
    render_value_quartile_gallery(
        gallery_df,
        "baseline_deviation_normalized",
        image_path_col="resolved_image_path",
        label_col="snip_id",
        title="analysis_ready — post-QC baseline_deviation_normalized value quartiles",
        output_path=Path(output_post_qc_baseline_deviation_gallery_png),
    )

    # ── 3. Experiment survival over stage ────────────────────────────────────────────────────
    stages, alive = _survival_over_stage(ar, death_event)
    plot_line(
        pd.Series(stages),
        pd.Series(alive),
        title="analysis_ready — total alive embryos over predicted_stage_hpf",
        output_path=Path(output_survival_over_stage_png),
        xlabel=_STAGE_COL,
        ylabel="total alive embryos",
        annotate_start_final=True,
    )

    # ── 4. Per-genotype survival panel (overlay + gallery) ───────────────────────────────────
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

    # ── 5. Per-well alive embryo counts over predicted_stage_hpf ───────────────────────────────
    # Separate all-genotype and per-genotype artifacts. Multi-embryo wells are still a mixed-well
    # review surface, so the title calls out that separate physical embryos should be inspected
    # independently when a well carries more than one embryo.
    well_stage = _alive_embryos_per_well_stage(ar)
    well_stage_all = _add_survival_fraction(
        well_stage.drop(columns=[_GENOTYPE_COL], errors="ignore")
        .groupby([_WELL_COL, "stage_hpf"], dropna=False)["alive_embryos"]
        .sum()
        .reset_index(),
        group_cols=[_WELL_COL],
    )
    well_stage_by_genotype = _add_survival_fraction(
        well_stage,
        group_cols=[_WELL_COL, _GENOTYPE_COL],
    )
    plot_well_survival_over_time(
        well_stage_all,
        time_col="stage_hpf",
        value_col="survival_fraction",
        value_label="relative alive fraction",
        title=(
            "analysis_ready — relative survival per well over predicted_stage_hpf (all genotypes)\n"
            "Note: wells with >1 physical embryo are aggregated here; inspect those embryos separately."
        ),
        output_path=Path(output_well_survival_over_stage_all_png),
        lead_value_col="alive_embryos",
        vmin=0.0,
        vmax=1.0,
    )
    plot_well_survival_over_time(
        well_stage_by_genotype,
        time_col="stage_hpf",
        value_col="survival_fraction",
        value_label="relative alive fraction",
        title=(
            "analysis_ready — relative survival per well over predicted_stage_hpf (per genotype)\n"
            "Note: wells with >1 physical embryo are aggregated here; inspect those embryos separately."
        ),
        output_path=Path(output_well_survival_over_stage_by_genotype_png),
        genotype_col=_GENOTYPE_COL,
        include_global=False,
        lead_value_col="alive_embryos",
        vmin=0.0,
        vmax=1.0,
    )
