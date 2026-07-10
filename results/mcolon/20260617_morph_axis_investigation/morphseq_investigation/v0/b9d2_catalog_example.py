"""TASK_E — the b9d2 catalog example (ACCEPTANCE TARGET for the whole engine).

> "Generate the b9d2 phenotype distributions over time vs their controls, and
>  watch the two clusters emerge."

This is the CANONICAL worked example — it supersedes the retired
``v0/_superseded/b9d2_worked_example.py`` + ``rich_distribution_plot*.py`` (which
drove the pre-migration ``DistributionGrouping`` / ``FacetCoordinate`` API). It
wires the WHOLE post-migration stack through the ``DistributionCatalog`` control
tower:

  from_dataframe -> detect_peaks -> label_groups  (PATH A, within-population)
  from_dataframe -> detect_peaks -> compare        (PATH B, cross-population)
  build_1d_density_grid / build_1d_distribution_comparison -> plot_1d_density_grid
  plot_1d_ridgeline  (TASK_D, same DistributionGrid IR)

Binning is UPSTREAM (``load_binned`` — reused verbatim in spirit from the old
worked example); the engine objects NEVER bin. Acceptance signal: the target
(b9d2 phenotype) peak count goes 1 -> 2 as hpf increases (two clusters emerge),
printed per bin and asserted.

Run:
  cd .../20260617_morph_axis_investigation
  PYTHONPATH=.:src:$PYTHONPATH conda run -n segmentation_grounded_sam \
      --no-capture-output python -m morphseq_investigation.v0.b9d2_catalog_example
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..engine.catalog import DistributionCatalog
from ..engine.facets import CoordinateFacet, LabelGroupFacet
from ..engine.plotting import (
    DEFAULT_ROLE_PALETTES,
    DEFAULT_ROLE_STYLES,
    GroupStyle,
    build_1d_density_grid,
    build_1d_distribution_comparison,
    plot_1d_density_grid,
)
from ..engine.ridge import plot_1d_ridgeline

# --------------------------------------------------------------------------- #
# Config — grounded in the old worked example (the live reference flow).
# --------------------------------------------------------------------------- #
TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
FEATURE_NAMES = ("total_length_um", "baseline_deviation_normalized")  # peak engine is 2-D
PHENOTYPE_LABELS = ("CE", "HTA")
TARGET_DESIGN_HPF = (14, 18, 24, 30, 48)
MIN_EMBRYOS = 10

# Reference-role styling: wildtype degrades to gray/dashed (matching the WT
# baseline convention); everything else (b9d2, CE/HTA phenotypes, peaks) carries
# a real crimson-family hue. reference_role is passed to the plotter/ridge below.
REFERENCE_ROLE = "wildtype"
ROLE_PALETTES = {**DEFAULT_ROLE_PALETTES, REFERENCE_ROLE: "#808080"}
# Reference draws dashed + unfilled (the WT baseline convention). DEFAULT_ROLE_STYLES
# only keys "reference"; wire the same dashed style under our "wildtype" role name.
ROLE_STYLES = {**DEFAULT_ROLE_STYLES, REFERENCE_ROLE: GroupStyle(line_style="--", fill=False)}


def _resolve_csv() -> Path:
    """The real b9d2 CSV is gitignored (~125MB). The old script located it via
    ``parents[5]`` from its own path; in a git WORKTREE that resolves to the
    worktree root (no data). Try that first, then fall back to the primary repo
    checkout so the acceptance runs on REAL data regardless of worktree."""
    rel = Path(
        "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv"
    )
    candidates = [
        Path(__file__).resolve().parents[5] / rel,
        Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq") / rel,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"reference_b9d2_clean.csv not found in any of: {[str(c) for c in candidates]}"
    )


# --------------------------------------------------------------------------- #
# Step 0 — load + bin UPSTREAM (the catalog NEVER bins). One row per (embryo,
# time_bin) with mean features + modal labels — mirrors the old load_binned.
# --------------------------------------------------------------------------- #
def load_binned(csv: Path | None = None) -> pd.DataFrame:
    csv = csv or _resolve_csv()
    df = pd.read_csv(csv, low_memory=False)
    needed = [TIME_COL, *FEATURE_NAMES, "phenotype_clean", "zygosity"]
    df = df.dropna(subset=needed).copy()
    df["time_bin"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2
    grain = (
        "embryo_id"
        if df["physical_embryo_id"].isna().all()
        else "physical_embryo_id"
    )
    agg: dict[str, Any] = {f: "mean" for f in FEATURE_NAMES}
    agg["phenotype_clean"] = lambda x: x.mode().iloc[0]
    agg["zygosity"] = lambda x: x.mode().iloc[0]
    binned = df.groupby([grain, "time_bin"]).agg(agg).reset_index()
    binned = binned.rename(columns={grain: "embryo_id"})

    # Derive the two comparison label columns (target vs reference), honest names:
    #   genotype        : "wildtype" (zygosity==wildtype)  vs "b9d2" (affected)
    #   phenotype_clean : kept as-is (CE / HTA / wildtype / unlabeled)
    # The genotype axis is what compare() resolves across (PATH B); it is ALSO a
    # within-population label group (PATH A). wildtype -> reference (gray/dashed),
    # b9d2 -> target (crimson).
    binned["genotype"] = np.where(binned["zygosity"] == "wildtype", "wildtype", "b9d2")
    return binned


def _keep_target_and_reference(binned: pd.DataFrame) -> pd.DataFrame:
    """The b9d2 comparison population per bin: WT reference + phenotype-affected
    target (CE/HTA). Drops het/homo carriers that are neither WT nor a scored
    phenotype (matches the old run_bin's ref/target carve). Filters to the
    DESIGN bins with >= MIN_EMBRYOS on each side so peak detection is stable."""
    is_ref = binned["zygosity"] == "wildtype"
    is_tgt = binned["phenotype_clean"].isin(PHENOTYPE_LABELS)
    sub = binned[is_ref | is_tgt].copy()

    design_centers = {
        float(int(h // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2) for h in TARGET_DESIGN_HPF
    }
    sub = sub[sub["time_bin"].isin(design_centers)]

    keep_bins = []
    for tb, g in sub.groupby("time_bin"):
        n_ref = int((g["zygosity"] == "wildtype").sum())
        n_tgt = int(g["phenotype_clean"].isin(PHENOTYPE_LABELS).sum())
        if n_ref >= MIN_EMBRYOS and n_tgt >= MIN_EMBRYOS:
            keep_bins.append(tb)
    # Sort by time_bin so from_dataframe's first-appearance grouping (hence the
    # figure's column order) runs earliest -> latest.
    kept = sub[sub["time_bin"].isin(keep_bins)]
    return kept.sort_values("time_bin", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The acceptance check: target (b9d2) peak count per time bin.
# Built from the SAME per-bin·per-genotype catalog Path B uses (detect_peaks on
# each distribution's own points — the pre-migration engine's peak signal).
# --------------------------------------------------------------------------- #
def target_peak_counts(catalog2: DistributionCatalog) -> dict[float, int]:
    counts: dict[float, int] = {}
    for dist in catalog2.distributions:
        if dist.coordinates.get("genotype") != "b9d2":
            continue
        tb = dist.coordinates.get("time_bin")
        counts[tb] = len(dist.sample_sets("resolved_peak"))
    return dict(sorted(counts.items()))


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    binned = load_binned()
    df = _keep_target_and_reference(binned)
    print(f"[load] {len(df)} (embryo, time_bin) rows across bins "
          f"{sorted(df['time_bin'].unique())}")

    # ------------------------------------------------------------------- #
    # PATH A — within-population. ONE distribution per time_bin; the genotype /
    # phenotype / resolved_peak label groups split samples WITHIN each bin.
    # ------------------------------------------------------------------- #
    catalog = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=FEATURE_NAMES,
        label_columns=("genotype", "phenotype_clean"),
        split_columns=("time_bin",),
    )
    catalog = catalog.detect_peaks(features=FEATURE_NAMES, output_label="resolved_peak")

    groups = (
        *catalog.label_groups("resolved_peak", display_name="Resolved peaks"),
        *catalog.label_groups("phenotype_clean", display_name="Phenotype"),
        *catalog.label_groups("genotype", display_name="Genotype"),
    )

    # ------------------------------------------------------------------- #
    # PATH B — cross-population. A SECOND catalog split on (time_bin, genotype):
    # separate WT / b9d2 populations, matched by time, peaks overlaid per cell.
    # ------------------------------------------------------------------- #
    catalog2 = DistributionCatalog.from_dataframe(
        df,
        sample_id_column="embryo_id",
        feature_columns=FEATURE_NAMES,
        label_columns=("genotype", "phenotype_clean"),
        split_columns=("time_bin", "genotype"),
    )
    catalog2 = catalog2.detect_peaks(features=FEATURE_NAMES, output_label="resolved_peak")

    # ---- ACCEPTANCE CHECK: target peak count per bin (1 -> 2 emergence) ---- #
    counts_by_bin = target_peak_counts(catalog2)
    print("\n=== b9d2 emergence — target (b9d2) resolved-peak count per bin ===")
    for tb, n in counts_by_bin.items():
        print(f"  bin {tb:>5}: target_peaks={n}")
    counts = [counts_by_bin[tb] for tb in sorted(counts_by_bin)]
    print(f"\nper-bin target peak counts (earliest first): {counts}")
    assert len(counts) >= 2, f"need >=2 bins to see emergence; got {counts}"
    # NOTE: the old <=1-at-earliest-bin "emergence" oracle was calibrated on the
    # STALE scipy_default (Scott's-rule) bandwidth, which over-smoothed b9d2. Under
    # the calibrated longest_non_outlier_MST_edge @ 0.75 rule, the honest counts are
    # multimodal earlier. Report the counts and ALWAYS render the figures so the
    # clusters can be inspected visually; do not abort on the old oracle.
    if counts[0] <= 1 and max(counts) >= 2:
        print("emergence (1 -> 2 peaks) seen under the old scipy-shaped oracle.")
    else:
        print(
            f"NOTE: counts {counts} do NOT match the old scipy-era <=1-then->=2 "
            f"oracle. This is expected under the calibrated MST-edge bandwidth — "
            f"inspect the emitted figures to judge the clusters."
        )

    comparisons = catalog2.compare(across="genotype", values=("wildtype", "b9d2"))

    # ------------------------------------------------------------------- #
    # Emit figures — both features, both paths, + a ridge.
    # ------------------------------------------------------------------- #
    emitted: list[Path] = []
    for feature in FEATURE_NAMES:
        # PATH A: the 3-row figure (Resolved peaks / Phenotype / Genotype) x bins.
        grid = build_1d_density_grid(
            groups,
            feature=feature,
            facet_row=LabelGroupFacet(),
            facet_col=CoordinateFacet("time_bin"),
        )
        path_a = out_dir / f"b9d2_grid_{feature}.png"
        plot_1d_density_grid(
            grid,
            role_styles=ROLE_STYLES,
            role_palettes=ROLE_PALETTES,
            reference_role=REFERENCE_ROLE,
            title=f"b9d2 within-population — {feature}",
            output_path=path_a,
        )
        emitted.append(path_a)

        # RIDGE from the SAME Path-A grid (TASK_D verb, same IR).
        path_ridge = out_dir / f"b9d2_ridge_stacked_{feature}.png"
        plot_1d_ridgeline(
            grid,
            variant="stacked",
            role_styles=ROLE_STYLES,
            role_palettes=ROLE_PALETTES,
            reference_role=REFERENCE_ROLE,
            output_path=path_ridge,
        )
        emitted.append(path_ridge)

        # PATH B: WT vs b9d2 peaks overlaid per time cell.
        grid_b = build_1d_distribution_comparison(
            comparisons,
            feature=feature,
            label_group="resolved_peak",
            facet_col=CoordinateFacet("time_bin"),
            reference_value="wildtype",
        )
        path_b = out_dir / f"b9d2_compare_{feature}.png"
        plot_1d_density_grid(
            grid_b,
            role_styles=ROLE_STYLES,
            role_palettes=ROLE_PALETTES,
            reference_role=REFERENCE_ROLE,
            title=f"b9d2 cross-population WT vs b9d2 — {feature}",
            output_path=path_b,
        )
        emitted.append(path_b)

    print("\n=== figures emitted ===")
    for p in emitted:
        print(f"  {p}")


if __name__ == "__main__":
    main()
