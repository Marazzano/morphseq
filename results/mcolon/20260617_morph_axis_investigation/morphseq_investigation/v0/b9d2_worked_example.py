"""TASK_E - the b9d2 worked example (ACCEPTANCE TARGET for the whole engine).

> "Generate the b9d2 phenotype distributions over time vs their controls, and
>  watch the two clusters emerge."

This script PROVES the engine composes: it runs the ontology construction walk
(PRIMITIVE_ONTOLOGY.md Steps 0-5) end-to-end on real b9d2 data, using the frozen
objects (TASK_0) + build_grid/evaluate_density (TASK_A) + labelers (TASK_B) +
compare_label_groups (TASK_C) + faceting-engine plotting (TASK_D).

Composition, not pipeline: per the repo owner, wiring this into Snakemake is a
separate pipeline-design task and explicitly OUT OF SCOPE. We source the current
hand-cleaned CSV with light column munging (allowed for this build) and focus on
composing the objects cleanly.

Acceptance signal: the target (phenotype) peak count goes 1 -> 2 as hpf increases
(two clusters emerge), visible in the emitted stacked KDE strip panel and printed
per bin. No `if genotype ... else if peak ...` anywhere - the same skeleton runs.

Run:
  cd .../20260617_morph_axis_investigation
  PYTHONPATH=.:src:$PYTHONPATH conda run -n segmentation_grounded_sam \
      --no-capture-output python -m morphseq_investigation.v0.b9d2_worked_example
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..engine.compare import compare_label_groups
from ..engine.grid import build_grid, evaluate_density
from ..engine.identifiers import make_distribution_id
from ..engine.labelers import label_genotype, label_peak_finding
from ..engine.objects import Distribution
from ..engine.plotting import strip_grid_figure

# --------------------------------------------------------------------------- #
# Config - grounded in valley_visualization.py (the live reference flow).
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CSV = (
    _PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv"
)

SCOPE_ID = "b9d2"
TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
FEATURE_NAMES = ("total_length_um", "baseline_deviation_normalized")  # 2-D (peak engine is 2-D)
PHENOTYPE_LABELS = ("CE", "HTA")
TARGET_DESIGN_HPF = (14, 18, 24, 30, 48)
MIN_EMBRYOS = 10
GRID_RESOLUTION = 40  # matches valley_visualization GRID


def _bin_center(hpf: float) -> float:
    return float(int(hpf // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2)


# --------------------------------------------------------------------------- #
# Step 0 - load + bin UPSTREAM (the engine objects NEVER bin). Embryo grain.
# --------------------------------------------------------------------------- #
def load_binned(csv: Path = _CSV) -> pd.DataFrame:
    """One row per (embryo, time_bin) with mean features + modal labels.

    Mirrors valley_visualization.load_bins' aggregation. Binning is upstream;
    the Distribution objects only ever receive a clean per-bin bag.
    """
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
    binned = binned.rename(columns={grain: "sample_id"})
    return binned


# --------------------------------------------------------------------------- #
# Step 1 - CARVE by label column -> one Distribution per (scope x bin x role).
#          Caller builds a SHARED feature representation BEFORE the split
#          (same feature_names on target & reference).
# --------------------------------------------------------------------------- #
def _carve_distribution(rows: pd.DataFrame, time_bin: float, role: str) -> Distribution:
    sample_ids = tuple(str(s) for s in rows["sample_id"].tolist())
    values = rows.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    return Distribution(
        distribution_id=make_distribution_id(SCOPE_ID, time_bin, role),
        scope_id=SCOPE_ID,
        time_bin=time_bin,
        role=role,
        sample_ids=sample_ids,
        feature_names=FEATURE_NAMES,
        feature_values=values,
        scope_tag=f"b9d2 {role} @ bin {time_bin}",
    )


# --------------------------------------------------------------------------- #
# The per-bin composition - Steps 1-4 of the construction walk.
# --------------------------------------------------------------------------- #
def run_bin(binned: pd.DataFrame, design_hpf: int) -> dict[str, Any] | None:
    """Carve -> label -> peak-on-pooled-grid -> compare, for ONE time bin.

    Returns a per-bin result dict, or None if the bin lacks enough embryos.
    """
    bin_center = _bin_center(design_hpf)
    sub = binned[binned["time_bin"] == bin_center]
    ref_rows = sub[sub["zygosity"] == "wildtype"]
    tgt_rows = sub[sub["phenotype_clean"].isin(PHENOTYPE_LABELS)]
    if len(ref_rows) < MIN_EMBRYOS or len(tgt_rows) < MIN_EMBRYOS:
        return None

    # Step 1 - carve two distributions on a shared feature representation.
    reference = _carve_distribution(ref_rows, bin_center, "reference")
    target = _carve_distribution(tgt_rows, bin_center, "target")

    # Step 2 - PROVIDED labeler on target: phenotype (CE / HTA) column.
    pheno_labels = [str(v) for v in tgt_rows["phenotype_clean"].tolist()]
    phenotype_lg, phenotype_sets = label_genotype(
        target,
        method="column",
        params={
            "labels": pheno_labels,
            "column": "phenotype_clean",
            # Name it "phenotype" so it is a distinct label group from "genotype"
            # (the label_group_name is the row identity in the distribution grid).
            "label_group_name": "phenotype",
        },
    )

    # Step 3 - DERIVED labeler on target AND reference, on ONE pooled grid.
    #          Pool target+reference features -> one grid -> same grid_id ->
    #          raster-comparable (ontology section 1b, construction walk Step 3).
    pooled_values = np.vstack([target.feature_values, reference.feature_values])
    pooled_fit_ids = list(target.sample_ids) + list(reference.sample_ids)
    pooled_grid = build_grid(
        feature_names=FEATURE_NAMES,
        pooled_values=pooled_values,
        fit_sample_ids=pooled_fit_ids,
        method="pooled_min_max",
        params={"resolution": GRID_RESOLUTION},
    )
    peak_params: Mapping[str, Any] = {"grid": pooled_grid}
    target_peak_lg, target_peak_sets = label_peak_finding(target, params=peak_params)
    reference_peak_lg, reference_peak_sets = label_peak_finding(reference, params=peak_params)

    # Both peak runs must reference the SAME grid_id (raster-comparability proof).
    assert target_peak_lg.artifacts is not None and reference_peak_lg.artifacts is not None
    assert target_peak_lg.artifacts.grid_id == reference_peak_lg.artifacts.grid_id == pooled_grid.grid_id

    # Step 4 - COMPARE the two peak LabelGroups (keeps provenance/artifacts alive).
    comparison = compare_label_groups(
        reference_peak_lg,
        target_peak_lg,
        "closest_center",
        reference_sample_sets={s.sample_set_id: s for s in reference_peak_sets},
        target_sample_sets={s.sample_set_id: s for s in target_peak_sets},
    )

    return {
        "design_hpf": design_hpf,
        "bin_center": bin_center,
        "n_target": len(target.sample_ids),
        "n_reference": len(reference.sample_ids),
        "grid_id": pooled_grid.grid_id,
        "pooled_grid": pooled_grid,
        "target_distribution": target,
        "reference_distribution": reference,
        "target_sample_ids": list(target.sample_ids),
        "reference_sample_ids": list(reference.sample_ids),
        "target_feature_values": target.feature_values,       # (n_target, n_feature)
        "reference_feature_values": reference.feature_values,  # (n_ref, n_feature)
        "target_peak_count": len(target_peak_lg.sample_set_ids),      # the emergence signal
        "reference_peak_count": len(reference_peak_lg.sample_set_ids),
        "target_peak_lg": target_peak_lg,
        "target_peak_sets": target_peak_sets,
        "reference_peak_lg": reference_peak_lg,
        "reference_peak_sets": reference_peak_sets,
        "phenotype_lg": phenotype_lg,
        "phenotype_sets": phenotype_sets,
        "comparison": comparison,
    }


# --------------------------------------------------------------------------- #
# Step 5 - LOOP over bins + emit the stacked KDE strip emergence panel.
# --------------------------------------------------------------------------- #
def build_emergence_strip_panel(results: list[dict[str, Any]]):
    """One 1-D KDE strip per time bin (rows), target phenotype density on the
    shared per-feature axis - a stacked strip panel to watch the clusters emerge.

    Uses feature 0 (total_length_um) as the strip axis: build a 1-D grid over
    that feature, evaluate the target's density on it per bin, and stack bins
    as rows via the faceting engine (TASK_D strip_grid_figure). Pure IR.
    """
    rows: list[dict[str, object]] = []
    feature_idx = 0
    fname = FEATURE_NAMES[feature_idx]
    for res in results:
        # 1-D grid over one feature. Bounds come from the pooled 2-D grid's axis
        # for this feature (so every bin shares the same feature-unit extent);
        # the fit set is the target samples whose density we evaluate.
        pooled_grid = res["pooled_grid"]
        col = np.asarray(pooled_grid.axis_values[feature_idx], dtype=float)
        target_vals = res["target_feature_col"]
        one_d_grid = build_grid(
            feature_names=(fname,),
            pooled_values=target_vals.reshape(-1, 1),
            fit_sample_ids=res["target_sample_ids"],
            method="fixed_bounds",
            params={
                "resolution": len(col),
                "bounds": [(float(col.min()), float(col.max()))],
            },
        )
        # Silverman's rule of thumb for a 1-D isotropic Gaussian KDE bandwidth.
        n = max(len(target_vals), 2)
        spread = float(np.std(target_vals)) or 1.0
        bandwidth = 1.06 * spread * n ** (-1.0 / 5.0)
        density = evaluate_density(one_d_grid, target_vals.reshape(-1, 1), bandwidth_spec=bandwidth)
        rows.append(
            {
                "grid": one_d_grid,
                "density_grids": [density],
                "labels": [f"{res['design_hpf']}hpf target (peaks={res['target_peak_count']})"],
                "title": f"{res['design_hpf']} hpf",
            }
        )
    return strip_grid_figure(rows, title=f"b9d2 {fname} KDE strip panel - phenotype emergence")


def main() -> None:
    binned = load_binned()
    results: list[dict[str, Any]] = []
    for design_hpf in TARGET_DESIGN_HPF:
        res = run_bin(binned, design_hpf)
        if res is None:
            print(f"[skip] {design_hpf} hpf - too few embryos")
            continue
        # Stash the target feature-0 column for the KDE strip panel.
        bin_center = res["bin_center"]
        sub = binned[binned["time_bin"] == bin_center]
        tgt_rows = sub[sub["phenotype_clean"].isin(PHENOTYPE_LABELS)]
        res["target_feature_col"] = tgt_rows[FEATURE_NAMES[0]].to_numpy(dtype=float)
        results.append(res)

    print("\n=== b9d2 emergence - target (phenotype) peak count per bin ===")
    for res in results:
        print(
            f"  {res['design_hpf']:>2} hpf: "
            f"target_peaks={res['target_peak_count']}  "
            f"reference_peaks={res['reference_peak_count']}  "
            f"(n_target={res['n_target']}, n_ref={res['n_reference']})"
        )
    counts = [r["target_peak_count"] for r in results]
    emerged = len(counts) >= 2 and counts[0] <= 1 and max(counts) >= 2
    print(f"\nEmergence (target peak count grows 1 -> 2 across bins): "
          f"{'YES' if emerged else 'NOT SEEN'}  {counts}")

    fig = build_emergence_strip_panel(results)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "b9d2_emergence_kde_strip_panel.png"
    try:
        from analyze.viz.plotting.faceting_engine import render, FacetSpec

        render(
            fig,
            backend="matplotlib",
            facet=FacetSpec(wrap=1, sharex=True, sharey=False),
            output_path=out_path,
        )
        print(f"\nKDE strip panel written: {out_path}")
    except Exception as exc:  # rendering is a smoke concern; composition is proven above
        print(f"\n[warn] figure render skipped ({type(exc).__name__}: {exc}); "
              f"IR built with {len(fig.subplots)} subplots")


if __name__ == "__main__":
    main()
