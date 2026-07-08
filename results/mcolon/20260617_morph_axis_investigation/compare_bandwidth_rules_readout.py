"""Bandwidth-rule bake-off for the resolved-peak reference readout.

Runs the SAME target-vs-WT resolved-peak permutation comparison on the real
cep290 and b9d2 data under three bandwidth rules, side by side per stage:

    scipy_default                     (Scott's rule; the V0 placeholder)
    median_kNN_distance   x1.0        (conservative geometry rule)
    longest_non_outlier_MST_edge x0.75 (the "best" geometry rule from V0)

For each gene it writes:
  - a CSV (one row per stage x rule x metric) with observed_difference / p / n_peaks
  - a compact figure: rows = stages, columns = rules, each cell the arrow readout,
    so you can eyeball where the rules agree and where scipy_default over/under-smooths.

Reuses load_bins + normalize_shape + build_distribution_overlay from
valley_visualization.py so the points and canonical grid match that figure exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from valley_visualization import GENES, load_bins, TARGET_DESIGN_HPF, GRID  # noqa: E402
from support_geometry import normalize_shape  # noqa: E402
from morphseq_investigation.plotting.modal_distribution_plotting import derive_shared_grid  # noqa: E402
from resolved_peak_reference_readout import (  # noqa: E402
    READOUT_METRICS, compute_reference_readout, render_readout_cell, _ARROW,
    _SIG_COLOR, _NS_COLOR, _INVALID_COLOR,
)

PLOT_DIR = RUN_DIR / "plots"
TABLE_DIR = RUN_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)

RULES = [
    ("scipy_default", 1.0, "scipy_default\n(Scott)"),
    ("median_kNN_distance", 1.0, "median_kNN\nx1.0"),
    ("longest_non_outlier_MST_edge", 0.75, "MST_edge\nx0.75"),
]
N_DRAWS = 200
LABEL_FS = 11


def _spec_for(rule, mult):
    from morphseq_investigation.core.resolved_peak_analysis import ResolvedPeakAnalysisSpec
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule=rule, bandwidth_multiplier=mult,
        peak_detector_method="kde_peak_basins_sample_support", min_sample_fraction=0.10,
    )


def run_gene(gene, cfg):
    bins = load_bins(cfg)
    hpfs = [h for h in TARGET_DESIGN_HPF if h in bins]
    rows = []
    # cells[hpf][rule_label] -> metric->MetricCell
    cells_by_stage_rule = {}

    for hpf in hpfs:
        grp_raw, phenos, wt_raw = bins[hpf]
        grp = normalize_shape(grp_raw)
        wt = normalize_shape(wt_raw)
        canonical_grid = derive_shared_grid(grp, wt, grid=GRID, kde=None)

        cells_by_stage_rule[hpf] = {}
        for rule, mult, label in RULES:
            cells = compute_reference_readout(
                target_points=grp, wt_points=wt, canonical_grid=canonical_grid,
                n_draws=N_DRAWS, seed=42, stage_id=f"{hpf}hpf", gene=gene,
                analysis_spec=_spec_for(rule, mult),
            )
            cells_by_stage_rule[hpf][label] = cells
            for spec in READOUT_METRICS:
                c = cells[spec.metric]
                rows.append({
                    "gene": gene, "hpf": hpf, "bandwidth_rule": rule,
                    "bandwidth_multiplier": mult, "metric": spec.metric,
                    "observed_difference": c.observed_difference, "p_value": c.p_value,
                    "significant": c.significant, "direction": c.direction,
                    "meaning": c.meaning, "valid": c.valid,
                })

    df = pd.DataFrame(rows)
    csv = TABLE_DIR / f"{gene}_bandwidth_rule_readout_comparison.csv"
    df.to_csv(csv, index=False)
    print(f"Saved: {csv.name}")
    _render(gene, hpfs, cells_by_stage_rule)
    return df


def _render(gene, hpfs, cells_by_stage_rule):
    n_rows, n_cols = len(hpfs), len(RULES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 1.9 * n_rows), squeeze=False)
    fig.subplots_adjust(bottom=0.06, top=0.90, left=0.10, right=0.99, hspace=0.28, wspace=0.06)

    for r, hpf in enumerate(hpfs):
        for c, (rule, mult, label) in enumerate(RULES):
            ax = axes[r][c]
            render_readout_cell(ax, cells_by_stage_rule[hpf][label], label_fs=LABEL_FS)
            if r == 0:
                ax.set_title(label, fontsize=LABEL_FS, fontweight="bold")
        axes[r][0].text(-0.06, 0.5, f"{hpf} hpf", transform=axes[r][0].transAxes,
                        fontsize=LABEL_FS, fontweight="bold", rotation=90, va="center", ha="right")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=r"$\uparrow$", color=_SIG_COLOR, lw=0, markersize=12, label="significant (p<0.05)"),
        Line2D([0], [0], marker=r"$\uparrow$", color=_NS_COLOR, lw=0, markersize=12, label="not significant"),
        Line2D([0], [0], marker=r"$-$", color=_INVALID_COLOR, lw=0, markersize=12, label="undefined (too few modes)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=LABEL_FS - 1, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    metric_order = " · ".join(f"{i+1}.{s.label}" for i, s in enumerate(READOUT_METRICS))
    fig.suptitle(
        f"{gene} — resolved-peak reference readout: bandwidth-rule bake-off  (target vs WT, {N_DRAWS} draws)\n"
        f"each cell top→bottom: {metric_order}   |   ↑/↓ = direction vs WT, bold+red = significant",
        fontsize=LABEL_FS, fontweight="bold", y=0.995)
    out = PLOT_DIR / f"{gene}_bandwidth_rule_readout_comparison.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.name}")


def main():
    all_df = []
    for gene, cfg in GENES.items():
        print(f"\n=== {gene} ===")
        all_df.append(run_gene(gene, cfg))
    combined = pd.concat(all_df, ignore_index=True)
    # quick agreement summary: per gene x stage x metric, do the 3 rules agree on direction?
    print("\n=== rule-agreement on direction (per gene/stage/metric) ===")
    g = combined.groupby(["gene", "hpf", "metric"])["direction"].nunique()
    print(f"cells where all 3 rules agree on direction: {(g == 1).sum()} / {len(g)}")
    print("Done.")


if __name__ == "__main__":
    main()
