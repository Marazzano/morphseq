"""Derived-label curvature grid: rows = phenotype, cols = pair.

Splitting phenotype onto the row axis isolates each class, so the question "does this pair's
High_to_Low actually fall?" is answerable per cell instead of being read out of overlapping
medians. Colored by pair so each cell keeps its identity when scanned across a row.

Uses the DERIVED (model-predicted) labels. The comparison that matters: pair_3 is 16/22
High/Low predicted vs 8/30 curated, so its High_to_Low row is where the ~8 mis-called
Low_to_High embryos should be visible as traces that rise instead of fall.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/35_derived_phenotype_by_pair_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

_both = __import__("32_penetrance_dots_both_genes")
_pair = __import__("33_penetrance_dots_by_pair")
load_b9d2, load_cep290 = _both.load_b9d2, _both.load_cep290
GENES, ID_COL, TIME_COL = _both.GENES, _both.ID_COL, _both.TIME_COL
score_with_pair = _pair.score_with_pair

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style, update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402

OUTPUT_DIR = RUN_DIR / "figures" / "trajectories_by_pair"
FEATURE = "baseline_deviation_normalized"
MIN_EMBRYOS = 8

PAIR_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2",
               "#EECA3B", "#9D755D"]


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=250,
        width_per_col=270,
        min_width=1100,
        individual_alpha=0.30,
        individual_width=0.8,
        trend_width=3.2,
        axis_label_fontsize=11,
        legend_fontsize=9,
    )


def build(gene: str, zygosity: str = "homozygous") -> None:
    df, z = (load_b9d2() if gene == "b9d2" else load_cep290())
    classes = GENES[gene]["classes"]

    emb = score_with_pair(df, z, classes)
    emb = emb[emb["zygosity"] == zygosity]
    if emb.empty:
        print(f"[{gene}] no {zygosity} embryos — skipped")
        return

    rows = df[df[ID_COL].isin(emb[ID_COL])].copy()
    rows["phenotype"] = rows[ID_COL].map(emb.set_index(ID_COL)["phenotype"])
    rows["pair"] = rows[ID_COL].map(emb.set_index(ID_COL)["pair"])
    rows = rows.dropna(subset=["phenotype", "pair", TIME_COL, FEATURE])

    keep = [p for p, n in emb.groupby("pair")[ID_COL].nunique().items() if n >= MIN_EMBRYOS]
    rows = rows[rows["pair"].isin(keep)]
    pairs = sorted(rows["pair"].unique())
    if not pairs:
        print(f"[{gene}] no pair with >= {MIN_EMBRYOS} embryos — skipped")
        return

    present = [c for c in classes if c in set(rows["phenotype"])]
    counts = emb[emb["pair"].isin(pairs)].groupby(["phenotype", "pair"]).size()
    print(f"\n[{gene}] derived labels, {zygosity} — rows=phenotype, cols=pair")
    print(counts.unstack("pair").fillna(0).astype(int).to_string())

    colors = {p: PAIR_COLORS[i % len(PAIR_COLORS)] for i, p in enumerate(pairs)}
    fig = plot_feature_over_time(
        rows,
        features=FEATURE,                 # scalar -> facet_row honored
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="pair",
        color_preset=ColorPreset(colors=colors, order=pairs),
        facet_row="phenotype",
        facet_col="pair",
        layout=FacetSpec(row_order=present, col_order=pairs, sharex=True, sharey=True),
        show_individual=True,
        show_trend=True,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"{gene} — {zygosity} curvature, DERIVED phenotype (rows) x pair (cols)"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    note = "   ".join(
        f"{p.replace(gene + '_', '')}: "
        + "/".join(f"{c}={int(counts.get((c, p), 0))}" for c in present)
        for p in pairs
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, color="#555555")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"derived_phenotype_x_pair__{gene}_{zygosity}_curvature.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    build("cep290", "homozygous")
    build("b9d2", "homozygous")


if __name__ == "__main__":
    main()
