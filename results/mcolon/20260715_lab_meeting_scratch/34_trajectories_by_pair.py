"""Trajectories by pair, colored by DERIVED phenotype.

Follow-up to the per-pair penetrance figure: pair_3 came out with an unusual phenotype split
(22 Low_to_High vs 16 High_to_Low, roughly balanced) where pair_1 and pair_2 are heavily
High_to_Low (27/9 and 42/9). This plots the actual trajectories so that split can be inspected
rather than inferred from counts.

Layout: rows = feature (curvature, length), cols = pair, colored by the derived 3-class label.
Homozygotes only by default -- the penetrant classes live there, and mixing in het/WT would
bury the phenotype contrast under non-penetrant traces.

Labels are the model's calls, carried over from script 32's scorer so they match the
penetrance figure exactly.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/34_trajectories_by_pair.py
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
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

_both = __import__("32_penetrance_dots_both_genes")
_pair = __import__("33_penetrance_dots_by_pair")
load_b9d2, load_cep290 = _both.load_b9d2, _both.load_cep290
GENES, NP_LABEL, ID_COL, TIME_COL = _both.GENES, _both.NP_LABEL, _both.ID_COL, _both.TIME_COL
score_with_pair = _pair.score_with_pair

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style, update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402

OUTPUT_DIR = RUN_DIR / "figures" / "trajectories_by_pair"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]

# Pairs below this many embryos are noise at trajectory resolution.
MIN_EMBRYOS = 8


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=290,
        width_per_col=300,
        min_width=1100,
        individual_alpha=0.30,
        individual_width=0.8,
        trend_width=3.2,
        axis_label_fontsize=11,
        legend_fontsize=10,
    )


def build(gene: str, zygosity: str = "homozygous", label_source: str = "curated") -> None:
    """label_source: 'curated' = ground-truth labels, 'derived' = the model's predictions.

    Worth generating both: pair_3 is 8/30 High/Low in the curated labels but 16/22 predicted,
    i.e. the model mis-calls ~8 Low_to_High embryos there. Comparing the two colourings shows
    exactly which trajectories it gets wrong.
    """
    df, z = (load_b9d2() if gene == "b9d2" else load_cep290())
    classes = GENES[gene]["classes"]
    colors = GENES[gene]["colors"]

    if label_source == "curated":
        emb = df.drop_duplicates(ID_COL)[[ID_COL, "zygosity", "pair", "train_label"]].copy()
        emb = emb.rename(columns={"train_label": "phenotype"})
        emb = emb[emb["phenotype"].notna()]
    else:
        emb = score_with_pair(df, z, classes)      # embryo -> predicted phenotype (+ pair)
    emb = emb[emb["zygosity"] == zygosity]
    if emb.empty:
        print(f"[{gene}] no {label_source} {zygosity} embryos — skipped")
        return

    rows = df[df[ID_COL].isin(emb[ID_COL])].copy()
    rows["phenotype"] = rows[ID_COL].map(emb.set_index(ID_COL)["phenotype"])
    rows["pair"] = rows[ID_COL].map(emb.set_index(ID_COL)["pair"])
    rows = rows.dropna(subset=["phenotype", "pair", TIME_COL, *FEATURES])

    keep = [p for p, n in emb.groupby("pair")[ID_COL].nunique().items() if n >= MIN_EMBRYOS]
    rows = rows[rows["pair"].isin(keep)]
    pairs = sorted(rows["pair"].unique())
    if not pairs:
        print(f"[{gene}] no pair with >= {MIN_EMBRYOS} {zygosity} embryos — skipped")
        return

    counts = emb[emb["pair"].isin(pairs)].groupby(["pair", "phenotype"]).size()
    print(f"\n[{gene}] {label_source} labels, {zygosity}, "
          f"{len(pairs)} pairs with >= {MIN_EMBRYOS} embryos")
    print(counts.unstack("phenotype").fillna(0).astype(int).to_string())

    present = [c for c in classes if c in set(rows["phenotype"])]
    fig = plot_feature_over_time(
        rows,
        features=FEATURES,                       # rows = curvature, length
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="phenotype",
        color_preset=ColorPreset(colors=colors, order=present),
        facet_col="pair",
        layout=FacetSpec(col_order=pairs, sharex=True, sharey=False),
        show_individual=True,
        show_trend=True,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"{gene} — {zygosity} trajectories by pair, "
               f"colored by {label_source.upper()} phenotype"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    note = "   ".join(f"{p.replace(gene + '_', '')}: "
                      + "/".join(f"{c}={int(counts.get((p, c), 0))}" for c in present)
                      for p in pairs)
    fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, color="#555555")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"trajectories_by_pair__{gene}_{zygosity}_{label_source}.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    for source in ("curated", "derived"):
        build("cep290", "homozygous", source)
        build("b9d2", "homozygous", source)


if __name__ == "__main__":
    main()
