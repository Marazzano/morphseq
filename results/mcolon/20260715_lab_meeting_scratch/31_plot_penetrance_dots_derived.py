"""Plot C (per-embryo dots) rebuilt on the DERIVED 3-class labels.

The original composite (script 21) used the old transferred phenotype labels, whose
"Not Penetrant" class was wildtype-relabeled and therefore polluted. This version uses the
predicted class from the 3-class simplex model instead:

    CE / HTA / Not Penetrant       (argmax of the out-of-fold probabilities)

so the penetrance rate printed on each rule is the classifier's own call. That is a MODEL
OUTPUT, not a measured biological penetrance -- it is labelled "% pen." and should be read
that way.

Layout per the plot brief: skinny columns, bolder dots with alpha, and the rate printed
directly on top of the existing penetrance rule (no new divider line).

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/31_plot_penetrance_dots_derived.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
SCORES = RUN_DIR / "figures/three_class_simplex/three_class_scores.csv"
OUTPUT_DIR = RUN_DIR / "figures" / "penetrance_composite"

ID_COL = "embryo_id"

# The scores CSV was written before the class rename; map the stored argmax onto the
# current phenotype-class vocabulary.
PRED_RENAME = {
    "CE": "CE",
    "homozygous NON-CE": "HTA",
    "HTA": "HTA",
    "wildtype": "Not Penetrant",
    "Not Penetrant": "Not Penetrant",
}
NP_LABEL = "Not Penetrant"
CLASS_ORDER = ["CE", "HTA", NP_LABEL]
CLASS_COLORS = {"CE": "#1B9E77", "HTA": "#D95F02", NP_LABEL: "#9AA0A6"}

ZYG_FROM_GENOTYPE = {
    "b9d2_wildtype": "wildtype",
    "b9d2_heterozygous": "heterozygous",
    "b9d2_homozygous": "homozygous",
    "b9d2_unknown": "unknown",
}
ZYGOSITY_ORDER = ["wildtype", "heterozygous", "homozygous", "unknown"]
ZYG_LABEL = {"wildtype": "WT", "heterozygous": "het",
             "homozygous": "homo", "unknown": "unk"}


def load() -> pd.DataFrame:
    df = pd.read_csv(SCORES)
    df["phenotype"] = df["pred"].map(PRED_RENAME)
    df["zygosity"] = df["genotype"].map(ZYG_FROM_GENOTYPE)
    return df.dropna(subset=["phenotype", "zygosity"])


def composition(df: pd.DataFrame):
    counts = (df.groupby(["zygosity", "phenotype"]).size()
                .unstack("phenotype")
                .reindex(index=ZYGOSITY_ORDER, columns=CLASS_ORDER)
                .fillna(0).astype(int))
    totals = counts.sum(axis=1)
    return counts, totals


def main(seed: int = 0) -> None:
    df = load()
    counts, totals = composition(df)
    print("derived class x zygosity (embryos):")
    print(counts.assign(total=totals).to_string())

    penetrant = counts[[c for c in CLASS_ORDER if c != NP_LABEL]].sum(axis=1)
    pen_pct = penetrant.div(totals.replace(0, np.nan)).fillna(0.0) * 100
    print("\npredicted penetrant %:", pen_pct.round(1).to_dict())

    rng = np.random.default_rng(seed)
    # Skinny: one narrow column per zygosity keeps the rule the dominant read.
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    for xi, zyg in enumerate(ZYGOSITY_ORDER):
        col = df[df["zygosity"] == zyg]
        total = max(int(totals[zyg]), 1)
        y0 = 0.0
        # NP at the bottom as the baseline band, penetrant classes stacked above it.
        for pheno in reversed(CLASS_ORDER):
            n = int((col["phenotype"] == pheno).sum())
            if n == 0:
                continue
            frac = n / total
            ys = y0 + rng.uniform(0.02, 0.98, size=n) * frac
            xs = xi + rng.uniform(-0.11, 0.11, size=n)
            ax.scatter(xs, ys, s=34, color=CLASS_COLORS[pheno], alpha=0.72,
                       edgecolors="white", linewidths=0.4, zorder=3,
                       label=pheno if xi == 0 else None)
            y0 += frac

        pen = pen_pct[zyg] / 100.0
        ax.plot([xi - 0.19, xi + 0.19], [1 - pen, 1 - pen],
                color="#222222", linewidth=2.0, zorder=5)
        # Rate printed directly ON TOP of the rule.
        ax.annotate(f"{pen_pct[zyg]:.0f}% pen.", (xi, 1 - pen),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#222222", zorder=6)

    ax.set_xticks(range(len(ZYGOSITY_ORDER)))
    ax.set_xticklabels([f"{ZYG_LABEL[z]}\nn={int(totals[z])}" for z in ZYGOSITY_ORDER],
                       fontsize=10)
    ax.set_xlim(-0.45, len(ZYGOSITY_ORDER) - 0.55)
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("% of embryos within zygosity", fontsize=11)
    ax.set_title("b9d2 — derived 3-class labels\n(rule = predicted penetrant fraction)",
                 fontsize=11, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor=CLASS_COLORS[c], markeredgecolor="white", label=c)
               for c in CLASS_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.11), ncol=len(CLASS_ORDER))
    ax.grid(True, axis="y", color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "C_embryo_dots_derived.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out.relative_to(RUN_DIR)}")


if __name__ == "__main__":
    main()
