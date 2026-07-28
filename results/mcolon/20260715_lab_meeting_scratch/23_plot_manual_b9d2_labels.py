"""Inspect the MANUALLY CURATED b9d2 phenotype labels before trusting/training on them.

Source: results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv
        (cluster_categories: CE / HTA / BA_rescue / wildtype / unlabeled)

Why this matters: the shipped b9d2 model defines "Not Penetrant" as WILDTYPE-relabeled, which
pollutes NP with phenotype-carrying hets. These manual labels are independent -- called by eye,
not by model -- and crucially they label CE in HETEROZYGOTES (15 of them), which is exactly the
claim the transferred labels could not establish on their own.

Before training on them, confirm by eye that the classes separate:
    fig 1  col = phenotype, colored by phenotype   -- do CE / HTA / BA_rescue / wildtype
                                                      actually look like distinct classes?
    fig 2  row = genotype, col = phenotype          -- does het CE look like homo CE?
                                                      (if yes, the het-penetrance claim is real)

Note BA_rescue is kept SEPARATE here rather than pooled into HTA, so the pooling decision can be
made from the plot instead of inherited.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/23_plot_manual_b9d2_labels.py
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

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style,
    update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402

SOURCE = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
OUTPUT_DIR = RUN_DIR / "figures" / "manual_b9d2_labels"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
LABEL_COL = "cluster_categories"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]

# Manual classes, kept unpooled. 'unlabeled' is dropped: it is "not looked at", not a class.
PHENO_ORDER = ["CE", "HTA", "BA_rescue", "wildtype"]
PHENO_COLORS = {
    "CE": "#1B9E77",
    "HTA": "#D95F02",
    "BA_rescue": "#7570B3",
    "wildtype": "#999999",
}

# genotype column is b9d2_heterozygous / b9d2_homozygous / b9d2_wildtype / b9d2_unknown
GENO_ORDER = ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous", "b9d2_unknown"]
GENO_COLORS = {
    "b9d2_wildtype": "#7F7F7F",
    "b9d2_heterozygous": "#F7B267",
    "b9d2_homozygous": "#B2182B",
    "b9d2_unknown": "#4C9F70",
}


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=300,
        width_per_col=340,
        min_width=1100,
        individual_alpha=0.25,
        individual_width=0.7,
        trend_width=3.4,
        axis_label_fontsize=12,
        legend_fontsize=10,
    )


def load() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, low_memory=False)
    df = df[df[LABEL_COL].isin(PHENO_ORDER)].copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *FEATURES])
    return df


def _counts(df: pd.DataFrame) -> str:
    per = df.drop_duplicates(ID_COL).groupby([LABEL_COL, "genotype"]).size()
    return "   ".join(f"{p}/{g.replace('b9d2_','')}={n}" for (p, g), n in per.items())


def plot_by_phenotype(df: pd.DataFrame, out: Path) -> None:
    """Both features, col = phenotype, colored by phenotype. Do the classes look distinct?"""
    phenos = [p for p in PHENO_ORDER if p in set(df[LABEL_COL])]
    fig = plot_feature_over_time(
        df,
        features=FEATURES,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by=LABEL_COL,
        color_preset=ColorPreset(colors=PHENO_COLORS, order=phenos),
        facet_col=LABEL_COL,
        layout=FacetSpec(col_order=phenos, sharex=True, sharey=False),
        show_individual=True,
        show_trend=True,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title="Manual b9d2 labels — curvature & length by curated phenotype",
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    fig.text(0.5, -0.02, _counts(df), ha="center", fontsize=8, color="#555555")
    _save(fig, out)


def plot_genotype_split(df: pd.DataFrame, feature: str, out: Path) -> None:
    """row = genotype, col = phenotype. Does het CE look like homo CE?"""
    phenos = [p for p in PHENO_ORDER if p in set(df[LABEL_COL])]
    genos = [g for g in GENO_ORDER if g in set(df["genotype"])]
    fig = plot_feature_over_time(
        df,
        features=feature,          # scalar -> facet_row honored
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="genotype",
        color_preset=ColorPreset(colors=GENO_COLORS, order=genos),
        facet_row="genotype",
        facet_col=LABEL_COL,
        layout=FacetSpec(row_order=genos, col_order=phenos, sharex=True, sharey=True),
        show_individual=True,
        show_trend=True,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"Manual b9d2 labels — {feature}: row = genotype, col = curated phenotype\n"
               f"het CE vs homo CE is the test: same shape => het penetrance is real"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    fig.text(0.5, -0.02, _counts(df), ha="center", fontsize=8, color="#555555")
    _save(fig, out)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    df = load()
    emb = df.drop_duplicates(ID_COL)
    print(f"{emb[ID_COL].nunique()} labeled embryos, {len(df)} rows")
    print("\ngenotype x phenotype (embryos):")
    print(pd.crosstab(emb["genotype"], emb[LABEL_COL]).to_string())

    plot_by_phenotype(df, OUTPUT_DIR / "01_by_phenotype.png")
    for feature in FEATURES:
        plot_genotype_split(df, feature, OUTPUT_DIR / f"02_genotype_split__{feature}.png")


if __name__ == "__main__":
    main()
