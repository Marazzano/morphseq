"""Look at the proposed 3-class b9d2 training labels BEFORE training on them.

Class definition (Option B):
    CE                  -- every manually curated CE embryo, any genotype
                           (15 het + 8 homo + 12 unknown + 3 wildtype-genotype)  n=38
    homozygous NON-CE   -- homozygotes curated as something else (22 HTA + 7 BA_rescue)  n=29
    wildtype            -- the 35 curated wildtype embryos                        n=35

The 2 wildtype-genotype HTA embryos (20251125_B06_e01, 20251125_F12_e01) are dropped: they fit
neither the homozygous class nor the wildtype class.

Note the built-in confound: CE spans all four genotypes while 'homozygous NON-CE' is by
construction 100% homozygous. So any CE-vs-homoNONCE separation may ride on genotype-correlated
signal rather than CE morphology. Colouring by zygosity here makes that visible up front --
if the CE panel's genotype medians diverge, CE is not one coherent morphological class.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/25_plot_three_class_training_labels.py
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
OUTPUT_DIR = RUN_DIR / "figures" / "three_class_training_labels"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
LABEL_COL = "cluster_categories"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]

CLASS_CE = "CE"
CLASS_HOMO = "homozygous NON-CE"
CLASS_WT = "wildtype"
CLASS_ORDER = [CLASS_CE, CLASS_HOMO, CLASS_WT]

# Embryos curated as a phenotype but genotyped wildtype; they fit neither non-CE class.
DROP_IDS = {"20251125_B06_e01", "20251125_F12_e01"}

GENO_ORDER = ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous", "b9d2_unknown"]
GENO_COLORS = {
    "b9d2_wildtype": "#7F7F7F",
    "b9d2_heterozygous": "#F7B267",
    "b9d2_homozygous": "#B2182B",
    "b9d2_unknown": "#4C9F70",
}


def build_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 3-class definition; return only embryos that land in a class."""
    df = df[~df[ID_COL].isin(DROP_IDS)].copy()

    is_ce = df[LABEL_COL] == "CE"
    is_homo_nonce = (df["genotype"] == "b9d2_homozygous") & df[LABEL_COL].isin(
        ["HTA", "BA_rescue"]
    )
    is_wt = df[LABEL_COL] == "wildtype"

    df["train_label"] = pd.NA
    df.loc[is_ce, "train_label"] = CLASS_CE
    df.loc[is_homo_nonce, "train_label"] = CLASS_HOMO
    df.loc[is_wt, "train_label"] = CLASS_WT
    return df[df["train_label"].notna()].copy()


def load() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, low_memory=False)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *FEATURES])
    return build_training_labels(df)


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=300,
        width_per_col=360,
        min_width=1000,
        individual_alpha=0.25,
        individual_width=0.7,
        trend_width=3.4,
        axis_label_fontsize=12,
        legend_fontsize=10,
    )


def main() -> None:
    df = load()
    emb = df.drop_duplicates(ID_COL)
    print(f"{emb[ID_COL].nunique()} training embryos")
    print()
    print(pd.crosstab(emb["train_label"], emb["genotype"].str.replace("b9d2_", ""),
                      margins=True).to_string())

    genos = [g for g in GENO_ORDER if g in set(df["genotype"])]
    note = "   ".join(
        f"{lab}/{g.replace('b9d2_','')}={n}"
        for (lab, g), n in emb.groupby(["train_label", "genotype"]).size().items()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common = dict(
        features=FEATURES,               # rows = curvature, length
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="genotype",             # colored by zygosity
        color_preset=ColorPreset(colors=GENO_COLORS, order=genos),
        facet_col="train_label",         # split by proposed training label
        layout=FacetSpec(col_order=CLASS_ORDER, sharex=True, sharey=False),
        show_individual=True,
        show_trend=True,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        title="Proposed 3-class b9d2 training labels — colored by zygosity",
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )

    # Interactive: hover shows "ID: <embryo_id>" on every individual trace, so specific
    # embryos can be identified and pulled for review. Self-contained HTML -- download and open.
    html_out = OUTPUT_DIR / "three_class_labels_by_zygosity.html"
    plot_feature_over_time(df, backend="plotly", output_path=str(html_out), **common)
    print(f"\nSaved: {html_out.relative_to(RUN_DIR)}")

    # Static companion for slides.
    fig = plot_feature_over_time(df, backend="matplotlib", **common)
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#555555")
    png_out = OUTPUT_DIR / "three_class_labels_by_zygosity.png"
    fig.savefig(png_out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_out.relative_to(RUN_DIR)}")

    # Embryo roster so the trajectories can be tied back to specific wells.
    roster = (emb[[ID_COL, "experiment_id", "genotype", LABEL_COL, "train_label"]]
              .sort_values(["train_label", "genotype", ID_COL]))
    roster_out = OUTPUT_DIR / "three_class_training_roster.csv"
    roster.to_csv(roster_out, index=False)
    print(f"Saved: {roster_out.relative_to(RUN_DIR)}  ({len(roster)} embryos)")


if __name__ == "__main__":
    main()
