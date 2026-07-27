"""Sanity-check the label transfer by LOOKING at the trajectories it produced.

The composition figure (script 21) says b9d2 hets are 13% CE / 20% HTA. The model's own
held-out confusion matrix says true-NP is called HTA 31% of the time, so that het HTA number
may be pure classifier leak. Percentages cannot settle it -- the trajectories can.

The diagnostic figure is faceted  row = phenotype  x  col = zygosity,  per gene:

    if the transfer is real   -- a given phenotype row looks like ITSELF across all three
                                 zygosity columns (het CE bends like homo CE, just rarer).
    if it is classifier leak  -- the het/WT cells look like the NP row instead: flat,
                                 wildtype-shaped traces that merely got an HTA sticker.

Homozygous is the reference column (that is what the model was trained on), so read each row
left-to-right and ask whether the shape survives.

Also emits an overlay view: same phenotype, all zygosities on ONE axis, so shape agreement is
judged directly rather than across panels.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/22_plot_transfer_check_trajectories.py
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

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style,
    update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402

_mess = __import__("14_plot_resolve_phenotypes_from_mess")
load_gene = _mess.load_gene
GENE_ORDER = _mess.GENE_ORDER
ZYGOSITY_ORDER = _mess.ZYGOSITY_ORDER
GENE_CONFIG = _mess.GENE_CONFIG
ID_COL = _mess.ID_COL
TIME_COL = _mess.TIME_COL
FEATURES = _mess.FEATURES

OUTPUT_DIR = RUN_DIR / "figures" / "transfer_check_trajectories"
NP_LABEL = "Not Penetrant"
NP_COLOR = "#BBBBBB"

GENE_PHENOTYPES = {
    "cep290": ["High_to_Low", "Low_to_High", NP_LABEL],
    "b9d2": ["CE", "HTA", NP_LABEL],
}
# The feature each gene's phenotype is actually defined on -- curvature for both, but keep the
# mapping explicit so the length view can be requested separately.
ZYG_LABEL = {"wildtype": "WT", "heterozygous": "het", "homozygous": "homo"}
ZYG_COLORS = {"wildtype": "#7F7F7F", "heterozygous": "#F7B267", "homozygous": "#B2182B"}


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=300,
        width_per_col=360,
        min_width=1100,
        individual_alpha=0.22,
        individual_width=0.7,
        trend_width=3.4,
        axis_label_fontsize=12,
        legend_fontsize=10,
    )


def phenotype_colors(gene: str) -> dict:
    colors = dict(GENE_CONFIG[gene]["pheno_colors"])
    colors[NP_LABEL] = NP_COLOR
    return colors


def _counts_note(df: pd.DataFrame) -> str:
    """n embryos per (phenotype, zygosity) so a sparse cell can't be over-read."""
    per = (df.drop_duplicates(ID_COL)
             .groupby(["phenotype_clean", "zygosity"]).size())
    return "   ".join(
        f"{p}/{ZYG_LABEL[z]}={n}" for (p, z), n in per.items() if z in ZYG_LABEL
    )


def plot_grid(df: pd.DataFrame, gene: str, feature: str, out: Path) -> None:
    """row = phenotype, col = zygosity. Colored by zygosity so leak reads as color-vs-shape."""
    phenos = [p for p in GENE_PHENOTYPES[gene] if p in set(df["phenotype_clean"])]
    fig = plot_feature_over_time(
        df,
        features=feature,   # scalar: a list forces multi-feature mode and overrides facet_row
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="zygosity",
        color_preset=ColorPreset(colors=ZYG_COLORS, order=ZYGOSITY_ORDER),
        facet_row="phenotype_clean",
        facet_col="zygosity",
        layout=FacetSpec(row_order=phenos, col_order=ZYGOSITY_ORDER,
                         sharex=True, sharey=True),
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"{gene} — {feature}: does each phenotype hold its shape across zygosity?\n"
               f"(homozygous = training reference; read each row left→right)"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    fig.text(0.5, -0.02, _counts_note(df[df["gene"] == gene]), ha="center",
             fontsize=8, color="#555555")
    _save(fig, out)


def plot_overlay(df: pd.DataFrame, gene: str, feature: str, out: Path) -> None:
    """One panel per phenotype, all zygosities overlaid -- direct shape comparison."""
    phenos = [p for p in GENE_PHENOTYPES[gene] if p in set(df["phenotype_clean"])]
    fig = plot_feature_over_time(
        df,
        features=[feature],
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="zygosity",
        color_preset=ColorPreset(colors=ZYG_COLORS, order=ZYGOSITY_ORDER),
        facet_col="phenotype_clean",
        facet_row=None,
        layout=FacetSpec(col_order=phenos, sharex=True, sharey=True),
        show_individual=False,
        show_trend=True,
        show_error_band=True,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"{gene} — {feature}: same phenotype, zygosities overlaid\n"
               f"(bands should COINCIDE if the transferred label means the same thing)"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    _save(fig, out)


def plot_overlay_by_phenotype(df: pd.DataFrame, gene: str, feature: str, out: Path) -> None:
    """Mirror of plot_overlay: col = phenotype, but colored BY PHENOTYPE instead of zygosity.

    Same panels, the other coloring. Here every trace in a panel shares a color, so the eye
    reads the phenotype's own shape without zygosity splitting it -- the reference view for
    'what is this class supposed to look like'.
    """
    phenos = [p for p in GENE_PHENOTYPES[gene] if p in set(df["phenotype_clean"])]
    fig = plot_feature_over_time(
        df,
        features=[feature],
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="phenotype_clean",
        color_preset=ColorPreset(colors=phenotype_colors(gene), order=phenos),
        facet_col="phenotype_clean",
        facet_row=None,
        layout=FacetSpec(col_order=phenos, sharex=True, sharey=True),
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=f"{gene} — {feature}: col = phenotype, colored by phenotype",
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    _save(fig, out)


def plot_multifeature(df: pd.DataFrame, gene: str, out: Path) -> None:
    """Both features at once: row = feature (curvature, length), col = phenotype, color = genotype.

    One figure per gene instead of one per feature -- curvature and length are read together,
    since a phenotype call that holds on curvature but not on length is a weaker call.
    """
    phenos = [p for p in GENE_PHENOTYPES[gene] if p in set(df["phenotype_clean"])]
    fig = plot_feature_over_time(
        df,
        features=FEATURES,          # rows: curvature then length
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="zygosity",
        color_preset=ColorPreset(colors=ZYG_COLORS, order=ZYGOSITY_ORDER),
        facet_col="phenotype_clean",
        facet_row=None,             # feature already occupies the row axis
        layout=FacetSpec(col_order=phenos, sharex=True, sharey=False),
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"{gene} — curvature & length by transferred phenotype, colored by genotype"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    fig.text(0.5, -0.02, _counts_note(df), ha="center", fontsize=8, color="#555555")
    _save(fig, out)


def plot_curvature_genotype_split(df: pd.DataFrame, gene: str, out: Path) -> None:
    """Curvature only, genotype SPLIT OUT into its own row: row = genotype, col = phenotype.

    The overlay hid the point by stacking genotypes on one axis. Splitting genotype onto the row
    axis isolates each cell -- so the het row of the Not-Penetrant column stands alone and you
    can see directly whether those hets carry a real curvature phenotype or are flat like WT.
    """
    feature = "baseline_deviation_normalized"
    phenos = [p for p in GENE_PHENOTYPES[gene] if p in set(df["phenotype_clean"])]
    fig = plot_feature_over_time(
        df,
        features=feature,   # scalar feature -> facet_row is honored
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by="zygosity",
        color_preset=ColorPreset(colors=ZYG_COLORS, order=ZYGOSITY_ORDER),
        facet_row="zygosity",
        facet_col="phenotype_clean",
        layout=FacetSpec(row_order=ZYGOSITY_ORDER, col_order=phenos,
                         sharex=True, sharey=True),
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=(f"{gene} — curvature, genotype split out (row=genotype, col=phenotype)\n"
               f"read the het row: do 'Not Penetrant' hets stay flat, or carry a phenotype?"),
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    fig.text(0.5, -0.02, _counts_note(df), ha="center", fontsize=8, color="#555555")
    _save(fig, out)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    frames = [load_gene(g) for g in GENE_ORDER]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["phenotype_clean"].notna()]
    df = df[df["zygosity"].isin(ZYGOSITY_ORDER)].copy()

    for gene in GENE_ORDER:
        sub = df[df["gene"] == gene]
        print(f"\n[{gene}] {_counts_note(sub)}")
        plot_multifeature(sub, gene, OUTPUT_DIR / f"{gene}__multifeature_by_genotype.png")
        plot_curvature_genotype_split(
            sub, gene, OUTPUT_DIR / f"{gene}__curvature_genotype_split.png")


if __name__ == "__main__":
    main()
