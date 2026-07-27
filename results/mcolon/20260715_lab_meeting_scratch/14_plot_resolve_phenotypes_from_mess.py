"""Demonstrate resolving dynamic phenotypes out of messy (mixed-zygosity) data.

Two figures, same grid (rows = feature [curvature, length], cols = gene [cep290, b9d2]), same
embryos -- only the coloring changes:

  figure 1  color = zygosity     -- the "mess": wildtype / het / homo trajectories overlaid, no
                                    phenotype structure visible.
  figure 2  color = phenotype    -- the same trajectories recolored by the TRANSFERRED phenotype
                                    label. The dynamic phenotype classes (cep290 HtL/LtH/NP,
                                    b9d2 CE/HTA) separate out of the mess.

The point: the phenotype models are trained on homozygotes, but here they are applied to ALL
zygosities (incl. heterozygotes and wildtype), showing dynamic phenotype structure can be pulled
even from carrier data that carries no curated phenotype label.

cep290 uses the 3-class NP model (script 11); b9d2 uses its shipped 2-class CE/HTA model.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/14_plot_resolve_phenotypes_from_mess.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
SOURCE = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style,
    update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time, ColorPreset  # noqa: E402
from analyze.viz.styling import (  # noqa: E402
    CEP290_PHENOTYPE_COLORS,
    B9D2_PHENOTYPE_COLORS,
    GENOTYPE_SUFFIX_COLORS,
)
from analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402

TABLE_DIR = SOURCE / "tables"
MODEL_DIR = SOURCE / "models"
OUTPUT_DIR = RUN_DIR / "figures" / "resolve_phenotypes_from_mess"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]
FEATURE_ORDER = FEATURES
GENE_ORDER = ["cep290", "b9d2"]

GENE_CONFIG = {
    "cep290": {
        "table": "reference_cep290_clean.csv",
        "model": "cep290_homozygous_phenotype_with_np.pkl",
        "pheno_colors": {k: CEP290_PHENOTYPE_COLORS[k]
                         for k in ("High_to_Low", "Low_to_High", "Not Penetrant")},
    },
    "b9d2": {
        "table": "reference_b9d2_clean.csv",
        # 3-class model (script 15): CE / HTA / Not Penetrant, NP defined from wildtype.
        "model": "b9d2_homozygous_phenotype_with_np.pkl",
        "pheno_colors": {**{k: B9D2_PHENOTYPE_COLORS[k] for k in ("CE", "HTA")},
                         "Not Penetrant": "#BBBBBB"},
    },
}

ZYGOSITY_ORDER = ["wildtype", "heterozygous", "homozygous"]
ZYGOSITY_COLORS = {k: GENOTYPE_SUFFIX_COLORS[k] for k in ZYGOSITY_ORDER}


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_gene(gene: str) -> pd.DataFrame:
    """All-zygosity trajectories + transferred phenotype label for one gene."""
    cfg = GENE_CONFIG[gene]
    with (MODEL_DIR / cfg["model"]).open("rb") as handle:
        model = pickle.load(handle)
    feature_cols = list(model["config"]["feature_cols"])

    df = pd.read_csv(
        TABLE_DIR / cfg["table"],
        usecols=lambda c: c in {ID_COL, TIME_COL, "zygosity", "use_embryo_flag", *FEATURES}
        or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    if "use_embryo_flag" in df.columns:
        df = df[_qc_mask(df["use_embryo_flag"])]
    df = df[df["zygosity"].isin(ZYGOSITY_ORDER)].copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *FEATURES, *feature_cols])

    # Transfer the phenotype label onto EVERY zygosity (the point of the demo).
    cross = transfer_labels_perbin(model, df, verbose=False)["embryo_support"][
        "embryo_cross_bin_prediction"
    ].set_index("query_embryo_id")
    df["phenotype_clean"] = df[ID_COL].map(cross["predicted_label"])

    df["gene"] = gene
    return df


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=320,
        width_per_col=380,
        min_width=1000,
        individual_alpha=0.14,
        individual_width=0.6,
        trend_width=3.6,
        axis_label_fontsize=13,
        legend_fontsize=11,
    )


def _plot(df: pd.DataFrame, color_by: str, color_lookup: dict, order: list, title: str, out: Path,
          *, show_individual: bool = True, show_error_band: bool = False) -> None:
    # ColorPreset.order sets the DRAW order: earlier = drawn first = underneath. Putting
    # wildtype first keeps it as the bottom layer, homozygous on top.
    preset = ColorPreset(colors=color_lookup, order=order)
    fig = plot_feature_over_time(
        df,
        features=FEATURE_ORDER,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by=color_by,
        color_preset=preset,
        facet_col="gene",
        facet_row=None,
        layout=FacetSpec(col_order=GENE_ORDER, sharex=True, sharey=False),
        show_individual=show_individual,
        show_trend=True,
        show_error_band=show_error_band,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=title,
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=True,
    )
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def main() -> None:
    frames = [load_gene(g) for g in GENE_ORDER]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["phenotype_clean"].notna()].copy()

    for gene in GENE_ORDER:
        sub = df[df["gene"] == gene].drop_duplicates(ID_COL)
        print(f"[{gene}] {sub[ID_COL].nunique()} embryos | "
              f"zygosity={sub['zygosity'].value_counts().to_dict()} | "
              f"phenotype={sub['phenotype_clean'].value_counts().to_dict()}")

    pheno_colors = {**GENE_CONFIG["cep290"]["pheno_colors"],
                    **GENE_CONFIG["b9d2"]["pheno_colors"]}

    # Three style variants x two colorings. Variants:
    #   traces        -- individual traces + trend (the original look)
    #   traces_bands  -- individual traces + error bands + trend
    #   bands         -- error bands + trend only (no individual traces)
    STYLE_VARIANTS = {
        "traces":       dict(show_individual=True,  show_error_band=False),
        "traces_bands": dict(show_individual=True,  show_error_band=True),
        "bands":        dict(show_individual=False, show_error_band=True),
    }
    STYLE_LABEL = {
        "traces": "traces", "traces_bands": "traces + error bands", "bands": "error bands only",
    }

    # Draw order (first = underneath). Wildtype under het under homozygous; Not Penetrant
    # under the active phenotype classes.
    zyg_order = ["wildtype", "heterozygous", "homozygous"]
    pheno_order = ["Not Penetrant", "High_to_Low", "Low_to_High", "CE", "HTA"]

    for variant, kwargs in STYLE_VARIANTS.items():
        _plot(
            df, "zygosity", ZYGOSITY_COLORS, zyg_order,
            f"The mess: mixed-zygosity trajectories, colored by zygosity ({STYLE_LABEL[variant]})",
            OUTPUT_DIR / f"01_colored_by_zygosity__{variant}.png",
            **kwargs,
        )
        _plot(
            df, "phenotype_clean", pheno_colors, pheno_order,
            f"Resolved: colored by transferred phenotype ({STYLE_LABEL[variant]})",
            OUTPUT_DIR / f"02_colored_by_phenotype__{variant}.png",
            **kwargs,
        )


if __name__ == "__main__":
    main()
