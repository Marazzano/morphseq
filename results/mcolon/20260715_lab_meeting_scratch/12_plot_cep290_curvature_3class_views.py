"""CEP290 F1-offspring curvature trends using the 3-class (HtL / LtH / Not Penetrant) model.

Uses the retrained ``cep290_homozygous_phenotype_with_np.pkl`` (script 11) so homozygotes can
receive a Not-Penetrant call instead of being force-split into HtL/LtH. Emits several faceted
views of curvature (``baseline_deviation_normalized``) over stage:

  view1  cols = pair,        one row (experiments pooled),   color = phenotype
  view2a rows = experiment,  cols = phenotype,               color = pair
  view2b rows = phenotype,   cols = experiment,              color = pair
  view2c cols = experiment,  one row,                        color = phenotype

READ BEFORE INTERPRETING: 20260208 and 20260219 only image ~10-47 hpf -- before the curvature
phenotype diverges -- so their homozygotes classify as ~100% Not Penetrant. That is a stage-
coverage effect (an undiverged embryo looks wildtype, and NP is trained on wildtype appearance),
not necessarily a penetrance finding. Only 20260210 (reaches 88 hpf) gives a clean 3-way split.
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.plotting.faceting_engine import FacetSpec  # noqa: E402
from analyze.viz.plotting.faceting_engine.style.defaults import (  # noqa: E402
    presentation_style,
    update_style,
)
from analyze.viz.plotting.feature_over_time import plot_feature_over_time  # noqa: E402
from analyze.viz.styling import CEP290_PHENOTYPE_COLORS  # noqa: E402
from analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402


BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
MODEL_PATH = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/models/"
    "cep290_homozygous_phenotype_with_np.pkl"
)
OUTPUT_DIR = RUN_DIR / "figures" / "cep290_curvature_3class_views"

EXPERIMENTS = ["20260208", "20260210", "20260219"]
PHENOTYPE_ORDER = ["High_to_Low", "Low_to_High", "Not Penetrant"]
FEATURE = "baseline_deviation_normalized"
Y_LABEL = "Curvature (normalized)"
EXPECTED_HOMOZYGOTES = {"20260208": 7, "20260210": 30, "20260219": 5}

PAIR_COLORS = {
    "cep290_pair_2_F1s": "#4C78A8",
    "cep290_pair_3_F1s": "#E45756",
    "cep290_spawn": "#72B043",
}


def _normalize_genotype(genotype: object) -> str:
    value = str(genotype).strip().lower().replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.replace("cep290_unkown", "cep290_unknown").replace(
        "cep290_homozyous", "cep290_homozygous"
    )


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_data() -> tuple[pd.DataFrame, list[str]]:
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    feature_cols = list(model["config"]["feature_cols"])
    base_cols = list(dict.fromkeys(
        ["embryo_id", "experiment_id", "pair", "genotype", "predicted_stage_hpf",
         "use_embryo_flag", FEATURE]
    ))

    frames = []
    for experiment in EXPERIMENTS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        frame = pd.read_csv(path, usecols=base_cols + feature_cols, low_memory=False)
        frame["experiment_id"] = experiment
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    df = df[_qc_mask(df["use_embryo_flag"])].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["pair"] = df["pair"].astype(str)
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")

    df = df[df["genotype"] == "cep290_homozygous"].dropna(
        subset=["predicted_stage_hpf", *feature_cols]
    )

    observed = df.groupby("experiment_id")["embryo_id"].nunique().to_dict()
    for experiment, expected in EXPECTED_HOMOZYGOTES.items():
        if observed.get(experiment, 0) != expected:
            raise RuntimeError(
                f"{experiment}: expected {expected} homozygotes, found "
                f"{observed.get(experiment, 0)}."
            )

    predictions = []
    for experiment, group in df.groupby("experiment_id", sort=True):
        cross = transfer_labels_perbin(model, group, verbose=False)["embryo_support"][
            "embryo_cross_bin_prediction"
        ]
        predictions.append(cross.assign(experiment_id=experiment))
    prediction = pd.concat(predictions, ignore_index=True).set_index("query_embryo_id")
    df["phenotype_clean"] = df["embryo_id"].map(prediction["predicted_label"])
    df = df[df["phenotype_clean"].isin(PHENOTYPE_ORDER)].copy()

    pair_order = sorted(df["pair"].dropna().unique().tolist())
    return df, pair_order


def _style() -> dict:
    return update_style(
        presentation_style(),
        height_per_row=330,
        width_per_col=330,
        min_width=1400,
        individual_alpha=0.28,
        individual_width=0.9,
        trend_width=3.6,
        axis_label_fontsize=13,
        legend_fontsize=12,
    )


def _base_kwargs() -> dict:
    return dict(
        features=FEATURE,
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        style=_style(),
        legend_loc="outside",
        repeat_xlabels=True,
        repeat_ylabels=False,
    )


def main() -> None:
    df, pair_order = load_data()
    experiment_order = sorted(df["experiment_id"].unique())
    pheno_colors = {k: CEP290_PHENOTYPE_COLORS[k] for k in PHENOTYPE_ORDER}
    pair_colors = {p: PAIR_COLORS.get(p, "#888888") for p in pair_order}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ax in fig.axes:
            ax.set_xlabel("Hours post fertilization")
        fig.axes[0].set_ylabel(Y_LABEL)
        path = OUTPUT_DIR / name
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # view1: cols = pair, one row (experiments pooled), color = phenotype, 40-90 hpf
    save(plot_feature_over_time(
        df, color_by="phenotype_clean", color_lookup=pheno_colors,
        facet_col="pair", xlim=(40.0, 90.0),
        layout=FacetSpec(col_order=pair_order, sharex=True, sharey=True),
        title="F1 curvature by pair (experiments pooled, 40-90 hpf), 3-class phenotype",
        **_base_kwargs(),
    ), "view1_col_pair_pooled_color_phenotype.png")

    # view2a: rows = experiment, cols = phenotype, color = pair
    save(plot_feature_over_time(
        df, color_by="pair", color_lookup=pair_colors,
        facet_row="experiment_id", facet_col="phenotype_clean",
        layout=FacetSpec(row_order=experiment_order, col_order=PHENOTYPE_ORDER,
                         sharex=True, sharey=True),
        title="F1 curvature: experiment x phenotype, colored by pair",
        **_base_kwargs(),
    ), "view2a_row_experiment_col_phenotype_color_pair.png")

    # view2b: rows = phenotype, cols = experiment, color = pair
    save(plot_feature_over_time(
        df, color_by="pair", color_lookup=pair_colors,
        facet_row="phenotype_clean", facet_col="experiment_id",
        layout=FacetSpec(row_order=PHENOTYPE_ORDER, col_order=experiment_order,
                         sharex=True, sharey=True),
        title="F1 curvature: phenotype x experiment, colored by pair",
        **_base_kwargs(),
    ), "view2b_row_phenotype_col_experiment_color_pair.png")

    # view2c: cols = experiment, one row, color = phenotype
    save(plot_feature_over_time(
        df, color_by="phenotype_clean", color_lookup=pheno_colors,
        facet_col="experiment_id",
        layout=FacetSpec(col_order=experiment_order, sharex=True, sharey=True),
        title="F1 curvature by experiment (one row), 3-class phenotype",
        **_base_kwargs(),
    ), "view2c_col_experiment_color_phenotype.png")

    summary = (
        df.drop_duplicates("embryo_id")
        .groupby(["experiment_id", "pair", "phenotype_clean"], observed=True)
        .size().rename("n_embryos").reset_index()
    )
    print("\nPer-facet counts:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
