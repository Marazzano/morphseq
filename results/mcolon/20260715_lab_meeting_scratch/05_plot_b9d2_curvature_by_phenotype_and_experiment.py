"""Facet B9D2 curvature by phenotype and color/group by experiment."""

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
from analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402


BUILD06_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
MODEL_PATH = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/models/b9d2_homozygous_phenotype.pkl"
)
OUTPUT_DIR = RUN_DIR / "figures" / "b9d2_curvature_by_phenotype_and_experiment"
OUTPUT_PNG = OUTPUT_DIR / "curvature_faceted_by_phenotype_colored_by_experiment.png"

PHENOTYPE_ORDER = ["CE", "HTA"]
# Build06 files discovered to contain pair values matching ``b9d2_pair_[0-9]+``.
B9D2_BUILD06_EXPERIMENTS = ["20251104", "20251119", "20251121", "20251125", "20260206"]
EXPERIMENT_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D"]


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_data(metric_col: str = "baseline_deviation_normalized") -> tuple[
    pd.DataFrame, list[str], list[str], dict[str, str]
]:
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    feature_cols = list(model["config"]["feature_cols"])
    base_cols = list(dict.fromkeys([
        "embryo_id",
        "experiment_id",
        "pair",
        "genotype",
        "predicted_stage_hpf",
        metric_col,
        "use_embryo_flag",
    ]))
    frames = []
    for experiment in B9D2_BUILD06_EXPERIMENTS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment}.csv"
        header = pd.read_csv(path, nrows=0).columns
        missing = sorted(set(base_cols + feature_cols) - set(header))
        if missing:
            raise RuntimeError(f"{path.name} is missing required columns: {missing}")
        frame = pd.read_csv(path, usecols=base_cols + feature_cols, low_memory=False)
        frame["experiment_id"] = experiment
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df[_qc_mask(df["use_embryo_flag"])].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["pair"] = df["pair"].astype(str)
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df[
        df["pair"].str.fullmatch(r"b9d2_pair_\d+", na=False)
        & df["genotype"].astype(str).str.endswith("_homozygous")
    ].dropna(subset=["predicted_stage_hpf", metric_col])

    # Retain every B9D2 pair represented in Build06, including pairs that occur
    # in only one experiment.
    embryo_meta = df.drop_duplicates("embryo_id")
    all_pairs = embryo_meta["pair"].dropna().unique().tolist()
    pair_order = sorted(all_pairs, key=lambda value: int(value.rsplit("_", 1)[-1]))
    df = df[df["pair"].isin(pair_order)].copy()

    # Cross-bin prediction pools phenotype probabilities across all supported
    # time bins to produce one CE/HTA label per embryo.
    prediction = transfer_labels_perbin(model, df, verbose=False)["embryo_support"][
        "embryo_cross_bin_prediction"
    ]
    predicted_labels = prediction.set_index("query_embryo_id")["predicted_label"]
    df["phenotype_clean"] = df["embryo_id"].map(predicted_labels)
    df = df[df["phenotype_clean"].isin(PHENOTYPE_ORDER)].copy()

    experiment_order = sorted(df["experiment_id"].unique())
    experiment_colors = dict(zip(experiment_order, EXPERIMENT_PALETTE))
    return df, pair_order, experiment_order, experiment_colors


def main(
    *,
    metric_col: str = "baseline_deviation_normalized",
    y_label: str = "Curvature (normalized)",
    output_path: Path = OUTPUT_PNG,
    title: str = "Build06 B9D2 predicted-phenotype curvature by pair and experiment",
) -> None:
    df, pair_order, experiment_order, experiment_colors = load_data(metric_col)
    counts = (
        df.drop_duplicates("embryo_id")
        .groupby(["phenotype_clean", "pair", "experiment_id"], observed=True)
        .size()
        .rename("n_embryos")
        .reset_index()
    )
    if df.empty:
        raise RuntimeError("No Build06 B9D2 embryos received CE/HTA phenotype predictions.")

    style = update_style(
        presentation_style(),
        height_per_row=330,
        width_per_col=320,
        min_width=1600,
        individual_alpha=0.16,
        individual_width=0.75,
        trend_width=3.0,
        axis_label_fontsize=13,
        legend_fontsize=12,
    )
    fig = plot_feature_over_time(
        df,
        features=metric_col,
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="experiment_id",
        color_lookup=experiment_colors,
        facet_row="phenotype_clean",
        facet_col="pair",
        layout=FacetSpec(
            row_order=PHENOTYPE_ORDER,
            col_order=pair_order,
            sharex=True,
            sharey=True,
        ),
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
        trend_linestyle="dotted",
        bin_width=3.0,
        smooth_method="gaussian",
        smooth_params={"sigma": 1.0},
        backend="matplotlib",
        title=title,
        style=style,
        legend_loc="outside",
        xlim=(11.0, 125.0),
        repeat_xlabels=False,
        repeat_ylabels=False,
        repeat_xticklabels=True,
        repeat_yticklabels=True,
    )
    fig.set_size_inches(25.0, 8.0, forward=True)
    for axis_index, ax in enumerate(fig.axes):
        row_index, col_index = divmod(axis_index, len(pair_order))
        ax.set_xlabel("Hours post fertilization" if row_index == len(PHENOTYPE_ORDER) - 1 else "")
        ax.set_ylabel(y_label if col_index == 0 else "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Build06 pairs: {pair_order}")
    print(f"Experiments: {experiment_order}")
    print(counts.to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
