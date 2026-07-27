"""Create the B9D2 CE/HTA curvature summary used for the lab meeting.

This mirrors the NWDB CEP290 phenotype-transition overlay: median curvature in
3 hpf bins, Gaussian-smoothed dashed trends, and interquartile error bands.
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

from analyze.utils.stats import normalize_arbitrary_feature  # noqa: E402
from analyze.viz.plotting.feature_over_time import plot_feature_over_time  # noqa: E402
from analyze.viz.styling import (  # noqa: E402
    B9D2_PHENOTYPE_COLORS,
    canonicalize_b9d2_phenotype,
)


INPUT_CSV = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv"
)
OUTPUT_DIR = RUN_DIR / "figures" / "b9d2_phenotype_transition" / "static_plots"
OUTPUT_CURVATURE_PNG = OUTPUT_DIR / "01_summary_error_band__CE__HTA_overlay.png"
OUTPUT_LENGTH_PNG = OUTPUT_DIR / "02_summary_error_band__CE__HTA_overlay__total_length_um.png"
OUTPUT_CURVATURE_INDIVIDUAL_PNG = OUTPUT_DIR / "03_individual_trajectories__CE__HTA_overlay.png"
OUTPUT_LENGTH_INDIVIDUAL_PNG = (
    OUTPUT_DIR / "04_individual_trajectories__CE__HTA_overlay__total_length_um.png"
)

PHENOTYPES = ("CE", "HTA")
TIME_MIN_HPF = 24.0
TIME_MAX_HPF = 120.0
BIN_WIDTH_HPF = 3.0
TREND_SMOOTH_SIGMA = 1.5


def _qc_mask(values: pd.Series) -> pd.Series:
    """Interpret bool-like use_embryo_flag values consistently."""
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_plot_data() -> pd.DataFrame:
    columns = [
        "embryo_id",
        "predicted_stage_hpf",
        "baseline_deviation_normalized",
        "total_length_um",
        "use_embryo_flag",
        "zygosity",
        "phenotype_clean",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=columns, low_memory=False)
    df = df[_qc_mask(df["use_embryo_flag"])].copy()
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df["baseline_deviation_normalized"] = pd.to_numeric(
        df["baseline_deviation_normalized"], errors="coerce"
    )
    df["total_length_um"] = pd.to_numeric(df["total_length_um"], errors="coerce")
    df = df.dropna(
        subset=["predicted_stage_hpf", "baseline_deviation_normalized", "total_length_um"]
    )

    # Match the CEP290 source figure: normalize before applying the time and
    # phenotype subsets, using the complete QC-passing reference distribution.
    df["curvature"] = normalize_arbitrary_feature(
        df["baseline_deviation_normalized"],
        low=0,
        high_percentile=100,
        clip=False,
    )
    df["phenotype_binary"] = df["phenotype_clean"].map(canonicalize_b9d2_phenotype)
    return df[
        (df["zygosity"] == "homozygous")
        & df["phenotype_binary"].isin(PHENOTYPES)
        & df["predicted_stage_hpf"].between(TIME_MIN_HPF, TIME_MAX_HPF, inclusive="both")
    ].copy()


def main() -> None:
    df = load_plot_data()
    counts = df.groupby("phenotype_binary")["embryo_id"].nunique().reindex(PHENOTYPES)
    if counts.isna().any() or (counts == 0).any():
        raise RuntimeError(f"Missing B9D2 phenotype data: {counts.to_dict()}")

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.labelsize": 17,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = (
        ("curvature", "Curvature", (0.0, 1.0), False, True, True, OUTPUT_CURVATURE_PNG),
        ("total_length_um", "Total length (µm)", None, False, True, True, OUTPUT_LENGTH_PNG),
        (
            "curvature",
            "Curvature",
            (0.0, 1.0),
            True,
            True,
            False,
            OUTPUT_CURVATURE_INDIVIDUAL_PNG,
        ),
        (
            "total_length_um",
            "Total length (µm)",
            None,
            True,
            True,
            False,
            OUTPUT_LENGTH_INDIVIDUAL_PNG,
        ),
    )
    for feature, y_label, ylim, show_individual, show_trend, show_band, output_path in outputs:
        fig = plot_feature_over_time(
            df,
            features=feature,
            time_col="predicted_stage_hpf",
            id_col="embryo_id",
            color_by="phenotype_binary",
            color_lookup={label: B9D2_PHENOTYPE_COLORS[label] for label in PHENOTYPES},
            show_individual=show_individual,
            show_trend=show_trend,
            show_error_band=show_band,
            trend_statistic="median",
            trend_smooth_sigma=TREND_SMOOTH_SIGMA,
            trend_linestyle="dashed",
            bin_width=BIN_WIDTH_HPF,
            error_type="iqr",
            backend="matplotlib",
            xlim=(TIME_MIN_HPF, TIME_MAX_HPF),
            ylim=ylim,
            legend_loc="outside",
        )
        fig.set_size_inches(5.0, 4.5, forward=True)
        for ax in fig.axes:
            ax.set_xlabel("Hours post fertilization")
            ax.set_ylabel(y_label)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        for legend in list(fig.legends):
            legend.remove()

        fig.savefig(output_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"Saved: {output_path}")
    print(f"B9D2 homozygous embryos: {counts.astype(int).to_dict()}")


if __name__ == "__main__":
    main()
