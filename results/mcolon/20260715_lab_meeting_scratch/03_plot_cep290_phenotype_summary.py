"""Create matching CEP290 curvature and length phenotype summaries."""

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
from analyze.viz.styling import CEP290_PHENOTYPE_COLORS  # noqa: E402


INPUT_CSV = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_cep290_clean.csv"
)
OUTPUT_DIR = RUN_DIR / "figures" / "cep290_phenotype_transition" / "static_plots"
OUTPUT_CURVATURE_PNG = (
    OUTPUT_DIR / "01_summary_error_band__High_to_Low__Low_to_High_overlay.png"
)
OUTPUT_LENGTH_PNG = (
    OUTPUT_DIR
    / "02_summary_error_band__High_to_Low__Low_to_High_overlay__total_length_um.png"
)
OUTPUT_CURVATURE_INDIVIDUAL_PNG = (
    OUTPUT_DIR / "03_individual_trajectories__High_to_Low__Low_to_High_overlay.png"
)
OUTPUT_LENGTH_INDIVIDUAL_PNG = (
    OUTPUT_DIR
    / "04_individual_trajectories__High_to_Low__Low_to_High_overlay__total_length_um.png"
)

PHENOTYPES = ("High_to_Low", "Low_to_High")
TIME_MIN_HPF = 24.0
TIME_MAX_HPF = 120.0
BIN_WIDTH_HPF = 3.0
TREND_SMOOTH_SIGMA = 1.5


def _qc_mask(values: pd.Series) -> pd.Series:
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
    for column in ("predicted_stage_hpf", "baseline_deviation_normalized", "total_length_um"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(
        subset=["predicted_stage_hpf", "baseline_deviation_normalized", "total_length_um"]
    )

    # Match the original NWDB summary: normalize on the complete QC-passing
    # CEP290 reference distribution before selecting time and phenotype.
    df["curvature"] = normalize_arbitrary_feature(
        df["baseline_deviation_normalized"],
        low=0,
        high_percentile=100,
        clip=False,
    )
    return df[
        (df["zygosity"] == "homozygous")
        & df["phenotype_clean"].isin(PHENOTYPES)
        & df["predicted_stage_hpf"].between(TIME_MIN_HPF, TIME_MAX_HPF, inclusive="both")
    ].copy()


def main() -> None:
    df = load_plot_data()
    counts = df.groupby("phenotype_clean")["embryo_id"].nunique().reindex(PHENOTYPES)
    if counts.isna().any() or (counts == 0).any():
        raise RuntimeError(f"Missing CEP290 phenotype data: {counts.to_dict()}")

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
            color_by="phenotype_clean",
            color_lookup={label: CEP290_PHENOTYPE_COLORS[label] for label in PHENOTYPES},
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
    print(f"CEP290 homozygous embryos: {counts.astype(int).to_dict()}")


if __name__ == "__main__":
    main()
