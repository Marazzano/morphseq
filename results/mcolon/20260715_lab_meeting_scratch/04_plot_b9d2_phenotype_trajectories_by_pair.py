"""Recreate the B9D2 pair trajectory grid, colored by phenotype."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.viz.styling import (  # noqa: E402
    B9D2_PHENOTYPE_COLORS,
    B9D2_PHENOTYPE_ORDER,
    canonicalize_b9d2_phenotype,
)


INPUT_CSV = (
    PROJECT_ROOT
    / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_b9d2_clean.csv"
)
OUTPUT_DIR = RUN_DIR / "figures" / "b9d2_phenotype_trajectories_by_pair"
OUTPUT_PNG = OUTPUT_DIR / "phenotype_trajectories_by_pair.png"

PAIR_ORDER = [f"b9d2_pair_{pair}" for pair in (2, 4, 5, 6, 7, 8)]
TIME_MIN_HPF = 11.0
TIME_MAX_HPF = 125.0
BIN_WIDTH_HPF = 3.0
TREND_SMOOTH_SIGMA = 1.5

FEATURES = (
    ("baseline_deviation_normalized", "Curvature (normalized)"),
    ("total_length_um", "Body length (µm)"),
)


def _qc_mask(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_data() -> pd.DataFrame:
    columns = [
        "embryo_id",
        "pair",
        "predicted_stage_hpf",
        "baseline_deviation_normalized",
        "total_length_um",
        "phenotype_clean",
        "zygosity",
        "use_embryo_flag",
    ]
    df = pd.read_csv(INPUT_CSV, usecols=columns, low_memory=False)
    df = df[_qc_mask(df["use_embryo_flag"]) & df["pair"].isin(PAIR_ORDER)].copy()
    for column in ("predicted_stage_hpf", "baseline_deviation_normalized", "total_length_um"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["predicted_stage_hpf"])
    df = df[df["predicted_stage_hpf"].between(TIME_MIN_HPF, TIME_MAX_HPF, inclusive="both")]
    df["phenotype_plot"] = df["phenotype_clean"].map(canonicalize_b9d2_phenotype)
    return df


def _phenotype_trend(panel: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray]:
    """Median across embryo-level bin medians, followed by light Gaussian smoothing."""
    work = panel[["embryo_id", "predicted_stage_hpf", feature]].dropna().copy()
    work["time_bin"] = (
        np.floor((work["predicted_stage_hpf"] - TIME_MIN_HPF) / BIN_WIDTH_HPF).astype(int)
    )
    embryo_bins = work.groupby(["embryo_id", "time_bin"], observed=True)[feature].median()
    trend = embryo_bins.groupby("time_bin").median().sort_index()
    if trend.empty:
        return np.array([]), np.array([])
    full_bins = np.arange(int(trend.index.min()), int(trend.index.max()) + 1)
    values = trend.reindex(full_bins).interpolate(limit_direction="both").to_numpy(dtype=float)
    values = gaussian_filter1d(values, sigma=TREND_SMOOTH_SIGMA, mode="nearest")
    times = TIME_MIN_HPF + (full_bins + 0.5) * BIN_WIDTH_HPF
    return times, values


def main() -> None:
    df = load_data()
    fig, axes = plt.subplots(
        len(FEATURES),
        len(PAIR_ORDER),
        figsize=(24, 9),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for row_index, (feature, y_label) in enumerate(FEATURES):
        for col_index, pair in enumerate(PAIR_ORDER):
            ax = axes[row_index, col_index]
            pair_df = df[df["pair"] == pair]
            for phenotype in B9D2_PHENOTYPE_ORDER:
                color = B9D2_PHENOTYPE_COLORS[phenotype]
                phenotype_df = pair_df[pair_df["phenotype_plot"] == phenotype]
                for _, embryo_df in phenotype_df.groupby("embryo_id", observed=True):
                    embryo_df = embryo_df.dropna(subset=[feature]).sort_values("predicted_stage_hpf")
                    is_wildtype = (embryo_df["zygosity"] == "wildtype").any()
                    ax.plot(
                        embryo_df["predicted_stage_hpf"],
                        embryo_df[feature],
                        color=color,
                        alpha=0.045 if is_wildtype else 0.27,
                        linewidth=0.6 if is_wildtype else 0.85,
                        zorder=0 if is_wildtype else 1,
                    )
                trend_x, trend_y = _phenotype_trend(phenotype_df, feature)
                if trend_x.size:
                    ax.plot(
                        trend_x,
                        trend_y,
                        color=color,
                        linewidth=2.8,
                        solid_capstyle="round",
                        zorder=3,
                    )

            ax.set_xlim(TIME_MIN_HPF, TIME_MAX_HPF)
            ax.grid(alpha=0.18, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
            if row_index == 0:
                ax.set_title(pair, fontsize=12, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
            if row_index == len(FEATURES) - 1:
                ax.set_xlabel("Time (hpf)", fontsize=11)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=B9D2_PHENOTYPE_COLORS[phenotype],
            linewidth=2.8,
            label="Not Penetrant" if phenotype == "non_penetrant" else phenotype,
        )
        for phenotype in B9D2_PHENOTYPE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        title="Phenotype",
        loc="upper left",
        bbox_to_anchor=(0.905, 0.91),
        fontsize=12,
        title_fontsize=12,
    )
    fig.suptitle("Trajectories by Pair, Grouped by Phenotype", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.04, 0.90, 0.94), w_pad=1.5, h_pad=1.6)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)

    counts = (
        df.drop_duplicates("embryo_id")
        .groupby(["pair", "phenotype_plot"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=PAIR_ORDER, columns=B9D2_PHENOTYPE_ORDER, fill_value=0)
    )
    print(counts.to_string())
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
