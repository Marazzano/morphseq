#!/usr/bin/env python
"""Generate slide-ready cumulative MorphSeq corpus-size figures.

The lower estimate applies the currently strict QC pass rate to every censusable acquisition.
The upper estimate applies only death and structural mask-geometry exclusions. Both estimates
therefore assume that every acquisition with a gross census count ultimately runs to completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "corpus_census_data"
REQUESTED_SLIDE_DIR = Path(
    "/Users/nick/Library/CloudStorage/"
    "GoogleDrive-nlammers@uw.edu/My Drive/projects/morphseq/slides/20260805/data_census"
)
DEFAULT_OUTPUT_DIR = (
    REQUESTED_SLIDE_DIR
    if REQUESTED_SLIDE_DIR.parent.is_dir()
    else HERE / "data_census"
)

VERSION_CUTOFFS: tuple[tuple[str, str | None], ...] = (
    ("v1", "2023-09-01"),
    ("v2", "2024-02-07"),
    ("v3", "2024-11-07"),
    ("v4", "2025-07-31"),
    ("v5", None),
)
VERSION_ORDER = [version for version, _ in VERSION_CUTOFFS]

UNIT_LABELS = {
    "embryos": "Embryos",
    "embryo_times": "Embryo-timepoints",
    "embryo_times_z": "Embryo-timepoints × z",
}
PERTURBATION_COLORS = {
    # ColorBrewer Set2, reordered so controls remain neutral.
    "control": "#B3B3B3",
    "environmental": "#66C2A5",
    "chemical": "#FC8D62",
    "genetic": "#8DA0CB",
    "environmental+chemical": "#E78AC3",
    "environmental+genetic": "#A6D854",
    "chemical+genetic": "#E5C494",
    "environmental+chemical+genetic": "#FFD92F",
}
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 15
ANNOTATION_FONTSIZE = 13.5
MEDIAN_COLOR = "#24557A"
PERTURBATION_TREND_COLOR = "#6B8F71"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--count-unit",
        choices=sorted(UNIT_LABELS),
        default="embryo_times",
    )
    parser.add_argument("--z-slice-multiplier", type=float, default=5.0)
    return parser.parse_args()


def compact_count(value: float, _position: object = None) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def load_projection_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    wells = pd.read_csv(data_dir / "well_census.csv", low_memory=False)
    policy = pd.read_csv(
        data_dir / "qc_flag_census/policy_sensitivity_summary.csv"
    )
    retention = pd.read_csv(data_dir / "qc_retention_summary.csv")
    return wells, _projection_rates(policy, retention)


def _projection_rates(
    policy: pd.DataFrame,
    retention: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scope in ("Keyence", "YX1"):
        strict = policy.loc[
            policy["scope"].eq(scope) & policy["policy"].eq("strict_current")
        ]
        upper = policy.loc[
            policy["scope"].eq(scope)
            & policy["policy"].eq("death_and_mask_geometry")
        ]
        if len(strict) != 1 or len(upper) != 1:
            raise ValueError(
                f"Expected one strict_current and death_and_mask_geometry row for {scope}."
            )
        strict_row = strict.iloc[0]
        upper_row = upper.iloc[0]
        rows.append(
            {
                "scope": scope,
                "lower_timepoint_rate": strict_row["timepoint_pass_rate"],
                "upper_timepoint_rate": upper_row["timepoint_pass_rate"],
                "lower_embryo_rate": strict_row["embryo_any_pass_rate"],
                "upper_embryo_rate": upper_row["embryo_any_pass_rate"],
                "lower_policy": "strict current QC",
                "upper_policy": "death + mask geometry",
            }
        )

    seahub = retention.loc[retention["scope"].eq("SeaHub")]
    if len(seahub) != 1:
        raise ValueError("Expected one SeaHub retention row.")
    detection_rate = float(seahub.iloc[0]["retention_rate"])
    rows.append(
        {
            "scope": "SeaHub",
            "lower_timepoint_rate": detection_rate,
            "upper_timepoint_rate": 1.0,
            "lower_embryo_rate": detection_rate,
            "upper_embryo_rate": 1.0,
            "lower_policy": "current detection completion",
            "upper_policy": "all reconciled embryos",
        }
    )
    rates = pd.DataFrame(rows)
    for grain in ("timepoint", "embryo"):
        if (
            rates[f"lower_{grain}_rate"]
            > rates[f"upper_{grain}_rate"] + 1e-12
        ).any():
            raise ValueError(f"Lower {grain} rate exceeds upper rate.")
    return rates


def prepare_projection_rows(
    wells: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    count_unit: str,
) -> pd.DataFrame:
    if count_unit not in UNIT_LABELS:
        raise ValueError(f"Unknown count unit {count_unit!r}.")
    frame = wells.merge(rates, on="scope", how="left", validate="many_to_one")
    if frame["lower_timepoint_rate"].isna().any():
        missing = sorted(frame.loc[frame["lower_timepoint_rate"].isna(), "scope"].unique())
        raise ValueError(f"Missing projection rates for scope(s): {missing}")

    date_token = frame["corpus_dataset_id"].astype(str).str.extract(r"^(\d{8})")[0]
    frame["acquisition_date"] = pd.to_datetime(
        date_token, format="%Y%m%d", errors="coerce"
    )
    if count_unit == "embryos":
        frame["gross_plot_count"] = frame["gross_embryos"]
        lower_rate = frame["lower_embryo_rate"]
        upper_rate = frame["upper_embryo_rate"]
    else:
        frame["gross_plot_count"] = frame["gross_embryo_timepoints"]
        lower_rate = frame["lower_timepoint_rate"]
        upper_rate = frame["upper_timepoint_rate"]

    # Rows without a gross count are not censusable and contribute zero by definition.
    frame["lower_count"] = frame["gross_plot_count"] * lower_rate
    frame["upper_count"] = frame["gross_plot_count"] * upper_rate
    frame["midpoint_count"] = (frame["lower_count"] + frame["upper_count"]) / 2
    frame["perturbation_type"] = (
        frame["perturbation_class"].fillna("control").replace("", "control")
    )
    return frame


def build_version_estimates(
    projection_rows: pd.DataFrame,
    *,
    z_slice_multiplier: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimate_rows = []
    perturbation_rows = []
    for version, cutoff in VERSION_CUTOFFS:
        selected = (
            projection_rows
            if cutoff is None
            else projection_rows.loc[
                projection_rows["acquisition_date"].le(pd.Timestamp(cutoff))
            ]
        )
        lower = float(selected["lower_count"].sum())
        upper = float(selected["upper_count"].sum())
        midpoint = (lower + upper) / 2
        traditional = selected.loc[selected["scope"].ne("SeaHub")]
        unique_perturbations: set[tuple[str, str]] = set()
        for serialized_atoms in traditional["atomic_perturbations"].dropna():
            for domain, label in json.loads(serialized_atoms):
                unique_perturbations.add((str(domain), str(label)))
        seahub_midpoint = float(
            selected.loc[selected["scope"].eq("SeaHub"), "midpoint_count"].sum()
        )
        traditional_midpoint = midpoint - seahub_midpoint
        estimate_rows.append(
            {
                "version": version,
                "cutoff_date": cutoff or "all",
                "censusable_dataset_count": selected.loc[
                    selected["gross_plot_count"].gt(0), "corpus_dataset_id"
                ].nunique(),
                "gross_count": float(selected["gross_plot_count"].sum()),
                "lower_estimate": lower,
                "midpoint_estimate": midpoint,
                "upper_estimate": upper,
                "unique_perturbation_count": len(unique_perturbations),
                "midpoint_5z_estimate": (
                    traditional_midpoint * z_slice_multiplier + seahub_midpoint
                ),
            }
        )
        grouped = selected.groupby("perturbation_type")["midpoint_count"].sum()
        for perturbation_type, count in grouped.items():
            perturbation_rows.append(
                {
                    "version": version,
                    "perturbation_type": perturbation_type,
                    "midpoint_estimate": float(count),
                }
            )
    return pd.DataFrame(estimate_rows), pd.DataFrame(perturbation_rows)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        paths.append(path)
    return paths


def _style_version_axis(ax: plt.Axes, *, ylabel: str) -> None:
    ax.set_xticks(np.arange(len(VERSION_ORDER)), VERSION_ORDER)
    ax.set_xlabel("Corpus version", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(compact_count))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="x", rotation=0)
    ax.margins(x=0.04)


def plot_solid_bars(
    estimates: pd.DataFrame,
    *,
    count_unit: str,
    output_dir: Path,
) -> list[Path]:
    values = estimates.set_index("version").reindex(VERSION_ORDER)["midpoint_estimate"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        np.arange(len(VERSION_ORDER)),
        values.to_numpy(),
        width=0.70,
        color="#3B82F6",
    )
    for bar, value in zip(bars, values):
        ax.annotate(
            compact_count(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )
    _style_version_axis(ax, ylabel=f"Projected {UNIT_LABELS[count_unit].lower()}")
    ax.set_title("Cumulative MorphSeq corpus size", fontsize=TITLE_FONTSIZE)
    ax.margins(y=0.12)
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "01a_cumulative_corpus_bar_solid")
    plt.close(fig)
    return paths


def plot_perturbation_bars(
    perturbation: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    count_unit: str,
    output_dir: Path,
    include_5z_outline: bool = False,
    y_axis_max: float | None = None,
) -> list[Path]:
    stacked = (
        perturbation.pivot_table(
            index="version",
            columns="perturbation_type",
            values="midpoint_estimate",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(VERSION_ORDER, fill_value=0)
    )
    preferred = list(PERTURBATION_COLORS)
    stack_order = [
        category for category in preferred if category in stacked.columns
    ] + sorted(set(stacked.columns) - set(preferred))
    stacked = stacked.reindex(columns=stack_order)

    fig, ax = plt.subplots(figsize=(9.8, 8.7))
    stacked.plot.bar(
        stacked=True,
        ax=ax,
        width=0.70,
        color=[
            PERTURBATION_COLORS.get(category, "#374151")
            for category in stacked.columns
        ],
    )
    ordered_estimates = estimates.set_index("version").reindex(VERSION_ORDER)
    totals = stacked.sum(axis=1)
    for index, value in enumerate(totals):
        perturbation_count = int(
            ordered_estimates.iloc[index]["unique_perturbation_count"]
        )
        truncated = y_axis_max is not None and value > y_axis_max
        annotation_y = y_axis_max if truncated else value
        annotation_box = (
            {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
            if truncated
            else None
        )
        ax.annotate(
            compact_count(value),
            (index, annotation_y),
            xytext=(0, -8 if truncated else 27),
            textcoords="offset points",
            ha="center",
            va="top" if truncated else "bottom",
            fontsize=ANNOTATION_FONTSIZE,
            fontweight="bold",
            bbox=annotation_box,
        )
        ax.annotate(
            f"{perturbation_count} perturbations",
            (index, annotation_y),
            xytext=(0, -34 if truncated else 5),
            textcoords="offset points",
            ha="center",
            va="top" if truncated else "bottom",
            fontsize=ANNOTATION_FONTSIZE,
            bbox=annotation_box,
        )
    if include_5z_outline:
        five_z = ordered_estimates["midpoint_5z_estimate"].to_numpy()
        ax.bar(
            [len(VERSION_ORDER) - 1],
            [five_z[-1]],
            width=0.70,
            facecolor="none",
            edgecolor="black",
            linewidth=2,
            linestyle="--",
            zorder=4,
        )
        ax.annotate(
            f"{five_z[-1] / 1_000_000:.1f} M",
            (len(VERSION_ORDER) - 1, five_z[-1]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="black",
            fontsize=ANNOTATION_FONTSIZE,
            fontweight="bold",
        )
    _style_version_axis(ax, ylabel=f"Projected {UNIT_LABELS[count_unit].lower()}")
    ax.set_title(
        (
            "Cumulative MorphSeq corpus by perturbation type "
            "with v5 five-z projection"
            if include_5z_outline
            else "Cumulative MorphSeq corpus size by perturbation type"
        ),
        fontsize=TITLE_FONTSIZE,
    )
    ax.legend(
        title="Perturbation type",
        frameon=False,
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=3,
        fontsize=ANNOTATION_FONTSIZE,
        title_fontsize=LABEL_FONTSIZE,
        columnspacing=1.2,
    )
    if y_axis_max is not None:
        ax.set_ylim(0, y_axis_max)
    else:
        ymax = (
            float(ordered_estimates["midpoint_5z_estimate"].iloc[-1])
            if include_5z_outline
            else float(totals.max())
        )
        ax.set_ylim(0, ymax * 1.22)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    paths = _save_figure(
        fig,
        output_dir,
        (
            "01c_cumulative_corpus_bar_by_perturbation_with_5z_outline"
            if include_5z_outline
            else (
                "01d_cumulative_corpus_bar_by_perturbation_ymax_200k"
                if y_axis_max is not None
                else "01b_cumulative_corpus_bar_by_perturbation"
            )
        ),
    )
    plt.close(fig)
    return paths


def _plot_range(
    ax: plt.Axes,
    estimates: pd.DataFrame,
    *,
    count_unit: str,
) -> None:
    ordered = estimates.set_index("version").reindex(VERSION_ORDER)
    x = np.arange(len(VERSION_ORDER))
    lower = ordered["lower_estimate"].to_numpy()
    midpoint = ordered["midpoint_estimate"].to_numpy()
    upper = ordered["upper_estimate"].to_numpy()
    ax.fill_between(
        x,
        lower,
        upper,
        color="#93C5FD",
        alpha=0.40,
        zorder=1,
    )
    ax.plot(
        x,
        midpoint,
        color=MEDIAN_COLOR,
        marker="o",
        markersize=8,
        linewidth=3,
        zorder=3,
    )
    _style_version_axis(ax, ylabel=f"Projected {UNIT_LABELS[count_unit].lower()}")


def plot_range_line(
    estimates: pd.DataFrame,
    *,
    count_unit: str,
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _plot_range(ax, estimates, count_unit=count_unit)
    ax.set_title(
        "Cumulative MorphSeq corpus: projected QC range",
        fontsize=TITLE_FONTSIZE,
    )
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "02_cumulative_corpus_qc_range")
    plt.close(fig)
    return paths


def plot_range_with_perturbations(
    estimates: pd.DataFrame,
    *,
    count_unit: str,
    output_dir: Path,
) -> list[Path]:
    ordered = estimates.set_index("version").reindex(VERSION_ORDER)
    x = np.arange(len(VERSION_ORDER))

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    _plot_range(ax, estimates, count_unit=count_unit)
    ax.set_title(
        "Cumulative MorphSeq corpus and perturbation diversity",
        fontsize=TITLE_FONTSIZE,
    )

    perturbation_ax = ax.twinx()
    perturbation_ax.plot(
        x,
        ordered["unique_perturbation_count"].to_numpy(),
        color=PERTURBATION_TREND_COLOR,
        marker="s",
        markersize=7,
        linewidth=2.7,
        zorder=4,
    )
    perturbation_ax.set_ylabel(
        "unique perturbations",
        color=PERTURBATION_TREND_COLOR,
        fontsize=LABEL_FONTSIZE,
    )
    perturbation_ax.tick_params(
        axis="y",
        colors=PERTURBATION_TREND_COLOR,
        labelsize=TICK_FONTSIZE,
    )
    perturbation_ax.spines["right"].set_visible(True)
    perturbation_ax.spines["right"].set_color(PERTURBATION_TREND_COLOR)
    perturbation_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    paths = _save_figure(
        fig,
        output_dir,
        "04_cumulative_corpus_qc_range_with_unique_perturbations",
    )
    plt.close(fig)
    return paths


def plot_range_with_z(
    estimates: pd.DataFrame,
    *,
    count_unit: str,
    z_slice_multiplier: float,
    output_dir: Path,
) -> list[Path]:
    if count_unit != "embryo_times":
        raise ValueError("The 5-z comparison is defined relative to embryo_times.")
    ordered = estimates.set_index("version").reindex(VERSION_ORDER)
    x = np.arange(len(VERSION_ORDER))
    midpoint = ordered["midpoint_estimate"].to_numpy()
    z_midpoint = ordered["midpoint_5z_estimate"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.6))
    _plot_range(ax, estimates, count_unit=count_unit)
    ax.plot(
        [x[-2], x[-1]],
        [midpoint[-2], z_midpoint[-1]],
        color=MEDIAN_COLOR,
        linestyle="--",
        linewidth=2.2,
        zorder=3,
    )
    ax.scatter(
        [x[-1]],
        [z_midpoint[-1]],
        color=MEDIAN_COLOR,
        marker="D",
        s=85,
        zorder=4,
    )
    ax.annotate(
        compact_count(z_midpoint[-1]),
        (x[-1], z_midpoint[-1]),
        xytext=(-4, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=MEDIAN_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
    )
    ax.set_ylabel("Projected image observations", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        "Cumulative MorphSeq corpus: effect of retaining five z-slices",
        fontsize=TITLE_FONTSIZE,
    )
    ax.margins(y=0.08)
    fig.tight_layout()
    paths = _save_figure(
        fig, output_dir, "03_cumulative_corpus_qc_range_with_5z"
    )
    plt.close(fig)
    return paths


def generate_version_figures(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    count_unit: str = "embryo_times",
    z_slice_multiplier: float = 5.0,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wells, rates = load_projection_inputs(data_dir)
    projection_rows = prepare_projection_rows(
        wells, rates, count_unit=count_unit
    )
    estimates, perturbation = build_version_estimates(
        projection_rows, z_slice_multiplier=z_slice_multiplier
    )
    estimates.to_csv(output_dir / "cumulative_corpus_version_estimates.csv", index=False)
    rates.to_csv(output_dir / "projection_rate_assumptions.csv", index=False)

    figure_paths = []
    figure_paths.extend(
        plot_solid_bars(estimates, count_unit=count_unit, output_dir=output_dir)
    )
    figure_paths.extend(
        plot_perturbation_bars(
            perturbation,
            estimates,
            count_unit=count_unit,
            output_dir=output_dir,
        )
    )
    figure_paths.extend(
        plot_perturbation_bars(
            perturbation,
            estimates,
            count_unit=count_unit,
            output_dir=output_dir,
            include_5z_outline=True,
        )
    )
    figure_paths.extend(
        plot_perturbation_bars(
            perturbation,
            estimates,
            count_unit=count_unit,
            output_dir=output_dir,
            y_axis_max=200_000,
        )
    )
    figure_paths.extend(
        plot_range_line(estimates, count_unit=count_unit, output_dir=output_dir)
    )
    figure_paths.extend(
        plot_range_with_z(
            estimates,
            count_unit=count_unit,
            z_slice_multiplier=z_slice_multiplier,
            output_dir=output_dir,
        )
    )
    figure_paths.extend(
        plot_range_with_perturbations(
            estimates,
            count_unit=count_unit,
            output_dir=output_dir,
        )
    )
    return {
        "version_estimates": estimates,
        "projection_rates": rates,
        "output_dir": output_dir,
        "figure_paths": figure_paths,
    }


def main() -> None:
    args = parse_args()
    result = generate_version_figures(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        count_unit=args.count_unit,
        z_slice_multiplier=args.z_slice_multiplier,
    )
    print(result["version_estimates"].to_string(index=False))
    print(f"\nOutputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
