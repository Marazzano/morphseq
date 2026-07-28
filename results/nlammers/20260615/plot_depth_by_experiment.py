"""Box plots of per-embryo sequencing depth and sheath-cell abundance,
split by experiment (dataset) and collection timepoint.

Reads the cached atlas count table and writes three panels:
  1. total recovered cells per embryo
  2. raw notochordal sheath-cell counts per embryo
  3. sheath cells per 1,000 recovered cells

Timepoints are restricted to the planned collection times (24/30/36 hpf).
"""

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COUNT_TABLE = (
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    "results/nlammers/20260615/embryo_cell_counts_long.csv"
)
OUT_DIR = (
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    "results/nlammers/20260615/depth_by_experiment"
)
TIMEPOINTS = [24, 30, 36]


def load_embryo_table(path):
    """Collapse the long count table to one record per embryo."""
    totals = defaultdict(int)
    sheath = defaultdict(int)
    hpf = {}

    with open(path) as handle:
        for row in csv.DictReader(handle):
            try:
                count = int(float(row["cell_count"]))
                stage = float(row["hpf"])
            except (TypeError, ValueError):
                continue

            key = (row["dataset"], row["embryo"])
            totals[key] += count
            hpf[key] = stage
            if "sheath" in row["cell_group"].lower():
                sheath[key] += count

    records = []
    for key, total in totals.items():
        if total <= 0:
            continue
        dataset, _ = key
        records.append(
            {
                "dataset": dataset,
                "hpf": hpf[key],
                "total_cells": total,
                "sheath": sheath.get(key, 0),
                "sheath_per_1000": 1000 * sheath.get(key, 0) / total,
            }
        )
    return records


def grouped_boxplot(ax, records, value_key, datasets, title, ylabel, log_y):
    """One box per dataset x timepoint, datasets grouped along the x axis."""
    palette = plt.get_cmap("tab10")
    width = 0.8 / len(TIMEPOINTS)
    handles = []

    for t_idx, stage in enumerate(TIMEPOINTS):
        positions, series = [], []
        for d_idx, dataset in enumerate(datasets):
            values = [
                r[value_key]
                for r in records
                if r["dataset"] == dataset and r["hpf"] == stage
            ]
            # A dataset with no embryos at this stage gets an empty slot.
            if len(values) < 3:
                continue
            positions.append(d_idx + (t_idx - (len(TIMEPOINTS) - 1) / 2) * width)
            series.append(values)

        if not series:
            continue

        bp = ax.boxplot(
            series,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(palette(t_idx))
            patch.set_alpha(0.65)
        handles.append((bp["boxes"][0], f"{stage:.0f} hpf"))

        # Overlay individual embryos so small-n groups are honest.
        for pos, values in zip(positions, series):
            jitter = np.random.default_rng(0).normal(0, width * 0.06, len(values))
            ax.scatter(
                pos + jitter,
                values,
                s=7,
                color="black",
                alpha=0.35,
                zorder=3,
                linewidths=0,
            )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(
        [d.replace("v3.1.0 ", "v3.1.0\n") for d in datasets], fontsize=9
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if handles:
        ax.legend(*zip(*handles), fontsize=9, title="Timepoint", title_fontsize=9)


def annotate_counts(ax, records, datasets):
    """Print n per dataset/timepoint under the axis."""
    labels = []
    for dataset in datasets:
        per_stage = [
            sum(1 for r in records if r["dataset"] == dataset and r["hpf"] == s)
            for s in TIMEPOINTS
        ]
        labels.append("n=" + "/".join(str(c) for c in per_stage))
    ax.set_xlabel(
        "Experiment    (" + "   ".join(labels) + ")",
        fontsize=9,
    )


def main():
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    records = [r for r in load_embryo_table(COUNT_TABLE) if r["hpf"] in TIMEPOINTS]
    datasets = sorted({r["dataset"] for r in records})

    panels = [
        ("total_cells", "Total recovered cells per embryo", "Total cells", True),
        ("sheath", "Notochordal sheath cells per embryo (raw count)", "Sheath cells", False),
        (
            "sheath_per_1000",
            "Sheath cells per 1,000 recovered cells",
            "Cells per 1,000",
            False,
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 15))
    for ax, (key, title, ylabel, log_y) in zip(axes, panels):
        grouped_boxplot(ax, records, key, datasets, title, ylabel, log_y)
        annotate_counts(ax, records, datasets)

    fig.suptitle(
        "Per-embryo depth and sheath-cell yield by experiment and timepoint",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"depth_and_sheath_by_experiment.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print("Saved:", path)


if __name__ == "__main__":
    main()
