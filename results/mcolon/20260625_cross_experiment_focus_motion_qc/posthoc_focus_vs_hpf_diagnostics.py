#!/usr/bin/env python
"""Stage/HPF diagnostics for current and candidate focus metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "focus_candidate_metrics"
FIGURES.mkdir(parents=True, exist_ok=True)

FOCUS_CSV = TABLES / "focus_metrics_by_experiment.csv"
CANDIDATE_CSV = TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv"

ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5


def focus_status(df: pd.DataFrame) -> pd.Series:
    ent = df["ff_rel_entropy"] < ENTROPY_CUT
    lap = df["ff_lap_abs_ratio"] < LAP_RATIO_CUT
    return np.select(
        [ent & lap, ent & ~lap, ~ent & lap],
        ["rejected_by_both", "entropy_only_reject", "sharpness_only_reject"],
        default="accepted",
    )


def load_focus() -> pd.DataFrame:
    df = pd.read_csv(FOCUS_CSV)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(subset=["predicted_stage_hpf", "ff_rel_entropy", "ff_lap_abs_ratio"]).copy()
    df["focus_status"] = focus_status(df)
    df["hpf_bin"] = (np.floor(df["predicted_stage_hpf"] / 5) * 5).astype(int)
    return df


def plot_current_reject_vs_hpf(df: pd.DataFrame) -> None:
    statuses = ["entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    colors = {
        "entropy_only_reject": "#b5651d",
        "sharpness_only_reject": "#1f77b4",
        "rejected_by_both": "#7b1fa2",
    }
    experiments = sorted(df["experiment_id"].astype(str).unique())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10, 3.3 * len(experiments)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, exp in zip(axes, experiments):
        sub = df[df["experiment_id"].astype(str) == exp].copy()
        bins = sorted(sub["hpf_bin"].unique())
        bottom = np.zeros(len(bins), dtype=float)
        for status in statuses:
            vals = []
            for b in bins:
                g = sub[sub["hpf_bin"] == b]
                vals.append(float((g["focus_status"] == status).mean() * 100) if len(g) else np.nan)
            vals = np.asarray(vals)
            ax.bar(bins, vals, bottom=bottom, width=4.5, color=colors[status], label=status, align="edge")
            bottom += np.nan_to_num(vals)
        ax.set_title(f"{exp}: current focus reject categories vs predicted HPF")
        ax.set_ylabel("% not-dead snips")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("predicted_stage_hpf, 5 hpf bins")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES / "current_focus_reject_categories_vs_predicted_hpf_all_experiments.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_current_metric_vs_hpf(df: pd.DataFrame) -> None:
    experiments = sorted(df["experiment_id"].astype(str).unique())
    fig, axes = plt.subplots(len(experiments), 2, figsize=(12, 3.4 * len(experiments)), sharex=True)
    axes = np.atleast_2d(axes)
    for r, exp in enumerate(experiments):
        sub = df[df["experiment_id"].astype(str) == exp].copy()
        grouped = sub.groupby("hpf_bin", observed=True)
        med = grouped[["ff_rel_entropy", "ff_lap_abs_ratio"]].median()
        q25 = grouped[["ff_rel_entropy", "ff_lap_abs_ratio"]].quantile(0.25)
        q75 = grouped[["ff_rel_entropy", "ff_lap_abs_ratio"]].quantile(0.75)
        x = med.index.to_numpy()
        for c, metric in enumerate(["ff_rel_entropy", "ff_lap_abs_ratio"]):
            ax = axes[r, c]
            ax.plot(x, med[metric], color="black", lw=1.4, label="median")
            ax.fill_between(x, q25[metric], q75[metric], color="#999999", alpha=0.25, label="IQR")
            if metric == "ff_rel_entropy":
                ax.axhline(ENTROPY_CUT, color="red", ls="--", lw=1)
            else:
                ax.axhline(LAP_RATIO_CUT, color="red", ls="--", lw=1)
            ax.set_title(f"{exp}: {metric}")
            ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("predicted_stage_hpf, 5 hpf bins")
    axes[-1, 1].set_xlabel("predicted_stage_hpf, 5 hpf bins")
    fig.tight_layout()
    out = FIGURES / "current_focus_metric_medians_vs_predicted_hpf_all_experiments.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_candidate_vs_hpf() -> None:
    if not CANDIDATE_CSV.exists():
        return
    df = pd.read_csv(CANDIDATE_CSV)
    df = df.dropna(subset=["predicted_stage_hpf"]).copy()
    df["hpf_bin"] = (np.floor(df["predicted_stage_hpf"] / 5) * 5).astype(int)
    metrics = [
        ("ff_rel_entropy", "current global rel entropy"),
        ("norm_rel_entropy_local_annulus", "normalized local-annulus rel entropy"),
        ("log_ratio_norm", "normalized LoG ratio"),
        ("sobel_ratio_norm", "normalized Sobel ratio"),
        ("emb_frac_gt245", "embryo fraction >=245"),
        ("emb_iqr_local", "embryo IQR"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axes = axes.ravel()
    grouped = df.groupby("hpf_bin", observed=True)
    for ax, (metric, label) in zip(axes, metrics):
        med = grouped[metric].median()
        q25 = grouped[metric].quantile(0.25)
        q75 = grouped[metric].quantile(0.75)
        x = med.index.to_numpy()
        ax.scatter(df["predicted_stage_hpf"], df[metric], s=8, alpha=0.18, color="#555555", linewidths=0)
        ax.plot(x, med, color="black", lw=1.4)
        ax.fill_between(x, q25, q75, color="#999999", alpha=0.25)
        if metric == "ff_rel_entropy":
            ax.axhline(ENTROPY_CUT, color="red", ls="--", lw=1)
        ax.set_title(label)
        ax.grid(alpha=0.25)
    axes[-2].set_xlabel("predicted_stage_hpf")
    axes[-1].set_xlabel("predicted_stage_hpf")
    fig.suptitle("20251125 5-minute benchmark candidate focus metrics vs predicted HPF", y=0.995)
    fig.tight_layout()
    out = FIGURES / "20251125_candidate_focus_metrics_vs_predicted_hpf_benchmark.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    focus = load_focus()
    plot_current_reject_vs_hpf(focus)
    plot_current_metric_vs_hpf(focus)
    plot_candidate_vs_hpf()
    print(f"Saved HPF diagnostics -> {FIGURES}")


if __name__ == "__main__":
    main()
