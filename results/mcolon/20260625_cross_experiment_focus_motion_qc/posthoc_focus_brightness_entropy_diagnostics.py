#!/usr/bin/env python
"""Brightness/saturation diagnostics for current focus entropy failures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "focus_brightness_entropy_diagnostics"
FIGURES.mkdir(parents=True, exist_ok=True)

FOCUS_CSV = TABLES / "focus_metrics_by_experiment.csv"
CANDIDATE_20251125_CSV = TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv"

ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5

STATUS_COLORS = {
    "accepted": "#1a8820",
    "entropy_only_reject": "#b5651d",
    "sharpness_only_reject": "#1f77b4",
    "rejected_by_both": "#7b1fa2",
}


def status_from_thresholds(df: pd.DataFrame) -> pd.Series:
    entropy_fail = df["ff_rel_entropy"] < ENTROPY_CUT
    sharp_fail = df["ff_lap_abs_ratio"] < LAP_RATIO_CUT
    status = pd.Series("accepted", index=df.index, dtype=object)
    status.loc[entropy_fail & ~sharp_fail] = "entropy_only_reject"
    status.loc[~entropy_fail & sharp_fail] = "sharpness_only_reject"
    status.loc[entropy_fail & sharp_fail] = "rejected_by_both"
    return status


def load_focus() -> pd.DataFrame:
    df = pd.read_csv(FOCUS_CSV, low_memory=False)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    cols = [
        "experiment_id",
        "predicted_stage_hpf",
        "ff_rel_entropy",
        "ff_lap_abs_ratio",
        "ff_mean_emb",
        "ff_mean_bg",
        "ff_rel_mean",
        "ff_iqr_emb",
        "ff_iqr_bg",
        "ff_rel_iqr",
    ]
    df = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    df["current_focus_status"] = status_from_thresholds(df)
    df["entropy_fail"] = df["ff_rel_entropy"] < ENTROPY_CUT
    df["any_focus_reject"] = df["current_focus_status"] != "accepted"
    return df


def load_candidate_20251125() -> pd.DataFrame:
    if not CANDIDATE_20251125_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(CANDIDATE_20251125_CSV)
    df = df[df["not_dead_snip"].astype(bool)].copy()
    df = df.dropna(
        subset=[
            "ff_rel_entropy",
            "ff_lap_abs_ratio",
            "emb_frac_gt245",
            "emb_frac_gt250",
            "ff_mean_emb",
            "emb_iqr_local",
            "norm_rel_entropy_local_annulus",
        ]
    ).copy()
    df["current_focus_status"] = status_from_thresholds(df)
    df["entropy_fail"] = df["ff_rel_entropy"] < ENTROPY_CUT
    df["any_focus_reject"] = df["current_focus_status"] != "accepted"
    return df


def plot_mean_intensity_histograms(df: pd.DataFrame) -> None:
    experiments = sorted(df["experiment_id"].astype(str).unique())
    metrics = [
        ("ff_mean_emb", "embryo mean intensity"),
        ("ff_mean_bg", "local background mean intensity"),
        ("ff_rel_mean", "embryo - background mean"),
    ]
    fig, axes = plt.subplots(len(experiments), len(metrics), figsize=(15, 3.2 * len(experiments)), squeeze=False)

    for r, exp in enumerate(experiments):
        sub = df[df["experiment_id"].astype(str) == exp]
        for c, (metric, label) in enumerate(metrics):
            ax = axes[r, c]
            vals = sub[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            if metric == "ff_rel_mean":
                bins = np.linspace(np.nanpercentile(vals, 1), np.nanpercentile(vals, 99), 65)
            else:
                bins = np.linspace(0, 255, 65)
            for status, grp in sub.groupby("current_focus_status", observed=True):
                gvals = grp[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if gvals.empty:
                    continue
                ax.hist(
                    gvals,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.7,
                    color=STATUS_COLORS.get(status, "#555555"),
                    label=status,
                )
            ax.set_title(f"{exp}: {label}")
            ax.set_xlabel(metric)
            ax.set_ylabel("density")
            ax.grid(alpha=0.2)

    handles = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="none", label=k)
        for k in ["accepted", "entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle("Not-dead snips: mean intensity distributions by current focus status", y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES / "mean_intensity_histograms_by_experiment_status.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved mean intensity histograms -> {out}")


def plot_entropy_vs_brightness(df: pd.DataFrame) -> None:
    experiments = sorted(df["experiment_id"].astype(str).unique())
    x_metrics = [
        ("ff_mean_emb", "embryo mean intensity"),
        ("ff_mean_bg", "local background mean intensity"),
        ("ff_rel_mean", "embryo - background mean"),
        ("ff_iqr_emb", "embryo IQR"),
    ]
    fig, axes = plt.subplots(len(experiments), len(x_metrics), figsize=(17, 3.4 * len(experiments)), squeeze=False)

    for r, exp in enumerate(experiments):
        sub = df[df["experiment_id"].astype(str) == exp]
        for c, (metric, label) in enumerate(x_metrics):
            ax = axes[r, c]
            ax.hexbin(
                sub[metric],
                sub["ff_rel_entropy"],
                gridsize=45,
                cmap="Blues",
                mincnt=1,
                linewidths=0,
            )
            ax.axhline(ENTROPY_CUT, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel(label)
            ax.set_ylabel("ff_rel_entropy")
            ax.set_title(f"{exp}")
            ax.grid(alpha=0.2)

    fig.suptitle("Current relative entropy vs brightness/contrast summaries", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIGURES / "ff_rel_entropy_vs_brightness_hexbin.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved entropy-vs-brightness plot -> {out}")


def binned_failure_summary(df: pd.DataFrame, metric: str, n_bins: int = 12) -> pd.DataFrame:
    rows = []
    for exp, sub in df.groupby("experiment_id", observed=True):
        tmp = sub.dropna(subset=[metric]).copy()
        if tmp.empty:
            continue
        tmp["bin"] = pd.qcut(tmp[metric].rank(method="first"), n_bins, labels=False, duplicates="drop") + 1
        for b, grp in tmp.groupby("bin", observed=True):
            rows.append(
                {
                    "experiment_id": exp,
                    "metric": metric,
                    "bin": int(b),
                    "n": int(len(grp)),
                    "metric_min": float(grp[metric].min()),
                    "metric_median": float(grp[metric].median()),
                    "metric_max": float(grp[metric].max()),
                    "entropy_fail_frac": float(grp["entropy_fail"].mean()),
                    "any_focus_reject_frac": float(grp["any_focus_reject"].mean()),
                    "median_ff_rel_entropy": float(grp["ff_rel_entropy"].median()),
                    "median_hpf": float(grp["predicted_stage_hpf"].median()),
                }
            )
    return pd.DataFrame(rows)


def plot_failure_rate_bins(df: pd.DataFrame) -> None:
    metrics = [
        ("ff_mean_emb", "embryo mean intensity"),
        ("ff_mean_bg", "local background mean intensity"),
        ("ff_rel_mean", "embryo - background mean"),
        ("ff_iqr_emb", "embryo IQR"),
    ]
    summaries = [binned_failure_summary(df, metric) for metric, _ in metrics]
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = TABLES / "focus_brightness_binned_failure_summary.csv"
    summary.to_csv(summary_path, index=False)

    experiments = sorted(df["experiment_id"].astype(str).unique())
    fig, axes = plt.subplots(len(experiments), len(metrics), figsize=(17, 3.2 * len(experiments)), squeeze=False)
    for r, exp in enumerate(experiments):
        for c, (metric, label) in enumerate(metrics):
            ax = axes[r, c]
            sub = summary[(summary["experiment_id"].astype(str) == exp) & (summary["metric"] == metric)]
            ax.plot(sub["metric_median"], 100 * sub["entropy_fail_frac"], marker="o", label="entropy fail")
            ax.plot(sub["metric_median"], 100 * sub["any_focus_reject_frac"], marker="s", label="any reject")
            ax.set_xlabel(label)
            ax.set_ylabel("% rejected")
            ax.set_title(exp)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(frameon=False)

    fig.suptitle("Not-dead snips: current focus reject rate vs brightness/contrast bins", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIGURES / "focus_reject_rate_vs_brightness_bins.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved binned failure summary -> {summary_path}")
    print(f"Saved reject-rate bins -> {out}")


def plot_20251125_saturation(df: pd.DataFrame) -> None:
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for status, grp in df.groupby("current_focus_status", observed=True):
        axes[0, 0].hist(
            grp["emb_frac_gt245"],
            bins=np.linspace(0, 0.6, 61),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=STATUS_COLORS.get(status, "#555555"),
            label=status,
        )
        axes[0, 1].hist(
            grp["ff_mean_emb"],
            bins=np.linspace(0, 255, 61),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=STATUS_COLORS.get(status, "#555555"),
        )

    axes[0, 0].set_title("embryo fraction >=245")
    axes[0, 0].set_xlabel("emb_frac_gt245")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_title("embryo mean intensity")
    axes[0, 1].set_xlabel("ff_mean_emb")

    axes[0, 2].scatter(df["emb_frac_gt245"], df["ff_rel_entropy"], s=12, alpha=0.35, linewidths=0)
    axes[0, 2].axhline(ENTROPY_CUT, color="red", linestyle="--", linewidth=1)
    axes[0, 2].set_xlabel("emb_frac_gt245")
    axes[0, 2].set_ylabel("ff_rel_entropy")
    axes[0, 2].set_title("global entropy vs saturation")

    axes[1, 0].scatter(df["emb_frac_gt245"], df["norm_rel_entropy_local_annulus"], s=12, alpha=0.35, linewidths=0)
    axes[1, 0].set_xlabel("emb_frac_gt245")
    axes[1, 0].set_ylabel("norm local entropy delta")
    axes[1, 0].set_title("local entropy vs saturation")

    axes[1, 1].scatter(df["emb_frac_gt245"], df["emb_iqr_local"], s=12, alpha=0.35, linewidths=0)
    axes[1, 1].set_xlabel("emb_frac_gt245")
    axes[1, 1].set_ylabel("emb_iqr_local")
    axes[1, 1].set_title("IQR vs saturation")

    tmp = df.copy()
    tmp["sat_bin"] = pd.qcut(tmp["emb_frac_gt245"].rank(method="first"), 12, labels=False, duplicates="drop") + 1
    bins = (
        tmp.groupby("sat_bin", observed=True)
        .agg(
            sat_median=("emb_frac_gt245", "median"),
            entropy_fail_frac=("entropy_fail", "mean"),
            any_focus_reject_frac=("any_focus_reject", "mean"),
            n=("emb_frac_gt245", "size"),
        )
        .reset_index()
    )
    axes[1, 2].plot(bins["sat_median"], 100 * bins["entropy_fail_frac"], marker="o", label="entropy fail")
    axes[1, 2].plot(bins["sat_median"], 100 * bins["any_focus_reject_frac"], marker="s", label="any reject")
    axes[1, 2].set_xlabel("median emb_frac_gt245")
    axes[1, 2].set_ylabel("% rejected")
    axes[1, 2].set_title("reject rate vs saturation")
    axes[1, 2].legend(frameon=False)

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    fig.suptitle("20251125 benchmark: clipping/saturation relationship to current entropy failures")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / "20251125_saturation_vs_entropy_diagnostics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved 20251125 saturation diagnostics -> {out}")

    sat_summary = bins.copy()
    sat_summary.to_csv(TABLES / "focus_20251125_saturation_binned_failure_summary.csv", index=False)


def write_correlations(df: pd.DataFrame, cand: pd.DataFrame) -> None:
    metrics = ["ff_mean_emb", "ff_mean_bg", "ff_rel_mean", "ff_iqr_emb", "ff_iqr_bg", "ff_rel_iqr"]
    rows = []
    for exp, sub in df.groupby("experiment_id", observed=True):
        for metric in metrics:
            if metric not in sub:
                continue
            corr_entropy = sub[[metric, "ff_rel_entropy"]].rank().corr().iloc[0, 1]
            corr_reject = sub[[metric, "entropy_fail"]].corr().iloc[0, 1]
            rows.append(
                {
                    "experiment_id": exp,
                    "metric": metric,
                    "spearman_vs_ff_rel_entropy": float(corr_entropy),
                    "pearson_vs_entropy_fail_binary": float(corr_reject),
                    "n": int(sub[[metric, "ff_rel_entropy"]].dropna().shape[0]),
                }
            )

    if not cand.empty:
        for metric in [
            "emb_frac_gt245",
            "emb_frac_gt250",
            "emb_iqr_local",
            "norm_rel_entropy_local_annulus",
            "ff_mean_emb",
        ]:
            corr_entropy = cand[[metric, "ff_rel_entropy"]].rank().corr().iloc[0, 1]
            corr_reject = cand[[metric, "entropy_fail"]].corr().iloc[0, 1]
            rows.append(
                {
                    "experiment_id": "20251125_benchmark",
                    "metric": metric,
                    "spearman_vs_ff_rel_entropy": float(corr_entropy),
                    "pearson_vs_entropy_fail_binary": float(corr_reject),
                    "n": int(cand[[metric, "ff_rel_entropy"]].dropna().shape[0]),
                }
            )

    out = TABLES / "focus_brightness_entropy_correlations.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved brightness/entropy correlations -> {out}")


def main() -> None:
    focus = load_focus()
    cand = load_candidate_20251125()
    plot_mean_intensity_histograms(focus)
    plot_entropy_vs_brightness(focus)
    plot_failure_rate_bins(focus)
    plot_20251125_saturation(cand)
    write_correlations(focus, cand)


if __name__ == "__main__":
    main()
