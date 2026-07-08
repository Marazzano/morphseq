#!/usr/bin/env python
"""Plots for 20251125 candidate focus metrics from the timed benchmark sample."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "focus_candidate_metrics"
FIGURES.mkdir(parents=True, exist_ok=True)

INPUT_CSV = TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv"
SCATTER_OUT = FIGURES / "20251125_sobel_log_vs_entropy_scatter_benchmark.png"
GALLERY_OUT = FIGURES / "20251125_log_ratio_norm_deciles_global_entropy_status_benchmark.png"
MANIFEST_OUT = TABLES / "focus_20251125_log_ratio_norm_decile_gallery_samples_benchmark.csv"

ENTROPY_CUT = -0.55
N_DECILES = 10
N_PER_DECILE = 10

STATUS_COLORS = {
    "accepted": "#1a8820",
    "entropy_only_reject": "#b5651d",
    "sharpness_only_reject": "#1f77b4",
    "rejected_by_both": "#7b1fa2",
}


def global_status(row: pd.Series) -> str:
    entropy_fail = bool(row["ff_rel_entropy"] < ENTROPY_CUT)
    sharp_fail = bool(row["ff_lap_abs_ratio"] < 1.5)
    if entropy_fail and sharp_fail:
        return "rejected_by_both"
    if entropy_fail:
        return "entropy_only_reject"
    if sharp_fail:
        return "sharpness_only_reject"
    return "accepted"


def load_df() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(
        subset=[
            "ff_rel_entropy",
            "norm_rel_entropy_local_annulus",
            "sobel_ratio_norm",
            "log_ratio_norm",
            "image_path",
        ]
    ).copy()
    df["global_entropy_status"] = df.apply(global_status, axis=1)
    return df


def plot_scatter(df: pd.DataFrame) -> None:
    y_metrics = [
        ("sobel_ratio_norm", "normalized Sobel ratio"),
        ("log_ratio_norm", "normalized LoG ratio"),
    ]
    x_metrics = [
        ("ff_rel_entropy", "current global relative entropy"),
        ("norm_rel_entropy_local_annulus", "normalized local-annulus relative entropy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    for r, (y_col, y_label) in enumerate(y_metrics):
        for c, (x_col, x_label) in enumerate(x_metrics):
            ax = axes[r, c]
            for status, grp in df.groupby("global_entropy_status", observed=True):
                ax.scatter(
                    grp[x_col],
                    grp[y_col],
                    s=12,
                    alpha=0.45,
                    color=STATUS_COLORS.get(status, "#555555"),
                    label=status,
                    linewidths=0,
                )
            if x_col == "ff_rel_entropy":
                ax.axvline(ENTROPY_CUT, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.25)

    handles = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="none", label=k)
        for k in ["accepted", "entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle("20251125 benchmark: Sobel/LoG structure ratios vs entropy metrics", y=0.965)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(SCATTER_OUT, dpi=180)
    plt.close(fig)


def assign_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("log_ratio_norm", ascending=True).reset_index(drop=True).copy()
    out["decile"] = pd.qcut(
        out["log_ratio_norm"].rank(method="first"),
        N_DECILES,
        labels=False,
        duplicates="drop",
    ).astype(int) + 1
    return out


def sample_deciles(df: pd.DataFrame) -> pd.DataFrame:
    with_deciles = assign_deciles(df)
    rows = []
    for decile in sorted(with_deciles["decile"].unique()):
        grp = with_deciles[with_deciles["decile"] == decile].sort_values("log_ratio_norm")
        if len(grp) <= N_PER_DECILE:
            picks = grp
        else:
            picks = grp.iloc[np.linspace(0, len(grp) - 1, N_PER_DECILE).astype(int)]
        picks = picks.copy()
        picks["sample_rank_within_decile"] = np.arange(1, len(picks) + 1)
        rows.append(picks)
    manifest = pd.concat(rows, ignore_index=True)
    keep = [
        "experiment_id",
        "decile",
        "sample_rank_within_decile",
        "well_id",
        "time_index",
        "snip_id",
        "image_id",
        "image_path",
        "mask_path",
        "log_ratio_norm",
        "sobel_ratio_norm",
        "ff_rel_entropy",
        "norm_rel_entropy_local_annulus",
        "global_entropy_status",
    ]
    manifest = manifest[[c for c in keep if c in manifest.columns]]
    manifest.to_csv(MANIFEST_OUT, index=False)
    return manifest


def read_image(path: object) -> np.ndarray | None:
    p = Path(str(path))
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("L"))


def plot_decile_gallery(manifest: pd.DataFrame) -> None:
    nrows = N_DECILES
    ncols = N_PER_DECILE
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.7, nrows * 1.75), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for r, decile in enumerate(sorted(manifest["decile"].unique())):
        grp = manifest[manifest["decile"] == decile].sort_values("sample_rank_within_decile")
        for c, (_, row) in enumerate(grp.head(ncols).iterrows()):
            ax = axes[r, c]
            img = read_image(row["image_path"])
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            else:
                ax.imshow(img, cmap="gray")
            ax.axis("on")
            status = str(row["global_entropy_status"])
            color = STATUS_COLORS.get(status, "#444444")
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.2)
            ax.set_title(
                f"{row.get('well_id', '')} t{row.get('time_index', '')}\n"
                f"LoG={row['log_ratio_norm']:.2f} glob={row['ff_rel_entropy']:.2f}",
                fontsize=5.5,
            )
        axes[r, 0].set_ylabel(f"D{decile}", fontsize=8)

    handles = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="none", label=k)
        for k in ["accepted", "entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=4, frameon=False)
    fig.suptitle(
        "20251125 benchmark: normalized LoG-ratio deciles, colored by current global entropy status",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(GALLERY_OUT, dpi=170)
    plt.close(fig)


def main() -> None:
    df = load_df()
    plot_scatter(df)
    manifest = sample_deciles(df)
    plot_decile_gallery(manifest)
    print(f"Saved scatter -> {SCATTER_OUT}")
    print(f"Saved LoG decile manifest -> {MANIFEST_OUT}")
    print(f"Saved LoG decile gallery -> {GALLERY_OUT}")


if __name__ == "__main__":
    main()
