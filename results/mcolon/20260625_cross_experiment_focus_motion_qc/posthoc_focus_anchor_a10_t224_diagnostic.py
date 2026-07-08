#!/usr/bin/env python
"""Focused diagnostic for A10 t224 as a true out-of-focus anchor."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "focus_candidate_metrics"
FIGURES.mkdir(parents=True, exist_ok=True)

BENCHMARK = TABLES / "focus_candidate_metrics_20251125_5min_benchmark.csv"
ANCHORS = TABLES / "focus_candidate_metrics_20251125.csv"
FOCUS = TABLES / "focus_metrics_by_experiment.csv"

OUT_FIG = FIGURES / "20251125_A10_t224_focus_anchor_diagnostic.png"
OUT_TRACE = FIGURES / "20251125_A10_current_focus_metrics_over_time.png"
OUT_TABLE = TABLES / "focus_A10_t224_anchor_metric_percentiles.csv"

ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5


def bbox(mask: np.ndarray, pad: int = 80) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    return (
        max(int(ys.min()) - pad, 0),
        min(int(ys.max()) + pad + 1, mask.shape[0]),
        max(int(xs.min()) - pad, 0),
        min(int(xs.max()) + pad + 1, mask.shape[1]),
    )


def robust_u8(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = img[valid & np.isfinite(img)]
    if vals.size == 0:
        vals = img[np.isfinite(img)]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)


def load_crop(row: pd.Series) -> dict:
    img = np.array(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
    mask = np.array(Image.open(row["mask_path"])).astype(bool)
    y0, y1, x0, x1 = bbox(mask)
    img_c = img[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]
    dil_outer = ndi.binary_dilation(mask_c, iterations=28)
    dil_inner = ndi.binary_dilation(mask_c, iterations=5)
    annulus = dil_outer & ~dil_inner
    valid = mask_c | annulus
    img_norm = robust_u8(img_c, valid)
    sobel = filters.sobel(img_norm.astype(np.float32) / 255.0)
    log = np.abs(ndi.gaussian_laplace(img_norm.astype(np.float32) / 255.0, sigma=1.0))
    return {
        "img": img_c,
        "mask": mask_c,
        "annulus": annulus,
        "img_norm": img_norm,
        "sobel": sobel,
        "log": log,
    }


def current_status(row: pd.Series) -> str:
    entropy_fail = bool(row["ff_rel_entropy"] < ENTROPY_CUT)
    sharp_fail = bool(row["ff_lap_abs_ratio"] < LAP_RATIO_CUT)
    if entropy_fail and sharp_fail:
        return "both fail"
    if entropy_fail:
        return "entropy only"
    if sharp_fail:
        return "sharpness only"
    return "pass"


def anchor_rows() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(BENCHMARK), pd.read_csv(ANCHORS)], ignore_index=True)
    df = df.drop_duplicates(["well_id", "time_index"], keep="last")
    picks = [
        ("A10", 37),
        ("A10", 173),
        ("A10", 224),
        ("A10", 226),
    ]
    rows = []
    for well, time_index in picks:
        hit = df[(df["well_id"] == well) & (df["time_index"].astype(int) == time_index)]
        if hit.empty:
            continue
        rows.append(hit.iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["current_status"] = out.apply(current_status, axis=1)
    return out


def write_percentiles(rows: pd.DataFrame) -> None:
    ref = pd.read_csv(BENCHMARK)
    ref = ref[ref["not_dead_snip"].astype(bool)].copy()
    metrics = [
        "ff_rel_entropy",
        "ff_lap_abs_ratio",
        "emb_iqr_local",
        "norm_rel_entropy_local_annulus",
        "sobel_ratio_norm",
        "log_ratio_norm",
        "emb_frac_gt245",
    ]
    records = []
    for _, row in rows.iterrows():
        for metric in metrics:
            pct = float((ref[metric] <= row[metric]).mean() * 100)
            records.append(
                {
                    "well_id": row["well_id"],
                    "time_index": int(row["time_index"]),
                    "current_status": row["current_status"],
                    "metric": metric,
                    "value": float(row[metric]),
                    "percentile_low_to_high": pct,
                }
            )
    pd.DataFrame(records).to_csv(OUT_TABLE, index=False)


def plot_anchor_panel(rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3.2 * len(rows)), squeeze=False)
    for r, (_, row) in enumerate(rows.iterrows()):
        data = load_crop(row)
        img = data["img"]
        mask = data["mask"]
        annulus = data["annulus"]

        panels = [
            ("raw crop", img, "gray", 0, 255),
            ("local contrast norm", data["img_norm"], "gray", 0, 255),
            ("Sobel", data["sobel"], "magma", None, None),
            ("LoG abs", data["log"], "magma", None, None),
        ]
        for c, (title, arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)

        ax = axes[r, 4]
        emb_px = img[mask]
        bg_px = img[annulus]
        ax.hist(bg_px, bins=50, range=(0, 255), density=True, alpha=0.45, label="annulus")
        ax.hist(emb_px, bins=50, range=(0, 255), density=True, alpha=0.55, label="embryo")
        ax.axvline(245, color="black", linestyle="--", linewidth=1)
        if r == 0:
            ax.set_title("intensity histogram", fontsize=10)
        ax.legend(fontsize=7, frameon=False)

        label = (
            f"{row['well_id']} t{int(row['time_index'])} {row['current_status']}\n"
            f"hpf={row['predicted_stage_hpf']:.1f} glob_ent={row['ff_rel_entropy']:.2f} "
            f"old_lap={row['ff_lap_abs_ratio']:.2f}\n"
            f"IQR={row['emb_iqr_local']:.0f} local_ent={row['norm_rel_entropy_local_annulus']:.2f} "
            f"Sobel={row['sobel_ratio_norm']:.2f} LoG={row['log_ratio_norm']:.2f} "
            f"sat245={row['emb_frac_gt245']:.2f}"
        )
        axes[r, 0].set_ylabel(label, fontsize=8)

    fig.suptitle("A10 t224 focus anchor: why old Laplacian ratio is too forgiving", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)


def plot_time_trace() -> None:
    df = pd.read_csv(FOCUS)
    df = df[(df["experiment_id"].astype(str) == "20251125") & (df["well_id"].astype(str) == "A10")].copy()
    df = df.sort_values("time_index")

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    series = [
        ("ff_rel_entropy", ENTROPY_CUT, "current global rel entropy"),
        ("ff_lap_abs_ratio", LAP_RATIO_CUT, "old Laplacian abs ratio"),
        ("ff_iqr_emb", None, "embryo IQR from current table"),
        ("ff_mean_emb", None, "embryo mean intensity"),
    ]
    for ax, (col, cut, label) in zip(axes, series, strict=True):
        ax.plot(df["time_index"], df[col], marker=".", linewidth=1)
        if cut is not None:
            ax.axhline(cut, color="black", linestyle="--", linewidth=1)
        ax.axvline(224, color="#d62728", linestyle="-", linewidth=1.5)
        ax.axvline(226, color="#2ca02c", linestyle="-", linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time_index")
    fig.suptitle("20251125 A10 current focus metrics over time; red=t224, green=t226")
    fig.tight_layout()
    fig.savefig(OUT_TRACE, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = anchor_rows()
    write_percentiles(rows)
    plot_anchor_panel(rows)
    plot_time_trace()
    print(f"Saved anchor diagnostic -> {OUT_FIG}")
    print(f"Saved A10 time trace -> {OUT_TRACE}")
    print(f"Saved percentile table -> {OUT_TABLE}")


if __name__ == "__main__":
    main()
