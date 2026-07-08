#!/usr/bin/env python
"""Decile galleries for local focus candidate metrics in 20251125."""

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

ENTROPY_CUT = -0.55
LAP_RATIO_CUT = 1.5
N_DECILES = 10
N_PER_DECILE = 10

STATUS_COLORS = {
    "accepted": "#1a8820",
    "entropy_only_reject": "#b5651d",
    "sharpness_only_reject": "#1f77b4",
    "rejected_by_both": "#7b1fa2",
}

METRICS = [
    (
        "emb_iqr_local",
        "local embryo IQR",
        "20251125_emb_iqr_local_deciles_current_focus_status_benchmark.png",
        "focus_20251125_emb_iqr_local_decile_gallery_samples_benchmark.csv",
    ),
    (
        "norm_rel_entropy_local_annulus",
        "normalized local-annulus relative entropy",
        "20251125_norm_rel_entropy_local_annulus_deciles_current_focus_status_benchmark.png",
        "focus_20251125_norm_rel_entropy_local_annulus_decile_gallery_samples_benchmark.csv",
    ),
]


def current_focus_status(row: pd.Series) -> str:
    entropy_fail = bool(row["ff_rel_entropy"] < ENTROPY_CUT)
    sharp_fail = bool(row["ff_lap_abs_ratio"] < LAP_RATIO_CUT)
    if entropy_fail and sharp_fail:
        return "rejected_by_both"
    if entropy_fail:
        return "entropy_only_reject"
    if sharp_fail:
        return "sharpness_only_reject"
    return "accepted"


def load_df(metric: str) -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    required = [
        metric,
        "ff_rel_entropy",
        "ff_lap_abs_ratio",
        "image_path",
        "mask_path",
        "not_dead_snip",
    ]
    df = df[df["not_dead_snip"].astype(bool)].dropna(subset=required).copy()
    df["current_focus_status"] = df.apply(current_focus_status, axis=1)
    df["current_entropy_fail"] = df["ff_rel_entropy"] < ENTROPY_CUT
    return df


def assign_deciles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.sort_values(metric, ascending=True).reset_index(drop=True).copy()
    out["decile"] = (
        pd.qcut(
            out[metric].rank(method="first"),
            N_DECILES,
            labels=False,
            duplicates="drop",
        ).astype(int)
        + 1
    )
    return out


def sample_deciles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    with_deciles = assign_deciles(df, metric)
    rows = []
    for decile in sorted(with_deciles["decile"].unique()):
        grp = with_deciles[with_deciles["decile"] == decile].sort_values(metric)
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
        "well",
        "time_index",
        "frame_index",
        "snip_id",
        "image_id",
        "image_path",
        "mask_path",
        "predicted_stage_hpf",
        metric,
        "ff_rel_entropy",
        "ff_lap_abs_ratio",
        "emb_frac_gt245",
        "emb_iqr_local",
        "norm_rel_entropy_local_annulus",
        "current_entropy_fail",
        "current_focus_status",
    ]
    keep_unique = list(dict.fromkeys(c for c in keep if c in manifest.columns))
    return manifest[keep_unique]


def read_crop(image_path: object, mask_path: object, pad: int = 60) -> tuple[np.ndarray | None, np.ndarray | None]:
    ip = Path(str(image_path))
    mp = Path(str(mask_path))
    if not ip.exists() or not mp.exists():
        return None, None

    img = np.array(Image.open(ip).convert("L"))
    mask = np.array(Image.open(mp)).astype(bool)
    if mask.shape != img.shape or not mask.any():
        return img, None

    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def show_crop(ax: plt.Axes, crop: np.ndarray, mask: np.ndarray | None) -> None:
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
    if mask is not None and mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=["#00ff33"], linewidths=0.5)


def plot_gallery(manifest: pd.DataFrame, metric: str, label: str, out_path: Path) -> None:
    nrows = N_DECILES
    ncols = N_PER_DECILE
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.05), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for r, decile in enumerate(sorted(manifest["decile"].unique())):
        grp = manifest[manifest["decile"] == decile].sort_values("sample_rank_within_decile")
        for c, (_, row) in enumerate(grp.head(ncols).iterrows()):
            ax = axes[r, c]
            crop, mask = read_crop(row["image_path"], row["mask_path"])
            if crop is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            else:
                show_crop(ax, crop, mask)
            ax.axis("on")

            status = str(row["current_focus_status"])
            for spine in ax.spines.values():
                spine.set_color(STATUS_COLORS.get(status, "#444444"))
                spine.set_linewidth(2.2)

            entropy_mark = "Efail" if bool(row["current_entropy_fail"]) else "Epass"
            ax.set_title(
                f"{row.get('well_id', row.get('well', ''))} t{int(row['time_index'])}\n"
                f"{metric}={row[metric]:.2f} {entropy_mark}\n"
                f"glob={row['ff_rel_entropy']:.2f} hpf={row['predicted_stage_hpf']:.0f}",
                fontsize=5.5,
            )
        axes[r, 0].set_ylabel(f"D{decile}", fontsize=8)

    handles = [
        Patch(facecolor=STATUS_COLORS[k], edgecolor="none", label=k)
        for k in ["accepted", "entropy_only_reject", "sharpness_only_reject", "rejected_by_both"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=4, frameon=False)
    fig.suptitle(
        f"20251125 benchmark: {label} deciles, low to high; borders show current focus status",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_deciles(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    with_deciles = assign_deciles(df, metric)
    rows = []
    for decile, grp in with_deciles.groupby("decile", observed=True):
        status_counts = grp["current_focus_status"].value_counts()
        rows.append(
            {
                "metric": metric,
                "decile": int(decile),
                "n": int(len(grp)),
                "metric_min": float(grp[metric].min()),
                "metric_median": float(grp[metric].median()),
                "metric_max": float(grp[metric].max()),
                "current_entropy_fail_frac": float(grp["current_entropy_fail"].mean()),
                "accepted_frac": float(status_counts.get("accepted", 0) / len(grp)),
                "entropy_only_reject_frac": float(status_counts.get("entropy_only_reject", 0) / len(grp)),
                "sharpness_only_reject_frac": float(status_counts.get("sharpness_only_reject", 0) / len(grp)),
                "rejected_by_both_frac": float(status_counts.get("rejected_by_both", 0) / len(grp)),
                "median_hpf": float(grp["predicted_stage_hpf"].median()),
                "median_frac_gt245": float(grp["emb_frac_gt245"].median()),
                "median_ff_rel_entropy": float(grp["ff_rel_entropy"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    summary_rows = []
    for metric, label, figure_name, manifest_name in METRICS:
        df = load_df(metric)
        manifest = sample_deciles(df, metric)
        manifest_path = TABLES / manifest_name
        figure_path = FIGURES / figure_name
        manifest.to_csv(manifest_path, index=False)
        plot_gallery(manifest, metric, label, figure_path)
        summary_rows.append(summarize_deciles(df, metric))
        print(f"Saved {label} manifest -> {manifest_path}")
        print(f"Saved {label} gallery -> {figure_path}")

    summary = pd.concat(summary_rows, ignore_index=True)
    summary_path = TABLES / "focus_20251125_local_metric_decile_summary_benchmark.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved decile summary -> {summary_path}")


if __name__ == "__main__":
    main()
