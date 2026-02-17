"""
Phase 0 Step 2: QC + Outlier Filtering.

Provides IQR-based outlier flagging on total_cost_C and the three
required QC deliverables (gate to proceed):
  QC-1: histogram/violin of total_cost_C (before/after filtering)
  QC-2: montage of top-N highest-cost samples with their cost maps
  QC-3: summary table of dropped samples

Gate: post-filter mean maps must not be dominated by alignment failures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from outlier_detection import OutlierDetectionConfig, detect_outliers

logger = logging.getLogger(__name__)


def compute_iqr_outliers(
    total_cost_C: np.ndarray,
    multiplier: float = 1.5,
) -> Tuple[np.ndarray, Dict]:
    """
    Flag outliers using IQR method on total_cost_C.

    Returns
    -------
    outlier_flag : (N,) bool
    stats : dict with q1, q3, iqr, lower, upper, n_outliers
    """
    q1 = float(np.percentile(total_cost_C, 25))
    q3 = float(np.percentile(total_cost_C, 75))
    iqr = q3 - q1

    result = detect_outliers(
        total_cost_C,
        OutlierDetectionConfig(method="iqr", iqr_multiplier=multiplier),
    )
    outlier_flag = result.outlier_flag

    stats = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": float(result.lower_bound),
        "upper_bound": float(result.upper_bound),
        "n_total": int(result.n_total),
        "n_outliers": int(result.n_outliers),
        "n_retained": int(result.n_total - result.n_outliers),
        "multiplier": multiplier,
        "method": "iqr",
    }
    logger.info(
        "IQR QC: %d/%d flagged (bounds=[%.4f, %.4f])",
        stats["n_outliers"],
        stats["n_total"],
        stats["lower_bound"],
        stats["upper_bound"],
    )
    return outlier_flag, stats


def plot_qc1_cost_histogram(
    total_cost_C: np.ndarray,
    outlier_flag: np.ndarray,
    stats: Dict,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    QC-1: Histogram/violin of total_cost_C showing before/after filtering.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full distribution with outlier bounds
    ax = axes[0]
    ax.hist(total_cost_C, bins=40, alpha=0.7, color="steelblue", edgecolor="k",
            label="All samples")
    ax.axvline(stats["lower_bound"], color="red", linestyle="--", linewidth=1.5,
               label=f"IQR bounds (×{stats['multiplier']:.1f})")
    ax.axvline(stats["upper_bound"], color="red", linestyle="--", linewidth=1.5)
    ax.axvline(stats["q1"], color="orange", linestyle=":", alpha=0.7, label="Q1/Q3")
    ax.axvline(stats["q3"], color="orange", linestyle=":", alpha=0.7)

    # Mark outliers
    outlier_costs = total_cost_C[outlier_flag]
    if len(outlier_costs) > 0:
        ax.hist(outlier_costs, bins=40, alpha=0.5, color="red", edgecolor="k",
                label=f"Outliers ({stats['n_outliers']})")
    ax.set_xlabel("Total OT Cost (C)")
    ax.set_ylabel("Count")
    ax.set_title(f"QC-1: Total Cost Distribution\n"
                 f"N={stats['n_total']}, outliers={stats['n_outliers']}")
    ax.legend(fontsize=8)

    # Right: retained only
    ax = axes[1]
    retained = total_cost_C[~outlier_flag]
    ax.hist(retained, bins=40, alpha=0.7, color="seagreen", edgecolor="k")
    ax.set_xlabel("Total OT Cost (C)")
    ax.set_ylabel("Count")
    ax.set_title(f"QC-1: After Filtering (N={stats['n_retained']})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved QC-1: {save_path}")
    return fig


def plot_qc2_worst_samples(
    X: np.ndarray,
    total_cost_C: np.ndarray,
    mask_ref: np.ndarray,
    sample_ids: List[str],
    top_n: int = 8,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    QC-2: Montage of top-N highest-cost samples with their cost density maps.

    X : (N, 512, 512, C) — channel 0 is cost_density
    """
    order = np.argsort(total_cost_C)[::-1][:top_n]

    ncols = min(4, top_n)
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, rank in enumerate(range(top_n)):
        if rank >= len(order):
            break
        i = order[rank]
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        cost_map = X[i, :, :, 0]
        # Mask outside embryo
        display = np.where(mask_ref.astype(bool), cost_map, np.nan)
        im = ax.imshow(display, cmap="hot", interpolation="nearest")
        ax.set_title(f"#{rank+1}: {sample_ids[i]}\nC={total_cost_C[i]:.4f}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused axes
    for idx in range(top_n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(f"QC-2: Top-{top_n} Highest Cost Samples", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved QC-2: {save_path}")
    return fig


def build_qc3_dropped_table(
    metadata_df: pd.DataFrame,
    outlier_flag: np.ndarray,
    total_cost_C: np.ndarray,
) -> pd.DataFrame:
    """
    QC-3: Summary table of dropped samples.

    Returns DataFrame with columns: sample_id, embryo_id, snip_id, total_cost_C, reason.
    """
    dropped = metadata_df[outlier_flag].copy()
    dropped["total_cost_C"] = total_cost_C[outlier_flag]
    dropped["reason"] = "IQR_outlier"

    # Sort by cost descending
    dropped = dropped.sort_values("total_cost_C", ascending=False).reset_index(drop=True)
    logger.info(f"QC-3: {len(dropped)} dropped samples")
    return dropped


def plot_qc_mean_maps(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    outlier_flag: np.ndarray,
    label_names: Dict[int, str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Post-filter mean cost density maps by class (gate visual check).

    Shows WT mean, mutant mean, and difference. This is the gate check:
    if these are dominated by alignment failures, do NOT proceed.
    """
    if label_names is None:
        label_names = {0: "WT", 1: "cep290"}

    valid = ~outlier_flag
    X_valid = X[valid]
    y_valid = y[valid]
    cost_ch = X_valid[:, :, :, 0]  # channel 0 = cost_density

    # Get proper extent for canonical grid display
    h, w = mask_ref.shape
    extent = [0, w, h, 0]  # [left, right, bottom, top] for upper origin
    origin = "upper"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for label_int, label_str in label_names.items():
        mask_class = y_valid == label_int
        if mask_class.sum() == 0:
            continue
        mean_map = np.mean(cost_ch[mask_class], axis=0)
        mean_map = np.where(mask_ref.astype(bool), mean_map, np.nan)

        ax_idx = label_int  # 0=WT, 1=mutant
        im = axes[ax_idx].imshow(
            mean_map,
            cmap="hot",
            aspect="equal",
            extent=extent,
            origin=origin,
            interpolation="nearest",
        )
        axes[ax_idx].set_title(f"Mean Cost: {label_str} (n={mask_class.sum()})")
        axes[ax_idx].set_xlabel("x (px)")
        axes[ax_idx].set_ylabel("y (px)")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)

    # Difference
    wt_mask = y_valid == 0
    mut_mask = y_valid == 1
    if wt_mask.sum() > 0 and mut_mask.sum() > 0:
        diff = np.mean(cost_ch[mut_mask], axis=0) - np.mean(cost_ch[wt_mask], axis=0)
        diff = np.where(mask_ref.astype(bool), diff, np.nan)
        vabs = np.nanmax(np.abs(diff)) if np.any(np.isfinite(diff)) else 1.0
        im = axes[2].imshow(
            diff,
            cmap="RdBu_r",
            vmin=-vabs,
            vmax=vabs,
            aspect="equal",
            extent=extent,
            origin=origin,
            interpolation="nearest",
        )
        axes[2].set_title(f"Difference ({label_names[1]} − {label_names[0]})")
        axes[2].set_xlabel("x (px)")
        axes[2].set_ylabel("y (px)")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Δ cost")

    fig.suptitle("QC Gate: Post-Filter Mean Cost Maps", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved QC mean maps: {save_path}")
    return fig


def run_qc_suite(
    X: np.ndarray,
    y: np.ndarray,
    total_cost_C: np.ndarray,
    mask_ref: np.ndarray,
    metadata_df: pd.DataFrame,
    sample_ids: List[str],
    out_dir: str | Path,
    iqr_multiplier: float = 1.5,
) -> Tuple[np.ndarray, Dict]:
    """
    Run the full QC suite (steps 2.1 + 2.2 from Phase 0 spec).

    Returns outlier_flag and stats dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 IQR filter
    outlier_flag, stats = compute_iqr_outliers(total_cost_C, multiplier=iqr_multiplier)

    # 2.2 QC deliverables
    plot_qc1_cost_histogram(total_cost_C, outlier_flag, stats,
                            save_path=out_dir / "qc1_cost_histogram.png")
    plot_qc2_worst_samples(X, total_cost_C, mask_ref, sample_ids,
                           save_path=out_dir / "qc2_worst_samples.png")

    dropped_df = build_qc3_dropped_table(metadata_df, outlier_flag, total_cost_C)
    dropped_df.to_csv(out_dir / "qc3_dropped_samples.csv", index=False)

    # Gate check: mean maps
    plot_qc_mean_maps(X, y, mask_ref, outlier_flag,
                      save_path=out_dir / "qc_gate_mean_maps.png")

    plt.close("all")
    logger.info(f"QC suite complete: {out_dir}")
    return outlier_flag, stats


__all__ = [
    "compute_iqr_outliers",
    "plot_qc1_cost_histogram",
    "plot_qc2_worst_samples",
    "build_qc3_dropped_table",
    "plot_qc_mean_maps",
    "run_qc_suite",
]
