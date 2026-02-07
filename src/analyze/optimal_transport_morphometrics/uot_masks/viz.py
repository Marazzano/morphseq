"""Visualization helpers for UOT mask transport."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from analyze.utils.optimal_transport import Coupling


def plot_creation_destruction(
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    output_path: Optional[str] = None,
) -> plt.Figure:
    mass_created_hw = np.maximum(mass_created_hw, 0.0)
    mass_destroyed_hw = np.maximum(mass_destroyed_hw, 0.0)
    vmax_created = float(mass_created_hw.max()) if mass_created_hw.size else 0.0
    vmax_destroyed = float(mass_destroyed_hw.max()) if mass_destroyed_hw.size else 0.0
    vmax_created = max(vmax_created, 1e-12)
    vmax_destroyed = max(vmax_destroyed, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(mass_created_hw, cmap="magma", vmin=0.0, vmax=vmax_created)
    axes[0].set_title("Mass created")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mass_destroyed_hw, cmap="magma", vmin=0.0, vmax=vmax_destroyed)
    axes[1].set_title("Mass destroyed")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def _downsample_mask_to_shape(mask_hw: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask_hw.shape == target_shape:
        return mask_hw
    h, w = mask_hw.shape
    th, tw = target_shape
    if th == 0 or tw == 0:
        raise ValueError("Target shape must be non-zero.")
    if h % th == 0 and w % tw == 0:
        fh = h // th
        fw = w // tw
        trimmed = mask_hw[: th * fh, : tw * fw]
        reshaped = trimmed.reshape(th, fh, tw, fw)
        return (reshaped.max(axis=(1, 3)) > 0).astype(mask_hw.dtype)

    zoom_factors = (th / float(h), tw / float(w))
    resized = ndimage.zoom(mask_hw.astype(np.float32), zoom=zoom_factors, order=0)
    if resized.shape != target_shape:
        resized = resized[:th, :tw]
        if resized.shape != target_shape:
            pad_h = th - resized.shape[0]
            pad_w = tw - resized.shape[1]
            resized = np.pad(resized, ((0, pad_h), (0, pad_w)), mode="constant")
    return (resized > 0.5).astype(mask_hw.dtype)


def _expand_mass_map_to_full(
    mass_hw: np.ndarray,
    transform_meta: Optional[dict],
) -> Optional[np.ndarray]:
    if not transform_meta:
        return None
    preprocess = transform_meta.get("preprocess", {})
    orig_shape = preprocess.get("orig_shape")
    bbox = preprocess.get("bbox_y0y1x0x1")
    pad_hw = preprocess.get("pad_hw", (0, 0))
    downsample_factor = int(transform_meta.get("downsample_factor", 1))

    if orig_shape is None or bbox is None:
        return None

    mass_up = mass_hw
    if downsample_factor > 1:
        mass_up = np.repeat(mass_up, downsample_factor, axis=0)
        mass_up = np.repeat(mass_up, downsample_factor, axis=1)

    pad_h, pad_w = pad_hw
    if pad_h or pad_w:
        mass_up = mass_up[: mass_up.shape[0] - pad_h, : mass_up.shape[1] - pad_w]

    y0, y1, x0, x1 = bbox
    target_h = y1 - y0
    target_w = x1 - x0
    h = min(target_h, mass_up.shape[0])
    w = min(target_w, mass_up.shape[1])

    full = np.zeros(orig_shape, dtype=mass_up.dtype)
    full[y0 : y0 + h, x0 : x0 + w] = mass_up[:h, :w]
    return full


def plot_creation_destruction_overlay(
    src_mask_hw: np.ndarray,
    tgt_mask_hw: np.ndarray,
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    transform_meta: Optional[dict] = None,
    alpha: float = 0.6,
    output_path: Optional[str] = None,
) -> plt.Figure:
    created_plot = _expand_mass_map_to_full(mass_created_hw, transform_meta)
    if created_plot is None:
        created_plot = mass_created_hw

    destroyed_plot = _expand_mass_map_to_full(mass_destroyed_hw, transform_meta)
    if destroyed_plot is None:
        destroyed_plot = mass_destroyed_hw

    created_plot = np.maximum(created_plot, 0.0)
    destroyed_plot = np.maximum(destroyed_plot, 0.0)
    vmax_created = float(created_plot.max()) if created_plot.size else 0.0
    vmax_destroyed = float(destroyed_plot.max()) if destroyed_plot.size else 0.0
    vmax_created = max(vmax_created, 1e-12)
    vmax_destroyed = max(vmax_destroyed, 1e-12)

    if created_plot.shape != tgt_mask_hw.shape:
        tgt_plot = _downsample_mask_to_shape(tgt_mask_hw, created_plot.shape)
    else:
        tgt_plot = tgt_mask_hw

    if destroyed_plot.shape != src_mask_hw.shape:
        src_plot = _downsample_mask_to_shape(src_mask_hw, destroyed_plot.shape)
    else:
        src_plot = src_mask_hw

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].imshow(tgt_plot, cmap="gray")
    im0 = axes[0].imshow(created_plot, cmap="magma", alpha=alpha, vmin=0.0, vmax=vmax_created)
    axes[0].set_title("Mass created (overlay on target)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(src_plot, cmap="gray")
    im1 = axes[1].imshow(destroyed_plot, cmap="magma", alpha=alpha, vmin=0.0, vmax=vmax_destroyed)
    axes[1].set_title("Mass destroyed (overlay on source)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def plot_velocity_overlay(
    mask_hw: np.ndarray,
    velocity_field_yx_hw2: np.ndarray,
    stride: int = 12,
    output_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    ax.imshow(mask_hw, cmap="gray")

    h, w = mask_hw.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    v = velocity_field_yx_hw2[0:h:stride, 0:w:stride]
    vy = v[..., 0]
    vx = v[..., 1]

    ax.quiver(xx, yy, vx, vy, color="cyan", angles="xy", scale_units="xy", scale=1.0)
    ax.set_title("Velocity field (quiver)")
    ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def _sample_distances(
    coupling: Coupling,
    support_src_yx: np.ndarray,
    support_tgt_yx: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if sp is not None and sp.issparse(coupling):
        coo = coupling.tocoo()
        if coo.nnz == 0:
            return np.array([]), None
        if coo.nnz <= max_samples:
            src = support_src_yx[coo.row]
            tgt = support_tgt_yx[coo.col]
            dists = np.linalg.norm(src - tgt, axis=1)
            return dists, coo.data
        idx = rng.choice(coo.nnz, size=max_samples, replace=False, p=coo.data / coo.data.sum())
        src = support_src_yx[coo.row[idx]]
        tgt = support_tgt_yx[coo.col[idx]]
        dists = np.linalg.norm(src - tgt, axis=1)
        return dists, None

    coupling = np.asarray(coupling)
    row_sums = coupling.sum(axis=1)
    total = float(row_sums.sum())
    if total <= 0:
        return np.array([]), None
    p_rows = row_sums / total
    n_samples = min(max_samples, coupling.shape[0] * coupling.shape[1])
    dists = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        r = rng.choice(len(p_rows), p=p_rows)
        row = coupling[r]
        row_sum = row_sums[r]
        if row_sum <= 0:
            continue
        p_cols = row / row_sum
        c = rng.choice(coupling.shape[1], p=p_cols)
        dists[i] = np.linalg.norm(support_src_yx[r] - support_tgt_yx[c])
    return dists, None


def plot_transport_spectrum(
    coupling: Coupling,
    support_src_yx: np.ndarray,
    support_tgt_yx: np.ndarray,
    bins: int = 30,
    max_samples: int = 50000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    rng = np.random.default_rng(0)
    dists, weights = _sample_distances(coupling, support_src_yx, support_tgt_yx, max_samples, rng)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    if dists.size == 0:
        ax.text(0.5, 0.5, "No transport mass", ha="center", va="center")
    else:
        ax.hist(dists, bins=bins, weights=weights, density=True, color="steelblue", alpha=0.8)
    ax.set_title("Transport spectrum")
    ax.set_xlabel("Transport distance (px)")
    ax.set_ylabel("Density")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Phase 2: NaN contract enforcement + summary plots
# ---------------------------------------------------------------------------

def apply_nan_mask(field: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    """Apply NaN masking: non-support pixels become NaN.

    Prevents confusion between "no data" and "zero motion/mass".

    Args:
        field: (H, W) or (H, W, C) array
        support_mask: (H, W) boolean-like array, True = valid

    Returns:
        Copy of field with NaN outside support.
    """
    mask_bool = np.asarray(support_mask).astype(bool)
    out = np.array(field, dtype=np.float64)
    if out.ndim == 2:
        out[~mask_bool] = np.nan
    elif out.ndim == 3:
        out[~mask_bool, :] = np.nan
    else:
        raise ValueError(f"field must be 2D or 3D, got {out.ndim}D")
    return out


def _build_support_mask_from_result(result) -> np.ndarray:
    """Build a combined support mask from a UOTResult."""
    shape = result.mass_created_px.shape[:2]
    mask = np.zeros(shape, dtype=bool)
    mask |= (result.mass_created_px > 0)
    mask |= (result.mass_destroyed_px > 0)
    vel_mag = np.linalg.norm(result.velocity_px_per_frame_yx, axis=-1)
    mask |= (vel_mag > 0)
    return mask


def plot_uot_summary(
    result,
    output_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """4-panel UOT summary with NaN masking and numeric annotations.

    Panel 1: Support mask
    Panel 2: Velocity quiver on support
    Panel 3: Mass creation heatmap (NaN masked)
    Panel 4: Mass destruction heatmap (NaN masked)
    """
    support_mask = _build_support_mask_from_result(result)

    created = apply_nan_mask(result.mass_created_px, support_mask)
    destroyed = apply_nan_mask(result.mass_destroyed_px, support_mask)
    velocity = result.velocity_px_per_frame_yx
    vel_mag = np.linalg.norm(velocity, axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=14)

    # Panel 1: Support mask
    ax = axes[0, 0]
    ax.imshow(support_mask.astype(float), cmap="gray", vmin=0, vmax=1)
    n_support = int(support_mask.sum())
    ax.set_title(f"Support mask ({n_support} px)")
    ax.axis("off")

    # Panel 2: Velocity quiver
    ax = axes[0, 1]
    ax.imshow(support_mask.astype(float), cmap="gray", vmin=0, vmax=1, alpha=0.3)
    h, w = support_mask.shape
    stride = max(1, min(h, w) // 20)
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    vy = velocity[0:h:stride, 0:w:stride, 0]
    vx = velocity[0:h:stride, 0:w:stride, 1]
    ax.quiver(xx, yy, vx, vy, color="cyan", angles="xy", scale_units="xy", scale=1.0)
    max_vel = float(np.nanmax(vel_mag)) if vel_mag.size else 0.0
    mean_vel = float(np.nanmean(vel_mag[support_mask])) if support_mask.any() else 0.0
    ax.set_title(f"Velocity (max={max_vel:.2f}, mean={mean_vel:.2f} px/fr)")
    ax.axis("off")

    # Panel 3: Mass created
    ax = axes[1, 0]
    vmax_c = float(np.nanmax(created)) if np.any(np.isfinite(created)) else 1e-12
    vmax_c = max(vmax_c, 1e-12)
    im = ax.imshow(created, cmap="magma", vmin=0, vmax=vmax_c)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    total_created = float(np.nansum(created))
    ax.set_title(f"Mass created (total={total_created:.4f})")
    ax.axis("off")

    # Panel 4: Mass destroyed
    ax = axes[1, 1]
    vmax_d = float(np.nanmax(destroyed)) if np.any(np.isfinite(destroyed)) else 1e-12
    vmax_d = max(vmax_d, 1e-12)
    im = ax.imshow(destroyed, cmap="magma", vmin=0, vmax=vmax_d)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    total_destroyed = float(np.nansum(destroyed))
    ax.set_title(f"Mass destroyed (total={total_destroyed:.4f})")
    ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def plot_velocity_histogram(
    result,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of velocity magnitudes on support pixels."""
    support_mask = _build_support_mask_from_result(result)
    vel_mag = np.linalg.norm(result.velocity_px_per_frame_yx, axis=-1)
    valid_mags = vel_mag[support_mask]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    if valid_mags.size == 0:
        ax.text(0.5, 0.5, "No velocity data", ha="center", va="center")
    else:
        ax.hist(valid_mags, bins=30, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axvline(float(np.mean(valid_mags)), color="red", linestyle="--", label=f"mean={np.mean(valid_mags):.2f}")
        ax.legend()
    ax.set_title("Velocity magnitude distribution")
    ax.set_xlabel("Velocity (px/frame)")
    ax.set_ylabel("Count")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def write_diagnostics_json(result, output_path: str):
    """Write UOT result diagnostics to JSON."""
    import json

    diagnostics = result.diagnostics or {}
    metrics = diagnostics.get("metrics", {})

    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)

    out["cost"] = float(result.cost)

    support_mask = _build_support_mask_from_result(result)
    out["n_support_pixels"] = int(support_mask.sum())
    out["total_mass_created"] = float(result.mass_created_px.sum())
    out["total_mass_destroyed"] = float(result.mass_destroyed_px.sum())

    vel_mag = np.linalg.norm(result.velocity_px_per_frame_yx, axis=-1)
    valid_mags = vel_mag[support_mask]
    if valid_mags.size > 0:
        out["mean_velocity_px_per_frame"] = float(np.mean(valid_mags))
        out["max_velocity_px_per_frame"] = float(np.max(valid_mags))
    else:
        out["mean_velocity_px_per_frame"] = 0.0
        out["max_velocity_px_per_frame"] = 0.0

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
