"""Visualization helpers for UOT mask transport."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .config import Coupling


def plot_creation_destruction(
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    output_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(mass_created_hw, cmap="magma")
    axes[0].set_title("Mass created")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mass_destroyed_hw, cmap="magma")
    axes[1].set_title("Mass destroyed")
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
