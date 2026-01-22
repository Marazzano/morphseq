#!/usr/bin/env python3
"""
UOT Parameter Debugging and Sensitization Script

Sequential testing approach:
1. Test 1: Identity (must pass first)
2. Test 2: Non-overlapping circles (using params from Test 1)
3. Test 3: Shape change (using params from Tests 1+2)
4. Test 4: Combined transport + shape change

Records all metrics for posterity and generates visualizations.
Real embryo testing is beyond scope - synthetic only.

USAGE:
    python results/mcolon/20260121_uot-mvp/debug_uot_params.py --test 1
    python results/mcolon/20260121_uot-mvp/debug_uot_params.py --test all

OUTPUT:
    - Results CSV per test with all parameter combinations
    - Visualizations per parameter combination
    - Parameter sensitivity plots
    - recommended_params.json with viable ranges
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import argparse

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from src.analyze.optimal_transport_morphometrics.uot_masks import run_uot_pair
from src.analyze.utils.optimal_transport import (
    UOTConfig, UOTFramePair, UOTFrame, MassMode, POTBackend
)
import ot

# ==== CONSTANTS ====

# CANONICAL GRID - All masks are created DIRECTLY on this grid
# NOTE: With pair_frame enabled, these are now properly tracked through the pipeline
# rather than hard-coded in every function
CANONICAL_GRID_SHAPE = (256, 576)  # Height x Width in pixels
CANONICAL_UM_PER_PX = 7.8  # Micrometers per pixel
# Physical dimensions: 1996.8 μm (2.0 mm) × 4492.8 μm (4.5 mm)

# All test masks should be created on canonical grid
IMAGE_SHAPE = CANONICAL_GRID_SHAPE
UM_PER_PX = CANONICAL_UM_PER_PX
COORD_SCALE = 1.0 / max(CANONICAL_GRID_SHAPE)  # Scale based on max dimension

# Parameter grid
EPSILON_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
REG_M_GRID = [0.1, 1.0, 10.0, 100.0]

OUTPUT_DIR = Path("results/mcolon/20260121_uot-mvp/debug_params")


# ==== TEST CASE DEFINITIONS ====

def make_circle(shape: Tuple[int, int], center_yx: Tuple[int, int], radius: int) -> np.ndarray:
    """Create a circle mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def make_ellipse(
    shape: Tuple[int, int], center_yx: Tuple[int, int], radius_y: int, radius_x: int
) -> np.ndarray:
    """Create an ellipse mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = ((yy - cy) / float(radius_y)) ** 2 + ((xx - cx) / float(radius_x)) ** 2 <= 1.0
    return mask.astype(np.uint8)


def make_identity_test(shape: Tuple[int, int], radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Test 1: Identity (null test) - Circle to same circle."""
    cy, cx = shape[0] // 2, shape[1] // 2
    circle = make_circle(shape, (cy, cx), radius)
    return circle, circle


def make_nonoverlap_test(
    shape: Tuple[int, int], radius: int, separation: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Test 2: Non-overlapping circles (pure transport) - No shape change."""
    cy, cx = shape[0] // 2, shape[1] // 2
    src = make_circle(shape, (cy - separation // 2, cx), radius)
    tgt = make_circle(shape, (cy + separation // 2, cx), radius)
    return src, tgt


def make_shape_change_test(shape: Tuple[int, int], radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Test 3: Circle to oval (shape change) - Same centroid."""
    cy, cx = shape[0] // 2, shape[1] // 2
    circle = make_circle(shape, (cy, cx), radius)
    # Oval with same area as circle: π*r² = π*ry*rx, so ry*rx = r²
    # Let's make it 1.5x wider and proportionally shorter
    radius_x = int(radius * 1.5)
    radius_y = int(radius * radius / radius_x)
    oval = make_ellipse(shape, (cy, cx), radius_y, radius_x)
    return circle, oval


def make_combined_test(
    shape: Tuple[int, int], radius: int, shift: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Test 4: Circle to shifted oval (combined transport + shape change)."""
    cy, cx = shape[0] // 2, shape[1] // 2
    circle = make_circle(shape, (cy, cx), radius)
    radius_x = int(radius * 1.5)
    radius_y = int(radius * radius / radius_x)
    oval = make_ellipse(shape, (cy + shift, cx + shift), radius_y, radius_x)
    return circle, oval


# ==== DIAGNOSTICS ====

def compute_surface_metrics(mask: np.ndarray, um_per_px: float) -> Dict[str, float]:
    """Track surface area in physical units."""
    area_px = float(mask.sum())
    area_um2 = area_px * (um_per_px ** 2)

    # Simple perimeter estimation via edge detection
    from scipy import ndimage
    edges = ndimage.sobel(mask.astype(float))
    perimeter_px = float((edges > 0).sum())
    perimeter_um = perimeter_px * um_per_px

    return {
        "area_px": area_px,
        "area_um2": area_um2,
        "perimeter_px": perimeter_px,
        "perimeter_um": perimeter_um,
    }


def diagnose_cost_matrix(cost_matrix: np.ndarray, epsilon: float) -> Dict[str, float]:
    """Analyze cost matrix for numerical health."""
    return {
        "cost_min": float(cost_matrix.min()),
        "cost_max": float(cost_matrix.max()),
        "cost_mean": float(cost_matrix.mean()),
        "cost_std": float(cost_matrix.std()),
        "cost_ratio_to_epsilon": float(cost_matrix.mean() / epsilon),
    }


def diagnose_gibbs_kernel(cost_matrix: np.ndarray, epsilon: float) -> Dict[str, float]:
    """
    Check K = exp(-C/epsilon) for numerical health.
    The solver operates on K, not C. If K is all zeros or ones, solver fails silently.
    """
    K = np.exp(-cost_matrix / epsilon)
    K_nonzero = K[K > 0]

    return {
        "K_min": float(K.min()),
        "K_max": float(K.max()),
        "K_mean": float(K.mean()),
        "K_zeros": int((K == 0).sum()),           # Underflow indicator
        "K_ones": int((K == 1).sum()),            # Epsilon too large
        "K_dynamic_range": float(K.max() / max(K_nonzero.min(), 1e-20)) if len(K_nonzero) > 0 else 0.0,
        "K_healthy": bool(K.min() > 1e-10 and K.max() < 1 - 1e-10),  # Has variation
    }


def diagnose_coupling_sparsity(coupling: np.ndarray, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Biological motion is local - coupling should be sparse.
    Low sparsity = mass diffusion = epsilon too high.
    """
    coupling_arr = np.asarray(coupling)
    total_entries = coupling_arr.size
    nonzero_entries = int((coupling_arr > threshold).sum())
    sparsity = 1 - (nonzero_entries / total_entries)

    return {
        "sparsity": sparsity,              # Should be > 0.9 for biological transport
        "nonzero_entries": nonzero_entries,
        "is_sparse": bool(sparsity > 0.8),       # Warning threshold
    }


def compute_velocity_metrics(velocity_field_yx_hw2: np.ndarray) -> Dict[str, float]:
    """Compute velocity field statistics."""
    velocity_mag = np.sqrt(velocity_field_yx_hw2[..., 0]**2 + velocity_field_yx_hw2[..., 1]**2)
    velocity_nonzero = velocity_mag[velocity_mag > 0]

    return {
        "mean_velocity_px": float(velocity_mag.mean()),
        "max_velocity_px": float(velocity_mag.max()),
        "mean_nonzero_velocity_px": float(velocity_nonzero.mean()) if len(velocity_nonzero) > 0 else 0.0,
        "velocity_has_nan": bool(np.isnan(velocity_mag).any()),
    }


def compute_mass_metrics(
    result,
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    um_per_px: float
) -> Dict[str, float]:
    """Compute creation/destruction metrics."""
    metrics = result.diagnostics.get("metrics", {}) if result.diagnostics else {}
    backend = result.diagnostics.get("backend", {}) if result.diagnostics else {}

    m_src = float(backend.get("m_src", np.nan))
    m_tgt = float(backend.get("m_tgt", np.nan))
    created_mass = float(metrics.get("created_mass", np.nan))
    destroyed_mass = float(metrics.get("destroyed_mass", np.nan))
    transported_mass = float(metrics.get("transported_mass", np.nan))

    # Compute surface areas
    created_area_um2 = float(result.mass_created_hw.sum()) * (um_per_px ** 2)
    destroyed_area_um2 = float(result.mass_destroyed_hw.sum()) * (um_per_px ** 2)

    return {
        "m_src": m_src,
        "m_tgt": m_tgt,
        "transported_mass": transported_mass,
        "created_mass": created_mass,
        "destroyed_mass": destroyed_mass,
        "created_area_um2": created_area_um2,
        "destroyed_area_um2": destroyed_area_um2,
    }


# ==== VISUALIZATION ====

def plot_input_masks_with_metrics(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    src_metrics: Dict[str, float],
    tgt_metrics: Dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    """Plot input masks with annotated metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(src_mask, cmap="gray")
    axes[0].set_title(
        f"Source\nArea: {src_metrics['area_um2']:.1f} μm²\nPerimeter: {src_metrics['perimeter_um']:.1f} μm"
    )
    axes[0].axis("off")

    axes[1].imshow(tgt_mask, cmap="gray")
    axes[1].set_title(
        f"Target\nArea: {tgt_metrics['area_um2']:.1f} μm²\nPerimeter: {tgt_metrics['perimeter_um']:.1f} μm"
    )
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cost_and_gibbs(
    cost_matrix: np.ndarray,
    epsilon: float,
    cost_diag: Dict[str, float],
    gibbs_diag: Dict[str, float],
    output_path: Path,
) -> None:
    """Plot cost matrix and Gibbs kernel with diagnostics."""
    K = np.exp(-cost_matrix / epsilon)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Cost matrix
    im0 = axes[0].imshow(cost_matrix, cmap="viridis")
    axes[0].set_title(f"Cost Matrix\nMean: {cost_diag['cost_mean']:.2e}")
    plt.colorbar(im0, ax=axes[0])

    # Log10(cost)
    log_cost = np.log10(cost_matrix + 1e-20)
    im1 = axes[1].imshow(log_cost, cmap="viridis")
    axes[1].set_title(f"Log10(Cost)\nRange: [{log_cost.min():.1f}, {log_cost.max():.1f}]")
    plt.colorbar(im1, ax=axes[1])

    # Gibbs kernel
    im2 = axes[2].imshow(K, cmap="viridis")
    axes[2].set_title(
        f"Gibbs Kernel (K=exp(-C/ε))\nHealthy: {gibbs_diag['K_healthy']}\n"
        f"Zeros: {gibbs_diag['K_zeros']}, Range: [{gibbs_diag['K_min']:.2e}, {gibbs_diag['K_max']:.2e}]"
    )
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Cost Matrix Analysis (ε={epsilon:.2e})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_flow_field(
    src_mask: np.ndarray,
    velocity_field_yx_hw2: np.ndarray,
    output_path: Path,
    stride: int = 8,
) -> None:
    """Plot velocity field as quiver plot overlaying source mask.

    CRITICAL: Data should already be on canonical grid from preprocessing.
    We just display it as-is with proper aspect ratio and coordinate labels.
    NO stretching or remapping - just display on canonical grid coordinates.
    """
    velocity_mag = np.sqrt(velocity_field_yx_hw2[..., 0]**2 + velocity_field_yx_hw2[..., 1]**2)

    # Use CANONICAL grid dimensions
    canon_h, canon_w = CANONICAL_GRID_SHAPE
    h_vel, w_vel = velocity_field_yx_hw2.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Velocity magnitude - display on canonical grid coords (no stretching via extent)
    # extent maps the data coordinates to the axis coordinates
    im0 = axes[0].imshow(velocity_mag, cmap="viridis", aspect='equal',
                         extent=[0, w_vel, h_vel, 0], interpolation='nearest')
    axes[0].set_title("Velocity Magnitude (px)")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    axes[0].set_xlim(0, canon_w)
    axes[0].set_ylim(canon_h, 0)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Quiver plot - display on canonical grid coords
    axes[1].imshow(src_mask, cmap="gray", alpha=0.3, aspect='equal',
                   extent=[0, src_mask.shape[1], src_mask.shape[0], 0], interpolation='nearest')

    # Subsample for quiver
    yy_vel, xx_vel = np.meshgrid(np.arange(0, h_vel, stride), np.arange(0, w_vel, stride), indexing='ij')

    u = velocity_field_yx_hw2[::stride, ::stride, 1]  # x component
    v = velocity_field_yx_hw2[::stride, ::stride, 0]  # y component

    # Only show arrows where velocity is non-zero
    mag_sub = velocity_mag[::stride, ::stride]
    mask_sub = mag_sub > 0

    axes[1].quiver(
        xx_vel[mask_sub], yy_vel[mask_sub],
        u[mask_sub], v[mask_sub],
        mag_sub[mask_sub],
        cmap="hot",
        scale=None,
        scale_units='xy',
        angles='xy',
    )
    axes[1].set_title("Velocity Field (arrows show transport direction)")
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("y (px)")
    axes[1].set_xlim(0, canon_w)
    axes[1].set_ylim(canon_h, 0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_creation_destruction_maps(
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    created_area_um2: float,
    destroyed_area_um2: float,
    output_path: Path,
) -> None:
    """Plot creation/destruction heatmaps with annotations.

    CRITICAL: Data should already be on canonical grid.
    Display as-is with canonical grid axis limits (no stretching).
    """
    # Use CANONICAL grid dimensions for axis limits
    canon_h, canon_w = CANONICAL_GRID_SHAPE
    h, w = mass_created_hw.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Display data as-is, set axis limits to canonical grid
    im0 = axes[0].imshow(mass_created_hw, cmap="Reds", aspect='equal',
                         extent=[0, w, h, 0], interpolation='nearest')
    axes[0].set_title(f"Mass Created\nTotal: {created_area_um2:.2f} μm²")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    axes[0].set_xlim(0, canon_w)
    axes[0].set_ylim(canon_h, 0)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mass_destroyed_hw, cmap="Blues", aspect='equal',
                         extent=[0, w, h, 0], interpolation='nearest')
    axes[1].set_title(f"Mass Destroyed\nTotal: {destroyed_area_um2:.2f} μm²")
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("y (px)")
    axes[1].set_xlim(0, canon_w)
    axes[1].set_ylim(canon_h, 0)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sensitivity_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
    log_scale: bool = False,
) -> None:
    """Create 2D heatmap of parameter sensitivity."""
    # Pivot table with epsilon as rows, reg_m as columns
    pivot = df.pivot_table(values=metric_col, index='epsilon', columns='marginal_relaxation')

    fig, ax = plt.subplots(figsize=(10, 6))

    if log_scale:
        im = ax.imshow(np.log10(pivot.values + 1e-20), aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'log10({metric_col})')
    else:
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_col)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_xlabel('marginal_relaxation')

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.0e}" for x in pivot.index])
    ax.set_ylabel('epsilon')

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_comparison_grid(
    output_dir: Path,
    df: pd.DataFrame,
    test_name: str,
    metric_to_show: str = "created_mass",
) -> None:
    """
    Create a comparison grid showing key results for all parameter combinations.

    This allows visual inspection of how different parameters perform.
    Shows velocity magnitude for each combination in a grid layout.
    """
    # Sort by epsilon and marginal_relaxation for consistent ordering
    df_sorted = df.sort_values(['epsilon', 'marginal_relaxation'])

    n_combos = len(df_sorted)
    n_cols = len(REG_M_GRID)
    n_rows = len(EPSILON_GRID)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each parameter combination
    for idx, row in df_sorted.iterrows():
        eps = row['epsilon']
        regm = row['marginal_relaxation']

        # Find grid position
        eps_idx = EPSILON_GRID.index(eps)
        regm_idx = REG_M_GRID.index(regm)

        ax = axes[eps_idx, regm_idx]

        # Try to load the velocity magnitude image
        param_dir = output_dir / f"eps_{eps:.0e}_regm_{regm:.0e}"
        flow_field_path = param_dir / "flow_field.png"

        # Show metric value as text
        metric_val = row.get(metric_to_show, np.nan)
        cost_val = row.get('cost', np.nan)
        stable = row.get('numerical_stable', False)

        status_icon = "✓" if stable else "✗"
        color = "green" if stable else "red"

        ax.text(
            0.5, 0.5,
            f"ε={eps:.0e}\nreg_m={regm:.0e}\n{status_icon}\n"
            f"cost={cost_val:.2e}\n{metric_to_show}={metric_val:.2e}",
            ha='center', va='center', fontsize=8,
            color=color, weight='bold',
            transform=ax.transAxes
        )
        # Use CANONICAL grid dimensions
        canon_h, canon_w = CANONICAL_GRID_SHAPE
        ax.set_xlim(0, canon_w)
        ax.set_ylim(canon_h, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add border color based on stability
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.suptitle(f"{test_name} - Parameter Comparison\n{metric_to_show}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_comparison_grid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved comparison grid to {output_dir / 'parameter_comparison_grid.png'}")


# ==== TEST EXECUTION ====

def run_single_param_combo(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    epsilon: float,
    marginal_relaxation: float,
    output_dir: Path,
    test_name: str,
) -> Dict:
    """Run UOT with a single parameter combination and collect all metrics."""

    # Create output directory
    param_dir = output_dir / f"eps_{epsilon:.0e}_regm_{marginal_relaxation:.0e}"
    param_dir.mkdir(parents=True, exist_ok=True)

    # Compute surface metrics for input masks
    src_metrics = compute_surface_metrics(src_mask, UM_PER_PX)
    tgt_metrics = compute_surface_metrics(tgt_mask, UM_PER_PX)

    # Plot input masks
    plot_input_masks_with_metrics(
        src_mask, tgt_mask, src_metrics, tgt_metrics,
        param_dir / "input_masks.png",
        f"{test_name} | ε={epsilon:.0e}, reg_m={marginal_relaxation:.0e}"
    )

    # Create UOT config
    config = UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=marginal_relaxation,
        downsample_factor=1,  # No downsampling on synthetic tests
        downsample_divisor=1,
        padding_px=16,  # Use proper padding (was 0)
        mass_mode=MassMode.UNIFORM,
        align_mode="none",
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=COORD_SCALE,
        use_pair_frame=True,  # NEW: Enable pair frame
    )

    # Create frame pair
    pair = UOTFramePair(
        src=UOTFrame(embryo_mask=src_mask, meta={"test": test_name}),
        tgt=UOTFrame(embryo_mask=tgt_mask, meta={"test": test_name}),
    )

    # Run UOT
    try:
        result = run_uot_pair(pair, config=config)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "epsilon": epsilon,
            "marginal_relaxation": marginal_relaxation,
            "cost": np.nan,
            "cost_is_nan": True,
            "error": str(e),
        }

    # NEW: Validate pair frame results
    if config.use_pair_frame and hasattr(result, 'transform_meta'):
        if result.transform_meta.get("preprocess", {}).get("pair_frame_used"):
            # Verify outputs are canonical-shaped
            assert result.mass_created_hw.shape == CANONICAL_GRID_SHAPE, \
                f"Mass created not canonical shaped: {result.mass_created_hw.shape}"
            assert result.velocity_field_yx_hw2.shape[:2] == CANONICAL_GRID_SHAPE, \
                f"Velocity not canonical shaped: {result.velocity_field_yx_hw2.shape}"

    # Collect all diagnostics
    cost_is_nan = np.isnan(result.cost)

    # Compute cost matrix for diagnostics
    backend = POTBackend()
    coords_src = result.support_src_yx.astype(np.float64) * float(COORD_SCALE)
    coords_tgt = result.support_tgt_yx.astype(np.float64) * float(COORD_SCALE)
    cost_matrix = ot.dist(coords_src, coords_tgt, metric="sqeuclidean")

    cost_diag = diagnose_cost_matrix(cost_matrix, epsilon)
    gibbs_diag = diagnose_gibbs_kernel(cost_matrix, epsilon)
    coupling_diag = diagnose_coupling_sparsity(result.coupling) if result.coupling is not None else {}
    velocity_diag = compute_velocity_metrics(result.velocity_field_yx_hw2)
    mass_diag = compute_mass_metrics(result, src_mask, tgt_mask, UM_PER_PX)

    # Numerical stability check
    numerical_stable = (
        not cost_is_nan
        and not velocity_diag["velocity_has_nan"]
        and gibbs_diag["K_healthy"]
    )

    # Plot diagnostics
    plot_cost_and_gibbs(cost_matrix, epsilon, cost_diag, gibbs_diag, param_dir / "cost_and_gibbs.png")
    plot_flow_field(src_mask, result.velocity_field_yx_hw2, param_dir / "flow_field.png", stride=6)
    plot_creation_destruction_maps(
        result.mass_created_hw, result.mass_destroyed_hw,
        mass_diag["created_area_um2"], mass_diag["destroyed_area_um2"],
        param_dir / "creation_destruction.png"
    )

    # Compile metrics
    metrics = {
        "epsilon": epsilon,
        "marginal_relaxation": marginal_relaxation,
        "coord_scale": COORD_SCALE,
        "cost": result.cost,
        "cost_is_nan": cost_is_nan,
        "numerical_stable": numerical_stable,
        **cost_diag,
        **gibbs_diag,
        **coupling_diag,
        **velocity_diag,
        **mass_diag,
        "src_area_um2": src_metrics["area_um2"],
        "tgt_area_um2": tgt_metrics["area_um2"],
    }

    return metrics


def run_test_with_grid(
    test_num: int,
    test_name: str,
    test_fn,
    test_params: Dict,
    viable_params: Optional[List[Dict]] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run a test across parameter grid and return results + viable params."""

    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {test_name}")
    print('='*60)

    output_dir = OUTPUT_DIR / f"test{test_num}_{test_name.lower().replace(' ', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test masks
    src_mask, tgt_mask = test_fn(**test_params)

    # Determine parameter combinations to test
    if viable_params is not None:
        print(f"Testing {len(viable_params)} viable parameter combinations from previous test...")
        param_combos = [(p['epsilon'], p['marginal_relaxation']) for p in viable_params]
    else:
        print(f"Testing full parameter grid ({len(EPSILON_GRID)} × {len(REG_M_GRID)} = {len(EPSILON_GRID) * len(REG_M_GRID)} combinations)...")
        param_combos = [(eps, regm) for eps in EPSILON_GRID for regm in REG_M_GRID]

    # Run all parameter combinations
    all_metrics = []
    for eps, regm in param_combos:
        print(f"  Running ε={eps:.0e}, reg_m={regm:.0e}...", end=" ")
        metrics = run_single_param_combo(src_mask, tgt_mask, eps, regm, output_dir, test_name)
        all_metrics.append(metrics)

        # Quick feedback
        if metrics.get("numerical_stable", False):
            print(f"✓ (cost={metrics['cost']:.4f})")
        else:
            print(f"✗ (unstable)")

    # Save results
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'results.csv'}")

    # Apply pass criteria based on test type
    viable_next = []

    if test_num == 1:  # Identity test
        # Pass criteria: minimal creation/destruction/velocity
        df_pass = df[
            (df['cost'] < 1e-6) &
            (df['created_mass'] < 1e-6) &
            (df['destroyed_mass'] < 1e-6) &
            (df['mean_velocity_px'] < 1e-6) &
            (df['K_healthy'] == True)
        ]
    elif test_num == 2:  # Non-overlapping circles
        # Pass criteria: transport happened, no creation/destruction
        df_pass = df[
            (df['cost'] > 0) &
            (df['created_mass'] < 1e-3) &
            (df['destroyed_mass'] < 1e-3) &
            (df['mean_velocity_px'] > 0) &
            (df['sparsity'] > 0.8)
        ]
    elif test_num == 3:  # Shape change
        # Pass criteria: creation/destruction at expected locations
        df_pass = df[
            (df['created_mass'] > 0) &
            (df['destroyed_mass'] > 0) &
            (df['numerical_stable'] == True)
        ]
    else:  # Test 4: Combined
        # Pass criteria: combined behavior
        df_pass = df[
            (df['cost'] > 0) &
            (df['created_mass'] > 0) &
            (df['destroyed_mass'] > 0) &
            (df['numerical_stable'] == True)
        ]

    print(f"\n{len(df_pass)} / {len(df)} parameter combinations passed test criteria")

    if len(df_pass) > 0:
        viable_next = df_pass[['epsilon', 'marginal_relaxation']].to_dict('records')

        # Save viable params
        with open(output_dir / "viable_params.json", 'w') as f:
            json.dump(viable_next, f, indent=2)
    else:
        print("WARNING: No parameter combinations passed!")

    # Create sensitivity plots
    if len(df) > 0:
        print("\nGenerating sensitivity plots...")

        # Plot for key metrics
        if test_num == 1:
            plot_sensitivity_heatmap(
                df, 'created_mass', output_dir / "sensitivity_created_mass.png",
                f"Test {test_num}: Created Mass Sensitivity", log_scale=True
            )
        elif test_num == 2:
            if 'sparsity' in df.columns:
                plot_sensitivity_heatmap(
                    df, 'sparsity', output_dir / "sensitivity_sparsity.png",
                    f"Test {test_num}: Coupling Sparsity Sensitivity", log_scale=False
                )

        # Always plot cost
        plot_sensitivity_heatmap(
            df, 'cost', output_dir / "sensitivity_cost.png",
            f"Test {test_num}: Cost Sensitivity", log_scale=True
        )

        # Create parameter comparison grid
        print("\nGenerating parameter comparison grid...")
        metric_to_show = 'created_mass' if test_num in [1, 3, 4] else 'transported_mass'
        plot_parameter_comparison_grid(output_dir, df, test_name, metric_to_show)

    return df, viable_next


# ==== MAIN WORKFLOW ====

def main():
    parser = argparse.ArgumentParser(description="Debug UOT parameters with synthetic tests")
    parser.add_argument(
        '--test',
        type=str,
        default='all',
        help='Which test to run: 1, 2, 3, 4, or "all"'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=40,
        help='Circle radius in pixels'
    )
    parser.add_argument(
        '--separation',
        type=int,
        default=120,
        help='Separation for non-overlapping test'
    )
    parser.add_argument(
        '--shift',
        type=int,
        default=10,
        help='Shift for combined test'
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track viable parameters across tests
    viable_params = None
    all_results = {}

    tests_to_run = ['1', '2', '3', '4'] if args.test == 'all' else [args.test]

    for test_id in tests_to_run:
        if test_id == '1':
            df, viable_params = run_test_with_grid(
                1, "Identity", make_identity_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius},
                viable_params=None  # Full grid for first test
            )
            all_results['test1'] = df

        elif test_id == '2':
            if viable_params is None or len(viable_params) == 0:
                print("\nERROR: Test 1 must pass before running Test 2")
                print("Run Test 1 first: python debug_uot_params.py --test 1")
                break

            df, viable_params = run_test_with_grid(
                2, "Non-overlapping Circles", make_nonoverlap_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius, "separation": args.separation},
                viable_params=viable_params
            )
            all_results['test2'] = df

        elif test_id == '3':
            if viable_params is None or len(viable_params) == 0:
                print("\nERROR: Test 2 must pass before running Test 3")
                print("Run Tests 1 and 2 first: python debug_uot_params.py --test all")
                break

            df, viable_params = run_test_with_grid(
                3, "Shape Change", make_shape_change_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius},
                viable_params=viable_params
            )
            all_results['test3'] = df

        elif test_id == '4':
            if viable_params is None or len(viable_params) == 0:
                print("\nERROR: Test 3 must pass before running Test 4")
                print("Run all tests first: python debug_uot_params.py --test all")
                break

            df, viable_params = run_test_with_grid(
                4, "Combined", make_combined_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius, "shift": args.shift},
                viable_params=viable_params
            )
            all_results['test4'] = df

    # Final recommendations
    if viable_params is not None and len(viable_params) > 0:
        print(f"\n{'='*60}")
        print("FINAL RECOMMENDATIONS")
        print('='*60)
        print(f"\n{len(viable_params)} parameter combinations passed all tests:")
        for p in viable_params[:10]:  # Show first 10
            print(f"  ε={p['epsilon']:.0e}, reg_m={p['marginal_relaxation']:.0e}")

        if len(viable_params) > 10:
            print(f"  ... and {len(viable_params) - 10} more")

        # Save recommendations
        with open(OUTPUT_DIR / "recommended_params.json", 'w') as f:
            json.dump(viable_params, f, indent=2)

        print(f"\nRecommendations saved to {OUTPUT_DIR / 'recommended_params.json'}")
    else:
        print("\nWARNING: No parameter combinations passed all tests!")

    print(f"\n{'='*60}")
    print("DIAGNOSTICS COMPLETE")
    print('='*60)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
