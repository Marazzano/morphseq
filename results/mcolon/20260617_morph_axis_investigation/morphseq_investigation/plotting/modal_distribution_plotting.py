"""Reusable visual QA helpers for modal-organization simulations.

These helpers are intentionally pre-metric: they make the generated density
landscape and sampled point cloud easy to inspect before running benchmark
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, to_rgba

from morphseq_investigation.core.support_geometry import evaluate_kde_on_grid


TARGET_DENS_COLOR = "#2166AC"
REFERENCE_DENS_COLOR = "#808080"
POINT_COLOR = "#2A9D9D"
POINT_EDGE_COLOR = "#333333"
HDR_COLOR = "#1B7837"
TRUE_DENS_COLOR = "#B05A7A"
DENS_CMAP = "Blues"
DENS_CMAP_HI = 0.62
SUPPORT_FRAC = 0.02
GRID = 70
HDR_MASS_FRAC = 0.5


@dataclass(frozen=True)
class DensityGrid:
    xx: np.ndarray
    yy: np.ndarray
    density: np.ndarray


@dataclass(frozen=True)
class DistributionVisualSpec:
    distribution_id: str
    points: np.ndarray
    labels: np.ndarray | None = None
    true_grid: DensityGrid | None = None
    note: str = ""


def density_box(
    point_sets: Iterable[np.ndarray],
    *,
    support_frac: float = SUPPORT_FRAC,
    margin: float = 0.06,
    scan_pct: float = 2.0,
    kde=None,
) -> tuple[float, float, float, float]:
    """Return a robust density-defined box around the mass of one or more clouds."""
    pts_list = [np.asarray(p, dtype=float) for p in point_sets if len(p) > 0]
    if not pts_list:
        raise ValueError("density_box requires at least one non-empty point set")

    allpts = np.vstack(pts_list)
    xr = np.percentile(allpts[:, 0], [scan_pct, 100 - scan_pct])
    yr = np.percentile(allpts[:, 1], [scan_pct, 100 - scan_pct])
    padx = (xr[1] - xr[0]) * 0.35
    pady = (yr[1] - yr[0]) * 0.35
    if padx <= 0:
        padx = 1.0
    if pady <= 0:
        pady = 1.0
    xs = np.linspace(xr[0] - padx, xr[1] + padx, 80)
    ys = np.linspace(yr[0] - pady, yr[1] + pady, 80)
    xx, yy = np.meshgrid(xs, ys)
    dens = evaluate_kde_on_grid(allpts, xx, yy, kde=kde)
    if not np.isfinite(dens).any() or float(np.nanmax(dens)) <= 0:
        xlo, ylo = np.min(allpts, axis=0)
        xhi, yhi = np.max(allpts, axis=0)
    else:
        mask = dens >= support_frac * float(np.nanmax(dens))
        xsel, ysel = xx[mask], yy[mask]
        xlo, xhi, ylo, yhi = xsel.min(), xsel.max(), ysel.min(), ysel.max()
    mx = (xhi - xlo) * margin if xhi > xlo else 1.0
    my = (yhi - ylo) * margin if yhi > ylo else 1.0
    return float(xlo - mx), float(xhi + mx), float(ylo - my), float(yhi + my)


def square_density_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Expand the shorter axis so equal-aspect panels do not collapse into strips."""
    xlo, xhi, ylo, yhi = box
    xmid = 0.5 * (xlo + xhi)
    ymid = 0.5 * (ylo + yhi)
    width = xhi - xlo
    height = yhi - ylo
    side = max(width, height)
    if side <= 0:
        side = 1.0
    half = 0.5 * side
    return xmid - half, xmid + half, ymid - half, ymid + half


def evaluate_density_grid(
    points: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    grid: int = GRID,
    kde=None,
) -> DensityGrid:
    """Evaluate a KDE on a fixed box."""
    xlo, xhi, ylo, yhi = box
    xs = np.linspace(xlo, xhi, grid)
    ys = np.linspace(ylo, yhi, grid)
    xx, yy = np.meshgrid(xs, ys)
    dens = evaluate_kde_on_grid(np.asarray(points, dtype=float), xx, yy, kde=kde)
    return DensityGrid(xx=xx, yy=yy, density=dens)


def hdr_mass_level(density: np.ndarray, mass_frac: float = HDR_MASS_FRAC) -> float | None:
    """Density level bounding the densest `mass_frac` of grid mass."""
    flat = np.sort(np.asarray(density, dtype=float).ravel())[::-1]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    csum = np.cumsum(flat)
    total = float(csum[-1])
    if total <= 0:
        return None
    idx = int(np.searchsorted(csum / total, mass_frac, side="left"))
    idx = min(idx, len(flat) - 1)
    return float(flat[idx])


def plot_kde_field(
    ax,
    grid: DensityGrid,
    *,
    color: str = TARGET_DENS_COLOR,
    cmap: str | None = DENS_CMAP,
    alpha_max: float = 0.46,
    zorder: int = 0,
) -> None:
    """Draw a soft filled KDE field without a hard rectangular edge."""
    dens = np.asarray(grid.density, dtype=float)
    peak = float(np.nanmax(dens)) if np.isfinite(dens).any() else 0.0
    if peak <= 0:
        return
    levels = np.linspace(0.06 * peak, peak, 14)
    if cmap is not None:
        light = ListedColormap(plt.get_cmap(cmap)(np.linspace(0.0, DENS_CMAP_HI, 256)))
        ax.contourf(grid.xx, grid.yy, dens, levels=levels, cmap=light, extend="max", zorder=zorder)
        return
    colors = [to_rgba(color, a) for a in np.linspace(0.08, alpha_max, len(levels) - 1)]
    ax.contourf(grid.xx, grid.yy, dens, levels=levels, colors=colors, antialiased=True, zorder=zorder)


def plot_hdr_contour(
    ax,
    grid: DensityGrid,
    *,
    mass_frac: float = HDR_MASS_FRAC,
    color: str = HDR_COLOR,
    linewidth: float = 1.6,
    linestyle: str = "-",
) -> None:
    """Draw the densest-mass HDR contour for visual concentration QA."""
    level = hdr_mass_level(grid.density, mass_frac=mass_frac)
    if level is None:
        return
    ax.contour(
        grid.xx,
        grid.yy,
        grid.density,
        levels=[level],
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        zorder=4,
    )


def plot_raw_points(ax, points: np.ndarray, labels: np.ndarray | None = None, *, s: float = 18) -> None:
    """Draw raw sampled points, optionally colored by labels."""
    pts = np.asarray(points, dtype=float)
    if labels is None:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=s,
            alpha=0.86,
            facecolors=POINT_COLOR,
            edgecolors=POINT_EDGE_COLOR,
            linewidths=0.35,
            zorder=3,
        )
        return

    labels = np.asarray(labels)
    palette = {
        "mode_0": "#2A9D9D",
        "mode_1": "#D05A8A",
        "mode_2": "#8E6BBE",
        "bridge": "#E0A11B",
        "artifact": "#666666",
    }
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            s=s,
            alpha=0.86,
            facecolors=palette.get(str(label), POINT_COLOR),
            edgecolors=POINT_EDGE_COLOR,
            linewidths=0.35,
            zorder=3,
        )


def plot_density_overlap(
    ax,
    grid_a: DensityGrid,
    grid_b: DensityGrid,
    *,
    color_a: str = TARGET_DENS_COLOR,
    color_b: str = REFERENCE_DENS_COLOR,
) -> None:
    """Draw two translucent density fields on the same grid."""
    for grid, color in ((grid_b, color_b), (grid_a, color_a)):
        dens = np.asarray(grid.density, dtype=float)
        peak = float(np.nanmax(dens)) if np.isfinite(dens).any() else 0.0
        if peak <= 0:
            continue
        levels = [f * peak for f in (0.20, 0.45, 0.70, 0.90)] + [peak]
        colors = [to_rgba(color, a) for a in (0.16, 0.24, 0.32, 0.42)]
        ax.contourf(grid.xx, grid.yy, dens, levels=levels, colors=colors, antialiased=True)


def format_density_axis(ax, box: tuple[float, float, float, float]) -> None:
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C8C8C8")
        spine.set_linewidth(0.8)


def plot_v0_distribution_qc_grid(
    specs: list[DistributionVisualSpec],
    out_path: str | Path,
    *,
    title: str = "V0 modal distribution visual QA",
    kde=None,
    show_hdr: bool = True,
) -> Path:
    """Plot one visual QA grid for V0 generated distributions.

    Rows:
      1. True density if provided, otherwise sampled KDE.
      2. Sampled points on sampled KDE.
      3. Densest-50% HDR contour on sampled KDE.
    """
    if not specs:
        raise ValueError("plot_v0_distribution_qc_grid requires at least one distribution")

    n = len(specs)
    fig, axes = plt.subplots(3, n, figsize=(2.45 * n, 6.95), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.12, top=0.84, wspace=0.12, hspace=0.18)

    for col, spec in enumerate(specs):
        points = np.asarray(spec.points, dtype=float)
        grids_for_box = [points]
        if spec.true_grid is not None:
            true_pts = np.column_stack([spec.true_grid.xx.ravel(), spec.true_grid.yy.ravel()])
            finite = np.isfinite(spec.true_grid.density.ravel()) & (spec.true_grid.density.ravel() > 0)
            if finite.any():
                grids_for_box.append(true_pts[finite])
        box = square_density_box(density_box(grids_for_box, kde=kde))
        sample_grid = evaluate_density_grid(points, box, kde=kde)

        ax = axes[0][col]
        if spec.true_grid is not None:
            plot_kde_field(ax, spec.true_grid, color=TRUE_DENS_COLOR, cmap=None, alpha_max=0.38)
        else:
            plot_kde_field(ax, sample_grid)
        format_density_axis(ax, box)
        ax.set_title(spec.distribution_id.replace("_", "\n"), fontsize=8.0, fontweight="bold")
        if spec.note:
            ax.text(0.02, 0.03, spec.note, transform=ax.transAxes, fontsize=6.2, color="#555", va="bottom")

        ax = axes[1][col]
        plot_kde_field(ax, sample_grid)
        plot_raw_points(ax, points, spec.labels)
        format_density_axis(ax, box)

        ax = axes[2][col]
        plot_kde_field(ax, sample_grid)
        if show_hdr:
            plot_hdr_contour(ax, sample_grid)
        plot_raw_points(ax, points, spec.labels, s=13)
        format_density_axis(ax, box)

    axes[0][0].set_ylabel("true density\nor sample KDE", fontsize=9)
    axes[1][0].set_ylabel("sampled points\n+ KDE", fontsize=9)
    axes[2][0].set_ylabel("sample KDE\n+ HDR 50%", fontsize=9)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2A9D9D",
               markeredgecolor=POINT_EDGE_COLOR, markersize=6, label="mode_0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D05A8A",
               markeredgecolor=POINT_EDGE_COLOR, markersize=6, label="mode_1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#8E6BBE",
               markeredgecolor=POINT_EDGE_COLOR, markersize=6, label="mode_2"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#E0A11B",
               markeredgecolor=POINT_EDGE_COLOR, markersize=6, label="bridge"),
        Line2D([0], [0], color=HDR_COLOR, lw=1.8, label="HDR 50% contour"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.5,
        0.015,
        "Point colors are known generator component labels for visual QA; they are not inferred modes.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555",
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path
