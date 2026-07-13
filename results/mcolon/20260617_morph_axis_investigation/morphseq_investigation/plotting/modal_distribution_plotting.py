"""Reusable visual QA helpers for modal-organization simulations.

These helpers are intentionally pre-metric: they make the generated density
landscape and sampled point cloud easy to inspect before running benchmark
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

from morphseq_investigation.core.density_composition import CanonicalGrid, DensityGrid
from morphseq_investigation.core.peak_counting import PeakCountDetail
from morphseq_investigation.plotting.v0_analysis import (
    compute_v0_metric_summary,
    compute_v0_peak_count_summary,
)


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
POINT_ALPHA = 0.28
POINT_ALPHA_HDR = 0.22
POINT_COLOR_BY_LABEL = {
    "mode_0": "#2A9D9D",
    "mode_1": "#D05A8A",
    "mode_2": "#8E6BBE",
    "mode_left": "#2A9D9D",
    "mode_right": "#D05A8A",
    "bridge": "#E0A11B",
    "bridge_left_right": "#E0A11B",
    "artifact": "#666666",
}
METRIC_COLOR_BY_NAME = {
    "hdr_concentration_auc": "#2166AC",
    "valley_depth": "#B2182B",
    "mst_max_edge": "#E0A11B",
    "fiedler": "#7A5CC6",
}


@dataclass(frozen=True)
class DistributionVisualSpec:
    distribution_id: str
    points: np.ndarray
    sampled_grid: DensityGrid | None = None
    component_labels: np.ndarray | None = None
    composed_grid: DensityGrid | None = None
    note: str = ""

    @property
    def true_grid(self) -> DensityGrid | None:
        """Backward-compatible alias for the composed density grid."""
        return self.composed_grid


@dataclass(frozen=True)
class DistributionOverlay:
    """Two density evaluations on a shared plotting frame."""

    box: tuple[float, float, float, float]
    target_grid: DensityGrid
    reference_grid: DensityGrid


RowOverlayFn = Callable[[plt.Axes, DistributionVisualSpec, DensityGrid, tuple[float, float, float, float]], None]


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


def density_box(
    point_sets: Iterable[np.ndarray],
    *,
    support_frac: float = SUPPORT_FRAC,
    margin: float = 0.06,
    scan_pct: float = 2.0,
) -> tuple[float, float, float, float]:
    """Return a robust point-quantile box around one or more clouds.

    ALWAYS square (via `square_density_box`): this is the one place a plotting
    box gets built from raw points, so squaring here means every caller --
    including ones that forget to wrap the call -- gets equal-aspect panels
    for free, instead of squaring being an opt-in step.
    """
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
    xlo, xhi = xr[0] - padx, xr[1] + padx
    ylo, yhi = yr[0] - pady, yr[1] + pady
    mx = (xhi - xlo) * margin if xhi > xlo else 1.0
    my = (yhi - ylo) * margin if yhi > ylo else 1.0
    box = (float(xlo - mx), float(xhi + mx), float(ylo - my), float(yhi + my))
    return square_density_box(box)


def derive_shared_grid(
    target_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    grid: int = GRID,
    support_frac: float = SUPPORT_FRAC,
    margin: float = 0.06,
    scan_pct: float = 2.0,
) -> CanonicalGrid:
    """Derive the canonical grid used for a target/reference comparison."""
    box = density_box(
        [np.asarray(target_points, dtype=float), np.asarray(reference_points, dtype=float)],
        support_frac=support_frac,
        margin=margin,
        scan_pct=scan_pct,
    )
    return CanonicalGrid(
        x_min=float(box[0]),
        x_max=float(box[1]),
        y_min=float(box[2]),
        y_max=float(box[3]),
        grid_size=int(grid),
    )


def canonical_box_from_specs(specs: list[DistributionVisualSpec]) -> tuple[float, float, float, float]:
    """Return a shared plotting box from the canonical grids on the specs."""
    boxes = []
    for spec in specs:
        if spec.composed_grid is not None and spec.composed_grid.grid is not None:
            grid = spec.composed_grid.grid
            boxes.append((grid.x_min, grid.x_max, grid.y_min, grid.y_max))
    if not boxes:
        raise ValueError("canonical_box_from_specs requires at least one spec with a canonical grid")
    x_lo = min(box[0] for box in boxes)
    x_hi = max(box[1] for box in boxes)
    y_lo = min(box[2] for box in boxes)
    y_hi = max(box[3] for box in boxes)
    return square_density_box((x_lo, x_hi, y_lo, y_hi))


def evaluate_density_grid(
    density_grid: DensityGrid,
) -> DensityGrid:
    """Validate/pass through an already-calculated density grid."""
    if not isinstance(density_grid, DensityGrid):
        raise TypeError("plotting requires a precomputed DensityGrid")
    return density_grid


def build_distribution_overlay(
    target_grid: DensityGrid,
    reference_grid: DensityGrid,
) -> DistributionOverlay:
    """Package two precomputed densities on one shared plotting frame."""
    if target_grid.xx.shape != reference_grid.xx.shape or not np.allclose(target_grid.xx, reference_grid.xx) or not np.allclose(target_grid.yy, reference_grid.yy):
        raise ValueError("overlay densities must share identical evaluation coordinates")
    box = (float(np.min(target_grid.xx)), float(np.max(target_grid.xx)),
           float(np.min(target_grid.yy)), float(np.max(target_grid.yy)))
    return DistributionOverlay(box=box, target_grid=target_grid, reference_grid=reference_grid)


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
    peak_ref: float | None = None,
    zorder: int = 0,
) -> None:
    """Draw a soft filled KDE field without a hard rectangular edge."""
    dens = np.asarray(grid.density, dtype=float)
    peak = float(peak_ref) if peak_ref is not None else (float(np.nanmax(dens)) if np.isfinite(dens).any() else 0.0)
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
            alpha=POINT_ALPHA,
            facecolors=POINT_COLOR,
            edgecolors=POINT_EDGE_COLOR,
            linewidths=0.35,
            zorder=3,
        )
        return

    labels = np.asarray(labels)
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            s=s,
            alpha=POINT_ALPHA,
            facecolors=POINT_COLOR_BY_LABEL.get(str(label), POINT_COLOR),
            edgecolors=POINT_EDGE_COLOR,
            linewidths=0.35,
            zorder=3,
        )


def plot_metric_summary_row(
    ax,
    metrics: dict[str, float | None],
    *,
    title: str = "metric probes",
) -> None:
    """Render a compact text summary of the V0 metrics for one distribution."""

    ax.set_axis_off()
    ax.text(
        0.03,
        0.95,
        f"{title}\n(shape-normalized sample)",
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color="#444",
        va="top",
        ha="left",
    )
    rows = [
        ("hdr_concentration_auc", "HDR conc", "higher = more concentrated"),
        ("valley_depth", "valley", "higher = more separated"),
        ("mst_max_edge", "MST edge", "higher = more separated"),
        ("fiedler", "fiedler", "higher = more separated"),
    ]
    y = 0.72
    for name, label, desc in rows:
        color = METRIC_COLOR_BY_NAME.get(name, "#444")
        value = metrics.get(name, float("nan"))
        ax.text(0.03, y, label, transform=ax.transAxes, fontsize=6.6, color=color, va="center", ha="left")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            rendered = "N/A"
        else:
            rendered = f"{float(value):.3f}"
        ax.text(
            0.48,
            y,
            rendered,
            transform=ax.transAxes,
            fontsize=6.7,
            color="#222",
            va="center",
            ha="right",
            fontfamily="monospace",
            fontweight="bold",
        )
        ax.text(0.52, y, desc, transform=ax.transAxes, fontsize=5.8, color="#777", va="center", ha="left")
        y -= 0.17


def _format_peak_count_value(detail: PeakCountDetail | None) -> tuple[str, str]:
    if detail is None:
        return "N/A", "no dens"
    if detail.n_modes < 2 or detail.split_fraction is None:
        if detail.n_modes > 1:
            return f"{detail.n_modes:d}", "ambig"
        return f"{detail.n_modes:d}", "split N/A"
    return f"{detail.n_modes:d}", f"split {detail.split_fraction:.3f}"


def _peak_method_label(method_name: str) -> str:
    labels = {
        "superlevel_cap_mass": "KDE cap",
        "hdr_component_persistence": "KDE HDR",
        "kde_peak_basins_sample_support": "KDE basin",
    }
    return labels.get(method_name, method_name)


def plot_peak_count_summary_row(
    ax,
    truth_detail: PeakCountDetail | None,
    observed_detail: PeakCountDetail | dict[str, PeakCountDetail | None] | None,
    *,
    title: str = "peak count audit",
    observed_method_order: tuple[str, ...] | None = None,
    primary_method: str | None = None,
) -> None:
    """Render the truth-versus-detected peak-count audit for one distribution."""

    ax.set_axis_off()
    ax.text(
        0.03,
        0.95,
        f"{title}\n(valley sweep)",
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color="#444",
        va="top",
        ha="left",
    )
    if isinstance(observed_detail, dict):
        rows = [("truth", truth_detail)]
        method_order = observed_method_order or tuple(observed_detail.keys())
        for method_name in method_order:
            rows.append((_peak_method_label(method_name), observed_detail.get(method_name)))
        primary_detail = None
        if primary_method is not None:
            primary_detail = observed_detail.get(primary_method)
        if primary_detail is None and method_order:
            primary_detail = observed_detail.get(method_order[0])
        if primary_detail is None:
            primary_detail = truth_detail
    else:
        rows = [("truth", truth_detail), ("det", observed_detail)]
        primary_detail = observed_detail

    y = 0.74
    step = 0.13 if len(rows) <= 4 else 0.10
    for label, detail in rows:
        count_txt, split_txt = _format_peak_count_value(detail)
        ax.text(0.03, y, label, transform=ax.transAxes, fontsize=6.2, color="#444", va="center", ha="left")
        ax.text(
            0.48,
            y,
            count_txt,
            transform=ax.transAxes,
            fontsize=6.4,
            color="#222",
            va="center",
            ha="right",
            fontfamily="monospace",
            fontweight="bold",
        )
        ax.text(0.52, y, split_txt, transform=ax.transAxes, fontsize=5.4, color="#777", va="center", ha="left")
        y -= step

    valley_state = "draw valley" if primary_detail is not None and primary_detail.n_modes >= 2 else "no valley"
    valley_reason = (
        "observed split" if primary_detail is not None and primary_detail.n_modes >= 2 else "single detected peak"
    )
    ax.text(0.03, y - 0.02, "ring", transform=ax.transAxes, fontsize=6.2, color="#B8860B", va="center", ha="left")
    ax.text(
        0.48,
        y - 0.02,
        valley_state,
        transform=ax.transAxes,
        fontsize=6.4,
        color="#222",
        va="center",
        ha="right",
        fontfamily="monospace",
        fontweight="bold",
    )
    ax.text(0.52, y - 0.02, valley_reason, transform=ax.transAxes, fontsize=5.4, color="#777", va="center", ha="left")
    y -= step
    ax.text(0.03, y - 0.02, "meth", transform=ax.transAxes, fontsize=6.2, color="#444", va="center", ha="left")
    ax.text(
        0.48,
        y - 0.02,
        "audit",
        transform=ax.transAxes,
        fontsize=6.4,
        color="#222",
        va="center",
        ha="right",
        fontfamily="monospace",
        fontweight="bold",
    )
    method_note = "KDE geom" if not isinstance(observed_detail, dict) else "KDE geom; sample support"
    ax.text(0.52, y - 0.02, method_note, transform=ax.transAxes, fontsize=5.2, color="#777", va="center", ha="left")


def plot_density_overlap(
    ax,
    grid_a: DensityGrid,
    grid_b: DensityGrid,
    *,
    color_a: str = TARGET_DENS_COLOR,
    color_b: str = REFERENCE_DENS_COLOR,
    alpha_scale: float = 1.0,
) -> None:
    """Draw two translucent density fields on the same grid.

    `alpha_scale` < 1 lightens both fields uniformly (e.g. 0.62 for a softer
    background that lets overlaid marks read).
    """
    for grid, color in ((grid_b, color_b), (grid_a, color_a)):
        dens = np.asarray(grid.density, dtype=float)
        peak = float(np.nanmax(dens)) if np.isfinite(dens).any() else 0.0
        if peak <= 0:
            continue
        levels = [f * peak for f in (0.20, 0.45, 0.70, 0.90)] + [peak]
        colors = [to_rgba(color, a * float(alpha_scale)) for a in (0.16, 0.24, 0.32, 0.42)]
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


_COMPATIBILITY_EXPORTS = {
    "plot_v0_distribution_qc_grid": ("v0_qc", "plot_v0_distribution_qc_grid"),
    "render_distribution_qc_grid": ("v0_qc", "render_distribution_qc_grid"),
    "draw_resolved_peak_basins": ("resolved_peaks", "draw_resolved_peak_basins"),
    "hdr_contour_for_field": ("resolved_peaks", "hdr_contour_for_field"),
    "mode_count_label": ("resolved_peaks", "mode_count_label"),
    "plot_resolved_peak_overlay": ("resolved_peaks", "plot_resolved_peak_overlay"),
}


def __getattr__(name: str):
    """Lazily preserve imports from the former plotting monolith."""
    export = _COMPATIBILITY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(name)
    module_name, attribute_name = export
    from importlib import import_module

    value = getattr(
        import_module(f"morphseq_investigation.plotting.{module_name}"),
        attribute_name,
    )
    globals()[name] = value
    return value

# Tech debt:
# - When auto_scale=False, add an explicit shared density peak reference so
#   cross-panel height comparisons use the same contour levels.
# - Either use POINT_ALPHA_HDR in the HDR row or remove it if the alpha split
#   is not needed.
