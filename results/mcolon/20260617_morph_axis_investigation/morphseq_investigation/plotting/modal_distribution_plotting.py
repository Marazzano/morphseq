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

from morphseq_investigation.core.density_composition import DensityGrid
from morphseq_investigation.core.peak_counting import PeakCountDetail, peak_count_detail
from morphseq_investigation.core.support_geometry import (
    fiedler_value,
    hdr_concentration_auc,
    mst_max_edge,
    normalize_shape,
    valley_depth,
)
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
    component_labels: np.ndarray | None = None
    composed_grid: DensityGrid | None = None
    note: str = ""

    @property
    def true_grid(self) -> DensityGrid | None:
        """Backward-compatible alias for the composed density grid."""
        return self.composed_grid


def compute_v0_metric_summary(
    points: np.ndarray,
    *,
    kde=None,
    distribution_id: str | None = None,
) -> dict[str, float | None]:
    """Compute the small V0 metric set on shape-normalized sampled points."""

    pts = normalize_shape(np.asarray(points, dtype=float))
    valley = float(valley_depth(pts, kde=kde))
    if distribution_id is not None and distribution_id.startswith("one_peak_"):
        valley = None
    return {
        "hdr_concentration_auc": float(hdr_concentration_auc(pts, relative=True, kde=kde)),
        "valley_depth": valley,
        "mst_max_edge": float(mst_max_edge(pts)),
        "fiedler": float(fiedler_value(pts)),
    }


def compute_v0_peak_count_summary(
    *,
    truth_density: np.ndarray | None = None,
    observed_density: np.ndarray | None = None,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
) -> dict[str, PeakCountDetail | None]:
    """Compute truth and observed peak-count probes for one V0 distribution."""

    truth_detail = None
    if truth_density is not None:
        truth_detail = peak_count_detail(
            truth_density,
            min_component_mass_frac=min_component_mass_frac,
            sweep_steps=sweep_steps,
        )
    observed_detail = None
    if observed_density is not None:
        observed_detail = peak_count_detail(
            observed_density,
            min_component_mass_frac=min_component_mass_frac,
            sweep_steps=sweep_steps,
        )
    return {"truth": truth_detail, "observed": observed_detail}


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
        return "N/A", "no density"
    if detail.n_modes < 2 or detail.split_fraction is None:
        return f"{detail.n_modes:d}", "split N/A"
    return f"{detail.n_modes:d}", f"split {detail.split_fraction:.3f}"


def plot_peak_count_summary_row(
    ax,
    truth_detail: PeakCountDetail | None,
    observed_detail: PeakCountDetail | None,
    *,
    title: str = "peak count audit",
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
    rows = [("truth peaks", truth_detail), ("detected peaks", observed_detail)]
    y = 0.72
    for label, detail in rows:
        count_txt, split_txt = _format_peak_count_value(detail)
        ax.text(0.03, y, label, transform=ax.transAxes, fontsize=6.6, color="#444", va="center", ha="left")
        ax.text(
            0.48,
            y,
            count_txt,
            transform=ax.transAxes,
            fontsize=6.7,
            color="#222",
            va="center",
            ha="right",
            fontfamily="monospace",
            fontweight="bold",
        )
        ax.text(0.52, y, split_txt, transform=ax.transAxes, fontsize=5.8, color="#777", va="center", ha="left")
        y -= 0.18

    valley_state = "draw valley" if observed_detail is not None and observed_detail.n_modes >= 2 else "no valley"
    valley_reason = (
        "observed split" if observed_detail is not None and observed_detail.n_modes >= 2 else "single detected peak"
    )
    ax.text(0.03, y - 0.02, "ring", transform=ax.transAxes, fontsize=6.6, color="#B8860B", va="center", ha="left")
    ax.text(
        0.48,
        y - 0.02,
        valley_state,
        transform=ax.transAxes,
        fontsize=6.7,
        color="#222",
        va="center",
        ha="right",
        fontfamily="monospace",
        fontweight="bold",
    )
    ax.text(0.52, y - 0.02, valley_reason, transform=ax.transAxes, fontsize=5.8, color="#777", va="center", ha="left")
    y -= 0.18
    ax.text(0.03, y - 0.02, "method", transform=ax.transAxes, fontsize=6.6, color="#444", va="center", ha="left")
    ax.text(
        0.48,
        y - 0.02,
        "sweep",
        transform=ax.transAxes,
        fontsize=6.7,
        color="#222",
        va="center",
        ha="right",
        fontfamily="monospace",
        fontweight="bold",
    )
    ax.text(0.52, y - 0.02, "super-level threshold", transform=ax.transAxes, fontsize=5.8, color="#777", va="center", ha="left")


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
    auto_scale: bool = True,
    include_peak_row: bool = True,
    include_metric_row: bool = False,
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
    n_rows = 3 + int(include_peak_row) + int(include_metric_row)
    fig_height = 6.95 + (1.35 if include_peak_row else 0.0) + (1.35 if include_metric_row else 0.0)
    fig, axes = plt.subplots(n_rows, n, figsize=(2.45 * n, fig_height), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.12, top=0.84, wspace=0.12, hspace=0.18)

    shared_box = None
    if not auto_scale:
        shared_box = canonical_box_from_specs(specs)

    for col, spec in enumerate(specs):
        points = np.asarray(spec.points, dtype=float)
        if auto_scale:
            grids_for_box = [points]
            if spec.composed_grid is not None:
                true_pts = np.column_stack([spec.composed_grid.xx.ravel(), spec.composed_grid.yy.ravel()])
                finite = np.isfinite(spec.composed_grid.density.ravel()) & (spec.composed_grid.density.ravel() > 0)
                if finite.any():
                    grids_for_box.append(true_pts[finite])
            box = square_density_box(density_box(grids_for_box, kde=kde))
        else:
            box = shared_box
        sample_grid = evaluate_density_grid(points, box, kde=kde)
        truth_detail = None
        if spec.composed_grid is not None:
            truth_detail = peak_count_detail(spec.composed_grid.density)
        observed_detail = peak_count_detail(sample_grid.density)

        ax = axes[0][col]
        if spec.composed_grid is not None:
            plot_kde_field(
                ax,
                spec.composed_grid,
                color=TRUE_DENS_COLOR,
                cmap=None,
                alpha_max=0.30,
            )
        else:
            plot_kde_field(ax, sample_grid)
        format_density_axis(ax, box)
        ax.set_title(spec.distribution_id.replace("_", "\n"), fontsize=8.0, fontweight="bold")
        if spec.note:
            ax.text(0.02, 0.03, spec.note, transform=ax.transAxes, fontsize=6.2, color="#555", va="bottom")

        ax = axes[1][col]
        plot_kde_field(ax, sample_grid)
        plot_raw_points(ax, points, spec.component_labels, s=14)
        format_density_axis(ax, box)

        ax = axes[2][col]
        plot_kde_field(ax, sample_grid)
        if show_hdr:
            plot_hdr_contour(ax, sample_grid)
        if observed_detail.n_modes >= 2 and observed_detail.split_level is not None:
            ax.contour(
                sample_grid.xx,
                sample_grid.yy,
                sample_grid.density,
                levels=[observed_detail.split_level],
                colors="#B8860B",
                linewidths=1.8,
                linestyles="--",
                zorder=5,
            )
        plot_raw_points(ax, points, spec.component_labels, s=11)
        format_density_axis(ax, box)

        row_idx = 3
        if include_peak_row:
            plot_peak_count_summary_row(axes[row_idx][col], truth_detail, observed_detail, title="peak count audit")
            row_idx += 1
        if include_metric_row:
            metric_summary = compute_v0_metric_summary(points, kde=kde, distribution_id=spec.distribution_id)
            plot_metric_summary_row(axes[row_idx][col], metric_summary, title="metric probes")

    axes[0][0].set_ylabel("true density\nor sample KDE", fontsize=9)
    axes[1][0].set_ylabel("sampled points\n+ KDE", fontsize=9)
    axes[2][0].set_ylabel("sample KDE\n+ HDR 50% / valley", fontsize=9)
    if include_peak_row:
        axes[3][0].set_ylabel("peak count\n(truth vs detected)", fontsize=9)
    if include_metric_row:
        axes[3 + int(include_peak_row)][0].set_ylabel("metrics\n(normalized)", fontsize=9)
    all_labels: list[str] = []
    for spec in specs:
        if spec.component_labels is None:
            continue
        all_labels.extend([str(label) for label in np.unique(spec.component_labels)])
    preferred = [
        "mode_0",
        "mode_1",
        "mode_2",
        "mode_left",
        "mode_right",
        "bridge_left_right",
        "bridge",
        "artifact",
    ]
    legend_labels: list[str] = []
    seen = set()
    for label in preferred + sorted(set(all_labels) - set(preferred)):
        if label in seen or label not in all_labels:
            continue
        seen.add(label)
        legend_labels.append(label)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=POINT_COLOR_BY_LABEL.get(label, POINT_COLOR),
            markeredgecolor=POINT_EDGE_COLOR,
            markersize=6,
            label=label,
        )
        for label in legend_labels
    ]
    handles.append(Line2D([0], [0], color="#B8860B", lw=1.8, ls="--", label="valley split floor"))
    handles.append(Line2D([0], [0], color=HDR_COLOR, lw=1.8, label="HDR 50% contour"))
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


# Tech debt:
# - When auto_scale=False, add an explicit shared density peak reference so
#   cross-panel height comparisons use the same contour levels.
# - Either use POINT_ALPHA_HDR in the HDR row or remove it if the alpha split
#   is not needed.
