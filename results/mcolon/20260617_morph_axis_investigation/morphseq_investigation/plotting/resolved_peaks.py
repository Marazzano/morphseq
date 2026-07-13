"""Resolved-peak labels and overlay renderers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from morphseq_investigation.plotting.modal_distribution_plotting import (
    plot_kde_field,
    plot_raw_points,
)


def mode_count_label(n_modes: int | None, *, frequency: float | None = None) -> str:
    """Compact label for vote-supported mode counts (row titles / legends).

    Extracted from `valley_visualization.py::_mode_count_label` -- shared
    across any figure that reports a `resolved_peak_count` next to its
    bootstrap-vote `mode_frequency`.
    """
    if n_modes is None:
        return "mode count unstable"
    n = int(n_modes)
    word = "mode" if n == 1 else "modes"
    suffix = f" ({frequency:.0%})" if frequency is not None else ""
    return f"{n} {word}{suffix}"


def hdr_contour_for_field(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    density: np.ndarray,
    *,
    color: str,
    linewidth: float = 2.0,
    hdr_mass: float = 0.60,
) -> None:
    """Draw one smooth HDR iso-density loop of `density` enclosing `hdr_mass`
    of its mass -- a closed curve that follows the KDE bump's shape.

    Extracted from `valley_visualization.py::_hdr_contour`. Unlike
    `plot_hdr_contour` (which reads a full `DensityGrid`'s mass), this takes a
    raw density array directly so callers can mask it to one basin's mass
    first (see `draw_resolved_peak_basins`).
    """
    density = np.asarray(density, dtype=float)
    if density.max() <= 0:
        return
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat)
    total = float(csum[-1])
    if total <= 0:
        return
    idx = min(int(np.searchsorted(csum / total, hdr_mass, side="left")), len(flat) - 1)
    level = float(flat[idx])
    if level <= 0:
        nonzero = flat[flat > 0]
        if nonzero.size == 0:
            return
        level = float(nonzero.min())
    ax.contour(xx, yy, density, levels=[level], colors=[color], linewidths=linewidth, zorder=5, alpha=0.95)


def draw_resolved_peak_basins(
    ax,
    distribution,
    *,
    color: str,
    linewidth: float = 2.0,
    hdr_mass: float = 0.60,
    n_modes: int | None,
) -> None:
    """Outline a `ResolvedPeakDistribution`'s modes with smooth KDE HDR loops
    that hug each bump's shape (extracted from
    `valley_visualization.py::_draw_basins`).

    `n_modes` is the vote-supported number of modes to actually display:
    - `None` -> draw nothing (no basin-count claim to make).
    - the typed resolved count controls how many supported basins are drawn;
    - a resolved count of zero or one collapses to the whole-density contour;
      extra detected peaks are not statistically supported; collapse to ONE
      outer HDR loop over the whole density rather than drawing unsupported
      sub-modes. This keeps the drawing consistent with a count shown
      elsewhere (e.g. via `mode_count_label`).

    Basin masking uses `distribution.empirical_basin_labels` so adjacent
    supported peaks stay separate loops.
    """
    density_grid = distribution.density_grid
    density = np.asarray(density_grid.density, dtype=float)
    labels = getattr(distribution, "empirical_basin_labels", None)
    labels = np.asarray(labels, dtype=int) if labels is not None else None
    peaks = list(distribution.peaks)
    if n_modes is None:
        return
    show = int(n_modes)

    if show <= 1 or distribution.resolved_peak_count <= 1:
        if peaks:
            cx, cy = peaks[0].geometry.center_coordinate
            ax.plot([cx], [cy], marker="+", color=color, ms=7, mew=1.6, zorder=6)
        hdr_contour_for_field(ax, density_grid.xx, density_grid.yy, density, color=color, linewidth=linewidth, hdr_mass=hdr_mass)
        return

    peaks_by_support = sorted(peaks, key=lambda p: p.geometry.total_support_fraction, reverse=True)
    for peak in peaks_by_support[:show]:
        cx, cy = peak.geometry.center_coordinate
        ax.plot([cx], [cy], marker="+", color=color, ms=7, mew=1.6, zorder=6)
        if labels is not None and peak.geometry.peak_id in np.unique(labels):
            mode_density = np.where(labels == peak.geometry.peak_id, density, 0.0)
        else:
            mode_density = density
        hdr_contour_for_field(ax, density_grid.xx, density_grid.yy, mode_density, color=color, linewidth=linewidth, hdr_mass=hdr_mass)


def plot_resolved_peak_overlay(ax, distribution, summary, *, title: str | None = None) -> None:
    """Overlay resolved-peak geometry on the KDE field for visual QC.

    Draws the KDE field, sample points colored by `sample_peak_ids` (accepted
    peak id vs. -1 for unassigned/outlier), and each resolved peak's center
    with its `radius` (R80) as a circle, annotated with `within_peak_r80_density`,
    `total_support_fraction`, and `cv_radius_from_center`.
    """
    plot_kde_field(ax, distribution.density_grid)

    if distribution.sample_points is not None:
        plot_raw_points(ax, distribution.sample_points, labels=distribution.sample_peak_ids)

    for peak in distribution.peaks:
        geometry = peak.geometry
        cx, cy = geometry.center_coordinate
        ax.plot(cx, cy, marker="x", color="black", markersize=8, markeredgewidth=1.6, zorder=5)
        if np.isfinite(geometry.radius) and geometry.radius > 0:
            circle = plt.Circle(
                (cx, cy), geometry.radius, fill=False, edgecolor="black", linewidth=1.2,
                linestyle="--", zorder=5,
            )
            ax.add_patch(circle)
        label = (
            f"peak {geometry.peak_id}\n"
            f"support={geometry.total_support_fraction:.2f}\n"
            f"r80_density={geometry.within_peak_r80_density:.3f}\n"
            f"cv={geometry.cv_radius_from_center:.2f}"
        )
        ax.annotate(
            label, (cx, cy), textcoords="offset points", xytext=(6, 6),
            fontsize=6.5, color="#222",
        )

    header = title if title is not None else distribution.distribution_id
    subtitle = (
        f"n_peaks={summary.number_of_peaks} "
        f"assigned={summary.assigned_support_fraction:.2f}"
    )
    ax.set_title(f"{header}\n{subtitle}", fontsize=8.5)
    ax.set_aspect("equal")
