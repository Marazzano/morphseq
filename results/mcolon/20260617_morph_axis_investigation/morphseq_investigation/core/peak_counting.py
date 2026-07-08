"""Empirical peak-count helpers for modal-organization density landscapes.

The detector layer is now method-aware. The legacy superlevel sweep remains the
default compatibility path, but callers can route to HDR- and basin-based
detectors to compare why a density field was counted the way it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label as ndi_label
from scipy.ndimage import maximum_filter

if TYPE_CHECKING:
    from .density_composition import DensityGrid


SUPPORTED_METHODS = (
    "superlevel_cap_mass",
    "hdr_component_persistence",
    "kde_peak_basins_sample_support",
)


@dataclass(frozen=True)
class PeakCandidateDetail:
    """Per-candidate evidence for a peak detector."""

    candidate_peak_id: int
    peak_x: float | None
    peak_y: float | None
    peak_height: float
    nearest_saddle_or_merge_height: float | None
    prominence_ratio: float | None
    basin_sample_count: int | None
    basin_sample_fraction: float | None
    basin_kde_mass: float | None
    superlevel_cap_mass_at_split: float | None
    accepted: bool
    reject_reason: str
    component_mass: float | None = None
    hdr_component_count: int | None = None
    component_area: float | None = None
    component_sample_count: int | None = None
    component_sample_fraction: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeakDetectionResult:
    """Routed peak-detection result for a single density field."""

    method_name: str
    n_modes: int
    peak_density: float
    total_mass: float
    split_fraction: float | None
    split_level: float | None
    n_components_at_split: int
    component_masses: tuple[float, ...]
    candidate_peak_count: int
    accepted_peak_count: int
    rejected_peak_count: int
    peak_locations: tuple[tuple[float, float], ...]
    peak_heights: tuple[float, ...]
    basin_sample_counts: tuple[int, ...]
    basin_sample_fractions: tuple[float, ...]
    basin_kde_masses: tuple[float, ...]
    hdr_component_counts: tuple[int, ...]
    reject_reasons: tuple[str, ...]
    notes: tuple[str, ...]
    candidate_details: tuple[PeakCandidateDetail, ...] = ()
    # Integer basin-label raster B(x,y) on the SAME grid as the input `density`
    # (label 0 = background, 1..K = basins). None for empty/degenerate results.
    basin_labels: np.ndarray | None = None

    @property
    def component_mass_fractions(self) -> tuple[float, ...]:
        return self.component_masses

    @property
    def accepted_component_count(self) -> int:
        return self.accepted_peak_count

    @property
    def rejected_component_count(self) -> int:
        return self.rejected_peak_count


# Backwards-compatible alias. Existing callers can keep importing PeakCountDetail.
PeakCountDetail = PeakDetectionResult


def _sanitize_density(density: np.ndarray) -> np.ndarray:
    dens = np.asarray(density, dtype=float)
    return np.where(np.isfinite(dens), dens, 0.0)


def _density_mass(dens: np.ndarray) -> float:
    return float(np.sum(dens)) if dens.size else 0.0


def _grid_mass(dens: np.ndarray, grid: "DensityGrid | None") -> float:
    total = _density_mass(dens)
    if total <= 0:
        return 0.0
    if grid is None or getattr(grid, "grid", None) is None:
        return total
    cell_area = float(grid.grid.cell_area)
    return float(np.sum(dens) * cell_area)


def _component_masses(dens: np.ndarray, labels: np.ndarray, n_components: int) -> np.ndarray:
    total = _density_mass(dens)
    if total <= 0 or n_components <= 0:
        return np.asarray([], dtype=float)
    return np.asarray(
        [float(np.sum(dens[labels == k]) / total) for k in range(1, n_components + 1)],
        dtype=float,
    )


def _component_centers_and_heights(
    dens: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    *,
    grid: "DensityGrid | None" = None,
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    peak_locations: list[tuple[float, float]] = []
    peak_heights: list[float] = []
    if n_components <= 0:
        return (), ()

    if grid is not None:
        xx = np.asarray(grid.xx, dtype=float)
        yy = np.asarray(grid.yy, dtype=float)
    else:
        xx = yy = None

    for component_id in range(1, n_components + 1):
        mask = labels == component_id
        if not np.any(mask):
            peak_locations.append((float("nan"), float("nan")))
            peak_heights.append(float("nan"))
            continue
        values = np.asarray(dens[mask], dtype=float)
        local_indices = np.argwhere(mask)
        best_idx = int(np.argmax(values))
        best_cell = local_indices[best_idx]
        peak_heights.append(float(values[best_idx]))
        if xx is not None and yy is not None:
            peak_locations.append((float(xx[tuple(best_cell)]), float(yy[tuple(best_cell)])))
        else:
            peak_locations.append((float(best_cell[1]), float(best_cell[0])))
    return tuple(peak_locations), tuple(peak_heights)


def _global_max_location(dens: np.ndarray, grid: "DensityGrid | None" = None) -> tuple[tuple[float, float], float]:
    if dens.size == 0:
        return (float("nan"), float("nan")), 0.0
    flat_index = int(np.argmax(dens))
    y_idx, x_idx = np.unravel_index(flat_index, dens.shape)
    peak_height = float(dens[y_idx, x_idx])
    if grid is not None:
        return (float(grid.xx[y_idx, x_idx]), float(grid.yy[y_idx, x_idx])), peak_height
    return (float(x_idx), float(y_idx)), peak_height


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n_components = ndi_label(mask)
    return labels, int(n_components)


def _local_maxima_candidates(dens: np.ndarray) -> list[tuple[int, int, float]]:
    if dens.size == 0:
        return []
    neighborhood = maximum_filter(dens, size=3, mode="nearest")
    maxima = np.asarray(dens == neighborhood, dtype=bool) & np.isfinite(dens) & (dens > 0)
    labels, n_labels = ndi_label(maxima)
    candidates: list[tuple[int, int, float]] = []
    for label_idx in range(1, n_labels + 1):
        mask = labels == label_idx
        if not np.any(mask):
            continue
        values = dens[mask]
        best_idx = int(np.argmax(values))
        local_indices = np.argwhere(mask)
        cell = local_indices[best_idx]
        y_idx, x_idx = int(cell[0]), int(cell[1])
        candidates.append((y_idx, x_idx, float(dens[y_idx, x_idx])))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates


def _dedupe_candidates(
    dens: np.ndarray,
    candidates: list[tuple[int, int, float]],
    *,
    min_separation_cells: float = 2.0,
) -> list[tuple[int, int, float]]:
    if not candidates:
        return []
    kept: list[tuple[int, int, float]] = []
    min_sep_sq = float(min_separation_cells) ** 2
    for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
        y_idx, x_idx, _ = candidate
        if any((y_idx - ky) ** 2 + (x_idx - kx) ** 2 < min_sep_sq for ky, kx, _ in kept):
            continue
        kept.append(candidate)
    return kept


def _approx_saddle_height(dens: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], *, grid: "DensityGrid | None" = None) -> float:
    if dens.size == 0:
        return 0.0
    if grid is None:
        return 0.0
    xs = np.asarray(grid.xx[0, :], dtype=float)
    ys = np.asarray(grid.yy[:, 0], dtype=float)
    if xs.size < 2 or ys.size < 2:
        return 0.0
    interpolator = RegularGridInterpolator((ys, xs), dens, bounds_error=False, fill_value=0.0)
    ts = np.linspace(0.0, 1.0, 64)
    line = np.column_stack([
        p0[1] + (p1[1] - p0[1]) * ts,
        p0[0] + (p1[0] - p0[0]) * ts,
    ])
    values = np.asarray(interpolator(line), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.min(values))


def _assign_cells_to_peaks(
    grid: "DensityGrid",
    peak_locations: tuple[tuple[float, float], ...],
) -> np.ndarray:
    xx = np.asarray(grid.xx, dtype=float)
    yy = np.asarray(grid.yy, dtype=float)
    coords = np.stack([xx, yy], axis=-1)
    labels = np.zeros(xx.shape, dtype=int)
    if not peak_locations:
        return labels
    peak_xy = np.asarray(peak_locations, dtype=float)
    d2 = np.sum((coords[..., None, :] - peak_xy[None, None, :, :]) ** 2, axis=-1)
    labels[:] = np.argmin(d2, axis=-1) + 1
    return labels


def _assign_points_to_peaks(
    points: np.ndarray,
    peak_locations: tuple[tuple[float, float], ...],
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0 or not peak_locations:
        return np.zeros(len(pts), dtype=int)
    peak_xy = np.asarray(peak_locations, dtype=float)
    d2 = np.sum((pts[:, None, :] - peak_xy[None, :, :]) ** 2, axis=-1)
    return np.argmin(d2, axis=-1) + 1


def _format_reject_reasons(reasons: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return tuple(seen)


def _superlevel_split(
    dens: np.ndarray,
    *,
    min_component_mass_frac: float,
    sweep_steps: int,
    start_frac: float,
    stop_frac: float,
) -> tuple[int, float | None, float | None, np.ndarray, np.ndarray, int, tuple[tuple[int, int, float], ...]]:
    peak = float(np.max(dens)) if dens.size else 0.0
    total = _density_mass(dens)
    if peak <= 0 or total <= 0:
        return 0, None, None, np.zeros_like(dens, dtype=int), np.asarray([], dtype=float), 0, ()

    for frac in np.linspace(float(start_frac), float(stop_frac), int(sweep_steps)):
        level = float(frac * peak)
        labels, n_components = _connected_components(dens >= level)
        if n_components < 2:
            continue
        masses = _component_masses(dens, labels, n_components)
        accepted = int(np.sum(masses >= float(min_component_mass_frac)))
        if accepted >= 2:
            details = tuple(
                (int(idx + 1), int(labels[labels == idx + 1].size), float(masses[idx]))
                for idx in range(n_components)
            )
            return accepted, float(frac), level, labels, masses, n_components, details

    return 1, None, 0.5 * peak, np.zeros_like(dens, dtype=int), np.asarray([1.0], dtype=float), 1, ((1, int(dens.size), 1.0),)


def _make_peak_details_from_labels(
    dens: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    *,
    grid: "DensityGrid | None" = None,
    candidate_peak_ids: list[int] | None = None,
    accepted_mask: list[bool] | None = None,
    reject_reasons: list[str] | None = None,
    component_masses: np.ndarray | None = None,
    component_sample_counts: np.ndarray | None = None,
    component_sample_fractions: np.ndarray | None = None,
    superlevel_cap_mass_at_split: np.ndarray | None = None,
    saddle_heights: np.ndarray | None = None,
    basin_kde_masses: np.ndarray | None = None,
    extra_by_candidate: list[dict[str, Any]] | None = None,
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...], tuple[PeakCandidateDetail, ...]]:
    if n_components <= 1 or not np.any(labels):
        location, height = _global_max_location(dens, grid=grid)
        mass = float(component_masses[0]) if component_masses is not None and len(component_masses) else 1.0
        accepted = bool(accepted_mask[0]) if accepted_mask else True
        reject_reason = str(reject_reasons[0]) if reject_reasons else ""
        candidate = PeakCandidateDetail(
            candidate_peak_id=int(candidate_peak_ids[0] if candidate_peak_ids else 1),
            peak_x=float(location[0]),
            peak_y=float(location[1]),
            peak_height=float(height),
            nearest_saddle_or_merge_height=None,
            prominence_ratio=None,
            basin_sample_count=int(component_sample_counts[0]) if component_sample_counts is not None and len(component_sample_counts) else None,
            basin_sample_fraction=float(component_sample_fractions[0]) if component_sample_fractions is not None and len(component_sample_fractions) else None,
            basin_kde_mass=float(basin_kde_masses[0]) if basin_kde_masses is not None and len(basin_kde_masses) else None,
            superlevel_cap_mass_at_split=float(superlevel_cap_mass_at_split[0]) if superlevel_cap_mass_at_split is not None and len(superlevel_cap_mass_at_split) else None,
            accepted=accepted,
            reject_reason=reject_reason,
            component_mass=mass,
            hdr_component_count=None,
            component_area=None,
            component_sample_count=int(component_sample_counts[0]) if component_sample_counts is not None and len(component_sample_counts) else None,
            component_sample_fraction=float(component_sample_fractions[0]) if component_sample_fractions is not None and len(component_sample_fractions) else None,
            extra=dict(extra_by_candidate[0]) if extra_by_candidate is not None and len(extra_by_candidate) else {},
        )
        return ((float(location[0]), float(location[1])),), (float(height),), (candidate,)

    peak_locations, peak_heights = _component_centers_and_heights(dens, labels, n_components, grid=grid)
    candidate_details: list[PeakCandidateDetail] = []
    for idx in range(n_components):
        peak_x, peak_y = peak_locations[idx]
        peak_h = float(peak_heights[idx])
        saddle = float(saddle_heights[idx]) if saddle_heights is not None and idx < len(saddle_heights) else None
        prominence = None
        if saddle is not None and peak_h > 0:
            prominence = float(max(0.0, 1.0 - saddle / peak_h))
        candidate_details.append(
            PeakCandidateDetail(
                candidate_peak_id=int(candidate_peak_ids[idx] if candidate_peak_ids is not None else idx + 1),
                peak_x=peak_x,
                peak_y=peak_y,
                peak_height=peak_h,
                nearest_saddle_or_merge_height=saddle,
                prominence_ratio=prominence,
                basin_sample_count=int(component_sample_counts[idx]) if component_sample_counts is not None and idx < len(component_sample_counts) else None,
                basin_sample_fraction=float(component_sample_fractions[idx]) if component_sample_fractions is not None and idx < len(component_sample_fractions) else None,
                basin_kde_mass=float(basin_kde_masses[idx]) if basin_kde_masses is not None and idx < len(basin_kde_masses) else None,
                superlevel_cap_mass_at_split=float(superlevel_cap_mass_at_split[idx]) if superlevel_cap_mass_at_split is not None and idx < len(superlevel_cap_mass_at_split) else None,
                accepted=bool(accepted_mask[idx]) if accepted_mask is not None and idx < len(accepted_mask) else True,
                reject_reason=str(reject_reasons[idx]) if reject_reasons is not None and idx < len(reject_reasons) else "",
                component_mass=float(component_masses[idx]) if component_masses is not None and idx < len(component_masses) else None,
                extra=dict(extra_by_candidate[idx]) if extra_by_candidate is not None and idx < len(extra_by_candidate) else {},
            )
        )
    return peak_locations, peak_heights, tuple(candidate_details)


def _detect_superlevel_cap_mass(
    density: np.ndarray,
    *,
    grid: "DensityGrid | None" = None,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    start_frac: float = 0.97,
    stop_frac: float = 0.02,
) -> PeakDetectionResult:
    dens = _sanitize_density(density)
    peak = float(np.max(dens)) if dens.size else 0.0
    total = _density_mass(dens)
    accepted, split_fraction, split_level, labels, masses, n_components, details = _superlevel_split(
        dens,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
        start_frac=start_frac,
        stop_frac=stop_frac,
    )
    if peak <= 0 or total <= 0:
        return PeakDetectionResult(
            method_name="superlevel_cap_mass",
            n_modes=0,
            peak_density=peak,
            total_mass=total,
            split_fraction=None,
            split_level=None,
            n_components_at_split=0,
            component_masses=(),
            candidate_peak_count=0,
            accepted_peak_count=0,
            rejected_peak_count=0,
            peak_locations=(),
            peak_heights=(),
            basin_sample_counts=(),
            basin_sample_fractions=(),
            basin_kde_masses=(),
            hdr_component_counts=(),
            reject_reasons=(),
            notes=("no positive density",),
            candidate_details=(),
        )

    reject_reasons: list[str] = []
    accepted_mask: list[bool] = []
    for component_id, _, mass in details:
        accepted_component = mass >= float(min_component_mass_frac)
        accepted_mask.append(accepted_component)
        if not accepted_component:
            reject_reasons.append(f"component_{component_id}_below_mass_cutoff")
    if split_fraction is None or n_components < 2:
        notes = ("no split found",)
        candidate_peak_count = 1
    else:
        notes = (f"first qualifying split at frac={split_fraction:.3f}",)
        candidate_peak_count = int(n_components)
    peak_locations, peak_heights, candidate_details = _make_peak_details_from_labels(
        dens,
        labels if labels.size else np.zeros_like(dens, dtype=int),
        n_components if n_components > 0 else 0,
        grid=grid,
        accepted_mask=accepted_mask if accepted_mask else None,
        reject_reasons=reject_reasons if reject_reasons else None,
        component_masses=masses,
    )
    n_modes = int(accepted)
    superlevel_basin_labels = labels if labels.size else np.zeros_like(dens, dtype=int)
    assert superlevel_basin_labels.shape == dens.shape
    return PeakDetectionResult(
        method_name="superlevel_cap_mass",
        n_modes=n_modes,
        peak_density=peak,
        total_mass=total,
        split_fraction=split_fraction,
        split_level=split_level,
        n_components_at_split=int(n_components),
        component_masses=tuple(float(m) for m in masses),
        candidate_peak_count=int(candidate_peak_count),
        accepted_peak_count=int(n_modes),
        rejected_peak_count=max(0, int(candidate_peak_count - n_modes)),
        peak_locations=peak_locations,
        peak_heights=peak_heights,
        basin_sample_counts=(),
        basin_sample_fractions=(),
        basin_kde_masses=(),
        hdr_component_counts=(),
        reject_reasons=_format_reject_reasons(reject_reasons),
        notes=notes,
        candidate_details=candidate_details,
        basin_labels=superlevel_basin_labels,
    )


def _hdr_level_components(
    dens: np.ndarray,
    *,
    mass_frac: float,
    grid: "DensityGrid | None" = None,
    sample_points: np.ndarray | None = None,
) -> dict[str, Any]:
    flat = np.sort(dens.ravel())[::-1]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {
            "mass_frac": float(mass_frac),
            "density_threshold": None,
            "n_components": 0,
            "component_masses": (),
            "component_areas": (),
            "component_sample_counts": (),
            "component_sample_fractions": (),
            "labels": np.zeros_like(dens, dtype=int),
        }
    csum = np.cumsum(flat)
    total = float(csum[-1])
    if total <= 0:
        return {
            "mass_frac": float(mass_frac),
            "density_threshold": None,
            "n_components": 0,
            "component_masses": (),
            "component_areas": (),
            "component_sample_counts": (),
            "component_sample_fractions": (),
            "labels": np.zeros_like(dens, dtype=int),
        }
    idx = int(np.searchsorted(csum / total, mass_frac, side="left"))
    idx = min(idx, len(flat) - 1)
    threshold = float(flat[idx])
    labels, n_components = _connected_components(dens >= threshold)
    masses = _component_masses(dens, labels, n_components)
    if grid is not None and getattr(grid, "grid", None) is not None:
        cell_area = float(grid.grid.cell_area)
    else:
        cell_area = 1.0
    areas = tuple(float(np.sum(labels == comp_idx) * cell_area) for comp_idx in range(1, n_components + 1))
    if sample_points is not None and len(sample_points) > 0 and n_components > 0 and grid is not None:
        peak_locations, _ = _component_centers_and_heights(dens, labels, n_components, grid=grid)
        point_assignments = _assign_points_to_peaks(sample_points, peak_locations)
        sample_counts = tuple(int(np.sum(point_assignments == comp_idx)) for comp_idx in range(1, n_components + 1))
        total_samples = float(len(sample_points))
        sample_fractions = tuple(float(count / total_samples) for count in sample_counts)
    else:
        sample_counts = tuple()
        sample_fractions = tuple()
    return {
        "mass_frac": float(mass_frac),
        "density_threshold": threshold,
        "n_components": int(n_components),
        "component_masses": tuple(float(m) for m in masses),
        "component_areas": areas,
        "component_sample_counts": sample_counts,
        "component_sample_fractions": sample_fractions,
        "labels": labels,
    }


def _detect_hdr_component_persistence(
    density: np.ndarray,
    *,
    grid: "DensityGrid | None" = None,
    sample_points: np.ndarray | None = None,
    hdr_mass_levels: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
) -> PeakDetectionResult:
    dens = _sanitize_density(density)
    peak = float(np.max(dens)) if dens.size else 0.0
    total = _density_mass(dens)
    if peak <= 0 or total <= 0:
        return PeakDetectionResult(
            method_name="hdr_component_persistence",
            n_modes=0,
            peak_density=peak,
            total_mass=total,
            split_fraction=None,
            split_level=None,
            n_components_at_split=0,
            component_masses=(),
            candidate_peak_count=0,
            accepted_peak_count=0,
            rejected_peak_count=0,
            peak_locations=(),
            peak_heights=(),
            basin_sample_counts=(),
            basin_sample_fractions=(),
            basin_kde_masses=(),
            hdr_component_counts=tuple(),
            reject_reasons=(),
            notes=("no positive density",),
            candidate_details=(),
        )

    level_results = [
        _hdr_level_components(dens, mass_frac=float(mass_frac), grid=grid, sample_points=sample_points)
        for mass_frac in hdr_mass_levels
    ]
    counts = tuple(int(result["n_components"]) for result in level_results)
    positive_counts = [count for count in counts if count > 0]
    if positive_counts:
        count_hist: dict[int, int] = {}
        for count in positive_counts:
            count_hist[count] = count_hist.get(count, 0) + 1
        final_count = max(count_hist.items(), key=lambda item: (item[1], item[0]))[0]
    else:
        final_count = 0
    selected_index = 0
    if counts:
        selected_index = int(np.argmax([count if count == final_count else -1 for count in counts]))
    selected = level_results[selected_index] if level_results else {
        "density_threshold": None,
        "component_masses": (),
        "component_areas": (),
        "component_sample_counts": (),
        "component_sample_fractions": (),
        "labels": np.zeros_like(dens, dtype=int),
        "n_components": 0,
    }
    labels = np.asarray(selected["labels"], dtype=int)
    n_components = int(selected["n_components"])
    component_masses = tuple(float(m) for m in selected["component_masses"])
    peak_locations, peak_heights, candidate_details = _make_peak_details_from_labels(
        dens,
        labels if labels.size else np.zeros_like(dens, dtype=int),
        n_components,
        grid=grid,
        component_masses=np.asarray(component_masses, dtype=float),
        component_sample_counts=np.asarray(selected["component_sample_counts"], dtype=int) if selected["component_sample_counts"] else None,
        component_sample_fractions=np.asarray(selected["component_sample_fractions"], dtype=float) if selected["component_sample_fractions"] else None,
        extra_by_candidate=[{"hdr_mass_level": level_results[selected_index]["mass_frac"]}] * max(1, n_components),
    )
    if n_components >= 2:
        split_level = float(selected["density_threshold"]) if selected["density_threshold"] is not None else None
        split_fraction = float(hdr_mass_levels[selected_index]) if hdr_mass_levels else None
    else:
        split_level = None
        split_fraction = None
    candidate_peak_count = int(max(counts)) if counts else 0
    accepted_peak_count = int(final_count)
    rejected_peak_count = max(0, candidate_peak_count - accepted_peak_count)
    reject_reasons = ()
    notes = (
        f"HDR counts across levels: {','.join(str(count) for count in counts)}",
    ) if counts else ("no HDR components",)
    return PeakDetectionResult(
        method_name="hdr_component_persistence",
        n_modes=accepted_peak_count,
        peak_density=peak,
        total_mass=total,
        split_fraction=split_fraction,
        split_level=split_level,
        n_components_at_split=int(n_components),
        component_masses=component_masses,
        candidate_peak_count=candidate_peak_count,
        accepted_peak_count=accepted_peak_count,
        rejected_peak_count=rejected_peak_count,
        peak_locations=peak_locations,
        peak_heights=peak_heights,
        basin_sample_counts=tuple(int(v) for v in selected["component_sample_counts"]),
        basin_sample_fractions=tuple(float(v) for v in selected["component_sample_fractions"]),
        basin_kde_masses=component_masses,
        hdr_component_counts=counts,
        reject_reasons=reject_reasons,
        notes=notes,
        candidate_details=candidate_details,
        basin_labels=(labels if labels.size else np.zeros_like(dens, dtype=int)),
    )


def _detect_kde_peak_basins_sample_support(
    density: np.ndarray,
    *,
    grid: "DensityGrid | None" = None,
    sample_points: np.ndarray | None = None,
    min_sample_fraction: float = 0.05,
    min_prominence_ratio: float = 0.10,
    min_separation_cells: float = 2.0,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    start_frac: float = 0.97,
    stop_frac: float = 0.02,
) -> PeakDetectionResult:
    dens = _sanitize_density(density)
    peak = float(np.max(dens)) if dens.size else 0.0
    total = _density_mass(dens)
    if peak <= 0 or total <= 0:
        return PeakDetectionResult(
            method_name="kde_peak_basins_sample_support",
            n_modes=0,
            peak_density=peak,
            total_mass=total,
            split_fraction=None,
            split_level=None,
            n_components_at_split=0,
            component_masses=(),
            candidate_peak_count=0,
            accepted_peak_count=0,
            rejected_peak_count=0,
            peak_locations=(),
            peak_heights=(),
            basin_sample_counts=(),
            basin_sample_fractions=(),
            basin_kde_masses=(),
            hdr_component_counts=(),
            reject_reasons=(),
            notes=("no positive density",),
            candidate_details=(),
        )

    candidates = _dedupe_candidates(dens, _local_maxima_candidates(dens), min_separation_cells=min_separation_cells)
    if not candidates:
        location, height = _global_max_location(dens, grid=grid)
        candidates = [(int(round(location[1])), int(round(location[0])), float(height))]

    superlevel_accepted, split_fraction, split_level, split_labels, split_masses, split_n_components, _ = _superlevel_split(
        dens,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
        start_frac=start_frac,
        stop_frac=stop_frac,
    )
    if split_level is None:
        split_level = 0.5 * peak
    if grid is not None:
        component_assignment = _assign_cells_to_peaks(grid, tuple())
    else:
        component_assignment = np.zeros_like(dens, dtype=int)
    # Determine the candidate peak locations first so the assignment can be built.
    peak_locations: list[tuple[float, float]] = []
    peak_heights: list[float] = []
    for y_idx, x_idx, height in candidates:
        if grid is not None:
            peak_locations.append((float(grid.xx[y_idx, x_idx]), float(grid.yy[y_idx, x_idx])))
        else:
            peak_locations.append((float(x_idx), float(y_idx)))
        peak_heights.append(float(height))
    peak_locations_tuple = tuple(peak_locations)
    peak_heights_tuple = tuple(peak_heights)
    if grid is not None and peak_locations_tuple:
        component_assignment = _assign_cells_to_peaks(grid, peak_locations_tuple)
        basin_kde_masses = tuple(
            float(np.sum(dens[component_assignment == comp_idx + 1]) / total)
            for comp_idx in range(len(peak_locations_tuple))
        )
    else:
        basin_kde_masses = tuple()

    if sample_points is not None and len(sample_points) > 0 and peak_locations_tuple:
        sample_assignments = _assign_points_to_peaks(sample_points, peak_locations_tuple)
        sample_counts = tuple(int(np.sum(sample_assignments == comp_idx + 1)) for comp_idx in range(len(peak_locations_tuple)))
        sample_total = float(len(sample_points))
        sample_fractions = tuple(float(count / sample_total) for count in sample_counts)
    else:
        sample_counts = tuple()
        sample_fractions = tuple(float(mass) for mass in basin_kde_masses)

    if grid is not None and peak_locations_tuple:
        interpolator = RegularGridInterpolator(
            (np.asarray(grid.yy[:, 0], dtype=float), np.asarray(grid.xx[0, :], dtype=float)),
            dens,
            bounds_error=False,
            fill_value=0.0,
        )
        saddle_heights: list[float] = []
        for idx, peak_loc in enumerate(peak_locations_tuple):
            if len(peak_locations_tuple) == 1:
                saddle_heights.append(float("nan"))
                continue
            distances = [
                (j, float((peak_loc[0] - other[0]) ** 2 + (peak_loc[1] - other[1]) ** 2))
                for j, other in enumerate(peak_locations_tuple)
                if j != idx
            ]
            nearest_index = min(distances, key=lambda item: item[1])[0]
            saddle_heights.append(
                float(
                    _approx_saddle_height(
                        dens,
                        peak_loc,
                        peak_locations_tuple[nearest_index],
                        grid=grid,
                    )
                )
            )
    else:
        saddle_heights = [float("nan") for _ in peak_locations_tuple]

    accepted_mask: list[bool] = []
    reject_reasons: list[str] = []
    candidate_details: list[PeakCandidateDetail] = []
    for idx, peak_loc in enumerate(peak_locations_tuple):
        peak_height = float(peak_heights_tuple[idx]) if idx < len(peak_heights_tuple) else float("nan")
        saddle_height = saddle_heights[idx] if idx < len(saddle_heights) else float("nan")
        prominence = None
        if np.isfinite(saddle_height) and peak_height > 0:
            prominence = float(max(0.0, 1.0 - (saddle_height / peak_height)))
        basin_fraction = float(sample_fractions[idx]) if idx < len(sample_fractions) else 0.0
        accepted = True
        reasons: list[str] = []
        if len(peak_locations_tuple) > 1:
            if basin_fraction < float(min_sample_fraction):
                accepted = False
                reasons.append(f"sample_fraction_below_{min_sample_fraction:.3f}")
            if prominence is not None and prominence < float(min_prominence_ratio):
                accepted = False
                reasons.append(f"prominence_below_{min_prominence_ratio:.3f}")
        if not accepted:
            reject_reasons.extend(reasons or ["rejected"])
        accepted_mask.append(accepted)
        candidate_details.append(
            PeakCandidateDetail(
                candidate_peak_id=idx + 1,
                peak_x=float(peak_loc[0]),
                peak_y=float(peak_loc[1]),
                peak_height=peak_height,
                nearest_saddle_or_merge_height=float(saddle_height) if np.isfinite(saddle_height) else None,
                prominence_ratio=prominence,
                basin_sample_count=int(sample_counts[idx]) if idx < len(sample_counts) else None,
                basin_sample_fraction=basin_fraction,
                basin_kde_mass=float(basin_kde_masses[idx]) if idx < len(basin_kde_masses) else None,
                superlevel_cap_mass_at_split=float(split_masses[idx]) if idx < len(split_masses) else None,
                accepted=accepted,
                reject_reason=";".join(reasons),
                component_mass=float(basin_kde_masses[idx]) if idx < len(basin_kde_masses) else None,
                extra={
                    "nearest_peak_index": int(idx + 1),
                    "superlevel_accepted_count": int(superlevel_accepted),
                    "split_fraction": split_fraction,
                    "split_level": split_level,
                    "support_proxy": "sample_fraction" if sample_points is not None and len(sample_points) > 0 else "basin_kde_mass",
                },
            )
        )

    candidate_peak_count = len(peak_locations_tuple)
    accepted_peak_count = int(np.sum(np.asarray(accepted_mask, dtype=bool)))
    rejected_peak_count = max(0, candidate_peak_count - accepted_peak_count)
    notes = (
        "nearest-peak sample assignment",
        f"superlevel baseline accepted={superlevel_accepted}",
        "basin support falls back to KDE mass when sample points are unavailable" if sample_points is None or len(sample_points) == 0 else "basin support uses observed sample assignments",
        "ambiguous multi-peak basin with no split" if len(peak_locations_tuple) > 1 and split_fraction is None else None,
    )
    return PeakDetectionResult(
        method_name="kde_peak_basins_sample_support",
        n_modes=accepted_peak_count,
        peak_density=peak,
        total_mass=total,
        split_fraction=split_fraction,
        split_level=split_level,
        n_components_at_split=int(split_n_components),
        component_masses=tuple(float(v) for v in basin_kde_masses),
        candidate_peak_count=int(candidate_peak_count),
        accepted_peak_count=int(accepted_peak_count),
        rejected_peak_count=int(rejected_peak_count),
        peak_locations=peak_locations_tuple,
        peak_heights=peak_heights_tuple,
        basin_sample_counts=tuple(int(v) for v in sample_counts),
        basin_sample_fractions=tuple(float(v) for v in sample_fractions),
        basin_kde_masses=tuple(float(v) for v in basin_kde_masses),
        hdr_component_counts=(),
        reject_reasons=_format_reject_reasons(reject_reasons),
        notes=tuple(note for note in notes if note is not None),
        candidate_details=tuple(candidate_details),
        basin_labels=np.asarray(component_assignment, dtype=int),
    )


def detect_peaks(
    density: np.ndarray,
    *,
    method: str,
    grid: "DensityGrid | None" = None,
    sample_points: np.ndarray | None = None,
    min_component_mass_frac: float = 0.10,
    min_sample_fraction: float = 0.05,
    min_prominence_ratio: float = 0.10,
    hdr_mass_levels: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    sweep_steps: int = 50,
    start_frac: float = 0.97,
    stop_frac: float = 0.02,
) -> PeakDetectionResult:
    """Route peak detection to the requested method."""

    method_name = str(method)
    if method_name not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported peak detector method: {method_name!r}")
    if method_name == "superlevel_cap_mass":
        return _detect_superlevel_cap_mass(
            density,
            grid=grid,
            min_component_mass_frac=min_component_mass_frac,
            sweep_steps=sweep_steps,
            start_frac=start_frac,
            stop_frac=stop_frac,
        )
    if method_name == "hdr_component_persistence":
        return _detect_hdr_component_persistence(
            density,
            grid=grid,
            sample_points=sample_points,
            hdr_mass_levels=hdr_mass_levels,
        )
    return _detect_kde_peak_basins_sample_support(
        density,
        grid=grid,
        sample_points=sample_points,
        min_sample_fraction=min_sample_fraction,
        min_prominence_ratio=min_prominence_ratio,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
        start_frac=start_frac,
        stop_frac=stop_frac,
    )


def peak_count_detail(
    density: np.ndarray,
    *,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    start_frac: float = 0.97,
    stop_frac: float = 0.02,
) -> PeakDetectionResult:
    """Return the compatibility peak-count result for a density field."""

    return detect_peaks(
        density,
        method="superlevel_cap_mass",
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
        start_frac=start_frac,
        stop_frac=stop_frac,
    )


def count_mass_significant_modes(
    density: np.ndarray,
    *,
    min_component_mass_frac: float = 0.10,
    sweep_steps: int = 50,
    start_frac: float = 0.97,
    stop_frac: float = 0.02,
) -> tuple[int, float | None]:
    """Return the compatibility peak count and the split level used for plotting."""

    detail = peak_count_detail(
        density,
        min_component_mass_frac=min_component_mass_frac,
        sweep_steps=sweep_steps,
        start_frac=start_frac,
        stop_frac=stop_frac,
    )
    return detail.n_modes, detail.split_level
