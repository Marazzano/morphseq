"""Distribution record/comparison containers for resolved-peak analyses.

This module provides the lean V0 object model described in
`Disturbion_Peak_broad_refactor.md`:

- a persistent empirical distribution record,
- typed density / detector / membership / resolved-peak slots,
- a lightweight comparison container, and
- pure `add_*` enrichment helpers.

The implementation stays close to the current resolved-peak stack. It does not
replace the existing KDE / detector / summary code; it just makes the derived
products explicit and owned by the distribution they describe.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .density_composition import CanonicalGrid
from .peak_counting import PeakDetectionResult, detect_peaks
from .resolved_peak_metrics import (
    RESOLVED_PEAK_METRICS,
    ResolvedPeakDistribution,
    ResolvedPeakDistributionSummary,
    summarize_resolved_peak_distribution,
    resolve_empirical_peak_distribution,
)
from .support_geometry import evaluate_kde_on_grid


def _readonly_array(values: np.ndarray | list[Any] | tuple[Any, ...], *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _readonly_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _assert_product_name_available(existing: Mapping[str, Any], name: str, *, replace_ok: bool) -> None:
    if name in existing and not replace_ok:
        raise KeyError(f"Product {name!r} already exists; pass replace=True to overwrite it.")


def _grid_signature(grid: CanonicalGrid) -> tuple[float, float, float, float, int]:
    return (
        float(grid.x_min),
        float(grid.x_max),
        float(grid.y_min),
        float(grid.y_max),
        int(grid.grid_size),
    )


def _assert_grid_compatibility(left: CanonicalGrid, right: CanonicalGrid) -> None:
    if _grid_signature(left) != _grid_signature(right):
        raise ValueError("DistributionComparison members must share an identical canonical grid.")


def _assign_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if len(points) == 0:
        return np.asarray([], dtype=int)
    if len(centers) == 0:
        return np.full(len(points), -1, dtype=int)
    distances_sq = np.empty((len(points), len(centers)), dtype=float)
    for center_idx in range(len(centers)):
        diff = points - centers[center_idx]
        distances_sq[:, center_idx] = np.einsum("ij,ij->i", diff, diff)
    return np.argmin(distances_sq, axis=1).astype(int)


@dataclass(frozen=True)
class DensityField:
    """Density values evaluated on a canonical grid."""

    grid: CanonicalGrid
    values: np.ndarray
    source_distribution_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _readonly_array(self.values, dtype=float)
        if values.shape != self.grid.xx.shape:
            raise ValueError("DensityField.values shape does not match grid.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    @property
    def xx(self) -> np.ndarray:
        return self.grid.xx

    @property
    def yy(self) -> np.ndarray:
        return self.grid.yy

    @property
    def density(self) -> np.ndarray:
        return self.values

    @property
    def cell_area(self) -> float:
        return float(self.grid.cell_area)


@dataclass(frozen=True)
class EmpiricalPeakMembership:
    distribution_id: str
    sample_peak_ids: np.ndarray
    assignment_rule: str
    candidate_peak_ids: tuple[int, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_peak_ids = _readonly_array(self.sample_peak_ids, dtype=int)
        object.__setattr__(self, "sample_peak_ids", sample_peak_ids)
        object.__setattr__(self, "candidate_peak_ids", tuple(int(value) for value in self.candidate_peak_ids))
        object.__setattr__(self, "provenance", _readonly_mapping(self.provenance))


@dataclass(frozen=True)
class DistributionRecord:
    distribution_id: str
    points: np.ndarray
    canonical_grid: CanonicalGrid
    densities: Mapping[str, DensityField] = field(default_factory=dict)
    peak_detections: Mapping[str, PeakDetectionResult] = field(default_factory=dict)
    peak_memberships: Mapping[str, EmpiricalPeakMembership] = field(default_factory=dict)
    resolved_peak_distributions: Mapping[str, ResolvedPeakDistribution] = field(default_factory=dict)
    resolved_peak_summaries: Mapping[str, ResolvedPeakDistributionSummary] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _readonly_array(self.points, dtype=float)
        if points.ndim != 2:
            raise ValueError("DistributionRecord.points must be a 2-D array.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "densities", _readonly_mapping(self.densities))
        object.__setattr__(self, "peak_detections", _readonly_mapping(self.peak_detections))
        object.__setattr__(self, "peak_memberships", _readonly_mapping(self.peak_memberships))
        object.__setattr__(self, "resolved_peak_distributions", _readonly_mapping(self.resolved_peak_distributions))
        object.__setattr__(self, "resolved_peak_summaries", _readonly_mapping(self.resolved_peak_summaries))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True)
class DistributionComparison:
    comparison_id: str
    members: Mapping[str, DistributionRecord]
    observed_metrics: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        members = _readonly_mapping(self.members)
        if len(members) < 2:
            raise ValueError("DistributionComparison requires at least two members.")
        grids = [member.canonical_grid for member in members.values()]
        first_grid = grids[0]
        for grid in grids[1:]:
            _assert_grid_compatibility(first_grid, grid)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "observed_metrics", _readonly_mapping(self.observed_metrics))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


def add_density(
    distribution: DistributionRecord,
    *,
    name: str,
    kde: Any = None,
    replace: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DistributionRecord:
    _assert_product_name_available(distribution.densities, name, replace_ok=replace)
    values = evaluate_kde_on_grid(distribution.points, distribution.canonical_grid.xx, distribution.canonical_grid.yy, kde=kde)
    field = DensityField(
        grid=distribution.canonical_grid,
        values=values,
        source_distribution_id=distribution.distribution_id,
        metadata=metadata or {},
    )
    updated = dict(distribution.densities)
    updated[name] = field
    return dc_replace(distribution, densities=_readonly_mapping(updated))


def add_peak_detection(
    distribution: DistributionRecord,
    *,
    name: str,
    density_name: str,
    method: str,
    min_component_mass_frac: float = 0.10,
    min_sample_fraction: float = 0.05,
    min_prominence_ratio: float = 0.10,
    outlier_density_floor_fraction: float | None = None,
    replace: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DistributionRecord:
    _assert_product_name_available(distribution.peak_detections, name, replace_ok=replace)
    density = distribution.densities[density_name]
    detection = detect_peaks(
        density.density,
        method=method,
        grid=density,
        sample_points=distribution.points,
        min_component_mass_frac=min_component_mass_frac,
        min_sample_fraction=min_sample_fraction,
        min_prominence_ratio=min_prominence_ratio,
    )
    if outlier_density_floor_fraction is not None:
        detection = dc_replace(
            detection,
            notes=tuple(list(detection.notes) + [f"outlier_density_floor_fraction={outlier_density_floor_fraction}"]),
        )
    if metadata:
        detection = dc_replace(detection, notes=tuple(list(detection.notes) + [f"metadata={dict(metadata)}"]))
    updated = dict(distribution.peak_detections)
    updated[name] = detection
    return dc_replace(distribution, peak_detections=_readonly_mapping(updated))


def add_peak_membership(
    distribution: DistributionRecord,
    *,
    name: str,
    detection_name: str,
    replace: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DistributionRecord:
    _assert_product_name_available(distribution.peak_memberships, name, replace_ok=replace)
    detection = distribution.peak_detections[detection_name]
    details = tuple(detection.candidate_details)
    candidate_peak_ids = tuple(int(detail.candidate_peak_id) for detail in details)
    if details:
        centers = np.asarray([(float(detail.peak_x), float(detail.peak_y)) for detail in details], dtype=float)
        assigned_candidate_indices = _assign_to_centers(distribution.points, centers)
        assigned_candidate_ids = np.asarray([candidate_peak_ids[idx] for idx in assigned_candidate_indices], dtype=int)
        rejected_ids = np.asarray([int(detail.candidate_peak_id) for detail in details if not bool(detail.accepted)], dtype=int)
        sample_peak_ids = np.where(np.isin(assigned_candidate_ids, rejected_ids), -1, assigned_candidate_ids).astype(int)
    else:
        sample_peak_ids = np.full(len(distribution.points), -1, dtype=int)

    membership = EmpiricalPeakMembership(
        distribution_id=distribution.distribution_id,
        sample_peak_ids=sample_peak_ids,
        assignment_rule="nearest_candidate_center_then_reject",
        candidate_peak_ids=candidate_peak_ids,
        provenance=dict(metadata or {}),
    )
    updated = dict(distribution.peak_memberships)
    updated[name] = membership
    return dc_replace(distribution, peak_memberships=_readonly_mapping(updated))


def add_resolved_peak_distribution(
    distribution: DistributionRecord,
    *,
    name: str,
    density_name: str,
    detection_name: str,
    membership_name: str,
    replace: bool = False,
    outlier_density_floor_fraction: float | None = None,
) -> DistributionRecord:
    _assert_product_name_available(distribution.resolved_peak_distributions, name, replace_ok=replace)
    density = distribution.densities[density_name]
    detection = distribution.peak_detections[detection_name]
    membership = distribution.peak_memberships[membership_name]
    resolved = resolve_empirical_peak_distribution(
        distribution_id=distribution.distribution_id,
        density_grid=density,
        sample_points=distribution.points,
        detection_result=detection,
        outlier_density_floor_fraction=outlier_density_floor_fraction,
    )
    if not np.array_equal(np.asarray(resolved.sample_peak_ids, dtype=int), np.asarray(membership.sample_peak_ids, dtype=int)):
        raise ValueError("Membership product does not match resolved peak assignment.")
    updated = dict(distribution.resolved_peak_distributions)
    updated[name] = resolved
    return dc_replace(distribution, resolved_peak_distributions=_readonly_mapping(updated))


def add_resolved_peak_summary(
    distribution: DistributionRecord,
    *,
    name: str,
    resolved_name: str,
    replace: bool = False,
) -> DistributionRecord:
    _assert_product_name_available(distribution.resolved_peak_summaries, name, replace_ok=replace)
    resolved = distribution.resolved_peak_distributions[resolved_name]
    summary = summarize_resolved_peak_distribution(resolved)
    updated = dict(distribution.resolved_peak_summaries)
    updated[name] = summary
    return dc_replace(distribution, resolved_peak_summaries=_readonly_mapping(updated))


def add_observed_metrics(
    comparison: DistributionComparison,
    *,
    name: str = "primary",
    summary_name: str = "primary",
    reference_role: str = "reference",
    target_role: str = "target",
    metric_registry: Mapping[str, Any] = RESOLVED_PEAK_METRICS,
    replace: bool = False,
) -> DistributionComparison:
    _assert_product_name_available(comparison.observed_metrics, name, replace_ok=replace)
    if reference_role not in comparison.members:
        raise KeyError(f"Missing comparison member role: {reference_role!r}")
    if target_role not in comparison.members:
        raise KeyError(f"Missing comparison member role: {target_role!r}")

    reference = comparison.members[reference_role]
    target = comparison.members[target_role]
    _assert_grid_compatibility(reference.canonical_grid, target.canonical_grid)

    try:
        reference_summary = reference.resolved_peak_summaries[summary_name]
    except KeyError as exc:
        raise KeyError(f"Missing summary {summary_name!r} on {reference_role!r}") from exc
    try:
        target_summary = target.resolved_peak_summaries[summary_name]
    except KeyError as exc:
        raise KeyError(f"Missing summary {summary_name!r} on {target_role!r}") from exc

    rows: list[dict[str, Any]] = []
    for metric_name, metric_def in metric_registry.items():
        if not hasattr(reference_summary, metric_name) or not hasattr(target_summary, metric_name):
            raise KeyError(f"Unknown resolved-peak metric: {metric_name!r}")
        reference_value = float(getattr(reference_summary, metric_name))
        target_value = float(getattr(target_summary, metric_name))
        rows.append(
            {
                "comparison_id": comparison.comparison_id,
                "reference_role": reference_role,
                "target_role": target_role,
                "summary_name": summary_name,
                "metric_name": metric_name,
                "minimum_peak_count": getattr(metric_def, "minimum_peak_count", None),
                "observed_reference_value": reference_value,
                "observed_target_value": target_value,
                "observed_difference": target_value - reference_value,
                "reference_distribution_id": reference.distribution_id,
                "target_distribution_id": target.distribution_id,
            }
        )

    table = pd.DataFrame(rows)
    updated = dict(comparison.observed_metrics)
    updated[name] = table
    return dc_replace(comparison, observed_metrics=_readonly_mapping(updated))


__all__ = [
    "DensityField",
    "DistributionComparison",
    "DistributionRecord",
    "EmpiricalPeakMembership",
    "add_density",
    "add_observed_metrics",
    "add_peak_detection",
    "add_peak_membership",
    "add_resolved_peak_distribution",
    "add_resolved_peak_summary",
]
