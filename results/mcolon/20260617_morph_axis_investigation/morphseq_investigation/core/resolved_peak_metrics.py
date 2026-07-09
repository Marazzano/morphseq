"""Resolved peak geometry and empirical-null helpers for modal organization V0.

This module turns detector output into canonical resolved-peak geometry and then
summarizes it into scalar metrics suitable for empirical-null testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
from scipy.spatial.distance import pdist

from .density_composition import DensityGrid
from .peak_counting import PeakCandidateDetail, PeakDetectionResult
from .peak_stability import PeakCountStability, PeakSeedSet


SOURCE_TYPES = ("truth", "empirical")

# `radius` is the R80 distance from the peak center (see `_radius_and_cv_from_distances`).
RADIUS_DEFINITION = "r80_distance_from_peak_center"
# within_peak_r80_density scopes both numerator and denominator to the R80 disk:
# 80% of the peak's support, divided by that disk's area. It is a radial
# concentration measure, not a basin/occupancy density estimate — for
# anisotropic peaks the circular disk includes empty area around the peak,
# so a lower reading there reflects elongation as much as reduced concentration.
R80_DENSITY_DEFINITION = "0.8_support_fraction_per_pi_r80_squared"


@dataclass(frozen=True)
class PeakGeometry:
    peak_id: int
    center_coordinate: tuple[float, float]
    total_support_fraction: float
    radius: float
    cv_radius_from_center: float
    within_peak_r80_density: float


@dataclass(frozen=True)
class ResolvedPeak:
    geometry: PeakGeometry
    source_type: Literal["truth", "empirical"]
    detector_detail: PeakCandidateDetail | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeakResolutionEvidence:
    """BACKGROUND sidecar: how the foreground `resolved_peak_count` /
    `is_reliable` answer was reached (COMPOSE_single_path_plan Sec 1.6).
    Only populated by the Stage-2a bootstrap-vote resolution path
    (`compute_resolved_peaks`'s MODE_VOTE_FULL_DATA strategy); `None` on
    distributions built via the plain single-pass constructors below
    (`resolve_empirical_peak_distribution` / `resolve_truth_peak_distribution`),
    which do not vote (plan Sec 1.5c: retention governs PERSISTED evidence).
    """

    target_peak_count: int | None
    resolution_succeeded: bool
    count_is_stable: bool
    count_stability: PeakCountStability
    consensus_seed_set: PeakSeedSet
    full_data_detection: PeakDetectionResult
    # CANDIDATES+ retention (Stage 2b) and STABILITY_GRAPH (Stage 5) are out
    # of scope for Stage 2a; both fields exist on the plan's target shape but
    # are intentionally left unpopulated here.
    bootstrap_tally: Any | None = None
    stability_graph: Any | None = None
    # Per-basin mass fractions computed at the final full-data carve, kept for
    # audit even when resolution failed (rejected basins are NOT exposed as
    # foreground peaks -- plan Sec 1.5d failure semantics).
    basin_component_mass_fractions: tuple[float, ...] = ()
    basin_validation: tuple[bool, ...] = ()


@dataclass(frozen=True)
class ResolvedPeakDistribution:
    distribution_id: str
    source_type: Literal["truth", "empirical"]
    density_grid: DensityGrid
    detection_result: PeakDetectionResult
    peaks: tuple[ResolvedPeak, ...]
    sample_points: np.ndarray | None = None
    sample_peak_ids: np.ndarray | None = None
    grid_peak_ids: np.ndarray | None = None
    # Detector basin-label raster B(x,y) on the density grid (label 0 =
    # background, 1..K = basins), forwarded from the empirical detection result
    # so plotting can draw basin boundaries without reconstruction. Only set on
    # the empirical path; None on the truth path (which uses grid_peak_ids).
    empirical_basin_labels: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    # ---- Stage 2a foreground fields (plan Sec 1.6) -----------------------
    # "This many peaks" -- None if resolution failed (no honest answer, NOT
    # forced to a smaller count). Falls back to `number_of_peaks` (the
    # accepted-candidate count already on `peaks`) for distributions built by
    # the single-pass constructors below, which have no vote: for those, the
    # single detected-and-accepted count IS the answer, so defaulting to it
    # (rather than None) keeps every pre-Stage-2a call site's
    # `resolved_peak_count` reading meaningful without requiring a vote.
    resolved_peak_count: int | None = None
    # "Trust it / don't." Defaults to True for single-pass (non-voting)
    # construction: those paths have no stability/failure concept to report,
    # and forcing a default of False would incorrectly read as "known
    # unreliable" to every existing (smoke test / array-job / null-path)
    # caller that never asked for a vote. Only the Stage-2a vote path can
    # actually set this to False (an unstable vote or a failed basin
    # validation).
    is_reliable: bool = True
    # BACKGROUND: audit / drill-down sidecar. None on all single-pass
    # construction paths; populated only by the Stage-2a vote resolution.
    resolution_evidence: PeakResolutionEvidence | None = None

    def __post_init__(self) -> None:
        if self.resolved_peak_count is None and self.resolution_evidence is None:
            # Single-pass (non-voting) construction: the accepted-candidate
            # count on `peaks` IS the answer (see field docstring above).
            object.__setattr__(self, "resolved_peak_count", len(self.peaks))

        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"Unknown source_type: {self.source_type!r}")

        if tuple(peak.source_type for peak in self.peaks) not in {(), (self.source_type,) * len(self.peaks)}:
            raise ValueError("All resolved peaks must match the distribution source_type.")

        peak_ids = [int(peak.geometry.peak_id) for peak in self.peaks]
        if len(set(peak_ids)) != len(peak_ids):
            raise ValueError("ResolvedPeak peak_id values must be unique.")

        density_shape = np.asarray(self.density_grid.density).shape

        if self.empirical_basin_labels is not None:
            basin_labels = _readonly_copy(self.empirical_basin_labels, dtype=int)
            if basin_labels.shape != density_shape:
                raise ValueError("empirical_basin_labels must match density_grid.density shape.")
            object.__setattr__(self, "empirical_basin_labels", basin_labels)

        if self.source_type == "empirical":
            if self.sample_points is None or self.sample_peak_ids is None:
                raise ValueError(
                    "Empirical distributions require sample_points and sample_peak_ids."
                )
            if self.grid_peak_ids is not None:
                raise ValueError("Empirical distributions must not provide grid_peak_ids.")

            sample_points = _readonly_copy(self.sample_points, dtype=float)
            sample_peak_ids = _readonly_copy(self.sample_peak_ids, dtype=int)
            if sample_points.ndim != 2 or sample_points.shape[1] != 2:
                raise ValueError("sample_points must have shape (n, 2).")
            if len(sample_points) != len(sample_peak_ids):
                raise ValueError("sample_points and sample_peak_ids must have equal length.")

            object.__setattr__(self, "sample_points", sample_points)
            object.__setattr__(self, "sample_peak_ids", sample_peak_ids)

            member_ids = set(int(value) for value in sample_peak_ids.ravel())
        else:
            if self.grid_peak_ids is None:
                raise ValueError("Truth distributions require grid_peak_ids.")
            if self.sample_points is not None or self.sample_peak_ids is not None:
                raise ValueError("Truth distributions must not provide empirical memberships.")

            grid_peak_ids = _readonly_copy(self.grid_peak_ids, dtype=int)
            if grid_peak_ids.shape != density_shape:
                raise ValueError("grid_peak_ids must match density_grid.density shape.")

            object.__setattr__(self, "grid_peak_ids", grid_peak_ids)
            member_ids = set(int(value) for value in grid_peak_ids.ravel())

        nonnegative_member_ids = {member_id for member_id in member_ids if member_id >= 0}
        if not nonnegative_member_ids.issubset(set(peak_ids)):
            missing = sorted(nonnegative_member_ids - set(peak_ids))
            raise ValueError(
                "Every nonnegative membership id must correspond to an accepted ResolvedPeak; "
                f"missing ids: {missing}"
            )

    @property
    def number_of_peaks(self) -> int:
        return len(self.peaks)

    @property
    def assigned_support_fraction(self) -> float:
        if self.source_type == "empirical":
            if self.sample_peak_ids is None:
                return float("nan")
            if len(self.sample_peak_ids) == 0:
                return float("nan")
            return float(np.mean(np.asarray(self.sample_peak_ids, dtype=int) != -1))

        if self.grid_peak_ids is None:
            return float("nan")

        density = np.asarray(self.density_grid.density, dtype=float)
        if self.density_grid.grid is None:
            weights = np.where(np.isfinite(density), density, 0.0)
        else:
            weights = np.where(np.isfinite(density), density, 0.0) * float(self.density_grid.grid.cell_area)
        total = float(np.sum(weights))
        if total <= 0:
            return float("nan")
        return float(np.sum(weights[np.asarray(self.grid_peak_ids) != -1]) / total)

    @property
    def unassigned_support_fraction(self) -> float:
        assigned = self.assigned_support_fraction
        if np.isnan(assigned):
            return float("nan")
        return float(1.0 - assigned)


@dataclass(frozen=True)
class ResolvedPeakDistributionSummary:
    distribution_id: str
    source_type: Literal["truth", "empirical"]
    number_of_peaks: int
    assigned_support_fraction: float
    unassigned_support_fraction: float
    across_peak_total_support_fraction_mean: float
    across_peak_total_support_fraction_skew: float
    across_peak_radius_mean: float
    across_peak_radius_skew: float
    across_peak_cv_radius_from_center_mean: float
    across_peak_distance_mean: float
    across_peak_r80_density_mean: float
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPeakMetricDefinition:
    name: str
    minimum_peak_count: int
    default_alternative: Literal["two-sided", "greater", "less"] = "two-sided"


@dataclass(frozen=True)
class ResolvedPeakRunContext:
    analysis_id: str
    scenario_id: str
    replicate_id: str
    seed: int
    n: int
    bandwidth_rule: str
    bandwidth_multiplier: float
    bandwidth_value: float
    peak_detector_method: str
    canonical_grid_id: str
    assignment_rule: str


@dataclass(frozen=True)
class EmpiricalNullResult:
    observed_value: float
    null_mean: float
    null_std: float
    null_median: float
    null_q025: float
    null_q975: float
    empirical_p_value: float
    standardized_effect: float
    n_null: int
    n_valid_null: int
    valid_null_fraction: float
    test_is_valid: bool


RESOLVED_PEAK_METRICS: dict[str, ResolvedPeakMetricDefinition] = {
    "number_of_peaks": ResolvedPeakMetricDefinition(
        name="number_of_peaks",
        minimum_peak_count=0,
    ),
    "assigned_support_fraction": ResolvedPeakMetricDefinition(
        name="assigned_support_fraction",
        minimum_peak_count=0,
    ),
    "across_peak_total_support_fraction_skew": ResolvedPeakMetricDefinition(
        name="across_peak_total_support_fraction_skew",
        minimum_peak_count=3,
    ),
    "across_peak_radius_mean": ResolvedPeakMetricDefinition(
        name="across_peak_radius_mean",
        minimum_peak_count=1,
    ),
    "across_peak_radius_skew": ResolvedPeakMetricDefinition(
        name="across_peak_radius_skew",
        minimum_peak_count=3,
    ),
    "across_peak_cv_radius_from_center_mean": ResolvedPeakMetricDefinition(
        name="across_peak_cv_radius_from_center_mean",
        minimum_peak_count=1,
    ),
    "across_peak_distance_mean": ResolvedPeakMetricDefinition(
        name="across_peak_distance_mean",
        minimum_peak_count=2,
    ),
    "across_peak_r80_density_mean": ResolvedPeakMetricDefinition(
        name="across_peak_r80_density_mean",
        minimum_peak_count=1,
    ),
}


def _readonly_copy(values: np.ndarray | list[Any] | tuple[Any, ...], *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _mean_or_nan(values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return float("nan")
    return float(np.mean(arr))


def _skew_or_nan(values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or not np.all(np.isfinite(arr)):
        return float("nan")
    mean = float(np.mean(arr))
    centered = arr - mean
    m2 = float(np.mean(centered ** 2))
    if not np.isfinite(m2) or m2 <= 0:
        return float("nan")
    m3 = float(np.mean(centered ** 3))
    skew = m3 / (m2 ** 1.5)
    return float(skew) if np.isfinite(skew) else float("nan")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    values = values[finite]
    weights = weights[finite]
    total = float(np.sum(weights))
    if values.size == 0 or total <= 0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    values = values[finite]
    weights = weights[finite]
    total = float(np.sum(weights))
    if values.size == 0 or total <= 0:
        return float("nan")
    mean = float(np.sum(values * weights) / total)
    variance = float(np.sum(weights * (values - mean) ** 2) / total)
    if variance < 0 or not np.isfinite(variance):
        return float("nan")
    return float(np.sqrt(variance))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights)
    values = values[finite]
    weights = weights[finite]
    total = float(np.sum(weights))
    if values.size == 0 or total <= 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / total
    idx = int(np.searchsorted(cdf, float(q), side="left"))
    idx = min(idx, len(sorted_values) - 1)
    return float(sorted_values[idx])


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.asarray([], dtype=float)
    return pdist(points)


def _validated_candidate_details(
    detection_result: PeakDetectionResult,
) -> tuple[PeakCandidateDetail, ...]:
    details = tuple(detection_result.candidate_details)
    if len(details) != int(detection_result.candidate_peak_count):
        raise ValueError(
            "detection_result.candidate_details must include every candidate peak."
        )

    candidate_ids: list[int] = []
    accepted_count = 0
    for detail in details:
        candidate_id = int(detail.candidate_peak_id)
        if candidate_id in candidate_ids:
            raise ValueError(f"Duplicate candidate_peak_id detected: {candidate_id}")
        candidate_ids.append(candidate_id)

        if not np.isfinite(detail.peak_x) or not np.isfinite(detail.peak_y):
            raise ValueError(f"Candidate {candidate_id} has non-finite peak coordinates.")
        if detail.accepted is None:
            raise ValueError(f"Candidate {candidate_id} is missing accepted status.")
        accepted_count += int(bool(detail.accepted))

    if accepted_count != int(detection_result.accepted_peak_count):
        raise ValueError(
            "Accepted candidate count does not match detection_result.accepted_peak_count."
        )

    return details


def _assign_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if len(points) == 0:
        return np.asarray([], dtype=int)
    if len(centers) == 0:
        return np.full(len(points), -1, dtype=int)
    # Per-center squared-distance columns avoid materializing the full
    # (n_points, n_centers, 2) broadcast array, which dominates for large
    # sample counts since this runs once per null draw.
    distances_sq = np.empty((len(points), len(centers)), dtype=float)
    for center_idx in range(len(centers)):
        diff = points - centers[center_idx]
        distances_sq[:, center_idx] = np.einsum("ij,ij->i", diff, diff)
    return np.argmin(distances_sq, axis=1).astype(int)


def _assign_grid_to_centers(grid: DensityGrid, centers: np.ndarray) -> np.ndarray:
    xx = np.asarray(grid.xx, dtype=float)
    yy = np.asarray(grid.yy, dtype=float)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    assignments = _assign_to_centers(coords, centers)
    return assignments.reshape(xx.shape)


def _radius_and_cv_from_distances(distances: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    distances = np.asarray(distances, dtype=float)
    if weights is None:
        weights = np.ones_like(distances, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    finite = np.isfinite(distances) & np.isfinite(weights)
    distances = distances[finite]
    weights = weights[finite]
    total = float(np.sum(weights))
    if distances.size == 0 or total <= 0:
        return float("nan"), float("nan")

    mean_distance = _weighted_mean(distances, weights)
    if not np.isfinite(mean_distance):
        return float("nan"), float("nan")

    radius = _weighted_quantile(distances, weights, 0.80)
    if not np.isfinite(radius):
        radius = float("nan")

    if mean_distance <= 1e-12:
        cv = float("nan")
    else:
        std_distance = _weighted_std(distances, weights)
        cv = float(std_distance / mean_distance) if np.isfinite(std_distance) else float("nan")
    return float(radius), float(cv)


def _within_peak_r80_density(total_support_fraction: float, radius: float) -> float:
    if not np.isfinite(radius) or radius <= 0:
        return float("nan")
    if not np.isfinite(total_support_fraction):
        return float("nan")
    r80_support_fraction = 0.8 * total_support_fraction
    area = float(np.pi) * radius ** 2
    return float(r80_support_fraction / area)


def _low_density_outlier_mask(
    sample_points: np.ndarray, density_grid: DensityGrid, outlier_density_floor_fraction: float
) -> np.ndarray:
    """Flag samples landing in near-zero-density grid cells as unresolved.

    Nearest-center assignment alone always assigns every sample to some peak,
    even one that is far out in the tail (e.g. a QC-fail embryo). This treats
    the KDE itself as the outlier gate: a sample below `outlier_density_floor_fraction`
    of the global peak height is excluded before assignment is trusted, regardless
    of which center it happens to be nearest to.
    """
    density = np.asarray(density_grid.density, dtype=float)
    peak_density = float(np.nanmax(density)) if density.size else 0.0
    if not np.isfinite(peak_density) or peak_density <= 0:
        return np.zeros(len(sample_points), dtype=bool)
    threshold = outlier_density_floor_fraction * peak_density

    xs = np.asarray(density_grid.grid.xs, dtype=float)
    ys = np.asarray(density_grid.grid.ys, dtype=float)
    x_idx = np.clip(np.searchsorted(xs, sample_points[:, 0]), 0, len(xs) - 1)
    y_idx = np.clip(np.searchsorted(ys, sample_points[:, 1]), 0, len(ys) - 1)
    sample_density = density[y_idx, x_idx]
    return ~np.isfinite(sample_density) | (sample_density < threshold)


def resolve_empirical_peak_distribution(
    *,
    distribution_id: str,
    density_grid: DensityGrid,
    sample_points: np.ndarray,
    detection_result: PeakDetectionResult,
    outlier_density_floor_fraction: float | None = None,
) -> ResolvedPeakDistribution:
    if density_grid.grid is None:
        raise ValueError("density_grid.grid is required for empirical peak resolution.")

    details = _validated_candidate_details(detection_result)
    accepted_details = [detail for detail in details if bool(detail.accepted)]
    centers = np.asarray([(float(detail.peak_x), float(detail.peak_y)) for detail in details], dtype=float)
    rejected_ids = {int(detail.candidate_peak_id) for detail in details if not bool(detail.accepted)}

    sample_points_array = _readonly_copy(sample_points, dtype=float)
    if sample_points_array.ndim != 2 or sample_points_array.shape[1] != 2:
        raise ValueError("sample_points must have shape (n, 2).")

    assignments = _assign_to_centers(sample_points_array, centers)
    if len(assignments) == 0:
        assigned_candidate_ids = np.asarray([], dtype=int)
    elif len(centers) == 0:
        assigned_candidate_ids = np.full(len(assignments), -1, dtype=int)
    else:
        assigned_candidate_ids = np.asarray([int(details[idx].candidate_peak_id) for idx in assignments], dtype=int)
    resolved_sample_peak_ids = np.where(
        np.isin(assigned_candidate_ids, np.asarray(list(rejected_ids), dtype=int)),
        -1,
        assigned_candidate_ids,
    ).astype(int, copy=False)

    if outlier_density_floor_fraction is not None and len(sample_points_array) > 0:
        outlier_mask = _low_density_outlier_mask(
            sample_points_array, density_grid, outlier_density_floor_fraction
        )
        resolved_sample_peak_ids = np.where(outlier_mask, -1, resolved_sample_peak_ids)

    density_total = float(len(resolved_sample_peak_ids))
    resolved_peaks: list[ResolvedPeak] = []
    for detail in accepted_details:
        member_mask = resolved_sample_peak_ids == int(detail.candidate_peak_id)
        member_points = sample_points_array[member_mask]
        support_fraction = float(len(member_points) / density_total) if density_total > 0 else float("nan")
        if len(member_points) == 0:
            radius = float("nan")
            cv = float("nan")
        else:
            distances = np.linalg.norm(
                member_points - np.asarray([detail.peak_x, detail.peak_y], dtype=float),
                axis=1,
            )
            radius, cv = _radius_and_cv_from_distances(distances)
        geometry = PeakGeometry(
            peak_id=int(detail.candidate_peak_id),
            center_coordinate=(float(detail.peak_x), float(detail.peak_y)),
            total_support_fraction=support_fraction,
            radius=radius,
            cv_radius_from_center=cv,
            within_peak_r80_density=_within_peak_r80_density(support_fraction, radius),
        )
        resolved_peaks.append(
            ResolvedPeak(
                geometry=geometry,
                source_type="empirical",
                detector_detail=detail,
                provenance={
                    "accepted": True,
                    "n_assigned_samples": int(len(member_points)),
                },
            )
        )

    return ResolvedPeakDistribution(
        distribution_id=distribution_id,
        source_type="empirical",
        density_grid=density_grid,
        detection_result=detection_result,
        peaks=tuple(resolved_peaks),
        sample_points=sample_points_array,
        sample_peak_ids=_readonly_copy(resolved_sample_peak_ids, dtype=int),
        empirical_basin_labels=(
            detection_result.basin_labels
            if detection_result.basin_labels is not None
            else None
        ),
        provenance={
            "assignment_rule": "nearest_candidate_center_then_reject",
            "outlier_density_floor_fraction": outlier_density_floor_fraction,
        },
    )


def resolve_truth_peak_distribution(
    *,
    distribution_id: str,
    density_grid: DensityGrid,
    detection_result: PeakDetectionResult,
) -> ResolvedPeakDistribution:
    if density_grid.grid is None:
        raise ValueError("density_grid.grid is required for truth peak resolution.")

    details = _validated_candidate_details(detection_result)
    accepted_details = [detail for detail in details if bool(detail.accepted)]
    centers = np.asarray([(float(detail.peak_x), float(detail.peak_y)) for detail in details], dtype=float)
    candidate_ids = np.asarray([int(detail.candidate_peak_id) for detail in details], dtype=int)
    rejected_ids = {int(detail.candidate_peak_id) for detail in details if not bool(detail.accepted)}

    grid_assignments = _assign_grid_to_centers(density_grid, centers).astype(int, copy=False)
    if len(candidate_ids) == 0:
        grid_peak_ids = np.full(grid_assignments.shape, -1, dtype=int)
    else:
        grid_peak_ids = candidate_ids[grid_assignments] if grid_assignments.size else np.asarray(grid_assignments, dtype=int)
    if rejected_ids:
        grid_peak_ids = np.where(np.isin(grid_peak_ids, np.asarray(list(rejected_ids), dtype=int)), -1, grid_peak_ids)

    density = np.asarray(density_grid.density, dtype=float)
    cell_area = float(density_grid.grid.cell_area)
    weights = np.where(np.isfinite(density), density, 0.0) * cell_area
    total_weight = float(np.sum(weights))

    resolved_peaks: list[ResolvedPeak] = []
    xx = np.asarray(density_grid.xx, dtype=float)
    yy = np.asarray(density_grid.yy, dtype=float)
    for detail in accepted_details:
        member_mask = grid_peak_ids == int(detail.candidate_peak_id)
        member_weights = weights[member_mask]
        support_fraction = float(np.sum(member_weights) / total_weight) if total_weight > 0 else float("nan")
        if not np.any(member_mask) or np.sum(member_weights) <= 0:
            radius = float("nan")
            cv = float("nan")
        else:
            member_coords = np.column_stack([xx[member_mask], yy[member_mask]])
            distances = np.linalg.norm(
                member_coords - np.asarray([detail.peak_x, detail.peak_y], dtype=float),
                axis=1,
            )
            radius, cv = _radius_and_cv_from_distances(distances, member_weights)
        geometry = PeakGeometry(
            peak_id=int(detail.candidate_peak_id),
            center_coordinate=(float(detail.peak_x), float(detail.peak_y)),
            total_support_fraction=support_fraction,
            radius=radius,
            cv_radius_from_center=cv,
            within_peak_r80_density=_within_peak_r80_density(support_fraction, radius),
        )
        resolved_peaks.append(
            ResolvedPeak(
                geometry=geometry,
                source_type="truth",
                detector_detail=detail,
                provenance={
                    "accepted": True,
                    "cell_area": cell_area,
                },
            )
        )

    return ResolvedPeakDistribution(
        distribution_id=distribution_id,
        source_type="truth",
        density_grid=density_grid,
        detection_result=detection_result,
        peaks=tuple(resolved_peaks),
        grid_peak_ids=_readonly_copy(grid_peak_ids, dtype=int),
        provenance={"assignment_rule": "nearest_candidate_center_then_reject"},
    )


def summarize_resolved_peak_distribution(
    distribution: ResolvedPeakDistribution,
) -> ResolvedPeakDistributionSummary:
    peaks = tuple(distribution.peaks)
    number_of_peaks = len(peaks)
    total_support = np.asarray([peak.geometry.total_support_fraction for peak in peaks], dtype=float)
    radii = np.asarray([peak.geometry.radius for peak in peaks], dtype=float)
    cv_radii = np.asarray([peak.geometry.cv_radius_from_center for peak in peaks], dtype=float)
    r80_densities = np.asarray([peak.geometry.within_peak_r80_density for peak in peaks], dtype=float)
    centers = np.asarray([peak.geometry.center_coordinate for peak in peaks], dtype=float)

    assigned_support_fraction = distribution.assigned_support_fraction
    unassigned_support_fraction = distribution.unassigned_support_fraction

    if number_of_peaks == 0:
        across_peak_total_support_fraction_mean = float("nan")
        across_peak_total_support_fraction_skew = float("nan")
        across_peak_radius_mean = float("nan")
        across_peak_radius_skew = float("nan")
        across_peak_cv_radius_from_center_mean = float("nan")
        across_peak_distance_mean = float("nan")
        across_peak_r80_density_mean = float("nan")
    else:
        across_peak_total_support_fraction_mean = _mean_or_nan(total_support)
        across_peak_total_support_fraction_skew = _skew_or_nan(total_support)
        across_peak_radius_mean = _mean_or_nan(radii)
        across_peak_radius_skew = _skew_or_nan(radii)
        across_peak_cv_radius_from_center_mean = _mean_or_nan(cv_radii)
        across_peak_distance_mean = _mean_or_nan(_pairwise_distances(centers))
        across_peak_r80_density_mean = _mean_or_nan(r80_densities)
        if number_of_peaks < 2:
            across_peak_distance_mean = float("nan")

    return ResolvedPeakDistributionSummary(
        distribution_id=distribution.distribution_id,
        source_type=distribution.source_type,
        number_of_peaks=number_of_peaks,
        assigned_support_fraction=float(assigned_support_fraction),
        unassigned_support_fraction=float(unassigned_support_fraction),
        across_peak_total_support_fraction_mean=float(across_peak_total_support_fraction_mean),
        across_peak_total_support_fraction_skew=float(across_peak_total_support_fraction_skew),
        across_peak_radius_mean=float(across_peak_radius_mean),
        across_peak_radius_skew=float(across_peak_radius_skew),
        across_peak_cv_radius_from_center_mean=float(across_peak_cv_radius_from_center_mean),
        across_peak_distance_mean=float(across_peak_distance_mean),
        across_peak_r80_density_mean=float(across_peak_r80_density_mean),
        provenance=dict(distribution.provenance),
    )


def run_empirical_null_test(
    *,
    observed_value: float,
    null_values: np.ndarray,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    min_valid_null_fraction: float = 0.8,
) -> EmpiricalNullResult:
    null_values = np.asarray(null_values, dtype=float).ravel()
    n_null = int(null_values.size)
    valid_null = null_values[np.isfinite(null_values)]
    n_valid_null = int(valid_null.size)
    valid_null_fraction = float(n_valid_null / n_null) if n_null > 0 else float("nan")

    null_mean = float(np.mean(valid_null)) if n_valid_null > 0 else float("nan")
    null_std = float(np.std(valid_null, ddof=0)) if n_valid_null > 0 else float("nan")
    null_median = float(np.median(valid_null)) if n_valid_null > 0 else float("nan")
    if n_valid_null > 0:
        null_q025, null_q975 = (float(v) for v in np.quantile(valid_null, [0.025, 0.975]))
    else:
        null_q025 = float("nan")
        null_q975 = float("nan")

    observed_value = float(observed_value)
    test_is_valid = bool(
        np.isfinite(observed_value)
        and n_valid_null > 0
        and np.isfinite(valid_null_fraction)
        and valid_null_fraction >= float(min_valid_null_fraction)
    )

    if not test_is_valid:
        empirical_p_value = float("nan")
        standardized_effect = float("nan")
        return EmpiricalNullResult(
            observed_value=observed_value,
            null_mean=null_mean,
            null_std=null_std,
            null_median=null_median,
            null_q025=null_q025,
            null_q975=null_q975,
            empirical_p_value=empirical_p_value,
            standardized_effect=standardized_effect,
            n_null=n_null,
            n_valid_null=n_valid_null,
            valid_null_fraction=valid_null_fraction,
            test_is_valid=False,
        )

    if alternative == "greater":
        count = int(np.sum(valid_null >= observed_value))
        empirical_p_value = float((1 + count) / (1 + n_valid_null))
    elif alternative == "less":
        count = int(np.sum(valid_null <= observed_value))
        empirical_p_value = float((1 + count) / (1 + n_valid_null))
    elif alternative == "two-sided":
        null_center = null_median
        observed_distance = abs(observed_value - null_center)
        null_distance = np.abs(valid_null - null_center)
        count = int(np.sum(null_distance >= observed_distance))
        empirical_p_value = float((1 + count) / (1 + n_valid_null))
    else:
        raise ValueError(f"Unsupported alternative: {alternative!r}")

    if np.isfinite(null_std) and null_std > 0:
        standardized_effect = float((observed_value - null_mean) / null_std)
    else:
        standardized_effect = float("nan")

    return EmpiricalNullResult(
        observed_value=observed_value,
        null_mean=null_mean,
        null_std=null_std,
        null_median=null_median,
        null_q025=null_q025,
        null_q975=null_q975,
        empirical_p_value=empirical_p_value,
        standardized_effect=standardized_effect,
        n_null=n_null,
        n_valid_null=n_valid_null,
        valid_null_fraction=valid_null_fraction,
        test_is_valid=True,
    )


__all__ = [
    "EmpiricalNullResult",
    "PeakGeometry",
    "PeakResolutionEvidence",
    "RADIUS_DEFINITION",
    "R80_DENSITY_DEFINITION",
    "RESOLVED_PEAK_METRICS",
    "ResolvedPeak",
    "ResolvedPeakDistribution",
    "ResolvedPeakDistributionSummary",
    "ResolvedPeakMetricDefinition",
    "ResolvedPeakRunContext",
    "resolve_empirical_peak_distribution",
    "resolve_truth_peak_distribution",
    "run_empirical_null_test",
    "summarize_resolved_peak_distribution",
]
