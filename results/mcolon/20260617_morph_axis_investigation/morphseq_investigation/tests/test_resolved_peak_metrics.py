"""Unit tests for the resolved-peak geometry, summary, and null-test layer.

These construct `PeakCandidateDetail`/`PeakDetectionResult`/`DensityGrid` by
hand rather than routing through the full KDE/detector pipeline, so most tests
here are fast and have hand-computable expected answers. The one integration
test (`test_points_to_resolved_pipeline_smoke`) uses fixed, deterministic
point clouds rather than randomly sampled ones, to avoid brittleness to
bandwidth/detector-threshold/SciPy-version drift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from morphseq_investigation.core.density_composition import CanonicalGrid, DensityGrid
from morphseq_investigation.core.peak_counting import PeakCandidateDetail, PeakDetectionResult
from morphseq_investigation.core.resolved_peak_analysis import ResolvedPeakAnalysisSpec, resolve_points_with_analysis_spec
from morphseq_investigation.core.resolved_peak_metrics import (
    resolve_empirical_peak_distribution,
    resolve_truth_peak_distribution,
    run_empirical_null_test,
    summarize_resolved_peak_distribution,
)


def _make_candidate(
    candidate_id: int, x: float, y: float, *, accepted: bool = True, reject_reason: str = ""
) -> PeakCandidateDetail:
    return PeakCandidateDetail(
        candidate_peak_id=candidate_id,
        peak_x=x,
        peak_y=y,
        peak_height=1.0,
        nearest_saddle_or_merge_height=None,
        prominence_ratio=None,
        basin_sample_count=None,
        basin_sample_fraction=None,
        basin_kde_mass=None,
        superlevel_cap_mass_at_split=None,
        accepted=accepted,
        reject_reason=reject_reason,
    )


def _make_detection_result(details: tuple[PeakCandidateDetail, ...]) -> PeakDetectionResult:
    accepted_count = sum(1 for d in details if d.accepted)
    return PeakDetectionResult(
        method_name="test",
        n_modes=accepted_count,
        peak_density=1.0,
        total_mass=1.0,
        split_fraction=None,
        split_level=None,
        n_components_at_split=accepted_count,
        component_masses=tuple(1.0 for _ in details),
        candidate_peak_count=len(details),
        accepted_peak_count=accepted_count,
        rejected_peak_count=len(details) - accepted_count,
        peak_locations=tuple((d.peak_x, d.peak_y) for d in details),
        peak_heights=tuple(d.peak_height for d in details),
        basin_sample_counts=tuple(0 for _ in details),
        basin_sample_fractions=tuple(0.0 for _ in details),
        basin_kde_masses=tuple(0.0 for _ in details),
        hdr_component_counts=tuple(1 for _ in details),
        reject_reasons=tuple(d.reject_reason for d in details),
        notes=(),
        candidate_details=details,
    )


def _make_density_grid(*, x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, grid_size=41) -> DensityGrid:
    grid = CanonicalGrid(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, grid_size=grid_size)
    xx, yy = grid.xx, grid.yy
    density = np.exp(-(xx**2 + yy**2) / 2.0)
    return DensityGrid(xx=xx, yy=yy, density=density, grid=grid)


# ---------------------------------------------------------------------------
# One-peak empirical resolution
# ---------------------------------------------------------------------------


def test_one_peak_empirical_resolution_assigns_all_samples():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    samples = np.array([[0.1, 0.1], [-0.1, 0.2], [0.3, -0.1]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )

    assert distribution.number_of_peaks == 1
    assert distribution.assigned_support_fraction == 1.0
    np.testing.assert_array_equal(distribution.sample_peak_ids, [0, 0, 0])


# ---------------------------------------------------------------------------
# Rejected candidate produces -1 membership
# ---------------------------------------------------------------------------


def test_rejected_candidate_produces_unassigned_membership():
    details = (
        _make_candidate(0, -3.0, 0.0),
        _make_candidate(1, 3.0, 0.0, accepted=False, reject_reason="low_mass"),
    )
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    # All samples are nearest to the rejected candidate at (3, 0).
    samples = np.array([[2.9, 0.0], [3.1, 0.1], [2.8, -0.1]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )

    assert distribution.number_of_peaks == 1
    np.testing.assert_array_equal(distribution.sample_peak_ids, [-1, -1, -1])
    assert distribution.assigned_support_fraction == 0.0


# ---------------------------------------------------------------------------
# Per-peak support fractions sum to assigned_support_fraction
# ---------------------------------------------------------------------------


def test_per_peak_support_fractions_sum_to_assigned_support_fraction():
    details = (_make_candidate(0, -3.0, 0.0), _make_candidate(1, 3.0, 0.0))
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    samples = np.array([[-3.0, 0.0], [-2.9, 0.1], [3.0, 0.0], [3.1, -0.1], [2.9, 0.0]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )
    summary = summarize_resolved_peak_distribution(distribution)

    per_peak_sum = sum(peak.geometry.total_support_fraction for peak in distribution.peaks)
    assert per_peak_sum == pytest.approx(summary.assigned_support_fraction)


# ---------------------------------------------------------------------------
# within_peak_r80_density on a hand-computable case
# ---------------------------------------------------------------------------


def test_within_peak_r80_density_hand_computable():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()

    # All samples at a fixed known radius from center -> R80 == that radius exactly.
    n = 100
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    known_radius = 2.0
    samples = np.column_stack([known_radius * np.cos(angles), known_radius * np.sin(angles)])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )

    peak = distribution.peaks[0]
    assert peak.geometry.radius == pytest.approx(known_radius, rel=1e-6)
    expected_density = (0.8 * peak.geometry.total_support_fraction) / (math.pi * known_radius**2)
    assert peak.geometry.within_peak_r80_density == pytest.approx(expected_density, rel=1e-6)


def test_within_peak_r80_density_nan_when_radius_zero_or_nonfinite():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    # A single sample exactly at the peak center -> radius quantile is 0.
    samples = np.array([[0.0, 0.0]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )
    peak = distribution.peaks[0]
    assert peak.geometry.radius == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(peak.geometry.within_peak_r80_density)


# ---------------------------------------------------------------------------
# Scale test: dimensional correctness of radius/density/cv/support
# ---------------------------------------------------------------------------


def test_within_peak_r80_density_scales_correctly_with_geometry():
    def build_distribution(scale: float):
        details = (_make_candidate(0, 0.0, 0.0),)
        detection_result = _make_detection_result(details)
        density_grid = _make_density_grid(
            x_min=-5.0 * scale, x_max=5.0 * scale, y_min=-5.0 * scale, y_max=5.0 * scale,
        )
        n = 100
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 1.0 * scale
        samples = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        return resolve_empirical_peak_distribution(
            distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
        )

    base = build_distribution(1.0)
    scaled = build_distribution(2.0)

    base_peak = base.peaks[0].geometry
    scaled_peak = scaled.peaks[0].geometry

    assert scaled_peak.radius == pytest.approx(2.0 * base_peak.radius, rel=1e-6)
    assert scaled_peak.total_support_fraction == pytest.approx(base_peak.total_support_fraction, rel=1e-6)
    # cv_radius_from_center is scale-invariant: all points sit at the same radius from
    # center in both cases, so std/mean of distances is ~0 (floating-point noise) regardless
    # of absolute scale -- check it's near-zero in both, not that it's exactly equal.
    assert base_peak.cv_radius_from_center == pytest.approx(0.0, abs=1e-9)
    assert scaled_peak.cv_radius_from_center == pytest.approx(0.0, abs=1e-9)
    # Area scales with radius**2, so a doubled radius quarters the density at fixed support.
    assert scaled_peak.within_peak_r80_density == pytest.approx(base_peak.within_peak_r80_density / 4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Truth-grid assignment shape
# ---------------------------------------------------------------------------


def test_truth_grid_assignment_shape_matches_density_shape():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()

    distribution = resolve_truth_peak_distribution(
        distribution_id="t", density_grid=density_grid, detection_result=detection_result,
    )

    assert distribution.grid_peak_ids.shape == np.asarray(density_grid.density).shape


# ---------------------------------------------------------------------------
# radius (R80) and cv_radius_from_center on a hand-computable case
# ---------------------------------------------------------------------------


def test_radius_and_cv_on_points_on_a_circle():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()

    n = 200
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius_value = 3.0
    samples = np.column_stack([radius_value * np.cos(angles), radius_value * np.sin(angles)])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )
    peak = distribution.peaks[0]
    assert peak.geometry.radius == pytest.approx(radius_value, rel=1e-6)
    # All points equidistant from center -> zero variance in distance -> cv is NaN
    # per the spec's edge-case convention (mean radius > epsilon but std == 0 -> cv finite 0,
    # but the spec calls for NaN when the underlying distances have no spread signal only
    # via the ddof=0 std path — here std is exactly 0, so cv == 0, not NaN).
    assert peak.geometry.cv_radius_from_center == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Summary NaN conventions for 0 / 1 / 2 peaks
# ---------------------------------------------------------------------------


def test_summary_nan_conventions_zero_peaks():
    details = (_make_candidate(0, 0.0, 0.0, accepted=False, reject_reason="low_mass"),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    samples = np.array([[0.0, 0.0], [0.1, 0.1]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )
    summary = summarize_resolved_peak_distribution(distribution)

    assert summary.number_of_peaks == 0
    assert math.isnan(summary.across_peak_radius_mean)
    assert math.isnan(summary.across_peak_distance_mean)
    assert math.isnan(summary.across_peak_r80_density_mean)
    assert math.isnan(summary.across_peak_cv_radius_from_center_mean)


def test_summary_nan_conventions_one_peak():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    samples = np.array([[0.1, 0.1], [-0.1, 0.2]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )
    summary = summarize_resolved_peak_distribution(distribution)

    assert summary.number_of_peaks == 1
    assert not math.isnan(summary.across_peak_radius_mean)
    # distance mean requires >= 2 peaks
    assert math.isnan(summary.across_peak_distance_mean)
    # skew requires >= 3 peaks
    assert math.isnan(summary.across_peak_radius_skew)


def test_summary_nan_conventions_two_peaks():
    details = (_make_candidate(0, -3.0, 0.0), _make_candidate(1, 3.0, 0.0))
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    samples = np.array([[-3.0, 0.1], [-2.9, -0.1], [3.0, 0.1], [3.1, -0.1]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )
    summary = summarize_resolved_peak_distribution(distribution)

    assert summary.number_of_peaks == 2
    assert not math.isnan(summary.across_peak_distance_mean)
    assert summary.across_peak_distance_mean == pytest.approx(6.0, abs=0.5)
    # skew requires >= 3 peaks
    assert math.isnan(summary.across_peak_radius_skew)


# ---------------------------------------------------------------------------
# run_empirical_null_test: p-value, n_valid_null / valid_null_fraction accounting
# ---------------------------------------------------------------------------


def test_empirical_null_test_basic_p_value():
    null_values = np.zeros(100)
    result = run_empirical_null_test(observed_value=5.0, null_values=null_values, alternative="greater")
    assert result.test_is_valid
    assert result.n_null == 100
    assert result.n_valid_null == 100
    assert result.valid_null_fraction == 1.0
    # observed_value (5.0) is far outside all null draws (0.0) -> smallest possible p-value.
    assert result.empirical_p_value == pytest.approx(1 / 101)


def test_empirical_null_test_invalid_when_too_many_nan_nulls():
    null_values = np.concatenate([np.full(90, np.nan), np.zeros(10)])
    result = run_empirical_null_test(
        observed_value=5.0, null_values=null_values, alternative="greater", min_valid_null_fraction=0.8,
    )
    assert result.n_null == 100
    assert result.n_valid_null == 10
    assert result.valid_null_fraction == pytest.approx(0.1)
    assert not result.test_is_valid
    assert math.isnan(result.empirical_p_value)
    assert math.isnan(result.standardized_effect)


def test_empirical_null_test_nan_standardized_effect_on_zero_variance_null():
    # Discrete-metric-like null: all draws identical -> null_std == 0.
    null_values = np.zeros(50)
    result = run_empirical_null_test(observed_value=0.0, null_values=null_values, alternative="two-sided")
    assert result.test_is_valid
    assert result.null_std == 0.0
    assert math.isnan(result.standardized_effect)


# ---------------------------------------------------------------------------
# outlier_density_floor_fraction gate
# ---------------------------------------------------------------------------


def test_outlier_density_floor_fraction_excludes_low_density_samples():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    # Three samples near the peak center, one far out in near-zero density.
    samples = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, 0.05], [4.9, 4.9]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
        outlier_density_floor_fraction=0.01,
    )

    np.testing.assert_array_equal(distribution.sample_peak_ids, [0, 0, 0, -1])
    assert distribution.assigned_support_fraction == pytest.approx(0.75)


def test_outlier_density_floor_fraction_none_keeps_all_assigned():
    details = (_make_candidate(0, 0.0, 0.0),)
    detection_result = _make_detection_result(details)
    density_grid = _make_density_grid()
    samples = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, 0.05], [4.9, 4.9]])

    distribution = resolve_empirical_peak_distribution(
        distribution_id="t", density_grid=density_grid, sample_points=samples, detection_result=detection_result,
    )

    # Without the gate, the far outlier is still assigned to the only (accepted) peak
    # by nearest-center assignment.
    np.testing.assert_array_equal(distribution.sample_peak_ids, [0, 0, 0, 0])


# ---------------------------------------------------------------------------
# Integration smoke test: points -> KDE -> detect_peaks -> resolve -> summarize
# ---------------------------------------------------------------------------


def test_points_to_resolved_pipeline_smoke():
    # Fixed, deterministic point clouds -- two tight, well-separated clusters.
    # Not randomly sampled, to avoid brittleness to bandwidth/detector-threshold/
    # SciPy-version drift; exact two-peak recovery is reasonable given the
    # well-separated, low-variance, fixed geometry.
    left_offsets = np.array(
        [[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05], [0.02, 0.02], [-0.02, -0.02]]
    )
    right_offsets = left_offsets.copy()
    left_cluster = np.array([-2.0, 0.0]) + left_offsets
    right_cluster = np.array([2.0, 0.0]) + right_offsets
    points = np.concatenate([left_cluster, right_cluster], axis=0)

    canonical_grid = CanonicalGrid(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, grid_size=121)
    spec = ResolvedPeakAnalysisSpec(
        bandwidth_rule="scipy_default",
        bandwidth_multiplier=1.0,
        peak_detector_method="kde_peak_basins_sample_support",
    )

    distribution = resolve_points_with_analysis_spec(
        distribution_id="smoke_test", points=points, canonical_grid=canonical_grid, analysis_spec=spec,
    )

    assert distribution.number_of_peaks == 2
    for peak in distribution.peaks:
        assert math.isfinite(peak.geometry.radius)
        assert math.isfinite(peak.geometry.total_support_fraction)


# ---------------------------------------------------------------------------
# Empirical basin-label raster exposure (Part 1: expose detector B(x,y))
# ---------------------------------------------------------------------------


def test_empirical_basin_labels_raster_exposed_on_two_mode_synthetic():
    # Two well-separated gaussians -> detector should resolve two basins and the
    # empirical resolved distribution should expose the basin-label raster that
    # the detector computed (label 0 = background, 1..K = basins).
    left_offsets = np.array(
        [[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05], [0.02, 0.02], [-0.02, -0.02]]
    )
    right_offsets = left_offsets.copy()
    left_cluster = np.array([-2.0, 0.0]) + left_offsets
    right_cluster = np.array([2.0, 0.0]) + right_offsets
    points = np.concatenate([left_cluster, right_cluster], axis=0)

    canonical_grid = CanonicalGrid(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, grid_size=121)
    spec = ResolvedPeakAnalysisSpec(
        bandwidth_rule="scipy_default",
        bandwidth_multiplier=1.0,
        peak_detector_method="kde_peak_basins_sample_support",
    )

    distribution = resolve_points_with_analysis_spec(
        distribution_id="basin_labels_test",
        points=points,
        canonical_grid=canonical_grid,
        analysis_spec=spec,
    )

    assert distribution.number_of_peaks == 2

    basin_labels = distribution.empirical_basin_labels
    assert basin_labels is not None
    # Same grid as the density surface.
    assert basin_labels.shape == np.asarray(distribution.density_grid.density).shape
    # Distinct nonzero label count == number_of_peaks.
    distinct_nonzero = sorted(int(v) for v in np.unique(basin_labels) if int(v) != 0)
    assert len(distinct_nonzero) == distribution.number_of_peaks
