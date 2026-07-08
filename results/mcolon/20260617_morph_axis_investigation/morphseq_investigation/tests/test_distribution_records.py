"""Tests for the typed distribution record/comparison layer."""

from __future__ import annotations

import numpy as np
import pytest

from morphseq_investigation.core.distribution_records import (
    DistributionComparison,
    DistributionRecord,
    add_density,
    add_observed_metrics,
    add_peak_detection,
    add_peak_membership,
    add_resolved_peak_distribution,
    add_resolved_peak_summary,
)
from morphseq_investigation.core.density_composition import CanonicalGrid
from morphseq_investigation.core.resolved_peak_analysis import ResolvedPeakAnalysisSpec
from morphseq_investigation.core.resolved_peak_metrics import RESOLVED_PEAK_METRICS
from morphseq_investigation.plotting.modal_distribution_plotting import build_distribution_overlay, derive_shared_grid
from morphseq_investigation.core.support_geometry import isotropic_geometry_kde_spec


def _two_cluster_points(center_offset: float = 0.0) -> np.ndarray:
    offsets = np.array(
        [[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05], [0.02, 0.02], [-0.02, -0.02]]
    )
    left = np.array([-2.0 - center_offset, 0.0]) + offsets
    right = np.array([2.0 + center_offset, 0.0]) + offsets
    return np.concatenate([left, right], axis=0)


def _analysis_spec() -> ResolvedPeakAnalysisSpec:
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule="scipy_default",
        bandwidth_multiplier=1.0,
        peak_detector_method="kde_peak_basins_sample_support",
    )


def _build_record(distribution_id: str, points: np.ndarray, grid: CanonicalGrid):
    record = DistributionRecord(
        distribution_id=distribution_id,
        points=points,
        canonical_grid=grid,
        metadata={"kind": "test"},
    )
    record = add_density(record, name="primary", kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))
    record = add_peak_detection(
        record,
        name="primary",
        density_name="primary",
        method=_analysis_spec().peak_detector_method,
        min_sample_fraction=0.10,
    )
    record = add_peak_membership(record, name="primary", detection_name="primary")
    record = add_resolved_peak_distribution(
        record,
        name="primary",
        density_name="primary",
        detection_name="primary",
        membership_name="primary",
    )
    record = add_resolved_peak_summary(record, name="primary", resolved_name="primary")
    return record


def test_distribution_record_pipeline_populates_typed_products():
    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid=61, kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))

    record = _build_record("sample", points, grid)

    assert record.points.flags.writeable is False
    assert set(record.densities) == {"primary"}
    assert set(record.peak_detections) == {"primary"}
    assert set(record.peak_memberships) == {"primary"}
    assert set(record.resolved_peak_distributions) == {"primary"}
    assert set(record.resolved_peak_summaries) == {"primary"}
    assert record.resolved_peak_summaries["primary"].number_of_peaks == record.resolved_peak_distributions["primary"].number_of_peaks


def test_duplicate_product_name_is_rejected_by_default():
    points = _two_cluster_points()
    grid = derive_shared_grid(points, points, grid=61, kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))
    record = DistributionRecord(distribution_id="sample", points=points, canonical_grid=grid)
    record = add_density(record, name="primary", kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))

    with pytest.raises(KeyError):
        add_density(record, name="primary", kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))


def test_distribution_comparison_observed_metrics_and_grid_compatibility():
    target_points = _two_cluster_points(center_offset=0.0)
    reference_points = _two_cluster_points(center_offset=0.5)
    grid = derive_shared_grid(target_points, reference_points, grid=61, kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))

    target = _build_record("target", target_points, grid)
    reference = _build_record("reference", reference_points, grid)
    comparison = DistributionComparison(
        comparison_id="cmp",
        members={"reference": reference, "target": target},
    )
    comparison = add_observed_metrics(comparison, name="primary")
    table = comparison.observed_metrics["primary"]

    assert set(table["metric_name"]) == set(RESOLVED_PEAK_METRICS)
    assert len(table) == len(RESOLVED_PEAK_METRICS)
    assert np.all(table["comparison_id"] == "cmp")
    assert table["observed_target_value"].notna().any()


def test_distribution_comparison_rejects_mismatched_grids():
    points = _two_cluster_points()
    grid_a = derive_shared_grid(points, points, grid=61, kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))
    grid_b = derive_shared_grid(points, points, grid=63, kde=isotropic_geometry_kde_spec("longest_non_outlier_MST_edge", 0.75))

    target = _build_record("target", points, grid_a)
    reference = _build_record("reference", points, grid_b)

    with pytest.raises(ValueError):
        DistributionComparison(comparison_id="cmp", members={"reference": reference, "target": target})


def test_build_distribution_overlay_always_carries_a_canonical_grid():
    points = _two_cluster_points()
    overlay = build_distribution_overlay(points, points, grid=51, kde=None)

    assert overlay.target_grid.grid is not None
    assert overlay.reference_grid.grid is not None
