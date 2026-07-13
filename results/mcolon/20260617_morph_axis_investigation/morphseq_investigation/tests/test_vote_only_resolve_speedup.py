"""Regression tests for the bootstrap-vote speed optimizations (Steps 1-2 of
the resolve-hot-path speedup plan): detection-only primitive extraction and
connectivity-computation gating.

These protect behavior, not just numeric output -- see test 3 in particular,
which guards the *architecture* (vote path must never construct a full
`ResolvedPeakDistribution`), not merely that the vote's final numbers match.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from morphseq_investigation.core import bandwidth_tuning
from morphseq_investigation.core.density_composition import CanonicalGrid
from morphseq_investigation.core.distribution_records import (
    _resolve_draw_for_vote,
    derive_shared_grid,
)
from morphseq_investigation.core.resolved_peak_analysis import (
    ResolvedPeakAnalysisSpec,
    _compute_peak_detection_with_analysis_spec,
    _resolve_points_single_pass,
)


def _two_cluster_points() -> np.ndarray:
    offsets = np.array(
        [[-0.05, 0.0], [0.05, 0.0], [0.0, -0.05], [0.0, 0.05], [0.02, 0.02], [-0.02, -0.02]]
    )
    left = np.array([-2.0, 0.0]) + offsets
    right = np.array([2.0, 0.0]) + offsets
    return np.concatenate([left, right], axis=0)


def _one_cluster_points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=[0.0, 0.0], scale=0.1, size=(20, 2))


def _grid_for(points: np.ndarray, grid_size: int = 61) -> CanonicalGrid:
    return derive_shared_grid(points, points, grid_size=grid_size)


def _geometry_spec() -> ResolvedPeakAnalysisSpec:
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule="median_kNN_distance",
        bandwidth_multiplier=1.0,
        peak_detector_method="kde_peak_basins_sample_support",
        min_sample_fraction=0.10,
    )


def _mst_spec() -> ResolvedPeakAnalysisSpec:
    return ResolvedPeakAnalysisSpec(
        bandwidth_rule="longest_non_outlier_MST_edge",
        bandwidth_multiplier=0.75,
        peak_detector_method="kde_peak_basins_sample_support",
        min_sample_fraction=0.10,
    )


# --- Test 1: detection-only equivalence (protects Step 1) -------------------


@pytest.mark.parametrize(
    "points_fn,spec_fn",
    [
        (_two_cluster_points, _mst_spec),
        (_two_cluster_points, _geometry_spec),
        (_one_cluster_points, _mst_spec),
    ],
)
def test_detection_only_matches_full_resolve_accepted_peaks(points_fn, spec_fn):
    points = points_fn()
    spec = spec_fn()
    grid = _grid_for(points)

    _density_grid, _points_array, detection_result = _compute_peak_detection_with_analysis_spec(
        points, grid, spec,
    )
    resolved = _resolve_points_single_pass(
        distribution_id="full", points=points, canonical_grid=grid, analysis_spec=spec,
    )

    accepted = tuple(d for d in detection_result.candidate_details if d.accepted)
    assert len(accepted) == resolved.number_of_peaks

    detection_centers = sorted((float(d.peak_x), float(d.peak_y)) for d in accepted)
    resolved_centers = sorted(peak.geometry.center_coordinate for peak in resolved.peaks)
    assert len(detection_centers) == len(resolved_centers)
    for (dx, dy), (rx, ry) in zip(detection_centers, resolved_centers):
        np.testing.assert_allclose([dx, dy], [rx, ry], rtol=0, atol=1e-9)


# --- Test 2: connectivity gating (protects Step 2) --------------------------


def test_connectivity_skipped_when_not_requested():
    points = _two_cluster_points()
    with patch.object(
        bandwidth_tuning, "_graph_connectivity_radius_from_distance_matrix",
        wraps=bandwidth_tuning._graph_connectivity_radius_from_distance_matrix,
    ) as spy:
        scales = bandwidth_tuning.bandwidth_geometry_scales(points, include_connectivity_radius=False)
    spy.assert_not_called()
    assert np.isnan(scales["connectivity_90_radius"])
    assert np.isnan(scales["graph_connectivity_radius"])


def test_connectivity_computed_and_reuses_distance_matrix_when_requested():
    points = _two_cluster_points()
    with patch(
        "morphseq_investigation.core.bandwidth_tuning.pdist",
        wraps=bandwidth_tuning.pdist,
    ) as pdist_spy:
        scales_with = bandwidth_tuning.bandwidth_geometry_scales(points, include_connectivity_radius=True)
        pdist_call_count_with = pdist_spy.call_count

    with patch(
        "morphseq_investigation.core.bandwidth_tuning.pdist",
        wraps=bandwidth_tuning.pdist,
    ) as pdist_spy_without:
        bandwidth_tuning.bandwidth_geometry_scales(points, include_connectivity_radius=False)
        pdist_call_count_without = pdist_spy_without.call_count

    # include_connectivity_radius=True must not trigger a second/independent
    # pdist call beyond the one bandwidth_geometry_scales already does for
    # median_kNN/MST -- i.e. connectivity reuses the shared distance matrix.
    assert pdist_call_count_with == pdist_call_count_without == 1
    assert np.isfinite(scales_with["connectivity_90_radius"])
    assert scales_with["connectivity_90_radius"] > 0


def test_connectivity_radius_value_unchanged_by_gating_refactor():
    # The gated path must produce the identical numeric connectivity value as
    # calling the (now matrix-based) helper directly on the same points.
    points = _two_cluster_points()
    scales = bandwidth_tuning.bandwidth_geometry_scales(points, include_connectivity_radius=True)

    from scipy.spatial.distance import pdist, squareform

    dmat = squareform(pdist(points))
    direct = bandwidth_tuning._graph_connectivity_radius_from_distance_matrix(
        dmat, target_mass=bandwidth_tuning.DEFAULT_CONNECTIVITY_MASS,
    )
    assert scales["connectivity_90_radius"] == pytest.approx(direct)


# --- Test 3: vote path never triggers full resolution (protects Step 1's ----
# --- architecture, not just its output) --------------------------------------


def test_vote_only_resolve_never_constructs_full_resolved_distribution():
    points = _two_cluster_points()
    grid = _grid_for(points)
    spec = _mst_spec()

    resolve_draw = _resolve_draw_for_vote(grid, spec)

    with patch(
        "morphseq_investigation.core.resolved_peak_analysis.resolve_empirical_peak_distribution",
        side_effect=AssertionError(
            "vote-only resolution must not construct a resolved distribution"
        ),
    ):
        count, centers = resolve_draw(points)

    assert count >= 0
    assert isinstance(centers, tuple)
