"""TASK_A — build_grid tests (ontology §1b, Invariant #4/#9)."""

import numpy as np
import pytest

from morphseq_investigation.engine.grid import build_grid
from morphseq_investigation.engine.objects import Grid


def _pooled_2d(seed=0, n=40):
    rng = np.random.default_rng(seed)
    pc1 = rng.normal(loc=0.0, scale=2.0, size=n)
    pc2 = rng.normal(loc=5.0, scale=1.0, size=n)
    values = np.stack([pc1, pc2], axis=1)
    sample_ids = tuple(f"s{i}" for i in range(n))
    return values, sample_ids


# --------------------------------------------------------------------------- #
# build_grid — all four methods, feature-unit axes, requested resolution
# --------------------------------------------------------------------------- #
def test_pooled_min_max_feature_unit_axes_and_resolution():
    values, sample_ids = _pooled_2d()
    grid = build_grid(
        ("PC1", "PC2"), values, sample_ids, "pooled_min_max", {"resolution": 25}
    )
    assert isinstance(grid, Grid)
    assert grid.feature_names == ("PC1", "PC2")
    assert len(grid.axis_values) == 2
    for axis in grid.axis_values:
        assert len(axis) == 25
    lo, hi = values.min(axis=0), values.max(axis=0)
    for j, axis in enumerate(grid.axis_values):
        assert axis.min() == pytest.approx(lo[j])
        assert axis.max() == pytest.approx(hi[j])


def test_pooled_quantile_bounds_from_quantiles():
    values, sample_ids = _pooled_2d()
    q_low, q_high = 0.1, 0.9
    grid = build_grid(
        ("PC1", "PC2"),
        values,
        sample_ids,
        "pooled_quantile",
        {"resolution": 15, "q_low": q_low, "q_high": q_high},
    )
    for axis in grid.axis_values:
        assert len(axis) == 15
    expected_lo = np.quantile(values, q_low, axis=0)
    expected_hi = np.quantile(values, q_high, axis=0)
    for j, axis in enumerate(grid.axis_values):
        assert axis.min() == pytest.approx(expected_lo[j])
        assert axis.max() == pytest.approx(expected_hi[j])
    # quantile bounds must be strictly inside the full pooled range
    full_lo, full_hi = values.min(axis=0), values.max(axis=0)
    for j, axis in enumerate(grid.axis_values):
        assert axis.min() > full_lo[j]
        assert axis.max() < full_hi[j]


def test_pooled_mad_scaled_axes_stay_feature_unit_not_whitened():
    values, sample_ids = _pooled_2d()
    grid = build_grid(
        ("PC1", "PC2"),
        values,
        sample_ids,
        "pooled_mad_scaled",
        {"resolution": 20, "mad_multiplier": 3.0},
    )
    for axis in grid.axis_values:
        assert len(axis) == 20
    # Axes must be in the SAME feature-unit range as the pooled data (roughly
    # PC1 in [-6, 6], PC2 in [2, 8]) -- NOT a z-scored/whitened range like
    # [-3, 3] centered at 0 for every axis.
    pc1_axis, pc2_axis = grid.axis_values
    assert pc1_axis.min() < -1.0 and pc1_axis.max() > 1.0
    assert pc2_axis.min() > 0.0 and pc2_axis.max() < 12.0
    # The two axes must NOT be identical ranges (which a whitened/z-scored
    # rendering would produce since both features would be unit-variance).
    assert not np.allclose(pc1_axis, pc2_axis)
    # centered on the pooled median for each feature
    median = np.median(values, axis=0)
    assert pc1_axis.mean() == pytest.approx(median[0], abs=1e-6)
    assert pc2_axis.mean() == pytest.approx(median[1], abs=1e-6)


def test_fixed_bounds_uses_explicit_bounds():
    values, sample_ids = _pooled_2d()
    grid = build_grid(
        ("PC1", "PC2"),
        values,
        sample_ids,
        "fixed_bounds",
        {"resolution": 10, "bounds": [(-10.0, 10.0), (0.0, 20.0)]},
    )
    pc1_axis, pc2_axis = grid.axis_values
    assert pc1_axis.min() == pytest.approx(-10.0)
    assert pc1_axis.max() == pytest.approx(10.0)
    assert pc2_axis.min() == pytest.approx(0.0)
    assert pc2_axis.max() == pytest.approx(20.0)
    assert len(pc1_axis) == 10 and len(pc2_axis) == 10


@pytest.mark.parametrize(
    "method,params",
    [
        ("pooled_min_max", {"resolution": 12}),
        ("pooled_quantile", {"resolution": 12, "q_low": 0.05, "q_high": 0.95}),
        ("pooled_mad_scaled", {"resolution": 12, "mad_multiplier": 4.0}),
        ("fixed_bounds", {"resolution": 12, "bounds": [(-5.0, 5.0), (-5.0, 5.0)]}),
    ],
)
def test_all_methods_produce_valid_grid(method, params):
    values, sample_ids = _pooled_2d()
    grid = build_grid(("PC1", "PC2"), values, sample_ids, method, params)
    assert grid.construction_method == method
    assert grid.fit_sample_ids == tuple(sample_ids)
    assert grid.grid_id.startswith("grid_")


# --------------------------------------------------------------------------- #
# grid_id determinism / order-independence (Invariant #9, TASK_0 contract)
# --------------------------------------------------------------------------- #
def test_shuffled_row_order_same_grid_id():
    values, sample_ids = _pooled_2d()
    grid_a = build_grid(
        ("PC1", "PC2"), values, sample_ids, "pooled_min_max", {"resolution": 20}
    )

    rng = np.random.default_rng(1)
    perm = rng.permutation(len(sample_ids))
    shuffled_values = values[perm]
    shuffled_ids = tuple(np.array(sample_ids)[perm])

    grid_b = build_grid(
        ("PC1", "PC2"), shuffled_values, shuffled_ids, "pooled_min_max", {"resolution": 20}
    )

    assert grid_a.grid_id == grid_b.grid_id
    # And produced axis coordinates must actually match (same bounds either way).
    for a, b in zip(grid_a.axis_values, grid_b.axis_values):
        assert np.allclose(a, b)


def test_different_pooled_values_different_grid_id():
    values, sample_ids = _pooled_2d(seed=0)
    other_values, _ = _pooled_2d(seed=99)
    grid_a = build_grid(
        ("PC1", "PC2"), values, sample_ids, "pooled_min_max", {"resolution": 20}
    )
    grid_b = build_grid(
        ("PC1", "PC2"), other_values, sample_ids, "pooled_min_max", {"resolution": 20}
    )
    assert grid_a.grid_id != grid_b.grid_id
