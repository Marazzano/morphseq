import dataclasses

import numpy as np
import pytest

from morphseq_investigation.engine.objects import (
    DensityEstimate,
    DensityEstimateSpec,
    Distribution,
    LabelGroup,
    LabelingProvenance,
    UNASSIGNED_LABEL,
)


def _distribution(distribution_id="d", features=("x", "y")):
    rng = np.random.default_rng(3)
    values = rng.normal(size=(16, len(features)))
    return Distribution(
        distribution_id=distribution_id,
        sample_ids=tuple(f"s{i}" for i in range(len(values))),
        feature_names=features,
        feature_values=values,
    )


def test_calc_density_computes_only_and_retains_identity_features_and_spec():
    distribution = _distribution()
    spec = DensityEstimateSpec(grid_params={"resolution": 12})
    density = distribution.calc_density(spec)
    assert density.distribution_id == distribution.distribution_id
    assert density.feature_names == distribution.feature_names
    assert density.spec is spec
    assert density.density_grid.density.shape == (12, 12)
    assert distribution.densities == ()
    assert distribution.shared_density is None
    assert distribution.label_groups == {}


def test_with_density_registers_and_selects_immutably():
    distribution = _distribution()
    first = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 10}))
    second = distribution.calc_density(
        DensityEstimateSpec(bandwidth_multiplier=1.0, grid_params={"resolution": 11})
    )
    registered = distribution.with_density(first)
    assert distribution.densities == ()
    assert registered.densities == (first,)
    assert registered.shared_density is first  # first-retained intelligent default
    selected = registered.with_density(second, select_as_shared=True)
    assert selected.densities == (first, second)
    assert selected.shared_density is second
    assert registered.shared_density is first


def test_reselect_retained_density_a_to_b_to_a_preserves_label_groups():
    distribution = _distribution()
    first = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    second = distribution.calc_density(
        DensityEstimateSpec(bandwidth_multiplier=1.1, grid_params={"resolution": 9})
    )
    retained = distribution.with_density(first).with_density(second)
    labeled = retained.with_label("provided", {"s0": "a"})

    selected_b = labeled.select_shared_density(second)
    selected_a = selected_b.select_shared_density(0)

    assert selected_b.shared_density is second
    assert selected_a.shared_density is first
    assert selected_a.densities == labeled.densities
    assert selected_a.label_groups["provided"] is labeled.label_groups["provided"]
    with pytest.raises(ValueError, match="not retained"):
        labeled.select_shared_density(distribution.calc_density())
    with pytest.raises(IndexError, match="must index"):
        labeled.select_shared_density(2)


@pytest.mark.parametrize("n_features", [1, 3])
def test_calc_density_supports_general_nd_bandwidth(n_features):
    distribution = _distribution(features=tuple(f"x{i}" for i in range(n_features)))
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 7}))
    assert density.density_grid.density.shape == (7,) * n_features
    assert np.all(np.isfinite(density.density_grid.density))


def test_density_registry_rejects_identity_feature_index_and_collision_errors():
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    with pytest.raises(ValueError, match="distribution identity"):
        _distribution("other").with_density(density)
    with pytest.raises(ValueError, match="ordered feature"):
        _distribution(features=("y", "x")).with_density(density)
    with pytest.raises(ValueError, match="equivalent"):
        distribution.with_density(density).with_density(density)
    with pytest.raises(ValueError, match="shared_density_index"):
        dataclasses.replace(distribution, densities=(density,), shared_density_index=-1)


def test_density_estimate_validates_grid_contract():
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    with pytest.raises(ValueError, match="ordered feature"):
        DensityEstimate(
            distribution_id="d",
            feature_names=("y", "x"),
            spec=density.spec,
            grid=density.grid,
            density_grid=density.density_grid,
        )


def test_unified_label_group_and_shared_density_fallback_are_stable():
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    distribution = distribution.with_density(density)
    labeled = distribution.with_label("provided", {"s0": "a"})
    group = labeled.label_groups["provided"]
    assert isinstance(group, LabelGroup)
    assert group.labeling_provenance == LabelingProvenance(method="provided")
    assert group.assignments["s1"] == UNASSIGNED_LABEL
    assert group.sample_set_count == 1
    assert group.peak_count is None and group.is_robust is None
    assert labeled.effective_density("provided") is density


def test_selecting_new_shared_density_does_not_change_resolved_group_density():
    distribution = _distribution()
    generating = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    replacement = distribution.calc_density(
        DensityEstimateSpec(bandwidth_multiplier=1.1, grid_params={"resolution": 9})
    )
    # A minimal object with resolved provenance is intentionally not constructed
    # here; object tests exercise its vote/geometry contracts. This isolates the
    # lifecycle invariant that a label-group override wins over shared fallback.
    group = LabelGroup(
        name="analysis", distribution_id="d",
        assignments={sid: UNASSIGNED_LABEL for sid in distribution.sample_ids},
        density=generating,
    )
    retained = dataclasses.replace(distribution, label_groups={"analysis": group})
    retained = retained.with_density(generating).with_density(replacement, select_as_shared=True)
    assert retained.shared_density is replacement
    assert retained.label_groups["analysis"].density is generating
    assert retained.effective_density("analysis") is generating
