import dataclasses

import numpy as np
import pytest

from morphseq_investigation.core.peak_stability import (
    PeakCountRobustnessPolicy,
    PeakVotingSpec,
)
from morphseq_investigation.engine.objects import (
    DensityEstimateSpec,
    Distribution,
    LabelGroup,
    UNASSIGNED_LABEL,
)


def _distribution(distribution_id="d", features=("x", "y")):
    rng = np.random.default_rng(13)
    values = rng.normal(size=(16, len(features)))
    return Distribution(
        distribution_id=distribution_id,
        sample_ids=tuple(f"s{i}" for i in range(len(values))),
        feature_names=features,
        feature_values=values,
    )


def _stub_labeler(monkeypatch, calls):
    def fake(distribution, *, output_label, density, voting_spec, robustness_policy):
        calls.append((distribution, output_label, density, voting_spec, robustness_policy))
        return LabelGroup(
            name=output_label,
            distribution_id=distribution.distribution_id,
            assignments={sid: UNASSIGNED_LABEL for sid in distribution.sample_ids},
            density=density,
        )

    monkeypatch.setattr(
        "morphseq_investigation.engine.labelers.detect_peaks", fake
    )


def _detect(distribution, **kwargs):
    return distribution.detect_peaks(output_label="peaks", **kwargs)


def test_explicit_density_routes_exact_object_and_attaches_immutably(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    result = _detect(distribution, density=density)
    assert distribution.label_groups == {}
    assert result.label_groups["peaks"].density is density
    assert calls[0][2] is density
    assert calls[0][3] == PeakVotingSpec(80, .8, 64)
    assert calls[0][4] == PeakCountRobustnessPolicy(.8)
    assert result.densities == distribution.densities == ()


def test_density_spec_calculates_analysis_local_without_registration(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    distribution = _distribution()
    spec = DensityEstimateSpec(grid_params={"resolution": 8})
    result = _detect(distribution, density_spec=spec)
    generated = calls[0][2]
    assert generated.spec is spec
    assert result.label_groups["peaks"].density is generated
    assert result.densities == distribution.densities == ()
    assert result.shared_density is None


def test_neither_density_input_uses_shared_density_without_registry_change(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    base = _distribution()
    density = base.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    distribution = base.with_density(density, select_as_shared=True)
    result = _detect(distribution)
    assert calls[0][2] is density
    assert result.densities == distribution.densities
    assert result.shared_density_index == distribution.shared_density_index


def test_zero_configuration_arguments_use_smart_defaults_and_default_label(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    base = _distribution()
    density = base.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    distribution = base.with_density(density)
    result = distribution.detect_peaks()
    assert "resolved_peaks" in result.label_groups
    _, output, passed_density, voting, policy = calls[0]
    assert output == "resolved_peaks" and passed_density is density
    assert voting == PeakVotingSpec(n_draws=80, sample_fraction=.80, min_valid_draws=64)
    assert policy == PeakCountRobustnessPolicy(min_mode_frequency=.80)


def test_scalar_overrides_compose_canonical_backend_contracts(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    base = _distribution()
    density = base.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    base.detect_peaks(
        density=density,
        n_draws=11,
        sample_fraction=.6,
        min_valid_draws=7,
        min_mode_frequency=.9,
    )
    assert calls[0][3] == PeakVotingSpec(11, .6, 7)
    assert calls[0][4] == PeakCountRobustnessPolicy(.9)


def test_implicit_min_valid_draws_uses_ceiling_of_eighty_percent(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    distribution.detect_peaks(density=density, n_draws=11)
    assert calls[0][3].min_valid_draws == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_draws": 0},
        {"sample_fraction": 0},
        {"sample_fraction": 1.1},
        {"n_draws": 4, "min_valid_draws": 5},
        {"min_mode_frequency": 1.1},
    ],
)
def test_invalid_scalar_configuration_uses_canonical_validation(monkeypatch, kwargs):
    calls = []
    _stub_labeler(monkeypatch, calls)
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    with pytest.raises(ValueError):
        distribution.detect_peaks(density=density, **kwargs)
    assert calls == []


def test_density_selection_errors_happen_before_labeler(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _detect(distribution, density=density, density_spec=density.spec)
    with pytest.raises(ValueError, match="requires density"):
        _detect(distribution)
    with pytest.raises(ValueError, match="distribution identity"):
        _detect(_distribution("other"), density=density)
    with pytest.raises(ValueError, match="ordered feature"):
        _detect(_distribution(features=("y", "x")), density=density)
    assert calls == []


def test_label_group_collision_raises_before_labeler(monkeypatch):
    calls = []
    _stub_labeler(monkeypatch, calls)
    distribution = _distribution().with_label("peaks", {})
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    with pytest.raises(ValueError, match="already exists"):
        _detect(distribution, density=density)
    assert calls == []


def test_labeler_must_return_group_with_exact_generating_density(monkeypatch):
    distribution = _distribution()
    density = distribution.calc_density(DensityEstimateSpec(grid_params={"resolution": 8}))
    other = distribution.calc_density(
        DensityEstimateSpec(bandwidth_multiplier=1.1, grid_params={"resolution": 9})
    )

    def wrong(*args, **kwargs):
        return LabelGroup(
            name="peaks", distribution_id="d",
            assignments={sid: UNASSIGNED_LABEL for sid in distribution.sample_ids},
            density=other,
        )

    monkeypatch.setattr("morphseq_investigation.engine.labelers.detect_peaks", wrong)
    with pytest.raises(ValueError, match="exact generating density"):
        _detect(distribution, density=density)
