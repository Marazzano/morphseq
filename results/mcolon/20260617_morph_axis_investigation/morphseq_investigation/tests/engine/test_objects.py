import dataclasses

import numpy as np
import pytest

from morphseq_investigation.engine.objects import (
    Distribution,
    LabelGroup,
    LabelingProvenance,
    SampleSetGeometry,
    UNASSIGNED_LABEL,
)
from morphseq_investigation.core.peak_stability import (
    PeakCountRobustnessPolicy,
    PeakCountVote,
    PeakResolutionSummary,
    PeakVotingSpec,
)


def _distribution(**changes):
    args = dict(
        distribution_id="d",
        sample_ids=("s0", "s1", "s2"),
        feature_names=("x", "y"),
        feature_values=np.arange(6.0).reshape(3, 2),
    )
    args.update(changes)
    return Distribution(**args)


def test_distribution_shape_identity_and_immutability():
    distribution = _distribution()
    assert distribution.feature_column("y").tolist() == [1, 3, 5]
    with pytest.raises(dataclasses.FrozenInstanceError):
        distribution.distribution_id = "x"
    with pytest.raises(ValueError):
        distribution.feature_values[0, 0] = 2
    with pytest.raises(ValueError, match="second axis"):
        _distribution(feature_names=("x",))
    with pytest.raises(ValueError, match="unique"):
        _distribution(sample_ids=("s0", "s0", "s2"))


def test_with_label_creates_only_unified_label_group_and_derives_sets():
    distribution = _distribution().with_label("provided", {"s0": "a", "s1": "a"})
    assert "labels" not in {field.name for field in dataclasses.fields(Distribution)}
    group = distribution.label_groups["provided"]
    assert isinstance(group, LabelGroup)
    assert group.assignments == {"s0": "a", "s1": "a", "s2": UNASSIGNED_LABEL}
    assert group.labeling_provenance == LabelingProvenance(method="provided")
    sets = distribution.sample_sets("provided")
    assert len(sets) == 1 and sets[0].sample_ids == ("s0", "s1")
    with pytest.raises(ValueError, match="already exists"):
        distribution.with_label("provided", {})


def test_label_group_constructor_validates_geometry_categories():
    geometry = SampleSetGeometry(
        feature_names=("x", "y"), center=np.zeros(2), radius=1,
        support_fraction=.5, r80_radial_concentration=.2,
        cv_radius_from_center=.1,
    )
    with pytest.raises(ValueError, match="assigned"):
        LabelGroup(
            name="g", distribution_id="d", assignments={"s0": "a"},
            sample_set_geometries={"b": geometry},
        )


def test_distribution_requires_total_label_assignment_coverage():
    group = LabelGroup(name="g", distribution_id="d", assignments={"s0": "a"})
    with pytest.raises(ValueError, match="cover every sample"):
        _distribution(label_groups={"g": group})


def _summary(count=2):
    voting = PeakVotingSpec(n_draws=5, sample_fraction=.8, min_valid_draws=4)
    return PeakResolutionSummary(
        peak_count_vote=PeakCountVote({count: 5}, 5, 5, .8),
        voting_spec=voting,
        robustness_policy=PeakCountRobustnessPolicy(.8),
        resolved_peak_count=count,
        is_robust=True,
    )


def _geometry(center):
    return SampleSetGeometry(
        feature_names=("x", "y"), center=np.asarray(center), radius=1,
        support_fraction=.5, r80_radial_concentration=.2,
        cv_radius_from_center=.1,
    )


def test_throwaway_is_explicit_unassigned_and_never_materializes():
    distribution = _distribution().with_label(
        "provided", {"s0": "kept", "s1": UNASSIGNED_LABEL}
    )
    sets = distribution.sample_sets("provided")
    assert [sample_set.sample_set_name for sample_set in sets] == ["kept"]
    assert {sid for sample_set in sets for sid in sample_set.sample_ids} == {"s0"}
    assert distribution.label_groups["provided"].assignments["s2"] == UNASSIGNED_LABEL


def test_resolved_peak_requires_density_count_geometry_and_local_ids():
    density = _distribution().calc_density()
    group = LabelGroup(
        name="run_a", distribution_id="d",
        assignments={"s0": "peak_0", "s1": "peak_1", "s2": UNASSIGNED_LABEL},
        density=density,
        sample_set_geometries={"peak_0": _geometry((0, 0)), "peak_1": _geometry((1, 1))},
        peak_resolution_summary=_summary(2),
        labeling_provenance=LabelingProvenance(method="resolved_peaks"),
    )
    distribution = _distribution(label_groups={"run_a": group})
    sets = distribution.sample_sets("run_a")
    assert group.peak_count == group.sample_set_count == len(sets) == len(group.sample_set_geometries)
    assert [item.sample_set_id for item in sets] == [
        "d__run_a__peak_0", "d__run_a__peak_1"
    ]
    assert distribution.sample_sets("run_a")[0].sample_set_id == sets[0].sample_set_id


def test_resolved_peak_constructor_rejects_missing_density_count_and_nondeterministic_ids():
    assignments = {"s0": "peak_0", "s1": "peak_1", "s2": UNASSIGNED_LABEL}
    geometries = {"peak_0": _geometry((0, 0)), "peak_1": _geometry((1, 1))}
    with pytest.raises(ValueError, match="generating density"):
        LabelGroup("r", "d", assignments, sample_set_geometries=geometries,
                   peak_resolution_summary=_summary(2))
    density = _distribution().calc_density()
    with pytest.raises(ValueError, match="count"):
        LabelGroup("r", "d", assignments, density=density,
                   sample_set_geometries=geometries, peak_resolution_summary=_summary(1))
    with pytest.raises(ValueError, match="deterministic"):
        LabelGroup(
            "r", "d", {"s0": "peak_1", "s1": "peak_0", "s2": UNASSIGNED_LABEL},
            density=density, sample_set_geometries=geometries,
            peak_resolution_summary=_summary(2),
            labeling_provenance=LabelingProvenance(method="resolved_peaks"),
        )
