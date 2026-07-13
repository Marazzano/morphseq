import numpy as np
import pytest

from morphseq_investigation.engine.catalog import DistributionComparison
from morphseq_investigation.engine.compare import label_group_agreement, summarize_label_groups
from morphseq_investigation.engine.objects import Distribution, UNASSIGNED_LABEL


def _distribution(distribution_id, condition):
    return Distribution(
        distribution_id=distribution_id,
        sample_ids=("s0", "s1", "s2", "s3"),
        feature_names=("x", "y"),
        feature_values=np.zeros((4, 2)),
        coordinates={"batch": "b1", "condition": condition},
    )


def test_agreement_reads_unified_assignments_and_explicit_unassigned():
    distribution = _distribution("d", "reference")
    distribution = distribution.with_label(
        "left", {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
    )
    distribution = distribution.with_label(
        "right", {"s0": "P", "s1": "P", "s2": UNASSIGNED_LABEL, "s3": "Q"}
    )
    result = label_group_agreement(
        distribution.get_label_group("left"), distribution.get_label_group("right")
    )
    assert result.n_samples_compared == 3
    assert result.left_only_sample_ids == ("s2",)
    assert result.cross_tab["A"]["P"] == 2
    assert result.cross_tab["B"]["Q"] == 1


def test_agreement_requires_same_complete_sample_membership():
    left = _distribution("d", "reference").with_label("x", {"s0": "A"}).get_label_group("x")
    right = _distribution("other", "target").with_label("x", {"s0": "A"}).get_label_group("x")
    with pytest.raises(ValueError, match="same distribution_id"):
        label_group_agreement(left, right)


def test_summary_preserves_structured_coordinates_and_derives_sample_sets():
    reference = _distribution("ref", "reference").with_label(
        "genotype", {"s0": "wt", "s1": "wt", "s2": "mut"}
    )
    target = _distribution("tgt", "target").with_label(
        "genotype", {"s0": "wt", "s1": "mut", "s2": "mut", "s3": "mut"}
    )
    comparison = DistributionComparison(
        coordinates={"batch": "b1"}, members={"reference": reference, "target": target}
    )
    rows = summarize_label_groups(comparison, "genotype")
    assert tuple(row.member_value for row in rows) == ("reference", "target")
    assert all(dict(row.comparison_coordinates) == {"batch": "b1"} for row in rows)
    assert rows[0].distribution_coordinates["condition"] == "reference"
    assert rows[0].sample_set_count == len(reference.sample_sets("genotype")) == 2
    assert rows[0].unassigned_sample_count == 1
    assert rows[0].peak_count is None
    assert rows[0].is_robust is None


def test_summary_missing_label_raises_instead_of_dropping_member():
    reference = _distribution("ref", "reference").with_label("group", {"s0": "A"})
    target = _distribution("tgt", "target")
    comparison = DistributionComparison(coordinates={"batch": "b1"}, members={"r": reference, "t": target})
    with pytest.raises(KeyError):
        summarize_label_groups(comparison, "group")
