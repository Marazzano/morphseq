import numpy as np
import pytest

from morphseq_investigation.engine.invariants import InvariantError, validate_label_group, validate_sample_sets
from morphseq_investigation.engine.objects import Distribution, LabelGroup, SampleSet


def _distribution():
    return Distribution(
        distribution_id="d", sample_ids=("s0", "s1", "s2"),
        feature_names=("x",), feature_values=np.arange(3.0).reshape(-1, 1),
    ).with_label("g", {"s0": "a", "s1": "b"})


def test_derived_sample_sets_validate_against_authoritative_group():
    distribution = _distribution()
    sets = distribution.sample_sets("g")
    validate_sample_sets(distribution, "g", sets)
    validate_label_group(distribution, distribution.label_groups["g"], sets)


def test_validate_label_group_rejects_wrong_membership():
    distribution = _distribution()
    sets = list(distribution.sample_sets("g"))
    sets[0] = SampleSet(
        sample_set_id=sets[0].sample_set_id, sample_set_name="a",
        distribution_id="d", sample_ids=("s1",),
    )
    with pytest.raises(InvariantError, match="disagrees"):
        validate_label_group(distribution, distribution.label_groups["g"], sets)


def test_validate_label_group_rejects_missing_materialized_category():
    distribution = _distribution()
    with pytest.raises(InvariantError, match="materialized"):
        validate_label_group(distribution, distribution.label_groups["g"], ())
