"""TASK_0 — central invariant guard tests (#3, #6b/#10, #7)."""

import numpy as np
import pytest

from morphseq_investigation.engine.objects import (
    Distribution,
    SampleSet,
    LabelGroup,
)
from morphseq_investigation.engine.invariants import (
    validate_label_group,
    InvariantError,
)


DID = "b9d2_30hpf_reference"


def _distribution(sample_ids=("s0", "s1", "s2", "s3")):
    n = len(sample_ids)
    return Distribution(
        distribution_id=DID,
        scope_id="b9d2",
        time_bin=30,
        role="reference",
        sample_ids=sample_ids,
        feature_names=("PC1",),
        feature_values=np.arange(n, dtype=float).reshape(n, 1),
    )


def _sample_set(name, members):
    return SampleSet(
        sample_set_id=f"{DID}__{name}",
        sample_set_name=name,
        distribution_id=DID,
        sample_ids=tuple(members),
    )


def _valid_group():
    d = _distribution()
    ssA = _sample_set("A", ("s0", "s1"))
    ssB = _sample_set("B", ("s2",))
    lg = LabelGroup(
        label_group_name="genotype",
        distribution_id=DID,
        sample_set_ids=(ssA.sample_set_id, ssB.sample_set_id),
        sample_id_to_sample_set_id={
            "s0": ssA.sample_set_id,
            "s1": ssA.sample_set_id,
            "s2": ssB.sample_set_id,
        },
        unassigned_sample_ids=("s3",),
    )
    return d, lg, [ssA, ssB]


def test_valid_group_passes():
    d, lg, sets = _valid_group()
    validate_label_group(d, lg, sets)  # no raise


def test_assignment_value_not_in_sample_set_ids_raises():
    d, lg, sets = _valid_group()
    bad = LabelGroup(
        label_group_name="genotype",
        distribution_id=DID,
        sample_set_ids=(f"{DID}__A",),
        sample_id_to_sample_set_id={"s0": f"{DID}__GHOST"},
        unassigned_sample_ids=("s1", "s2", "s3"),
    )
    with pytest.raises(InvariantError):
        validate_label_group(d, bad, sets)


def test_assigned_unassigned_overlap_raises():
    d, _, sets = _valid_group()
    bad = LabelGroup(
        label_group_name="genotype",
        distribution_id=DID,
        sample_set_ids=(f"{DID}__A",),
        sample_id_to_sample_set_id={"s0": f"{DID}__A", "s1": f"{DID}__A"},
        unassigned_sample_ids=("s1", "s2", "s3"),  # s1 both assigned AND unassigned
    )
    with pytest.raises(InvariantError):
        validate_label_group(d, bad, [sets[0]])


def test_incomplete_coverage_raises():
    d = _distribution()
    ssA = _sample_set("A", ("s0",))
    bad = LabelGroup(
        label_group_name="genotype",
        distribution_id=DID,
        sample_set_ids=(ssA.sample_set_id,),
        sample_id_to_sample_set_id={"s0": ssA.sample_set_id},
        unassigned_sample_ids=(),  # s1,s2,s3 unaccounted
    )
    with pytest.raises(InvariantError):
        validate_label_group(d, bad, [ssA])


def test_sample_set_members_disagree_with_map_raises():
    d = _distribution()
    ssA = _sample_set("A", ("s0", "s1"))
    bad = LabelGroup(
        label_group_name="genotype",
        distribution_id=DID,
        sample_set_ids=(ssA.sample_set_id,),
        sample_id_to_sample_set_id={"s0": ssA.sample_set_id},  # map says only s0
        unassigned_sample_ids=("s2", "s3"),
    )
    with pytest.raises(InvariantError):
        validate_label_group(d, bad, [ssA])


def test_foreign_distribution_sample_set_raises():
    d = _distribution()
    foreign = SampleSet(
        sample_set_id="other__A",
        sample_set_name="A",
        distribution_id="other_30hpf_target",
        sample_ids=("s0",),
    )
    lg = LabelGroup(
        label_group_name="genotype",
        distribution_id=DID,
        sample_set_ids=("other__A",),
        sample_id_to_sample_set_id={"s0": "other__A"},
        unassigned_sample_ids=("s1", "s2", "s3"),
    )
    with pytest.raises(InvariantError):
        validate_label_group(d, lg, [foreign])
