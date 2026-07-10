"""TASK_0 — central invariant guard tests.

Two guards:
  - ``validate_label_group`` (kept) — coverage / assignment-consistency / FK on a
    labeler RUN-result (LabelGroup + its SampleSets).
  - ``validate_sample_sets`` (new) — derived-view consistency: the sets
    ``Distribution.sample_sets(label)`` returns cover EXACTLY the label column's
    assigned samples.
"""

import numpy as np
import pytest

from morphseq_investigation.engine.objects import (
    Distribution,
    SampleSet,
    LabelGroup,
)
from morphseq_investigation.engine.identifiers import make_distribution_id
from morphseq_investigation.engine.invariants import (
    validate_label_group,
    validate_sample_sets,
    InvariantError,
)


DID = make_distribution_id({"scope_id": "b9d2", "time_bin": 30})


def _distribution(sample_ids=("s0", "s1", "s2", "s3")):
    n = len(sample_ids)
    return Distribution(
        distribution_id=DID,
        sample_ids=sample_ids,
        feature_names=("PC1",),
        feature_values=np.arange(n, dtype=float).reshape(n, 1),
        coordinates={"scope_id": "b9d2", "time_bin": 30},
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


# --------------------------------------------------------------------------- #
# validate_label_group (kept)
# --------------------------------------------------------------------------- #
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
        distribution_id="dist_other",
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


# --------------------------------------------------------------------------- #
# validate_sample_sets (new — derived-view consistency)
# --------------------------------------------------------------------------- #
def test_validate_sample_sets_passes_on_real_derived_view():
    d = _distribution().with_label(
        "genotype", {"s0": "wildtype", "s1": "wildtype", "s2": "b9d2"}
    )  # s3 unassigned
    validate_sample_sets(d, "genotype", d.sample_sets("genotype"))  # no raise


def test_validate_sample_sets_rejects_extra_member():
    d = _distribution().with_label("genotype", {"s0": "wildtype"})
    real = list(d.sample_sets("genotype"))
    # Tamper: add an unassigned sample to the set.
    tampered = SampleSet(
        sample_set_id=real[0].sample_set_id,
        sample_set_name=real[0].sample_set_name,
        distribution_id=DID,
        sample_ids=("s0", "s1"),  # s1 was unassigned
    )
    with pytest.raises(InvariantError):
        validate_sample_sets(d, "genotype", [tampered])


def test_validate_sample_sets_rejects_missing_category():
    d = _distribution().with_label("genotype", {"s0": "wildtype", "s2": "b9d2"})
    real = list(d.sample_sets("genotype"))
    # Drop the b9d2 set -> union no longer covers all assigned samples.
    with pytest.raises(InvariantError):
        validate_sample_sets(d, "genotype", real[:1])


def test_validate_sample_sets_rejects_foreign_distribution_id():
    d = _distribution().with_label("genotype", {"s0": "wildtype"})
    foreign = SampleSet(
        sample_set_id="dist_other__wildtype",
        sample_set_name="wildtype",
        distribution_id="dist_other",
        sample_ids=("s0",),
    )
    with pytest.raises(InvariantError):
        validate_sample_sets(d, "genotype", [foreign])
