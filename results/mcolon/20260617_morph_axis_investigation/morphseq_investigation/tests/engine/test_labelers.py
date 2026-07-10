"""TASK_B — labeler tests (genotype + peak_finding).

Both labelers run the SAME skeleton and pass the SAME ``validate_label_group``
(peer symmetry). Genotype proves the skeleton; peak_finding folds the live
``compute_resolved_peaks`` machinery into the ontology.
"""

import numpy as np
import pytest

from morphseq_investigation.engine.identifiers import make_distribution_id
from morphseq_investigation.engine.invariants import validate_label_group
from morphseq_investigation.engine.objects import (
    Distribution,
    LabelGroup,
    SampleSet,
    SampleSetGeometry,
)
from morphseq_investigation.engine.labelers import label_genotype


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_distribution(values, sample_ids, *, scope="b9d2", time_bin=30, role="target"):
    dist_id = make_distribution_id(scope, time_bin, role)
    return Distribution(
        distribution_id=dist_id,
        scope_id=scope,
        time_bin=time_bin,
        role=role,
        sample_ids=tuple(sample_ids),
        feature_names=("PC1", "PC2"),
        feature_values=np.asarray(values, dtype=float),
    )


def _genotype_distribution():
    # 6 samples, 2 features. Labels exercise: two real categories, the literal
    # string "unknown", and one NA (None).
    values = np.arange(12, dtype=float).reshape(6, 2)
    sample_ids = [f"e{i}" for i in range(6)]
    return _make_distribution(values, sample_ids)


# --------------------------------------------------------------------------- #
# genotype labeler
# --------------------------------------------------------------------------- #
def test_genotype_category_counts_match():
    dist = _genotype_distribution()
    labels = ["wildtype", "wildtype", "homo", "homo", "homo", "wildtype"]
    lg, sample_sets = label_genotype(dist, "column", {"labels": labels, "column": "genotype"})

    by_name = {s.sample_set_name: s for s in sample_sets}
    assert set(by_name) == {"wildtype", "homo"}
    assert len(by_name["wildtype"].sample_ids) == 3
    assert len(by_name["homo"].sample_ids) == 3
    # peer shape
    assert isinstance(lg, LabelGroup)
    assert all(isinstance(s, SampleSet) for s in sample_sets)
    # provided labels carry no measured geometry
    assert all(s.geometry is None for s in sample_sets)
    # feature_names the labeler USED to form groups is empty for a provided labeler
    assert lg.provenance["labeler"]["feature_names"] == ()
    assert lg.artifacts is None


def test_genotype_unknown_is_a_real_sample_set():
    dist = _genotype_distribution()
    # a LITERAL "unknown" string (not NA) must become a real SampleSet, not abstention
    labels = ["wildtype", "unknown", "unknown", "homo", "homo", "wildtype"]
    lg, sample_sets = label_genotype(dist, "column", {"labels": labels})

    by_name = {s.sample_set_name: s for s in sample_sets}
    assert "unknown" in by_name
    unknown_set = by_name["unknown"]
    assert len(unknown_set.sample_ids) == 2
    # a literal "unknown" is a real category, NOT a missing value
    assert unknown_set.provenance["evidence"]["is_missing_value"] is False
    # and it is NOT in unassigned
    assert unknown_set.sample_ids[0] not in lg.unassigned_sample_ids


def test_genotype_na_source_marks_is_missing_value():
    dist = _genotype_distribution()
    labels = ["wildtype", None, "homo", np.nan, "homo", "wildtype"]
    lg, sample_sets = label_genotype(
        dist, "column", {"labels": labels, "missing_name": "na_bucket"}
    )
    by_name = {s.sample_set_name: s for s in sample_sets}
    assert "na_bucket" in by_name
    na_set = by_name["na_bucket"]
    # both the None and the NaN sample landed in the missing bucket
    assert len(na_set.sample_ids) == 2
    assert na_set.provenance["evidence"]["is_missing_value"] is True
    # the missing bucket is a REAL SampleSet, not abstention
    assert na_set.sample_set_id in lg.sample_set_ids


def test_genotype_coverage_invariant_holds():
    dist = _genotype_distribution()
    labels = ["wildtype", "wildtype", "homo", "homo", None, "unknown"]
    lg, sample_sets = label_genotype(dist, "column", {"labels": labels})
    # validate_label_group already ran inside the labeler; assert coverage directly too
    validate_label_group(dist, lg, sample_sets)
    covered = set(lg.sample_id_to_sample_set_id) | set(lg.unassigned_sample_ids)
    assert covered == set(dist.sample_ids)


def test_genotype_forced_abstention_goes_to_unassigned():
    dist = _genotype_distribution()
    labels = ["wildtype"] * 6
    lg, sample_sets = label_genotype(
        dist, "column", {"labels": labels, "unassigned": {"e0", "e5"}}
    )
    assert set(lg.unassigned_sample_ids) == {"e0", "e5"}
    # the abstained samples are in NO SampleSet
    for s in sample_sets:
        assert "e0" not in s.sample_ids and "e5" not in s.sample_ids
    validate_label_group(dist, lg, sample_sets)

