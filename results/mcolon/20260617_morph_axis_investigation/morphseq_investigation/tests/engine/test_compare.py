"""TASK_C -- compare_label_groups + correspondence policies + partition agreement.

Fixture-driven (TASK_0 objects only) -- hand-built LabelGroups/SampleSets with
known geometry, per the TASK_C brief. Does not depend on TASK_A/B.
"""

import numpy as np
import pytest

from morphseq_investigation.engine.objects import (
    HDR,
    LabelGroup,
    SampleSet,
    SampleSetGeometry,
)
from morphseq_investigation.engine.compare import (
    ComparisonResult,
    PartitionAgreementResult,
    compare_label_groups,
    label_group_agreement,
)


REF_DID = "b9d2_30hpf_reference"
TGT_DID = "b9d2_30hpf_target"
GRID_A = "grid_aaa"
GRID_B = "grid_bbb"


def _geom(center, grid_id=GRID_A, feature_names=("PC1", "PC2")):
    return SampleSetGeometry(
        grid_id=grid_id,
        feature_names=feature_names,
        center=np.asarray(center, dtype=float),
        radius=1.0,
        r80=0.8,
        cv_radius_from_center=0.1,
    )


def _hdr(mask, grid_id=GRID_A, feature_names=("PC1", "PC2")):
    return HDR(grid_id=grid_id, feature_names=feature_names, level=0.5, mask=np.asarray(mask, dtype=bool))


def _sample_set(
    distribution_id,
    name,
    members,
    *,
    geometry=None,
    hdr=None,
):
    return SampleSet(
        sample_set_id=f"{distribution_id}__{name}",
        sample_set_name=name,
        distribution_id=distribution_id,
        sample_ids=tuple(members),
        geometry=geometry,
        hdr=hdr,
    )


def _label_group(distribution_id, name, sample_sets, unassigned=()):
    assignment = {}
    for s in sample_sets:
        for sid in s.sample_ids:
            assignment[sid] = s.sample_set_id
    return LabelGroup(
        label_group_name=name,
        distribution_id=distribution_id,
        sample_set_ids=tuple(s.sample_set_id for s in sample_sets),
        sample_id_to_sample_set_id=assignment,
        unassigned_sample_ids=tuple(unassigned),
    )


def _sample_set_map(*sample_sets):
    return {s.sample_set_id: s for s in sample_sets}


# --------------------------------------------------------------------------- #
# Fixture: two peak LabelGroups (reference: 1 big + 1 small peak; target: 2 peaks)
# --------------------------------------------------------------------------- #
def _peak_fixture():
    ref_big = _sample_set(
        REF_DID, "peak_0", [f"r{i}" for i in range(8)],
        geometry=_geom([0.0, 0.0]),
        hdr=_hdr(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])),
    )
    ref_small = _sample_set(
        REF_DID, "peak_1", [f"r{i}" for i in range(8, 10)],
        geometry=_geom([10.0, 10.0]),
        hdr=_hdr(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])),
    )
    reference_lg = _label_group(REF_DID, "peak", [ref_big, ref_small], unassigned=("r10",))

    tgt_near_big = _sample_set(
        TGT_DID, "peak_0", [f"t{i}" for i in range(5)],
        geometry=_geom([0.5, 0.5]),
        hdr=_hdr(np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])),
    )
    tgt_near_small = _sample_set(
        TGT_DID, "peak_1", [f"t{i}" for i in range(5, 7)],
        geometry=_geom([9.5, 9.5]),
        hdr=_hdr(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])),
    )
    target_lg = _label_group(TGT_DID, "peak", [tgt_near_big, tgt_near_small], unassigned=("t7",))

    ref_sets = _sample_set_map(ref_big, ref_small)
    tgt_sets = _sample_set_map(tgt_near_big, tgt_near_small)
    return reference_lg, target_lg, ref_sets, tgt_sets, {
        "ref_big": ref_big, "ref_small": ref_small,
        "tgt_near_big": tgt_near_big, "tgt_near_small": tgt_near_small,
    }


# --------------------------------------------------------------------------- #
# largest_reference
# --------------------------------------------------------------------------- #
def test_largest_reference_pairs_every_target_to_biggest_ref():
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    result = compare_label_groups(
        reference_lg, target_lg, "largest_reference",
        reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
    )
    assert isinstance(result, ComparisonResult)
    assert len(result.pairs) == 2
    for pair in result.pairs:
        assert pair.reference_sample_set_id == named["ref_big"].sample_set_id
    tgt_ids = {p.target_sample_set_id for p in result.pairs}
    assert tgt_ids == {named["tgt_near_big"].sample_set_id, named["tgt_near_small"].sample_set_id}


# --------------------------------------------------------------------------- #
# closest_center
# --------------------------------------------------------------------------- #
def test_closest_center_pairs_by_nearest_geometry():
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    result = compare_label_groups(
        reference_lg, target_lg, "closest_center",
        reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
    )
    assert len(result.pairs) == 2
    by_tgt = {p.target_sample_set_id: p.reference_sample_set_id for p in result.pairs}
    assert by_tgt[named["tgt_near_big"].sample_set_id] == named["ref_big"].sample_set_id
    assert by_tgt[named["tgt_near_small"].sample_set_id] == named["ref_small"].sample_set_id
    # center_distance metric present and small for the correctly-matched pair
    for pair in result.pairs:
        assert "center_distance" in pair.metrics
        assert pair.metrics["center_distance"] < 1.0  # 0.5,0.5 offset -> sqrt(0.5)


# --------------------------------------------------------------------------- #
# matched_by_overlap
# --------------------------------------------------------------------------- #
def test_matched_by_overlap_pairs_by_best_hdr_overlap():
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    result = compare_label_groups(
        reference_lg, target_lg, "matched_by_overlap",
        reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
    )
    assert len(result.pairs) == 2
    by_tgt = {p.target_sample_set_id: p.reference_sample_set_id for p in result.pairs}
    assert by_tgt[named["tgt_near_big"].sample_set_id] == named["ref_big"].sample_set_id
    assert by_tgt[named["tgt_near_small"].sample_set_id] == named["ref_small"].sample_set_id
    for pair in result.pairs:
        assert "overlap_fraction" in pair.metrics
        assert pair.metrics["overlap_fraction"] > 0.0


# --------------------------------------------------------------------------- #
# all_pairs
# --------------------------------------------------------------------------- #
def test_all_pairs_is_full_cross_product():
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    result = compare_label_groups(
        reference_lg, target_lg, "all_pairs",
        reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
    )
    assert len(result.pairs) == 4  # 2 ref x 2 tgt, no reduction
    got = {(p.reference_sample_set_id, p.target_sample_set_id) for p in result.pairs}
    expected = {
        (r, t)
        for r in (named["ref_big"].sample_set_id, named["ref_small"].sample_set_id)
        for t in (named["tgt_near_big"].sample_set_id, named["tgt_near_small"].sample_set_id)
    }
    assert got == expected
    # all_pairs leaves nothing unmatched
    assert result.unmatched_reference_sample_set_ids == ()
    assert result.unmatched_target_sample_set_ids == ()


def test_unknown_correspondence_spec_raises():
    reference_lg, target_lg, ref_sets, tgt_sets, _ = _peak_fixture()
    with pytest.raises(ValueError):
        compare_label_groups(
            reference_lg, target_lg, "not_a_real_spec",
            reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
        )


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def test_mismatched_feature_names_raises_on_closest_center():
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    # swap in a target set whose geometry claims different feature_names
    bad_tgt = _sample_set(
        TGT_DID, "peak_0", named["tgt_near_big"].sample_ids,
        geometry=_geom([0.5, 0.5], feature_names=("umap_1", "umap_2")),
        hdr=named["tgt_near_big"].hdr,
    )
    tgt_sets = dict(tgt_sets)
    tgt_sets[bad_tgt.sample_set_id] = bad_tgt
    with pytest.raises(ValueError):
        compare_label_groups(
            reference_lg, target_lg, "closest_center",
            reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
        )


def test_mismatched_grid_id_raises_on_matched_by_overlap():
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    bad_tgt = _sample_set(
        TGT_DID, "peak_0", named["tgt_near_big"].sample_ids,
        geometry=named["tgt_near_big"].geometry,
        hdr=_hdr(named["tgt_near_big"].hdr.mask, grid_id=GRID_B),
    )
    tgt_sets = dict(tgt_sets)
    tgt_sets[bad_tgt.sample_set_id] = bad_tgt
    with pytest.raises(ValueError):
        compare_label_groups(
            reference_lg, target_lg, "matched_by_overlap",
            reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
        )


def test_mismatched_feature_names_scalar_pair_metric_raises_even_in_all_pairs():
    # all_pairs still computes per-pair metrics -> guardrail fires there too.
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    bad_tgt = _sample_set(
        TGT_DID, "peak_0", named["tgt_near_big"].sample_ids,
        geometry=_geom([0.5, 0.5], feature_names=("umap_1", "umap_2")),
        hdr=None,
    )
    tgt_sets = dict(tgt_sets)
    tgt_sets[bad_tgt.sample_set_id] = bad_tgt
    with pytest.raises(ValueError):
        compare_label_groups(
            reference_lg, target_lg, "all_pairs",
            reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
        )


# --------------------------------------------------------------------------- #
# unassigned + unmatched survive
# --------------------------------------------------------------------------- #
def test_unassigned_and_unmatched_survive_into_result():
    reference_lg, target_lg, ref_sets, tgt_sets, named = _peak_fixture()
    result = compare_label_groups(
        reference_lg, target_lg, "largest_reference",
        reference_sample_sets=ref_sets, target_sample_sets=tgt_sets,
    )
    # largest_reference pairs both targets to ref_big -> ref_small unmatched
    assert result.unmatched_reference_sample_set_ids == (named["ref_small"].sample_set_id,)
    assert result.unmatched_target_sample_set_ids == ()
    assert result.reference_unassigned_sample_ids == ("r10",)
    assert result.target_unassigned_sample_ids == ("t7",)


# --------------------------------------------------------------------------- #
# Genotype-vs-peak agreement
# --------------------------------------------------------------------------- #
def _partition_label_group(distribution_id, name, id_to_group, group_names, unassigned=()):
    sets = []
    for g in group_names:
        members = tuple(sid for sid, grp in id_to_group.items() if grp == g)
        sets.append(_sample_set(distribution_id, g, members))
    return _label_group(distribution_id, name, sets, unassigned=unassigned), _sample_set_map(*sets)


def test_agreement_identical_partitions_is_max():
    sample_ids = [f"s{i}" for i in range(20)]
    id_to_group = {sid: ("A" if i % 2 == 0 else "B") for i, sid in enumerate(sample_ids)}
    left_lg, _ = _partition_label_group(REF_DID, "genotype", id_to_group, ["A", "B"])
    right_lg, _ = _partition_label_group(REF_DID, "peak", id_to_group, ["A", "B"])

    result = label_group_agreement(left_lg, right_lg)
    assert isinstance(result, PartitionAgreementResult)
    assert result.agreement_metric == "adjusted_rand_score"
    assert result.agreement_score == pytest.approx(1.0)
    assert result.n_samples_compared == 20


def test_agreement_orthogonal_partitions_near_chance():
    # 4x4 grid design: left splits into quadrant-rows, right into a scrambled
    # partition uncorrelated with left -> ARI near 0.
    rng = np.random.default_rng(0)
    n = 200
    sample_ids = [f"s{i}" for i in range(n)]
    left_labels = rng.integers(0, 3, size=n)
    right_labels = rng.permutation(left_labels)  # same multiset, shuffled -> independent
    id_to_left = {sid: f"g{left_labels[i]}" for i, sid in enumerate(sample_ids)}
    id_to_right = {sid: f"p{right_labels[i]}" for i, sid in enumerate(sample_ids)}

    left_lg, _ = _partition_label_group(REF_DID, "genotype", id_to_left, ["g0", "g1", "g2"])
    right_lg, _ = _partition_label_group(REF_DID, "peak", id_to_right, ["p0", "p1", "p2"])

    result = label_group_agreement(left_lg, right_lg)
    assert abs(result.agreement_score) < 0.2  # near chance


def test_agreement_cross_tab_shape_and_counts():
    id_to_group = {
        "s0": "A", "s1": "A", "s2": "B", "s3": "B",
    }
    left_lg, _ = _partition_label_group(REF_DID, "genotype", id_to_group, ["A", "B"])
    # right partition: s0,s1,s2 -> P; s3 -> Q
    id_to_right = {"s0": "P", "s1": "P", "s2": "P", "s3": "Q"}
    right_lg, _ = _partition_label_group(REF_DID, "peak", id_to_right, ["P", "Q"])

    result = label_group_agreement(left_lg, right_lg)
    left_a = f"{REF_DID}__A"
    left_b = f"{REF_DID}__B"
    right_p = f"{REF_DID}__P"
    right_q = f"{REF_DID}__Q"
    assert result.cross_tab[left_a][right_p] == 2
    assert result.cross_tab[left_a][right_q] == 0
    assert result.cross_tab[left_b][right_p] == 1
    assert result.cross_tab[left_b][right_q] == 1


def test_agreement_unassigned_on_one_side_excluded_but_reported():
    id_to_group = {"s0": "A", "s1": "A", "s2": "B"}
    left_lg, _ = _partition_label_group(REF_DID, "genotype", id_to_group, ["A", "B"])
    # right: s0,s1 assigned to P; s2 unassigned (labeler declined to place it)
    id_to_right = {"s0": "P", "s1": "P"}
    right_lg, _ = _partition_label_group(
        REF_DID, "peak", id_to_right, ["P"], unassigned=("s2",)
    )

    result = label_group_agreement(left_lg, right_lg)
    assert result.n_samples_compared == 2
    # s2 is assigned on the LEFT (genotype) but unassigned on the RIGHT (peak)
    assert result.left_only_sample_ids == ("s2",)
    assert result.right_only_sample_ids == ()
    assert result.unassigned_both_sample_ids == ()


def test_agreement_requires_same_distribution_id():
    id_to_group = {"s0": "A", "s1": "B"}
    left_lg, _ = _partition_label_group(REF_DID, "genotype", id_to_group, ["A", "B"])
    right_lg, _ = _partition_label_group(TGT_DID, "peak", id_to_group, ["A", "B"])
    with pytest.raises(ValueError):
        label_group_agreement(left_lg, right_lg)
