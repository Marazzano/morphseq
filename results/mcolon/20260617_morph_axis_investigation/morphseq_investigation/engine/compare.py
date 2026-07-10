"""Cross-run / cross-distribution comparison layer (ontology Invariant #10,
"TBD -- Comparison").

Compares :class:`~engine.objects.LabelGroup` s, NOT naked ``SampleSet`` tuples --
this keeps ``artifacts``/``provenance``/``across_sample_set_metrics``/
``unassigned_sample_ids`` alive through the comparison instead of collapsing to
bare geometry. Roles (``reference``/``target``) are assigned AT CALL TIME -- a
``LabelGroup`` is not intrinsically one or the other.

Two distinct comparison shapes live in this module:

1. ``compare_label_groups`` -- REF-vs-TARGET correspondence between two
   LabelGroups (usually from two different Distributions/roles). A
   ``correspondence_spec`` policy decides which SampleSets pair up.
2. ``label_group_agreement`` -- partition-vs-partition agreement between two
   TOTAL partitions of the SAME Distribution's samples (e.g. genotype vs peak)
   -- "do they carve alike?"

Both are cross-run relations and therefore live HERE, not on
``LabelGroup.across_sample_set_metrics`` (ontology #8: that slot is WITHIN one
run only).

Because a labeler's SampleSets are not stored on the LabelGroup itself (the
LabelGroup only holds ``sample_set_ids`` -- foreign keys), every function here
additionally takes ``{sample_set_id: SampleSet}`` maps so it can look up the
actual geometry/hdr/members. This mirrors how ``invariants.validate_label_group``
already takes a separate ``sample_sets`` iterable alongside the LabelGroup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .objects import HDR, LabelGroup, SampleSet, SampleSetGeometry

CorrespondenceSpec = str  # one of _VALID_SPECS, kept as a plain str (see module doc)

_VALID_SPECS = ("largest_reference", "closest_center", "matched_by_overlap", "all_pairs")


def _readonly_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Same idiom as ``engine.objects`` -- immutable view, not a naked dict."""
    return MappingProxyType(dict(values))


# --------------------------------------------------------------------------- #
# ComparisonResult -- frozen, typed, same discipline as the ontology objects.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CorrespondencePair:
    """One matched (reference_set, target_set) pair + its per-pair metrics.

    ``metrics`` holds whichever of {center_distance, overlap_fraction,
    valley_depth} are computable for this pair given what's available on the
    two SampleSets (geometry / hdr). Never all three are guaranteed.
    """

    reference_sample_set_id: str
    target_sample_set_id: str
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _readonly_mapping(self.metrics))


@dataclass(frozen=True)
class ComparisonResult:
    """Result of :func:`compare_label_groups`.

    Keeps unmatched sets and unassigned samples VISIBLE (never silently
    dropped) alongside the correspondence pairs themselves.
    """

    correspondence_spec: str
    reference_distribution_id: str
    target_distribution_id: str
    pairs: tuple[CorrespondencePair, ...] = ()
    unmatched_reference_sample_set_ids: tuple[str, ...] = ()
    unmatched_target_sample_set_ids: tuple[str, ...] = ()
    reference_unassigned_sample_ids: tuple[str, ...] = ()
    target_unassigned_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(self.pairs))
        object.__setattr__(
            self,
            "unmatched_reference_sample_set_ids",
            tuple(self.unmatched_reference_sample_set_ids),
        )
        object.__setattr__(
            self,
            "unmatched_target_sample_set_ids",
            tuple(self.unmatched_target_sample_set_ids),
        )
        object.__setattr__(
            self,
            "reference_unassigned_sample_ids",
            tuple(self.reference_unassigned_sample_ids),
        )
        object.__setattr__(
            self,
            "target_unassigned_sample_ids",
            tuple(self.target_unassigned_sample_ids),
        )


@dataclass(frozen=True)
class PartitionAgreementResult:
    """Result of :func:`label_group_agreement` -- partition-vs-partition on the
    SAME Distribution's samples (e.g. genotype vs peak): do they carve alike?

    Distinct from :class:`ComparisonResult` (ref-vs-target correspondence).
    """

    distribution_id: str
    left_label_group_name: str
    right_label_group_name: str
    cross_tab: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    agreement_metric: str = "adjusted_rand_score"
    agreement_score: float = 0.0
    n_samples_compared: int = 0
    left_only_sample_ids: tuple[str, ...] = ()   # unassigned on left, assigned on right
    right_only_sample_ids: tuple[str, ...] = ()   # unassigned on right, assigned on left
    unassigned_both_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cross_tab",
            _readonly_mapping({k: _readonly_mapping(v) for k, v in dict(self.cross_tab).items()}),
        )
        object.__setattr__(self, "left_only_sample_ids", tuple(self.left_only_sample_ids))
        object.__setattr__(self, "right_only_sample_ids", tuple(self.right_only_sample_ids))
        object.__setattr__(
            self, "unassigned_both_sample_ids", tuple(self.unassigned_both_sample_ids)
        )


# --------------------------------------------------------------------------- #
# Guardrails -- assert, fail loudly (#10 / TASK_C brief).
# --------------------------------------------------------------------------- #
def _require_same_feature_names(a: SampleSetGeometry, b: SampleSetGeometry) -> None:
    if tuple(a.feature_names) != tuple(b.feature_names):
        raise ValueError(
            "scalar comparison (center distance) requires identical feature_names: "
            f"{a.feature_names!r} != {b.feature_names!r}"
        )


def _require_same_grid_id(a: HDR, b: HDR) -> None:
    if a.grid_id != b.grid_id:
        raise ValueError(
            "raster comparison (matched_by_overlap) requires identical grid_id: "
            f"{a.grid_id!r} != {b.grid_id!r}"
        )


# --------------------------------------------------------------------------- #
# Per-pair metrics
# --------------------------------------------------------------------------- #
def _center_distance(ref_set: SampleSet, tgt_set: SampleSet) -> float | None:
    """Euclidean distance between geometry centers, feature units, no frame.

    Requires both sets to HAVE geometry and share ``feature_names`` (#4/#10).
    Returns ``None`` if either set lacks geometry (e.g. a genotype set).
    """
    if ref_set.geometry is None or tgt_set.geometry is None:
        return None
    _require_same_feature_names(ref_set.geometry, tgt_set.geometry)
    return float(np.linalg.norm(ref_set.geometry.center - tgt_set.geometry.center))


def _hdr_overlap_fraction(ref_set: SampleSet, tgt_set: SampleSet) -> float | None:
    """Jaccard overlap of two boolean HDR masks on the SAME grid (raster-comparable).

    Requires both sets to have an ``hdr`` and share ``grid_id`` (#4/#10). Returns
    ``None`` if either set lacks an HDR.
    """
    if ref_set.hdr is None or tgt_set.hdr is None:
        return None
    _require_same_grid_id(ref_set.hdr, tgt_set.hdr)
    ref_mask = np.asarray(ref_set.hdr.mask, dtype=bool)
    tgt_mask = np.asarray(tgt_set.hdr.mask, dtype=bool)
    if ref_mask.shape != tgt_mask.shape:
        raise ValueError(
            f"HDR mask shapes differ despite same grid_id: {ref_mask.shape} vs {tgt_mask.shape}"
        )
    union = np.logical_or(ref_mask, tgt_mask).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(ref_mask, tgt_mask).sum()
    return float(intersection) / float(union)


def _pair_metrics(ref_set: SampleSet, tgt_set: SampleSet) -> dict[str, float]:
    """Compute whichever per-pair metrics are available for this pair.

    ``valley_depth`` is a within-run (across_sample_set_metrics) concept in the
    ontology (#8) and is not computable purely from two SampleSets here; if a
    caller wants it surfaced per-pair it must already live in one side's
    provenance/across metrics -- out of scope for this generic per-pair metric
    computation (documented, not silently invented).
    """
    metrics: dict[str, float] = {}
    center_distance = _center_distance(ref_set, tgt_set)
    if center_distance is not None:
        metrics["center_distance"] = center_distance
    overlap = _hdr_overlap_fraction(ref_set, tgt_set)
    if overlap is not None:
        metrics["overlap_fraction"] = overlap
    return metrics


# --------------------------------------------------------------------------- #
# Correspondence policies -- each returns [(ref_set_id, tgt_set_id), ...]
# --------------------------------------------------------------------------- #
def _policy_largest_reference(
    reference_lg: LabelGroup,
    target_lg: LabelGroup,
    reference_sample_sets: Mapping[str, SampleSet],
) -> list[tuple[str, str]]:
    """Pair each target set to the biggest reference set (by member count)."""
    if not reference_lg.sample_set_ids:
        return []
    biggest_ref_id = max(
        reference_lg.sample_set_ids,
        key=lambda sid: len(reference_sample_sets[sid].sample_ids),
    )
    return [(biggest_ref_id, tgt_id) for tgt_id in target_lg.sample_set_ids]


def _policy_closest_center(
    reference_lg: LabelGroup,
    target_lg: LabelGroup,
    reference_sample_sets: Mapping[str, SampleSet],
    target_sample_sets: Mapping[str, SampleSet],
) -> list[tuple[str, str]]:
    """Pair by nearest geometry.center, feature units -- no frame (#4).

    Every reference set involved in a comparison must share ``feature_names``
    with the target set it's compared to (guardrail raised inside
    ``_center_distance`` via ``_require_same_feature_names``). Sets without
    geometry (e.g. a genotype SampleSet) cannot participate and are skipped
    (they surface as unmatched via the caller's bookkeeping).
    """
    pairs: list[tuple[str, str]] = []
    for tgt_id in target_lg.sample_set_ids:
        tgt_set = target_sample_sets[tgt_id]
        if tgt_set.geometry is None:
            continue
        best_ref_id = None
        best_dist = None
        for ref_id in reference_lg.sample_set_ids:
            ref_set = reference_sample_sets[ref_id]
            if ref_set.geometry is None:
                continue
            _require_same_feature_names(ref_set.geometry, tgt_set.geometry)
            dist = float(np.linalg.norm(ref_set.geometry.center - tgt_set.geometry.center))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_ref_id = ref_id
        if best_ref_id is not None:
            pairs.append((best_ref_id, tgt_id))
    return pairs


def _policy_matched_by_overlap(
    reference_lg: LabelGroup,
    target_lg: LabelGroup,
    reference_sample_sets: Mapping[str, SampleSet],
    target_sample_sets: Mapping[str, SampleSet],
) -> list[tuple[str, str]]:
    """Pair by best HDR/basin raster overlap -- requires identical grid_id (#4)."""
    pairs: list[tuple[str, str]] = []
    for tgt_id in target_lg.sample_set_ids:
        tgt_set = target_sample_sets[tgt_id]
        if tgt_set.hdr is None:
            continue
        best_ref_id = None
        best_overlap = None
        for ref_id in reference_lg.sample_set_ids:
            ref_set = reference_sample_sets[ref_id]
            if ref_set.hdr is None:
                continue
            _require_same_grid_id(ref_set.hdr, tgt_set.hdr)
            overlap = _hdr_overlap_fraction(ref_set, tgt_set)
            if overlap is not None and (best_overlap is None or overlap > best_overlap):
                best_overlap = overlap
                best_ref_id = ref_id
        if best_ref_id is not None:
            pairs.append((best_ref_id, tgt_id))
    return pairs


def _policy_all_pairs(
    reference_lg: LabelGroup,
    target_lg: LabelGroup,
) -> list[tuple[str, str]]:
    """Full cross product -- no reduction."""
    return [
        (ref_id, tgt_id)
        for ref_id in reference_lg.sample_set_ids
        for tgt_id in target_lg.sample_set_ids
    ]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compare_label_groups(
    reference_lg: LabelGroup,
    target_lg: LabelGroup,
    correspondence_spec: CorrespondenceSpec,
    *,
    reference_sample_sets: Mapping[str, SampleSet],
    target_sample_sets: Mapping[str, SampleSet],
) -> ComparisonResult:
    """Compare two LabelGroups (NOT naked SampleSet tuples, ontology #10).

    Roles (``reference``/``target``) are assigned AT CALL TIME -- a LabelGroup
    is not intrinsically one or the other.

    Because a labeler's SampleSets are not stored on the LabelGroup (it only
    holds ``sample_set_ids`` foreign keys), the caller must additionally pass
    ``reference_sample_sets`` / ``target_sample_sets`` as
    ``{sample_set_id: SampleSet}`` maps -- the same shape
    ``invariants.validate_label_group`` already expects for its ``sample_sets``
    argument.

    ``unmatched`` sets and ``unassigned`` samples from BOTH sides survive into
    the result untouched -- they are never silently dropped.
    """
    if correspondence_spec not in _VALID_SPECS:
        raise ValueError(
            f"unknown correspondence_spec {correspondence_spec!r}; must be one of {_VALID_SPECS}"
        )

    if correspondence_spec == "largest_reference":
        raw_pairs = _policy_largest_reference(reference_lg, target_lg, reference_sample_sets)
    elif correspondence_spec == "closest_center":
        raw_pairs = _policy_closest_center(
            reference_lg, target_lg, reference_sample_sets, target_sample_sets
        )
    elif correspondence_spec == "matched_by_overlap":
        raw_pairs = _policy_matched_by_overlap(
            reference_lg, target_lg, reference_sample_sets, target_sample_sets
        )
    else:  # all_pairs
        raw_pairs = _policy_all_pairs(reference_lg, target_lg)

    pairs = []
    for ref_id, tgt_id in raw_pairs:
        ref_set = reference_sample_sets[ref_id]
        tgt_set = target_sample_sets[tgt_id]
        metrics = _pair_metrics(ref_set, tgt_set)
        pairs.append(
            CorrespondencePair(
                reference_sample_set_id=ref_id,
                target_sample_set_id=tgt_id,
                metrics=metrics,
            )
        )

    matched_ref_ids = {p.reference_sample_set_id for p in pairs}
    matched_tgt_ids = {p.target_sample_set_id for p in pairs}
    unmatched_ref = tuple(
        sid for sid in reference_lg.sample_set_ids if sid not in matched_ref_ids
    )
    unmatched_tgt = tuple(
        sid for sid in target_lg.sample_set_ids if sid not in matched_tgt_ids
    )

    return ComparisonResult(
        correspondence_spec=correspondence_spec,
        reference_distribution_id=reference_lg.distribution_id,
        target_distribution_id=target_lg.distribution_id,
        pairs=tuple(pairs),
        unmatched_reference_sample_set_ids=unmatched_ref,
        unmatched_target_sample_set_ids=unmatched_tgt,
        reference_unassigned_sample_ids=reference_lg.unassigned_sample_ids,
        target_unassigned_sample_ids=target_lg.unassigned_sample_ids,
    )


# --------------------------------------------------------------------------- #
# Genotype-vs-peak agreement -- partition-vs-partition on the SAME Distribution.
# --------------------------------------------------------------------------- #
def label_group_agreement(
    left_lg: LabelGroup,
    right_lg: LabelGroup,
) -> PartitionAgreementResult:
    """Agreement between two TOTAL partitions of the SAME Distribution's samples
    (e.g. genotype vs peak) -- do they carve alike?

    Distinct from :func:`compare_label_groups` (that's ref-vs-target
    correspondence between two possibly-different Distributions; this is
    partition-vs-partition on ONE Distribution, per the TASK_C brief).

    Agreement metric: **adjusted Rand index** (``sklearn.metrics.
    adjusted_rand_score``) -- chosen over normalized MI because ARI is
    chance-corrected AND bounded at exactly 1.0 for identical partitions
    (NMI is also chance-agnostic in its raw form and less interpretable at the
    "near chance" end); ARI ~ 0 for independent random partitions, which is
    exactly the "near chance" contract the TASK_C tests ask for.

    Only samples assigned in BOTH LabelGroups are used for the cross-tab and
    the agreement score (an ARI over an artificially-added "unassigned"
    category would conflate labeler abstention with disagreement). Samples
    unassigned on one or both sides are reported separately, never silently
    dropped.
    """
    if left_lg.distribution_id != right_lg.distribution_id:
        raise ValueError(
            "label_group_agreement compares two partitions of the SAME Distribution: "
            f"{left_lg.distribution_id!r} != {right_lg.distribution_id!r}"
        )

    left_assignment = dict(left_lg.sample_id_to_sample_set_id)
    right_assignment = dict(right_lg.sample_id_to_sample_set_id)
    left_unassigned = set(left_lg.unassigned_sample_ids)
    right_unassigned = set(right_lg.unassigned_sample_ids)

    common_ids = sorted(set(left_assignment) & set(right_assignment))

    left_only = sorted(set(left_assignment) & right_unassigned)
    right_only = sorted(left_unassigned & set(right_assignment))
    unassigned_both = sorted(left_unassigned & right_unassigned)

    cross_tab: dict[str, dict[str, int]] = {
        left_id: {right_id: 0 for right_id in right_lg.sample_set_ids}
        for left_id in left_lg.sample_set_ids
    }
    left_labels = []
    right_labels = []
    for sample_id in common_ids:
        left_set_id = left_assignment[sample_id]
        right_set_id = right_assignment[sample_id]
        cross_tab[left_set_id][right_set_id] += 1
        left_labels.append(left_set_id)
        right_labels.append(right_set_id)

    if common_ids:
        agreement_score = float(adjusted_rand_score(left_labels, right_labels))
    else:
        agreement_score = float("nan")

    return PartitionAgreementResult(
        distribution_id=left_lg.distribution_id,
        left_label_group_name=left_lg.label_group_name,
        right_label_group_name=right_lg.label_group_name,
        cross_tab=cross_tab,
        agreement_metric="adjusted_rand_score",
        agreement_score=agreement_score,
        n_samples_compared=len(common_ids),
        left_only_sample_ids=tuple(left_only),
        right_only_sample_ids=tuple(right_only),
        unassigned_both_sample_ids=tuple(unassigned_both),
    )
