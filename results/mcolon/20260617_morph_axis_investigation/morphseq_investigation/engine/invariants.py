"""Cross-object invariant guards for derived SampleSets and density grids."""

from __future__ import annotations

from typing import Iterable

from .objects import (
    UNASSIGNED_LABEL,
    Distribution,
    DensityGrid,
    Grid,
    LabelGroup,
    SampleSet,
)


class InvariantError(ValueError):
    """Raised when an ontology invariant is violated. Message is precise."""


def validate_sample_sets(
    distribution: Distribution,
    label_name: str,
    sample_sets: Iterable[SampleSet],
) -> None:
    """Derived-view consistency (spec §"What must stay in sync": SampleSets are
    DERIVED from labels).

    Guarantees the ``distribution.sample_sets(label_name)`` output is a faithful
    view of the authoritative label group:
      - one set per assigned category, no set for unassigned;
      - the union of set members == exactly the group's assigned samples
        (disjoint across sets, unassigned excluded);
      - every set's ``distribution_id`` / ``sample_set_name`` matches the group.
    The labelers call this after deriving sets, so no caller hand-rolls the check.
    """
    column = distribution.get_label_group(label_name)
    sets = list(sample_sets)

    assigned = {
        sid
        for sid, call in column.assignments.items()
        if call != UNASSIGNED_LABEL and sid in set(distribution.sample_ids)
    }
    expected_categories = [str(c) for c in column.categories()]

    got_names = [s.sample_set_name for s in sets]
    if got_names != expected_categories:
        raise InvariantError(
            "derived-view: sample_sets names "
        f"{got_names} != label-group categories {expected_categories}"
        )

    seen: set[str] = set()
    union: set[str] = set()
    for sset in sets:
        if sset.distribution_id != distribution.distribution_id:
            raise InvariantError(
                f"derived-view: SampleSet {sset.sample_set_id!r}.distribution_id "
                f"{sset.distribution_id!r} != {distribution.distribution_id!r}"
            )
        members = set(sset.sample_ids)
        overlap = union & members
        if overlap:
            raise InvariantError(
                f"derived-view: sample overlaps across sets: {sorted(overlap)}"
            )
        union |= members
        seen.add(sset.sample_set_name)
        # Members must be the group's samples assigned to this category.
        expected_members = {
            sid for sid, call in column.assignments.items() if str(call) == sset.sample_set_name
        }
        if members != (expected_members & assigned):
            raise InvariantError(
                f"derived-view: SampleSet {sset.sample_set_name!r} members disagree "
                "with the label-group assignment"
            )

    if union != assigned:
        missing = sorted(assigned - union)
        extra = sorted(union - assigned)
        raise InvariantError(
            "derived-view: sample_sets do not cover exactly the assigned samples: "
            + (f"missing {missing}; " if missing else "")
            + (f"extra {extra}" if extra else "")
        )


def _check_ordered_features(distribution: Distribution) -> None:
    """#3 — feature_values[:, j] <-> feature_names[j]."""
    if distribution.feature_values.shape[1] != len(distribution.feature_names):
        raise InvariantError(
            "ordered-features (#3): feature_values has "
            f"{distribution.feature_values.shape[1]} columns but "
            f"{len(distribution.feature_names)} feature_names"
        )


def check_density_shape(grid: Grid, density_grid: DensityGrid) -> None:
    """Grid/DensityGrid shape: density.shape == per-axis lengths, ids agree."""
    if density_grid.grid_id != grid.grid_id:
        raise InvariantError(
            f"DensityGrid.grid_id {density_grid.grid_id!r} != Grid.grid_id {grid.grid_id!r}"
        )
    expected = tuple(len(a) for a in grid.axis_values)
    if tuple(density_grid.density.shape) != expected:
        raise InvariantError(
            f"density.shape {tuple(density_grid.density.shape)} != axis lengths {expected}"
        )


def validate_label_group(
    distribution: Distribution,
    label_group: LabelGroup,
    sample_sets: Iterable[SampleSet],
) -> None:
    """Run every applicable guard; raise :class:`InvariantError` on the first fail.

    Checks:
      - #3   ordered features on the Distribution.
      - FK   every SampleSet.distribution_id == distribution.distribution_id.
      - #6b/#10 assignment consistency:
          * every assignment *value* in label_group.sample_set_ids,
          * every assignment *key* in Distribution.sample_ids,
          * SampleSet.sample_ids agrees with the assignment map,
      - #7   coverage: assigned members ∪ unassigned == Distribution.sample_ids,
             assigned ∩ unassigned == ∅ (disjoint, total).
      - unassigned entries are real Distribution samples.
    """
    _check_ordered_features(distribution)

    dist_samples = set(distribution.sample_ids)
    sets = list(sample_sets)
    sets_by_name = {s.sample_set_name: s for s in sets}
    expected_names = {str(category) for category in label_group.categories()}
    if set(sets_by_name) != expected_names:
        raise InvariantError("materialized SampleSets must equal assigned non-unassigned categories")

    # FK: each SampleSet belongs to this Distribution.
    for sset in sets:
        if sset.distribution_id != distribution.distribution_id:
            raise InvariantError(
                f"SampleSet {sset.sample_set_id!r}.distribution_id {sset.distribution_id!r} != "
                f"Distribution {distribution.distribution_id!r}"
            )

    assignment = dict(label_group.assignments)
    for sample_id in assignment:
        # #10 — every assignment key is a real Distribution sample.
        if sample_id not in dist_samples:
            raise InvariantError(
                f"assignment key {sample_id!r} is not a Distribution sample"
            )

    unassigned = {sid for sid, value in assignment.items() if value == UNASSIGNED_LABEL}

    # unassigned must be real samples.
    stray = unassigned - dist_samples
    if stray:
        raise InvariantError(
            f"unassigned_sample_ids not in Distribution: {sorted(stray)}"
        )

    # #7 — assigned ∩ unassigned == ∅.
    assigned_keys = set(assignment) - unassigned
    overlap = assigned_keys & unassigned
    if overlap:
        raise InvariantError(
            f"assigned and unassigned_sample_ids overlap: {sorted(overlap)}"
        )

    # #7 — coverage is total: assigned ∪ unassigned == Distribution.sample_ids.
    covered = assigned_keys | unassigned
    missing = dist_samples - covered
    extra = covered - dist_samples
    if missing or extra:
        raise InvariantError(
            "coverage (#7) failed: "
            + (f"unaccounted samples {sorted(missing)}; " if missing else "")
            + (f"non-Distribution samples {sorted(extra)}" if extra else "")
        )

    # #6b — each SampleSet.sample_ids agrees with the assignment map.
    for name, sset in sets_by_name.items():
        members_from_assignment = {
            sid for sid, tgt in assignment.items() if str(tgt) == name
        }
        members_declared = set(sset.sample_ids)
        if members_from_assignment != members_declared:
            only_map = sorted(members_from_assignment - members_declared)
            only_set = sorted(members_declared - members_from_assignment)
            raise InvariantError(
                f"SampleSet {sset.sample_set_id!r}.sample_ids disagrees with assignment map: "
                + (f"in map only {only_map}; " if only_map else "")
                + (f"in set only {only_set}" if only_set else "")
            )
