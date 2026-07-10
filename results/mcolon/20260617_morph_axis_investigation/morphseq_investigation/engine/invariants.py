"""Central invariant guards — called by labelers before returning, not by users.

Every check the ontology's Invariants list (#3, #6b/#10, #7, and the Grid shape
rule) makes enforceable is here, behind one entry point
:func:`validate_label_group`, so no labeler hand-rolls its own drift check.
"""

from __future__ import annotations

from typing import Iterable

from .objects import (
    Distribution,
    DensityGrid,
    Grid,
    LabelGroup,
    SampleSet,
)


class InvariantError(ValueError):
    """Raised when an ontology invariant is violated. Message is precise."""


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
    sets_by_id = {s.sample_set_id: s for s in sample_sets}

    # sample_set_ids on the group must correspond to provided SampleSets.
    for set_id in label_group.sample_set_ids:
        if set_id not in sets_by_id:
            raise InvariantError(
                f"sample_set_id {set_id!r} in label_group.sample_set_ids has no "
                "matching SampleSet passed to validate_label_group"
            )

    # FK: each SampleSet belongs to this Distribution.
    for set_id in label_group.sample_set_ids:
        sset = sets_by_id[set_id]
        if sset.distribution_id != distribution.distribution_id:
            raise InvariantError(
                f"SampleSet {set_id!r}.distribution_id {sset.distribution_id!r} != "
                f"Distribution {distribution.distribution_id!r}"
            )

    assignment = dict(label_group.sample_id_to_sample_set_id)
    valid_set_ids = set(label_group.sample_set_ids)

    # #10 — every assignment value is a declared sample_set_id.
    for sample_id, set_id in assignment.items():
        if set_id not in valid_set_ids:
            raise InvariantError(
                f"assignment value {set_id!r} (sample {sample_id!r}) not in "
                f"label_group.sample_set_ids"
            )
        # #10 — every assignment key is a real Distribution sample.
        if sample_id not in dist_samples:
            raise InvariantError(
                f"assignment key {sample_id!r} is not a Distribution sample"
            )

    unassigned = set(label_group.unassigned_sample_ids)

    # unassigned must be real samples.
    stray = unassigned - dist_samples
    if stray:
        raise InvariantError(
            f"unassigned_sample_ids not in Distribution: {sorted(stray)}"
        )

    # #7 — assigned ∩ unassigned == ∅.
    assigned_keys = set(assignment)
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
    for set_id in label_group.sample_set_ids:
        sset = sets_by_id[set_id]
        members_from_assignment = {
            sid for sid, tgt in assignment.items() if tgt == set_id
        }
        members_declared = set(sset.sample_ids)
        if members_from_assignment != members_declared:
            only_map = sorted(members_from_assignment - members_declared)
            only_set = sorted(members_declared - members_from_assignment)
            raise InvariantError(
                f"SampleSet {set_id!r}.sample_ids disagrees with assignment map: "
                + (f"in map only {only_map}; " if only_map else "")
                + (f"in set only {only_set}" if only_set else "")
            )
