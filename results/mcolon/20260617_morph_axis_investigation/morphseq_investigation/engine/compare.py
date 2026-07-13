"""Purpose-specific comparisons over the unified label ontology.

Peak correspondence is deliberately absent: peak matching has its own deferred
policy design.  This module currently owns partition agreement and a structured
summary of label groups across a catalog ``DistributionComparison``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Hashable, Mapping

from sklearn.metrics import adjusted_rand_score

from .catalog import DistributionComparison
from .objects import LabelGroup, UNASSIGNED_LABEL


def _readonly_mapping(values):
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class LabelGroupSummaryRow:
    """One comparison member's label summary with structured coordinates."""

    comparison_coordinates: Mapping[str, Hashable]
    member_value: Hashable
    distribution_id: str
    distribution_coordinates: Mapping[str, Hashable]
    label_group_name: str
    sample_set_count: int
    assigned_sample_count: int
    unassigned_sample_count: int
    peak_count: int | None
    is_robust: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison_coordinates", _readonly_mapping(self.comparison_coordinates))
        object.__setattr__(self, "distribution_coordinates", _readonly_mapping(self.distribution_coordinates))


@dataclass(frozen=True)
class PartitionAgreementResult:
    distribution_id: str
    left_label_group_name: str
    right_label_group_name: str
    cross_tab: Mapping[Hashable, Mapping[Hashable, int]] = field(default_factory=dict)
    agreement_metric: str = "adjusted_rand_score"
    agreement_score: float = 0.0
    n_samples_compared: int = 0
    left_only_sample_ids: tuple[str, ...] = ()
    right_only_sample_ids: tuple[str, ...] = ()
    unassigned_both_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cross_tab",
            _readonly_mapping({key: _readonly_mapping(value) for key, value in self.cross_tab.items()}),
        )
        object.__setattr__(self, "left_only_sample_ids", tuple(self.left_only_sample_ids))
        object.__setattr__(self, "right_only_sample_ids", tuple(self.right_only_sample_ids))
        object.__setattr__(self, "unassigned_both_sample_ids", tuple(self.unassigned_both_sample_ids))


def summarize_label_groups(
    comparison: DistributionComparison,
    label_group_name: str,
) -> tuple[LabelGroupSummaryRow, ...]:
    """Summarize one label across comparison members without flattening coordinates.

    Sample-set counts are taken from the authoritative derived view
    ``Distribution.sample_sets``. Missing labels raise rather than silently
    dropping a comparison member.
    """
    rows = []
    for member_value, distribution in comparison.members.items():
        group = distribution.get_label_group(label_group_name)
        sample_sets = distribution.sample_sets(label_group_name)
        assigned = sum(value != UNASSIGNED_LABEL for value in group.assignments.values())
        rows.append(
            LabelGroupSummaryRow(
                comparison_coordinates=comparison.coordinates,
                member_value=member_value,
                distribution_id=distribution.distribution_id,
                distribution_coordinates=distribution.coordinates,
                label_group_name=group.name,
                sample_set_count=len(sample_sets),
                assigned_sample_count=assigned,
                unassigned_sample_count=len(group.assignments) - assigned,
                peak_count=group.peak_count,
                is_robust=group.is_robust,
            )
        )
    return tuple(rows)


def label_group_agreement(left_lg: LabelGroup, right_lg: LabelGroup) -> PartitionAgreementResult:
    """Chance-corrected agreement between two partitions of one distribution."""
    if left_lg.distribution_id != right_lg.distribution_id:
        raise ValueError("label_group_agreement requires the same distribution_id")
    if set(left_lg.assignments) != set(right_lg.assignments):
        raise ValueError("label_group_agreement requires identical sample membership")

    sample_ids = tuple(left_lg.assignments)
    left_assigned = {sid for sid in sample_ids if left_lg.assignments[sid] != UNASSIGNED_LABEL}
    right_assigned = {sid for sid in sample_ids if right_lg.assignments[sid] != UNASSIGNED_LABEL}
    common = tuple(sid for sid in sample_ids if sid in left_assigned and sid in right_assigned)
    left_only = tuple(sid for sid in sample_ids if sid in left_assigned - right_assigned)
    right_only = tuple(sid for sid in sample_ids if sid in right_assigned - left_assigned)
    neither = tuple(sid for sid in sample_ids if sid not in left_assigned | right_assigned)

    cross_tab = {
        left: {right: 0 for right in right_lg.categories()}
        for left in left_lg.categories()
    }
    for sid in common:
        cross_tab[left_lg.assignments[sid]][right_lg.assignments[sid]] += 1

    score = (
        float(adjusted_rand_score(
            [left_lg.assignments[sid] for sid in common],
            [right_lg.assignments[sid] for sid in common],
        ))
        if common else float("nan")
    )
    return PartitionAgreementResult(
        distribution_id=left_lg.distribution_id,
        left_label_group_name=left_lg.name,
        right_label_group_name=right_lg.name,
        cross_tab=cross_tab,
        agreement_score=score,
        n_samples_compared=len(common),
        left_only_sample_ids=left_only,
        right_only_sample_ids=right_only,
        unassigned_both_sample_ids=neither,
    )


__all__ = [
    "LabelGroupSummaryRow",
    "PartitionAgreementResult",
    "label_group_agreement",
    "summarize_label_groups",
]
