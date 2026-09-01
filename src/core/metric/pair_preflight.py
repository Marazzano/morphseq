"""Cohort-wide legal-positive preflight for split-local metric indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.core.metric.pairing import MetricPairIndex, PairCandidateCounts


class MetricPairPreflightError(ValueError):
    """Base error for pair preflight failures."""


class NoLegalPositiveError(MetricPairPreflightError):
    """One or more accepted anchors have no legal positive."""


@dataclass(frozen=True)
class PairPreflightReport:
    policy_name: str
    policy_version: str
    split: str
    anchor_count: int
    anchors_with_same_embryo_candidates: int
    anchors_with_different_embryo_candidates: int
    minimum_candidate_count: int
    maximum_candidate_count: int


def preflight_pair_index(pair_index: MetricPairIndex) -> PairPreflightReport:
    """Validate every accepted anchor and return deterministic count diagnostics."""

    if not isinstance(pair_index, MetricPairIndex):
        raise TypeError("pair_index must be a MetricPairIndex")
    counts: list[PairCandidateCounts] = []
    missing: list[str] = []
    for anchor_index, snip_id in enumerate(pair_index.snip_ids):
        candidate_counts = pair_index.candidate_counts(anchor_index)
        counts.append(candidate_counts)
        if not candidate_counts.total:
            missing.append(snip_id)
    if missing:
        policy = pair_index.policy
        raise NoLegalPositiveError(
            f"pair policy {policy.name!r}@{policy.version} split={pair_index.split!r} has "
            f"anchors with no legal positive: snip_id values={missing!r}; "
            f"stage_source={policy.stage_source!r}, stage_column={policy.stage_column!r}, "
            f"sampler_age_window={float(policy.sampler_age_window):g}, "
            f"same_embryo_enabled={policy.same_embryo.enabled}, "
            f"different_embryo_enabled={policy.different_embryo.enabled}"
        )
    totals = [candidate_count.total for candidate_count in counts]
    return PairPreflightReport(
        policy_name=pair_index.policy.name,
        policy_version=pair_index.policy.version,
        split=pair_index.split,
        anchor_count=len(pair_index),
        anchors_with_same_embryo_candidates=sum(
            candidate_count.same_embryo > 0 for candidate_count in counts
        ),
        anchors_with_different_embryo_candidates=sum(
            candidate_count.different_embryo > 0 for candidate_count in counts
        ),
        minimum_candidate_count=min(totals),
        maximum_candidate_count=max(totals),
    )


def preflight_pair_indices(
    pair_indices: Mapping[str, MetricPairIndex],
) -> tuple[PairPreflightReport, ...]:
    """Preflight an explicitly supplied mapping of split name to split-local index."""

    reports = []
    for split in sorted(pair_indices):
        pair_index = pair_indices[split]
        if pair_index.split != split:
            raise MetricPairPreflightError(
                f"pair index mapping key={split!r} disagrees with index split={pair_index.split!r}"
            )
        reports.append(preflight_pair_index(pair_index))
    return tuple(reports)
