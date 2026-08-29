"""Explicit metric-group mapping, relations, and provenance payloads."""

from __future__ import annotations

from typing import Any, Mapping

from .mapping import (
    CompiledMetricMapping,
    MappedMetricGroups,
    MetricMappingArtifact,
    MetricMappingCoverageError,
    MetricMappingEntry,
    MetricMappingError,
    MetricMappingValidationError,
)
from .relations import (
    MetricGroupDefinition,
    MetricRelation,
    MetricRelationCoverageError,
    MetricRelationError,
    MetricRelationPolicy,
    MetricRelationValidationError,
    PairRelation,
    UnknownMetricGroupError,
    build_d35_compatibility_candidate,
    build_test_only_policy,
    validate_policy_for_preset,
)


METRIC_PROVENANCE_FORMAT_VERSION = "1.0"


def build_metric_provenance_payload(
    *, mapping: MetricMappingArtifact, relation_policy: MetricRelationPolicy
) -> dict[str, Any]:
    """Return the exact mapping and relation policy for run provenance."""

    _validate_bundle_coverage(mapping, relation_policy)
    return {
        "format_version": METRIC_PROVENANCE_FORMAT_VERSION,
        "mapping": mapping.to_dict(),
        "relation_policy": relation_policy.to_dict(),
    }


def load_metric_provenance_payload(
    payload: Mapping[str, Any],
) -> tuple[MetricMappingArtifact, MetricRelationPolicy]:
    """Round-trip a persisted metric mapping/relation provenance payload."""

    if payload.get("format_version") != METRIC_PROVENANCE_FORMAT_VERSION:
        raise MetricRelationValidationError(
            "unsupported metric provenance format_version="
            f"{payload.get('format_version')!r}; expected "
            f"{METRIC_PROVENANCE_FORMAT_VERSION!r}"
        )
    try:
        mapping = MetricMappingArtifact.from_dict(payload["mapping"])
        relation_policy = MetricRelationPolicy.from_dict(payload["relation_policy"])
    except KeyError as exc:
        raise MetricRelationValidationError(
            f"metric provenance payload is missing field {exc.args[0]!r}"
        ) from exc
    _validate_bundle_coverage(mapping, relation_policy)
    return mapping, relation_policy


def validate_metric_bundle_for_preset(
    *,
    mapping: MetricMappingArtifact,
    relation_policy: MetricRelationPolicy,
    scientific_preset: bool,
) -> None:
    """Validate group coverage and isolate non-scientific policies from presets."""

    _validate_bundle_coverage(mapping, relation_policy)
    validate_policy_for_preset(relation_policy, scientific_preset=scientific_preset)
    if scientific_preset and not mapping.scientific_policy:
        raise MetricMappingValidationError(
            f"scientific preset cannot use mapping {mapping.name!r}@{mapping.version} "
            "because scientific_policy is false"
        )


def _validate_bundle_coverage(
    mapping: MetricMappingArtifact, relation_policy: MetricRelationPolicy
) -> None:
    mapped_groups = set(mapping.group_names)
    relation_groups = set(relation_policy.group_names)
    if mapped_groups != relation_groups:
        raise MetricRelationCoverageError(
            "mapping and relation policy group identities differ; "
            f"mapping_only={sorted(mapped_groups - relation_groups)!r}, "
            f"relation_only={sorted(relation_groups - mapped_groups)!r}"
        )


__all__ = [
    "CompiledMetricMapping",
    "MappedMetricGroups",
    "MetricGroupDefinition",
    "MetricMappingArtifact",
    "MetricMappingCoverageError",
    "MetricMappingEntry",
    "MetricMappingError",
    "MetricMappingValidationError",
    "MetricRelation",
    "MetricRelationCoverageError",
    "MetricRelationError",
    "MetricRelationPolicy",
    "MetricRelationValidationError",
    "PairRelation",
    "UnknownMetricGroupError",
    "build_d35_compatibility_candidate",
    "build_metric_provenance_payload",
    "build_test_only_policy",
    "load_metric_provenance_payload",
    "validate_metric_bundle_for_preset",
    "validate_policy_for_preset",
]
