from __future__ import annotations

import json

import pandas as pd
import pytest

from src.core.metric.mapping import (
    CompiledMetricMapping,
    MetricMappingArtifact,
    MetricMappingCoverageError,
    MetricMappingEntry,
    MetricMappingValidationError,
)


def _mapping(*entries: MetricMappingEntry) -> MetricMappingArtifact:
    return MetricMappingArtifact(
        name="test_only_exact_perturbation_mapping",
        version="fixture-1",
        metadata_columns=("perturbation_label", "application_hpf"),
        entries=entries,
        scientific_policy=False,
    )


def test_complete_explicit_mapping_preserves_canonical_names_and_row_order() -> None:
    artifact = _mapping(
        MetricMappingEntry(("control", 0), "control_group"),
        MetricMappingEntry(("SU5402", 24), "fgfr_inhibitor_24hpf"),
    )

    mapped = CompiledMetricMapping(artifact).map_records(
        [
            {"perturbation_label": "SU5402", "application_hpf": 24},
            {"perturbation_label": "control", "application_hpf": 0},
        ]
    )

    assert mapped.group_names == ("fgfr_inhibitor_24hpf", "control_group")
    assert mapped.code_to_group == ("control_group", "fgfr_inhibitor_24hpf")
    assert mapped.group_codes == (1, 0)


def test_uncovered_values_are_collected_and_named_without_default_group() -> None:
    artifact = _mapping(MetricMappingEntry(("control", 0), "control_group"))

    with pytest.raises(MetricMappingCoverageError) as exc_info:
        CompiledMetricMapping(artifact).map_records(
            [
                {"perturbation_label": "odd-control-label", "application_hpf": 0},
                {"perturbation_label": "SU5402", "application_hpf": 30},
                {"perturbation_label": "odd-control-label", "application_hpf": 0},
            ]
        )

    message = str(exc_info.value)
    assert "odd-control-label" in message
    assert "SU5402" in message
    assert "30" in message
    assert "uncovered non-null metadata tuples" in message


def test_same_inhibitor_at_different_times_requires_each_exact_tuple() -> None:
    artifact = _mapping(MetricMappingEntry(("SU5402", 24), "fgfr_inhibitor"))
    mapper = CompiledMetricMapping(artifact)

    with pytest.raises(MetricMappingCoverageError, match=r"SU5402.*30"):
        mapper.map_records(
            [{"perturbation_label": "SU5402", "application_hpf": 30}]
        )


def test_same_inhibitor_times_can_be_explicitly_combined_or_separated() -> None:
    combined = _mapping(
        MetricMappingEntry(("SU5402", 24), "fgfr_inhibitor"),
        MetricMappingEntry(("SU5402", 30), "fgfr_inhibitor"),
    )
    separated = _mapping(
        MetricMappingEntry(("SU5402", 24), "fgfr_inhibitor_24hpf"),
        MetricMappingEntry(("SU5402", 30), "fgfr_inhibitor_30hpf"),
    )
    records = [
        {"perturbation_label": "SU5402", "application_hpf": 24},
        {"perturbation_label": "SU5402", "application_hpf": 30},
    ]

    assert CompiledMetricMapping(combined).map_records(records).group_names == (
        "fgfr_inhibitor",
        "fgfr_inhibitor",
    )
    assert CompiledMetricMapping(separated).map_records(records).group_names == (
        "fgfr_inhibitor_24hpf",
        "fgfr_inhibitor_30hpf",
    )


def test_aliases_are_only_combined_by_explicit_entries() -> None:
    artifact = _mapping(
        MetricMappingEntry(("wild-type", 0), "wild_type"),
        MetricMappingEntry(("WT", 0), "wild_type"),
    )
    records = [
        {"perturbation_label": "WT", "application_hpf": 0},
        {"perturbation_label": "wild-type", "application_hpf": 0},
    ]

    assert CompiledMetricMapping(artifact).map_records(records).group_names == (
        "wild_type",
        "wild_type",
    )
    with pytest.raises(MetricMappingCoverageError, match="wt"):
        CompiledMetricMapping(artifact).map_records(
            [{"perturbation_label": "wt", "application_hpf": 0}]
        )


def test_group_codes_are_stable_under_observation_and_artifact_reordering() -> None:
    entries = (
        MetricMappingEntry(("zeta", 0), "group_zeta"),
        MetricMappingEntry(("alpha", 0), "group_alpha"),
    )
    records = [
        {"perturbation_label": "zeta", "application_hpf": 0},
        {"perturbation_label": "alpha", "application_hpf": 0},
    ]
    forward = CompiledMetricMapping(_mapping(*entries)).map_records(records)
    reversed_order = CompiledMetricMapping(_mapping(*reversed(entries))).map_records(
        list(reversed(records))
    )

    assert forward.code_to_group == reversed_order.code_to_group == (
        "group_alpha",
        "group_zeta",
    )
    assert dict(zip(forward.group_names, forward.group_codes)) == dict(
        zip(reversed_order.group_names, reversed_order.group_codes)
    )


def test_all_null_metadata_remains_unassigned_but_partial_null_is_uncovered() -> None:
    artifact = _mapping(MetricMappingEntry(("control", 0), "control"))
    mapper = CompiledMetricMapping(artifact)

    mapped = mapper.map_records(
        [{"perturbation_label": None, "application_hpf": float("nan")}]
    )
    assert mapped.group_names == (None,)
    assert mapped.group_codes == (None,)

    with pytest.raises(MetricMappingCoverageError, match="control.*null"):
        mapper.map_records(
            [{"perturbation_label": "control", "application_hpf": pd.NA}]
        )


def test_missing_configured_metadata_columns_fail_by_name() -> None:
    artifact = _mapping(MetricMappingEntry(("control", 0), "control"))

    with pytest.raises(MetricMappingCoverageError, match="application_hpf"):
        CompiledMetricMapping(artifact).map_records(
            [{"perturbation_label": "control"}]
        )


def test_mapping_serialization_is_deterministic_and_round_trips() -> None:
    first = _mapping(
        MetricMappingEntry(("zeta", 30), "group_zeta"),
        MetricMappingEntry(("alpha", 24), "group_alpha"),
    )
    second = _mapping(*reversed(first.entries))

    assert first.to_json() == second.to_json()
    assert MetricMappingArtifact.from_json(first.to_json()) == first
    payload = json.loads(first.to_json())
    assert payload["version"] == "fixture-1"
    assert payload["scientific_policy"] is False


def test_duplicate_tuples_and_string_number_coercion_are_rejected_or_exact() -> None:
    with pytest.raises(MetricMappingValidationError, match="duplicate exact metadata"):
        _mapping(
            MetricMappingEntry(("control", 0), "control_a"),
            MetricMappingEntry(("control", 0), "control_b"),
        )

    artifact = _mapping(MetricMappingEntry(("treatment", 24), "treatment_24"))
    with pytest.raises(MetricMappingCoverageError, match="'24'"):
        CompiledMetricMapping(artifact).map_records(
            [{"perturbation_label": "treatment", "application_hpf": "24"}]
        )

    assert CompiledMetricMapping(artifact).map_records(
        [{"perturbation_label": "treatment", "application_hpf": 24.0}]
    ).group_names == ("treatment_24",)


def test_required_mapping_artifact_loader_fails_specific_path(tmp_path) -> None:
    missing = tmp_path / "missing-mapping.json"
    with pytest.raises(FileNotFoundError, match="required metric mapping artifact.*missing"):
        MetricMappingArtifact.read_json(missing)

    artifact = _mapping(MetricMappingEntry(("control", 0), "control"))
    path = tmp_path / "mapping.json"
    path.write_text(artifact.to_json(), encoding="utf-8")
    assert MetricMappingArtifact.read_json(path) == artifact
