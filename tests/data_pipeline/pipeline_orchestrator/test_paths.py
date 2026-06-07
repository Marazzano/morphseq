"""Tests for the orchestration path registry (orchestration/paths.py).

These pin the resolved paths to the worked examples in
docs/refactors/streamline-snakemake/target/front_end_naming_and_flow.md (§ Resolved paths),
so the registry and that spec cannot drift apart silently.

Run with: PYTHONPATH=src:$PYTHONPATH pytest tests/data_pipeline/pipeline_orchestrator/test_paths.py
"""

from pathlib import Path

import pytest

from data_pipeline.pipeline_orchestrator.orchestration import (
    PER_WELL_DIRNAME,
    PIPELINE_STEPS,
    artifact_path,
    known_artifacts,
    known_steps,
    provenance_path,
    step_dir,
    validated_path,
)

ROOT = Path("/ROOT")
EXP = "20250912"
WELL = "20250912_B01"


# ── resolved paths match the front-end doc's worked examples ────────────────────────────────

def test_ingest_plate_metadata():
    assert artifact_path(ROOT, "ingest_plate_metadata", "csv", EXP) == \
        ROOT / "experiment_metadata" / EXP / "plate_metadata.csv"


def test_ingest_scope_metadata_scope_token():
    assert artifact_path(ROOT, "ingest_scope_metadata", "raw", EXP, format_vars={"scope": "yx1"}) == \
        ROOT / "experiment_metadata" / EXP / "scope_metadata__yx1.csv"


def test_map_series_to_wells_and_provenance():
    assert artifact_path(ROOT, "map_series_to_wells", "mapping", EXP) == \
        ROOT / "experiment_metadata" / EXP / "series_well_mapping.csv"
    assert provenance_path(ROOT, "map_series_to_wells", "mapping", EXP) == \
        ROOT / "experiment_metadata" / EXP / "series_well_mapping.csv.provenance.json"


def test_join_and_validated_sentinel():
    assert artifact_path(ROOT, "join_series_mapping_to_scope_metadata", "mapped", EXP) == \
        ROOT / "experiment_metadata" / EXP / "scope_metadata_mapped.csv"
    assert validated_path(ROOT, "join_series_mapping_to_scope_metadata", "mapped", EXP) == \
        ROOT / "experiment_metadata" / EXP / "scope_metadata_mapped.csv.validated"


def test_discover_wells():
    assert artifact_path(ROOT, "discover_wells", "wells", EXP) == \
        ROOT / "experiment_metadata" / EXP / "discovered_wells.txt"


def test_frame_inventory_per_well_embeds_well_id():
    assert artifact_path(ROOT, "frame_inventory", "inventory", EXP,
                         path_mode="per_well", well_id=WELL) == \
        ROOT / "experiment_metadata" / EXP / PER_WELL_DIRNAME / WELL / f"{WELL}_frame_inventory.csv"
    assert validated_path(ROOT, "frame_inventory", "inventory", EXP,
                          path_mode="per_well", well_id=WELL) == \
        ROOT / "experiment_metadata" / EXP / PER_WELL_DIRNAME / WELL / f"{WELL}_frame_inventory.csv.validated"


def test_frame_inventory_merged_uses_experiment_id():
    # The concatenated experiment view is named with the experiment id, not a well_id.
    assert artifact_path(ROOT, "frame_inventory", "inventory", EXP, path_mode="merged") == \
        ROOT / "experiment_metadata" / EXP / f"{EXP}_frame_inventory.csv"


def test_step_dir_per_well():
    assert step_dir(ROOT, "frame_inventory", EXP, path_mode="per_well", well_id=WELL) == \
        ROOT / "experiment_metadata" / EXP / PER_WELL_DIRNAME / WELL


# ── error paths fail loudly ─────────────────────────────────────────────────────────────────

def test_unknown_stage_raises():
    with pytest.raises(KeyError):
        artifact_path(ROOT, "no_such_stage", "x", EXP)


def test_unknown_artifact_raises():
    with pytest.raises(KeyError):
        artifact_path(ROOT, "discover_wells", "no_such_artifact", EXP)


def test_per_well_without_well_id_raises():
    with pytest.raises(ValueError):
        artifact_path(ROOT, "frame_inventory", "inventory", EXP, path_mode="per_well")


def test_unknown_path_mode_raises():
    with pytest.raises(ValueError):
        step_dir(ROOT, "discover_wells", EXP, path_mode="sideways")


# ── explicit mode and token errors ───────────────────────────────────────────────────────────

def test_frame_inventory_requires_explicit_path_mode():
    with pytest.raises(ValueError, match="Pass path_mode='per_well' or path_mode='merged' explicitly"):
        artifact_path(ROOT, "frame_inventory", "inventory", EXP)


def test_identity_tokens_cannot_be_passed_via_format_vars():
    with pytest.raises(ValueError, match="Do not pass identity tokens via format_vars"):
        artifact_path(
            ROOT,
            "ingest_plate_metadata",
            "csv",
            EXP,
            format_vars={"experiment_id": "fake", "well_id": "fake"},
        )


def test_missing_template_token_names_template():
    with pytest.raises(ValueError) as excinfo:
        artifact_path(ROOT, "ingest_scope_metadata", "raw", EXP)

    message = str(excinfo.value)
    assert "scope" in message
    assert "scope_metadata__{scope}.csv" in message


def test_known_steps_and_artifacts_are_sorted():
    assert "frame_inventory" in known_steps()
    assert known_artifacts("frame_inventory") == ("inventory",)
    assert known_artifacts("discover_wells") == ("wells",)


# ── registry shape invariants ───────────────────────────────────────────────────────────────

def test_every_step_has_required_keys():
    for step, spec in PIPELINE_STEPS.items():
        assert "stage" in spec, f"{step} missing stage"
        assert spec["fanout"] in ("experiment", "per_well_then_merge"), f"{step} bad fanout"
        assert spec["artifacts"], f"{step} has no artifacts"
