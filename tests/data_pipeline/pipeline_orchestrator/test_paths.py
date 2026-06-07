"""Tests for the orchestration path registry (orchestration/paths.py).

These pin the PUBLIC CONTRACT (resolved paths + the rules of the system), not the private
implementation. Resolved paths are pinned hard (a filename change SHOULD make these yell);
error-message assertions check for the important words, not exact prose (so rewording an error
does not break a test). Spec:
docs/refactors/streamline-snakemake/target/front_end_naming_and_flow.md (§ Resolved paths).

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


class TestResolvedArtifactPaths:
    """Pin the exact resolved path for each step — the registry's core job."""

    def test_ingest_plate_metadata(self):
        assert artifact_path(ROOT, "ingest_plate_metadata", "csv", EXP) == \
            ROOT / "experiment_metadata" / EXP / "plate_metadata.csv"

    def test_ingest_scope_metadata_scope_token(self):
        # `scope` is a format token supplied via format_vars.
        assert artifact_path(ROOT, "ingest_scope_metadata", "raw", EXP,
                             format_vars={"scope": "yx1"}) == \
            ROOT / "experiment_metadata" / EXP / "scope_metadata__yx1.csv"

    def test_map_series_to_wells(self):
        assert artifact_path(ROOT, "map_series_to_wells", "mapping", EXP) == \
            ROOT / "experiment_metadata" / EXP / "series_well_mapping.csv"

    def test_join_series_mapping_to_scope_metadata(self):
        assert artifact_path(ROOT, "join_series_mapping_to_scope_metadata", "mapped", EXP) == \
            ROOT / "experiment_metadata" / EXP / "scope_metadata_mapped.csv"

    def test_discover_wells(self):
        assert artifact_path(ROOT, "discover_wells", "wells", EXP) == \
            ROOT / "experiment_metadata" / EXP / "discovered_wells.txt"

    def test_frame_inventory_per_well_names_the_well(self):
        # per-well shard uses {well_id}.
        assert artifact_path(ROOT, "frame_inventory", "inventory", EXP,
                             path_mode="per_well", well_id=WELL) == \
            ROOT / "experiment_metadata" / EXP / PER_WELL_DIRNAME / WELL / f"{WELL}_frame_inventory.csv"

    def test_frame_inventory_merged_names_the_experiment(self):
        # merged view uses {experiment_id}, NOT a fabricated well_id.
        assert artifact_path(ROOT, "frame_inventory", "inventory", EXP, path_mode="merged") == \
            ROOT / "experiment_metadata" / EXP / f"{EXP}_frame_inventory.csv"


class TestDerivedSidecarPaths:
    """validated_/provenance_ are general helpers: artifact_path + a fixed suffix. Test the
    suffix behavior against both an experiment-grain and a per-well artifact."""

    def test_validated_path_appends_suffix_to_experiment_artifact(self):
        base = artifact_path(ROOT, "join_series_mapping_to_scope_metadata", "mapped", EXP)
        assert validated_path(ROOT, "join_series_mapping_to_scope_metadata", "mapped", EXP) == \
            base.with_name(base.name + ".validated")

    def test_validated_path_appends_suffix_to_per_well_artifact(self):
        base = artifact_path(ROOT, "frame_inventory", "inventory", EXP,
                             path_mode="per_well", well_id=WELL)
        assert validated_path(ROOT, "frame_inventory", "inventory", EXP,
                              path_mode="per_well", well_id=WELL) == \
            base.with_name(base.name + ".validated")

    def test_provenance_path_appends_suffix_to_artifact(self):
        base = artifact_path(ROOT, "map_series_to_wells", "mapping", EXP)
        assert provenance_path(ROOT, "map_series_to_wells", "mapping", EXP) == \
            base.with_name(base.name + ".provenance.json")


class TestPathModeRules:
    """fanout is executable: experiment steps accept only `experiment`; per_well_then_merge steps
    accept only `per_well` or `merged`. These prove that rule in both directions."""

    def test_experiment_step_defaults_to_experiment_mode(self):
        # path_mode=None on an experiment-grain step resolves to the one legal mode.
        assert step_dir(ROOT, "discover_wells", EXP) == \
            ROOT / "experiment_metadata" / EXP

    def test_frame_inventory_requires_explicit_path_mode(self):
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "frame_inventory", "inventory", EXP)
        message = str(excinfo.value)
        assert "frame_inventory" in message
        assert "per_well" in message
        assert "merged" in message

    def test_experiment_mode_rejected_for_per_well_then_merge_step(self):
        # experiment and merged land in the same dir, but a per-well step must say `merged`.
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "frame_inventory", "inventory", EXP, path_mode="experiment")
        message = str(excinfo.value)
        assert "frame_inventory" in message
        assert "per_well" in message
        assert "merged" in message
        assert "experiment" in message

    def test_per_well_mode_rejected_for_experiment_step(self):
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "discover_wells", "wells", EXP, path_mode="per_well")
        message = str(excinfo.value)
        assert "discover_wells" in message
        assert "experiment" in message
        assert "per_well" in message

    def test_per_well_requires_well_id(self):
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "frame_inventory", "inventory", EXP, path_mode="per_well")
        assert "well_id" in str(excinfo.value)

    def test_unknown_path_mode_raises(self):
        with pytest.raises(ValueError):
            step_dir(ROOT, "discover_wells", EXP, path_mode="sideways")


class TestTemplateAndIdentityRules:
    """The no-leakage boundary + clear template errors."""

    def test_identity_tokens_cannot_be_passed_via_format_vars(self):
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "ingest_plate_metadata", "csv", EXP,
                          format_vars={"experiment_id": "fake", "well_id": "fake"})
        message = str(excinfo.value)
        assert "format_vars" in message
        assert "experiment_id" in message
        assert "well_id" in message

    def test_missing_template_token_names_the_template(self):
        # ingest_scope_metadata's template needs {scope}; omitting it must name both the token
        # and the template, not raise a bare KeyError.
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "ingest_scope_metadata", "raw", EXP)
        message = str(excinfo.value)
        assert "scope" in message
        assert "scope_metadata__{scope}.csv" in message


class TestErrorPaths:
    """Unknown keys fail loudly."""

    def test_unknown_step_raises(self):
        with pytest.raises(KeyError):
            artifact_path(ROOT, "no_such_step", "x", EXP)

    def test_unknown_artifact_raises(self):
        with pytest.raises(KeyError):
            artifact_path(ROOT, "discover_wells", "no_such_artifact", EXP)


class TestRegistryIntrospection:
    """Introspection helpers + registry-shape invariants (the only place that pokes at
    PIPELINE_STEPS directly)."""

    def test_known_steps_and_artifacts(self):
        assert "frame_inventory" in known_steps()
        assert known_steps() == tuple(sorted(known_steps()))
        assert known_artifacts("frame_inventory") == ("inventory",)
        assert known_artifacts("discover_wells") == ("wells",)

    def test_every_step_has_required_keys(self):
        for step, spec in PIPELINE_STEPS.items():
            assert "stage" in spec, f"{step} missing stage"
            assert spec["fanout"] in ("experiment", "per_well_then_merge"), f"{step} bad fanout"
            assert spec["artifacts"], f"{step} has no artifacts"

    def test_every_artifact_template_is_string_or_mode_dict(self):
        # The registry's shape convention: a template is a string, or a {path_mode: string} dict.
        valid_modes = ("per_well", "merged", "experiment")
        for step, spec in PIPELINE_STEPS.items():
            for artifact, template in spec["artifacts"].items():
                assert isinstance(template, (str, dict)), f"{step}/{artifact} bad template type"
                if isinstance(template, dict):
                    assert template, f"{step}/{artifact} empty mode dict"
                    for mode, mode_template in template.items():
                        assert mode in valid_modes, f"{step}/{artifact} bad mode {mode!r}"
                        assert isinstance(mode_template, str), f"{step}/{artifact}/{mode} not str"
