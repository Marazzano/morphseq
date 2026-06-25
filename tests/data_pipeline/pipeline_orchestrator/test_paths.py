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
    EXECUTION_PER_WELL,
    EXECUTION_RUN_BATCH,
    PER_WELL_DIRNAME,
    PER_WELL_THEN_MERGE,
    PIPELINE_STEPS,
    artifact_path,
    execution_mode,
    known_artifacts,
    known_steps,
    per_well_step_dir,
    provenance_path,
    step_dir,
    validated_path,
)

# Doctrine-allowed regime names and forbidden legacy names (output_tree_doctrine.md)
_ALLOWED_STAGES = frozenset({
    "acquisition", "object_extraction", "features", "quality_control", "analysis_ready"
})
_FORBIDDEN_STAGE_NAMES = frozenset({
    "experiment_metadata", "built_image_data", "detection", "segmentation"
})

ROOT = Path("/ROOT")
EXP = "20250912"
WELL = "20250912_B01"
PRODUCT_KEY = "BF__z_stack"


class TestResolvedArtifactPaths:
    """Pin the exact resolved path for each step — the registry's core job."""

    def test_ingest_plate_metadata(self):
        assert artifact_path(ROOT, "ingest_plate_metadata", "csv", EXP) == \
            ROOT / "acquisition" / EXP / "plate_metadata.csv"

    def test_ingest_scope_metadata_scope_token(self):
        # `scope` is a format token supplied via format_vars.
        assert artifact_path(ROOT, "ingest_scope_metadata", "raw", EXP,
                             format_vars={"scope": "yx1"}) == \
            ROOT / "acquisition" / EXP / "scope_metadata__yx1.csv"

    def test_map_positions_to_wells(self):
        assert artifact_path(ROOT, "map_positions_to_wells", "mapping", EXP) == \
            ROOT / "acquisition" / EXP / "position_well_mapping.csv"

    def test_apply_position_to_well_mapping(self):
        assert artifact_path(ROOT, "apply_position_to_well_mapping", "mapped", EXP) == \
            ROOT / "acquisition" / EXP / "scope_metadata_mapped.csv"

    def test_discover_wells(self):
        assert artifact_path(ROOT, "discover_wells", "wells", EXP) == \
            ROOT / "acquisition" / EXP / "discovered_wells.txt"

    def test_frame_inventory_per_well_names_the_well(self):
        # per-well shard: acquisition/<exp>/frame_inventory/per_well/<well_id>/...
        assert artifact_path(ROOT, "frame_inventory", "inventory", EXP,
                             path_mode="per_well", well_id=WELL) == \
            ROOT / "acquisition" / EXP / "frame_inventory" / PER_WELL_DIRNAME / WELL / f"{WELL}_frame_inventory.csv"

    def test_frame_inventory_merged_names_the_experiment(self):
        # merged view uses {experiment_id}, NOT a fabricated well_id.
        assert artifact_path(ROOT, "frame_inventory", "inventory", EXP, path_mode="merged") == \
            ROOT / "acquisition" / EXP / "frame_inventory" / f"{EXP}_frame_inventory.csv"

    def test_materialize_well_done_under_materialized_images(self):
        # done sentinel: acquisition/<exp>/materialized_images/per_well/<well_id>/...
        assert artifact_path(ROOT, "materialize_well", "done", EXP,
                             path_mode="per_well", well_id=WELL) == \
            ROOT / "acquisition" / EXP / "materialized_images" / PER_WELL_DIRNAME / WELL / f"{WELL}.materialize_well.done"

    def test_resolved_product_plan_uses_product_key_filename(self):
        assert artifact_path(
            ROOT,
            "resolved_product_plans",
            "json",
            EXP,
            path_mode="per_well",
            well_id=WELL,
            format_vars={"product_key": PRODUCT_KEY},
        ) == (
            ROOT / "acquisition" / EXP / "resolved_product_plans" / PER_WELL_DIRNAME
            / WELL / f"{PRODUCT_KEY}_resolved_product_plan.json"
        )

    def test_frame_inventory_product_shard_uses_well_and_product_key_filename(self):
        assert artifact_path(
            ROOT,
            "frame_inventory_products",
            "inventory",
            EXP,
            path_mode="per_well",
            well_id=WELL,
            format_vars={"product_key": PRODUCT_KEY},
        ) == (
            ROOT / "acquisition" / EXP / "frame_inventory_products" / PER_WELL_DIRNAME
            / WELL / f"{WELL}_{PRODUCT_KEY}_frame_inventory.csv"
        )

    def test_discovered_product_shards_names_the_well(self):
        assert artifact_path(
            ROOT,
            "discovered_product_shards",
            "csv",
            EXP,
            path_mode="per_well",
            well_id=WELL,
        ) == (
            ROOT / "acquisition" / EXP / "discovered_product_shards" / PER_WELL_DIRNAME
            / WELL / f"{WELL}_discovered_product_shards.csv"
        )

    def test_frame_detections_per_well_names_the_well(self):
        assert artifact_path(ROOT, "frame_detections", "frame_detections", EXP,
                             path_mode="per_well", well_id=WELL) == \
            ROOT / "object_extraction" / EXP / "frame_detections" / PER_WELL_DIRNAME / WELL / f"{WELL}_frame_detections.csv"

    def test_frame_detections_merged_names_the_experiment(self):
        assert artifact_path(ROOT, "frame_detections", "frame_detections", EXP, path_mode="merged") == \
            ROOT / "object_extraction" / EXP / "frame_detections" / f"{EXP}_frame_detections.csv"

    def test_frame_masks_per_well_names_the_well(self):
        assert artifact_path(ROOT, "frame_masks", "frame_masks", EXP,
                             path_mode="per_well", well_id=WELL) == \
            ROOT / "object_extraction" / EXP / "frame_masks" / PER_WELL_DIRNAME / WELL / f"{WELL}_frame_masks.csv"

    def test_frame_masks_merged_names_the_experiment(self):
        assert artifact_path(ROOT, "frame_masks", "frame_masks", EXP, path_mode="merged") == \
            ROOT / "object_extraction" / EXP / "frame_masks" / f"{EXP}_frame_masks.csv"

    def test_snip_auxiliary_masks_manifest_lands_under_object_extraction(self):
        assert artifact_path(ROOT, "snip_auxiliary_masks", "manifest", EXP, path_mode="merged") == \
            ROOT / "object_extraction" / EXP / "snip_auxiliary_masks" / f"{EXP}_snip_auxiliary_masks.csv"

    def test_snip_auxiliary_masks_per_well_manifest_lands_under_per_well(self):
        assert artifact_path(ROOT, "snip_auxiliary_masks", "manifest", EXP, path_mode="per_well", well_id=WELL) == \
            ROOT / "object_extraction" / EXP / "snip_auxiliary_masks" / PER_WELL_DIRNAME / WELL / f"{WELL}_snip_auxiliary_masks.csv"

    def test_prompt_seeds_sidecar_is_per_well_only(self):
        assert artifact_path(ROOT, "frame_masks", "prompt_seeds", EXP,
                             path_mode="per_well", well_id=WELL) == \
            ROOT / "object_extraction" / EXP / "frame_masks" / PER_WELL_DIRNAME / WELL / f"{WELL}_prompt_seeds.csv"


class TestDerivedSidecarPaths:
    """validated_/provenance_ are general helpers: artifact_path + a fixed suffix. Test the
    suffix behavior against both an experiment-grain and a per-well artifact."""

    def test_validated_path_appends_suffix_to_experiment_artifact(self):
        base = artifact_path(ROOT, "apply_position_to_well_mapping", "mapped", EXP)
        assert validated_path(ROOT, "apply_position_to_well_mapping", "mapped", EXP) == \
            base.with_name(base.name + ".validated")

    def test_validated_path_appends_suffix_to_per_well_artifact(self):
        base = artifact_path(ROOT, "frame_inventory", "inventory", EXP,
                             path_mode="per_well", well_id=WELL)
        assert validated_path(ROOT, "frame_inventory", "inventory", EXP,
                              path_mode="per_well", well_id=WELL) == \
            base.with_name(base.name + ".validated")

    def test_validated_path_appends_suffix_to_product_frame_inventory_shard(self):
        base = artifact_path(
            ROOT,
            "frame_inventory_products",
            "inventory",
            EXP,
            path_mode="per_well",
            well_id=WELL,
            format_vars={"product_key": PRODUCT_KEY},
        )
        assert validated_path(
            ROOT,
            "frame_inventory_products",
            "inventory",
            EXP,
            path_mode="per_well",
            well_id=WELL,
            format_vars={"product_key": PRODUCT_KEY},
        ) == base.with_name(base.name + ".validated")

    def test_provenance_path_appends_suffix_to_artifact(self):
        base = artifact_path(ROOT, "map_positions_to_wells", "mapping", EXP)
        assert provenance_path(ROOT, "map_positions_to_wells", "mapping", EXP) == \
            base.with_name(base.name + ".provenance.json")


class TestPathModeRules:
    """fanout is executable: experiment steps accept only `experiment`; per_well_then_merge steps
    accept only `per_well` or `merged`. These prove that rule in both directions."""

    def test_experiment_step_defaults_to_experiment_mode(self):
        # path_mode=None on an experiment-grain step resolves to the one legal mode.
        assert step_dir(ROOT, "discover_wells", EXP) == \
            ROOT / "acquisition" / EXP

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

    def test_prompt_seeds_has_no_merged_view(self):
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "frame_masks", "prompt_seeds", EXP, path_mode="merged")
        message = str(excinfo.value)
        assert "prompt_seeds" in message
        assert "merged" in message

    def test_per_well_requires_well_id(self):
        with pytest.raises(ValueError) as excinfo:
            artifact_path(ROOT, "frame_inventory", "inventory", EXP, path_mode="per_well")
        assert "well_id" in str(excinfo.value)

    def test_unknown_path_mode_raises(self):
        with pytest.raises(ValueError):
            step_dir(ROOT, "discover_wells", EXP, path_mode="sideways")


class TestPerWellStepDir:
    """per_well_step_dir names the directory holding ALL of a step's per-well shards
    ({stage}/{exp}/per_well) — the place the merge lists to discover which wells have shards.
    It composes with step_dir so a specific shard dir is always per_well_step_dir(...) / well_id."""

    def test_per_well_step_dir_returns_directory_containing_shards(self):
        # frame_inventory has product_dir="frame_inventory" → per_well sits inside it.
        assert per_well_step_dir(ROOT, "frame_inventory", EXP) == \
            ROOT / "acquisition" / EXP / "frame_inventory" / PER_WELL_DIRNAME

    def test_step_dir_per_well_is_per_well_step_dir_plus_well_id(self):
        # The composition rule: a specific well's shard dir is the per-well dir + the well_id.
        assert step_dir(ROOT, "frame_inventory", EXP, path_mode="per_well", well_id=WELL) == \
            per_well_step_dir(ROOT, "frame_inventory", EXP) / WELL

    def test_per_well_step_dir_rejects_experiment_step(self):
        # An experiment-grain step has no per-well directory — fail loud.
        with pytest.raises(ValueError) as excinfo:
            per_well_step_dir(ROOT, "discover_wells", EXP)
        message = str(excinfo.value)
        assert "discover_wells" in message
        assert "per_well" in message


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

    def test_missing_product_key_format_var_fails_loud(self):
        with pytest.raises(ValueError) as excinfo:
            artifact_path(
                ROOT,
                "resolved_product_plans",
                "json",
                EXP,
                path_mode="per_well",
                well_id=WELL,
            )
        message = str(excinfo.value)
        assert "product_key" in message
        assert "{product_key}_resolved_product_plan.json" in message


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
        assert "frame_detections" in known_steps()
        assert "frame_masks" in known_steps()
        assert "snip_auxiliary_masks" in known_steps()
        assert known_steps() == tuple(sorted(known_steps()))
        assert known_artifacts("frame_inventory") == ("inventory",)
        assert known_artifacts("resolved_product_plans") == ("json",)
        assert known_artifacts("frame_inventory_products") == ("inventory",)
        assert known_artifacts("discovered_product_shards") == ("csv",)
        assert known_artifacts("frame_detections") == ("frame_detections",)
        assert known_artifacts("frame_masks") == ("frame_masks", "prompt_seeds")
        assert known_artifacts("snip_auxiliary_masks") == ("manifest",)
        assert known_artifacts("discover_wells") == ("wells",)

    def test_every_step_has_required_keys(self):
        valid_executions = (EXECUTION_PER_WELL, EXECUTION_RUN_BATCH)
        for step, spec in PIPELINE_STEPS.items():
            assert "stage" in spec, f"{step} missing stage"
            assert spec["fanout"] in ("experiment", "per_well_then_merge"), f"{step} bad fanout"
            assert "execution" in spec, f"{step} missing execution"
            assert spec["execution"] in valid_executions, (
                f"{step} has unknown execution {spec['execution']!r}; "
                f"must be one of {valid_executions}"
            )
            assert spec["artifacts"], f"{step} has no artifacts"

    def test_execution_mode_helper_returns_correct_value(self):
        # per-well step: snip_inventory is one job per well
        assert execution_mode("snip_inventory") == EXECUTION_PER_WELL
        # run-batch step: frame_masks loads SAM2 once for all run wells
        assert execution_mode("frame_masks") == EXECUTION_RUN_BATCH

    def test_run_batch_steps_are_per_well_then_merge(self):
        # A batch execution step that doesn't produce per-well shards is incoherent.
        for step, spec in PIPELINE_STEPS.items():
            if spec["execution"] == EXECUTION_RUN_BATCH:
                assert spec["fanout"] == PER_WELL_THEN_MERGE, (
                    f"{step} has execution=run_batch but fanout={spec['fanout']!r}; "
                    f"run_batch steps must fan out per_well_then_merge"
                )

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


class TestOutputTreeDoctrine:
    """Pin the output_tree_doctrine.md regime rules against the live registry."""

    def test_all_stages_are_doctrine_regimes(self):
        for step, spec in PIPELINE_STEPS.items():
            assert spec["stage"] in _ALLOWED_STAGES, (
                f"{step!r} has non-doctrine stage {spec['stage']!r}. "
                f"Allowed: {sorted(_ALLOWED_STAGES)}"
            )

    def test_no_legacy_regime_names_in_product_dir(self):
        for step, spec in PIPELINE_STEPS.items():
            product_dir = spec.get("product_dir")
            assert product_dir not in _FORBIDDEN_STAGE_NAMES, (
                f"{step!r} has legacy product_dir {product_dir!r}. "
                f"Forbidden: {sorted(_FORBIDDEN_STAGE_NAMES)}"
            )

    def test_frame_inventory_lands_under_acquisition_frame_inventory(self):
        p = artifact_path(ROOT, "frame_inventory", "inventory", EXP,
                          path_mode="per_well", well_id=WELL)
        assert str(p).startswith(str(ROOT / "acquisition" / EXP / "frame_inventory"))

    def test_materialize_well_done_lands_under_acquisition_materialized_images(self):
        p = artifact_path(ROOT, "materialize_well", "done", EXP,
                          path_mode="per_well", well_id=WELL)
        assert str(p).startswith(str(ROOT / "acquisition" / EXP / "materialized_images"))

    def test_frame_detections_lands_under_object_extraction(self):
        p = artifact_path(ROOT, "frame_detections", "frame_detections", EXP,
                          path_mode="per_well", well_id=WELL)
        assert str(p).startswith(str(ROOT / "object_extraction" / EXP / "frame_detections"))

    def test_frame_masks_lands_under_object_extraction(self):
        p = artifact_path(ROOT, "frame_masks", "frame_masks", EXP,
                          path_mode="per_well", well_id=WELL)
        assert str(p).startswith(str(ROOT / "object_extraction" / EXP / "frame_masks"))
