"""Frame-inventory product-family rules.

This is the live microscope handoff family. ``materialize_well`` emits one validated per-well
``{well_id}_frame_inventory.csv`` shard; downstream Beat-2 consumers should depend on that shard
and its sentinel. The merged experiment-level frame inventory is an aggregate view over those same
materializer-emitted shards.

Do not split detection/segmentation logic here. This file wires the handoff product only.
"""

import importlib.util

_paths_spec = importlib.util.spec_from_file_location(
    "_pipeline_orchestrator_paths",
    PROJECT_ROOT / "src" / "data_pipeline" / "pipeline_orchestrator" / "orchestration" / "paths.py",
)
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)

FRAME_INVENTORY_STEP = "frame_inventory"
FRAME_INVENTORY_ARTIFACT = "inventory"

# Step 6 — the live per-well materializer + its emitted frame-inventory shard.
MATERIALIZE_WELL_STEP = "materialize_well"


def _materialize_well_inventory(experiment: str, *, well_id: str):
    return paths.artifact_path(
        DATA_ROOT,
        MATERIALIZE_WELL_STEP,
        "inventory",
        experiment,
        path_mode=paths.PATH_MODE_PER_WELL,
        well_id=well_id,
    )


def _materialize_well_done(experiment: str, *, well_id: str):
    return paths.artifact_path(
        DATA_ROOT,
        MATERIALIZE_WELL_STEP,
        "done",
        experiment,
        path_mode=paths.PATH_MODE_PER_WELL,
        well_id=well_id,
    )


def _materialize_well_validated(experiment: str, *, well_id: str):
    return paths.validated_path(
        DATA_ROOT,
        MATERIALIZE_WELL_STEP,
        "inventory",
        experiment,
        path_mode=paths.PATH_MODE_PER_WELL,
        well_id=well_id,
    )


def _materialize_well_validated_for_run(wc):
    return [
        str(_materialize_well_validated(wc.experiment, well_id=well_id))
        for well_id in _frame_inventory_run_wells(wc)
    ]


def _frame_inventory_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return paths.artifact_path(
        DATA_ROOT,
        FRAME_INVENTORY_STEP,
        FRAME_INVENTORY_ARTIFACT,
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _frame_inventory_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return paths.validated_path(
        DATA_ROOT,
        FRAME_INVENTORY_STEP,
        FRAME_INVENTORY_ARTIFACT,
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _frame_inventory_run_wells(wc):
    # The run set comes from well_runner via wells_for_experiment(): discovered ∩ config targets.
    return wells_for_experiment(wc)


def _frame_inventory_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT,
        MATERIALIZE_WELL_STEP,
        "inventory",
        wc.experiment,
        _frame_inventory_run_wells(wc),
    )


def _frame_inventory_validated_for_run(wc):
    return [
        _materialize_well_validated(wc.experiment, well_id=well_id)
        for well_id in _frame_inventory_run_wells(wc)
    ]


rule materialize_well:
    """Step 6 — the LIVE per-well YX1 materializer.

    Fanned over discovered_wells.txt (one well per job). Reads the experiment's acquisition
    inventory + position→well mapping, materializes the configured product set (Step 6: BF /
    projection / focus_stack), writes pixel files into the live built_image_data tree
    (candidate=False), and emits the per-well frame-inventory shard the validator consumes.
    The ND2 source travels inside the acquisition inventory (source_nd2_path) — not a CLI arg.
    """
    input:
        acquisition_inventory_csv=SCOPE_ACQUISITION_INVENTORY_CSV,
        position_well_mapping_csv=POSITION_WELL_MAPPING_CSV,
    output:
        inventory=str(_materialize_well_inventory("{experiment}", well_id="{well_id}")),
        done=str(_materialize_well_done("{experiment}", well_id="{well_id}")),
    params:
        device=lambda wc: str(config.get("image_building", {}).get("device", "cuda")),
        smoke_max=lambda wc: int(
            config.get("image_materialization", {}).get("smoke_max_time_indices", 0)
        ),
    shell:
        # well_index is derived from well_id inside the task (identity parser) — not passed here.
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks materialize-well \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --scope "yx1" \
          --acquisition-inventory-csv "{input.acquisition_inventory_csv}" \
          --position-well-mapping-csv "{input.position_well_mapping_csv}" \
          --built-image-data-dir "{BUILT_IMAGE_DATA_DIR}" \
          --frame-inventory-csv "{output.inventory}" \
          --done-flag "{output.done}" \
          --candidate "false" \
          --device "{params.device}" \
          --smoke-max-time-indices "{params.smoke_max}"
        """


rule validate_frame_inventory_for_well:
    input:
        # Step 6: consume the materializer-emitted shard (NOT the frame_contract.csv adapter).
        inventory=str(_materialize_well_inventory("{experiment}", well_id="{well_id}")),
        done=str(_materialize_well_done("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_materialize_well_validated("{experiment}", well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_frame_inventory:
    input:
        per_well_inventory=_frame_inventory_artifacts_for_run,
        per_well_validated=_frame_inventory_validated_for_run,
    output:
        inventory=str(_frame_inventory_artifact(
            "{experiment}",
            path_mode=paths.PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks merge-frame-inventory \
          --inputs {input.per_well_inventory} \
          --output-csv "{output.inventory}"
        """


rule validate_frame_inventory:
    input:
        inventory=str(_frame_inventory_artifact(
            "{experiment}",
            path_mode=paths.PATH_MODE_MERGED,
        )),
    output:
        validated=str(_frame_inventory_validated(
            "{experiment}",
            path_mode=paths.PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}"
        """
