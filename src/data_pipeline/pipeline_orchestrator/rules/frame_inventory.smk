"""Frame-inventory product-family rules.

This is the live microscope handoff family. ``materialize_well`` emits one validated per-well
``{well_id}_frame_inventory.csv`` shard; downstream Beat-2 consumers should depend on that shard
and its sentinel. The merged experiment-level frame inventory is an aggregate view over those same
materializer-emitted shards.

Do not split detection/segmentation logic here. This file wires the handoff product only.
"""

FRAME_INVENTORY_STEP = "frame_inventory"
FRAME_INVENTORY_ARTIFACT = "inventory"

# Step 6 — the live per-well materializer + its emitted frame-inventory shard.
MATERIALIZE_WELL_STEP = "materialize_well"


def _materialize_well_done(experiment: str, *, well_id: str):
    # materialize_well owns only the done sentinel (pixel materialization state).
    return rule_artifact(MATERIALIZE_WELL_STEP, "done", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _materialize_well_validated(experiment: str, *, well_id: str):
    # frame_inventory owns the contract path — validated sentinel lives there too.
    return rule_validated(FRAME_INVENTORY_STEP, FRAME_INVENTORY_ARTIFACT, experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _materialize_well_validated_for_run(wc):
    return [_materialize_well_validated(wc.experiment, well_id=well_id) for well_id in _frame_inventory_run_wells(wc)]

def _frame_inventory_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(FRAME_INVENTORY_STEP, FRAME_INVENTORY_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _frame_inventory_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(FRAME_INVENTORY_STEP, FRAME_INVENTORY_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)


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


# The NATIVE per-well producers (materialize_well + validate_frame_inventory_for_well) live in
# rules/materialize_well_native.smk and are included ONLY in native mode by the Snakefile. In dropin
# mode the dropin_handoff.smk producers write the SAME canonical shard + validated sentinel instead.
# The merge rules below are producer-AGNOSTIC: they consume shards + sentinels by PATH, so they work
# unchanged under either producer family. The shared helpers above (_frame_inventory_artifact,
# _materialize_well_validated, _materialize_well_done, _frame_inventory_run_wells, …) are reused by
# both producer files.


rule merge_frame_inventory:
    input:
        per_well_inventory=_frame_inventory_artifacts_for_run,
        per_well_validated=_frame_inventory_validated_for_run,
    output:
        inventory=str(_frame_inventory_artifact(
            "{experiment}",
            path_mode=PATH_MODE_MERGED,
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
            path_mode=PATH_MODE_MERGED,
        )),
    output:
        validated=str(_frame_inventory_validated(
            "{experiment}",
            path_mode=PATH_MODE_MERGED,
        )),
    shell:
        # Merged node = aggregate view over already-strict per-well shards: skip L4 (re-opening every
        # image is wasteful) but run L0–L3 grouped by well. validation_scope=merged allows MANY wells
        # and applies the per-well temporal checks within each well_id group.
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}" \
          --check-sources "false" \
          --validation-scope "merged"
        """
