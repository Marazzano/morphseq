"""Frame-inventory product-family rules.

This is the live microscope handoff family. Native product materialization emits validated
product-grain shards; discovery + assembly write the canonical per-well
``{well_id}_frame_inventory.csv`` shard. Drop-in mode writes that same canonical shard directly.
Downstream Beat-2 consumers should depend on the canonical shard and its sentinel.

Do not split detection/segmentation logic here. This file wires the handoff product only.
"""

FRAME_INVENTORY_STEP = "frame_inventory"
FRAME_INVENTORY_ARTIFACT = "inventory"
RESOLVED_PRODUCT_PLANS_STEP = "resolved_product_plans"
RESOLVED_PRODUCT_PLAN_ARTIFACT = "json"
FRAME_INVENTORY_PRODUCTS_STEP = "frame_inventory_products"
FRAME_INVENTORY_PRODUCT_ARTIFACT = "inventory"
DISCOVERED_PRODUCT_SHARDS_STEP = "discovered_product_shards"
DISCOVERED_PRODUCT_SHARDS_ARTIFACT = "csv"

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


def _resolved_product_plan(experiment: str, *, well_id: str, product_key: str):
    return rule_artifact(
        RESOLVED_PRODUCT_PLANS_STEP,
        RESOLVED_PRODUCT_PLAN_ARTIFACT,
        experiment,
        path_mode=PATH_MODE_PER_WELL,
        well_id=well_id,
        format_vars={"product_key": product_key},
    )


def _frame_inventory_product_artifact(experiment: str, *, well_id: str, product_key: str):
    return rule_artifact(
        FRAME_INVENTORY_PRODUCTS_STEP,
        FRAME_INVENTORY_PRODUCT_ARTIFACT,
        experiment,
        path_mode=PATH_MODE_PER_WELL,
        well_id=well_id,
        format_vars={"product_key": product_key},
    )


def _frame_inventory_product_validated(experiment: str, *, well_id: str, product_key: str):
    return rule_validated(
        FRAME_INVENTORY_PRODUCTS_STEP,
        FRAME_INVENTORY_PRODUCT_ARTIFACT,
        experiment,
        path_mode=PATH_MODE_PER_WELL,
        well_id=well_id,
        format_vars={"product_key": product_key},
    )


def _discovered_product_shards_artifact(experiment: str, *, well_id: str):
    return rule_artifact(
        DISCOVERED_PRODUCT_SHARDS_STEP,
        DISCOVERED_PRODUCT_SHARDS_ARTIFACT,
        experiment,
        path_mode=PATH_MODE_PER_WELL,
        well_id=well_id,
    )


def _frame_inventory_run_wells(wc):
    # The run set comes from well_runner via wells_for_experiment(): discovered ∩ config targets.
    return wells_for_experiment(wc)


def _frame_inventory_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT,
        FRAME_INVENTORY_STEP,
        FRAME_INVENTORY_ARTIFACT,
        wc.experiment,
        _frame_inventory_run_wells(wc),
    )


def _frame_inventory_validated_for_run(wc):
    return [
        _materialize_well_validated(wc.experiment, well_id=well_id)
        for well_id in _frame_inventory_run_wells(wc)
    ]


# The NATIVE per-well producers (product materialization -> discovery -> assembly ->
# validate_frame_inventory_for_well) live in rules/materialize_well_native.smk and are included ONLY
# in native mode by the Snakefile. In dropin mode the dropin_handoff.smk producers write the SAME
# canonical shard + validated sentinel instead. The merge rules below are producer-AGNOSTIC: they
# consume shards + sentinels by PATH, so they work unchanged under either producer family.


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
