"""snip_auxiliary_masks product rules.

Per-snip UNet auxiliary masks (via/yolk/focus/bubble/foreground), predicted on the snip crop and
native to snip resolution. Consumes the validated per-well snip_inventory; produces one manifest
row per (snip_id, auxiliary_mask_type) plus the PNG masks beside the shard. Runs AFTER
snip_processing (replaces the retired full-frame auxiliary_masks step). Per-well build -> validate
-> merge, mirroring the feature-product template.
"""

SNIP_AUX_STEP = "snip_auxiliary_masks"


def _sam_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(SNIP_AUX_STEP, "manifest", experiment, path_mode=path_mode, well_id=well_id)

def _sam_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(SNIP_AUX_STEP, "manifest", experiment, path_mode=path_mode, well_id=well_id)

def _sam_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _sam_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _sam_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, SNIP_AUX_STEP, "manifest", wc.experiment, wells_for_experiment(wc),
    )


rule build_snip_auxiliary_masks_for_well:
    """Run per-snip UNet auxiliary-mask inference for one well from the validated snip_inventory."""
    input:
        snip_inventory=str(_sam_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_sam_snip_inventory_validated("{experiment}", well_id="{well_id}")),
    output:
        manifest=str(_sam_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    params:
        output_root=str(DATA_ROOT),
        config_yaml=str(CONFIG_YAML),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-auxiliary-masks \
          --snip-inventory-csv "{input.snip_inventory}" \
          --output-root "{params.output_root}" \
          --output-csv "{output.manifest}" \
          --config-yaml "{params.config_yaml}"
        """


rule validate_snip_auxiliary_masks_for_well:
    """Validate the per-well snip_auxiliary_masks shard (contract + cross-check) and write .validated."""
    input:
        manifest=str(_sam_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        snip_inventory=str(_sam_snip_inventory("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_sam_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-snip-auxiliary-masks \
          --input-csv "{input.manifest}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_snip_auxiliary_masks:
    """Row-stack per-well snip_auxiliary_masks shards into the experiment-level merged manifest."""
    input:
        per_well=_sam_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_sam_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_sam_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'snip_auxiliary_masks', 'manifest', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id', 'auxiliary_mask_type'])
"
        """
