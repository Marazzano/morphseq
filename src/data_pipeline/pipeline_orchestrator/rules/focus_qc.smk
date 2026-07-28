"""focus_qc quality-control rules.

Interior structural-edge-content heuristic per snip (ghost/structureless embryo detection).
Consumes the validated snip_inventory (universe + snip->mask handoff), the per-well frame_masks
shard (canonical RLE source), the per-well frame_inventory shard (projection pixels via
materialized_image_readers), and the per-well physical_embryo_registry shard (identity verifier).
Per-well build -> validate -> merge, mirroring the mask_quality_qc template.
"""

FOCUS_QC_STEP = "focus_qc"


def _fqc_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(FOCUS_QC_STEP, "focus_qc", experiment, path_mode=path_mode, well_id=well_id)

def _fqc_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(FOCUS_QC_STEP, "focus_qc", experiment, path_mode=path_mode, well_id=well_id)

def _fqc_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fqc_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fqc_frame_masks(experiment, *, well_id):
    return rule_artifact("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fqc_frame_masks_validated(experiment, *, well_id):
    return rule_validated("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fqc_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fqc_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fqc_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, FOCUS_QC_STEP, "focus_qc", wc.experiment, wells_for_experiment(wc))


rule build_focus_qc_for_well:
    """Compute the per-well focus_qc shard from the universe + canonical frame_masks + frame_inventory."""
    input:
        snip_inventory=str(_fqc_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_fqc_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_masks=str(_fqc_frame_masks("{experiment}", well_id="{well_id}")),
        frame_masks_validated=str(_fqc_frame_masks_validated("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_frame_inventory_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
        frame_inventory_validated=str(_frame_inventory_validated("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
        physical_embryo_registry=str(_fqc_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_fqc_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        focus_qc=str(_fqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks focus-qc \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.focus_qc}"
        """


rule validate_focus_qc_for_well:
    """Validate the per-well focus_qc shard (spine + metric + flag, registry as verifier) and write .validated."""
    input:
        focus_qc=str(_fqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_fqc_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_fqc_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-focus-qc \
          --input-csv "{input.focus_qc}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_focus_qc:
    """Row-stack per-well focus_qc shards into the experiment-level merged table."""
    input:
        per_well=_fqc_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_fqc_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_fqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.quality_control.focus_qc.contract import FOCUS_QC_TABLE_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'focus_qc', 'focus_qc', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=FOCUS_QC_TABLE_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
