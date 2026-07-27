"""motion_blur_qc quality-control rules.

Adjacent z-plane mask-pixel NCC per snip. Consumes the validated snip_inventory, the per-well
frame_masks shard, the per-well frame_inventory shard (z-stack pixels via materialized_image_readers),
and the per-well physical_embryo_registry shard. Per-well build -> validate -> merge.
"""

MOTION_BLUR_QC_STEP = "motion_blur_qc"


def _mbqc_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(
        MOTION_BLUR_QC_STEP, "motion_blur_qc", experiment, path_mode=path_mode, well_id=well_id
    )

def _mbqc_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(
        MOTION_BLUR_QC_STEP, "motion_blur_qc", experiment, path_mode=path_mode, well_id=well_id
    )

def _mbqc_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mbqc_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mbqc_frame_masks(experiment, *, well_id):
    return rule_artifact("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mbqc_frame_masks_validated(experiment, *, well_id):
    return rule_validated("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mbqc_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mbqc_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mbqc_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, MOTION_BLUR_QC_STEP, "motion_blur_qc", wc.experiment, wells_for_experiment(wc))


rule build_motion_blur_qc_for_well:
    """Compute per-well motion blur from z planes, or emit not-applicable for single-z frames."""
    input:
        snip_inventory=str(_mbqc_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_mbqc_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_masks=str(_mbqc_frame_masks("{experiment}", well_id="{well_id}")),
        frame_masks_validated=str(_mbqc_frame_masks_validated("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_mbqc_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_mbqc_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        motion_blur_qc=str(_mbqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks motion-blur-qc \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.motion_blur_qc}"
        """


rule validate_motion_blur_qc_for_well:
    """Validate the per-well motion_blur_qc shard and write .validated."""
    input:
        motion_blur_qc=str(_mbqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_mbqc_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_mbqc_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-motion-blur-qc \
          --input-csv "{input.motion_blur_qc}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_motion_blur_qc:
    """Row-stack per-well motion_blur_qc shards into the experiment-level merged table."""
    input:
        per_well=_mbqc_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_mbqc_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_mbqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.quality_control.motion_blur_qc.contract import MOTION_BLUR_QC_TABLE_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'motion_blur_qc', 'motion_blur_qc', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=MOTION_BLUR_QC_TABLE_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """


rule motion_blur_qc_report:
    """TERMINAL: histogram + cutoff-relative gallery for motion_blur_qc."""
    input:
        motion_blur_qc=str(_mbqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
        snip_inventory=str(rule_artifact("snip_inventory", "snip_inventory", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        histogram_png=str(rule_artifact("motion_blur_qc_report", "histogram_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        gallery_png=str(rule_artifact("motion_blur_qc_report", "gallery_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks motion-blur-qc-report \
          --motion-blur-qc-csv "{input.motion_blur_qc}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --output-root "{DATA_ROOT}" \
          --output-histogram-png "{output.histogram_png}" \
          --output-gallery-png "{output.gallery_png}"
        """
