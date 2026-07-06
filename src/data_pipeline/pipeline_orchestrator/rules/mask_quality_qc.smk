"""mask_quality_qc quality-control rules.

Structural mask-trustworthiness flags per snip (edge / discontinuous / overlapping), decoded
from the CANONICAL frame_masks RLE via the shared decoder (migrated off the raw SAM2 path).
Consumes the validated snip_inventory (universe + snip->mask handoff), the per-well frame_masks
shard (the RLE source), and the per-well physical_embryo_registry shard (identity verifier).
Per-well build -> validate -> merge, mirroring the feature-product template.
"""

MASK_QUALITY_QC_STEP = "mask_quality_qc"


def _mqqc_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(MASK_QUALITY_QC_STEP, "mask_quality_qc", experiment, path_mode=path_mode, well_id=well_id)

def _mqqc_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(MASK_QUALITY_QC_STEP, "mask_quality_qc", experiment, path_mode=path_mode, well_id=well_id)

def _mqqc_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mqqc_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mqqc_frame_masks(experiment, *, well_id):
    return rule_artifact("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mqqc_frame_masks_validated(experiment, *, well_id):
    return rule_validated("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mqqc_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mqqc_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mqqc_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, MASK_QUALITY_QC_STEP, "mask_quality_qc", wc.experiment, wells_for_experiment(wc))


rule build_mask_quality_qc_for_well:
    """Compute the per-well mask_quality_qc shard from the universe + canonical frame_masks."""
    input:
        snip_inventory=str(_mqqc_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_mqqc_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_masks=str(_mqqc_frame_masks("{experiment}", well_id="{well_id}")),
        frame_masks_validated=str(_mqqc_frame_masks_validated("{experiment}", well_id="{well_id}")),
        physical_embryo_registry=str(_mqqc_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_mqqc_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        mask_quality_qc=str(_mqqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks mask-quality-qc \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-masks-csv "{input.frame_masks}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.mask_quality_qc}"
        """


rule validate_mask_quality_qc_for_well:
    """Validate the per-well mask_quality_qc shard (spine + flags, registry as verifier) and write .validated."""
    input:
        mask_quality_qc=str(_mqqc_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_mqqc_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_mqqc_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-mask-quality-qc \
          --input-csv "{input.mask_quality_qc}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_mask_quality_qc:
    """Row-stack per-well mask_quality_qc shards into the experiment-level merged table."""
    input:
        per_well=_mqqc_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_mqqc_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_mqqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'mask_quality_qc', 'mask_quality_qc', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """


rule mask_quality_qc_report:
    """TERMINAL: per-flag histograms and quartile galleries with canonical snip-mask overlays."""
    input:
        mask_quality_qc=str(_mqqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
        snip_inventory=str(rule_artifact("snip_inventory", "snip_inventory", "{experiment}", path_mode=PATH_MODE_MERGED)),
        frame_masks=str(rule_artifact("frame_masks", "frame_masks", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        edge_flag_histogram_png=str(
            rule_artifact("mask_quality_qc_report", "edge_flag_histogram_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)
        ),
        edge_flag_gallery_png=str(
            rule_artifact("mask_quality_qc_report", "edge_flag_gallery_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)
        ),
        discontinuous_mask_flag_histogram_png=str(
            rule_artifact(
                "mask_quality_qc_report",
                "discontinuous_mask_flag_histogram_png",
                "{experiment}",
                path_mode=PATH_MODE_EXPERIMENT,
            )
        ),
        discontinuous_mask_flag_gallery_png=str(
            rule_artifact(
                "mask_quality_qc_report",
                "discontinuous_mask_flag_gallery_png",
                "{experiment}",
                path_mode=PATH_MODE_EXPERIMENT,
            )
        ),
        overlapping_mask_flag_histogram_png=str(
            rule_artifact(
                "mask_quality_qc_report",
                "overlapping_mask_flag_histogram_png",
                "{experiment}",
                path_mode=PATH_MODE_EXPERIMENT,
            )
        ),
        overlapping_mask_flag_gallery_png=str(
            rule_artifact(
                "mask_quality_qc_report",
                "overlapping_mask_flag_gallery_png",
                "{experiment}",
                path_mode=PATH_MODE_EXPERIMENT,
            )
        ),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks mask-quality-qc-report \
          --mask-quality-qc-csv "{input.mask_quality_qc}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-masks-csv "{input.frame_masks}" \
          --output-root "{DATA_ROOT}" \
          --output-edge-flag-histogram-png "{output.edge_flag_histogram_png}" \
          --output-edge-flag-gallery-png "{output.edge_flag_gallery_png}" \
          --output-discontinuous-mask-flag-histogram-png "{output.discontinuous_mask_flag_histogram_png}" \
          --output-discontinuous-mask-flag-gallery-png "{output.discontinuous_mask_flag_gallery_png}" \
          --output-overlapping-mask-flag-histogram-png "{output.overlapping_mask_flag_histogram_png}" \
          --output-overlapping-mask-flag-gallery-png "{output.overlapping_mask_flag_gallery_png}"
        """
