"""curvature_metrics feature-product rules.

First computed feature: one row per snip_id with micron-aware geometry, decoded from the
canonical frame_masks RLE. Consumes the validated snip_inventory (the universe), the per-well
frame_masks shard (the RLE source), the per-well frame_inventory shard (pixel calibration), and
the per-well physical_embryo_registry shard (identity verifier). Per-well build -> validate ->
merge, mirroring the snip_processing template.

The TERMINAL curvature_metrics_report rule reads the merged curvature_metrics (the numbers) plus
snip_inventory for both the snip image and its snip-space embryo mask (``embryo_mask``; legacy
alias ``embryo_mask_snip_path``); it redraws the spine on the snip from that pre-cropped mask
(no frame_masks / RLE at report time).
"""

CURVATURE_METRICS_STEP = "curvature_metrics"


def _curvature_metrics_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(CURVATURE_METRICS_STEP, "curvature_metrics", experiment, path_mode=path_mode, well_id=well_id)

def _curvature_metrics_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(CURVATURE_METRICS_STEP, "curvature_metrics", experiment, path_mode=path_mode, well_id=well_id)

def _curvature_metrics_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _curvature_metrics_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _curvature_metrics_frame_masks(experiment, *, well_id):
    return rule_artifact("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _curvature_metrics_frame_inventory(experiment, *, well_id):
    return rule_artifact("frame_inventory", "inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _curvature_metrics_frame_inventory_validated(experiment, *, well_id):
    return rule_validated("frame_inventory", "inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _curvature_metrics_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _curvature_metrics_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _curvature_metrics_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, CURVATURE_METRICS_STEP, "curvature_metrics", wc.experiment, wells_for_experiment(wc),
    )


rule build_curvature_metrics_for_well:
    """Compute the per-well curvature_metrics shard from validated snip_inventory + frame_masks + frame_inventory."""
    input:
        snip_inventory=str(_curvature_metrics_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_curvature_metrics_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_masks=str(_curvature_metrics_frame_masks("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_curvature_metrics_frame_inventory("{experiment}", well_id="{well_id}")),
        frame_inventory_validated=_curvature_metrics_frame_inventory_validated("{experiment}", well_id="{well_id}"),
        physical_embryo_registry=str(_curvature_metrics_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_curvature_metrics_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        curvature_metrics=str(_curvature_metrics_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks curvature-metrics \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.curvature_metrics}"
        """


rule validate_curvature_metrics_for_well:
    """Validate the per-well curvature_metrics shard (spine + features, registry as verifier) and write .validated."""
    input:
        curvature_metrics=str(_curvature_metrics_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_curvature_metrics_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_curvature_metrics_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-curvature-metrics \
          --input-csv "{input.curvature_metrics}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_curvature_metrics:
    """Row-stack per-well curvature_metrics shards into the experiment-level merged table."""
    input:
        per_well=_curvature_metrics_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_curvature_metrics_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_curvature_metrics_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'curvature_metrics', 'curvature_metrics', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """


rule curvature_metrics_report:
    """TERMINAL: feature histogram grid + baseline_deviation_normalized quartile gallery with the
    geodesic spine overlaid. Consumed by nothing — only the `reports` aggregate target requests
    this. See viz/report_world.md."""
    input:
        curvature_metrics=str(_curvature_metrics_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
        snip_inventory=str(rule_artifact("snip_inventory", "snip_inventory", "{experiment}", path_mode=PATH_MODE_MERGED)),
        frame_inventory=str(rule_artifact("frame_inventory", "inventory", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        feature_grid_png=str(rule_artifact("curvature_metrics_report", "feature_grid_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        gallery_png=str(rule_artifact("curvature_metrics_report", "gallery_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks curvature-metrics-report \
          --curvature-metrics-csv "{input.curvature_metrics}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-root "{DATA_ROOT}" \
          --output-feature-grid-png "{output.feature_grid_png}" \
          --output-gallery-png "{output.gallery_png}"
        """
