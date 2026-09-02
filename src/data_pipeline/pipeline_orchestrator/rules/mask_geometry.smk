"""mask_geometry feature-product rules.

First computed feature: one row per snip_id with micron-aware geometry, decoded from the
canonical frame_masks RLE. Consumes the validated snip_inventory (the universe), the per-well
frame_masks shard (the RLE source), the per-well frame_inventory shard (pixel calibration), and
the per-well physical_embryo_registry shard (identity verifier). Per-well build -> validate ->
merge, mirroring the snip_processing template.
"""

MASK_GEOMETRY_STEP = "mask_geometry"


def _mg_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(MASK_GEOMETRY_STEP, "mask_geometry", experiment, path_mode=path_mode, well_id=well_id)

def _mg_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(MASK_GEOMETRY_STEP, "mask_geometry", experiment, path_mode=path_mode, well_id=well_id)

def _mg_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "legacy_default_snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mg_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "legacy_default_snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mg_frame_masks(experiment, *, well_id):
    return rule_artifact("frame_masks", "frame_masks", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mg_frame_inventory(experiment, *, well_id):
    return rule_artifact("frame_inventory", "inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mg_frame_inventory_validated(experiment, *, well_id):
    return rule_validated("frame_inventory", "inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mg_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mg_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _mg_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, MASK_GEOMETRY_STEP, "mask_geometry", wc.experiment, wells_for_experiment(wc))


rule build_mask_geometry_for_well:
    """Compute the per-well mask_geometry shard from validated snip_inventory + frame_masks + frame_inventory."""
    input:
        snip_inventory=str(_mg_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_mg_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_masks=str(_mg_frame_masks("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_mg_frame_inventory("{experiment}", well_id="{well_id}")),
        frame_inventory_validated=_mg_frame_inventory_validated("{experiment}", well_id="{well_id}"),
        physical_embryo_registry=str(_mg_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_mg_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        mask_geometry=str(_mg_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks mask-geometry \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.mask_geometry}"
        """


rule validate_mask_geometry_for_well:
    """Validate the per-well mask_geometry shard (spine + features, registry as verifier) and write .validated."""
    input:
        mask_geometry=str(_mg_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_mg_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_mg_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-mask-geometry \
          --input-csv "{input.mask_geometry}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_mask_geometry:
    """Row-stack per-well mask_geometry shards into the experiment-level merged table."""
    input:
        per_well=_mg_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_mg_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_mg_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.feature_extraction.mask_geometry.contract import MASK_GEOMETRY_TABLE_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'mask_geometry', 'mask_geometry', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=MASK_GEOMETRY_TABLE_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """


rule mask_geometry_report:
    """TERMINAL: feature histogram grid + area_um2 quartile gallery. Consumed by nothing — only the
    `reports` aggregate target requests this. See viz/report_world.md."""
    input:
        mask_geometry=str(_mg_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
        snip_inventory=str(rule_artifact("snip_inventory", "snip_inventory", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        geometry_feature_grid_png=str(rule_artifact("mask_geometry_report", "geometry_feature_grid_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        area_um2_quartile_gallery_png=str(rule_artifact("mask_geometry_report", "area_um2_quartile_gallery_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks mask-geometry-report \
          --mask-geometry-csv "{input.mask_geometry}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --output-root "{DATA_ROOT}" \
          --output-geometry-feature-grid-png "{output.geometry_feature_grid_png}" \
          --output-area-um2-quartile-gallery-png "{output.area_um2_quartile_gallery_png}"
        """
