"""Snip-processing product-family rules.

Consumes the per-well frame_masks shard and the per-well frame_inventory shard,
extracts per-embryo crops, mints snip_inventory rows, and emits the per-well
snip_inventory shard + pixel files under the same per-well directory.

Yolk masks are optional: when absent the rotation step falls back to a
mass-distribution heuristic and extraction uses a zero yolk mask.
"""

from data_pipeline.object_extraction.snip_processing.snip_frame_shape import (
    resolve_snip_frame_shape as _resolve_snip_frame_shape,
)
from data_pipeline.object_extraction.snip_processing.defaults import (
    DEFAULT_BLEND_RADIUS_UM,
    DEFAULT_TARGET_PIXEL_SIZE_UM,
)

SNIP_INVENTORY_STEP = "snip_inventory"

# Upstream identity source: physical_embryo_registry owns the track_id -> physical_embryo_id
# resolution. snip_processing JOINS the PER-WELL shard (not the merged table) — the crop loop is
# per-well and joins on (well_id, track_id), so it depends only on this well's registry; depending
# on the merged table would force every well to finish before any well could crop.
PHYSICAL_EMBRYO_REGISTRY_STEP = "physical_embryo_registry"


def _physical_embryo_registry_per_well(experiment: str, *, well_id: str):
    return rule_artifact(PHYSICAL_EMBRYO_REGISTRY_STEP, "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _physical_embryo_registry_per_well_validated(experiment: str, *, well_id: str):
    return rule_validated(PHYSICAL_EMBRYO_REGISTRY_STEP, "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _snip_inventory_artifact(experiment: str, artifact: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(SNIP_INVENTORY_STEP, artifact, experiment, path_mode=path_mode, well_id=well_id)

def _snip_inventory_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(SNIP_INVENTORY_STEP, "snip_inventory", experiment, path_mode=path_mode, well_id=well_id)

def _snip_inventory_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, SNIP_INVENTORY_STEP, "snip_inventory", wc.experiment, wells_for_experiment(wc))

def _snip_inventory_validated_for_run(wc):
    return [_snip_inventory_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w) for w in wells_for_experiment(wc)]

def _snip_inventory_snips_dir(experiment: str, well_id: str) -> str:
    """Per-well pixel directory: sits beside the shard CSV under per_well/{well_id}/."""
    return rule_step_dir(SNIP_INVENTORY_STEP, experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id) + "/snips"


rule snip_processing_per_well:
    """Extract per-embryo crops from validated frame_masks for one well.

    Reads frame_masks + frame_inventory shards, JOINS physical_embryo_id from
    the per-well physical_embryo_registry shard (identity is no longer minted
    here), builds embryo_id / snip_id via shared/identifiers, runs the
    extraction/rotation/augmentation stack, and emits the per-well
    snip_inventory shard + pixel files. Yolk masks are optional; rotation
    degrades gracefully without them.
    """
    input:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
        frame_masks_validated=str(_frame_masks_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_physical_embryo_registry_per_well(
            "{experiment}", well_id="{well_id}"
        )),
        physical_embryo_registry_validated=str(_physical_embryo_registry_per_well_validated(
            "{experiment}", well_id="{well_id}"
        )),
    output:
        snip_inventory=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    params:
        snips_dir=lambda wc: _snip_inventory_snips_dir(wc.experiment, wc.well_id),
        target_pixel_size_um=lambda wc: float(
            config.get("snip_processing", {}).get(
                "target_pixel_size_um", DEFAULT_TARGET_PIXEL_SIZE_UM
            )
        ),
        # Crop output (H, W) comes from the single snip_frame_shape source of truth so the snip
        # image, the saved embryo mask, and the per-snip via mask all share one grid.
        output_height_px=lambda wc: int(_resolve_snip_frame_shape(config)[0]),
        output_width_px=lambda wc: int(_resolve_snip_frame_shape(config)[1]),
        background_noise_scale=lambda wc: float(
            config.get("snip_processing", {}).get("background_noise_scale", 0.1)
        ),
        blend_radius_um=lambda wc: float(
            config.get("snip_processing", {}).get(
                "blend_radius_um", DEFAULT_BLEND_RADIUS_UM
            )
        ),
        # Legacy microscopy/checkpoint behavior stays ON by default. SeaHub's
        # runtime overlay explicitly sets this false for already-normalized
        # inverted 8-bit source images.
        apply_clahe=lambda wc: str(
            config.get("snip_processing", {}).get("apply_clahe", True)
        ).lower(),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-processing \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.snip_inventory}" \
          --snips-dir "{params.snips_dir}" \
          --output-root "{DATA_ROOT}" \
          --target-pixel-size-um "{params.target_pixel_size_um}" \
          --output-height-px "{params.output_height_px}" \
          --output-width-px "{params.output_width_px}" \
          --background-noise-scale "{params.background_noise_scale}" \
          --blend-radius-um "{params.blend_radius_um}" \
          --apply-clahe "{params.apply_clahe}"
        """


rule validate_snip_inventory_for_well:
    """Validate the per-well snip_inventory shard and write a .validated sentinel."""
    input:
        snip_inventory=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    output:
        validated=str(_snip_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-snip-inventory \
          --input-csv "{input.snip_inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_snip_inventory:
    """Row-stack per-well snip_inventory shards into the experiment-level merged table."""
    input:
        per_well=_snip_inventory_artifacts_for_run,
        # Wait on each shard's .validated sentinel too, not just the artifact — otherwise the merge
        # can fire in the window after the shard CSV exists but before validation writes the
        # sentinel, and collect_well_shard_paths (which requires both) sees zero validated shards
        # and raises "no shards to concatenate". Matches the SAFE merge rules (e.g. merge_frame_masks).
        per_well_validated=_snip_inventory_validated_for_run,
    output:
        merged=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import SNIP_INVENTORY_COLUMNS
from pathlib import Path
shards = collect_well_shard_paths('{DATA_ROOT}', 'snip_inventory', 'snip_inventory', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=SNIP_INVENTORY_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
