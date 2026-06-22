"""Snip-processing product-family rules.

Consumes the per-well frame_masks shard and the per-well frame_inventory shard,
extracts per-embryo crops, mints snip_inventory rows, and emits the per-well
snip_inventory shard + pixel files under the same per-well directory.

Yolk masks are optional: when absent the rotation step falls back to a
mass-distribution heuristic and extraction uses a zero yolk mask.
"""

import importlib.util

_paths_spec = importlib.util.spec_from_file_location(
    "_pipeline_orchestrator_paths",
    PROJECT_ROOT / "src" / "data_pipeline" / "pipeline_orchestrator" / "orchestration" / "paths.py",
)
_paths_mod = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(_paths_mod)

SNIP_INVENTORY_STEP = "snip_inventory"


def _snip_inventory_artifact(experiment: str, artifact: str, *, path_mode: str, well_id: str | None = None):
    return _paths_mod.artifact_path(
        DATA_ROOT,
        SNIP_INVENTORY_STEP,
        artifact,
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _snip_inventory_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return _paths_mod.validated_path(
        DATA_ROOT,
        SNIP_INVENTORY_STEP,
        "snip_inventory",
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _snip_inventory_for_run(wc):
    return [
        str(_snip_inventory_artifact(
            wc.experiment, "snip_inventory",
            path_mode=_paths_mod.PATH_MODE_PER_WELL,
            well_id=w,
        ))
        for w in wells_for_experiment(wc)
    ]


def _snip_inventory_snips_dir(experiment: str, well_id: str) -> str:
    """Per-well pixel directory: sits beside the shard CSV under per_well/{well_id}/."""
    shard_dir = _paths_mod.step_dir(
        DATA_ROOT, SNIP_INVENTORY_STEP, experiment,
        path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id=well_id,
    )
    return str(shard_dir / "snips")


rule snip_processing_per_well:
    """Extract per-embryo crops from validated frame_masks for one well.

    Reads frame_masks + frame_inventory shards, mints physical_embryo_id /
    embryo_id / snip_id via shared/identifiers, runs the extraction/rotation/
    augmentation stack, and emits the per-well snip_inventory shard + pixel
    files. Yolk masks are optional; rotation degrades gracefully without them.
    """
    input:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=_paths_mod.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        snip_inventory=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=_paths_mod.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    params:
        snips_dir=lambda wc: _snip_inventory_snips_dir(wc.experiment, wc.well_id),
        target_pixel_size_um=lambda wc: float(
            config.get("snip_processing", {}).get("target_pixel_size_um", 2.17)
        ),
        output_height_px=lambda wc: int(
            config.get("snip_processing", {}).get("output_height_px", 512)
        ),
        output_width_px=lambda wc: int(
            config.get("snip_processing", {}).get("output_width_px", 512)
        ),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-processing \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-csv "{output.snip_inventory}" \
          --snips-dir "{params.snips_dir}" \
          --output-root "{DATA_ROOT}" \
          --target-pixel-size-um "{params.target_pixel_size_um}" \
          --output-height-px "{params.output_height_px}" \
          --output-width-px "{params.output_width_px}"
        """


rule validate_snip_inventory_for_well:
    """Validate the per-well snip_inventory shard and write a .validated sentinel."""
    input:
        snip_inventory=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=_paths_mod.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    output:
        validated=str(_snip_inventory_validated(
            "{experiment}", path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id="{well_id}"
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
        per_well=_snip_inventory_for_run,
    output:
        merged=str(_snip_inventory_artifact(
            "{experiment}", "snip_inventory",
            path_mode=_paths_mod.PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from pathlib import Path
shards = collect_well_shard_paths('{DATA_ROOT}', 'snip_inventory', 'snip_inventory', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
