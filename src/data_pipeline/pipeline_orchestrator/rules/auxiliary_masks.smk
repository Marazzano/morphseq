"""Auxiliary-mask product rules.

Consumes the per-well validated frame_inventory shard and runs UNet inference to produce
per-family PNG masks (via/yolk/focus/bubble) plus a per-well manifest CSV. The merged
experiment-level manifest is the aggregate view.

Per-well output layout (under object_extraction/{experiment}/auxiliary_masks/per_well/{well_id}/):
    {well_id}_auxiliary_masks.csv                -- per-well manifest
    {well_id}_auxiliary_masks.csv.validated      -- sentinel (written after manifest validates)
    via/{image_id}_via.png
    yolk/{image_id}_yolk.png
    focus/{image_id}_focus.png
    bubble/{image_id}_bubble.png

Merged output layout (under object_extraction/{experiment}/auxiliary_masks/):
    {experiment_id}_auxiliary_masks.csv          -- experiment-level manifest
    {experiment_id}_auxiliary_masks.csv.validated -- merged sentinel
"""

import importlib.util

_paths_spec = importlib.util.spec_from_file_location(
    "_pipeline_orchestrator_paths",
    PROJECT_ROOT / "src" / "data_pipeline" / "pipeline_orchestrator" / "orchestration" / "paths.py",
)
_paths_mod = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(_paths_mod)

AUXILIARY_MASKS_STEP = "auxiliary_masks"


def _aux_masks_artifact(experiment: str, artifact: str, *, path_mode: str, well_id: str | None = None):
    return _paths_mod.artifact_path(
        DATA_ROOT,
        AUXILIARY_MASKS_STEP,
        artifact,
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _aux_masks_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return _paths_mod.validated_path(
        DATA_ROOT,
        AUXILIARY_MASKS_STEP,
        "manifest",
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _aux_masks_manifests_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT,
        AUXILIARY_MASKS_STEP,
        "manifest",
        wc.experiment,
        wells_for_experiment(wc),
    )


def _aux_masks_validated_for_run(wc):
    return [
        str(_aux_masks_validated(wc.experiment, path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id=w))
        for w in wells_for_experiment(wc)
    ]


rule auxiliary_masks_per_well:
    """Run UNet auxiliary-mask inference for one well.

    Reads the per-well validated frame_inventory shard, filters to the requested well,
    and produces per-family PNG masks + a per-well manifest CSV + sentinel.
    """
    input:
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        manifest=str(_aux_masks_artifact(
            "{experiment}", "manifest",
            path_mode=_paths_mod.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
        sentinel=str(_aux_masks_validated(
            "{experiment}",
            path_mode=_paths_mod.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    params:
        output_root=lambda wc: str(
            _paths_mod.step_dir(
                DATA_ROOT, AUXILIARY_MASKS_STEP, wc.experiment,
                path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id=wc.well_id,
            )
        ),
        model_root=lambda wc: str(MODELS_DIR / "auxiliary_masks"),
        batch_size=lambda wc: int(config.get("auxiliary_masks", {}).get("batch_size", 64)),
        num_workers=lambda wc: int(config.get("auxiliary_masks", {}).get("num_workers", 1)),
    shell:
        """
        {RUN} -m data_pipeline.auxiliary_masks.materialize_auxiliary_masks \
          --frame-contract "{input.frame_inventory}" \
          --output-root "{params.output_root}" \
          --output-manifest-csv "{output.manifest}" \
          --output-sentinel "{output.sentinel}" \
          --model-root "{params.model_root}" \
          --well-id "{wildcards.well_id}" \
          --batch-size "{params.batch_size}" \
          --num-workers "{params.num_workers}"
        """


rule merge_auxiliary_masks:
    """Row-stack per-well auxiliary-mask manifests into the experiment-level table."""
    input:
        per_well=_aux_masks_manifests_for_run,
        per_well_validated=_aux_masks_validated_for_run,
    output:
        merged=str(_aux_masks_artifact(
            "{experiment}", "manifest",
            path_mode=_paths_mod.PATH_MODE_MERGED,
        )),
        sentinel=str(_aux_masks_validated(
            "{experiment}",
            path_mode=_paths_mod.PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -m data_pipeline.auxiliary_masks.merge_auxiliary_masks \
          --inputs {input.per_well} \
          --output-manifest-csv "{output.merged}" \
          --output-sentinel "{output.sentinel}"
        """
