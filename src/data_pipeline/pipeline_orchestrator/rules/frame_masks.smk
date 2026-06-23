"""Frame-masks product-family rules.

Consumes the per-well frame_inventory shard and the per-well frame_detections shard,
runs SAM2 video segmentation, and emits the per-well frame_masks shard. The merged
experiment-level table is an aggregate view for audit/reporting.

SAM2 path contract — see models/sam2.py for the full explanation. The short version:
  - sam2_models_root must point to the directory that *contains* the sam2/ package subdir
    (typically MODELS_DIR / "sam2", which is a symlink to the checkout).
  - sam2_config is a *relative* path like "configs/sam2.1/sam2.1_hiera_s.yaml" — the
    loader resolves it against the package subdir and passes it to Hydra in that form.
  - sam2_checkpoint is a *relative* path like "checkpoints/sam2.1_hiera_small.pt" —
    resolved against sam2_models_root by the loader.
  Never expand these to absolute paths before passing; the loader's chdir+Hydra dance
  requires the relative form.
"""

FRAME_MASKS_STEP = "frame_masks"


def _frame_masks_artifact(experiment: str, artifact: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(FRAME_MASKS_STEP, artifact, experiment, path_mode=path_mode, well_id=well_id)

def _frame_masks_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(FRAME_MASKS_STEP, "frame_masks", experiment, path_mode=path_mode, well_id=well_id)

def _frame_masks_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, FRAME_MASKS_STEP, "frame_masks", wc.experiment, wells_for_experiment(wc))

def _frame_masks_validated_for_run(wc):
    return [_frame_masks_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w) for w in wells_for_experiment(wc)]


rule frame_masks_per_well:
    """Run SAM2 segmentation over a validated per-well frame_inventory + frame_detections.

    One job per well. Reads the frame_inventory and frame_detections shards, builds
    RGB JPEG frames in a temp dir, seeds SAM2 from kept detections, propagates forward,
    and emits the per-well frame_masks shard + prompt_seeds audit sidecar.
    """
    input:
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_detections=str(_frame_detections_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
        prompt_seeds=str(_frame_masks_artifact(
            "{experiment}", "prompt_seeds",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    params:
        device=lambda wc: str(config.get("frame_masks", {}).get("device", DEVICE)),
        sam2_models_root=lambda wc: str(MODELS_DIR / "sam2"),
        sam2_config=lambda wc: str(config.get("frame_masks", {}).get(
            "sam2_config", "configs/sam2.1/sam2.1_hiera_s.yaml"
        )),
        sam2_checkpoint=lambda wc: str(config.get("frame_masks", {}).get(
            "sam2_checkpoint", "checkpoints/sam2.1_hiera_small.pt"
        )),
        sam2_model_id=lambda wc: str(config.get("frame_masks", {}).get(
            "sam2_model_id", "sam2.1_hiera_s"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks frame-masks \
          --frame-inventory-csv "{input.frame_inventory}" \
          --frame-detections-csv "{input.frame_detections}" \
          --output-csv "{output.frame_masks}" \
          --prompt-seeds-csv "{output.prompt_seeds}" \
          --sam2-models-root "{params.sam2_models_root}" \
          --sam2-config "{params.sam2_config}" \
          --sam2-checkpoint "{params.sam2_checkpoint}" \
          --sam2-model-id "{params.sam2_model_id}" \
          --device "{params.device}"
        """


rule validate_frame_masks_for_well:
    """Validate the per-well frame_masks shard against its frame_inventory; write the .validated sentinel.

    The merge (collect_well_shard_paths) only picks up shards that carry a .validated sentinel, so
    this rule is what makes a built frame_masks shard eligible for the merged table — and what lets
    downstream consumers (e.g. physical_embryo_registry, snip_processing) depend on a *validated*
    frame_masks shard rather than a raw one.
    """
    input:
        frame_masks=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated=str(_frame_masks_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-masks \
          --input-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_frame_masks:
    """Row-stack per-well frame_masks shards into the experiment-level merged table."""
    input:
        per_well=_frame_masks_artifacts_for_run,
        per_well_validated=_frame_masks_validated_for_run,
    output:
        merged=str(_frame_masks_artifact(
            "{experiment}", "frame_masks",
            path_mode=PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.segmentation.frame_masks_contract import FRAME_MASKS_REQUIRED_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'frame_masks', 'frame_masks', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=list(FRAME_MASKS_REQUIRED_COLUMNS), sort_columns=['experiment_id', 'well_id', 'time_index'])
"
        """
