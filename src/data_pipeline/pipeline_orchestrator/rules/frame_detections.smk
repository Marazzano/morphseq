"""Frame-detections product-family rules.

Consumes the per-well frame_inventory shard (the microscope-agnostic handoff seam) and runs
GroundingDINO to produce the per-well frame_detections shard. The merged experiment-level table
is an aggregate view for audit/reporting; downstream per-well consumers (frame_masks — not yet
wired) depend on the per-well shard directly.
"""

FRAME_DETECTIONS_STEP = "frame_detections"
FRAME_DETECTIONS_ARTIFACT = "frame_detections"


def _frame_detections_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(FRAME_DETECTIONS_STEP, FRAME_DETECTIONS_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _frame_detections_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(FRAME_DETECTIONS_STEP, FRAME_DETECTIONS_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _frame_detections_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, FRAME_DETECTIONS_STEP, FRAME_DETECTIONS_ARTIFACT, wc.experiment, wells_for_experiment(wc))

def _frame_detections_validated_for_run(wc):
    return [_frame_detections_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w) for w in wells_for_experiment(wc)]


rule frame_detections_per_well:
    """Run GroundingDINO detection over a validated per-well frame_inventory shard.

    One job per well. Reads the frame_inventory shard, runs the detection router, and emits the
    per-well frame_detections shard. The merged experiment table is produced by
    merge_frame_detections.
    """
    input:
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        detections=str(_frame_detections_artifact(
            "{experiment}",
            path_mode=PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    params:
        device=lambda wc: str(config.get("frame_detections", {}).get("device", "cuda")),
        gdino_repo=lambda wc: str(MODELS_DIR / "GroundingDINO"),
        gdino_config=lambda wc: str(MODELS_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
        gdino_weights=lambda wc: str(MODELS_DIR / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"),
    # gpu=1: this process itself loads GroundingDINO onto the GPU and holds that memory for the
    # job's duration. Caps concurrent GPU jobs to 1 (pass --resources gpu=1 at the CLI to enforce
    # it; see rules/frame_masks.smk for the fuller note). If a future model-server design puts the
    # model in a resident process instead, do NOT copy this onto the per-well client rule — the
    # client no longer holds GPU memory itself, and double-claiming the slot deadlocks scheduling.
    resources:
        gpu=1,
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks frame-detections \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-csv "{output.detections}" \
          --gdino-repo-dir "{params.gdino_repo}" \
          --gdino-config "{params.gdino_config}" \
          --gdino-weights "{params.gdino_weights}" \
          --device "{params.device}"
        """


rule validate_frame_detections_for_well:
    """Validate the per-well frame_detections shard against its frame_inventory; write the .validated sentinel.

    Symmetric with validate_frame_masks_for_well. The merge (collect_well_shard_paths) only picks up
    shards that carry a .validated sentinel, so this rule is what makes a built frame_detections shard
    eligible for the merged table. (frame_masks_per_well consumes the per-well detection shard
    directly, so it does not require this sentinel — only the merge does.)
    """
    input:
        detections=str(_frame_detections_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}",
        )),
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated=str(_frame_detections_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-detections \
          --input-csv "{input.detections}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_frame_detections:
    """Row-stack per-well frame_detections shards into the experiment-level merged table."""
    input:
        per_well=_frame_detections_artifacts_for_run,
        # Wait on each shard's .validated sentinel too (written by validate_frame_detections_for_well),
        # matching the other merge rules — collect_well_shard_paths only collects validated shards.
        per_well_validated=_frame_detections_validated_for_run,
    output:
        merged=str(_frame_detections_artifact(
            "{experiment}",
            path_mode=PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.object_extraction.detection.frame_detections_contract import REQUIRED_FRAME_DETECTIONS_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'frame_detections', 'frame_detections', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=list(REQUIRED_FRAME_DETECTIONS_COLUMNS), sort_columns=['experiment_id', 'well_id', 'time_index'])
"
        """
