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

def _frame_detections_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, FRAME_DETECTIONS_STEP, FRAME_DETECTIONS_ARTIFACT, wc.experiment, wells_for_experiment(wc))


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


rule merge_frame_detections:
    """Row-stack per-well frame_detections shards into the experiment-level merged table."""
    input:
        per_well=_frame_detections_artifacts_for_run,
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
