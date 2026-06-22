"""Frame-detections product-family rules.

Consumes the per-well frame_inventory shard (the microscope-agnostic handoff seam) and runs
GroundingDINO to produce the per-well frame_detections shard. The merged experiment-level table
is an aggregate view for audit/reporting; downstream per-well consumers (frame_masks — not yet
wired) depend on the per-well shard directly.
"""

import importlib.util

_paths_spec = importlib.util.spec_from_file_location(
    "_pipeline_orchestrator_paths",
    PROJECT_ROOT / "src" / "data_pipeline" / "pipeline_orchestrator" / "orchestration" / "paths.py",
)
_paths_mod = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(_paths_mod)

FRAME_DETECTIONS_STEP = "frame_detections"
FRAME_DETECTIONS_ARTIFACT = "frame_detections"


def _frame_detections_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return _paths_mod.artifact_path(
        DATA_ROOT,
        FRAME_DETECTIONS_STEP,
        FRAME_DETECTIONS_ARTIFACT,
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _frame_detections_for_run(wc):
    return [
        str(_frame_detections_artifact(wc.experiment, path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id=w))
        for w in wells_for_experiment(wc)
    ]


rule frame_detections_per_well:
    """Run GroundingDINO detection over a validated per-well frame_inventory shard.

    One job per well. Reads the frame_inventory shard, runs the detection router, and emits the
    per-well frame_detections shard. The merged experiment table is produced by
    merge_frame_detections.
    """
    input:
        frame_inventory=str(_frame_inventory_artifact(
            "{experiment}", path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        frame_inventory_validated=str(_frame_inventory_validated(
            "{experiment}", path_mode=_paths_mod.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        detections=str(_frame_detections_artifact(
            "{experiment}",
            path_mode=_paths_mod.PATH_MODE_PER_WELL,
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
        per_well=_frame_detections_for_run,
    output:
        merged=str(_frame_detections_artifact(
            "{experiment}",
            path_mode=_paths_mod.PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -c "
import pandas as pd, sys
frames = [pd.read_csv(p) for p in {input.per_well!r}]
pd.concat(frames, ignore_index=True).to_csv('{output.merged}', index=False)
"
        """
