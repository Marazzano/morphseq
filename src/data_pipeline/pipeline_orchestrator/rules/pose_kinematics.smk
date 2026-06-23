"""pose_kinematics feature-product rules.

First computed feature: one row per snip_id with micron-aware geometry, decoded from the
canonical frame_masks RLE. Consumes the validated snip_inventory (the universe), the per-well
frame_masks shard (the RLE source), the per-well frame_inventory shard (pixel calibration), and
the per-well physical_embryo_registry shard (identity verifier). Per-well build -> validate ->
merge, mirroring the snip_processing template.
"""

import importlib.util

_pose_kinematics_paths_spec = importlib.util.spec_from_file_location(
    "_pipeline_orchestrator_paths_pose_kinematics",
    PROJECT_ROOT / "src" / "data_pipeline" / "pipeline_orchestrator" / "orchestration" / "paths.py",
)
_pose_kinematics_paths = importlib.util.module_from_spec(_pose_kinematics_paths_spec)
_pose_kinematics_paths_spec.loader.exec_module(_pose_kinematics_paths)

POSE_KINEMATICS_STEP = "pose_kinematics"


def _pose_kinematics_artifact(experiment, *, path_mode, well_id=None):
    return _pose_kinematics_paths.artifact_path(
        DATA_ROOT, POSE_KINEMATICS_STEP, "pose_kinematics", experiment,
        path_mode=path_mode, well_id=well_id,
    )


def _pose_kinematics_validated(experiment, *, path_mode, well_id=None):
    return _pose_kinematics_paths.validated_path(
        DATA_ROOT, POSE_KINEMATICS_STEP, "pose_kinematics", experiment,
        path_mode=path_mode, well_id=well_id,
    )


def _pose_kinematics_snip_inventory(experiment, *, well_id):
    return _pose_kinematics_paths.artifact_path(
        DATA_ROOT, "snip_inventory", "snip_inventory", experiment,
        path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _pose_kinematics_snip_inventory_validated(experiment, *, well_id):
    return _pose_kinematics_paths.validated_path(
        DATA_ROOT, "snip_inventory", "snip_inventory", experiment,
        path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _pose_kinematics_frame_masks(experiment, *, well_id):
    return _pose_kinematics_paths.artifact_path(
        DATA_ROOT, "frame_masks", "frame_masks", experiment,
        path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _pose_kinematics_frame_inventory(experiment, *, well_id):
    return _pose_kinematics_paths.artifact_path(
        DATA_ROOT, "frame_inventory", "inventory", experiment,
        path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _pose_kinematics_registry(experiment, *, well_id):
    return _pose_kinematics_paths.artifact_path(
        DATA_ROOT, "physical_embryo_registry", "physical_embryo_registry", experiment,
        path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _pose_kinematics_registry_validated(experiment, *, well_id):
    return _pose_kinematics_paths.validated_path(
        DATA_ROOT, "physical_embryo_registry", "physical_embryo_registry", experiment,
        path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _pose_kinematics_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, POSE_KINEMATICS_STEP, "pose_kinematics", wc.experiment, wells_for_experiment(wc),
    )


rule build_pose_kinematics_for_well:
    """Compute the per-well pose_kinematics shard from validated snip_inventory + frame_masks + frame_inventory."""
    input:
        snip_inventory=str(_pose_kinematics_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_pose_kinematics_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_masks=str(_pose_kinematics_frame_masks("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_pose_kinematics_frame_inventory("{experiment}", well_id="{well_id}")),
        frame_inventory_validated=str(_pose_kinematics_paths.validated_path(
            DATA_ROOT, "frame_inventory", "inventory", "{experiment}",
            path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id="{well_id}",
        )),
        physical_embryo_registry=str(_pose_kinematics_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_pose_kinematics_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        pose_kinematics=str(_pose_kinematics_artifact(
            "{experiment}", path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks pose-kinematics \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-masks-csv "{input.frame_masks}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.pose_kinematics}"
        """


rule validate_pose_kinematics_for_well:
    """Validate the per-well pose_kinematics shard (spine + features, registry as verifier) and write .validated."""
    input:
        pose_kinematics=str(_pose_kinematics_artifact(
            "{experiment}", path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_pose_kinematics_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_pose_kinematics_validated(
            "{experiment}", path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-pose-kinematics \
          --input-csv "{input.pose_kinematics}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_pose_kinematics:
    """Row-stack per-well pose_kinematics shards into the experiment-level merged table."""
    input:
        per_well=_pose_kinematics_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_pose_kinematics_validated(wc.experiment, path_mode=_pose_kinematics_paths.PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_pose_kinematics_artifact("{experiment}", path_mode=_pose_kinematics_paths.PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'pose_kinematics', 'pose_kinematics', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
