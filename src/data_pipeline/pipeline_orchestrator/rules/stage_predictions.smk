"""stage_predictions feature-product rules.

Kimmel1995 developmental stage (hpf) per snip from plate_metadata (start_age_hpf + temperature)
and frame timing. No mask reading. Per-well build -> validate -> merge.
"""

import importlib.util

_stage_paths_spec = importlib.util.spec_from_file_location(
    "_pipeline_orchestrator_paths_stage_predictions",
    PROJECT_ROOT / "src" / "data_pipeline" / "pipeline_orchestrator" / "orchestration" / "paths.py",
)
_stage_paths = importlib.util.module_from_spec(_stage_paths_spec)
_stage_paths_spec.loader.exec_module(_stage_paths)

STAGE_PREDICTIONS_STEP = "stage_predictions"


def _stage_artifact(experiment, *, path_mode, well_id=None):
    return _stage_paths.artifact_path(
        DATA_ROOT, STAGE_PREDICTIONS_STEP, "stage_predictions", experiment,
        path_mode=path_mode, well_id=well_id,
    )


def _stage_validated(experiment, *, path_mode, well_id=None):
    return _stage_paths.validated_path(
        DATA_ROOT, STAGE_PREDICTIONS_STEP, "stage_predictions", experiment,
        path_mode=path_mode, well_id=well_id,
    )


def _stage_snip_inventory(experiment, *, well_id):
    return _stage_paths.artifact_path(
        DATA_ROOT, "snip_inventory", "snip_inventory", experiment,
        path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _stage_snip_inventory_validated(experiment, *, well_id):
    return _stage_paths.validated_path(
        DATA_ROOT, "snip_inventory", "snip_inventory", experiment,
        path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _stage_frame_inventory(experiment, *, well_id):
    return _stage_paths.artifact_path(
        DATA_ROOT, "frame_inventory", "inventory", experiment,
        path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _stage_registry(experiment, *, well_id):
    return _stage_paths.artifact_path(
        DATA_ROOT, "physical_embryo_registry", "physical_embryo_registry", experiment,
        path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _stage_registry_validated(experiment, *, well_id):
    return _stage_paths.validated_path(
        DATA_ROOT, "physical_embryo_registry", "physical_embryo_registry", experiment,
        path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id=well_id,
    )


def _stage_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, STAGE_PREDICTIONS_STEP, "stage_predictions", wc.experiment, wells_for_experiment(wc),
    )


rule build_stage_predictions_for_well:
    """Compute the per-well stage_predictions shard from snip_inventory + frame_inventory + plate_metadata."""
    input:
        snip_inventory=str(_stage_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_stage_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_stage_frame_inventory("{experiment}", well_id="{well_id}")),
        plate_metadata=PLATE_METADATA_CSV,
        physical_embryo_registry=str(_stage_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_stage_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        stage_predictions=str(_stage_artifact(
            "{experiment}", path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks stage-predictions \
          --snip-inventory-csv "{input.snip_inventory}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --plate-metadata-csv "{input.plate_metadata}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.stage_predictions}"
        """


rule validate_stage_predictions_for_well:
    """Validate the per-well stage_predictions shard (spine + features, registry verifier)."""
    input:
        stage_predictions=str(_stage_artifact(
            "{experiment}", path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_stage_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_stage_validated(
            "{experiment}", path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-stage-predictions \
          --input-csv "{input.stage_predictions}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_stage_predictions:
    """Row-stack per-well stage_predictions shards into the experiment-level merged table."""
    input:
        per_well=_stage_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_stage_validated(wc.experiment, path_mode=_stage_paths.PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_stage_artifact("{experiment}", path_mode=_stage_paths.PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'stage_predictions', 'stage_predictions', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
