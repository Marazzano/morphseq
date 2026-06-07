"""Frame-inventory product-family rules.

This is the behavior-preserving bridge from the legacy ``frame_contract.csv`` table to the
``frame_inventory`` product family declared in ``orchestration.paths``. The current live pipeline
still builds ``frame_contract.csv`` as the physical frame table; these rules split that table into
per-well inventory shards and merge those shards back to the experiment-level inventory.

Do not split detection/segmentation here. Do not bulk-import stale rule fragments.
"""

import importlib.util

_paths_spec = importlib.util.spec_from_file_location(
    "_pipeline_orchestrator_paths",
    PROJECT_ROOT / "src" / "data_pipeline" / "pipeline_orchestrator" / "orchestration" / "paths.py",
)
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)

FRAME_INVENTORY_STEP = "frame_inventory"
FRAME_INVENTORY_ARTIFACT = "inventory"


def _frame_inventory_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return paths.artifact_path(
        DATA_ROOT,
        FRAME_INVENTORY_STEP,
        FRAME_INVENTORY_ARTIFACT,
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _frame_inventory_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return paths.validated_path(
        DATA_ROOT,
        FRAME_INVENTORY_STEP,
        FRAME_INVENTORY_ARTIFACT,
        experiment,
        path_mode=path_mode,
        well_id=well_id,
    )


def _frame_inventory_run_wells(wc):
    # Legacy bridge: the live checkpoint still writes wells.txt and wells_for_experiment reads it.
    # When discover_wells is moved onto paths.py, this should become run_well_ids_for_experiment().
    return wells_for_experiment(wc)


def _frame_inventory_artifacts_for_run(wc):
    return [
        _frame_inventory_artifact(
            wc.experiment,
            path_mode=paths.PATH_MODE_PER_WELL,
            well_id=well_id,
        )
        for well_id in _frame_inventory_run_wells(wc)
    ]


def _frame_inventory_validated_for_run(wc):
    return [
        _frame_inventory_validated(
            wc.experiment,
            path_mode=paths.PATH_MODE_PER_WELL,
            well_id=well_id,
        )
        for well_id in _frame_inventory_run_wells(wc)
    ]


rule build_frame_inventory_for_well:
    input:
        frame_contract_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_contract.csv",
        frame_contract_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".frame_contract.validated",
    output:
        inventory=str(_frame_inventory_artifact(
            "{experiment}",
            path_mode=paths.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks build-frame-inventory-for-well \
          --frame-contract-csv "{input.frame_contract_csv}" \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --output-csv "{output.inventory}"
        """


rule validate_frame_inventory_for_well:
    input:
        inventory=str(_frame_inventory_artifact(
            "{experiment}",
            path_mode=paths.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    output:
        validated=str(_frame_inventory_validated(
            "{experiment}",
            path_mode=paths.PATH_MODE_PER_WELL,
            well_id="{well_id}",
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_frame_inventory:
    input:
        per_well_inventory=_frame_inventory_artifacts_for_run,
        per_well_validated=_frame_inventory_validated_for_run,
    output:
        inventory=str(_frame_inventory_artifact(
            "{experiment}",
            path_mode=paths.PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks merge-frame-inventory \
          --inputs {input.per_well_inventory} \
          --output-csv "{output.inventory}"
        """


rule validate_frame_inventory:
    input:
        inventory=str(_frame_inventory_artifact(
            "{experiment}",
            path_mode=paths.PATH_MODE_MERGED,
        )),
    output:
        validated=str(_frame_inventory_validated(
            "{experiment}",
            path_mode=paths.PATH_MODE_MERGED,
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-frame-inventory \
          --input-csv "{input.inventory}" \
          --output-flag "{output.validated}"
        """
