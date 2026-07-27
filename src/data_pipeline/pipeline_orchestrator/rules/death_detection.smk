"""death_detection quality-control rules — ONE build, TWO output grains.

A single build rule emits both per-well shards (the per-snip death_detection_qc table and the
per-physical_embryo death_event table) because they share one compute pass over fraction_alive +
frame timing. Each output then validates at its OWN grain and merges separately. Consumes
fraction_alive (snip spine + time_index + fraction_alive), frame_inventory (elapsed_time_s for the
hours-based lead-time), stage_predictions (death_event only), and the snip_inventory universe.
"""

DEATH_DETECTION_QC_STEP = "death_detection_qc"
DEATH_EVENT_STEP = "death_event"


def _dd_qc_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(DEATH_DETECTION_QC_STEP, "death_detection_qc", experiment, path_mode=path_mode, well_id=well_id)

def _dd_qc_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(DEATH_DETECTION_QC_STEP, "death_detection_qc", experiment, path_mode=path_mode, well_id=well_id)

def _dd_event_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(DEATH_EVENT_STEP, "death_event", experiment, path_mode=path_mode, well_id=well_id)

def _dd_event_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(DEATH_EVENT_STEP, "death_event", experiment, path_mode=path_mode, well_id=well_id)

def _dd_fraction_alive(experiment, *, well_id):
    return rule_artifact("fraction_alive", "fraction_alive", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_fraction_alive_validated(experiment, *, well_id):
    return rule_validated("fraction_alive", "fraction_alive", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_frame_inventory(experiment, *, well_id):
    return rule_artifact("frame_inventory", "inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_stage_predictions(experiment, *, well_id):
    return rule_artifact("stage_predictions", "stage_predictions", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_stage_predictions_validated(experiment, *, well_id):
    return rule_validated("stage_predictions", "stage_predictions", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _dd_qc_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, DEATH_DETECTION_QC_STEP, "death_detection_qc", wc.experiment, wells_for_experiment(wc))

def _dd_event_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, DEATH_EVENT_STEP, "death_event", wc.experiment, wells_for_experiment(wc))


rule build_death_detection_for_well:
    """Compute BOTH per-well death outputs (qc flags + death_event) in one pass."""
    input:
        fraction_alive=str(_dd_fraction_alive("{experiment}", well_id="{well_id}")),
        fraction_alive_validated=str(_dd_fraction_alive_validated("{experiment}", well_id="{well_id}")),
        frame_inventory=str(_dd_frame_inventory("{experiment}", well_id="{well_id}")),
        stage_predictions=str(_dd_stage_predictions("{experiment}", well_id="{well_id}")),
        stage_predictions_validated=str(_dd_stage_predictions_validated("{experiment}", well_id="{well_id}")),
        snip_inventory=str(_dd_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_dd_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        physical_embryo_registry=str(_dd_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_dd_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        qc=str(_dd_qc_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
        death_event=str(_dd_event_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks death-detection \
          --fraction-alive-csv "{input.fraction_alive}" \
          --frame-inventory-csv "{input.frame_inventory}" \
          --stage-predictions-csv "{input.stage_predictions}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-qc-csv "{output.qc}" \
          --output-death-event-csv "{output.death_event}"
        """


rule validate_death_detection_qc_for_well:
    """Validate the per-well death_detection_qc shard (snip grain + flags) and write .validated."""
    input:
        qc=str(_dd_qc_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
        physical_embryo_registry=str(_dd_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_dd_qc_validated("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-death-detection-qc \
          --input-csv "{input.qc}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule validate_death_event_for_well:
    """Validate the per-well death_event shard (physical-embryo grain, no embryo_id) and write .validated."""
    input:
        death_event=str(_dd_event_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
        physical_embryo_registry=str(_dd_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_dd_event_validated("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-death-event \
          --input-csv "{input.death_event}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_death_detection_qc:
    """Row-stack per-well death_detection_qc shards into the experiment-level merged table."""
    input:
        per_well=_dd_qc_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_dd_qc_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_dd_qc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.quality_control.death_detection.contract import DEATH_DETECTION_QC_TABLE_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'death_detection_qc', 'death_detection_qc', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=DEATH_DETECTION_QC_TABLE_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """


rule merge_death_event:
    """Row-stack per-well death_event shards into the experiment-level merged table."""
    input:
        per_well=_dd_event_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_dd_event_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_dd_event_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.quality_control.death_detection.contract import DEATH_EVENT_TABLE_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'death_event', 'death_event', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=DEATH_EVENT_TABLE_COLUMNS, sort_columns=['experiment_id', 'well_id', 'physical_embryo_id'])
"
        """


rule death_detection_report:
    """TERMINAL: mortality report (survival curve + curtain + death-time histogram). Consumed by
    nothing — only the `reports` aggregate target requests this. See viz/report_world.md."""
    input:
        qc=str(_dd_qc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
        fraction_alive=str(rule_artifact("fraction_alive", "fraction_alive", "{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        experiment_png=str(rule_artifact("death_detection_report", "experiment_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        curtain_png=str(rule_artifact("death_detection_report", "curtain_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        death_time_png=str(rule_artifact("death_detection_report", "death_time_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
        well_survival_png=str(rule_artifact("death_detection_report", "well_survival_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks death-detection-report \
          --death-detection-qc-csv "{input.qc}" \
          --fraction-alive-csv "{input.fraction_alive}" \
          --output-experiment-png "{output.experiment_png}" \
          --output-curtain-png "{output.curtain_png}" \
          --output-death-time-png "{output.death_time_png}" \
          --output-well-survival-png "{output.well_survival_png}"
        """
