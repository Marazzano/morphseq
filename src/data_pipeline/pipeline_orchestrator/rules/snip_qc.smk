"""snip_qc quality-control rules — the final per-snip verdict.

Builds use_snip + qc_fail_reasons by ORing the MVP exclusion flags (death_detection_qc,
surface_area_qc, mask_quality_qc). The build verb loads those source shards through the path
registry (it takes output_root + experiment + well_id, like consolidated_features), so the rule
only declares the upstream shards + their validated sentinels as DAG dependencies. Per-well
build -> validate -> merge.
"""

SNIP_QC_STEP = "snip_qc"
# MVP exclusion flag sources (must match snip_qc/entrypoint._MVP_FLAG_SOURCES).
_SNIP_QC_SOURCE_STEPS = ["death_detection_qc", "surface_area_qc", "mask_quality_qc"]


def _snipqc_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(SNIP_QC_STEP, "verdict", experiment, path_mode=path_mode, well_id=well_id)

def _snipqc_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(SNIP_QC_STEP, "verdict", experiment, path_mode=path_mode, well_id=well_id)

def _snipqc_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _snipqc_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _snipqc_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _snipqc_source_shards(experiment, well_id):
    """The per-well source flag shards + validated sentinels the verdict depends on."""
    deps = []
    for step in _SNIP_QC_SOURCE_STEPS:
        deps.append(rule_artifact(step, step, experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id))
        deps.append(rule_validated(step, step, experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id))
    return deps

def _snipqc_artifacts_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, SNIP_QC_STEP, "verdict", wc.experiment, wells_for_experiment(wc))


rule build_snip_qc_for_well:
    """Build the per-well snip_qc verdict by ORing the MVP exclusion flags (loads sources via registry)."""
    input:
        source_shards=lambda wc: _snipqc_source_shards(wc.experiment, wc.well_id),
        snip_inventory=str(_snipqc_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_snipqc_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        physical_embryo_registry=str(_snipqc_registry("{experiment}", well_id="{well_id}")),
    output:
        verdict=str(_snipqc_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-qc \
          --output-root "{DATA_ROOT}" \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.verdict}"
        """


rule validate_snip_qc_for_well:
    """Validate the per-well snip_qc verdict shard (spine + verdict, registry verifier) and write .validated."""
    input:
        verdict=str(_snipqc_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
        physical_embryo_registry=str(_snipqc_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_snipqc_validated("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-snip-qc \
          --input-csv "{input.verdict}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_snip_qc:
    """Row-stack per-well snip_qc verdict shards into the experiment-level merged table."""
    input:
        per_well=_snipqc_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_snipqc_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_snipqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'snip_qc', 'verdict', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
