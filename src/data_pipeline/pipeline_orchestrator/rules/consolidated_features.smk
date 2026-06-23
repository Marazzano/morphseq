"""consolidated_features feature-product rules.

Merges the per-well feature shards one-to-one on snip_id. The build verb loads the upstream
shards through the path registry (it takes output_root + experiment + well_id, not raw CSV paths),
so the rule only declares the upstream shards + their validated sentinels as DAG dependencies.

NOTE: consolidates mask_geometry + curvature_metrics + pose_kinematics + stage_predictions +
fraction_alive. fraction_alive joined once its masks moved to snip-native resolution (per-snip
snip_auxiliary_masks), resolving the old VIA/embryo mask resolution mismatch.
"""

CONSOLIDATED_FEATURES_STEP = "consolidated_features"
# Upstream feature products merged by the MVP (must match DEFAULT_FEATURE_STEPS minus fraction_alive).
_CF_SOURCE_STEPS = ["mask_geometry", "curvature_metrics", "pose_kinematics", "stage_predictions", "fraction_alive"]


def _cf_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(CONSOLIDATED_FEATURES_STEP, "consolidated_features", experiment, path_mode=path_mode, well_id=well_id)

def _cf_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(CONSOLIDATED_FEATURES_STEP, "consolidated_features", experiment, path_mode=path_mode, well_id=well_id)

def _cf_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _cf_source_shards(experiment, well_id):
    """The per-well feature shards + validated sentinels the consolidation depends on."""
    deps = []
    for step in _CF_SOURCE_STEPS:
        deps.append(rule_artifact(step, step, experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id))
        deps.append(rule_validated(step, step, experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id))
    return deps


def _cf_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, CONSOLIDATED_FEATURES_STEP, "consolidated_features", wc.experiment, wells_for_experiment(wc),
    )


rule build_consolidated_features_for_well:
    """Merge the per-well feature shards into the consolidated table for one well (loads via registry)."""
    input:
        source_shards=lambda wc: _cf_source_shards(wc.experiment, wc.well_id),
        physical_embryo_registry=str(_cf_registry("{experiment}", well_id="{well_id}")),
    output:
        consolidated=str(_cf_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks consolidated-features \
          --output-root "{DATA_ROOT}" \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.consolidated}"
        """


rule validate_consolidated_features_for_well:
    """Validate the per-well consolidated_features shard (spine + core columns, registry verifier)."""
    input:
        consolidated=str(_cf_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_cf_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_cf_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-consolidated-features \
          --input-csv "{input.consolidated}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_consolidated_features:
    """Row-stack per-well consolidated_features shards into the experiment-level merged table."""
    input:
        per_well=_cf_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_cf_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_cf_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'consolidated_features', 'consolidated_features', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
