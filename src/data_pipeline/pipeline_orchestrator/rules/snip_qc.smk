"""snip_qc quality-control rules — the final per-snip verdict.

Builds use_snip + qc_fail_reasons by ORing the MVP exclusion flags. The eligible source
steps and their flag columns are declared in flag_input_resolver._SOURCE_PAYLOADS; the
policy (which flags matter) comes from SNIP_QC_EXCLUSION_FLAGS or a config
override. The resolver joins those two truths at parse time to determine which source
artifacts are needed — no hardcoded source step list here.

DAG shape per well:
  1. write_snip_qc_resolved_sources_for_well
       input:  source QC CSVs + sentinels (DAG gate — resolver does not read them)
       output: {well_id}_snip_qc_resolved_sources.json  (tracked planning artifact)
  2. build_snip_qc_for_well
       input:  resolved_sources JSON + snip_inventory + physical_embryo_registry
       output: {well_id}_snip_qc.parquet
  3. validate_snip_qc_for_well  →  merge_snip_qc

See: docs/refactors/streamline-snakemake/target/specs/quality_control/snip_qc_verdict_and_flag_resolver.md
"""

import json as _json
import shlex as _shlex

from data_pipeline.quality_control.snip_qc.contract import SNIP_QC_EXCLUSION_FLAGS
from data_pipeline.quality_control.snip_qc.flag_input_resolver import (
    ResolvedFlagSource,
    resolve_snip_qc_flag_sources,
)

# Resolve config override vs default at parse time — both planning and runtime use this.
_SNIP_QC_EXCLUSION_FLAGS: tuple = tuple(
    config.get("snip_qc", {}).get("exclusion_flags") or SNIP_QC_EXCLUSION_FLAGS
)
# Shell-safe JSON string for passing to the writer rule's shell command.
_EXCLUSION_FLAGS_JSON_QUOTED = _shlex.quote(_json.dumps(list(_SNIP_QC_EXCLUSION_FLAGS)))

SNIP_QC_STEP = "snip_qc"


def _snipqc_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(SNIP_QC_STEP, "verdict", experiment, path_mode=path_mode, well_id=well_id)


def _snipqc_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(SNIP_QC_STEP, "verdict", experiment, path_mode=path_mode, well_id=well_id)


def _snipqc_resolved_sources(experiment, *, well_id):
    return rule_artifact(SNIP_QC_STEP, "resolved_sources", experiment,
                         path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _snipqc_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment,
                         path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _snipqc_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment,
                          path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _snipqc_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment,
                         path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _snipqc_source_shards(experiment, well_id):
    """Return source flag CSV paths + validated sentinels for all steps needed by the resolver.

    These are DAG gating inputs to write_snip_qc_resolved_sources_for_well — they ensure
    upstream QC products are built and validated before snip_qc runs. The resolver itself
    does not read these files; it only resolves their paths.
    """
    resolved = resolve_snip_qc_flag_sources(
        _SNIP_QC_EXCLUSION_FLAGS,
        output_root=DATA_ROOT,
        experiment_id=experiment,
        well_id=well_id,
    )
    deps = []
    for src in resolved:
        deps.append(str(src.path))
        deps.append(str(validated_path(
            DATA_ROOT, src.step, src.artifact_key, experiment,
            path_mode=PATH_MODE_PER_WELL, well_id=well_id,
        )))
    return deps


def _snipqc_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, SNIP_QC_STEP, "verdict", wc.experiment, wells_for_experiment(wc)
    )


rule write_snip_qc_resolved_sources_for_well:
    """Serialize the resolver output for this well as a tracked JSON artifact.

    This rule runs during DAG execution (not planning). The source QC CSVs and their
    validated sentinels are listed as inputs to gate DAG execution — Snakemake will not
    run this rule until all upstream QC products are built and validated. The resolver
    itself is pure (no disk reads); it derives source paths from paths.py alone.

    Output contains both exclusion_flags and resolved_sources so build_snip_qc_for_well
    receives the exact same plan the DAG was declared with — no split-brain between
    planning and runtime.
    """
    input:
        source_shards=lambda wc: _snipqc_source_shards(wc.experiment, wc.well_id),
    output:
        resolved_sources=str(_snipqc_resolved_sources("{experiment}", well_id="{well_id}")),
    params:
        exclusion_flags_json=_EXCLUSION_FLAGS_JSON_QUOTED,
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks write-snip-qc-resolved-sources \
          --output-root "{DATA_ROOT}" \
          --experiment "{wildcards.experiment}" \
          --well-id "{wildcards.well_id}" \
          --exclusion-flags-json {params.exclusion_flags_json} \
          --output-json "{output.resolved_sources}"
        """


rule build_snip_qc_for_well:
    """Build the per-well snip_qc verdict from the resolved flag sources + snip_inventory."""
    input:
        resolved_sources=str(_snipqc_resolved_sources("{experiment}", well_id="{well_id}")),
        snip_inventory=str(_snipqc_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_snipqc_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        physical_embryo_registry=str(_snipqc_registry("{experiment}", well_id="{well_id}")),
    output:
        verdict=str(_snipqc_artifact("{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}")),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-qc \
          --resolved-sources-json-path "{input.resolved_sources}" \
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


rule snip_qc_report:
    """TERMINAL: exclusion-reason fraction over time, all snips + not-dead-only as side-by-side
    panels in ONE PNG (shared y-axis + legend, for direct comparison). Consumed by nothing — only
    the `reports` aggregate target requests this. See viz/report_world.md."""
    input:
        snip_qc=str(_snipqc_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        exclusion_reasons_png=str(rule_artifact("snip_qc_report", "exclusion_reasons_png", "{experiment}", path_mode=PATH_MODE_EXPERIMENT)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-qc-report \
          --snip-qc-path "{input.snip_qc}" \
          --output-exclusion-reasons-png "{output.exclusion_reasons_png}"
        """
