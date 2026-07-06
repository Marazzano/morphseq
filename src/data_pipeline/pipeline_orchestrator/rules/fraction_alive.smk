"""fraction_alive feature-product rules.

Continuous viability fraction per snip: the per-snip ``foreground`` (whole-embryo) mask AND-ed
against the per-snip ``via`` (dead-tissue) mask, both from the snip-resolution snip_auxiliary_masks
product (co-keyed, co-resolution). Consumes validated snip_inventory + validated
snip_auxiliary_masks + the per-well physical_embryo_registry (identity verifier). Per-well build ->
validate -> merge, mirroring the feature-product template.
"""

FRACTION_ALIVE_STEP = "fraction_alive"


def _fa_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(FRACTION_ALIVE_STEP, "fraction_alive", experiment, path_mode=path_mode, well_id=well_id)

def _fa_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(FRACTION_ALIVE_STEP, "fraction_alive", experiment, path_mode=path_mode, well_id=well_id)

def _fa_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fa_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fa_snip_aux(experiment, *, well_id):
    return rule_artifact("snip_auxiliary_masks", "manifest", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fa_snip_aux_validated(experiment, *, well_id):
    return rule_validated("snip_auxiliary_masks", "manifest", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fa_registry(experiment, *, well_id):
    return rule_artifact("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _fa_registry_validated(experiment, *, well_id):
    return rule_validated("physical_embryo_registry", "physical_embryo_registry", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _fa_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, FRACTION_ALIVE_STEP, "fraction_alive", wc.experiment, wells_for_experiment(wc),
    )


rule build_fraction_alive_for_well:
    """Compute the per-well fraction_alive shard from validated snip_inventory + snip_auxiliary_masks."""
    input:
        snip_inventory=str(_fa_snip_inventory("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(_fa_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        snip_auxiliary_masks=str(_fa_snip_aux("{experiment}", well_id="{well_id}")),
        snip_auxiliary_masks_validated=str(_fa_snip_aux_validated("{experiment}", well_id="{well_id}")),
        physical_embryo_registry=str(_fa_registry("{experiment}", well_id="{well_id}")),
        physical_embryo_registry_validated=str(_fa_registry_validated("{experiment}", well_id="{well_id}")),
    output:
        fraction_alive=str(_fa_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    params:
        output_root=str(DATA_ROOT),
        config_yaml=str(CONFIG_YAML),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks fraction-alive \
          --snip-inventory-csv "{input.snip_inventory}" \
          --snip-auxiliary-masks-csv "{input.snip_auxiliary_masks}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-csv "{output.fraction_alive}" \
          --output-root "{params.output_root}" \
          --config-yaml "{params.config_yaml}"
        """


rule validate_fraction_alive_for_well:
    """Validate the per-well fraction_alive shard (spine + features, registry as verifier) and write .validated."""
    input:
        fraction_alive=str(_fa_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        physical_embryo_registry=str(_fa_registry("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_fa_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-fraction-alive \
          --input-csv "{input.fraction_alive}" \
          --physical-embryo-registry-csv "{input.physical_embryo_registry}" \
          --output-flag "{output.validated}"
        """


rule merge_fraction_alive:
    """Row-stack per-well fraction_alive shards into the experiment-level merged table."""
    input:
        per_well=_fa_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_fa_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_fa_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
shards = collect_well_shard_paths('{DATA_ROOT}', 'fraction_alive', 'fraction_alive', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', sort_columns=['experiment_id', 'well_id', 'snip_id'])
"
        """
