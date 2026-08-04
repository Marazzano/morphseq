"""snip_auxiliary_masks product rules.

Per-snip UNet auxiliary masks (via/yolk/focus/bubble/foreground), predicted on the snip crop and
native to snip resolution. Consumes the validated per-well snip_inventory; produces one manifest
row per (snip_id, auxiliary_mask_type) plus the PNG masks beside the shard. Runs AFTER
snip_processing (replaces the retired full-frame auxiliary_masks step). Per-well build -> validate
-> merge, mirroring the feature-product template.
"""

from data_pipeline.model_servers.socket_paths import service_socket_pattern

SNIP_AUX_STEP = "snip_auxiliary_masks"

# --- resident model server (opt-in) -----------------------------------------------------------
# The HIGHEST-VALUE amortization in the pipeline. This step loads FOUR UNet checkpoints
# (via/yolk/focus/bubble) per well: measured 69.94s of load against 2.3-12.6s of actual work.
# A 3-well GPU A/B ran 280.39s per-well vs 97.47s served (2.88x) with all 2,100 mask PNGs
# byte-identical. Over 576 wells that is ~11h of repeated loading collapsing to ~70s.
# See docs/MODEL_LOAD_BENCHMARKS.md and docs/MODEL_SERVER_WIRING.md.
#
# Before editing either rule, read model_servers/socket_paths.py -- it lists the three traps that
# all fail silently.
SNIP_AUX_SERVED = bool(config.get("unet_snip", {}).get("use_model_server", False))

def _snip_aux_socket_pattern() -> str:
    """The socket pattern for this service -- see model_servers.socket_paths for WHY it is a
    wildcard and not a hash. Both the service `output:` and the client `input:` call this, which
    is what makes them agree; that agreement is the whole contract."""
    return service_socket_pattern("unetaux")


def _sam_artifact(experiment, *, path_mode, well_id=None):
    return rule_artifact(SNIP_AUX_STEP, "manifest", experiment, path_mode=path_mode, well_id=well_id)

def _sam_validated(experiment, *, path_mode, well_id=None):
    return rule_validated(SNIP_AUX_STEP, "manifest", experiment, path_mode=path_mode, well_id=well_id)

def _sam_snip_inventory(experiment, *, well_id):
    return rule_artifact("snip_inventory", "legacy_default_snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _sam_snip_inventory_validated(experiment, *, well_id):
    return rule_validated("snip_inventory", "legacy_default_snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)


def _sam_artifacts_for_run(wc):
    return run_well_shard_paths(
        DATA_ROOT, SNIP_AUX_STEP, "manifest", wc.experiment, wells_for_experiment(wc),
    )


def _snip_aux_unet_config_json(wc) -> str:
    """The unet_snip config block as JSON, built exactly as cmd_snip_auxiliary_masks builds it.

    Mirrors tasks.py: read the block, then let the ENVIRONMENT models_root (env.yaml
    paths.models_root) REPLACE any config models_root -- checkpoints may live anywhere on a
    machine, not under the data tree. Keeping this identical to the per-well path is what makes
    the served output equivalent; if these two ever drift, the served masks drift with them.
    """
    import json
    block = config.get("unet_snip") or config.get("auxiliary_masks", {}).get("unet_snip", {})
    block = dict(block)
    block["models_root"] = str(MODELS_DIR)
    return json.dumps(block)


if SNIP_AUX_SERVED:
    rule service_unet_aux_masks:
        """Resident 4x-UNet server: load via/yolk/focus/bubble ONCE, serve per-well requests.

        Only instantiated when unet_snip.use_model_server is true. Holds the GPU for its lifetime,
        which is why it -- and NOT the client rule -- declares gpu=1.

        The harness creates the socket only after all four checkpoints are loaded, so a client that
        connects is guaranteed a ready model. That ordering is the readiness handshake.
        """
        output:
            socket=service(_snip_aux_socket_pattern()),
        params:
            unet_config_json=_snip_aux_unet_config_json,
            device=lambda wc: str(config.get("unet_snip", {}).get("device", DEVICE)),
        resources:
            gpu=1,
        shell:
            """
            {RUN} -m data_pipeline.model_servers.harness \
              --adapter snip_auxiliary_masks \
              --socket-path "{output.socket}" \
              --adapter-arg unet_snip_config_json='{params.unet_config_json}' \
              --adapter-arg device="{params.device}"
            """


    rule build_snip_auxiliary_masks_for_well_served:
        """Per-well auxiliary masks via the resident UNet server (thin socket client).

        Same inputs, same manifest output, same DAG position as the in-process rule; only the
        location of inference differs. Declares NO gpu resource -- this process never touches the
        GPU, and claiming the resource the service holds would deadlock scheduling silently.
        """
        input:
            snip_inventory=str(_sam_snip_inventory("{experiment}", well_id="{well_id}")),
            snip_inventory_validated=str(_sam_snip_inventory_validated("{experiment}", well_id="{well_id}")),
            socket=_snip_aux_socket_pattern(),
        threads: 0  # see model_servers/socket_paths.py, trap 4
        output:
            manifest=str(_sam_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
        params:
            output_root=str(DATA_ROOT),
        shell:
            """
            {RUN} -m data_pipeline.model_servers.client \
              --socket-path "{input.socket}" \
              --payload-json '{{"snip_inventory_csv": "{input.snip_inventory}", "output_root": "{params.output_root}", "output_csv": "{output.manifest}"}}'
            """



else:
    rule build_snip_auxiliary_masks_for_well:
        """Run per-snip UNet auxiliary-mask inference for one well from the validated snip_inventory.

        --models-root is the ENVIRONMENT model root (env.yaml.paths.models_root); it is authoritative and
        replaces any config unet_snip.models_root — model weights may live anywhere on a machine, not
        under the data tree. The per-family `checkpoint` key (config) carries the family sub-route. This
        seam first ran continuously raw->snip_qc in the Tier-1 through-line proof; see
        docs/refactors/streamline-snakemake/target/specs/data_flow_test_plan.md (Tier 1).
        """
        input:
            snip_inventory=str(_sam_snip_inventory("{experiment}", well_id="{well_id}")),
            snip_inventory_validated=str(_sam_snip_inventory_validated("{experiment}", well_id="{well_id}")),
        output:
            manifest=str(_sam_artifact(
                "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
            )),
        params:
            output_root=str(DATA_ROOT),
            config_yaml=str(CONFIG_YAML),
            models_root=str(MODELS_DIR),
        # gpu=1: this process loads the 4x UNet auxiliary-mask models onto the GPU and holds that
        # memory for the job's duration. Do not copy onto a future model-server client rule.
        resources:
            gpu=1,
        shell:
            """
            {RUN} -m data_pipeline.pipeline_orchestrator.tasks snip-auxiliary-masks \
              --snip-inventory-csv "{input.snip_inventory}" \
              --output-root "{params.output_root}" \
              --output-csv "{output.manifest}" \
              --config-yaml "{params.config_yaml}" \
              --models-root "{params.models_root}"
            """


# The served/in-process gate is the `if SNIP_AUX_SERVED:` / `else:` split above, NOT ruleorder --
# only one rule is ever DEFINED. See model_servers/socket_paths.py, trap 3.


rule validate_snip_auxiliary_masks_for_well:
    """Validate the per-well snip_auxiliary_masks shard (contract + cross-check) and write .validated."""
    input:
        manifest=str(_sam_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
        snip_inventory=str(_sam_snip_inventory("{experiment}", well_id="{well_id}")),
    output:
        validated=str(_sam_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-snip-auxiliary-masks \
          --input-csv "{input.manifest}" \
          --snip-inventory-csv "{input.snip_inventory}" \
          --output-flag "{output.validated}"
        """


rule merge_snip_auxiliary_masks:
    """Row-stack per-well snip_auxiliary_masks shards into the experiment-level merged manifest."""
    input:
        per_well=_sam_artifacts_for_run,
        per_well_validated=lambda wc: [
            str(_sam_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w))
            for w in wells_for_experiment(wc)
        ],
    output:
        merged=str(_sam_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -c "
from data_pipeline.pipeline_orchestrator.orchestration.well_runner import (
    collect_well_shard_paths, concat_well_shards_to_file,
)
from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import SNIP_AUXILIARY_MASKS_REQUIRED_COLUMNS
shards = collect_well_shard_paths('{DATA_ROOT}', 'snip_auxiliary_masks', 'manifest', '{wildcards.experiment}')
concat_well_shards_to_file(shards, '{output.merged}', required_columns=SNIP_AUXILIARY_MASKS_REQUIRED_COLUMNS, sort_columns=['experiment_id', 'well_id', 'snip_id', 'auxiliary_mask_type'])
"
        """
