"""Latent-embeddings (legacy VAE) product-family rules.

Consumes the per-well snip_inventory shards, encodes each valid snip through the legacy VAE, and
emits one per-well latents parquet, then the merged experiment table. The PRODUCT is
`latent_embeddings`; the BACKEND (legacy VAE, Python-3.9-pinned) is named only in code/config.

Three things make this step unlike the geometry features (spec: legacy_embeddings.md):
  1. The encode body runs under MODEL_RUN (Python 3.9) — model pickles don't survive 3.10. Only
     FILES cross the line: snip_inventory + PNGs in, latents parquet out.
  2. The weights are a machine-path dependency (env.yaml.paths.models_root + config model_name),
     resolved at runtime by model_paths.resolve_legacy_model_dir — never a registry path.
  3. execution=RUN_BATCH: ONE 3.9 process loads the model once and writes EVERY run well's shard
     before exiting (model load dominates per-well encode cost). The encode rule is therefore a
     BATCH rule over the run set, not one job per well — same shape as SAM2 segmentation.

The validate + merge rules run under the normal RUN env (3.10) — they only read/validate parquet.
"""

LATENT_EMBEDDINGS_STEP = "latent_embeddings"
LATENTS_ARTIFACT = "latents"

# Upstream: snip_inventory owns the per-well snip table the encoder reads (snip_id +
# processed_snip_path). The encoder consumes the VALIDATED shard.
SNIP_INVENTORY_STEP = "snip_inventory"

_LE_CFG = config.get("feature_extraction", {}).get("legacy_embeddings", {})


def _latents_artifact(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_artifact(LATENT_EMBEDDINGS_STEP, LATENTS_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _latents_validated(experiment: str, *, path_mode: str, well_id: str | None = None):
    return rule_validated(LATENT_EMBEDDINGS_STEP, LATENTS_ARTIFACT, experiment, path_mode=path_mode, well_id=well_id)

def _snip_inventory_per_well(experiment: str, *, well_id: str):
    return rule_artifact(SNIP_INVENTORY_STEP, "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _snip_inventory_per_well_validated(experiment: str, *, well_id: str):
    return rule_validated(SNIP_INVENTORY_STEP, "snip_inventory", experiment, path_mode=PATH_MODE_PER_WELL, well_id=well_id)

def _latents_shards_for_run(wc):
    return run_well_shard_paths(DATA_ROOT, LATENT_EMBEDDINGS_STEP, LATENTS_ARTIFACT, wc.experiment, wells_for_experiment(wc))

def _latents_validated_for_run(wc):
    return [_latents_validated(wc.experiment, path_mode=PATH_MODE_PER_WELL, well_id=w) for w in wells_for_experiment(wc)]


if MODEL_RUN is None:
    # Fail loud at PARSE time only if a model rule is actually requested would be ideal, but the
    # prefix is needed in the shell string below; guard here so a missing interpreter is a clear
    # error the moment this family is included with no 3.9 env configured.
    _MODEL_RUN_OR_FAIL = (
        'echo "ERROR: latent_embeddings needs a Python-3.9 model interpreter; set '
        'runtime.model_python_executable or runtime.model_python_env in env.yaml '
        '(model_input_handoff_contract.md §9)." >&2; exit 1; #'
    )
else:
    _MODEL_RUN_OR_FAIL = MODEL_RUN


rule encode_latent_embeddings_for_well:
    """Encode one well's snips through the legacy VAE (Python 3.9) → per-well latents parquet.

    Mirrors the per-well shape every other product uses (one {well_id} job). The encode body runs
    under MODEL_RUN; the entrypoint loads the model, encodes this well's valid snips in
    manifest order, stamps provenance, validates, and writes the shard. The entrypoint accepts a
    list of (inventory, output) pairs; here it is a single pair.

    DOCTRINE NOTE: legacy_embeddings.md specifies execution=RUN_BATCH (load once across wells).
    Realizing a true single-process-all-wells batch needs Snakemake's --batch mechanism, which no
    rule in this repo uses yet; like frame_masks (also RUN_BATCH in the registry), the rule is
    declared per-well. The `execution` field documents intent; the batch optimization is deferred.
    """
    input:
        snip_inventory=str(_snip_inventory_per_well("{experiment}", well_id="{well_id}")),
        snip_inventory_validated=str(
            _snip_inventory_per_well_validated("{experiment}", well_id="{well_id}")
        ),
    output:
        latents=str(_latents_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    params:
        model_run=_MODEL_RUN_OR_FAIL,
        output_root=str(DATA_ROOT),
        models_root=lambda wc: str(_LE_CFG.get("models_root_override") or MODELS_DIR),
        model_name=lambda wc: str(_LE_CFG.get("model_name", "")),
        model_input_height=lambda wc: int(_LE_CFG.get("model_input_shape", [288, 128])[0]),
        model_input_width=lambda wc: int(_LE_CFG.get("model_input_shape", [288, 128])[1]),
        model_input_channels=lambda wc: int(_LE_CFG.get("model_input_channels", 1)),
        batch_size=lambda wc: int(_LE_CFG.get("batch_size", 64)),
        device=lambda wc: str(_LE_CFG.get("device", "cpu")),
    # No gpu=1: this stage is CPU-only in production. config.yaml's
    # feature_extraction.legacy_embeddings has no device override, so params.device defaults to
    # "cpu"; the Python-3.9 model env (mseq_pipeline_py3.9) that runs the encode body ships a
    # torch build with no CUDA support (torch.version.cuda is None) regardless. Claiming gpu=1
    # here just serializes a CPU job behind real GPU work under --resources gpu=1. Do not re-add
    # by pattern-matching the other model-heavy rules — verify device/torch build first.
    shell:
        """
        {params.model_run} -m data_pipeline.feature_extraction.legacy_embeddings.entrypoint \
          --snip-inventory-csv "{input.snip_inventory}" \
          --output-parquet "{output.latents}" \
          --output-root "{params.output_root}" \
          --models-root "{params.models_root}" \
          --model-name "{params.model_name}" \
          --model-input-height {params.model_input_height} \
          --model-input-width {params.model_input_width} \
          --model-input-channels {params.model_input_channels} \
          --batch-size {params.batch_size} \
          --device "{params.device}"
        """


rule validate_latent_embeddings_for_well:
    """Validate a per-well latents shard against the contract; write the .validated sentinel."""
    input:
        latents=str(_latents_artifact(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    output:
        validated=str(_latents_validated(
            "{experiment}", path_mode=PATH_MODE_PER_WELL, well_id="{well_id}"
        )),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-latent-embeddings \
          --input-parquet "{input.latents}" \
          --output-flag "{output.validated}"
        """


rule merge_latent_embeddings:
    """Concat the validated per-well latents shards into the experiment-level parquet."""
    input:
        per_well=_latents_shards_for_run,
        per_well_validated=_latents_validated_for_run,
    output:
        merged=str(_latents_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks merge-latent-embeddings \
          --inputs {input.per_well} \
          --output-parquet "{output.merged}"
        """


rule validate_latent_embeddings:
    input:
        merged=str(_latents_artifact("{experiment}", path_mode=PATH_MODE_MERGED)),
    output:
        validated=str(_latents_validated("{experiment}", path_mode=PATH_MODE_MERGED)),
    shell:
        """
        {RUN} -m data_pipeline.pipeline_orchestrator.tasks validate-latent-embeddings \
          --input-parquet "{input.merged}" \
          --output-flag "{output.validated}"
        """
