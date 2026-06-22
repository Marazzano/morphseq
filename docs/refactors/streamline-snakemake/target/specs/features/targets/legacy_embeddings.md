# Legacy Embeddings — computed feature spec

**Status:** spec, 2026-06-22. Foundation laid (snip source seam, model path resolver,
load-smoke script). Encode loop and pipeline wiring are next.
**Companion to:** `model_input_handoff_contract.md` §9 (the loading boundary doctrine),
`feature_world.md` (the shared feature/QC stage pattern this step follows).

> Read `model_input_handoff_contract.md` §9 before this doc. It locks the 3.9/3.10
> env boundary, the file-only coupling rule, and why `load_model_subprocess.py` must
> not exist. This spec does not relitigate those decisions.

---

## Naming Doctrine

Three names, three distinct scopes — do not blur them:

| Name | Scope | Meaning |
|------|-------|---------|
| `legacy_embeddings` | Backend / implementation | The specific encoder path: the legacy VAE, Python-3.9-pinned. Names the code package and the config key. |
| `latent_embeddings` | Pipeline product | What downstream consumers care about: a table of latent vectors keyed by `snip_id`. The `PIPELINE_STEPS` key and the `product_dir` on disk. |
| `latents` | Artifact | The concrete parquet file. The `artifacts` key in the registry row. |

> **Legacy embeddings** is the backend. **Latent embeddings** is the product. **Latents** is the file.
> The 3.9 process makes files, not objects.

---

## What It Computes

A per-snip latent representation: for each processed snip PNG the legacy VAE encoder
produces a fixed-length vector of latent means (`z_mu_*`) and, when the model emits them,
log-variance outputs (`z_sigma_*`). These are the morphological embeddings consumed by
downstream analysis and analysis-ready joins.

**Output grain:** one row per `snip_id`. Key: `snip_id`. Feature columns: `z_mu_b_*` /
`z_mu_n_*` (or plain `z_mu_*` for non-MetricVAE/SeqVAE models), mirroring the column
naming from `extract_embeddings_legacy`. No biological metadata is joined at encode time
— identity is carried entirely by `snip_id`.

---

## Code Layout

The legacy embedding code is a product-local vertical slice under `feature_extraction/`,
following the same pattern as mask geometry, curvature metrics, etc.:

```text
src/data_pipeline/feature_extraction/legacy_embeddings/
  contract.py          # latent table schema, required columns, and validator
  snip_source.py       # reads snip_inventory and collects processed snip paths
  model_paths.py       # resolves models_root/legacy/<model_name>
  transforms.py        # snip_to_model_input_tensor(): PNG -> model input tensor
  encode.py            # pure encode loop around a loaded encoder
  entrypoint.py        # 3.9 CLI: load model, collect snips, encode, write shards
  load_model_smoke.py  # 3.9 CLI: load model and print metadata only
  __init__.py
```

> The existing foundation files (`snip_source.py`, `model_paths.py`, `load_model_smoke.py`)
> currently live under `src/data_pipeline/features/legacy_embeddings/`. They must be
> migrated to `feature_extraction/legacy_embeddings/` before the encode loop is wired —
> `features/` is a naming dead-end that conflicts with the `feature_extraction/` package
> convention. Do not let both roots grow code.

---

## Output Location (`PIPELINE_STEPS` row)

```python
"latent_embeddings": {
    "stage": "features",
    "product_dir": "latent_embeddings",
    "fanout": PER_WELL_THEN_MERGE,
    "artifacts": {
        "latents": {
            PATH_MODE_PER_WELL: "{well_id}_latents.parquet",
            PATH_MODE_MERGED:   "{experiment_id}_latents.parquet",
        },
    },
}
```

On disk (all paths resolved via `artifact_path()` / `validated_path()` — never typed raw):
```
features/<exp>/latent_embeddings/per_well/<well_id>/<well_id>_latents.parquet
features/<exp>/latent_embeddings/per_well/<well_id>/<well_id>_latents.parquet.validated
features/<exp>/latent_embeddings/<exp>_latents.parquet
features/<exp>/latent_embeddings/<exp>_latents.parquet.validated
```

The `product_dir` names the artifact type, not the backend. `latent_embeddings` not
`legacy_vae`. Method provenance belongs in code and config, not paths.

---

## Latents Contract (`contract.py`)

```python
REQUIRED_COLUMNS = ["snip_id"]   # z_mu_* detected dynamically
```

**Validation rules:**
- `snip_id` present, non-null, and unique in the shard
- at least one `z_mu_*` column exists
- all `z_mu_*` columns are numeric and non-null
- `z_sigma_*` columns are optional (included when the model emits them); if present,
  numeric and non-null
- no biological metadata columns required or expected here

**MVP output:** `z_mu_*` required; `z_sigma_*` included when available. Downstream uses
that need only means are not broken by a model that omits sigma.

---

## Config Knobs (`config.yaml`)

```yaml
features:
  legacy_embeddings:
    model_name: 20241107_ds_sweep01_optimum   # which VAE — a science choice
    model_input_shape: [288, 128]             # [height, width] — NOT [width, height]
    model_input_channels: 1                  # 1=grayscale (MetricVAEConfig default); 3=RGB
    batch_size: 64
```

`model_input_shape` is `[height, width]` following the NumPy/torchvision (H, W) convention.
Do not interpret it as `[width, height]`. This drives `snip_to_model_input_tensor()` in
`transforms.py` — the explicit, config-driven replacement for `basic_transform(target_size=(288, 128))`.

`model_input_channels` must match the channel count the model was trained with.
The production model (`MetricVAEConfig`) defaults to `input_dim=(1, 288, 128)` — grayscale.
The legacy pipeline used `transforms.Grayscale(num_output_channels=1)` throughout; passing
`model_input_channels: 3` to a model trained on 1-channel input will silently produce garbage
embeddings or a shape error at the first conv layer.

`model_name` is a science knob (determines the embedding space). `models_root` is a
machine path. They live in different files and never swap.

---

## Encoding Input Policy

`collect_snip_inputs()` in `snip_source.py` applies this filter before any snip is encoded:

- include rows where `is_valid_snip == True`
- require `processed_snip_path` non-null and the resolved file exists
- require `snip_id` non-null and unique in the encoded input set
- **preserve snip_inventory row order** — the output is a faithful ordered transform of
  the manifest; tests pin this

---

## Three Hard Constraints (what makes this step unlike geometry features)

### 1. Python 3.9 env boundary — only files cross

The legacy VAE encoder is 3.9-pinned (custom encoder/decoder pickles do not survive
3.9 → 3.10). The whole encode command body runs in Python 3.9; the orchestrating
Snakemake process (3.10) only schedules the job and validates the output artifact.
**No model object ever crosses the version line.** Only the snip_inventory + PNG files
go in; only the latents parquet comes out.

Interpreter resolution: prefer `env.yaml.runtime.model_python_executable` (direct, no
conda-run shell layer); fall back to `conda run -n {model_python_env}`.

### 2. Model weights are a machine-path dependency

`models_root` and the weights under it are not committed and are not science config.
They live in `env.yaml.paths.models_root`. The registry row names the artifact location;
`model_paths.resolve_legacy_model_dir` constructs the weights path from
`models_root + model_name` at runtime.

### 3. Batch execution — model loads once, writes per-well shards

Loading the legacy VAE through the 3.9 sub-env (process spawn + checkpoint deserialize)
dominates the cost of encoding a few hundred snips per well. The encode job therefore
uses:

- **`fanout=per_well_then_merge` for output** — one validated shard per well, same as
  every other feature step
- **`execution=RUN_BATCH_WRITES_PER_WELL_SHARDS`** — one Python 3.9 process loads the
  model once, iterates over all run wells, and writes each well's shard before exiting

`PER_WELL_THEN_MERGE` does NOT mean one shell process per well here. The Snakemake rule
is a batch rule over the run set; the entrypoint iterates wells internally. This is the
same shape as stitching.

---

## Needs Before Coding (encode loop)

- Migrate foundation files from `features/legacy_embeddings/` →
  `feature_extraction/legacy_embeddings/` (snip_source, model_paths, load_model_smoke).
- `config.yaml` `features.legacy_embeddings` section (model_name, model_input_shape,
  batch_size).
- `PIPELINE_STEPS` row for `latent_embeddings` in `orchestration/paths.py`.
- `contract.py` — latents schema and `validate_latent_embeddings(df)`.
- `transforms.py` — `snip_to_model_input_tensor(model_input_shape)`.
- `encode.py` — pure encode loop (takes a loaded encoder + list of SnipInputs, returns
  a DataFrame).
- `entrypoint.py` — 3.9 CLI: reads config, loads model once, iterates wells, writes shards.
- Legacy VAE weights staged at `models_root/legacy/<model_name>/` so the load-smoke
  confirms a real in-3.9 load.

---

## Done When

- `load_model_smoke.py` exits 0 with real weights (proves the 3.9 env + weights work).
- `entrypoint.py` (3.9): given a snip_inventory CSV, models_root, and model_name, encodes
  all valid snips and writes `<well_id>_latents.parquet` with `snip_id` + `z_mu_*` columns.
- Per-well shard validates against `contract.py`.
- Merged `<exp>_latents.parquet` is a concat of validated per-well shards, with `.validated`
  sidecar written.
- `analysis_ready` gains `latents.parquet` as an input, joins on `snip_id`, and sets
  `embedding_calculated=True` for matched rows.
- Tests in `tests/data_pipeline/feature_extraction/legacy_embeddings/` cover: transform
  shape + dtype contract (including `[height, width]` axis order), snip source gating and
  row-order preservation, model path resolution, contract validation, and a synthetic
  encode-loop smoke (CPU, tiny fixture model or mocked encoder).

---

## Not This Step

- Biological metadata joins (`genotype`, `predicted_stage_hpf`, `use_snip`) happen in
  `analysis_ready`, not here. Encode all valid snips; gate downstream.
- Training-time metadata (`pert_id`, `metric_array`, train/eval splits) is out of scope
  for inference. See `model_input_handoff_contract.md` §7.
- The `bf_embryo_snips/<exp>/` symlink view (Decision 1 in the original handoff contract)
  is legacy-compat only. New code reads from the manifest directly.
