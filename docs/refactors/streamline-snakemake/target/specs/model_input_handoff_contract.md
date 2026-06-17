# Model Input Handoff Contract

**Status:** Design / plan only (no code yet)
**Date:** 2026-06-05
**Scope:** How processed snips are passed from the data pipeline into the
representation model. **This pass targets INFERENCE only** (encode snips →
latents), mirroring what the original pipeline's Build06 step does. Training-time
metadata (`pert_id`, `metric_array`, splits) is documented for context but is
out of scope here.

---

## 1. Problem statement

The data pipeline now produces, **per well**, fully model-ready snip *images*
plus a per-well/merged *manifest*. But there is no defined seam by which the
model actually consumes them. The model code expects a different shape of input
than the pipeline currently emits, and the gap has two halves:

1. **Image discovery.** The model's data configs locate images by globbing a
   flat legacy tree, not by reading the manifest.
2. **Per-snip biological metadata.** The training datasets need index-aligned
   vectors (`age_hpf`, perturbation id, embryo id, train/eval split) that the
   snip manifest does **not** carry. Snip processing is purely geometric.

This document defines the handoff contract that closes both halves.

---

## 2. What the pipeline produces today

From `pipeline_orchestrator/Snakefile`, rules
`run_snip_processing_per_well` → `merge_snip_manifests`
(code: `src/data_pipeline/snip_processing/run_per_well.py`):

```
data_pipeline_output/processed_snips/<experiment>/
  per_well/<well_id>/
    snips/processed/<snip_id>.jpg      # MODEL INPUT IMAGE
    snips/raw_crops/<snip_id>.tif
    snips/masks/embryo_mask/<snip_id>.png
    contracts/snip_manifest.parquet    # per-well manifest
  contracts/snip_manifest.parquet      # merged (experiment-level)
```

**Processed image properties** (from `process_single_snip`):
- grayscale, `576 × 256` (H×W), rescaled to `target_pixel_size_um = 7.8`
- PCA-rotated to canonical orientation, embryo-centered crop
- CLAHE + background-noise blending applied
- `.jpg`

**Snip manifest columns** (geometric/provenance only — *no biology*):
`snip_id, experiment_id, well_id, well_index, image_id, frame_index,
channel_id, embryo_id, instance_id, source_image_path, embryo_mask_path,
yolk_mask_path, source_micrometers_per_pixel, frame_snapshot_hash,
processed_snip_path, raw_crop_path, target_pixel_size_um, output_height_px,
output_width_px, blend_radius_um, background_mean, background_std,
rotation_angle_rad, rotation_angle_deg, rotation_used_yolk,
snip_processing_run_id, snip_processing_version`

`snip_id` invariant: `f"{embryo_id}_{channel_id}_f{frame_index:04d}"`.

---

## 3. What the model expects today

### 3.1 Image source (`src/data/dataset_configs.py`)
`_collect_snip_paths` globs a **flat legacy tree**:

```
<root>/training_data/bf_embryo_snips/<experiment>/*.jpg | *.png
```

The legacy training datasets in `src/core/functions/dataset_utils.py`
(`DatasetCached`, `SeqPairDatasetCached`, `TripletDatasetCached`) subclass
`torchvision.datasets.ImageFolder`, which:
- expects a `root/<class>/<file>` layout, and
- assigns each file an **integer index by sort order** — that index is the join
  key into all metadata.

### 3.2 Per-snip metadata, index-aligned (`model_config.seq_key_dict`)
The pair/triplet datasets do **not** read the manifest. They index into parallel
vectors that must be aligned 1:1 with the `ImageFolder` sample order:
- `pert_id_vec` — integer perturbation/class id per snip
- `e_id_vec`    — integer embryo id per snip
- `age_hpf_vec` — developmental age per snip
- `metric_array` — perturbation×perturbation pos/neg relationship matrix
- `train_indices` / `train_bool` / `eval_bool` — split membership

In the real training config these come from `make_metadata()` reading
`metadata/age_key.csv` (see `_find_data_root` / `load_trained_model` in
`src/core/run/run_utils.py`). The analysis-time shim
`BaseDataConfig.make_metadata()` is a **no-op**.

### 3.3 The mismatch
| Need | Pipeline emits | Gap |
|------|----------------|-----|
| Images under flat `bf_embryo_snips/<exp>/` | Per-well `processed/` dirs | Layout differs |
| Stable per-snip metadata join | `ImageFolder` integer index (sort-order, fragile) | No durable key used |
| age / perturbation / genotype | Manifest has none | Lives in `analysis_ready` |

---

## 3.5 What the ORIGINAL pipeline does at inference (Build06)

This is the behavior we are mirroring. Inference is much simpler than training —
it does **not** touch `pert_id`, `metric_array`, or splits at all.

Entry: `src/run_morphseq_pipeline/steps/run_build06.py` → `run_build06()`, which
per experiment:
1. Loads QC'd embryos: `metadata/build04_output/qc_staged_<exp>.csv`, filtered to
   `use_embryo_flag == True`.
2. `ensure_latents_for_experiments()` →
   `generate_latents_with_repo_images()` (`services/gen_embeddings.py`):
   - Loads the legacy model from `<data_root>/models/legacy/<model_name>`.
   - Builds an `EvalDataConfig(experiments=[...], root=repo_root)` dataset, which
     globs images from `repo_root/training_data/bf_embryo_snips/<exp>/*.jpg`.
   - `transform = basic_transform(target_size=(288, 128))`, `batch_size=64`,
     `shuffle=False`, grayscale.
   - `extract_embeddings_legacy()` runs `lit_model.encoder(x)` under
     `torch.no_grad()`/eval and emits a DataFrame with
     `experiment_date, embryo_id, snip_id, z_mu_*` (and `z_sigma_*`).
     **All identity comes from the image file stem (`snip_id`)** — there is no
     metadata join at encode time.
   - Writes `analysis/latent_embeddings/legacy/<model_name>/morph_latents_<exp>.csv`.
3. Merges latents back onto the QC table on `snip_id`.

**Inference contract, distilled:**
> Given a set of processed snip JPGs discoverable per experiment, run the model
> encoder over them (no metadata needed) and produce a `snip_id → z_mu_*` table.

So for the inference build step, the *only* thing the model needs from the
pipeline is **the images, discoverable by experiment, named by `snip_id`** — which
our symlink view (Decision 1) provides directly. No `model_inputs` join is
required to *generate* latents; the join to biology happens afterward when the
latents are merged back for analysis.

### `pert_id` — answer

`pert_id` is a **training-time** integer, *not* used at inference. In the original
code (`src/vae/auxiliary_scripts/make_training_key.py::make_seq_key`):

```python
pert_u = np.unique(seq_key["short_pert_name"])          # categories
pert_df["perturbation_id"] = np.arange(len(pert_u))     # factorized -> int
seq_key = seq_key.merge(pert_df, on="short_pert_name")  # pert_id per snip
```

i.e. `pert_id = factorize(short_pert_name)`. `short_pert_name` is the
human-readable perturbation label assembled during QC (Build04), effectively a
**composite of genotype + treatment/chemical** (see
`build04_perform_embryo_qc.py`: `short_pert_name`, `master_perturbation`,
`temperature`, `medium`). Your intuition is right: it's a genotype×treatment
composite, collapsed to one string then integer-encoded.

For the new pipeline that composite would be derived from `analysis_ready`'s
`genotype` + `treatment` columns — **but that is only needed when we build the
training step, not now.**

---

## 4. Where the biological metadata actually lives (for later / training)

The pipeline already joins biology downstream of snip processing. In
`consolidate_features` and then `assemble_analysis_ready`, keyed by `snip_id`:

- `predicted_stage_hpf` — the **age signal** (→ `age_hpf_vec`)
- `start_age_hpf` — plate-declared start age
- `genotype`, `treatment` (a.k.a. `chem_perturbation`) — the **perturbation
  signal** (→ `pert_id_vec`)
- `embryo_id` — (→ `e_id_vec`)
- `use_snip` + QC exclusion flags — the **usability gate**

So `analysis_ready.csv` (`data_pipeline_output/analysis_ready/<exp>/`) is the
natural source for the model's per-snip metadata, joined to the snip manifest on
`snip_id`.

---

## 5. Decisions (locked)

1. **Image seam — manifest is source of truth; the legacy
   `bf_embryo_snips/<exp>/` tree is a symlink *view*.**
   New code reads images via the manifest's `processed_snip_path`. A thin
   materializer symlinks (not copies) those paths into the legacy flat tree so
   existing `ImageFolder`/`EvalDataConfig` code works unchanged.

2. **Inference needs only the images.** The encode step takes the symlink view
   per experiment and produces `snip_id → z_mu_*`. No metadata join is required
   to generate latents.

3. **Metadata — join from `analysis_ready` (← consolidated) on `snip_id`.**
   This applies to (a) merging latents back for analysis, and (b) the future
   training step (`pert_id` etc.). Do **not** widen the snip manifest with
   biology; keep geometric prep decoupled from annotation.

4. **First deliverable — this design doc only.** No code in this pass.

---

## 6. The inference build step (build_06 equivalent) — THIS PASS

Goal: reproduce Build06's *inference* behavior on the new pipeline's outputs —
encode processed snips into latents keyed by `snip_id`. No training, no
`pert_id`, no `model_inputs` join required to encode.

### 6.0 Spine position (IMPORTANT — embeddings FEEDS analysis_ready)
Embeddings is a **late spine stage**, not a terminal tail. Legacy:
`build06: df02 + latents → df03`; and `analysis_ready` reserves
`embedding_calculated` (today hardcoded False). So:
```
… snips → features → qc → EMBEDDINGS → analysis_ready
                          (latents join on snip_id, flip embedding_calculated)
```
**Dependency-ordering consequence:** the `use_snip` gate **cannot** come from
`analysis_ready` (that would be circular — analysis_ready is downstream of us).
Gate from the **QC consolidation** instead (`quality_control/.../qc_flags.csv`,
which defines `use_snip` upstream of analysis_ready), or simply encode **all**
snips and let analysis_ready do the gating on join. (See §8 Q2.)

### 6.1 Inputs / outputs (per-well shard + symlink view)
```
INPUT (per RUN well):
  processed_snips/<exp>/per_well/<well_id>/contracts/snip_manifest.parquet
    -> processed_snip_path  (the JPGs to encode)
  quality_control/<exp>/consolidated/qc_flags.csv   (optional use_snip gate; NOT analysis_ready)

OUTPUT (per-well shard — fanout=per_well):
  embeddings/<exp>/per_well/<well_id>/latents.parquet   # snip_id, z_mu_*, (z_sigma_*)
  embeddings/<exp>/per_well/<well_id>/.latents.validated
OUTPUT (merged publication, Zone C):
  embeddings/<exp>/contracts/latents.parquet           # thin concat of shards
  embeddings/<exp>/bf_embryo_snips/<exp>/<snip_id>.jpg  # symlink view (Decision 1)
```

### 6.2 Execution — fanout=per_well, execution=SINGLE (batched). ⭐ the key call.
Latents land **per well** (spine + incremental staleness), but **ONE job loads
the model once** and encodes all *run* wells, writing each well's shard. Loading
the legacy model through the Py-3.9 `conda run` sub-env **per well** would dominate
the encode (process spawn + checkpoint deserialize ≫ encoding a few hundred snips).
This is the canonical `per_well` + `single` stage — same shape as stitching.

Steps (mirrors `generate_latents_with_repo_images`):
1. **Resolve run wells** (well-runner `run_wells`), and for each, its snip shard +
   `processed_snip_path`s; symlink them into the `bf_embryo_snips/<exp>/` view.
2. **Load the model ONCE** from `models_root/.../<model_name>` (reuse
   `resolve_model_dir` / `load_legacy_model_safe` → the `conda run` sub-env once).
3. **Encode all run wells** in batches (`basic_transform(target_size=(288,128))`,
   `batch_size=64`, `shuffle=False`, eval + `no_grad`, `extract_embeddings_legacy`).
4. **Write each well's shard** `per_well/<well_id>/latents.parquet` (`snip_id, z_mu_*`).

> Identity is carried entirely by the `snip_id` file stem — no metadata join is
> needed to produce latents. Biology is joined *after*, in analysis_ready.

### 6.3 Reuse vs. rebuild
The existing `services/gen_embeddings.py` already implements the encode loop; the
new pieces are (a) **sourcing images from the manifest** (→ symlink view) instead
of the hand-built `training_data/bf_embryo_snips`, and (b) **partitioning output
per well** while keeping the single model load. Prefer wrapping the service over
rewriting the encoder.

### 6.4 Snakemake integration
Two rules following the spine rhythm:
- `compute_embeddings` (**fanout=per_well, execution=single**): declared inputs =
  the run wells' snip shards (via `merge_trigger_inputs`-style fan), loads the model
  once, writes each `per_well/<well_id>/latents.parquet` + `.validated`.
- `merge_embeddings` (Zone C): thin concat of present `.validated` shards →
  `embeddings/<exp>/contracts/latents.parquet`.

Then `assemble_analysis_ready` gains `latents.parquet` (merged) as an input,
joins on `snip_id`, and sets `embedding_calculated=True` for matched snips.

### 6.5 Environment / sub-environment (IMPORTANT)

Today the Snakefile pins **one** interpreter for every rule:
```python
PYTHON = ".../envs/segmentation_grounded_sam/bin/python"   # hardcoded, no config
```
There is no env setting in `config.yaml`. This is fine while every step lives in
one env — but **the legacy model encoder does not.**

The legacy model is pickle/TorchScript-pinned to **Python 3.9**, so the existing
inference code already shells out to a *sub-environment*
(`services/legacy_model_utils.py::load_legacy_model_safe`):
```python
cmd = ['conda', 'run', '-n', 'mseq_pipeline_py3.9', 'python', subprocess_script, ...]
```
i.e. the orchestrating step runs in the main env, and **model loading is routed
to `conda run -n mseq_pipeline_py3.9`** as a subprocess. We reuse that whole
stack rather than reinventing it.

**Decision — align with Scope 3 (`env.yaml`), do NOT invent a `config.yaml`
block.** The target refactor already specifies the environment story as **Scope
3**: machine knobs (`python`, `device`, paths, `models_root`) live in a
gitignored **`env.yaml`** beside `config.yaml`; science knobs stay in
`config.yaml`. The model interpreter is a *machine knob*, so it belongs in
`env.yaml.runtime`, **not** in `config.yaml`. (See
`well_id_throughline_refactor_plan.md` Scope 3.)

The build_06 env therefore slots into the planned `env.yaml`:
```yaml
# env.yaml — gitignored, per-machine (Scope 3)
runtime:
  python: "/net/.../envs/segmentation_grounded_sam/bin/python"   # main pipeline
  model_python_env: "mseq_pipeline_py3.9"   # legacy model encoder sub-env (Py 3.9)  ← ADD
  device: "cuda"
paths:
  models_root: "/net/.../models"            # where the legacy model weights live
```

How build_06 uses it:
- Orchestration runs under `runtime.python` (the main env).
- The encoder model load uses `runtime.model_python_env` as the
  `target_python_env` passed into `load_legacy_model_safe`, replacing the
  hardcoded `"mseq_pipeline_py3.9"` default. The existing `conda run -n <env>`
  sub-environment machinery does the rest.

**Interim (before Scope 3 `env.yaml` exists):** since Scope 3 is a separate
later track and `env.yaml` is not built yet, build_06 may read the model env name
from a single value (default `"mseq_pipeline_py3.9"`) so it is not hardcoded
inside `legacy_model_utils`. When `env.yaml` lands, move that value to
`runtime.model_python_env`. **Do not add an `environments:` block to
`config.yaml`** — that would diverge from the Scope 3 design.

---

## 7. Deferred: the `model_inputs` table (TRAINING, not this pass)

When we build the training step, add a `model_inputs` table = 1:1 join of merged
snip manifest ⋈ `analysis_ready` on `snip_id`, carrying the index-aligned
training metadata:

| Column | Source | Purpose |
|--------|--------|---------|
| `snip_id` | both (join key) | stable per-snip identity |
| `embryo_id` | manifest | → `e_id_vec` |
| `processed_snip_path` | manifest | image to load |
| `predicted_stage_hpf` | analysis_ready | → `age_hpf_vec` |
| `short_pert_name` (= `genotype` × `treatment`) | derived from analysis_ready | → `pert_id_vec` |
| `use_snip` | analysis_ready | usability gate |

Invariants when we get there: stable `snip_id` join key (never sort-order),
`use_snip` filtered once before index assignment, deterministic persisted
categorical maps for `pert_id`/`e_id`.

---

## 8. Open questions

**For the inference pass (this one):**
1. **Where do trained models live** — leaning Scope-3 `models_root` (machine path
   in `env.yaml`), reusing the legacy `models/legacy/<model_name>` layout under it.
   Confirm when wiring.
2. **`use_snip` gating — RESOLVED (ordering):** embeddings **feeds** analysis_ready,
   so the gate can NOT come from analysis_ready (circular). Either gate from QC
   consolidation (`qc_flags.csv`, upstream) or encode all snips and let
   analysis_ready gate on join. Default: **encode all**, gate downstream. (§6.0)
3. **Output location — RESOLVED:** per-well shard
   `embeddings/<exp>/per_well/<well_id>/latents.parquet` + merged
   `embeddings/<exp>/contracts/latents.parquet`. (§6.1)
   - **Execution — RESOLVED:** `fanout=per_well`, `execution=single` (batched;
     model loaded once). (§6.2)
4. **Env config — RESOLVED:** the model interpreter is a machine knob → belongs
   in Scope 3's `env.yaml` (`runtime.model_python_env`), **not** `config.yaml`.
   Interim: read it from a single non-hardcoded value defaulting to
   `mseq_pipeline_py3.9`. (§6.5)
5. **Confirm the model env name** — is `mseq_pipeline_py3.9` still the correct/
   current env on this machine, or has it been renamed?

**Deferred to the training pass:**
4. **`metric_array`** pos/neg relationships from `short_pert_name`.
5. **Train/eval split** source and grain (per-embryo vs per-snip).
6. **Age source** — `predicted_stage_hpf` vs clock-based.
7. **Cross-experiment** merge vs list-of-tables for the loader.
```
