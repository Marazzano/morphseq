# Orientation for Final Review — morphseq pipeline refactor

**Purpose:** a single entry point for a model doing a final scan of the pipeline
refactor. It says **where the authoritative docs are**, **which files matter**,
and **what is legacy vs. target** so you don't have to rediscover the layout.
**Date:** 2026-06-05.

> **Read this first, then the `target/` docs.** This file is a map, not the spec.
> The design lives in `target/` (symlinked). Some top-level docs here are stale —
> see "Doc trust order" below.
>
> **Auditing this refactor?** Start with [`AUDIT_PASSOFF.md`](AUDIT_PASSOFF.md) (this dir) — reading order, the spine claims to scrutinize, and your deliverable (`target/OVERALL_PLAN.md`).

---

## 0. The one-paragraph mental model

There are **two pipelines in this repo**: the **legacy** procedural pipeline
(`src/run_morphseq_pipeline/` — the `buildNN` steps, still the source of truth
for *behavior*) and the **target** Snakemake pipeline (`src/data_pipeline/` —
contract-driven, per-well, the thing the refactor is building). The refactor
*ports legacy behavior into the Snakemake pipeline*. The Snakemake pipeline is
wired end-to-end **through analysis-ready**, but the **model/embedding step
(legacy "build_06") is not yet ported** — that is the active frontier.

---

## 1. Doc trust order (IMPORTANT — some docs are stale)

| Trust | Location | Notes |
|---|---|---|
| ✅ **Authoritative target** | `target/` (symlink → morphseq-docs) | Verified 2026-06-04. Start at `target/current_state_and_next_steps.md`. |
| ✅ **Active frontier spec** | `target/model_input_handoff_contract.md` | The model/embedding handoff (legacy build_06), inference-first. 2026-06-05. Moved into `target/` 2026-06-05. |
| 🗄️ **Archived (stale terms, useful reference)** | `_archive_2026-06-05_pre-target/` | Pre-`target/` structure docs. They say `frame_manifest.csv` / `stitched_image_index.csv`; the **real Snakefile uses `frame_contract.csv` / `stitched_inventory.csv`**, and downstream **is** now wired. Kept for the **dataset-output tree** (`data_output_structure.md`) and **`src/` script structure** (`processing_files_pipeline_structure_and_plan.md`). See that folder's README. |
| 🗄️ **Historical** | `_Archive/`, `logs/`, `supplementary_files_(...)` | Older planning/review context only. |

**`target/` contents (the real spec):**
- `current_state_and_next_steps.md` — verified on-disk state + recommended order (the "where are we" anchor).
- `per_well_throughline_findings.md` — north star: grain model, registry, well-runner, DAG mechanics.
- `well_id_throughline_refactor_plan.md` — the formal **Scopes 1–5** (incl. Scope 3 = config/env).
- `front_end_naming_and_flow.md` — ingest lineages, fan-out, post-fan tail.
- `stitched_handoff_contract.md` — the microscope-agnostic stitched drop-in seam.

---

## 2. Legacy vs. target — code map

| Stage | Legacy (behavior source) | Target (Snakemake refactor) | State |
|---|---|---|---|
| Orchestration | `src/run_morphseq_pipeline/cli.py` + `steps/run_buildNN.py` | `src/data_pipeline/pipeline_orchestrator/Snakefile` | target wired through analysis-ready |
| Build01 images | `steps/run_build01.py` | rules `build_stitched_images_yx1*` | ✅ ported |
| Build02 segment | `steps/run_build02.py`, `run_sam2.py` | `segment_and_track_per_well`, `merge_segmentation_tracking` | ✅ ported (per-well) |
| Build03 snips | `steps/run_build03.py` | `run_snip_processing_per_well`, `merge_snip_manifests` | ✅ ported (per-well) |
| Build04 QC | `steps/run_build04.py` | the 13 `compute_*_qc` / `compute_*` + `consolidate_*` rules | ✅ ported; some QC rules read merged contract (the "merge-wall" — see findings doc Scope 5) |
| Build05 training snips | `steps/run_build05.py`, `vae/auxiliary_scripts/make_training_key.py` | **not ported** (training; deferred) | ❌ future |
| **Build06 embeddings** | `steps/run_build06.py` → `services/gen_embeddings.py` → `services/legacy_model_utils.py` | **not ported** → `src/data_pipeline/embeddings/` is **empty** | ❌ **active frontier** (see `target/model_input_handoff_contract.md`) |
| Analysis-ready | `analysis_ready/assemble*.py` (legacy refs) | `assemble_analysis_ready` rule + `src/data_pipeline/analysis_ready/assemble.py` | ✅ ported |

---

## 3. The files that matter (where to look)

**Target pipeline (the refactor):**
- `src/data_pipeline/pipeline_orchestrator/Snakefile` — **the whole DAG**. Read top to bottom.
- `src/data_pipeline/pipeline_orchestrator/config.yaml` — science knobs. Note: **interpreter is hardcoded at Snakefile:23**, not in config (Scope 3 fixes this).
- `src/data_pipeline/snip_processing/run_per_well.py` + `process_snips.py` — produces the model-input JPGs + `snip_manifest.parquet`.
- `src/data_pipeline/analysis_ready/assemble.py` + `schemas/analysis_ready.py` — the snip_id-keyed biology/QC join (`predicted_stage_hpf`, `genotype`, `treatment`, `use_snip`).
- `src/data_pipeline/embeddings/` — **empty**; where build_06 lands.

**Legacy model/inference stack (reuse, don't rewrite):**
- `src/run_morphseq_pipeline/steps/run_build06.py` — the inference orchestration to mirror.
- `src/run_morphseq_pipeline/services/gen_embeddings.py` — `generate_latents_with_repo_images` (the encoder loop).
- `src/run_morphseq_pipeline/services/legacy_model_utils.py` — **the sub-environment switch** (`conda run -n mseq_pipeline_py3.9`).
- `src/analyze/analysis_utils.py::extract_embeddings_legacy` — encoder → `snip_id, z_mu_*` DataFrame.

**Model + data-loading contract:**
- `src/data/dataset_configs.py` — `EvalDataConfig` / `_SnipDataset` (globs `training_data/bf_embryo_snips/<exp>/*.jpg`).
- `src/core/functions/dataset_utils.py` — legacy `ImageFolder`-based training datasets + `seq_key_dict` (`pert_id_vec`, `e_id_vec`, `age_hpf_vec`) — **training only**.
- `src/vae/auxiliary_scripts/make_training_key.py::make_seq_key` — where `pert_id = factorize(short_pert_name)` is built (`short_pert_name` = genotype×treatment composite).

---

## 4. Cross-cutting concerns the review must keep consistent

1. **Identifiers (Scope 1).** `well_id`/`image_id`/`snip_id` are minted inline as
   f-strings today; `src/data_pipeline/identifiers/` is empty. `snip_id`
   invariant: `f"{embryo_id}_{channel_id}_f{frame_index:04d}"`. This is the join
   key for everything (manifest ⋈ analysis_ready ⋈ latents). **DO NEXT** per the
   target.
2. **Config vs. environment (Scope 3).** `config.yaml` = science (committed);
   `env.yaml` = machine (`python`, `device`, `input_root`/`output_root`/
   `models_root`) — gitignored, **does not exist yet**. The build_06 model env
   (`mseq_pipeline_py3.9`) belongs in `env.yaml.runtime.model_python_env`, **not**
   a `config.yaml` block. (`target/model_input_handoff_contract.md` §6.5 was reconciled
   to this on 2026-06-05.)
3. **Per-well grain + the merge-wall.** Features/QC are one copy-paste template
   (13 near-identical rules); a few QC rules read the *merged* contract instead of
   per-well shards (findings doc Scope 5). No algorithm changes — rewiring only.
4. **`snip_id` everywhere, never sort-order index.** The legacy `ImageFolder`
   path keyed metadata by file sort order; the target keys by `snip_id`. Any new
   loader must preserve this.

---

## 5. Active frontier: the model/embedding step (build_06)

Spec: `target/model_input_handoff_contract.md`. In one breath:
- **Inference only this pass.** Encode snips → `snip_id, z_mu_*`. No `pert_id`,
  no `metric_array`, no splits (those are training, deferred).
- **Image seam:** manifest `processed_snip_path` is source of truth; symlink a
  legacy `bf_embryo_snips/<exp>/` *view* so existing `EvalDataConfig` works.
- **Reuse** `gen_embeddings.py` encoder loop + `legacy_model_utils` sub-env
  switch; the only new piece is sourcing images from the manifest.
- **Output:** `data_pipeline_output/embeddings/<exp>/morph_latents_<exp>.csv`.
- Open items: model location vs `models_root`, `use_snip` gating at inference,
  confirm `mseq_pipeline_py3.9` env name. (Contract §8.)

---

## 6. Quick verification commands (ground truth on disk)

```bash
# What the Snakefile actually calls its contracts (NOT frame_manifest):
grep -c "frame_contract\|stitched_inventory" src/data_pipeline/pipeline_orchestrator/Snakefile

# Confirm the model/embedding step is unbuilt:
ls -la src/data_pipeline/embeddings/        # only __init__.py (0 bytes)

# Confirm identifiers package is empty (Scope 1 not started):
cat src/data_pipeline/identifiers/__init__.py

# The legacy sub-environment switch:
grep -n "conda', 'run'\|mseq_pipeline_py3.9" src/run_morphseq_pipeline/services/legacy_model_utils.py
```
