# Refactor-010: Standardize Morphological Embedding Generation (Build06)

**Created**: 2025-09-01  
**Status**: Proposal  
**Depends On**: Refactor-009 SAM2 Pipeline Validation

## **Executive Summary**
- Goal: Centralize how we generate and publish morphological embeddings and provide a canonical pipeline artifact (df03) that merges df02 with z_mu_*.
- Decision: Adopt the legacy latent store as the source of truth for per‑experiment embeddings and add a Build06 step to standardize generation + merging.
- Model: Use legacy model `models/legacy/20241107_ds_sweep01_optimum` (AutoModel‑compatible).
- Outputs: 
  - Pipeline artifact: `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv` (df02 + z_mu_* by snip_id)
  - Per‑experiment copies for analysis users under central data root: `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/df03_{experiment}.csv` (non‑destructive)
  - Training‑run join (optional): `<root>/training_data/<train_name>/embryo_metadata_with_embeddings.csv`

## **Background & Problem**
- Today embeddings are generated via:
  - Ad‑hoc assessment scripts (`assess_image_set.py`, `assess_vae_results.py`) using `final_model/` folders
  - Batch analysis path (`analysis_utils.calculate_morph_embeddings`) writing per‑experiment `morph_latents_{exp}.csv` under `analysis/latent_embeddings/legacy/<model_name>/`
- There is no standardized pipeline step to merge embeddings into df02. Downstream users rely on notebooks or ad‑hoc merges.

## **Scope (This Refactor)**
- Add a Build06 step that:
  - Ensures latents exist for experiments present in df02 (optionally generates missing via legacy batch path)
  - Aggregates per‑experiment latents and merges them into df02 to produce a canonical df03
  - Optionally writes a per‑experiment df03 under the central data root for analysis users
  - Provides clear CLI, safety checks, and non‑destructive defaults

## **Source of Truth & Paths**
- Legacy model (for embedding):
  - `<DATA_ROOT>/models/legacy/20241107_ds_sweep01_optimum`
- Per‑experiment latent tables (reference and/or generation target):
  - `<DATA_ROOT>/analysis/latent_embeddings/legacy/20241107_ds_sweep01_optimum/morph_latents_{experiment}.csv` 
- Pipeline inputs/outputs:
  - df02 (input): `<root>/metadata/combined_metadata_files/embryo_metadata_df02.csv`
  - df03 (output): `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv`
  - Training metadata (optional join): `<root>/training_data/<train_name>/embryo_metadata_df_train.csv`
- Analysis copies (optional, non‑destructive):
  - `<DATA_ROOT>/metadata/metadata_n_embeddings/20241107_ds_sweep01_optimum/df03_{experiment}.csv`

## **Design**
- Inputs:
  - `--root`: pipeline root containing df02
  - `--data-root`: central data root (default via `MORPHSEQ_DATA_ROOT`)
  - `--model-name`: default `20241107_ds_sweep01_optimum`
  - `--experiments`: optional explicit list; otherwise inferred from df02 `experiment_date`
  - `--generate-missing-latents`: ensure missing `morph_latents_{exp}.csv` are created using legacy batch path
  - `--train-name`: optional; if present, write joined train metadata with embeddings
  - `--dry-run`: list planned actions without writing
  - `--overwrite`: allow overwriting df03 or analysis copies; default is non‑destructive
- Algorithm:
  1) Resolve df02 and experiment list
  2) For each experiment, check `<DATA_ROOT>/analysis/latent_embeddings/legacy/<model_name>/morph_latents_{exp}.csv`
     - If missing and `--generate-missing-latents`, call `calculate_morph_embeddings(data_root, model_name, model_class="legacy", experiments=[exp])`
  3) Load all per‑experiment latents; select columns `[snip_id] + z_mu_*` (and `z_mu_b_*`/`z_mu_n_*` if present)
  4) Normalize `snip_id` stems to match Build05 naming
  5) Merge df02 with combined latents on `snip_id` → df03
  6) Write df03 to `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv` (respect `--overwrite`)
  7) If `--train-name` provided and train metadata exists, also write `<root>/training_data/<train_name>/embryo_metadata_with_embeddings.csv`
  8) If `--export-analysis-copies`, write per‑experiment df03 copies to `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/df03_{exp}.csv` (respect `--overwrite`)

## **CLI Sketch**
- Build06 entry (proposed):
  - `python -m src.run_morphseq_pipeline.cli build06 \
      --root <root> \
      --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
      --model-name 20241107_ds_sweep01_optimum \
      --generate-missing-latents \
      --train-name <optional> \
      --export-analysis-copies \
      [--experiments 20250612_30hpf_ctrl_atf6 20240626] \
      [--dry-run] [--overwrite]`

## **Safety & Validation**
- Non‑destructive by default; writing df03 or analysis copies requires `--overwrite` if files exist
- Coverage report: % of df02 rows with a matched embedding by `snip_id` (warn if < 95%)
- Hygiene: verify z columns are finite (no NaN/Inf)
- Determinism: legacy embedding path runs models in eval mode; matching to existing `morph_latents_{exp}.csv` ensures stable results
- Clear logs for: missing latents, generated latents, skipped writes, and path summaries

## **Testing Strategy**
- Unit (simulate wiring): create tmp df02 + tiny latent CSV; assert df03 columns and coverage
- Integration (reference compare): load an existing latent file, run Build06 with `--dry-run` then real run (non‑overwrite), assert outputs and schema
- Negative: missing latent for an experiment without `--generate-missing-latents` → graceful warning and partial merge

## **Notes on Current State & Improvements**
- Current generation is scattered across assessment scripts and analysis utilities; this refactor centralizes merging and clearly documents the latent source of truth (legacy store)
- Improvements:
  - Optional: cache a manifest of available `morph_latents_{exp}.csv`
  - Optional: emit a metadata sidecar (JSON) describing model, commit hash, and run parameters for reproducibility

## **Acceptance Criteria**
- Build06 produces `embryo_metadata_df03.csv` with z columns and >=95% join coverage on representative datasets
- Running with `--export-analysis-copies` creates per‑experiment df03 files under `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/` without overwrites unless `--overwrite` is set
- Clear CLI UX and logs; dry‑run mode lists planned actions

## **Example Commands**
- Minimal merge (no latent generation):
  - `python -m src.run_morphseq_pipeline.cli build06 --root <root> --data-root <DATA_ROOT> --model-name 20241107_ds_sweep01_optimum`
- Generate missing latents then merge:
  - `python -m src.run_morphseq_pipeline.cli build06 --root <root> --data-root <DATA_ROOT> --model-name 20241107_ds_sweep01_optimum --generate-missing-latents`
- Export per‑experiment df03 copies (safe):
  - `python -m src.run_morphseq_pipeline.cli build06 --root <root> --data-root <DATA_ROOT> --model-name 20241107_ds_sweep01_optimum --export-analysis-copies`

---

## **Implementation Plan: gen_embeddings.py (Pipeline‑First Service)**

Module: `src/run_morphseq_pipeline/services/gen_embeddings.py`

Purpose: Centralize embedding ingestion/generation and df02 merge under clear, testable functions used by Build06. Keeps existing assessment scripts intact.

Proposed API (clean names):
- `resolve_model_dir(data_root, model_name) -> Path`
  - Resolves `<DATA_ROOT>/models/legacy/<model_name>`; accepts direct or `final_model/` layout; validates `model_config.json`.
- `ensure_latents_for_experiments(data_root, model_name, experiments, generate_missing=False, logger=None) -> dict[exp, Path]`
  - Ensures `<DATA_ROOT>/analysis/latent_embeddings/legacy/<model_name>/morph_latents_{exp}.csv` exists for each experiment; optionally generates via `analysis_utils.calculate_morph_embeddings`.
- `load_latents(latent_paths: dict[str, Path], z_schema: str = "auto") -> pd.DataFrame`
  - Reads per‑experiment latent CSVs, keeps `snip_id` + z columns per schema policy, validates finite values, de‑dups by `snip_id`.
- `merge_df02_with_embeddings(root, latents_df, overwrite=False, out_name="embryo_metadata_df03.csv") -> Path`
  - Joins df02 with embeddings on `snip_id`; writes df03 alongside df02; reports join coverage.
- `merge_train_with_embeddings(root, train_name, latents_df, overwrite=False) -> Optional[Path]`
  - Joins `training_data/<train_name>/embryo_metadata_df_train.csv` with embeddings; writes `embryo_metadata_with_embeddings.csv` if train metadata exists.
- `export_df03_copies_by_experiment(df03, data_root, model_name, overwrite=False) -> None`
  - Writes per‑experiment `df03_{experiment}.csv` under `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/` (non‑destructive by default).
- `build_df03_with_embeddings(root, data_root, model_name, experiments=None, generate_missing=False, export_analysis=False, train_name=None, overwrite=False, dry_run=False, z_schema="auto") -> Path`
  - One‑shot orchestrator used by Build06; returns df03 path.

Schema policy (z_schema):
- `auto` (default): keep both flat `z_mu_XX` and bio/nuisance `z_mu_b_XX` / `z_mu_n_XX` when present (requested standard).
- `flat`: keep only `z_mu_XX` columns.
- `bio_nuisance`: keep only `z_mu_b_XX` / `z_mu_n_XX` when present, else fall back to any `z_mu_XX`.

Join key:
- `snip_id` — exact filename stem of Build05 snips (no extension). We will normalize stems as needed to match Build05 naming.

Safety & UX:
- Non‑destructive writes (`--overwrite` required to replace existing files).
- `--dry-run` prints actions (experiments found, files to read/write, missing latents) and exits.
- Coverage report printed after merge; hygiene check enforces finite numeric z’s.

Build06 integration:
- Build06 invokes `build_df03_with_embeddings(...)` with `z_schema="auto"` by default.

Testing (unit‑first):
- Simulated latent CSVs + tiny df02: assert df03 join coverage, z schema presence, and path handling; negative case for missing latents without generation.

---

## **Analysis & Rationale (for audit)**

- Current embedding generation is split across ad‑hoc assessment scripts and analysis utilities. The pipeline lacks a canonical df02+z merge.
- We standardize around the legacy per‑experiment latent store as source of truth and expose a pipeline‑first service (`gen_embeddings.py`) so Build06 can be thin and deterministic.
- Keeping both flat and bio/nuisance z’s by default (`z_schema="auto"`) preserves information for downstream tasks while enabling later simplifications.
- This reduces duplication, keeps legacy scripts for research workflows, and provides a single, safe, documented path for pipeline outputs (df03).

---

## **Open Questions (.insight)**

- .insight: Should Build06 always prefer existing legacy latents, and only generate when `--generate-missing-latents` is set (default yes)?
- .insight: Do we also want an option to embed directly from a provided `--model-dir` (bypassing the legacy store) as a future enhancement?
- .insight: For z schema, keep `auto` as default (flat + bio/nuisance) — do we need a downstream consumer that prefers a single flattened schema?
- .insight: Do we want a metadata sidecar (JSON) alongside df03 capturing `model_name`, data_root, timestamp, git commit, and coverage stats for reproducibility?
- .insight: Should per‑experiment df03 copies be enabled by default, or remain opt‑in via `--export-analysis-copies`?
