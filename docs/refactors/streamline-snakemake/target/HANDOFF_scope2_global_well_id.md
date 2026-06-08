# Handoff — Scope 2: Global `well_id` Migration

**Branch:** `mdcolon/20260222_docs_snakemake_remake`
**Date:** 2026-06-07
**Prior session:** YX1 Phase 1 recompose + two-well smoke run (DONE — see `recompose_yx1_front_end.md`).
**This handoff:** everything needed to implement Scope 2 and leave the next agent a clean landing.

---

## What Scope 2 is

Flip `well_id` from a **local** bare label (`A01`) to a **global** prefixed id
(`20240418_A01 = {experiment_id}_{well_index}`). This is the one semantic change that makes
`well_id` the right key for the per-well DAG: the well-runner, file paths, and every
downstream join all require a globally unique `well_id`.

**The grammar (Scope 1) is already done.** `shared/identifiers/constructors.py` already has:
```python
def build_well_id(experiment_id: str, well_index: str) -> str:
    return f"{sanitize_experiment_id(experiment_id)}_{str(well_index).strip()}"
```
Scope 2 is the **data and call-site migration** — making every piece of code and every stored
CSV emit and consume the global form.

---

## What is already done (do NOT redo)

From `well_id_global_migration_map.md` (STATUS section, 2026-06-05):

- **Section A (grammar):** `build_well_id(experiment_id, well_index)` is implemented with
  the 2-arg signature. `split_well_id`, `validate_well_id` implemented. 13 tests pass.
- **Section C (mint sites):** All metadata-ingest mint sites already pass both args:
  - `keyence/extract_scope_metadata.py:287` — `build_well_id(experiment_id, well_index)` ✅
  - `keyence/map_series_to_wells.py:179` — `build_well_id(experiment_id, well_index)` ✅
  - `scope/shared/apply_series_mapping.py:91` — `build_well_id(experiment_id, well_index)` ✅
  - `build_image_id` calls in `apply_series_mapping.py` and `keyence/extract_scope_metadata.py` use well_id-first form ✅
  - `plate/plate_processing.py:65` — **fixed 2026-06-07**: was `.map(build_well_id)` (single-arg), now `lambda w: build_well_id(experiment_id, w)` ✅
  - YX1's `extract_scope_metadata.py` was renamed/rewritten (Phase 1) — **never minted well_id** ✅
- **Smoke run verified (2026-06-07):** end-to-end pipeline for `20250912` produces
  `well_id = 20250912_A01` (global) in `scope_metadata_mapped.csv` and `discovered_wells.txt`.

---

## What remains (your job)

The full spec is in `well_id_global_migration_map.md`. Sections B, D, and E are NOT started.
Section F (Snakefile + docs) is minimal. Here is the ordered work:

### Section B — Schemas (9 files)

Update schema definitions to reflect that `well_id` is now semantically global.
Most are semantics-only (no column rename needed). The one structural change is `segmentation.py`:
delete `video_id` from 6 schema constants (it IS `well_id`).

Key files:
- `src/data_pipeline/schemas/scope_metadata.py` — semantics only (already updated for Phase 1)
- `src/data_pipeline/schemas/segmentation.py` — **delete `video_id` from** `SEGMENTATION_TRACKING`, `FRAME_DETECTIONS`, `SEED_SELECTION`, `TRACK_INSTANCES`, `MASK_RLE`, `V2`; UNIQUE_KEYs already use `well_id`
- `src/data_pipeline/schemas/plate_metadata.py` — confirm `well_id` global
- `src/data_pipeline/schemas/frame_contract.py` — note: the frame-contract rename to `frame_inventory` (Decision in `front_end_naming_and_frame_inventory_flow.md`) is a separate concern; fold it in if doing both at once, otherwise leave for later

### Section D — Defensive re-derivation landmines

These sites today defensively re-derive local vs. global `well_id` via string splitting.
Once `well_id` is globally unique everywhere, these patterns become wrong and must be deleted.

| File | Lines | Pattern | Fix |
|---|---|---|---|
| `segmentation_and_tracking/pipelines/segmentation_and_tracking.py` | 70–91 | `startswith(f"{exp}_")` + `split("_",1)[-1]` + `isin({local, global})` | filter `well_id == requested_well_id` (both global) |
| `segmentation_and_tracking/pipelines/segmentation_and_tracking.py` | 107, 153 | `well_slug = canonical_well_id.split("_")[-1]` | `_, well_index = split_well_id(well_id)` |
| `merge_segmentation_and_tracking_contracts.py` | 22, 29 | `startswith(f"{experiment}_")` + `split("_")[-1]` | shards already keyed on global `well_id` |
| `merge_snip_manifests.py` | 22, 28 | same | same |
| `grounded_sam2/csv_formatter.py` | 206 | `well_id = video_id.split("_")[-1]` | `well_id = video_id` (it IS global); use `split_well_id` for local |
| `compute_stage_predictions.py` | 71, 77 | `well_id == well` (local compare) | compare global well_id |
| `scope/shared/apply_series_mapping.py` | 105 | `isin(selected_wells_set)` | `selected_wells` must be global well_ids |

**The INVARIANT:** fix all of these in the same pass. A half-fixed state where some code emits
global `20240418_A01` and other code filters for local `A01` will silently drop all wells.

### Section E — video_id collapse (in-scope: `src/data_pipeline/` only)

`video_id` is already the global well_id (constructed as `f"{exp}_{well_slug}"`). Absorb it:

| Area | Files | Change |
|---|---|---|
| `segmentation/video_generation/` | `video_generator`, `render_eval_video`, `results_adapter`, `service`, `models`, `overlay_manager` | rename param `video_id → well_id`; `re.match(r"^(.+)_([A-H]\d{2})$", …)` + `rsplit("_",1)` → `split_well_id`. **Audit each — load-bearing parsing.** |
| `segmentation_and_tracking/` normalizers | `normalize_{seed_selection,track_instances,frame_detections,mask_rle}`, `gdino_ingestor`, `raw_types` | drop `video_id` field; thread `well_id` |
| `segmentation_and_tracking/pipelines/segmentation_and_tracking.py` | 108, 218, 226, 310, 327 | delete `video_id = f"{exp}_{slug}"`; use `well_id` directly |

**OUT OF SCOPE:** `src/build/build03A_process_images.py`, `src/run_morphseq_pipeline/combine_experiments_parts.py`, `src/build/pipeline_objects.py`, anything under `src/_Archive/`. Do not touch legacy build.

### Section F — Snakefile + docs (minimal)

- Snakefile: `well_index` appears only in a docstring on this branch — confirm no functional
  use; no change expected.
- `identifier_and_wildcard_contract.md`: flip worked examples to global `well_id`, drop the
  CURRENT-vs-TARGET banner's "not yet" framing.
- `well_id_global_migration_map.md`: update the STATUS block to reflect sections completed.

---

## Recommended order

```
1. Section B schemas — start with segmentation.py (video_id deletion), then others
2. Section D landmines — fix ALL of them in one pass (the INVARIANT)
3. Section E video_id collapse — follows D (video_id is already well_id once D is clean)
4. Section F — docs/Snakefile cleanup
5. Run tests: PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output python -m pytest src/ -x -q
```

Do NOT start Section D or E without first verifying the tests still pass after Section B.
Doing D+E as one atomic commit is safer than two half-states.

---

## Key files to read first

1. `well_id_global_migration_map.md` — the full spec with exact line numbers (the canonical reference for this work)
2. `well_id_throughline_refactor_plan.md` — the motivating design (Scope 2 = §"Scope 2" starting at line ~287)
3. `shared/identifiers/constructors.py` — the 2-arg `build_well_id` + `split_well_id` already implemented
4. `segmentation_and_tracking.py` — the heaviest single file in Sections D+E

---

## How to run the pipeline to verify

No GPU needed for the metadata stages. The command that ran the smoke run:
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/src/data_pipeline/pipeline_orchestrator
PYTHONPATH=/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/src \
  /net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/conda run \
    -n segmentation_grounded_sam --no-capture-output \
    snakemake --cores 4 -p \
    /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output/experiment_metadata/20250912/discovered_wells.txt
```

For the per-well segmentation stages (Sections D/E), those require GPU. Use qsub if needed:
see `src/run_morphseq_pipeline/run_experiment_manager_qsub.sh` for the qsub template.

---

## What success looks like

- `discovered_wells.txt` still lists `20250912_A01` (global ids) — smoke run still passes
- No `split("_")[-1]` or `startswith(f"{exp}_")` patterns remain in `src/data_pipeline/`
- `video_id` field is gone from all segmentation schema constants
- All tests pass
