# Handoff — Segmentation & Tracking: paths.py wiring + video_generation collapse

**Branch:** `mdcolon/20260222_docs_snakemake_remake`
**Date:** 2026-06-07
**Prior session:** Scope 2 global well_id migration (Sections B, D, E partial — see below)
**This handoff:** what was done, what remains, and the exact next job.

---

## What was done this session

### Section B — Schemas ✅
- Removed `video_id` from 6 schema constants in `segmentation.py`:
  `SEGMENTATION_TRACKING`, `FRAME_DETECTIONS`, `SEED_SELECTION`, `TRACK_INSTANCES`,
  `MASK_RLE`, `V2` (via `SEGMENTATION_TRACKING_V2`).
- Removed `video_id` from `UNIQUE_KEY_SEED_SELECTION`.
- Dropped `experiment_id` from `UNIQUE_KEY_FRAME_CONTRACT` in `frame_contract.py`
  (`well_id` is now globally unique; the key is `(well_id, channel_id, time_int)`).

### Section D — Landmines ✅
- `segmentation_and_tracking/pipelines/segmentation_and_tracking.py`:
  - Per-well shard dir now uses global `well_id` directly (`per_well/20250912_A01/`).
  - Manifest filter replaced: two-candidate `isin({local, global})` → `well_id == requested_well_id`.
- `merge_segmentation_and_tracking_contracts.py`: removed `startswith(exp_)` pre-filter and
  `_slug()` function; views symlinks now keyed on full `well_id`.
- `snip_processing/pipelines/merge_snip_manifests.py`: same — removed `startswith` + `_slug`.
- `segmentation/grounded_sam2/csv_formatter.py` line 206: `well_id = video_id.split("_")[-1]`
  → `well_id = video_id` (it IS the global well_id).

### Section E — video_id collapse (partial) ✅
Within `segmentation_and_tracking/` (the non-video_generation half):
- `raw_types.py`: removed `video_id` field from `SeedSelection`.
- `ingestors/gdino_ingestor.py`: removed `video_id` param from `ingest_seed_selection`.
- `normalizers/normalize_frame_detections.py`: removed `video_id` param + output column.
- `normalizers/normalize_seed_selection.py`: removed `video_id` output column.
- `normalizers/normalize_track_instances.py`: removed `video_id` param + column;
  changed `well_index: int` → `well_index: str` (spec: `well_index = "A01"`, not int).
- `normalizers/normalize_mask_rle.py`: removed `video_id` param + column.
- `normalizers/normalize_segmentation_tracking.py`: `well_index: int` → `well_index: str`.
- `segmentation_and_tracking.py` pipeline: removed all `video_id` construction and threading;
  `well_index` now derived via `split_well_id(canonical_well_id)` (string `"A01"`).
- MP4 output filenames now use full `well_id` (`20250912_A01_raw.mp4`).

---

## What remains

### 1. `video_generation/` subsystem — Section E audit (load-bearing, do carefully)

Six files still use `video_id` with `re.match`/`rsplit` parsing that must become `split_well_id`:
- `segmentation/video_generation/video_generator.py`
- `segmentation/video_generation/render_eval_video.py`
- `segmentation/video_generation/results_adapter.py`
- `segmentation/video_generation/service.py`
- `segmentation/video_generation/models.py`
- `segmentation/video_generation/overlay_manager.py`

Pattern to replace in each:
- `re.match(r"^(.+)_([A-H][0-9]{2})$", video_id)` → `split_well_id(well_id)`
- `video_id.rsplit("_", 1)` → `split_well_id(well_id)`
- param rename `video_id` → `well_id` throughout

**Audit each file before editing** — the parsing is used for both experiment_id extraction and
well_index extraction. `split_well_id` returns `(experiment_id, well_index)` which covers both.

### 2. `paths.py` wiring — the big structural gap ⚠️

**Per `pipeline_file_philosophy.md` constraint 1:** no artifact path is ever typed as a raw
string — all via `orchestration/paths.py`. Today, `segmentation_and_tracking.py` and the merge
scripts hardcode paths like `output_root / "segmentation_and_tracking" / exp / "per_well" / well_id`.
These need to go through `artifact_path` / `step_dir` from `PIPELINE_STEPS`.

**What to do:**
1. Add back-half step rows to `PIPELINE_STEPS` in `orchestration/paths.py`:
   - `"segmentation_tracking"` — `PER_WELL_THEN_MERGE`, stage `"segmentation_and_tracking"`,
     artifacts: `per_well`: `"{well_id}_segmentation_tracking.csv"`, `merged`: `"{exp}_segmentation_tracking.csv"`
   - `"frame_detections"`, `"seed_selection"`, `"track_instances"`, `"mask_rle"` — parquet shards,
     same stage, `PER_WELL_THEN_MERGE`
   - `"snip_manifest"` — stage `"processed_snips"`, `PER_WELL_THEN_MERGE`
   - (other back-half steps: aux masks, features, stage predictions — add as you go)
2. Replace hardcoded path construction in:
   - `segmentation_and_tracking/pipelines/segmentation_and_tracking.py` (shard_dir, contracts_dir, etc.)
   - `segmentation_and_tracking/pipelines/merge_segmentation_and_tracking_contracts.py`
   - `snip_processing/pipelines/merge_snip_manifests.py`
3. Update the Snakefile `output:` declarations to use `artifact_path` / `validated_path` from
   `paths.py` for these steps (same pattern as the front-end rules already use).

**Reference:** `pipeline_file_philosophy.md` + `orchestration/paths.py` + `orchestration/well_runner.py`
are the worked examples. The "add a stage" recipe (§ THE "ADD A STAGE" RECIPE) is the template.

### 3. Section F — docs ✅ (do last, after paths wiring)
- `well_id_global_migration_map.md`: update STATUS block to reflect B, D, E done.
- `well_id_throughline_refactor_plan.md`: update stale banner; record DECISION 1 kept.
- `identifier_and_wildcard_contract.md`: flip worked examples to global `well_id`.

---

## Smoke test to verify paths wiring is correct

After adding back-half steps to `paths.py` and wiring the Snakefile:

```bash
# 1. DAG parses cleanly (no missing output paths)
cd src/data_pipeline/pipeline_orchestrator
PYTHONPATH=/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/src \
  /net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/conda run \
    -n segmentation_grounded_sam --no-capture-output \
    snakemake -n --cores 1

# 2. Front-end smoke run still green (no GPU needed)
snakemake --cores 4 -p \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output/experiment_metadata/20250912/discovered_wells.txt

# 3. No remaining landmine patterns
grep -rn 'split("_")' src/data_pipeline/ | grep -v '#\|__pycache__'
grep -rn 'startswith.*exp.*_' src/data_pipeline/ | grep -v '#\|__pycache__'
grep -rn 'video_id' src/data_pipeline/ | grep -v '__pycache__\|video_generation'
```

Expected after full wiring: all three greps return empty (or only comments / the not-yet-done
`video_generation/` subsystem).
