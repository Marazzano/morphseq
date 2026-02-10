# Streamline-Snakemake Refactor Documentation

**Organization Date:** 2025-11-06
**Status:** Current docs now reflect `stitched_image_index.csv` + `frame_manifest.csv` architecture

---

## 2026-02-10 - Addendum, highlighting what we need to change in the original doc
- Keep all downstream (segmentation, snip processing, features, QC, embeddings, analysis-ready) logic as-is.
- Limit conceptual updates to ingest and pre-segmentation handoff only.
- Treat ingest as scope-first (YX1 and Keyence stay separate through extraction and mapping).
- Keep `materialize_stitched_images_*` as the stage name; scope builders implement the behavior.
- Keep frame-level handoff on `stitched_image_index.csv` + `frame_manifest.csv`.
- Use `channel_id`, preserve `channel_raw_name`, use `temperature_c`, and require `micrometers_per_pixel` in frame-level metadata.

---

## Quick TL;DR (For Scientists)
If you only remember one thing:

1. `stitched_image_index.csv` tells you what stitched images were materialized.
2. `frame_manifest.csv` is the canonical frame table that segmentation should trust.
3. `embryo_id` starts at segmentation, not during metadata ingest.

---

## Core Documentation (Current)

### 1. `processing_files_pipeline_structure_and_plan.md`
**Architecture spec**
- Why this design changed
- Full module structure
- Contract definitions and validation checks
- Scientist-friendly debugging flow

### 2. `snakemake_rules_data_flow.md`
**Rule-by-rule implementation spec**
- Exact stage flow through `frame_manifest.csv`
- Rule purposes and I/O expectations
- Naming conventions (`channel_id`, `channel_raw_name`, `temperature_c`)

### 3. `data_ouput_strcutre.md`
**Output file and directory spec**
- Canonical output tree
- Contract files and required columns
- Practical checklist for experiment validation

### 4. `DATA_INGESTION_AND_TESTING_STRATEGY.md`
**Data setup and testing guidance**
- Symlink strategy
- Test dataset guidance
- Stepwise validation approach

---

## Recommended Reading Order

1. `processing_files_pipeline_structure_and_plan.md`
2. `snakemake_rules_data_flow.md`
3. `data_ouput_strcutre.md`
4. `DATA_INGESTION_AND_TESTING_STRATEGY.md`

---

## Current Pre-Segmentation Flow

1. Normalize plate metadata.
2. Extract scope metadata.
3. Run scope-specific series mapping.
4. Apply mapping to produce `scope_metadata_mapped.csv`.
5. Materialize stitched images (scope-specific).
6. Emit and validate `stitched_image_index.csv`.
7. Build and validate `frame_manifest.csv`.
8. Start segmentation using `frame_manifest.csv`.

---

## Notes on Legacy Documents

- `logs/` and `_Archive/` retain historical planning and review context.
- Historical references to `experiment_image_manifest.json` are deprecated.
- Use the three core docs above for current implementation decisions.
