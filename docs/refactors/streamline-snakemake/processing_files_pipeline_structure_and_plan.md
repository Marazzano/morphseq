# MorphSeq Pipeline Refactor: Structure and Implementation Plan

**Status:** Current Architecture Spec
**Audience:** Scientists and developers implementing the new pipeline
**Last Updated:** 2026-02-10

## TL;DR
The pipeline now uses two CSV handoff contracts instead of `experiment_image_manifest.json`:

1. `stitched_image_index.csv`
2. `frame_manifest.csv`

Why this change:
- It avoids fragile filename parsing.
- It keeps microscope-specific logic in microscope-specific modules.
- It gives downstream stages one clear, validated frame-level source of truth.

In practical terms:
- YX1 and Keyence each build/symlink stitched images in their own way.
- While they do that, they also report rows for `stitched_image_index.csv`.
- A shared join step builds `frame_manifest.csv` from stitched index + scope metadata + plate metadata.
- Segmentation consumes `frame_manifest.csv`.

Important boundary:
- `embryo_id` is not available before segmentation and must not appear in pre-segmentation contracts.

---

## Design Choices and Why

### 1) CSV contracts over JSON manifest
- CSV is easier to validate, diff, and inspect during troubleshooting.
- The old JSON path required extra conversion logic and was not yet deeply integrated.

### 2) Reporter pattern for stitched images
- Materializers write/symlink images and emit index rows at the same time.
- No generic crawler guesses metadata from filenames.
- This prevents silent drift when naming templates change.

### 3) Scope-specific behavior, shared schema
- YX1 and Keyence remain independent where they should be.
- Contracts are shared so downstream logic is microscope-agnostic.

### 4) Naming consistency
- Use `channel_id` for normalized channels (`BF`, `GFP`, etc.).
- Use `channel_raw_name` for microscope-native labels.
- Use `temperature_c` (not mixed `temperature`/`temperature_c` variants).

### 5) Calibration fidelity
- `micrometers_per_pixel` is required in `frame_manifest.csv`.
- Downstream feature extraction and QC depend on this being present.

---

## Pipeline Tasks Overview

```
PHASE 1: METADATA INGEST
  Plate metadata -> plate_metadata.csv
  Scope metadata -> scope_metadata_raw.csv
  Scope-specific mapping -> series_well_mapping.csv
  Mapping applied -> scope_metadata_mapped.csv

PHASE 2: IMAGE BUILD + FRAME CONTRACT
  Scope-specific image build/symlink -> stitched_ff_images/
  Reporter rows from builders -> stitched_image_index.csv
  Shared validation -> stitched_image_index.csv (validated)
  Shared join + validation -> frame_manifest.csv (canonical)

PHASE 3+: DOWNSTREAM PROCESSING
  Segmentation consumes frame_manifest.csv
  Segmentation -> snip processing -> features -> QC -> embeddings -> analysis-ready
```

---

## Finalized Directory Structure

```
src/data_pipeline/

├── schemas/
│   ├── __init__.py
│   ├── channel_normalization.py
│   ├── plate_metadata.py
│   ├── scope_metadata.py
│   ├── scope_and_plate_metadata.py
│   ├── stitched_image_index.py          # NEW
│   ├── frame_manifest.py                # NEW
│   ├── segmentation.py
│   ├── snip_processing.py
│   ├── features.py
│   ├── quality_control.py
│   └── analysis_ready.py
│
├── metadata_ingest/
│   ├── plate/
│   │   └── plate_processing.py
│   ├── scope/                           # Scope-first ingest pipelines
│   │   ├── yx1/
│   │   │   ├── extract_scope_metadata.py
│   │   │   ├── map_series_to_wells.py
│   │   │   └── apply_series_mapping.py
│   │   ├── keyence/
│   │   │   ├── extract_scope_metadata.py
│   │   │   ├── map_series_to_wells.py
│   │   │   └── apply_series_mapping.py
│   │   └── shared/
│   │       └── align_scope_plate.py
│   └── frame_manifest/                  # Shared handoff into segmentation
│       ├── build_frame_manifest.py
│       └── validate_frame_manifest.py
│
├── image_building/
│   ├── scope/                           # Scope-first image pipelines
│   │   ├── yx1/
│   │   │   └── stitched_ff_builder.py   # Updated: reporter pattern
│   │   └── keyence/
│   │       └── stitched_ff_builder.py   # Updated: reporter pattern
│   └── handoff/                         # NEW
│       ├── io.py
│       └── validate_stitched_index.py
│
├── segmentation/
│   └── ...
├── snip_processing/
│   └── ...
├── feature_extraction/
│   └── ...
├── quality_control/
│   └── ...
├── embeddings/
│   └── ...
├── analysis_ready/
│   └── ...
└── pipeline_orchestrator/
    ├── Snakefile
    └── config.yaml
```

Note:
- This layout intentionally emphasizes separate scope pipelines (YX1 and Keyence).
- Shared logic is limited to handoff/validation steps.

---

## Agreed Implementation Touch List (Scope-First)

This is the current agreed migration set for code changes.

### Modify Existing Files
1. `src/data_pipeline/metadata_ingest/scope/yx1/extract_scope_metadata.py`
2. `src/data_pipeline/metadata_ingest/scope/keyence/extract_scope_metadata.py`
3. `src/data_pipeline/metadata_ingest/scope/yx1/map_series_to_wells.py`
4. `src/data_pipeline/metadata_ingest/scope/keyence/map_series_to_wells.py`
5. `src/data_pipeline/metadata_ingest/scope/yx1/apply_series_mapping.py`
6. `src/data_pipeline/metadata_ingest/scope/keyence/apply_series_mapping.py`
7. `src/data_pipeline/image_building/scope/yx1/stitched_ff_builder.py`
8. `src/data_pipeline/image_building/scope/keyence/stitched_ff_builder.py`
9. `src/data_pipeline/pipeline_orchestrator/config.yaml`

### Add New Files
1. `src/data_pipeline/schemas/stitched_image_index.py`
2. `src/data_pipeline/schemas/frame_manifest.py`
3. `src/data_pipeline/image_building/handoff/io.py`
4. `src/data_pipeline/image_building/handoff/validate_stitched_index.py`
5. `src/data_pipeline/metadata_ingest/frame_manifest/build_frame_manifest.py`

### Remove Legacy Files
1. `src/data_pipeline/metadata_ingest/manifests/generate_image_manifest.py`
2. `src/data_pipeline/schemas/image_manifest.py`
3. `src/data_pipeline/metadata_ingest/manifests/__init__.py` (optional)

---

## Canonical Contracts

### `stitched_image_index.csv`
Produced during image materialization (reporter pattern).

Required columns:
- `experiment_id`
- `microscope_id`
- `well_id`
- `well_index`
- `channel_id`
- `time_int`
- `frame_index`
- `image_id`
- `stitched_image_path`
- `materialization_status` (`written`, `symlinked`, `skipped`, `failed`)
- `source_artifact_path`
- `source_artifact_kind` (`nd2_container`, `tile_dir`, `tile_file`)

Optional columns:
- `image_width_px`
- `image_height_px`

### `frame_manifest.csv`
Canonical frame-level input for segmentation and downstream phases.

Required columns:
- `experiment_id`
- `microscope_id`
- `well_id`
- `well_index`
- `channel_id`
- `channel_raw_name`
- `time_int`
- `frame_index`
- `image_id`
- `stitched_image_path`
- `micrometers_per_pixel`
- `frame_interval_s`
- `absolute_start_time`
- `experiment_time_s`
- `image_width_px`
- `image_height_px`
- `objective_magnification`
- `genotype`
- `treatment`
- `medium`
- `temperature_c`
- `start_age_hpf`
- `embryos_per_well`

Uniqueness key for both contracts:
- `(experiment_id, well_id, channel_id, time_int)`

Frame semantics:
- `time_int` is the acquisition time key.
- `frame_index` is contiguous 0-based order after sorting by `time_int` per `(experiment_id, well_id, channel_id)`.

---

## Revised Rule Flow

1. Normalize plate metadata -> `plate_metadata.csv`
2. Extract scope metadata -> `scope_metadata_raw.csv`
3. Scope-specific mapping -> `series_well_mapping.csv`
4. Apply mapping -> `scope_metadata_mapped.csv`
5. Scope-specific stitched image materialization + emit `stitched_image_index.csv`
6. Validate stitched index
7. Build `frame_manifest.csv` by joining:
   - `scope_metadata_mapped.csv`
   - `stitched_image_index.csv`
   - `plate_metadata.csv`

Then segmentation consumes `frame_manifest.csv`.

---

## Validation and Acceptance Checks

1. No duplicate frame keys in `stitched_image_index.csv`.
2. No duplicate frame keys in `frame_manifest.csv`.
3. Every `stitched_image_path` in `frame_manifest.csv` exists.
4. Every row has non-null `micrometers_per_pixel`.
5. Every row has non-null `temperature_c` and `start_age_hpf` when required.
6. No `embryo_id` appears before segmentation outputs.

---

## Files Removed by This Architecture Update

- `src/data_pipeline/metadata_ingest/manifests/generate_image_manifest.py`
- `src/data_pipeline/schemas/image_manifest.py`
- `src/data_pipeline/metadata_ingest/manifests/__init__.py` (optional cleanup)

These are safe to remove once `frame_manifest.csv` path is wired in Snakefile.

---

## Guidance for Future Scientists

Use this mental model:
- If the question is "What image files were produced and from what source?" read `stitched_image_index.csv`.
- If the question is "What frame-level table should I trust for analysis/segmentation inputs?" read `frame_manifest.csv`.
- If the question is "Where do embryos start?" that begins at segmentation, not metadata ingest.

When debugging:
1. Check `plate_metadata.csv`.
2. Check `scope_metadata_mapped.csv`.
3. Check `stitched_image_index.csv` for missing/failed frames.
4. Check `frame_manifest.csv` for join completeness and calibration columns.
