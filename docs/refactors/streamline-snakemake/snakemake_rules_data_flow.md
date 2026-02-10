# MorphSeq Snakemake Rules and Data Flow

**Status:** Current Implementation Spec
**Audience:** Scientists and developers wiring/maintaining the pipeline
**Last Updated:** 2026-02-10

## TL;DR
Use this sequence for pre-segmentation data flow:

1. Plate metadata ingest
2. Scope metadata ingest (YX1/Keyence)
3. Scope-specific series mapping
4. Apply mapping
5. Scope-specific stitched image materialization + emit `stitched_image_index.csv`
6. Validate stitched index
7. Build `frame_manifest.csv` from:
   - `scope_metadata_mapped.csv`
   - `stitched_image_index.csv`
   - `plate_metadata.csv`

Downstream segmentation consumes `frame_manifest.csv`.

The old `experiment_image_manifest.json` flow is deprecated and removed.

---

## Rule Flow (High Level)

```
PHASE 1: METADATA
  normalize_plate_metadata
  extract_scope_metadata_yx1 | extract_scope_metadata_keyence
  map_series_to_wells_yx1 | map_series_to_wells_keyence
  apply_series_mapping_yx1 | apply_series_mapping_keyence

PHASE 2: IMAGES + FRAME CONTRACT
  materialize_stitched_images_yx1 | materialize_stitched_images_keyence
  validate_stitched_image_index
  build_frame_manifest

PHASE 3+
  segmentation_and_downstream (consumes frame_manifest.csv)
```

---

## Phase 1 Rules

### `rule normalize_plate_metadata`
**Input**
- `data_pipeline_output/inputs/plate_metadata/{experiment}_well_metadata.xlsx`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/plate_metadata.csv` `[VALIDATED]`

**Module**
- `metadata_ingest/plate/plate_processing.py`

**Purpose**
- Parse/normalize plate annotations.
- Ensure `temperature_c` and `start_age_hpf` are available for downstream joining.

---

### `rule extract_scope_metadata_yx1`
### `rule extract_scope_metadata_keyence`
**Input**
- `data_pipeline_output/inputs/raw_image_data/YX1/{experiment}/` or
- `data_pipeline_output/inputs/raw_image_data/Keyence/{experiment}/`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/scope_metadata_raw.csv` `[VALIDATED]`

**Modules**
- `metadata_ingest/scope/yx1/extract_scope_metadata.py`
- `metadata_ingest/scope/keyence/extract_scope_metadata.py`

**Purpose**
- Extract microscope timing, calibration, dimensions, and raw channel provenance.
- Keep this microscope-specific, but emit a shared column contract.

---

### `rule map_series_to_wells_yx1`
### `rule map_series_to_wells_keyence`
**Input**
- `plate_metadata.csv`
- `scope_metadata_raw.csv`
- (and raw Keyence path for Keyence mapping)

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/series_well_mapping.csv` `[VALIDATED]`
- `data_pipeline_output/experiment_metadata/{experiment}/series_well_mapping_provenance.json`

**Modules**
- `metadata_ingest/scope/yx1/map_series_to_wells.py`
- `metadata_ingest/scope/keyence/map_series_to_wells.py`

**Purpose**
- Resolve microscope series/position IDs into plate well IDs.
- Keep logic scope-specific.

---

### `rule apply_series_mapping_yx1`
### `rule apply_series_mapping_keyence`
**Input**
- `scope_metadata_raw.csv`
- `series_well_mapping.csv`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/scope_metadata_mapped.csv` `[VALIDATED]`

**Purpose**
- Produce final well-linked scope metadata used by all later joins.
- Modules:
  - `metadata_ingest/scope/yx1/apply_series_mapping.py`
  - `metadata_ingest/scope/keyence/apply_series_mapping.py`

---

## Phase 2 Rules

### `rule materialize_stitched_images_yx1`
### `rule materialize_stitched_images_keyence`
**Input**
- raw scope data path
- `scope_metadata_mapped.csv`

**Output**
- `data_pipeline_output/built_image_data/{experiment}/stitched_ff_images/` (directory)
- `data_pipeline_output/experiment_metadata/{experiment}/stitched_image_index.csv` `[VALIDATED]`

**Modules**
- `image_building/scope/yx1/stitched_ff_builder.py`
- `image_building/scope/keyence/stitched_ff_builder.py`
- shared writer: `image_building/handoff/io.py`

**Purpose**
- Perform scope-specific stitching/materialization.
- Emit reporter rows during write/symlink operations.

**Important**
- Do not crawl the output folder and infer metadata from filenames.
- Builders report rows as they process each frame.

---

### `rule validate_stitched_image_index`
**Input**
- `stitched_image_index.csv`
- `stitched_ff_images/` directory

**Output**
- validation marker file (example):
  - `data_pipeline_output/experiment_metadata/{experiment}/.stitched_image_index.validated`

**Module**
- `image_building/handoff/validate_stitched_index.py`

**Purpose**
- Ensure referenced image paths exist and rows satisfy schema/uniqueness checks.

---

### `rule build_frame_manifest`
**Input**
- `plate_metadata.csv`
- `scope_metadata_mapped.csv`
- `stitched_image_index.csv`
- `.stitched_image_index.validated`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/frame_manifest.csv` `[VALIDATED]`

**Module**
- `metadata_ingest/frame_manifest/build_frame_manifest.py`

**Purpose**
- Build one canonical frame-level table for segmentation and downstream logic.
- Join scope calibration/timing with plate annotations and stitched paths.

---

### `rule validate_frame_manifest`
**Input**
- `frame_manifest.csv`

**Output**
- validation marker file (example):
  - `data_pipeline_output/experiment_metadata/{experiment}/.frame_manifest.validated`

**Module**
- `metadata_ingest/frame_manifest/validate_frame_manifest.py`

**Purpose**
- Enforce required columns, non-null checks, uniqueness key, and basic path integrity.

---

## Contract Columns

### `stitched_image_index.csv`
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
- `materialization_status`
- `source_artifact_path`
- `source_artifact_kind`

Optional columns:
- `image_width_px`
- `image_height_px`

### `frame_manifest.csv`
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

Uniqueness key for both:
- `(experiment_id, well_id, channel_id, time_int)`

---

## Naming and ID Conventions

- `channel_id`: normalized channel (`BF`, `GFP`, etc.)
- `channel_raw_name`: microscope-native channel label
- `image_id`: `{well_id}_{channel_id}_t{frame_index:04d}`

Frame semantics:
- `time_int` = acquisition ordering key
- `frame_index` = contiguous 0-based index after sorting by `time_int` per `(experiment_id, well_id, channel_id)`

---

## Scientist-Friendly Mental Model

If you want to know:
- "What files were produced?" -> read `stitched_image_index.csv`
- "What frame table should segmentation trust?" -> read `frame_manifest.csv`
- "Where embryo IDs start?" -> segmentation stage, not metadata stage

---

## Deprecated Rules and Files (Removed)

Deprecated rule:
- `rule generate_image_manifest`

Removed files:
- `metadata_ingest/manifests/generate_image_manifest.py`
- `schemas/image_manifest.py`

Do not add new dependencies on `experiment_image_manifest.json`.

---

## Acceptance Checks

1. No duplicate frame keys in stitched index.
2. No duplicate frame keys in frame manifest.
3. Every stitched path in frame manifest exists.
4. `micrometers_per_pixel` is non-null for every row.
5. `temperature_c` and `start_age_hpf` are present where required.
6. No `embryo_id` appears before segmentation outputs.
