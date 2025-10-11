# Preliminary Snakemake Rules for MorphSeq Pipeline

**Author:** Claude Code + User
**Date:** 2025-10-11
**Status:** DRAFT - Phases 1-5 complete, Phase 6+ pending

---

## Phase 1: Metadata Input Validation

### `rule normalize_plate_metadata`
**Input:**
- `raw_plate_layout.xlsx` (user-provided, various formats)

**Output:**
- `experiment_metadata/{exp}/plate_metadata.csv` [VALIDATED]

**Module:**
- `metadata/plate_processing.py`
- Schema: `REQUIRED_COLUMNS_PLATE_METADATA`

**Purpose:**
Normalize plate layout Excel files (96-well, 24-well, various column naming) into standardized CSV with experiment_id, well_id, genotype, treatment, temperature_c, etc.

---

### `rule extract_scope_metadata`
**Input:**
- `raw_image_data/{microscope}/{exp}/` (raw microscope files)

**Output:**
- `experiment_metadata/{exp}/scope_metadata.csv` [VALIDATED]

**Module:**
- `preprocessing/{keyence|yx1}/extract_scope_metadata.py` (microscope-specific)
- Schema: `REQUIRED_COLUMNS_SCOPE_METADATA`

**Purpose:**
Extract microscope metadata (micrometers_per_pixel, frame_interval_s, image dimensions, timestamps) from raw file headers. Microscope-specific extraction logic.

---

### `rule consolidate_experiment_metadata`
**Input:**
- `experiment_metadata/{exp}/plate_metadata.csv`
- `experiment_metadata/{exp}/scope_metadata.csv`

**Output:**
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv` [VALIDATED]

**Module:**
- `preprocessing/consolidate_plate_n_scope_metadata.py` (SHARED - microscope-agnostic)
- Schema: `REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA`

**Purpose:**
Join validated plate + scope metadata on experiment_id + well_id. This is the authoritative metadata source for downstream processing.

---

## Phase 2: Image Preprocessing

### `rule preprocess_images`
**Input:**
- `raw_image_data/{microscope}/{exp}/`

**Output:**
- `built_image_data/{exp}/stitched_images/`

**Module:**
- `preprocessing/{keyence|yx1}/stitching.py` (microscope-specific)

**Purpose:**
Stitch tiles, perform z-stacking, export flat-field corrected images with normalized channel names in paths (e.g., `A01/BF/A01_BF_t0000.tif`).

**Note:** Channel normalization happens during `extract_scope_metadata` (Phase 1), so stitched images use normalized channel names.

---

### `rule generate_image_manifest`
**Input:**
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv` [VALIDATED, includes normalized channels]
- `built_image_data/{exp}/stitched_images/`

**Output:**
- `experiment_metadata/{exp}/experiment_image_manifest.json` [VALIDATED]

**Module:**
- `metadata/generate_image_manifest.py`
- Schema: `schemas/image_manifest.py` (REQUIRED_EXPERIMENT_FIELDS, REQUIRED_WELL_FIELDS, REQUIRED_CHANNEL_FIELDS, REQUIRED_FRAME_FIELDS)

**Purpose:**
- Build experiment-level manifest with all wells, channels, and frames
- Validate channel normalization (BF must be present, all names in VALID_CHANNEL_NAMES)
- Sort frames by time_int (required for SAM2 ordering)
- Single source of truth for image inventory consumed by segmentation

**Key point:** This is where channel normalization is validated. All downstream rules use normalized channel names from this manifest.

---

## Phase 3: Segmentation (SAM2 Pipeline)

**Note:** Processing happens **per-well** basis using `experiment_image_manifest.json` to get per-well frame lists.

---

### `rule gdino_detection`
**Input:**
- `built_image_data/{exp}/stitched_images/`
- `experiment_metadata/{exp}/experiment_image_manifest.json` [VALIDATED]

**Output:**
- `segmentation/{exp}/gdino_detections.json` (per-well)

**Module:**
- `segmentation/grounded_sam2/gdino_detection.py`

**Purpose:**
- Run GroundingDINO on **all frames** in the well
- Detect embryos (count, bounding boxes)
- Determine **seed frame** (good quality frame with clear embryo detection)
- Generate bounding boxes to prompt SAM2

**Key point:** Runs on ALL frames to assess embryo presence/count and select best seed frame. Uses manifest to get per-well frame lists.

---

### `rule sam2_segmentation_and_tracking`
**Input:**
- `gdino_detections.json` (seed frame bboxes)
- `experiment_metadata/{exp}/experiment_image_manifest.json` [VALIDATED]
- `built_image_data/{exp}/stitched_images/`

**Output:**
- `segmentation/{exp}/sam2_raw_output.json` (nested: video/embryo/frame structure)

**Modules:**
- `segmentation/grounded_sam2/propagation.py` (main entry point)
- `segmentation/grounded_sam2/frame_organization_for_sam2.py` (utility functions - NOT a separate rule)

**Purpose:**
- Track embryos across time using SAM2 video propagation
- Uses seed frame bboxes from GroundingDINO as prompts
- **Custom bidirectional propagation:** backward + forward from seed frame to accommodate SAM2's strict ordering requirements
- **Internal workflow:**
  1. `frame_organization_for_sam2.py` creates temp directory with SAM2-compatible frame ordering
  2. Runs bidirectional propagation (backward from seed, then forward from seed)
  3. Cleans up temp directory
  4. Outputs nested JSON with tracking results

**Key point:** `organize_frames_for_sam2` is a utility function called internally, NOT a separate Snakemake rule.

---

### `rule export_sam2_masks`
**Input:**
- `segmentation/{exp}/sam2_raw_output.json`

**Output:**
- `segmentation/{exp}/mask_images/{image_id}_masks.png` (integer-labeled PNGs)

**Module:**
- `segmentation/grounded_sam2/mask_export.py`

**Purpose:**
- Export masks as integer-labeled PNG images for visualization/QC
- Each embryo gets a unique integer label
- Useful for debugging, visual inspection, and downstream QC

---

### `rule flatten_sam2_to_csv`
**Input:**
- `segmentation/{exp}/sam2_raw_output.json`
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv` (to inject well_id, experiment_id, calibration)

**Output:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]

**Module:**
- `segmentation/grounded_sam2/csv_formatter.py`
- Schema: `REQUIRED_COLUMNS_SEGMENTATION_TRACKING`

**Purpose:**
- Flatten nested JSON → row-per-mask CSV
- Add critical columns:
  - `mask_rle` (compressed mask string)
  - `well_id` (from metadata join)
  - `experiment_id` (from metadata join)
  - `is_seed_frame` (boolean flag)
  - `source_image_path` (original stitched image)
  - `exported_mask_path` (PNG mask path)
- Validate against schema (column existence + non-empty checks)

**Key point:** This is the authoritative segmentation output consumed by all downstream steps (snip processing, features, QC).

---

## Phase 3b: UNet Auxiliary Masks

### `rule unet_auxiliary_masks`
**Input:**
- `built_image_data/{exp}/stitched_images/`

**Output:**
- `segmentation/{exp}/unet_masks/via/{image_id}_via.png` (viability/dead regions)
- `segmentation/{exp}/unet_masks/yolk/{image_id}_yolk.png` (yolk sac)
- `segmentation/{exp}/unet_masks/focus/{image_id}_focus.png` (out-of-focus)
- `segmentation/{exp}/unet_masks/bubble/{image_id}_bubble.png` (air bubbles)
- `segmentation/{exp}/unet_masks/mask/{image_id}_mask.png` (embryo - alternative segmentation)

**Modules:**
- `segmentation/unet/inference.py` (main entry point)
- `segmentation/unet/model_loader.py` (loads 5 different checkpoints)

**Purpose:**
- Generate auxiliary masks for QC purposes ONLY (not primary segmentation)
- All 5 models use same inference pipeline, just different checkpoints:
  - `mask_v0_0100` (embryo)
  - `via_v1_0100` (viability/dead regions)
  - `yolk_v1_0050` (yolk sac)
  - `focus_v0_0100` (out-of-focus regions)
  - `bubble_v0_0100` (air bubbles)

**Key point:** UNet masks are used downstream in `auxiliary_mask_qc` module (imaging quality, viability detection). SAM2 remains the authoritative embryo segmentation.

---

---

## Image Manifest Design Discussion

### **Channel Normalization Strategy**

**Problem:** Microscopes use inconsistent channel naming:
- YX1: "BF", "EYES - Dia", "Empty", "GFP" (all need normalization)
- Keyence: "BF", "Brightfield", "GFP" (need standardization)

**Solution:** Normalize during preprocessing, validate in manifest, use normalized names everywhere downstream.

### **Channel Normalization Mapping**

```python
# schemas/channel_normalization.py (NEW)

# Universal channel mapping (all microscopes)
CHANNEL_NORMALIZATION_MAP = {
    # Brightfield variants
    "bf": "BF",
    "brightfield": "BF",
    "eyes - dia": "BF",      # YX1 mislabeling
    "empty": "BF",           # YX1 mislabeling
    "phase": "Phase",

    # Fluorescence channels
    "gfp": "GFP",
    "rfp": "RFP",
    "cfp": "CFP",
    "yfp": "YFP",
    "mcherry": "mCherry",
    "dapi": "DAPI",
}

# Valid normalized names (microscope-agnostic)
VALID_CHANNEL_NAMES = [
    "BF", "Phase",                    # Brightfield
    "GFP", "RFP", "CFP", "YFP",      # Fluorescence
    "mCherry", "DAPI"
]

# BF must always be present
BRIGHTFIELD_CHANNELS = {"BF", "Phase"}
```

### **Structure (Experiment-level, indexed by well_index + normalized channel name)**

```json
{
  "experiment_id": "20250529_30hpf_ctrl",
  "microscope": "yx1",
  "wells": {
    "A01": {
      "well_index": "A01",
      "well_id": "20250529_30hpf_ctrl_A01",
      "genotype": "WT",
      "treatment": "ctrl",
      "temperature_c": 28.5,
      "embryos_per_well": 1,
      "micrometers_per_pixel": 0.65,
      "pixels_per_micrometer": 1.538,
      "frame_interval_s": 180,
      "image_width_px": 2048,
      "image_height_px": 2048,
      "channels": {
        "BF": {
          "channel_name": "BF",
          "raw_name": "EYES - Dia",
          "microscope_channel_index": 0,
          "frames": [
            {
              "image_id": "20250529_30hpf_ctrl_A01_BF_t0000",
              "time_int": 0,
              "absolute_start_time": "2025-05-29T10:00:00",
              "experiment_time_s": 0,
              "image_path": "built_image_data/20250529_30hpf_ctrl/stitched_images/A01/BF/A01_BF_t0000.tif"
            },
            {
              "image_id": "20250529_30hpf_ctrl_A01_BF_t0001",
              "time_int": 1,
              "absolute_start_time": "2025-05-29T10:03:00",
              "experiment_time_s": 180,
              "image_path": "built_image_data/20250529_30hpf_ctrl/stitched_images/A01/BF/A01_BF_t0001.tif"
            }
          ]
        },
        "GFP": {
          "channel_name": "GFP",
          "raw_name": "GFP",
          "microscope_channel_index": 1,
          "frames": [
            {
              "image_id": "20250529_30hpf_ctrl_A01_GFP_t0000",
              "time_int": 0,
              "image_path": "built_image_data/20250529_30hpf_ctrl/stitched_images/A01/GFP/A01_GFP_t0000.tif"
            }
          ]
        }
      }
    },
    "A02": {...}
  }
}
```

### **Key Design Points:**
1. **`wells` indexed by `well_index`** (e.g., "A01", not full well_id)
2. **`channels` indexed by normalized name** ("BF", "GFP" - NOT "ch00", "ch01")
3. **`image_id` uses normalized channel name** ("..._BF_t0000" - self-documenting!)
4. **`well_id`** = `experiment_id_{well_index}` (full identifier)
5. **Full metadata** from `scope_and_plate_metadata.csv` at well level
6. **Provenance preserved:** `raw_name` + `microscope_channel_index` track original values
7. **Frames list per channel** (chronological order for SAM2)

### **Why normalized channel names in image_id?**

**Before (unclear):**
```
image_id: "20250529_30hpf_ctrl_A01_ch00_t0000"  # What is ch00?
image_id: "20250529_30hpf_ctrl_A01_ch01_t0000"  # What is ch01?
```

**After (self-documenting):**
```
image_id: "20250529_30hpf_ctrl_A01_BF_t0000"   # Brightfield
image_id: "20250529_30hpf_ctrl_A01_GFP_t0000"  # GFP channel
```

**Benefits:**
- ✅ Self-documenting (no channel lookup needed)
- ✅ Easier debugging (grep for "_BF_" or "_GFP_")
- ✅ Microscope-agnostic (YX1 and Keyence both use "BF")
- ✅ Biologically meaningful (channel matters, not index)

### **Storage Location:**
```
experiment_metadata/{exp}/
  ├── plate_metadata.csv
  ├── scope_metadata.csv
  ├── scope_and_plate_metadata.csv
  └── experiment_image_manifest.json  ← Single file per experiment
```

### **Schema Validation:**
```python
# schemas/image_manifest.py

REQUIRED_EXPERIMENT_FIELDS = [
    'experiment_id',
    'microscope',
    'wells'
]

REQUIRED_WELL_FIELDS = [
    'well_index',
    'well_id',
    'genotype',
    'treatment',
    'temperature_c',
    'embryos_per_well',
    'micrometers_per_pixel',
    'frame_interval_s',
    'image_width_px',
    'image_height_px',
    'channels'
]

REQUIRED_CHANNEL_FIELDS = [
    'channel_name',              # Normalized: "BF", "GFP", etc.
    'raw_name',                  # Original: "EYES - Dia", "Empty", etc.
    # 'microscope_channel_index',  # Original index (0, 1, 2...) for provenance
    'frames'
]

REQUIRED_FRAME_FIELDS = [
    'image_id',
    'frame_index', 
    'absolute_start_time',
    'experiment_time_s',
    'image_path'
]

def validate_channels(channels_dict):
    """
    Validate that channels have been properly normalized.

    Args:
        channels_dict: {"BF": {...}, "GFP": {...}}  # Indexed by normalized name
    """
    from data_pipeline.schemas.channel_normalization import (
        VALID_CHANNEL_NAMES,
        BRIGHTFIELD_CHANNELS
    )

    # 1. Must have at least one brightfield channel
    bf_present = any(ch in BRIGHTFIELD_CHANNELS for ch in channels_dict.keys())
    if not bf_present:
        raise ValueError(f"Missing brightfield channel. Found: {list(channels_dict.keys())}")

    # 2. All channel names must be normalized
    for ch_name, ch_data in channels_dict.items():
        if ch_name not in VALID_CHANNEL_NAMES:
            raise ValueError(f"Invalid channel name: {ch_name}. Must be one of {VALID_CHANNEL_NAMES}")

        # Check required fields
        missing = set(REQUIRED_CHANNEL_FIELDS) - set(ch_data.keys())
        if missing:
            raise ValueError(f"Channel {ch_name} missing fields: {missing}")
```

### **How it's generated:**
```
rule generate_image_manifest:
    input:
        - experiment_metadata/{exp}/scope_and_plate_metadata.csv [FULL METADATA]
        - built_image_data/{exp}/stitched_images/
    output:
        - experiment_metadata/{exp}/experiment_image_manifest.json [VALIDATED]

    # Module: metadata/generate_image_manifest.py
    # 1. Read scope_and_plate_metadata.csv (includes normalized channel info from preprocessing)
    # 2. Group by well_index
    # 3. Scan stitched_images/ directory for each well + channel
    # 4. Sort frames by time_int (required for SAM2 ordering)
    # 5. Build nested JSON structure (wells → channels → frames)
    # 6. Validate against schema (REQUIRED_*_FIELDS + validate_channels())
    # 7. Write experiment_image_manifest.json [VALIDATED]
```

### **Pipeline Flow for Channel Normalization:**

```
1. Preprocessing (microscope-specific normalization)
   ├─ preprocessing/yx1/extract_scope_metadata.py
   │  ├─ Import CHANNEL_NORMALIZATION_MAP from schemas/
   │  ├─ Detect raw channel names from ND2 metadata
   │  ├─ Normalize: "EYES - Dia" → "BF", "GFP" → "GFP"
   │  └─ Write scope_metadata.csv with normalized channel_name column
   │
   └─ preprocessing/keyence/extract_scope_metadata.py
      ├─ Import CHANNEL_NORMALIZATION_MAP from schemas/
      ├─ Detect raw channel names from Keyence file structure
      ├─ Normalize: "Brightfield" → "BF", "gfp" → "GFP"
      └─ Write scope_metadata.csv with normalized channel_name column

2. Manifest Generation (shared validation)
   └─ metadata/generate_image_manifest.py
      ├─ Read scope_and_plate_metadata.csv (includes normalized channels)
      ├─ Build nested JSON (wells → channels → frames)
      ├─ Validate channels using schemas/image_manifest.py
      │  ├─ Check BF channel present (BRIGHTFIELD_CHANNELS)
      │  ├─ Check all channel names in VALID_CHANNEL_NAMES
      │  └─ Check REQUIRED_CHANNEL_FIELDS present
      └─ Write experiment_image_manifest.json [VALIDATED]

3. Downstream Rules (consume normalized names)
   └─ All rules use normalized channel names ("BF", "GFP")
      └─ No microscope-specific logic needed
```

---

## Phase 4: Snip Processing

**Note:** Snips are **processed** embryo crops, not just extracted. Processing includes: crop + rotation + noise augmentation + CLAHE equalization + Gaussian blending for training data quality.

**Current Implementation:** `src/build/build03A_process_images.py` lines 257-414 (export_embryo_snips function)

---

### `rule extract_snips`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `built_image_data/{exp}/stitched_images/`

**Output:**
- `processed_snips/{exp}/raw_crops/{snip_id}.tif` (unprocessed crops)

**Module:**
- `snip_processing/extraction.py`

**Purpose:**
- Crop embryo regions using SAM2 masks + bounding boxes from segmentation_tracking.csv
- No rotation or augmentation applied
- Save as raw TIF files for subsequent processing
- Useful for debugging and provenance (can inspect pre-processing crops)

**Key point:** Creates raw crops only. All processing (crop + rotation + augmentation) happens in next rule.

---

### `rule process_snips`
**Input:**
- `processed_snips/{exp}/raw_crops/{snip_id}.tif`
- `segmentation/{exp}/segmentation_tracking.csv` [needed for mask_rle data]

**Output:**
- `processed_snips/{exp}/processed/{snip_id}.jpg` (fully processed)

**Module:**
- `snip_processing/rotation.py` (PCA-based orientation)
- `snip_processing/augmentation.py` (noise + CLAHE + blending)

**Purpose:**
- Apply crop + PCA-based rotation for standardized orientation
- Add Gaussian noise to background regions (training data augmentation)
- Apply CLAHE histogram equalization (contrast enhancement)
- Gaussian blending at edges (smooth transitions)
- Save as JPEGs with snip_id naming

**Key processing steps (from build03A lines 367-384):**
1. Crop to bounding box region
2. PCA rotation using mask contour (angle stored for manifest)
3. CLAHE equalization (clipLimit=2.0, tileGridSize=(8,8))
4. Gaussian noise addition to background (mean=0, std=10)
5. Gaussian blur blending at edges (sigma=3)

**Key point:** Only saves processed JPEGs. Manifest generation happens separately to allow validation without reprocessing.

---

### `rule generate_snip_manifest`
**Input:**
- `processed_snips/{exp}/processed/` (directory of processed snips)
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]

**Output:**
- `processed_snips/{exp}/snip_manifest.csv` [VALIDATED]

**Module:**
- `snip_processing/manifest_generation.py`
- Schema: `REQUIRED_COLUMNS_SNIP_MANIFEST`

**Purpose:**
- Scan processed_snips/ directory to inventory all processed JPEGs
- Join with segmentation_tracking.csv to get experiment_id, well_id, embryo_id, time_int
- Validate completeness (all expected snips present, no missing files)
- Add file metadata (file size, dimensions, processing timestamp)
- Validate schema and write snip_manifest.csv [VALIDATED]

**Required manifest columns:**
```python
REQUIRED_COLUMNS_SNIP_MANIFEST = [
    'snip_id',
    'experiment_id',
    'well_id',
    'embryo_id',
    'time_int',
    'raw_crop_path',          # Path to raw crop TIF
    'processed_snip_path',    # Path to processed JPEG
    'file_size_bytes',        # Validate files exist and are non-empty
    'image_width_px',         # Actual snip dimensions
    'image_height_px',
    'processing_timestamp',   # When processing occurred
]
```

**Output structure:**
```
processed_snips/{exp}/
├── raw_crops/
│   └── {snip_id}.tif         # Unprocessed crops (for debugging)
├── processed/
│   └── {snip_id}.jpg         # Fully processed (crop + rotate + augment)
└── snip_manifest.csv         # [VALIDATED] - Authoritative snip inventory
```

**Key point:** Separate manifest generation allows validation without reprocessing. Can regenerate manifest to add new columns or verify file integrity.

---

## Phase 5: Feature Extraction

**Note:** Features are computed from validated segmentation data and consolidated into a single analysis-ready table. All feature modules run in parallel (independent computations), then consolidation merges results.

**Current Implementation:** `src/build/build03A_process_images.py` (compile_embryo_stats function, lines 771-863)

---

### `rule compute_mask_geometry`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv` [for pixel_size calibration]

**Output:**
- `computed_features/{exp}/mask_geometry_metrics.csv`

**Module:**
- `feature_extraction/mask_geometry_metrics.py`

**Purpose:**
- Compute geometric features from SAM2 masks:
  - `area_px`, `area_um2` (using micrometers_per_pixel)
  - `perimeter_px`, `perimeter_um`
  - `length_um`, `width_um` (via PCA on mask contour)
  - `centroid_x_um`, `centroid_y_um`
- **Critical:** Must convert area_px → area_um2 using micrometers_per_pixel from scope_and_plate_metadata.csv
- **Critical:** Fail if pixel-based areas are used without calibration (downstream stage inference requires um2)

**Key columns:**
```python
OUTPUT_COLUMNS_MASK_GEOMETRY = [
    'snip_id',
    'area_px',
    'area_um2',           # Required for stage inference
    'perimeter_px',
    'perimeter_um',
    'length_um',
    'width_um',
    'centroid_x_um',
    'centroid_y_um',
]
```

---

### `rule compute_pose_kinematics`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv` [for pixel_size + frame_interval_s]

**Output:**
- `computed_features/{exp}/pose_kinematics_metrics.csv`

**Module:**
- `feature_extraction/pose_kinematics_metrics.py`

**Purpose:**
- Compute pose and motion features:
  - Bounding box dimensions (bbox_width_um, bbox_height_um)
  - Orientation angle (from PCA or SAM2 mask)
  - Frame-to-frame deltas:
    - `displacement_um` (Euclidean distance between centroids)
    - `speed_um_per_s` (displacement / frame_interval_s)
    - `angular_velocity_deg_per_s`
- Requires temporal ordering (sort by embryo_id + time_int)

**Key columns:**
```python
OUTPUT_COLUMNS_POSE_KINEMATICS = [
    'snip_id',
    'bbox_width_um',
    'bbox_height_um',
    'orientation_deg',
    'displacement_um',
    'speed_um_per_s',
    'angular_velocity_deg_per_s',
]
```

---

### `rule compute_fraction_alive`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `segmentation/{exp}/unet_masks/viability/`

**Output:**
- `computed_features/{exp}/fraction_alive.csv`

**Module:**
- `feature_extraction/fraction_alive.py`

**Purpose:**
- Measure the proportion of viable pixels per snip using UNet viability masks
- Aggregate by `snip_id` using SAM2 masks to normalize for embryo area
- Emits continuous `fraction_alive` plus helper metadata (e.g., total viability pixels)

---

### `rule predict_developmental_stage`
**Input:**
- `computed_features/{exp}/mask_geometry_metrics.csv` [needs area_um2]

**Output:**
- `computed_features/{exp}/stage_predictions.csv`

**Module:**
- `feature_extraction/stage_inference.py`

**Purpose:**
- Predict developmental stage (HPF - hours post fertilization) from area_um2 growth curves
- Uses Kimmel et al. (1995) formula or trained model
- **Must use area_um2** - fail if pixel-based areas detected
- Outputs predicted_stage_hpf for each snip

**Key columns:**
```python
OUTPUT_COLUMNS_STAGE_PREDICTIONS = [
    'snip_id',
    'predicted_stage_hpf',
    'stage_confidence',      # Optional confidence score
]
```

**Key point:** This rule depends on `compute_mask_geometry` completing first (needs area_um2 input).

---

### `rule consolidate_features`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [base table with snip_id]
- `computed_features/{exp}/mask_geometry_metrics.csv`
- `computed_features/{exp}/pose_kinematics_metrics.csv`
- `computed_features/{exp}/fraction_alive.csv`
- `computed_features/{exp}/stage_predictions.csv`
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv` [for joining experiment_id, well_id]

**Output:**
- `computed_features/{exp}/consolidated_snip_features.csv` [VALIDATED]

**Module:**
- `feature_extraction/consolidate_features.py`
- Schema: `REQUIRED_COLUMNS_CONSOLIDATED_FEATURES`

**Purpose:**
- Merge all feature tables on snip_id
- Add experiment metadata (experiment_id, well_id, genotype, treatment, temperature_c)
- Validate completeness:
  - All snips from segmentation_tracking.csv have features
  - No missing critical columns (area_um2, predicted_stage_hpf)
  - No NaN values in required fields
- This is the **single source of truth** consumed by all QC and analysis modules

---

## Phase 6: Quality Control

QC modules compute independent quality flags from various sources (SAM2, UNet masks, features). A final consolidation rule combines all flags into `use_embryo_flag`.

---

### `rule compute_auxiliary_mask_qc`
**Input**
- `segmentation/{exp}/segmentation_tracking.csv` `[VALIDATED]`
- `segmentation/{exp}/unet_masks/` (viability, yolk, focus, bubble from Phase 3b)
- `computed_features/{exp}/fraction_alive.csv`
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv`

**Output**
- `quality_control/{exp}/auxiliary_mask_qc.csv`

**Module**
- `quality_control/auxiliary_mask_qc.py`
- Uses `qc_utils.compute_qc_flags()`, `qc_utils.compute_fraction_alive()`

- **Purpose**
  - Join `fraction_alive` with auxiliary mask overlays
  - Threshold viability (`fraction_alive < dead_threshold`, default 0.9) to produce the canonical `dead_flag`
  - Emit imaging QC flags derived from UNet masks (`no_yolk_flag`, `focus_flag`, `bubble_flag`, `frame_flag` legacy)

**Columns**
`snip_id`, `fraction_alive`, `dead_flag`, `no_yolk_flag`, `focus_flag`, `bubble_flag`, `frame_flag`

---

### `rule compute_surface_area_outliers`
**Input**
- `computed_features/{exp}/consolidated_snip_features.csv`
- `metadata/sa_reference_curves.csv` (reference SA curves)

**Output**
- `quality_control/{exp}/sa_outlier_qc.csv`

**Module**
- `quality_control/surface_area_outlier_detection.py`
- Function: `compute_sa_outlier_flag()`

**Purpose**
- Flag embryos with abnormal surface area for their stage
- Two-sided outlier detection:
  - Upper: `area_um2 > k_upper × reference(stage_hpf)` (k = 1.2)
  - Lower: `area_um2 < k_lower × reference(stage_hpf)` (k = 0.9)
- Uses control embryos (wt, control_flag) to build reference curve
- Fallback to `stage_ref.csv` if insufficient controls

**Columns**
`snip_id`, `sa_outlier_flag`

---

### `rule compute_death_lead_time_flag`
**Input**
- `quality_control/{exp}/auxiliary_mask_qc.csv` (for `dead_flag`)
- `computed_features/{exp}/consolidated_snip_features.csv` (for `embryo_id`, `predicted_stage_hpf`)

**Output**
- `quality_control/{exp}/death_lead_time_qc.csv`

**Module**
- `quality_control/death_detection.py`
- Function: `compute_death_lead_time_flag()`

**Purpose**
- Retroactively flag embryos in hours leading up to the canonical `dead_flag`
- For embryos with any `dead_flag == True`:
  - Find first death time
  - Flag all timepoints within `dead_lead_time` hours before death (default 2.0)
- Column is named `death_lead_time_flag` to avoid introducing a second death flag; gating treats it as a separate QC signal.

**Columns**
`snip_id`, `death_lead_time_flag`

**Parameter**
`dead_lead_time = 2.0` hours (configurable)

---

### `rule consolidate_qc_flags`
**Input**
- `computed_features/{exp}/consolidated_snip_features.csv` (base table)
- `quality_control/{exp}/auxiliary_mask_qc.csv`
- `quality_control/{exp}/sa_outlier_qc.csv`
- `quality_control/{exp}/death_lead_time_qc.csv`
- `segmentation/{exp}/segmentation_tracking.csv` (for `sam2_qc_flags`)

**Output**
- `quality_control/{exp}/consolidated_qc_flags.csv` `[VALIDATED]`

**Module**
- `quality_control/consolidate_qc.py`
- Schema: `REQUIRED_COLUMNS_QC_FLAGS`

**Purpose**
- Merge all QC flags onto consolidated features
- Compute final `use_embryo_flag`:
  ```python
  use_embryo_flag = NOT (
      dead_flag OR death_lead_time_flag OR sa_outlier_flag OR
      sam2_qc_flag OR focus_flag OR bubble_flag OR no_yolk_flag
      # frame_flag DISABLED (too sensitive; SAM2 edge detection preferred)
  )
  ```
- Generate QC summary statistics (counts per flag)
- Authoritative QC table consumed by analysis modules

**Validation**
1. All snips from consolidated_features present
2. No duplicate snip_ids
3. All flags are boolean (fillna with False)
4. `use_embryo_flag` correctly computed

**Key point:** Consolidates QC from three sources:
- SAM2 QC (edge detection, tracking issues)
- Auxiliary mask QC (viability, focus, bubbles)
- Feature-based QC (SA outliers, death lead-time)

---

## Phase 7: Embeddings - PENDING

---

## Phase 8: Analysis-Ready - PENDING

---

## Summary of New Files and Data Outputs

### **New Schema Files**
```
src/data_pipeline/schemas/
├── __init__.py
├── channel_normalization.py          # NEW - Channel name mappings
├── image_manifest.py                 # NEW - Image manifest schema
├── plate_metadata.py
├── scope_metadata.py
├── scope_and_plate_metadata.py
├── segmentation.py
├── snip_processing.py
├── features.py
├── quality_control.py
└── analysis_ready.py
```

### **New Processing Modules**
```
src/data_pipeline/
├── metadata/
│   ├── plate_processing.py
│   └── generate_image_manifest.py
├── preprocessing/
│   ├── keyence/
│   │   └── extract_scope_metadata.py
│   ├── yx1/
│   │   └── extract_scope_metadata.py
│   └── consolidate_plate_n_scope_metadata.py
├── snip_processing/
│   ├── extraction.py
│   ├── rotation.py
│   ├── augmentation.py
│   └── manifest_generation.py
└── feature_extraction/
    ├── mask_geometry_metrics.py
├── pose_kinematics_metrics.py
├── fraction_alive.py
├── stage_inference.py
└── consolidate_features.py
```

### **New Data Outputs**
```
experiment_metadata/{exp}/
├── plate_metadata.csv [VALIDATED]
├── scope_metadata.csv [VALIDATED]
├── scope_and_plate_metadata.csv [VALIDATED]
└── experiment_image_manifest.json [VALIDATED]  # NEW - Single manifest per experiment

built_image_data/{exp}/
└── stitched_images/
    └── {well_index}/
        └── {channel_name}/              # NEW - Organized by normalized channel
            └── {well_index}_{channel_name}_t{time}.tif

segmentation/{exp}/
├── gdino_detections.json
├── sam2_raw_output.json
├── segmentation_tracking.csv [VALIDATED]
└── mask_images/

processed_snips/{exp}/
├── raw_crops/{snip_id}.tif
├── processed/{snip_id}.jpg
└── snip_manifest.csv [VALIDATED]

computed_features/{exp}/
├── mask_geometry_metrics.csv
├── pose_kinematics_metrics.csv
├── fraction_alive.csv
├── stage_predictions.csv
└── consolidated_snip_features.csv [VALIDATED]
```

### **Key Changes from Original Plan**
1. ✅ **Renamed directory:** `processed_metadata/` → `experiment_metadata/`
2. ✅ **Added rule:** `generate_image_manifest` (Phase 2)
3. ✅ **Channel normalization:** YX1/Keyence extract + normalize → manifest validates → downstream consumes
4. ✅ **Self-documenting image_ids:** `_BF_t0000` instead of `_ch00_t0000`
5. ✅ **Provenance preserved:** `raw_name` + `microscope_channel_index` in manifest
6. ✅ **Single manifest file:** experiment_image_manifest.json (all wells + channels)

---

## Notes
- [VALIDATED] = Schema enforcement at consolidation point
- Microscope-specific vs shared logic explicitly noted
- Explicit consolidation steps tracked
- Channel normalization happens in Phase 1 (extract_scope_metadata)
- Manifest generation validates normalization in Phase 2
