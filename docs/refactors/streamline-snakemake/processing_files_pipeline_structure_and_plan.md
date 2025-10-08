# MorphSeq Pipeline Refactor: Final Structure & Implementation Plan

**Author:** Claude Code Analysis
**Date:** 2025-10-06
**Status:** APPROVED - READY FOR IMPLEMENTATION

---

## Pipeline Tasks Overview

The MorphSeq pipeline processes zebrafish embryo timelapse data through these major tasks:

```
IMAGE PREPROCESSING
  Raw microscope data → Stitched FF images + metadata
  (Keyence or YX1 microscope-specific processing)

SEGMENTATION
  Stitched images → Embryo masks + auxiliary masks
  - SAM2: Embryo detection, tracking, propagation (PRIMARY METHOD)
  - UNet: Auxiliary masks (yolk, bubble, focus, viability)

SNIP PROCESSING & FEATURE EXTRACTION
  Masks + images → Cropped/rotated snips + morphology features
  - Extract embryo regions
  - Compute shape, spatial features
  - Infer developmental stage

QUALITY CONTROL
  Features + masks → QC flags + validation
  - Imaging quality (frame, focus, bubbles, yolk)
  - Viability tracking (death detection)
  - Tracking validation (movement, trajectory)
  - Segmentation quality (mask validation)
  - Size validation (area thresholds)

EMBEDDING GENERATION
  QC-approved snips (`use_embryo == True`) → Latent embeddings
  (VAE-based, note Python 3.9 subprocess)
```

**Key Principles:**
- SAM2 is the **primary segmentation method** (tracking + masks)
- UNet provides **auxiliary masks** for QC (yolk, bubble, focus, viability, embryo)
  - UNet embryo masks used for validation/comparison only
  - SAM2 embryo masks are authoritative for the pipeline
- Each task maps cleanly to Snakemake rules
- Flexible, extensible structure (no hardcoded stage numbers)

---

## Experiment Inventory Strategy

### Legacy behaviour (`ExperimentManager`)
- Auto-discovers experiments by scanning `raw_image_data/{microscope}/{experiment_id}` (`src/build/pipeline_objects.py:1181-1202`).
- Any directory that is not hidden/ignored becomes an experiment key (e.g., `20240915_keyence`).
- Experiment state is then tracked via JSON under `metadata/experiments/`.
- Manual overrides require writing custom scripts or filtering the `ExperimentManager.experiments` dict after discovery.

### Proposed Snakemake approach
- Keep automatic discovery as the *default*: we still glob `raw_image_data/{microscope}/{experiment_id}` when no extra config is supplied, mirroring `ExperimentManager` behaviour.
- Optionally honor a curated inventory file (`metadata/experiments/experiments.csv` or YAML) when present. Columns can include:
  - `experiment_id`
  - `microscope`
  - optional overrides (e.g., `sam2_model`, `skip_unet`)
- A loader in `data_pipeline.config.registry` implements the logic:
  1. If `--config experiments=<list>` is provided, use that explicit list (after validating the directories).
  2. Else if the inventory file exists, use the curated rows.
  3. Else fall back to the raw-directory discovery.
- Snakemake command options:
  - Default run: `snakemake all` → auto-discovery.
  - Manual subset: `snakemake all --config experiments=20240915_keyence,20240918_yx1`.
  - Force use of the curated inventory: `snakemake all --config use_inventory=true`.
  - Re-run the legacy glob explicitly: `snakemake all --config experiments=discover` (alias for path scanning).

### Manual override workflow
- To process a handful of experiments, either:
  1. Edit/duplicate `experiments.csv`, or
  2. Pass `--config experiments=exp_a,exp_b` on the CLI.
- The `registry` helper merges command-line overrides with the canonical list and validates that directories exist under `raw_image_data/`.
- For ad-hoc new experiments, ship a small script (`scripts/discover_experiments.py`) that scans raw data, emits a CSV stub, and allows curating metadata before running Snakemake.

### Why this layout
- Keeps discovery logic declarative and version-controlled.
- Aligns with Snakemake’s `config` semantics while retaining a single source of truth for microscope metadata.
- Makes it easy to exclude problem experiments without renaming directories or editing the workflow.

---

## Finalized Directory Structure

```
src/data_pipeline/

├── preprocessing/                              # Raw → Stitched Images
│   ├── keyence/
│   │   ├── stitching.py                       # Tile stitching logic
│   │   ├── z_stacking.py                      # Z-slice focus stacking
│   │   └── metadata.py                        # Scrape Keyence metadata
│   └── yx1/
│       ├── processing.py                      # YX1-specific processing
│       └── metadata.py                        # YX1 metadata extraction
│
├── segmentation/                               # Segmentation
│   ├── grounded_sam2/                                  # SAM2 embryo tracking (PRIMARY)
│   │   ├── frame_organization_for_sam2.py    # Organize images into video structure
│   │   ├── gdino_detection.py                # Grounded DINO embryo detection
│   │   ├── propagation.py                    # SAM2 mask propagation
│   │   ├── mask_export.py                    # Export masks to PNG
│   │   └── sam2_output_csv_formatter.py                  # Flatten JSON to CSV
│   ├── unet/                                  # UNet auxiliary masks
│   │   ├── inference.py                      # Core inference pipeline
│   │   └── model_loader.py                   # Load 5 models (mask, via, yolk, focus, bubble)
│   └── mask_utilities.py                     # Shared RLE/polygon/bbox utilities
│
├── snip_processing/                            # Snips & Features
│   ├── extraction.py                          # Crop embryo regions from images
│   ├── rotation.py                            # PCA-based rotation alignment
│   ├── augmentation.py                        # Synthetic noise (training data)
│   ├── io.py                                  # Save snip images
│   └── embryo_features/                       # Features from snips
│       ├── shape.py                           # Area, perimeter, contours
│       ├── spatial.py                         # Centroids, bboxes, orientation
│       └── stage_inference.py                 # HPF (developmental stage) prediction
│
├── quality_control/                            # Quality Control signals
│   ├── auxiliary_unet_imaging_quality_qc.py  # Frame, yolk, focus, bubble flags (from UNet)
│   ├── viability_tracking_qc.py              # Death detection with persistence validation
│   ├── tracking_metrics_qc.py                # Movement speed, trajectory smoothing, tracking validation
│   ├── segmentation_quality_qc.py            # SAM2 mask quality checks
│   └── size_validation_qc.py                 # Area thresholds, abnormal growth detection
│
├── embeddings/                                 # Latent Embeddings (QC-passed snips only)
│   ├── inference.py                           # VAE embedding generation
│   ├── subprocess_wrapper.py                  # Python 3.9 subprocess orchestration
│   └── file_validation.py                     # Check existing embeddings
│
├── pipeline_orchestrator/                     # Snakemake entry point & helpers
│   ├── Snakefile                             # Task DAG importing data_pipeline modules
│   ├── config/                               # Snakemake defaults + execution profiles
│   ├── experiment_discovery.py               # Resolve experiment lists (override → inventory → glob)
│   └── cli.py                                # Thin wrapper for launching the workflow
├── identifiers/                                # Shared utilities
│   └── parsing.py                             # ID parsing (from parsing_utils.py)
│
├── metadata/                                   # Metadata operations
│   └── enrichment.py                          # Perturbation metadata merging
│
└── io/                                         # File I/O utilities
    ├── loaders.py                             # Load images, masks, CSVs
    ├── savers.py                              # Save outputs
    └── validators.py                          # Validate file formats
```

---

## Module Descriptions

### **preprocessing/**
**Purpose:** Convert raw microscope data into standardized stitched FF images

**Microscope-specific modules:**
- **keyence/**: Keyence BZ-X800 processing (tile stitching, z-stacking)
- **yx1/**: YX1 microscope processing

**Why separate?** Different microscopes have different:
- File formats
- Tile arrangements
- Metadata structures
- Stitching requirements

**Future-proof:** Easy to add new microscope types

---

### **segmentation/**
**Purpose:** Detect and track embryos, generate masks

#### **grounded_sam2/** - SAM2 + GroundingDINO pipeline (PRIMARY)
- **frame_organization_for_sam2.py**: Reorganize well/timepoint images into video frame sequences
- **gdino_detection.py**: Grounded DINO detection for seed annotations
- **propagation.py**: SAM2 mask propagation (forward/bidirectional)
- **bounding_box_utils.py**: Convert GroundingDINO detections into SAM2 prompt boxes
- **mask_export.py**: Export integer-labeled PNG masks
- **csv_formatter.py**: Flatten nested JSON to tabular CSV

#### **unet/** - AUXILIARY MASKS FOR QC
- **inference.py**: Run inference on all 5 UNet models
- **model_loader.py**: Load different model checkpoints
  - `mask_v0_0100` (embryo)
  - `via_v1_0100` (viability/dead regions)
  - `yolk_v1_0050` (yolk sac)
  - `focus_v0_0100` (out-of-focus regions)
  - `bubble_v0_0100` (air bubbles)

**Note:** All 5 models use the same inference pipeline, just different checkpoints (verify in build02B)

#### **mask_utilities.py**
Shared utilities for all segmentation methods:
- RLE encoding/decoding
- Polygon conversion
- Bounding box extraction

---

### **snip_processing/**
**Purpose:** Extract, align, and measure embryo regions

**Snip operations:**
- **extraction.py**: Crop embryo regions with padding
- **rotation.py**: PCA-based alignment to standard orientation
- **augmentation.py**: Add synthetic noise for training data
- **io.py**: Save snip images to disk

**Feature extraction (subfolder):**
- **embryo_features/shape.py**: Area, perimeter, aspect ratio, contours
- **embryo_features/spatial.py**: Centroids, bounding boxes, orientation angles
- **embryo_features/stage_inference.py**: Predict HPF (hours post-fertilization) from morphology

**Why nested?** embryo_features are extracted FROM processed snips (logical hierarchy)

---

### **quality_control/**
**Purpose:** Validate data quality, flag issues

#### **auxiliary_unet_imaging_quality_qc.py**
Checks imaging quality and biological context using UNet auxiliary masks:
- `frame_flag`: Embryo near image boundary
- `no_yolk_flag`: Yolk sac missing (abnormal development)
- `focus_flag`: Out-of-focus regions nearby (imaging issue)
- `bubble_flag`: Air bubbles nearby (contamination)
- `out_of_frame_flag`: Embryo truncated at edge (legacy, Build03)

**Source:** UNet auxiliary masks + spatial proximity analysis

#### **viability_tracking_qc.py**
Tracks embryo viability over time:
- Detects persistent decline in `fraction_alive`
- Validates death inflection points (25% persistence threshold)
- Flags dead embryos with 2hr buffer
- `dead_flag`: Initial viability threshold (Build03)
- `dead_flag2`: Death with persistence validation (Build04)
- Adds `dead_inflection_time_int` column

**Source:** UNet viability mask + persistence algorithm

#### **tracking_metrics_qc.py**
Validates embryo tracking quality:
- `compute_speed()`: Movement speed between frames
- `smooth_trajectory()`: Savitzky-Golay trajectory smoothing
- `detect_tracking_errors()`: Identify jumps, discontinuities

**Note:** This is QC for tracking results, NOT the tracking itself

#### **segmentation_quality_qc.py**
Validates SAM2 segmentation quality using multiple checks:
- `HIGH_SEGMENTATION_VAR_SNIP`: High area variance vs nearby frames (>20% change)
- `MASK_ON_EDGE`: Mask touches image edges (within 2 pixels)
- `DETECTION_FAILURE`: Missing expected embryos in frame
- `OVERLAPPING_MASKS`: Embryo masks overlap (IoU > 0.1)
- `LARGE_MASK`: Unusually large mask (>15% of frame area)
- `SMALL_MASK`: Unusually small mask (<0.1% of frame area)
- `DISCONTINUOUS_MASK`: Multiple disconnected mask components
- `sam2_qc_flag`: General SAM2 quality issue (legacy)

**Source:** SAM2 output JSON from `segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py`

#### **size_validation_qc.py**
Validates embryo sizes are biologically plausible:
- `sa_outlier_flag`: Surface area outlier (Build04)
  - Compares to internal controls or stage reference
  - One-sided detection (flags abnormally large only)

**Source:** Morphology measurements + HPF stage predictions

---

### **embeddings/**
**Purpose:** Generate latent representations for QC-approved snips

**Complex subprocess logic:**
- Uses Python 3.9 for legacy model compatibility
- Subprocess orchestration with error handling
- File validation to avoid recomputation
- Filters inputs to `use_embryo == True` before launching inference

**Already well-organized** - just needs to move from `src/analyze/gen_embeddings/`

---

### **pipeline_orchestrator/**
**Purpose:** Provide Snakemake-driven workflow orchestration

- `Snakefile`: Defines DAG, imports task functions from `data_pipeline`
- `config/defaults.yaml`: Baseline Snakemake configuration (experiments, device, thresholds)
- `experiment_discovery.py`: Shared experiment resolver (CLI override → inventory → glob)
- `cli.py`: Python shim to launch Snakemake with friendly arguments

**Planning reference:** `docs/refactors/streamline-snakemake/populate_process_files/pipeline_orchestrator.md`

---

### **identifiers/**
**Purpose:** Canonical ID parsing and validation

**Already excellent** - just move from `segmentation_sandbox/scripts/utils/parsing_utils.py`

Functions:
- `parse_entity_id()`: Auto-detect and parse any ID type
- `build_*_id()`: Construct IDs from components
- `extract_*()`: Extract parent IDs
- `validate_id_format()`: Validate ID structure

---

### **metadata/**
**Purpose:** Metadata enrichment operations

- **enrichment.py**: Merge perturbation metadata, genotype/phenotype mappings

---

### **io/**
**Purpose:** File I/O utilities shared across pipeline

- **loaders.py**: Load images, masks, CSVs, JSONs
- **savers.py**: Save outputs with proper formatting
- **validators.py**: Validate file formats, schemas

---

## Snakemake Rule Mapping

Each folder maps to a logical group of Snakemake rules:

```python
# Preprocessing rules
rule preprocess_keyence:
    run: from data_pipeline.preprocessing.keyence import stitch, extract_metadata

rule preprocess_yx1:
    run: from data_pipeline.preprocessing.yx1 import process_images

# Segmentation rules
# NOTE: Frame organization integrated into propagation (not a separate rule)
# propagate_masks() internally creates temp video structure, runs SAM2, cleans up

rule gdino_detect:
    run: from data_pipeline.segmentation.sam2.gdino_detection import detect_embryos

rule sam2_segment_and_track:
    run: from data_pipeline.segmentation.sam2.propagation import propagate_masks

rule sam2_export_masks:
    run: from data_pipeline.segmentation.sam2.mask_export import export_masks

rule unet_segment:
    run: from data_pipeline.segmentation.unet.inference import run_all_models

# Snip processing & features
rule extract_snips:
    run: from data_pipeline.snip_processing.extraction import crop_embryos

rule compute_morphology_features:
    run: from data_pipeline.snip_processing.embryo_features.shape import compute_morphology

rule infer_embryo_stage:
    run: from data_pipeline.snip_processing.embryo_features.stage_inference import infer_hpf_stage

# Quality control
rule qc_imaging:
    run: from data_pipeline.quality_control.auxiliary_unet_imaging_quality_qc import compute_qc_flags

rule qc_viability:
    run: from data_pipeline.quality_control.viability_tracking_qc import compute_dead_flag2_persistence

rule qc_tracking:
    run: from data_pipeline.quality_control.tracking_metrics_qc import compute_speed

rule qc_segmentation:
    run: from data_pipeline.quality_control.segmentation_quality_qc import validate_masks

rule qc_size:
    run: from data_pipeline.quality_control.size_validation_qc import validate_sizes

# Embeddings (QC-gated)
rule generate_embeddings:
    run: from data_pipeline.embeddings.inference import ensure_embeddings
```

---

## 5-Week Implementation Plan

### **Week 1: Move Core Utilities & Critical Verifications**
**Goal:** Validate approach with minimal changes + verify critical assumptions

**Move as-is:**
1. `segmentation_sandbox/scripts/utils/parsing_utils.py` → `identifiers/parsing.py` (~800 lines)
2. `src/build/qc_utils.py` → `quality_control/auxiliary_unet_imaging_quality_qc.py` (135 lines)
3. `src/data_pipeline/quality_control/death_detection.py` → `quality_control/viability_tracking_qc.py` (317 lines)
4. `segmentation_sandbox/scripts/utils/mask_utils.py` → `segmentation/mask_utilities.py` (~200 lines)
5. `src/analyze/gen_embeddings/*.py` → `embeddings/*.py` (~300 lines total)

**Create shared infrastructure:**
6. `config/runtime.py` with `resolve_device(prefer_gpu: bool) -> torch.device`
   - Single implementation reused by all preprocessing modules
   - Handles CUDA availability checking, fallback to CPU
   - Supports string inputs ("cuda", "cpu", "auto")

**Critical verifications (BLOCKING for Week 2):**
7. **Verify UNet post-processing assumption:**
   - Read `build02B_segment_bf_main.py` carefully
   - Confirm all 5 models (mask, via, yolk, focus, bubble) use identical post-processing
   - Document any differences found
   - If assumption is wrong, revise Week 2 module structure plan

8. **Document microscope filename issues:**
   - Audit Keyence and YX1 filename patterns in raw data
   - Document ad-hoc naming variations discovered
   - Create normalization strategy for `config/microscopes.py`

**Update imports in 2-3 existing scripts to verify**

**Deliverable:**
- Core utilities importable from new locations, all tests pass
- Device resolver ready for use
- UNet verification complete (green/red light for Week 2)
- Microscope filename issues documented

---

### **Week 2: Extract Preprocessing & UNet Modules**

**Preprocessing**
- Extract from `build01A_compile_keyence_torch.py` → `preprocessing/keyence/`
- Extract from `build01B_compile_yx1_images_torch.py` → `preprocessing/yx1/`

**UNet segmentation (auxiliary masks)**
- Extract from `build02B_segment_bf_main.py` → `segmentation/unet/`
- **Verify:** All 5 models use same post-processing (just different checkpoints)

**Create Snakemake rules for preprocessing + UNet masks**

**Deliverable:** Can stitch images and run UNet via Snakemake

---

### **Week 3: Extract SAM2 Pipeline**

**Extract from sandbox scripts:**
- `01_prepare_videos.py` → `segmentation/grounded_sam2/frame_organization_for_sam2.py`
- `03_gdino_detection.py` → `segmentation/grounded_sam2/gdino_detection.py`
- `04_sam2_video_processing.py` → `segmentation/grounded_sam2/propagation.py`
- `05_sam2_qc_analysis.py` → `quality_control/segmentation_quality_qc.py`
- `06_export_masks.py` → `segmentation/grounded_sam2/mask_export.py`
- `export_sam2_metadata_to_csv.py` → `segmentation/grounded_sam2/csv_formatter.py`
- Bounding box conversions handled in new `segmentation/grounded_sam2/bounding_box_utils.py` (extracted from detection + propagation scripts)

**Delete:**
- `GroundedSamAnnotations` class
- `GroundedDinoAnnotations` class
- `SimpleMaskExporter` class
- `SAM2MetadataExporter` class
- `BaseFileHandler` usage
- Entity tracking systems

**Extract pure functions, delete class hierarchies**

**Deliverable:** SAM2 pipeline runs via Snakemake

---

### **Week 4: Extract Snip Processing & QC Modules**

**Snip processing & features**
- Extract from `build03A_process_images.py` (1753 lines → ~200 lines across 7 files)
  - `snip_processing/extraction.py`
  - `snip_processing/rotation.py`
  - `snip_processing/augmentation.py`
  - `snip_processing/io.py`
  - `snip_processing/embryo_features/*.py`

**Quality control**
- Extract from `build04_perform_embryo_qc.py`:
  - `quality_control/tracking_metrics_qc.py`
  - `quality_control/size_validation_qc.py`
  - `snip_processing/embryo_features/stage_inference.py`

**Deliverable:** Build03 and Build04 run via Snakemake

---

### **Week 5: Workflow Wiring & Cleanup**
- Ensure Snakemake rules enforce QC gating before embeddings (use `use_embryo` filters)
- Update CLI entry points to call Snakemake targets instead of `ExperimentManager`
- Remove deprecated orchestration layers once parity is achieved
- Add regression checks or smoke tests covering preprocessing → QC → embeddings handoff

**Delete overengineering:**
- `pipeline_objects.py` (1593 lines) → Replaced by Snakemake
- `base_file_handler.py` → Not needed
- `entity_id_tracker.py` → Not needed
- All sandbox scripts 01-07 → Become Snakemake rules
- `build03A_process_images.py` → After extraction
- `build04_perform_embryo_qc.py` → After extraction

**Create full Snakemake workflow**

**Deliverable:** Complete end-to-end pipeline, clean codebase

---

## Key Design Decisions

### ✅ **SAM2 is Primary, UNet is Auxiliary**
- SAM2: Embryo detection, tracking, propagation (main segmentation)
- UNet: Yolk, bubble, focus, viability masks (QC support)

### ✅ **Microscope-Specific Preprocessing**
- Separate modules for Keyence vs YX1
- Easy to add new microscope types

### ✅ **Explicit, Descriptive Names**
- Long names OK if they eliminate ambiguity
- `auxiliary_unet_imaging_quality_qc.py` > `spatial_qc.py`
- `frame_organization_for_sam2.py` > `video_prep.py`

### ✅ **Functions Over Classes**
- Extract pure functions from overengineered classes
- Delete annotation management, entity tracking systems
- Let Snakemake handle orchestration

### ✅ **Logical Hierarchies**
- `snip_processing/embryo_features/` - features extracted FROM snips
- `segmentation/grounded_sam2/` - SAM2 + GDINO components grouped together
- `segmentation/unet/` - UNet components grouped together

### ✅ **QC Naming Clarity**
- `*_qc.py` suffix for all QC modules
- Distinguish QC from actual operations (tracking_metrics_qc vs tracking)

---

## Next Steps

1. ✅ **Get final approval on structure**
2. ✅ **Verify UNet post-processing** (read build02B)
3. ✅ **Create Week 1 detailed migration guide**
4. ✅ **Begin file moves**

---

## Summary

**This structure is:**
- ✅ **Clear** - Names describe purpose
- ✅ **Flexible** - No hardcoded stage numbers
- ✅ **Extensible** - Easy to add new methods
- ✅ **Snakemake-ready** - Each folder maps to rules
- ✅ **Simple** - Functions over classes, minimal abstraction

**Expected outcome:**
- ~4000 lines of overengineering deleted
- ~2000 lines of clean, reusable functions
- Snakemake handles orchestration
- Easy to understand and extend

**The goal: Boring, predictable code that works.**
