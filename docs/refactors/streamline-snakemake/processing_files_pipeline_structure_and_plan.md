# MorphSeq Pipeline Refactor: Final Structure & Implementation Plan

**Author:** Claude Code Analysis
**Date:** 2025-10-06
**Status:** APPROVED - READY FOR IMPLEMENTATION

---

## Pipeline Tasks Overview

The MorphSeq pipeline processes zebrafish embryo timelapse data through these major tasks:

```
METADATA NORMALIZATION & VALIDATION
  Plate layout Excel → normalized plate_metadata.csv (schema-backed)
  Microscope headers → scope_metadata.csv (per-microscope extractors, schema-backed)
  Join plate + scope → scope_and_plate_metadata.csv (shared, schema-backed)

IMAGE PREPROCESSING
  Raw microscope data → Stitched FF images + metadata
  (Keyence or YX1 microscope-specific processing)

SEGMENTATION
  Stitched images → Embryo masks + auxiliary masks
  - SAM2: Embryo detection, tracking, propagation (PRIMARY METHOD)
  - UNet: Auxiliary masks (yolk, bubble, focus, viability)

SNIP PROCESSING & FEATURE EXTRACTION
  Masks + images → Cropped snips + SAM2-derived features
  - Extract embryo regions and assign stable snip_id
  - Compute mask-geometry metrics + pose/kinematics metrics from SAM2 masks and tracking table
  - Infer developmental stage (HPF)
  - Consolidate per-snip features into one table

QUALITY CONTROL
  Features + masks → QC flags grouped by dependency
  - Auxiliary mask QC (UNet viability & imaging signals)
  - Segmentation QC (SAM2-only validation + tracking metrics)
  - Morphology QC (feature-based surface area outliers)

QC CONSOLIDATION
  Merge all QC flags → consolidated_qc_flags + use_embryo gating

EMBEDDING GENERATION
  QC-approved snips (`use_embryo == True`) → Latent embeddings
  (VAE-based, note Python 3.9 subprocess)

ANALYSIS-READY TABLE
  Features + QC flags + embeddings → features_qc_embeddings.csv
  (`embedding_calculated` column for downstream filtering)
```

**Key Principles:**
- Step boundaries stay explicit: preprocessing → segmentation → feature extraction → QC → QC consolidation → embeddings → analysis-ready hand-off
- Schema-backed validation at every consolidation point: REQUIRED_COLUMNS_* live in `src/data_pipeline/schemas/` and each writer enforces column existence + non-null.
- Plate metadata normalization is isolated: `metadata/plate_processing.py` standardizes Excel inputs, microscopes export only scope metadata, and `preprocessing/consolidate_plate_n_scope_metadata.py` performs the shared join.
- The experiment image manifest (`metadata/generate_image_manifest.py`) is the single source of truth for per-well, per-channel frame ordering; all segmentation rules consume `experiment_metadata/{exp}/experiment_image_manifest.json`.
- `consolidated_snip_features.csv` is the single feature source for every QC module (no duplicate joins)
- Surface-area metrics must be converted to `area_um2` using microscope metadata before downstream use (no pure pixel-area logic)
- QC modules are grouped by dependency (`auxiliary_mask_qc`, `segmentation_qc`, `morphology_qc`), and their merge is tracked in `consolidated_qc_flags.csv`
- `use_embryo_flags.csv` is the only gate for embeddings; no rule reaches back into individual QC tables
- `features_qc_embeddings.csv` contains everything analysis notebooks need, with an `embedding_calculated` helper column when embeddings lag the QC outputs
- SAM2 remains the **authoritative segmentation method**; UNet masks exist strictly for QC/auxiliary logic
- `segmentation_tracking.csv` (SAM2) replaces the legacy tracking table and retains mask_rle, is_seed_frame, source paths, and well identifiers for downstream validation.

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

├── schemas/                                     # REQUIRED_COLUMNS_* definitions (imported everywhere)
│   ├── __init__.py
│   ├── plate_metadata.py
│   ├── scope_metadata.py
│   ├── scope_and_plate_metadata.py
│   ├── segmentation.py
│   ├── snip_processing.py
│   ├── features.py
│   ├── quality_control.py
│   └── analysis_ready.py
│
├── metadata/                                   # Metadata operations
│   ├── plate_processing.py                    # Normalize Excel layouts → plate_metadata.csv (schema-backed)
│   ├── generate_image_manifest.py            # Build experiment_image_manifest.json (channel-normalized frame inventory)
│   └── enrichment.py                          # Perturbation metadata merging
│
├── preprocessing/                              # Raw → stitched images + scope metadata
│   ├── keyence/
│   │   ├── stitching.py                       # Tile stitching logic
│   │   ├── z_stacking.py                      # Z-slice focus stacking
│   │   └── extract_scope_metadata.py          # Keyence scope metadata extractor (schema-backed)
│   ├── yx1/
│   │   ├── processing.py                      # YX1-specific processing
│   │   └── extract_scope_metadata.py          # YX1 scope metadata extractor (schema-backed)
│   └── consolidate_plate_n_scope_metadata.py  # Join validated plate + scope metadata (shared)
│
├── segmentation/                               # Segmentation
│   ├── grounded_sam2/                          # SAM2 embryo tracking (PRIMARY)
│   │   ├── frame_organization_for_sam2.py     # Organize manifest-derived frames into SAM2 temp dirs
│   │   ├── gdino_detection.py                 # Grounded DINO embryo detection
│   │   ├── propagation.py                     # SAM2 mask propagation
│   │   ├── mask_export.py                     # Export masks to PNG
│   │   └── csv_formatter.py                   # Flatten JSON → segmentation_tracking.csv (mask_rle, source paths, seed flag)
│   ├── unet/                                  # UNet auxiliary masks
│   │   ├── inference.py                       # Core inference pipeline
│   │   └── model_loader.py                    # Load 5 models (mask, via, yolk, focus, bubble)
│   └── mask_utilities.py                      # Shared RLE/polygon/bbox utilities (for example redonling unet derived masks and sam2 derived masks )
│
├── snip_processing/                            # Snip extraction utilities
│   ├── extraction.py                          # Crop embryo regions from images
│   ├── rotation.py                            # PCA-based rotation alignment
│   └── io.py                                  # Save snip images + manifest helpers (schema-backed)
│
├── feature_extraction/                         # SAM2-derived feature computations
│   ├── mask_geometry_metrics.py               # Area, perimeter, contour stats + px→μm² conversion → mask_geometry_metrics.csv
│   ├── pose_kinematics_metrics.py             # Centroid, bbox, orientation, deltas → pose_kinematics_metrics.csv
│   ├── stage_inference.py                     # HPF (developmental stage) prediction
│   └── consolidate_features.py                # Assemble consolidated_snip_features.csv (imports REQUIRED_COLUMNS_FEATURES)
│
├── quality_control/                            # Quality Control signals
│   ├── auxiliary_mask_qc/
│   │   ├── imaging_quality_qc.py              # Frame, yolk, focus, bubble flags (from UNet)
│   │   └── embryo_viability_qc.py             # fraction_alive + dead_flag (UNet viability + SAM2)
│   ├── segmentation_qc/
│   │   ├── segmentation_quality_qc.py         # SAM2 mask quality checks
│   │   └── tracking_metrics_qc.py             # Movement speed, trajectory validation
│   ├── morphology_qc/
│   │   └── size_validation_qc.py              # Surface area outlier detection
│   └── consolidation/
│       ├── consolidate_qc.py                  # Merge all QC CSVs per snip_id (imports REQUIRED_COLUMNS_QC)
│       └── compute_use_embryo.py              # Apply gating logic → use_embryo_flags.csv
│
├── embeddings/                                 # Latent Embeddings (QC-passed snips only)
│   ├── inference.py                           # VAE embedding generation
│   ├── subprocess_wrapper.py                  # Python 3.9 subprocess orchestration
│   └── file_validation.py                     # Check existing embeddings
│
├── analysis_ready/                             # Final analysis hand-off helpers
│   └── assemble_features_qc_embeddings.py     # Join features + QC + embeddings (imports REQUIRED_COLUMNS_ANALYSIS_READY)
│
├── pipeline_orchestrator/                     # Snakemake entry point & helpers
│   ├── Snakefile                             # Task DAG importing data_pipeline modules
│   ├── config/                               # Snakemake defaults + execution profiles
│   ├── experiment_discovery.py               # Resolve experiment lists (override → inventory → glob)
│   └── cli.py                                # Thin wrapper for launching the workflow
├── identifiers/                                # Shared utilities
│   └── parsing.py                             # ID parsing (from parsing_utils.py)
│
└── io/                                         # File I/O utilities
    ├── loaders.py                             # Load images, masks, CSVs
    ├── savers.py                              # Save outputs
    └── validators.py                          # Validate file formats
```

---

## Module Descriptions

### **schemas/**
**Purpose:** Centralize `REQUIRED_COLUMNS_*` lists used by consolidation steps. Each schema module is imported by its writer to enforce column presence and non-null constraints (see data_validation_plan).

- `plate_metadata.py`, `scope_metadata.py`, `scope_and_plate_metadata.py`: Guard metadata normalization before it reaches downstream logic.
- `segmentation.py`, `snip_processing.py`, `features.py`: Define expectations for SAM2 outputs, snip manifests, and consolidated feature tables.
- `quality_control.py`, `analysis_ready.py`: Keep QC merges and final analysis hand-off aligned with the validation contract.

---

### **preprocessing/**
**Purpose:** Convert raw microscope data into standardized stitched FF images **and** extract scope metadata needed for downstream schema validation.

**Microscope-specific modules:**
- **keyence/**: Keyence BZ-X800 processing (tile stitching, z-stacking) plus `extract_scope_metadata.py` for CSV export aligned with `REQUIRED_COLUMNS_SCOPE_METADATA`.
- **yx1/**: YX1 microscope processing plus `extract_scope_metadata.py` tailored to the YX1 header formats.

**Shared joiner:**
- `consolidate_plate_n_scope_metadata.py` merges validated plate + scope CSVs (microscope-agnostic) and enforces `REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA`.

**Why separate?** Different microscopes have different:
- File formats
- Tile arrangements
- Metadata structures
- Stitching requirements

**Future-proof:** Easy to add new microscope types while keeping shared validation logic untouched.

---

### **segmentation/**
**Purpose:** Detect and track embryos, generate masks

**Execution model:** Every SAM2 tracking job operates **per well**. Upstream Snakemake rules will eventually emit an `image_manifest` (future task) that enumerates stitched frames for a given well in chronological order. For now, the segmentation modules assume they receive the list of frame file paths and metadata for exactly one well at a time.

#### **grounded_sam2/** - SAM2 + GroundingDINO pipeline (PRIMARY)
- **frame_organization_for_sam2.py**: Reorganize well/timepoint images into video frame sequences
- **gdino_detection.py**: Grounded DINO detection for seed annotations
- **propagation.py**: SAM2 mask propagation (forward/bidirectional)
- **bounding_box_utils.py**: Convert GroundingDINO detections into SAM2 prompt boxes
- **mask_export.py**: Export integer-labeled PNG masks
- **csv_formatter.py**: Flatten nested JSON to schema-aligned `segmentation_tracking.csv` (adds mask_rle, source paths, well_id, `is_seed_frame`)

**Flow detail:**
- `gdino_detection.py` runs first, producing detections for a single well (its own model + step) using the per-channel frame lists from `experiment_image_manifest.json`.
- `frame_organization_for_sam2.py` takes the per-well frame list from `experiment_image_manifest.json` and enforces the strict ordering SAM2 expects by symlinking frames into a temporary directory (no classes—just context-manager helpers). Each temp directory corresponds to the subset of frames we are about to process (seed→end for forward, reversed 0→seed for backward).
- The SAM2 propagation code (`propagation.py`) consumes the temp directory plus GDINO seed boxes to run inference. `propagate_forward()` stores results keyed by the original frame indices by passing a `start_index`; `propagate_bidirectional()` slices the frame list twice, runs forward propagation on each slice, and remaps backward results with `original_idx = seed_idx - offset` before merging. The modules stay separate, but Snakemake still calls them within one `rule_track`.
- Downstream helpers (`mask_export.py`, `csv_formatter.py`) simply read the SAM2 outputs and export masks/metadata.

**Core modules in play:**
- `frame_organization_for_sam2.py`
  - Consumes the per-well frame list and creates the SAM2-compliant temp directory + JSON manifest.
  - Handles forward/backward ordering quirks needed for bidirectional propagation.
- `gdino_detection.py`
  - Runs Grounded DINO to produce seed detections (`initial_detections.json`) used to prompt SAM2.
  - Outputs include bounding boxes persisted via `bounding_box_utils.py`.
- `propagation.py`
  - Invokes SAM2 with the organized frames + GDINO seeds.
  - Handles per-well propagation loops and writes raw SAM2 outputs to disk.
- `mask_export.py`
  - Converts the SAM2 tensor outputs into labeled PNG masks per frame.
- `csv_formatter.py`
  - Flattens SAM2 metadata into `segmentation_tracking.csv` with schema-aligned columns (`mask_rle`, `source_image_path`, `well_id`, `is_seed_frame`).
- `bounding_box_utils.py`
  - Shared helper for translating GDINO detections into SAM2 prompt boxes and for any coordinate conversions across the pipeline.

**Minimal functional interface (no classes):**

```python
@contextmanager
def sam2_frame_dir(frame_paths: list[Path]):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        for idx, src in enumerate(frame_paths):
            (tmp_dir / f"{idx:05d}.jpg").symlink_to(src)
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

def propagate_forward(predictor, temp_dir: Path, start_index: int,
                      seed_boxes, embryo_ids, verbose=False):
    state = predictor.init_state(video_path=str(temp_dir))
    predictor.add_new_points_or_box(state, frame_idx=0, box=seed_boxes, obj_id=..., ...)
    raw = {}
    for frame_offset, obj_ids, mask_logits in predictor.propagate_in_video(state):
        raw[start_index + frame_offset] = decode_masks(obj_ids, mask_logits, embryo_ids)
    return raw

def propagate_bidirectional(predictor, frame_paths: list[Path], seed_idx: int,
                            seed_boxes, embryo_ids, verbose=False):
    with sam2_frame_dir(frame_paths[seed_idx:]) as tmp_fwd:
        forward = propagate_forward(
            predictor, tmp_fwd,
            start_index=seed_idx,
            seed_boxes=seed_boxes,
            embryo_ids=embryo_ids,
            verbose=verbose,
        )

    if seed_idx == 0:
        return forward

    with sam2_frame_dir(frame_paths[:seed_idx + 1][::-1]) as tmp_rev:
        backward_raw = propagate_forward(
            predictor, tmp_rev,
            start_index=0,
            seed_boxes=seed_boxes,
            embryo_ids=embryo_ids,
            verbose=verbose,
        )

    backward = {seed_idx - offset: data for offset, data in backward_raw.items()}
    return merge_results(forward, backward)  # prefers forward when both exist
```

`propagate_forward` always receives a `start_index`, so offset `0` from SAM2 maps back to the real frame index (the seed frame in the forward pass, the seed frame in reversed space for backward). When we do the reverse slice, we remap offsets with `seed_idx - offset` before merging, keeping everything keyed by true chronology.

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
**Purpose:** Extract and align embryo crops ahead of feature/QC stages

**Core operations:**
- **extraction.py**: Crop embryo regions with padding using SAM2 masks
- **rotation.py**: PCA-based alignment to standard orientation
- **augmentation.py**: CLAHE, background noise injection, and edge blending for model-ready crops
- **manifest_generation.py**: Scan processed snips, attach metadata, and write schema-validated manifest

**snip_id management:**
- `snip_id` is assigned during extraction using `segmentation_tracking.csv`
- Format: `{embryo_id}_s{frame:04d}` (e.g., `embryo_001_s0005`)
- Manifest resides in `processed_snips/{experiment_id}/snip_manifest.csv` (schema-backed) combining raw + processed paths

---

### **feature_extraction/**
**Purpose:** Derive per-snip metrics directly from SAM2 masks and tracking outputs

**Modules:**
- **mask_geometry_metrics.py**: Area, perimeter, contour stats derived from SAM2 masks with pixel-size metadata to produce `area_um2` → `mask_geometry_metrics.csv`
- **pose_kinematics_metrics.py**: Centroid, orientation, bbox geometry and frame deltas derived from `segmentation_tracking.csv` → `pose_kinematics_metrics.csv`
- **fraction_alive.py**: Aggregates UNet viability masks to compute per-snip viability fraction → `fraction_alive.csv`
- **stage_inference.py**: HPF prediction + confidence using surface-area reference curves (requires `area_um2`)
- **consolidate_features.py**: Merge segmentation_tracking + mask_geometry + pose_kinematics + fraction_alive + stage data (plus plate/scope metadata columns) into `consolidated_snip_features.csv` with schema validation

**Key principle:** Features operate on masks/tracking metadata (not raw snip pixels); the merge output is the single source consumed by all QC modules.

---

### **quality_control/**
**Purpose:** Validate data quality with dependency-scoped subpackages

#### **auxiliary_mask_qc/**
- **imaging_quality_qc.py**: Boundary, yolk, focus, and bubble flags computed from UNet auxiliary masks with SAM2 geometry for proximity
- **embryo_viability_qc.py**: Ingests `fraction_alive` from `feature_extraction/fraction_alive.py`, thresholds it to produce the single canonical `dead_flag`, and computes `death_inflection_time_int`. All downstream rules consume this `dead_flag`; no other module emits alternate death flags.

#### **segmentation_qc/**
- **segmentation_quality_qc.py**: Mask integrity checks for SAM2 output (edge contact, overlaps, area drift, disconnected components, etc.)
- **tracking_metrics_qc.py**: Trajectory smoothness, speed outliers, and ID persistence derived from SAM2 tracking data

#### **morphology_qc/**
- **size_validation_qc.py**: Surface-area outlier detection leveraging consolidated features (SA outlier remains a critical QC signal)

#### **consolidation/**
- **consolidate_qc.py**: Row-wise merge of all QC CSVs (keys on `snip_id`) into `consolidated_qc_flags.csv`, enforcing `REQUIRED_COLUMNS_QC` + `QC_FAIL_FLAGS`.
- **compute_use_embryo.py**: Applies gating logic to produce `use_embryo_flags.csv`, the single contract consumed by embeddings and analysis

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

### **analysis_ready/**
**Purpose:** Assemble the final analysis table per experiment

- **assemble_features_qc_embeddings.py**: Join `consolidated_snip_features.csv`, `consolidated_qc_flags.csv`, `use_embryo_flags.csv`, and embedding latents.
- Adds `embedding_calculated` boolean for rows missing embeddings (e.g., re-runs or alternate models).
- Validates against `REQUIRED_COLUMNS_ANALYSIS_READY` (features + QC + plate/scope metadata + embeddings) so downstream notebooks can assume a consistent schema.
- Optional `embedding_model` metadata when multiple latent spaces are mixed; default pipeline produces one CSV per experiment in `analysis_ready/{experiment_id}/`.

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
**Purpose:** Handle experiment-level metadata outside of Snakemake orchestration.

- **plate_processing.py**: Normalize raw plate layout Excel files, fill experiment IDs, and write `plate_metadata.csv` using `REQUIRED_COLUMNS_PLATE_METADATA`.
- **generate_image_manifest.py**: Read `scope_and_plate_metadata.csv` + stitched images, normalize channel names, and emit `experiment_image_manifest.json` validated by `schemas/image_manifest.py`.
- **enrichment.py**: Merge perturbation metadata, genotype/phenotype mappings (downstream of validated plate metadata)

---

### **io/**
**Purpose:** File I/O utilities shared across pipeline

- **loaders.py**: Load images, masks, CSVs, JSONs
- **savers.py**: Save outputs with proper formatting
- **validators.py**: Validate file formats, schemas

---


---

## 5-Week Implementation Plan

### **Week 1: Move Core Utilities & Critical Verifications**
**Goal:** Validate approach with minimal changes + verify critical assumptions

**Move as-is:**
1. `segmentation_sandbox/scripts/utils/parsing_utils.py` → `identifiers/parsing.py` (~800 lines)
2. `src/build/qc_utils.py` → `quality_control/auxiliary_mask_qc/imaging_quality_qc.py` (135 lines)
3. `src/data_pipeline/quality_control/death_detection.py` → `quality_control/auxiliary_mask_qc/embryo_viability_qc.py` (317 lines)
4. `segmentation_sandbox/scripts/utils/mask_utils.py` → `segmentation/mask_utilities.py` (~200 lines)
5. `src/analyze/gen_embeddings/*.py` → `embeddings/*.py` (~300 lines total)

**Create shared infrastructure:**
6. `config/runtime.py` with `resolve_device(prefer_gpu: bool) -> torch.device`
   - Single implementation reused by all preprocessing modules
   - Handles CUDA availability checking, fallback to CPU
   - Supports string inputs ("cuda", "cpu", "auto")
7. `src/data_pipeline/schemas/` package with placeholder `REQUIRED_COLUMNS_*` lists for plate, scope, segmentation, snip, features, QC, analysis-ready tables, plus seeded `channel_normalization.py` and `image_manifest.py`.
8. `metadata/plate_processing.py` scaffold that reads Excel layouts, normalizes column names, and emits schema-aligned `plate_metadata.csv` (without full edge-case handling yet).
9. `metadata/generate_image_manifest.py` stub that walks stitched images, applies channel normalization, and writes schema-aligned `experiment_image_manifest.json`.

**Critical verifications (BLOCKING for Week 2):**
10. **Verify UNet post-processing assumption:**
   - Read `build02B_segment_bf_main.py` carefully
   - Confirm all 5 models (mask, via, yolk, focus, bubble) use identical post-processing
   - Document any differences found
   - If assumption is wrong, revise Week 2 module structure plan

11. **Document microscope filename issues:**
   - Audit Keyence and YX1 filename patterns in raw data
   - Document ad-hoc naming variations discovered
   - Create normalization strategy for `config/microscopes.py`

**Update imports in 2-3 existing scripts to verify**

**Deliverable:**
- Core utilities importable from new locations, all tests pass
- Device resolver ready for use
- UNet verification complete (green/red light for Week 2)
- Image manifest scaffolding in place (channel normalization + schema hook)
- Microscope filename issues documented

---

### **Week 2: Extract Preprocessing & UNet Modules**

**Preprocessing**
- Extract from `build01A_compile_keyence_torch.py` → `preprocessing/keyence/` (stitching + `extract_scope_metadata.py` that emits schema-aligned CSV)
- Extract from `build01B_compile_yx1_images_torch.py` → `preprocessing/yx1/` (processing + `extract_scope_metadata.py` for YX1 headers)
- Implement shared `preprocessing/consolidate_plate_n_scope_metadata.py` joiner that consumes validated plate + scope metadata and enforces `REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA`
- Wire up `metadata/generate_image_manifest.py` to read consolidated metadata + stitched images and emit `experiment_image_manifest.json`

**UNet segmentation (auxiliary masks)**
- Extract from `build02B_segment_bf_main.py` → `segmentation/unet/`
- **Verify:** All 5 models use same post-processing (just different checkpoints)

**Create Snakemake rules for preprocessing + UNet masks**

**Deliverable:** Can stitch images, publish schema-validated scope metadata, emit experiment_image_manifest.json, and run UNet via Snakemake

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
- Rename output CSV to `segmentation_tracking.csv` and add schema-required fields (`mask_rle`, `source_image_path`, `well_id`, `is_seed_frame`) before validation.

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

**Snip processing & feature extraction**
- Extract from `build03A_process_images.py` (1753 lines → ~200 lines across focused modules)
  - `snip_processing/extraction.py`
  - `snip_processing/rotation.py`
  - `snip_processing/augmentation.py`
  - `snip_processing/manifest_generation.py`
  - `feature_extraction/mask_geometry_metrics.py`
  - `feature_extraction/pose_kinematics_metrics.py`
  - `feature_extraction/fraction_alive.py`
  - `feature_extraction/stage_inference.py`
  - `feature_extraction/consolidate_features.py` (schema-backed joiner for consolidated_snip_features)

**Quality control**
- Extract from `build04_perform_embryo_qc.py` into dependency-scoped packages:
  - `quality_control/auxiliary_mask_qc/imaging_quality_qc.py`
  - `quality_control/auxiliary_mask_qc/embryo_viability_qc.py`
  - `quality_control/segmentation_qc/tracking_metrics_qc.py`
  - `quality_control/segmentation_qc/segmentation_quality_qc.py`
  - `quality_control/morphology_qc/size_validation_qc.py`
  - `quality_control/consolidation/consolidate_qc.py`
  - `quality_control/consolidation/compute_use_embryo.py`

**Deliverable:** Build03 and Build04 run via Snakemake with schema-validated manifests, feature tables, and QC merges

---

### **Week 5: Workflow Wiring & Cleanup**
- Ensure Snakemake rules enforce QC gating before embeddings (use `use_embryo` filters) and finish the `combine_features_qc_embeddings` hand-off
- Wire `analysis_ready/assemble_features_qc_embeddings.py` to import `REQUIRED_COLUMNS_ANALYSIS_READY` and fail fast on schema drift
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

### ✅ **Schema-Backed Validation**
- `src/data_pipeline/schemas/` provides REQUIRED_COLUMNS_* lists consumed by every consolidation point.
- Writers fail fast on missing/empty columns, catching schema drift (plate metadata, scope metadata, segmentation tracking, snip manifests, features, QC, analysis-ready).

### ✅ **Single-Source Tables**
- `consolidated_snip_features.csv` joins SAM2 tracking data with mask_geometry/pose_kinematics/stage metrics
- `consolidated_qc_flags.csv` + `use_embryo_flags.csv` provide the only QC inputs for embeddings/analysis
- `analysis_ready/{experiment_id}/features_qc_embeddings.csv` adds `embedding_calculated` so downstream consumers can filter when embeddings lag behind QC
- `segmentation_tracking.csv` replaces legacy tracking tables and carries mask_rle + provenance columns needed for downstream joins.

### ✅ **Explicit, Descriptive Names**
- Long names OK if they eliminate ambiguity
- `quality_control/auxiliary_mask_qc/imaging_quality_qc.py` > `spatial_qc.py`
- `segmentation/grounded_sam2/frame_organization_for_sam2.py` > `video_prep.py`

### ✅ **Functions Over Classes**
- Extract pure functions from overengineered classes
- Delete annotation management, entity tracking systems
- Let Snakemake handle orchestration

### ✅ **Logical Hierarchies**
- `feature_extraction/` - SAM2-derived per-snip metrics + consolidation
- `quality_control/auxiliary_mask_qc|segmentation_qc|morphology_qc` - QC grouped by dependency footprint
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
