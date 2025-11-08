# MorphSeq Pipeline: Data Output Structure
**Status:** APPROVED - Aligned with Snakemake rules

This document defines the complete data output structure for the MorphSeq pipeline refactor. It is synchronized with:
- `preliminary_rules.md` - Snakemake rule definitions
- `**processing_files_pipeline_structure_and_plan**.md` - Module organization and implementation plan

---

## Directory Structure
**
{data_pipeline_root}/
│
├── inputs/                                           # ALL USER-PROVIDED INPUTS
│   ├── raw_image_data/                               # Raw microscope exports
│   │   ├── Keyence/{experiment_id}/
│   │   └── YX1/{experiment_id}/
│   │
│   ├── reference_data/
│   │   ├── surface_area_ref.csv                      # HPF stage reference (former stage_ref_df.csv)
│   │   └── perturbation_catalog.csv                  # Perturbation metadata (former perturbation_name_key.csv)
│   │
│   └── plate_metadata/                               # Per-experiment well annotations
│       └── {experiment_id}_plate_layout.xlsx
│
├── experiment_metadata/                              # VALIDATED EXPERIMENT CONTEXT (schema-backed)
│   └── {experiment_id}/
│       ├── plate_metadata.csv                        # REQUIRED_COLUMNS_PLATE_METADATA (experiment, well, genotype, temp, etc.)
│       ├── scope_metadata.csv                        # REQUIRED_COLUMNS_SCOPE_METADATA (microscope + timing calibration)
│       ├── scope_and_plate_metadata.csv              # REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA (joins prior two)
│       └── experiment_image_manifest.json            # REQUIRED_IMAGE_MANIFEST_* (per-well channel inventory, normalized frame list)
│
**├── models/                                           # PRE-TRAINED MODEL CHECKPOINTS (symlink targets)
│   ├── segmentation/
│   │   ├── unet/
│   │   │   ├── embryo_mask_v0_0100/
│   │   │   ├── viability_v1_0100/
│   │   │   ├── yolk_v1_0050/
│   │   │   ├── focus_v0_0100/
│   │   │   └── bubble_v0_0100/
│   │   ├── sam2/
│   │   └── grounding_dino/
│   └── embeddings/
│       ├── morphology_vae_2024/                      # Current production VAE
│       └── legacy/
├── identifiers/                                # Shared utilities
│   └── parsing.py                             # ID parsing (from parsing_utils.py)
│
├── segmentation/                                     # Segmentation outputs (per experiment) embryo_ids and snip_ids generated here
│   └── {experiment_id}/
│       ├── gdino_detections.json                     # GroundingDINO seed bounding boxes (per-well)
│       ├── sam2_raw_output.json                      # SAM2 tracked masks (nested: video/embryo/frame)
│       ├── segmentation_tracking.csv [VALIDATED]     # Flattened table: snip_id, embryo_id, well_id, mask_rle, is_seed_frame, source paths (embryo_id = {image_id}_{embryo_index (e.g. eNN)}, snip_id = {embryo_id}_t{frame_index})
│       ├── mask_images/                              # Integer-labeled PNG masks
│       │   └── {image_id}_masks.png
│       └── unet_masks/                               # UNet auxiliary masks for QC
│           ├── via/                                  # Viability/dead regions
│           │   └── {image_id}_via.png
│           ├── yolk/                                 # Yolk sac
│           │   └── {image_id}_yolk.png
│           ├── focus/                                # Out-of-focus regions
│           │   └── {image_id}_focus.png
│           ├── bubble/                               # Air bubbles
│           │   └── {image_id}_bubble.png
│           └── mask/                                 # UNet embryo (validation only, SAM2 is authoritative)
│               └── {image_id}_mask.png
│
├── processed_snips/                                  # Cropped embryo images + manifest
│   └── {experiment_id}/
│       ├── raw_crops/                                # Intermediate TIF crops (pre-rotation)
│       │   └── {snip_id}.tif
│       ├── processed/                                # Final JPEG crops (rotation + CLAHE + noise)
│       │   └── {snip_id}.jpg
│       └── snip_manifest.csv                         # REQUIRED_COLUMNS_SNIP_MANIFEST (paths, rotation angle, timestamps)
│
├── computed_features/                                # Feature extraction outputs (per snip_id)
│   └── {experiment_id}/
│       ├── mask_geometry_metrics.csv                 # SAM2 mask geometry w/ px→μm² conversions (feature_extraction/mask_geometry_metrics.py)
│       ├── pose_kinematics_metrics.csv               # Embryo pose + motion (feature_extraction/pose_kinematics_metrics.py)
│       ├── stage_predictions.csv                     # Predicted HPF + confidence (predict_developmental_stage.py)
│       └── consolidated_snip_features.csv            # Merge of segmentation_tracking + mask_geometry + pose_kinematics + stage (+ experiment & well IDs)
│
├── quality_control/                                  # QC assessments organised by dependency
│   └── {experiment_id}/
│       ├── segmentation_quality_qc.csv               # SAM2 mask integrity (edge, discontinuous, overlap flags)
│       ├── auxiliary_mask_qc.csv                     # UNet imaging quality (yolk, focus, bubble flags)
│       ├── embryo_death_qc.csv                        # fraction_alive + dead_flag (THE ONLY death flag source)
│       ├── surface_area_outliers_qc.csv              # Surface area outlier flags
│       └── consolidated_qc_flags.csv [VALIDATED]     # Merged QC flags + use_embryo_flag
│
├── latent_embeddings/                                # VAE latents (QC-passed `use_embryo == True`)
│   └── {model_name}/
│       ├── {experiment_id}_embedding_manifest.csv    # Filtered inputs + file metadata
│       └── {experiment_id}_latents.csv               # snip_id, embedding_model, z0 … z{dim-1}
│
├── analysis_ready/                                   # Final analysis tables (per experiment)
│   └── {experiment_id}/
│       └── features_qc_embeddings.csv                # REQUIRED_COLUMNS_ANALYSIS_READY (features + QC + plate/scope metadata + embeddings)
│                                                     # Includes `embedding_calculated` column for filtering and, when multi-model,
│                                                     # optional `embedding_model` metadata
│
└── (optional downstream tables handled per analysis)

---

## Key Naming Conventions

### **ID Hierarchy**
```
experiment_id                                     # User-defined experiment identifier
└── well_index                                    # e.g., A01, B02 (plate position)
    └── well_id = {experiment_id}_{well_index}   # Full well identifier
        └── channel_name                          # Normalized: BF, GFP, etc.
            └── frame_index                       # Temporal index (0-based)
                └── image_id = {well_id}_{channel_name}_t{frame_index}  # e.g., exp_A01_BF_t0000
                    └── embryo_index              # Embryo within frame (e.g., e01, e02)
                        └── embryo_id = {image_id}_{embryo_index}       # e.g., exp_A01_BF_t0000_e01
                            └── snip_id = {embryo_id}_t{frame_index}    # e.g., exp_A01_BF_t0000_e01_t0000
```

### **Channel Normalization**
All microscope-specific channel names are normalized during Phase 1 (`extract_scope_metadata`):

**YX1 Normalization:**
- "EYES - Dia" → "BF"
- "Empty" → "BF"
- "GFP" → "GFP"

**Keyence Normalization:**
- "Brightfield" → "BF"
- "bf" → "BF"
- "gfp" → "GFP"

**Valid normalized names:**
- Brightfield: `BF`, `Phase`
- Fluorescence: `GFP`, `RFP`, `CFP`, `YFP`, `mCherry`, `DAPI`

### **Validation Markers**
Files marked `[VALIDATED]` enforce schema on write:
- Column existence checks (all `REQUIRED_COLUMNS_*` present)
- Non-null validation (required fields cannot be empty)
- Schema alignment (correct data types, value ranges when applicable)

**Validated outputs:**
- Phase 1-2: `plate_metadata.csv`, `scope_metadata.csv`, `scope_and_plate_metadata.csv`, `experiment_image_manifest.json`
- Phase 3: `segmentation_tracking.csv`
- Phase 4: `snip_manifest.csv`
- Phase 5: `consolidated_snip_features.csv`
- Phase 6: `consolidated_qc_flags.csv`
- Phase 7: `{exp}_embedding_manifest.csv`, `{exp}_latents.csv`
- Phase 8: `features_qc_embeddings.csv`

---

## Cross-Reference

- **Snakemake Rules:** See `preliminary_rules.md` for rule-by-rule pipeline definition
- **Module Implementation:** See `processing_files_pipeline_structure_and_plan.md` for code organization
- **Schema Definitions:** All schemas in `src/data_pipeline/schemas/`
