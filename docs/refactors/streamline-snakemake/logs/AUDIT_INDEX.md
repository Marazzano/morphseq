# Snakemake Refactor Drift Audit - Index

**Audit Date:** 2025-11-10
**Auditor:** Claude Code (Session: claude/audit-snakemake-refactor-drift-011CUyL3xzfD6bFvuq61UKbT)
**Follow-up to:** PR from session_011CUqrQGuCEkKWvjcRgsyYr
**Purpose:** Systematically audit refactored code (`src/data_pipeline/`) against working implementation (`src/build/`, `src/run_morphseq_pipeline/`) and design docs

---

## Audit Scope

**Refactored Code:**
- `src/data_pipeline/` (new implementation)

**Working Implementation:**
- `src/build/` (legacy build scripts)
- `src/run_morphseq_pipeline/` (current pipeline orchestration)

**Design Documents:**
- `docs/refactors/streamline-snakemake/README.md`
- `docs/refactors/streamline-snakemake/processing_files_pipeline_structure_and_plan.md`
- `docs/refactors/streamline-snakemake/snakemake_rules_data_flow.md`
- `docs/refactors/streamline-snakemake/data_ouput_strcutre.md`

---

## Audit Checklist

### Phase 1: Metadata Input Validation ✅ COMPLETE

- [x] **plate_processing.py** - Plate metadata normalization
  - Log: `logs/phase1_plate_processing.md`
  - Status: ✅ Complete - More complete than working impl (adds series_map)
  - Drift: Moderate (column name normalization)

- [x] **keyence_scope_metadata.py** - Keyence scope metadata extraction
  - Log: `logs/phase1_keyence_scope_metadata.md`
  - Status: ✅ Complete - Identical core logic, enhanced features
  - Drift: Low (phase boundary shift is intentional)

- [x] **yx1_scope_metadata.py** - YX1 scope metadata extraction
  - Log: `logs/phase1_yx1_scope_metadata.md`
  - Status: ✅ Complete - Identical timestamp extraction, enhanced
  - Drift: None (core logic identical)

- [x] **series_well_mapper.py** - Series to well mapping
  - Log: `logs/phase1_series_well_mapper.md`
  - Status: ✅ Complete - YX1 extracted from inline, Keyence new
  - Drift: Low (extracted to module)

- [x] **align_scope_plate.py** - Scope and plate metadata alignment
  - Log: `logs/phase1_align_scope_plate.md`
  - Status: ✅ Complete - Identical join logic, enhanced validation
  - Drift: None (core logic identical)

- [x] **generate_image_manifest.py** - Experiment image manifest generation
  - Log: `logs/phase1_generate_image_manifest.md`
  - Status: ✅ Complete - New design feature (no working equivalent)
  - Drift: N/A (design innovation)

**Phase 1 Summary:** `logs/PHASE1_SUMMARY.md`

### Phase 2: Image Building

- [ ] **keyence/stitched_ff_builder.py** - Keyence image stitching
  - Log: `logs/phase2_keyence_stitched_ff_builder.md`
  - Status: Pending

- [ ] **keyence/z_stacking.py** - Keyence Z-slice focus stacking
  - Log: `logs/phase2_keyence_z_stacking.md`
  - Status: Pending

- [ ] **yx1/stitched_ff_builder.py** - YX1 image building
  - Log: `logs/phase2_yx1_stitched_ff_builder.md`
  - Status: Pending

### Phase 3: Segmentation

- [ ] **grounded_sam2/frame_organization_for_sam2.py** - SAM2 frame organization
  - Log: `logs/phase3_frame_organization_for_sam2.md`
  - Status: Pending

- [ ] **grounded_sam2/gdino_detection.py** - GroundingDINO detection
  - Log: `logs/phase3_gdino_detection.md`
  - Status: Pending

- [ ] **grounded_sam2/propagation.py** - SAM2 mask propagation
  - Log: `logs/phase3_propagation.md`
  - Status: Pending

- [ ] **grounded_sam2/mask_export.py** - SAM2 mask export
  - Log: `logs/phase3_mask_export.md`
  - Status: Pending

- [ ] **grounded_sam2/csv_formatter.py** - Segmentation tracking CSV
  - Log: `logs/phase3_csv_formatter.md`
  - Status: Pending

- [ ] **unet/inference.py** - UNet auxiliary masks
  - Log: `logs/phase3_unet_inference.md`
  - Status: Pending

### Phase 4: Snip Processing & Feature Extraction

- [ ] **snip_processing/extraction.py** - Snip extraction
  - Log: `logs/phase4_snip_extraction.md`
  - Status: Pending

- [ ] **snip_processing/rotation.py** - PCA rotation
  - Log: `logs/phase4_snip_rotation.md`
  - Status: Pending

- [ ] **snip_processing/augmentation.py** - CLAHE & augmentation
  - Log: `logs/phase4_snip_augmentation.md`
  - Status: Pending

- [ ] **feature_extraction/mask_geometry_metrics.py** - Mask geometry
  - Log: `logs/phase4_mask_geometry_metrics.md`
  - Status: Pending

- [ ] **feature_extraction/pose_kinematics_metrics.py** - Pose & kinematics
  - Log: `logs/phase4_pose_kinematics_metrics.md`
  - Status: Pending

- [ ] **feature_extraction/fraction_alive.py** - Fraction alive metric
  - Log: `logs/phase4_fraction_alive.md`
  - Status: Pending

- [ ] **feature_extraction/stage_inference.py** - Developmental stage (HPF)
  - Log: `logs/phase4_stage_inference.md`
  - Status: Pending

- [ ] **feature_extraction/consolidate_features.py** - Feature consolidation
  - Log: `logs/phase4_consolidate_features.md`
  - Status: Pending

### Phase 5: Quality Control

- [ ] **quality_control/auxiliary_mask_qc/imaging_quality_qc.py** - Imaging quality QC
  - Log: `logs/phase5_imaging_quality_qc.md`
  - Status: Pending

- [ ] **quality_control/auxiliary_mask_qc/death_detection.py** - Death detection
  - Log: `logs/phase5_death_detection.md`
  - Status: Pending

- [ ] **quality_control/segmentation_qc/segmentation_quality_qc.py** - Segmentation QC
  - Log: `logs/phase5_segmentation_quality_qc.md`
  - Status: Pending

- [ ] **quality_control/morphology_qc/size_validation_qc.py** - Size validation QC
  - Log: `logs/phase5_size_validation_qc.md`
  - Status: Pending

- [ ] **quality_control/consolidation/consolidate_qc.py** - QC consolidation
  - Log: `logs/phase5_consolidate_qc.md`
  - Status: Pending

### Phase 6+: Embeddings & Analysis

- [ ] **embeddings/** - Embedding generation modules
  - Log: `logs/phase6_embeddings.md`
  - Status: Pending

- [ ] **analysis_ready/assemble_features_qc_embeddings.py** - Final assembly
  - Log: `logs/phase6_assemble_features_qc_embeddings.md`
  - Status: Pending

### Schemas & Supporting Infrastructure

- [ ] **schemas/** - All schema definitions
  - Log: `logs/schemas_audit.md`
  - Status: Pending

- [ ] **identifiers/** - ID parsing utilities
  - Log: `logs/identifiers_audit.md`
  - Status: Pending

- [ ] **io/** - File I/O utilities
  - Log: `logs/io_audit.md`
  - Status: Pending

- [ ] **config/** - Configuration management
  - Log: `logs/config_audit.md`
  - Status: Pending

---

## Audit Progress

**Last Updated:** 2025-11-10
**Modules Audited:** 6 / 40+ (Phase 1 complete)
**Critical Issues Found:** 1 (column name normalization strategy)
**Blocking Issues:** 1 (column naming decision needed before Phase 2)

---

## Key Findings Summary

### Critical Issues
- 🚨 **Column Name Normalization** (plate_processing, align_scope_plate)
  - Refactor uses: `well_index`, `experiment_id`, `treatment`
  - Working uses: `well`, `experiment_date`, `chem_perturbation`
  - **DECISION NEEDED:** Normalized names vs backward compatibility
  - Impact: Affects ALL downstream modules

### Moderate Drift
- None detected (phase boundary shifts are intentional per design docs)

### Minor Drift / Documentation Mismatches
- None detected

### Modules to Deprecate
- None (all refactored modules are improvements)

### Missing Implementations in Working Code
- ❌ series_number_map extraction (plate_processing)
- ❌ Metadata validation (all Phase 1 modules)
- ❌ Channel normalization (scope metadata modules)
- ❌ Frame ordering validation (working has no manifest)
- ❌ BF channel presence validation
- ❌ Centralized image manifest

### New Design Features (Not Drift)
- ✅ generate_image_manifest.py - Single source of truth for images
- ✅ Schema validation throughout Phase 1
- ✅ Provenance tracking (series mapping)
- ✅ Channel normalization
- ✅ Frame interval computation
- ✅ Phase boundary shifts (metadata before image building)

---

## Next Steps

1. Begin systematic audit starting with Phase 1 (metadata processing)
2. Log findings continuously for each module
3. Update this index as modules are completed
4. Create consolidated action items document at end
