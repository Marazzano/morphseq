### 🎯 SAM2 Integration - FULLY IMPLEMENTED! 

SAM2 segmentation is now a **first-class CLI citizen** with complete Python-based orchestration! The `sam2` subcommand runs the entire segmentation_sandbox pipeline and integrates seamlessly with the hybrid mask approach.

**Key Features:**
- ✅ Direct SAM2 pipeline execution via `sam2` subcommand
- ✅ Auto-discovery of SAM2 outputs by Build03  
- ✅ Hybrid approach: SAM2 embryo masks + Build02 QC masks
- ✅ Complete E2E orchestration with `--run-sam2` flag
- ✅ Batch processing for multiple experiments

---

# MorphSeq Centralized Runner

Centralized CLI to invoke MorphSeq pipeline steps (Build01→Build05) with a consistent, parameterized interface. Keeps “results/” runners optional by exposing a single module entrypoint.

Key features:
- **Complete pipeline**: `build01`, `build02`, `sam2`, `build03`, `build04`, `build05`, `e2e`, `validate`
- **SAM2 Integration**: Direct `sam2` subcommand with Python orchestration
- **Hybrid Segmentation**: SAM2 embryo masks + Build02 QC masks (yolk, focus, bubble, viability)
- **Auto-Discovery**: Build03 automatically finds SAM2 outputs
- **Enhanced Build02**: Runs all 5 UNet models (embryo, yolk, focus, bubble, viability)
- **E2E Orchestration**: Complete Build01→Build02→SAM2→Build03→Build04→Build05 flow

## Install/Run

- Run via module:
  - `python -m src.run_morphseq_pipeline.cli <subcommand> [args]`

## Data Structure & Path Conventions

### Build Pipeline Data
- Stitched FF images: `<data_root>/built_image_data/stitched_FF_images/{exp}/`
- Build02 QC masks: `<data_root>/segmentation/{model}_predictions/{exp}/`
  - `mask_v0_0100_predictions/` - embryo masks
  - `yolk_v1_0050_predictions/` - yolk masks  
  - `focus_v0_0100_predictions/` - focus masks
  - `bubble_v0_0100_predictions/` - bubble masks
  - `via_v1_0100_predictions/` - viability masks

### SAM2 Pipeline Data
- SAM2 root: `<data_root>/sam2_pipeline_files/`
- Exported masks: `<data_root>/sam2_pipeline_files/exported_masks/{exp}/`
- Metadata CSV: `<data_root>/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv`
- Pipeline files: `<data_root>/sam2_pipeline_files/detections/`, `/embryo_metadata/`

### Metadata Files
- Per-experiment metadata: `<data_root>/metadata/built_metadata_files/{exp}_metadata.csv`
- Combined df01: `<data_root>/metadata/combined_metadata_files/embryo_metadata_df01.csv`

## Subcommands

### Core Pipeline Steps

**`build01`** - Image stitching and metadata
```bash
python -m src.run_morphseq_pipeline.cli build01 \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --microscope keyence
```

**`build02`** - Complete QC mask suite (5 UNet models)
```bash  
python -m src.run_morphseq_pipeline.cli build02 \
  --data-root morphseq_playground \
  --mode legacy
```
Runs all 5 models: embryo, yolk, focus, bubble, viability

**`sam2`** - SAM2 segmentation pipeline ⭐ NEW!
```bash
# Single experiment
python -m src.run_morphseq_pipeline.cli sam2 \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --confidence-threshold 0.45 \
  --workers 8

# Batch mode (all experiments)  
python -m src.run_morphseq_pipeline.cli sam2 \
  --data-root morphseq_playground \
  --batch
```

**`build03`** - Embryo processing (hybrid masks)
```bash
# Auto-discovers SAM2 CSV or falls back to legacy
python -m src.run_morphseq_pipeline.cli build03 \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --by-embryo 5 --frames-per-embryo 3
```

**`build04`** - QC analysis and stage inference
```bash
python -m src.run_morphseq_pipeline.cli build04 \
  --data-root morphseq_playground
```

**`build05`** - Training data preparation  
```bash
python -m src.run_morphseq_pipeline.cli build05 \
  --data-root morphseq_playground \
  --train-name test_sam2_20250903
```

### End-to-End Orchestration

**`e2e`** - Complete pipeline with SAM2 ⭐ ENHANCED!
```bash
# Full pipeline with SAM2
python -m src.run_morphseq_pipeline.cli e2e \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --microscope keyence \
  --run-sam2 \
  --train-name test_sam2_20250903

# Legacy pipeline (no SAM2)
python -m src.run_morphseq_pipeline.cli e2e \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --microscope keyence \
  --train-name legacy_test_20250903
```

### Utility Commands

**`validate`** - Validation checks
```bash
python -m src.run_morphseq_pipeline.cli validate \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --checks schema,units,paths
```

## Key Features & Benefits

### 🎯 SAM2 Integration
- **Direct orchestration**: `sam2` subcommand runs complete segmentation_sandbox pipeline
- **Auto-discovery**: Build03 automatically finds SAM2 outputs at standard paths
- **Hybrid approach**: Superior SAM2 embryo masks + Build02 QC masks for complete analysis
- **Batch processing**: Process multiple experiments with `--batch` flag

### 🏗️ Enhanced Build02
- **Complete QC suite**: Runs all 5 UNet models (embryo, yolk, focus, bubble, viability)
- **Robust error handling**: Continues with partial success if some models fail
- **Quality control**: Enables full QC flag calculation including dead_flag from viability masks

### 🔄 End-to-End Orchestration
- **Full pipeline**: Build01→Build02→SAM2→Build03→Build04→Build05
- **Flexible control**: Skip any step with `--skip-*` flags
- **Legacy compatibility**: Can run without SAM2 for existing workflows
- **Progress tracking**: Clear step-by-step progress indicators

### 🔍 Validation & Quality
- **Pre-flight checks**: Validate inputs before pipeline execution
- **SAM2 validation**: Check outputs, mask files, and CSV integrity
- **Error handling**: Clear error messages and actionable feedback

## Environment Setup

```bash
# Activate the MorphSeq environment
conda activate mseq_data_pipeline_env

# Verify SAM2 sandbox is available
ls segmentation_sandbox/scripts/pipelines/
```

## Troubleshooting

- **SAM2 not found**: Ensure `segmentation_sandbox/` exists in repo root
- **Missing models**: Check Build02 model availability in conda environment  
- **Auto-discovery fails**: Manually provide `--sam2-csv` path to Build03
- **Permission errors**: Ensure write access to data root directory
