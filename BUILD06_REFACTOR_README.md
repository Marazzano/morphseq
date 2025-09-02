# Build06 Refactor Implementation (Refactor-010)

This document describes the completed implementation of Refactor-010: Standardize Morphological Embedding Generation.

## Overview

Build06 has been refactored to centralize morphological embedding generation around the **legacy latent store as source of truth**. Instead of generating embeddings for training subsets, it now aggregates per-experiment latents and merges them into df02 to produce canonical df03.

## Key Changes

### 1. New Service Module
- **Created**: `src/run_morphseq_pipeline/services/gen_embeddings.py`
- Contains all embedding-related functions with clean, testable APIs
- Handles model resolution, latent loading, merging, and validation

### 2. Updated CLI Interface
**Old Build06:**
```bash
python -m src.run_morphseq_pipeline.cli build06 --root <root> --train-name <name>
```

**New Build06:**
```bash
python -m src.run_morphseq_pipeline.cli build06 \
    --root <root> \
    --data-root <DATA_ROOT> \
    --model-name 20241107_ds_sweep01_optimum \
    [--experiments exp1 exp2 ...] \
    [--generate-missing-latents] \
    [--export-analysis-copies] \
    [--train-run <name> --write-train-output] \
    [--dry-run] [--overwrite]
```

### 3. Rewritten Build06 Step
- **File**: `src/run_morphseq_pipeline/steps/run_build06.py`
- Now delegates to `build_df03_with_embeddings()` orchestrator
- Supports full df02 aggregation instead of training-subset only

## Usage Examples

### Basic Usage (Minimal)
```bash
python -m src.run_morphseq_pipeline.cli build06 \
    --root /path/to/morphseq/project \
    --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq
```

### Generate Missing Latents
```bash
python -m src.run_morphseq_pipeline.cli build06 \
    --root /path/to/morphseq/project \
    --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
    --generate-missing-latents
```

### Export Analysis Copies
```bash
python -m src.run_morphseq_pipeline.cli build06 \
    --root /path/to/morphseq/project \
    --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
    --export-analysis-copies
```

### Dry Run (Test Mode)
```bash
python -m src.run_morphseq_pipeline.cli build06 \
    --root /path/to/morphseq/project \
    --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
    --dry-run
```

### Training Join Output
```bash
python -m src.run_morphseq_pipeline.cli build06 \
    --root /path/to/morphseq/project \
    --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
    --train-run my_training_run \
    --write-train-output
```

## File Paths

### Inputs
- **df02**: `<root>/metadata/combined_metadata_files/embryo_metadata_df02.csv`
- **Model**: `<DATA_ROOT>/models/legacy/<model_name>/`
- **Latents**: `<DATA_ROOT>/analysis/latent_embeddings/legacy/<model_name>/morph_latents_{experiment}.csv`

### Outputs
- **df03**: `<root>/metadata/combined_metadata_files/embryo_metadata_df03.csv`
- **Training join** (optional): `<root>/training_data/<train_run>/embryo_metadata_with_embeddings.csv`
- **Analysis copies** (optional): `<DATA_ROOT>/metadata/metadata_n_embeddings/<model_name>/df03_{experiment}.csv`

## Environment Variables

- **MORPHSEQ_DATA_ROOT**: Default data root if `--data-root` not provided

## Safety Features

- **Non-destructive by default**: Files are not overwritten unless `--overwrite` is specified
- **Coverage reporting**: Reports join coverage and warns if < 95%
- **Validation**: Checks for finite values in embedding columns
- **Dry run mode**: Preview actions without making changes

## Testing

Run the test script to validate the implementation:

```bash
python test_build06_simple.py --root /path/to/your/project --data-root /path/to/data --dry-run
```

## Dependencies

The refactor relies on existing infrastructure:
- `src.analyze.analysis_utils.calculate_morph_embeddings()` for generating missing latents
- Legacy model loading via `AutoModel.load_from_folder()`
- Existing df02 structure and metadata conventions

## Next Steps

1. **Install dependencies** (einops, torch, etc.) if missing
2. **Test with real data** using `--generate-missing-latents`
3. **Validate df03 output** for coverage and schema correctness
4. **Integrate with existing workflows**

## Architecture

```
Build06 CLI
    ↓
build_df03_with_embeddings() [orchestrator]
    ↓
├── resolve_model_dir()
├── ensure_latents_for_experiments()
│   └── calculate_morph_embeddings() [existing]
├── load_latents()
├── merge_df02_with_embeddings()
├── merge_train_with_embeddings() [optional]
└── export_df03_copies_by_experiment() [optional]
```

This refactor successfully implements the standardization goals while maintaining backward compatibility with existing research workflows.