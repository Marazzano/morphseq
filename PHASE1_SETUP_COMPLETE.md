# Phase 1 Setup Complete - 20250912 YX1 Dataset

**Date:** 2025-11-08
**Status:** ✅ Ready for testing

---

## What Was Set Up

### 1. Data Pipeline Output Structure

Created `data_pipeline_output/` with the following structure:

```
data_pipeline_output/
├── inputs/
│   ├── raw_image_data/
│   │   ├── YX1/ -> /net/trapnell/.../raw_image_data/YX1  (symlink)
│   │   └── Keyence/ -> /net/trapnell/.../raw_image_data/Keyence  (symlink)
│   ├── plate_metadata/
│   │   ├── 20250912_plate_layout.xlsx
│   │   └── 20250612_24hpf_ctrl_atf6_plate_layout.xlsx
│   ├── reference_data/  (empty, for future)
│   └── models/  (empty, for future)
├── experiment_metadata/  (outputs go here)
├── built_image_data/
├── segmentation/
├── processed_snips/
├── computed_features/
├── quality_control/
├── latent_embeddings/
└── analysis_ready/
```

**Key Points:**
- ✅ Raw image data symlinked from network (read-only, no copies)
- ✅ Plate metadata copied for 20250912 and atf6 experiments
- ✅ All output directories created

### 2. Snakefile Configuration

Updated `src/data_pipeline/pipeline_orchestrator/`:
- **config.yaml** - Set `data_root: "data_pipeline_output"` and `experiments: [20250912]`
- **Snakefile** - Updated to:
  - Use `data_pipeline_output` as data root
  - Added `phase1_extract_scope_metadata_yx1` rule for YX1 datasets
  - Fixed all output paths to match directory structure
  - Fixed Python f-string escaping for Snakemake

### 3. Phase 1 Rules Configured

The following rules are ready:
1. `phase1_normalize_plate_metadata` - Process Excel → validated CSV
2. `phase1_extract_scope_metadata_yx1` - Extract metadata from ND2 files
3. `phase1_map_series_to_wells` - Create series-to-well mapping
4. `phase1_align_scope_and_plate` - Join plate + scope metadata

**Expected Outputs:**
```
data_pipeline_output/experiment_metadata/20250912/
├── plate_metadata.csv
├── scope_metadata.csv
├── series_well_mapping.csv
└── scope_and_plate_metadata.csv
```

---

## How to Test

### Option 1: Use test script (recommended)
```bash
./test_phase1.sh
```

### Option 2: Manual commands
```bash
cd src/data_pipeline/pipeline_orchestrator

# Dry run to check DAG
snakemake -n phase1_complete

# Run Phase 1
snakemake --cores 1 phase1_complete

# Check outputs
ls -lh ../../../data_pipeline_output/experiment_metadata/20250912/
```

---

## Expected Behavior

### Successful Run:
1. **Plate metadata normalization** - Reads Excel, validates schema, writes CSV
2. **YX1 scope extraction** - Reads ND2 file metadata, extracts calibration/timing
3. **Series mapping** - Maps ND2 series numbers to well IDs
4. **Alignment** - Joins plate + scope metadata into final table

### Common Issues:

#### If you see schema validation errors:
- Check `src/data_pipeline/schemas/scope_and_plate_metadata.py`
- Known issue from consolidated action items: `embryo_id` should NOT be required in Phase 1 (it's generated in Phase 3)

#### If ND2 reading fails:
- Ensure `nd2` Python package is installed in your conda environment
- Check that ND2 file is accessible: `ls -lh data_pipeline_output/inputs/raw_image_data/YX1/20250912/`

#### If import errors occur:
- Ensure you're in the correct conda environment with all dependencies
- Path should be set correctly: `sys.path.insert(0, "{PROJECT_ROOT}/src")`

---

## Dataset Details

**20250912 YX1 Dataset:**
- **File:** `20250912_WT_tricane_serial_dilution_experiment.nd2` (1.5TB)
- **Location:** `/net/trapnell/.../raw_image_data/YX1/20250912/`
- **Plate metadata:** `metadata/plate_metadata/20250912_well_metadata.xlsx`
- **Built metadata exists:** Yes, at `/net/trapnell/.../built_metadata_files/20250912_metadata.csv` (10,736 rows)
- **Microscope:** YX1 (ND2 format, multi-series file)

---

## Next Steps After Phase 1 Success

1. **Review outputs:**
   ```bash
   head -20 data_pipeline_output/experiment_metadata/20250912/scope_and_plate_metadata.csv
   ```

2. **Validate against existing:**
   ```bash
   # Compare with old metadata
   wc -l /net/trapnell/.../built_metadata_files/20250912_metadata.csv
   wc -l data_pipeline_output/experiment_metadata/20250912/scope_and_plate_metadata.csv
   ```

3. **Check for critical blocking issues** (from consolidated action items):
   - Issue #1: Does schema require `embryo_id`? (it shouldn't)
   - Issue #5: YX1 series mapping working correctly?
   - Issue #7: CSV formatter includes all required columns?

4. **Move to Phase 2** (Image Building) - NOT YET IMPLEMENTED

---

## References

- **Docs:** `docs/refactors/streamline-snakemake/`
  - `README.md` - Overview
  - `snakemake_rules_data_flow.md` - Rule specifications
  - `data_ouput_strcutre.md` - Output structure
  - `DATA_INGESTION_AND_TESTING_STRATEGY.md` - This approach
- **Consolidated Action Items:** `docs/refactors/streamline-snakemake/logs/20251106_CONSOLIDATED_ACTION_ITEMS.md`
- **Schemas:** `src/data_pipeline/schemas/`

---

**Setup completed by:** Claude Code
**Ready for:** Phase 1 testing on 20250912 YX1 dataset
