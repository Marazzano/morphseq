# Audit Log: plate_processing.py

**Module:** `src/data_pipeline/metadata_ingest/plate/plate_processing.py`
**Audit Date:** 2025-11-10
**Status:** ✅ COMPLETE

---

## Audit Notes / Scratchpad

Working implementation found in `src/build/export_utils.py:build_experiment_metadata()` (lines 22-226).

Key observations:
- Both implementations use same 8×12 plate extraction logic
- Refactor adds series_number_map extraction (NEW feature for YX1 data)
- Refactor normalizes column names (working code uses raw names)
- Refactor adds schema validation
- Column name differences: `well` vs `well_index`, `experiment_date` vs `experiment_id`

---

## Role / Intended Function

**Summary:** Normalize and validate plate layout metadata from Excel/CSV files into schema-compliant format

**Reference docs:**
- `processing_files_pipeline_structure_and_plan.md` - Lines 125-127
- `snakemake_rules_data_flow.md` - Lines 10-24 (rule normalize_plate_metadata)

**Intended behavior:**
1. Read multi-sheet Excel files (8×12 plate format)
2. Extract sheets: medium, genotype, chem_perturbation, start_age_hpf, embryos_per_well, temperature
3. Optionally extract series_number_map sheet
4. Flatten 96-well plate data into long-format CSV
5. Normalize column names to match schema
6. Validate against REQUIRED_COLUMNS_PLATE_METADATA
7. Output: `plate_metadata.csv`

---

## Working Implementation

**File(s):** `src/build/export_utils.py:build_experiment_metadata()`
**Lines:** 22-226
**Entry points:** Called from build scripts (build01A_compile_keyence_torch.py, build01B_compile_yx1_images_torch.py)

**How it's invoked:**
```python
meta_df = build_experiment_metadata(repo_root, exp_name, meta_df)
```

**Key invariants / expectations:**
1. **Sheet list (hardcoded):** `["medium", "genotype", "chem_perturbation", "start_age_hpf", "embryos_per_well", "temperature"]`
2. **File location search:**
   - First: `metadata/well_metadata/{exp_name}_well_metadata.xlsx`
   - Fallback: `metadata/plate_metadata/{exp_name}_well_metadata.xlsx`
3. **8×12 extraction:** `df.iloc[:8, 1:13]` for each sheet
4. **Numeric sheets:** `temperature`, `embryos_per_well` → parsed as float
5. **String sheets:** All others → parsed as string (except start_age_hpf which can be either)
6. **Empty well filtering:** Rows with empty `start_age_hpf` are dropped
7. **Output columns (raw names):**
   - `well` (e.g., "A01")
   - `experiment_date` (e.g., "20240509_24hpf")
   - `medium`, `genotype`, `chem_perturbation`, `start_age_hpf`, `embryos_per_well`, `temperature`
8. **well_id generation:** `experiment_date + "_" + well`
9. **NO series_number_map extraction** in working version

---

## Drift Summary

**Type(s) of drift:**
- [x] Incompatible signature (column names)
- [x] Behavioral change (new feature: series_number_map)
- [x] Behavioral change (schema validation)

**Details:**

### 1. Column Name Normalization (MODERATE DRIFT)
- **Working:** Uses raw column names from Excel: `well`, `experiment_date`, `chem_perturbation`, etc.
- **Refactor:** Normalizes to schema names: `well_index`, `experiment_id`, `treatment`, etc.
- **Impact:**
  - Any code expecting `well` will break (should use `well_index`)
  - Any code expecting `experiment_date` will break (should use `experiment_id`)
  - `chem_perturbation` → `treatment` rename could break downstream code
- **Location of change:** `_normalize_column_names()` lines 172-223

### 2. Series Number Map Extraction (NEW FEATURE)
- **Working:** Does NOT extract or save series_number_map
- **Refactor:** Extracts series_number_map sheet if present, saves to `series_number_map.csv`
- **Impact:**
  - This is a REQUIRED feature for YX1 microscope data (per docs)
  - Working implementation is INCOMPLETE - missing critical YX1 support
  - Refactor is CORRECT behavior
- **Location of change:** Lines 113-119, 49-52

### 3. Schema Validation (NEW FEATURE)
- **Working:** No validation - silently proceeds with whatever columns exist
- **Refactor:** Validates against `REQUIRED_COLUMNS_PLATE_METADATA` schema
- **Impact:**
  - Refactor will FAIL FAST on malformed/incomplete data
  - Working code silently passes bad data downstream (causes failures later)
  - Refactor is CORRECT behavior per design docs
- **Location of change:** Line 67

### 4. File Format Support (NEW FEATURE)
- **Working:** Excel (.xlsx) only
- **Refactor:** Supports Excel (.xlsx, .xls) AND CSV (.csv)
- **Impact:**
  - More flexible input handling
  - No breaking changes (still supports Excel)
- **Location of change:** Lines 44-60

### 5. Skip Sheets Handling (MINOR DIFFERENCE)
- **Working:** Implicitly skips non-existent sheets (fills with np.nan)
- **Refactor:** Explicitly skips 'Export Summary' and handles series_number_map separately
- **Impact:** Minimal - same end result
- **Location of change:** Lines 99-123

### 6. Well ID Format (IDENTICAL)
- **Working:** `experiment_date + "_" + well`
- **Refactor:** `experiment_id + "_" + well_index` (after normalization, same values)
- **Impact:** None (column names normalized but values same)
- **Location of change:** Lines 63-64

---

## Recommended Actions

### CRITICAL: Address Column Name Incompatibility
- [ ] **Decision needed:** Align refactored column names with working implementation OR update all downstream code?
  - **Option A (Recommended):** Keep refactor's normalized names, update downstream code
    - Rationale: Schema-backed column names are clearer and more maintainable
    - Impact: Must update all code expecting `well`, `experiment_date`, `chem_perturbation`
  - **Option B:** Make refactor output raw column names to match working code
    - Rationale: Zero breaking changes
    - Impact: Schemas must use raw names (less ideal)

### HIGH PRIORITY: Validate series_number_map extraction
- [ ] Test series_number_map extraction on real YX1 data
- [ ] Verify saved CSV format matches expectations of `series_well_mapper.py`
- [ ] Confirm this sheet exists in YX1 well metadata Excel files

### MEDIUM PRIORITY: Update working implementation
- [ ] Add series_number_map extraction to `build_experiment_metadata()` if needed before deprecation
- [ ] Add schema validation to catch bad data earlier

### TESTING: Validate refactor against real data
- [ ] Test on Keyence experiment Excel files (without series_number_map)
- [ ] Test on YX1 experiment Excel files (with series_number_map)
- [ ] Verify empty well filtering works correctly
- [ ] Confirm numeric vs string sheet parsing matches working behavior

---

## Priority

**HIGH** - This is Phase 1, foundational module. Column name incompatibility will cascade to all downstream code.

---

## Status

**Reviewed:** 2025-11-10
**Changes implemented:** No
**Blocker:** Need decision on column name normalization strategy

---

## Conclusion

The refactored `plate_processing.py` is **MORE COMPLETE** than the working implementation:
- ✅ Adds critical series_number_map extraction for YX1 data
- ✅ Adds schema validation for early error detection
- ✅ Adds CSV input support for flexibility
- ⚠️  Normalizes column names (breaking change, but GOOD change)

**Recommendation:** Keep refactored version, update downstream code to use normalized column names.

**Critical Question for User:**
> Should we align the refactor to use raw column names (`well`, `experiment_date`) to maintain backward compatibility, or update all downstream code to use normalized names (`well_index`, `experiment_id`)?
