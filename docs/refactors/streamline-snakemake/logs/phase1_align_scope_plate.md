# Audit Log: align_scope_plate.py

**Module:** `src/data_pipeline/metadata_ingest/mapping/align_scope_plate.py`
**Audit Date:** 2025-11-10
**Status:** ✅ COMPLETE

---

## Audit Notes / Scratchpad

Working implementation found in:
- `src/build/export_utils.py:build_experiment_metadata()` (lines 206-212)

Key observations:
- Core join logic is IDENTICAL: `pd.merge(scope_df, plate_df, on=["well", "experiment_date"], how="left")`
- Both validate that ALL scope rows have matching plate metadata
- Refactor normalizes column names first (working uses raw names)
- Refactor adds series_well_mapping integration
- Refactor outputs to TWO locations (Phase 1 aligned + legacy experiment_metadata/)

---

## Role / Intended Function

**Summary:** Join validated plate and scope metadata using series→well mapping

**Reference docs:**
- `processing_files_pipeline_structure_and_plan.md` - Lines 134-135
- `snakemake_rules_data_flow.md` - Lines 59-76 (rule align_scope_and_plate)

**Intended behavior:**
1. Load plate_metadata.csv (validated)
2. Load scope_metadata.csv (validated)
3. Load series_well_mapping.csv
4. Join scope + plate using well identifiers
5. Validate all scope rows have matching plate data
6. Validate against REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA
7. Output to TWO locations:
   - `input_metadata_alignment/{exp}/aligned_metadata/scope_and_plate.csv`
   - `experiment_metadata/{exp}/scope_and_plate_metadata.csv` (legacy path)

---

## Working Implementation

**File:** `src/build/export_utils.py:build_experiment_metadata()`
**Lines:** 206-212

```python
# Join scope metadata (meta_df) with plate metadata (plate_df)
meta_df = meta_df.merge(
    plate_df, on=["well","experiment_date"], how="left", indicator=True
)

# Strict validation: ALL scope rows must have matching plate metadata
if not (meta_df["_merge"]=="both").all():
    missing = meta_df.loc[meta_df["_merge"]!="both", ["well","experiment_date"]]
    raise ValueError(f"Missing well metadata for:\n{missing}")

meta_df = meta_df.drop(columns=["_merge"])

# Create well_id
meta_df["well_id"] = meta_df["experiment_date"] + "_" + meta_df["well"]
```

**Key invariants:**
1. **Join keys:** `["well", "experiment_date"]`
2. **Join type:** LEFT join (scope is left side)
3. **Validation:** STRICT - all scope rows must match plate rows
4. **Failure mode:** ValueError with list of missing wells
5. **well_id creation:** After join, format `{experiment_date}_{well}`

---

## Drift Summary

**Type(s) of drift:**
- [ ] Missing behavior (core join logic IDENTICAL ✅)
- [x] Incompatible signature (column names)
- [x] New feature (series_well_mapping integration)
- [x] New feature (dual output paths)
- [x] New feature (schema validation)

**Details:**

### 1. Core Join Logic (IDENTICAL ✅)
- **Working:** `merge(on=["well", "experiment_date"], how="left", indicator=True)`
- **Refactor:** Same logic (after column normalization)
- **Impact:** **NO DRIFT** in core joining behavior

### 2. Strict Validation (IDENTICAL ✅)
- **Working:** Raises ValueError if any scope row lacks plate match
- **Refactor:** Same validation
- **Impact:** **NO DRIFT** in validation logic

### 3. Column Name Normalization (BREAKING CHANGE)
- **Working:** Uses raw column names
  - Join keys: `["well", "experiment_date"]`
  - Output: `well`, `experiment_date`, `well_id`
- **Refactor:** Uses normalized column names
  - Join keys (after normalization): `["well_index", "experiment_id"]`
  - Output: `well_index`, `experiment_id`, `well_id`
- **Impact:**
  - Depends on upstream normalization (plate_processing, scope_metadata)
  - Downstream code must use normalized names
  - **Same issue as plate_processing.py** - need consistent strategy

### 4. Series Well Mapping Integration (NEW FEATURE)
- **Working:** No explicit series mapping table used
- **Refactor:** Reads `series_well_mapping.csv` to validate well assignments
- **Impact:**
  - Enables cross-validation of series → well assignments
  - Catches mismatches between scope and plate data

### 5. Dual Output Paths (NEW FEATURE)
- **Working:** Single output to `built_metadata_files/{exp}_metadata.csv`
- **Refactor:** TWO outputs:
  1. `input_metadata_alignment/{exp}/aligned_metadata/scope_and_plate.csv` (Phase 1 output)
  2. `experiment_metadata/{exp}/scope_and_plate_metadata.csv` (legacy compatibility)
- **Impact:**
  - Maintains backward compatibility
  - New pipeline uses Phase 1 output
  - Legacy code can still use experiment_metadata/

### 6. Schema Validation (NEW FEATURE)
- **Working:** No schema validation
- **Refactor:** Validates against `REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA`
- **Impact:** Catches malformed joins early

### 7. well_id Format (IDENTICAL AFTER NORMALIZATION)
- **Working:** `experiment_date + "_" + well`
- **Refactor:** `experiment_id + "_" + well_index`
- **Impact:** Same format (column names different but values same)

---

## Recommended Actions

### CRITICAL: Resolve column name normalization strategy
- [ ] **Same decision as plate_processing.py audit:**
  - Option A: Keep normalized names, update downstream code
  - Option B: Make align module output raw names for compatibility
- [ ] This blocks integration with legacy pipeline

### HIGH PRIORITY: Validate join keys
- [ ] Verify join keys match between scope and plate metadata
- [ ] Test on experiments with:
  - All wells present
  - Missing wells (should raise ValueError)
  - Extra plate wells (should be OK - left join)

### HIGH PRIORITY: Test series mapping integration
- [ ] Verify series_well_mapping.csv is loaded correctly
- [ ] Test validation of series → well assignments
- [ ] Check error messages for mismatches

### MEDIUM PRIORITY: Validate dual output paths
- [ ] Verify both output CSVs are written
- [ ] Confirm they have identical content
- [ ] Test legacy pipeline can still read `experiment_metadata/` path

### TESTING: End-to-end validation
- [ ] Run on Keyence experiment (implicit series mapping)
- [ ] Run on YX1 experiment (explicit series mapping)
- [ ] Compare output with working implementation metadata
- [ ] Verify schema compliance

---

## Priority

**CRITICAL** - This is the metadata alignment bottleneck. Column name normalization issue affects all downstream modules.

---

## Status

**Reviewed:** 2025-11-10
**Changes implemented:** No
**Blocker:** Column name normalization strategy (same as plate_processing.py)

---

## Conclusion

The refactored `align_scope_plate.py` has **IDENTICAL CORE LOGIC** to working implementation:
- ✅ **Join logic: IDENTICAL** (left join on well + experiment identifiers)
- ✅ **Validation: IDENTICAL** (strict checking, all scope rows must match)
- ✅ **well_id format: IDENTICAL** (after normalization)
- ✅ Adds series mapping validation (NEW)
- ✅ Adds dual output paths for backward compatibility (NEW)
- ✅ Adds schema validation (NEW)
- ⚠️  **Column name normalization** (same breaking change as plate_processing.py)

**Recommendation:** Keep refactored version. Core behavior is proven correct. Column name issue must be resolved consistently across all Phase 1 modules.

**No behavioral drift in core join logic** - only enhancement and normalization.

**Critical dependency:** Resolution of column name normalization strategy from plate_processing.py audit.
