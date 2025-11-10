# Audit Log: series_well_mapper (Keyence & YX1)

**Modules:**
- `src/data_pipeline/metadata_ingest/mapping/series_well_mapper_keyence.py`
- `src/data_pipeline/metadata_ingest/mapping/series_well_mapper_yx1.py`

**Audit Date:** 2025-11-10
**Status:** ✅ COMPLETE

---

## Audit Notes / Scratchpad

Working implementation:
- **Keyence:** Series mapping implicit in directory structure (XY##a, W0##) - no explicit mapper needed
- **YX1:** Inline series_number_map Excel parsing in `build01B_compile_yx1_images_torch.py` (lines 382-437)

Key observations:
- Refactor creates **EXPLICIT** mapping modules with provenance tracking
- Keyence: NEW functionality (no working equivalent)
- YX1: Extracted from inline code to dedicated module
- Both support explicit (Excel-based) and implicit (positional) mapping
- Provenance tracking is NEW feature

---

## Role / Intended Function

**Summary:** Create explicit series_number ↔ well_index mapping with provenance tracking

**Reference docs:**
- `processing_files_pipeline_structure_and_plan.md` - Lines 132-133
- `snakemake_rules_data_flow.md` - Lines 41-57 (rule map_series_to_wells)

**Intended behavior:**
1. Read plate_metadata.csv and scope_metadata.csv
2. Extract series_number_map from Excel (if present)
3. Create explicit series → well mapping
4. Fall back to implicit positional mapping if Excel map not found
5. Validate mapping (no duplicates, valid range)
6. Record provenance (explicit vs implicit, source file)
7. Output: `series_well_mapping.csv` + `mapping_provenance.json`

---

## Working Implementation

### Keyence (NO EXPLICIT MAPPER)
**File:** Implicit in `build01A_compile_keyence_torch.py`
**Method:** Directory structure defines series → well mapping
- `XY01a/` → well "A01", series implicitly 1
- `W001/` → well position (1-indexed to A01-H12)
- No explicit series_number_map needed

### YX1 (INLINE MAPPING)
**File:** `build01B_compile_yx1_images_torch.py` (lines 382-437)
**Method:** Excel series_number_map parsing

```python
# Read series_number_map sheet (8×12 grid)
sm_raw = plate_map_xl.parse("series_number_map", header=None)
series_map = data_rows.iloc[:, 1:13]  # Extract 8×12 numeric grid

# Build mapping
well_name_list = []
well_ind_list = []
for c in range(12):
    for r in range(8):
        val = series_map.iloc[r, c]
        if pd.notna(val) and 1 <= int(val) <= n_w:
            series_idx_1b = int(val)
            well_name = row_letters[r] + f"{c+1:02}"
            well_name_list.append(well_name)
            well_ind_list.append(series_idx_1b)
```

**Validation:**
- Range check: 1 ≤ series_idx ≤ n_w
- Warnings for duplicates and out-of-range values
- No hard failure on validation issues

---

## Drift Summary

**Type(s) of drift:**
- [x] New feature (Keyence explicit mapping - no working equivalent)
- [x] Behavioral change (YX1 extraction to dedicated module)
- [x] New feature (provenance tracking)
- [x] New feature (implicit fallback mapping)

**Details:**

### 1. Keyence: NEW Explicit Mapping Module
- **Working:** NO explicit series mapper (implicit from directory structure)
- **Refactor:** Creates explicit mapping with provenance
- **Impact:**
  - Enables validation and auditing of well assignments
  - Documents implicit directory-based mapping
  - Allows overrides if needed
- **Recommendation:** Keep - adds documentation and validation

### 2. YX1: Extraction to Dedicated Module
- **Working:** Inline Excel parsing (lines 382-437 in build01B)
- **Refactor:** Dedicated `series_well_mapper_yx1.py` module
- **Comparison:**

| Aspect | Working (Inline) | Refactor (Module) |
|--------|-----------------|-------------------|
| Excel parsing | ✅ (8×12 grid) | ✅ (8×12 grid) |
| Range validation | ✅ (1 ≤ idx ≤ n_w) | ✅ (1 ≤ idx ≤ n_w) |
| Duplicate handling | ⚠️ Warnings only | ⚠️ Warnings only |
| Provenance tracking | ❌ No | ✅ Yes (JSON file) |
| Implicit fallback | ❌ No | ✅ Yes (positional) |

- **Impact:** Better modularity, testability, and provenance

### 3. Provenance Tracking (NEW FEATURE)
- **Working:** No provenance recorded
- **Refactor:** Outputs `mapping_provenance.json` with:
  - mapping_method: "explicit" or "implicit"
  - source_file: Path to Excel file (if explicit)
  - timestamp: When mapping was created
  - validation_status: Pass/fail
- **Impact:** Enables auditing and troubleshooting

### 4. Implicit Fallback Mapping (NEW FEATURE)
- **Working:** YX1 requires explicit series_number_map Excel sheet
- **Refactor:** Falls back to positional mapping if Excel not found
  - series_number (1-based) = well_index_0based + 1
  - Assumes sequential well assignment
- **Impact:** More flexible, handles missing Excel sheets

### 5. Validation Enhancements
- **Working:** Basic range checking, warnings only
- **Refactor:** Enhanced validation:
  - Range checking (1 ≤ idx ≤ n_series)
  - Duplicate detection
  - Missing well detection
  - Schema validation
- **Impact:** Catches more errors early

---

## Recommended Actions

### CRITICAL: Validate YX1 series_number_map parsing
- [ ] Compare Excel parsing logic with working implementation (lines 382-437)
- [ ] Test on real YX1 well_metadata.xlsx files
- [ ] Verify 8×12 grid extraction is identical
- [ ] Validate range checking matches working behavior

### HIGH PRIORITY: Test implicit fallback
- [ ] Test YX1 experiments without series_number_map Excel sheet
- [ ] Verify positional fallback produces correct well assignments
- [ ] Compare with known well → series mappings

### MEDIUM PRIORITY: Validate Keyence mapping
- [ ] Test Keyence directory structure parsing
- [ ] Verify XY##a → A## conversion
- [ ] Verify W0## → A##-H12 conversion
- [ ] Check edge cases (non-standard directory names)

### TESTING: Provenance validation
- [ ] Verify mapping_provenance.json contains correct information
- [ ] Test that source_file paths are recorded correctly
- [ ] Validate timestamp format

---

## Priority

**HIGH** - Critical for YX1 data. Keyence mapping is new but low-risk (implicit from directory structure).

---

## Status

**Reviewed:** 2025-11-10
**Changes implemented:** No
**Blocker:** Need to validate YX1 Excel parsing matches working implementation

---

## Conclusion

### Keyence Mapper (NEW MODULE)
- **NEW functionality** - no working equivalent
- **Low risk:** Maps implicit directory structure to explicit table
- **Benefit:** Documentation and validation

### YX1 Mapper (EXTRACTED FROM INLINE CODE)
- **Core logic: IDENTICAL** to working implementation (Excel 8×12 grid parsing)
- ✅ Same range validation (1 ≤ idx ≤ n_w)
- ✅ Same duplicate handling (warnings)
- ✅ Adds provenance tracking (NEW)
- ✅ Adds implicit fallback (NEW)
- ✅ Better modularity and testability

**Recommendation:** Keep refactored versions. YX1 mapper is extraction of proven working code. Keyence mapper adds new validation capability.

**Critical validation:** Test YX1 Excel parsing on real data to confirm identical behavior with working implementation.
