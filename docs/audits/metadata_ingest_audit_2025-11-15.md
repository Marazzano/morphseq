# Metadata Ingest Module Audit Report

**Date**: 2025-11-15  
**Baseline**: Working implementation in `src/build/`  
**Refactored**: New modules in `src/data_pipeline/metadata_ingest/`

---

## Executive Summary

This audit identifies **critical drift** between the refactored metadata_ingest modules and the working Build01 implementation. Several refactored modules make incompatible assumptions about data structures, file paths, and processing workflows that will cause failures when integrated.

**Priority Findings**:
- 🔴 **CRITICAL**: Image manifest assumes wrong directory structure
- 🔴 **CRITICAL**: Keyence scope metadata doesn't handle multi-tile z-stacks
- 🟡 **HIGH**: YX1 series mapping loses well assignment validation
- 🟡 **HIGH**: Missing P*/T* subdirectory hierarchy handling
- 🟢 **MEDIUM**: Plate processing compatible with minor adjustments

---

## 1. Plate Processing (`plate/plate_processing.py`)

### Intended Role
Process multi-sheet Excel plate metadata files, normalize column names, validate against schema, and output standardized CSV.

### Working Implementation Location
- **Primary**: `build_experiment_metadata()` in `src/build/export_utils.py:22-200`
- **Called from**: 
  - `build_ff_from_keyence()` line 317
  - `build_ff_from_yx1()` line 722

### Drift Analysis

#### ✅ **Compatible Features**
- Multi-sheet Excel parsing (8×12 plate layout)
- Standard sheets: medium, genotype, chem_perturbation, start_age_hpf, embryos_per_well, temperature
- Filters empty wells by `start_age_hpf`
- Handles both `well_metadata/` and `plate_metadata/` directories

#### ⚠️ **Behavioral Differences**
1. **Series Number Map Handling** (Lines 49-52)
   - **Refactored**: Extracts and saves `series_number_map` as separate CSV
   - **Working**: Doesn't extract series_number_map (handled later in YX1 pipeline)
   - **Impact**: Low - beneficial addition for YX1 data

2. **Error Messages** (Lines 40-72 vs export_utils:156-185)
   - **Refactored**: Generic error messages
   - **Working**: Detailed, user-friendly error messages explaining required sheets and format
   - **Impact**: Medium - reduced user experience

3. **Directory Search** (Lines 17-19 vs export_utils:28-69)
   - **Refactored**: Assumes input_file path provided directly
   - **Working**: Searches both `well_metadata/` and `plate_metadata/` with tolerance for name variations
   - **Impact**: Medium - less flexible for edge cases

4. **Column Normalization** (Lines 172-223)
   - **Refactored**: Uses explicit mapping dict
   - **Working**: Hardcoded column names
   - **Impact**: Low - refactored is more maintainable

#### 🔴 **Missing Functionality**
- None critical

### Recommended Actions
1. **Priority: MEDIUM** - Adopt detailed error messages from working implementation
2. **Priority: LOW** - Add directory search tolerance for non-standard filenames
3. **Priority: LOW** - Validate series_number_map extraction doesn't break YX1 pipeline

---

## 2. Keyence Scope Metadata (`scope/keyence_scope_metadata.py`)

### Intended Role
Extract microscope metadata from Keyence BZ-X TIFF files, including spatial/temporal calibration, channel info, and well assignments.

### Working Implementation Location
- **Primary**: `get_image_paths()` in `src/build/build01A_compile_keyence_torch.py:101-160`
- **Metadata scraper**: `scrape_keyence_metadata()` in `src/build/export_utils.py`
- **Output**: Returns `sample_list` (list of dicts with tile_zpaths) + `meta_df`

### Drift Analysis

#### 🔴 **CRITICAL: Incompatible Architecture**

**Problem 1: Multi-Tile Z-Stack Structure Lost**
- **Working** (lines 101-160): Builds `sample_dict` with `tile_zpaths` arrays
  ```python
  sample_dict[key] = {
      "tile_zpaths": [[str(im_files[idx]) for idx in pos_idx]],  # Nested list of tiles
      "well": well_conv,
      "time_id": t_idx,
  }
  ```
- **Refactored** (lines 205-333): Extracts metadata per individual TIFF
  ```python
  for tiff_path in tiff_files:
      meta = _scrape_keyence_metadata(tiff_path)  # One file at a time
  ```
- **Impact**: 🔴 **BREAKS BUILD01** - FF projection requires all tiles in a z-stack together

**Problem 2: Missing Directory Hierarchy**
- **Working**: Handles `XY*/P*/T*` subdirectory structure explicitly
  ```python
  pos_dirs = sorted(well_dir.glob("P*")) or [well_dir]
  for pos_dir in pos_dirs:
      time_dirs = sorted(pos_dir.glob("T*")) or [pos_dir]
  ```
- **Refactored** (line 144): Uses `rglob("*CH*.tif")` - flattens structure
- **Impact**: 🔴 **CRITICAL** - Loses position/time organization

**Problem 3: Sample Data Structure**
- **Working**: Returns `List[Dict]` where each dict is one well-timepoint with all tiles
- **Refactored**: Returns `pd.DataFrame` with one row per TIFF file
- **Impact**: 🔴 **INCOMPATIBLE** - Downstream FF projection expects grouped tiles

#### ⚠️ **Behavioral Differences**

1. **Well Extraction** (Lines 157-203)
   - **Working**: Derives from directory name (`XY01a` → `A01`)
   - **Refactored**: Extracts from path with fallbacks
   - **Impact**: Low - should be compatible if working correctly

2. **Cytometer Flag** (build01A line 302)
   - **Working**: Detects `W0*` vs `XY*` directories, sets `cytometer_flag`
   - **Refactored**: No explicit cytometer detection
   - **Impact**: Medium - may handle cytometer data incorrectly

3. **Time Index** (Lines 252-255)
   - **Working**: Parses from directory structure (`T0001`)
   - **Refactored**: Regex on filename
   - **Impact**: Low - should work if files named consistently

#### 🔴 **Missing Functionality**
1. **Multi-position subsampling** - Working code handles `P*` subdirs
2. **Time series grouping** - Working code groups by `T*` subdirs
3. **Tile assembly metadata** - Working code tracks which tiles belong together
4. **Resume capability** - Working code can skip already-processed frames (build01A:322-335)

### Recommended Actions
1. **Priority: CRITICAL** - Redesign to return sample_list structure matching working implementation
2. **Priority: CRITICAL** - Add P*/T* directory hierarchy parsing
3. **Priority: HIGH** - Group tiles by well-timepoint for FF projection
4. **Priority: MEDIUM** - Add cytometer_flag detection logic
5. **Consider**: May need to split into two modules: (1) metadata extraction, (2) sample list building

---

## 3. YX1 Scope Metadata (`scope/yx1_scope_metadata.py`)

### Intended Role
Extract microscope metadata from YX1 Nikon ND2 files, including well-to-series mapping from Excel.

### Working Implementation Location
- **Primary**: `build_ff_from_yx1()` in `src/build/build01B_compile_yx1_images_torch.py:284-738`
- **Key sections**:
  - Well mapping: lines 423-519
  - Timestamp extraction: lines 87-182
  - QC validation: lines 66-84, 566-576

### Drift Analysis

#### ✅ **Compatible Features**
1. **Timestamp Extraction** (Lines 31-103)
   - Matches working implementation's `_get_imputed_time_vector()` (build01B:87-182)
   - Handles NaN gaps, computes cycle time, imputes missing values
   - **Status**: ✅ Excellent match

2. **Channel Normalization** (Lines 106-136)
   - Similar logic to working implementation
   - Handles 'EYES - Dia', 'Empty' special cases
   - **Status**: ✅ Compatible

#### 🔴 **CRITICAL: Missing Well Mapping Logic**

**Problem 1: Series Number Map Not Read**
- **Working** (lines 423-471): Reads `series_number_map` sheet from Excel
  ```python
  series_map = data_rows.iloc[:, 1:13]  # 8x12 numeric grid
  for c in range(len(col_id_list)):
      for r in range(len(row_letter_list)):
          val = series_map.iloc[r, c]
          well_name = row_letter_list[r] + f"{col_id_list[c]:02}"
          well_ind_list.append(series_idx_1b)  # 1-based series index
  ```
- **Refactored** (lines 196-228): Generates sequential well indices
  ```python
  for w_idx in range(n_w):
      well_index = f"{w_idx:02d}"  # Just 00, 01, 02...
  ```
- **Impact**: 🔴 **CRITICAL** - Well IDs will be wrong for YX1 data

**Problem 2: No Stage Position QC**
- **Working** (lines 66-84, 566-576): Validates well assignments using KMeans clustering on stage XYZ
  ```python
  row_clusters = KMeans(n_clusters=len(row_index)).fit(stage_xyz_array[:, 1])
  assert np.all(row_letter_pd == row_letter_vec)
  ```
- **Refactored**: No validation of well assignments
- **Impact**: 🟡 **HIGH** - Silent well mapping errors won't be caught

**Problem 3: Missing Frame Arrays**
- **Working** (lines 535-562): Builds `stage_xyz_array`, `well_id_array`, `time_id_array`
  ```python
  well_id_array = np.empty((total_entries,), dtype=np.uint16)
  time_id_array = np.empty((total_entries,), dtype=np.uint16)
  ```
- **Refactored**: Creates DataFrame directly
- **Impact**: 🟡 **HIGH** - Different data structure for FF processing

#### ⚠️ **Behavioral Differences**

1. **Per-Well Frame Diagnostics** (build01B:489-515)
   - **Working**: Checks which timepoints are actually available per well
   - **Refactored**: Assumes all T×W frames exist
   - **Impact**: Medium - may fail on incomplete ND2 files

2. **Output Structure**
   - **Working**: Returns `well_df` with columns matching `sample_dict` structure
   - **Refactored**: Returns scope_metadata.csv with image-level rows
   - **Impact**: Medium - different downstream processing

3. **Z-Buffer Handling** (build01B:604)
   - **Working**: Special case for exp "20231206" (`z_buff=True`)
   - **Refactored**: No special handling
   - **Impact**: Low - may be obsolete

### Recommended Actions
1. **Priority: CRITICAL** - Add series_number_map parsing from Excel (copy from working impl)
2. **Priority: CRITICAL** - Generate correct well_index from series mapping, not sequential
3. **Priority: HIGH** - Add stage position QC validation
4. **Priority: MEDIUM** - Add per-well frame availability diagnostics
5. **Priority: MEDIUM** - Validate output structure matches downstream expectations

---

## 4. Keyence Series Well Mapper (`mapping/series_well_mapper_keyence.py`)

### Intended Role
Create explicit mapping between Keyence microscope series numbers and plate well positions.

### Working Implementation Location
- **Implicit**: Well names extracted inline in `get_image_paths()` (build01A:101-160)
- No separate mapping stage in working implementation

### Drift Analysis

#### ✅ **New Functionality**
- Refactored creates explicit series-to-well mapping CSV (not in working impl)
- Provides provenance tracking via JSON
- Validates mapping against plate and scope metadata

#### ⚠️ **Architectural Difference**

**Working Approach**: Inline well extraction
```python
well_name = Path(well_dir).name[-4:]  # e.g. 'A01a'
well_conv = sorted(well_dir.glob("_*"))[0].name[-3:]
```

**Refactored Approach**: Separate mapping stage
```python
def map_series_to_wells_keyence(...) -> pd.DataFrame:
    # Discover wells, count positions, create mapping CSV
```

**Implications**:
- 🟢 **Benefit**: Explicit mapping is more auditable
- 🟡 **Risk**: Adds pipeline stage that may be redundant
- 🟡 **Risk**: Different well name extraction may not match working code

#### 🔴 **Concerns**

1. **Position Counting** (Lines 74-100)
   - Refactored counts P* subdirs or image files
   - May not match working implementation's position handling
   - **Impact**: Medium - could create duplicate series entries

2. **Series Number Assignment** (Lines 148-162)
   - Refactored assigns sequential series numbers
   - Working code derives from directory structure
   - **Impact**: Medium - series numbers may not match

### Recommended Actions
1. **Priority: MEDIUM** - Validate series numbers match working implementation
2. **Priority: MEDIUM** - Cross-check position counting logic
3. **Priority: LOW** - Consider if separate mapping stage is necessary (could be done inline)

---

## 5. YX1 Series Well Mapper (`mapping/series_well_mapper_yx1.py`)

### Intended Role
Map YX1 ND2 series numbers to plate well positions using Excel metadata.

### Working Implementation Location
- **Primary**: Lines 423-519 in `build_ff_from_yx1()`
- Reads series_number_map from Excel, validates with KMeans

### Drift Analysis

#### ✅ **Core Logic Match**
- Both read series_number_map from Excel sheet
- Both build series → well_name mapping dict

#### 🔴 **CRITICAL: Missing Validation**

**Working Implementation** (build01B:472-486):
```python
# Log a concise mapping summary
pairs = sorted(zip(well_name_list, well_ind_list), key=lambda x: x[1])
log.info("YX1 series_number_map: selected_wells=%d, series_min=%d, series_max=%d", ...)

# QC well assignments with KMeans (lines 66-84)
_qc_well_assignments(stage_xyz_array, well_name_list_long)
```

**Refactored** (lines 50-91):
```python
def _build_implicit_mapping(...):
    # Build positional mapping
    mapping = {}
    for series_idx, well_index in enumerate(scope_wells):
        series_num = int(well_index) + 1
```

**Problems**:
1. **No explicit parsing** (Line 18-47)
   - Refactored checks for 'series_number_map' column in plate_df
   - Working code reads 'series_number_map' Excel **sheet** (8×12 grid)
   - **Impact**: 🔴 **CRITICAL** - Won't find the mapping

2. **Implicit mapping fallback** (Lines 50-91)
   - Creates sequential mapping if explicit not found
   - Working code would fail loudly if series_number_map missing
   - **Impact**: 🔴 **CRITICAL** - Silent incorrect mapping

3. **No stage position validation**
   - Working code validates with KMeans clustering
   - Refactored only checks for gaps/duplicates
   - **Impact**: 🟡 **HIGH** - Won't catch well assignment errors

#### ⚠️ **Behavioral Differences**

1. **Duplicate Detection** (Lines 172-177)
   - Refactored: Checks after building full mapping
   - Working (build01B:462-467): Prevents duplicates during construction
   - **Impact**: Low - both detect, different timing

2. **Output Format**
   - Refactored: Separate mapping CSV + provenance JSON
   - Working: Integrated into well_df
   - **Impact**: Medium - different downstream usage

### Recommended Actions
1. **Priority: CRITICAL** - Fix series_number_map parsing to read Excel **sheet**, not DataFrame column
2. **Priority: CRITICAL** - Remove implicit mapping fallback (should fail if no explicit map)
3. **Priority: HIGH** - Add stage position QC validation
4. **Priority: MEDIUM** - Align output format with working implementation

---

## 6. Align Scope Plate (`mapping/align_scope_plate.py`)

### Intended Role
Merge scope metadata and plate metadata into unified table.

### Working Implementation Location
- **Implicit**: Merge happens in `build_experiment_metadata()` (export_utils:103-110)
- Later used in FF building when combining metadata

### Drift Analysis

#### ✅ **Core Logic Compatible**
- Both use left merge on experiment_id and well_id
- Both validate merge completeness
- Both keep all scope metadata rows

#### ⚠️ **Architectural Difference**

**Working Approach**: Merge during metadata building
```python
# In build_experiment_metadata()
meta_df = build_experiment_metadata(...)  # Includes merge
# Later merged with scope metadata in FF building
```

**Refactored Approach**: Separate merge stage
```python
def align_scope_and_plate_metadata(...):
    merged_df = scope_df.merge(plate_df, on=['experiment_id', 'well_id'], ...)
```

**Implications**:
- 🟢 **Benefit**: Explicit merge is more transparent
- 🟢 **Benefit**: Better error handling (lines 65-71)
- 🟡 **Neutral**: Adds pipeline stage but improves modularity

#### 🔴 **Concerns**

1. **Timing of Merge** (Lines 48-57)
   - Refactored merges scope and plate metadata early
   - Working code merges at different point in pipeline
   - **Impact**: Low - should be compatible if schemas match

2. **Validation** (Lines 65-71)
   - Refactored checks for unmatched wells
   - Working code assumes merge will succeed
   - **Impact**: 🟢 **Improvement** - better error detection

### Recommended Actions
1. **Priority: LOW** - Validate merge timing doesn't affect downstream processing
2. **Priority: LOW** - Ensure schema compatibility with both scope metadata extractors

---

## 7. Generate Image Manifest (`manifests/generate_image_manifest.py`)

### Intended Role
Generate `experiment_image_manifest.json` - single source of truth for per-well, per-channel frame ordering.

### Working Implementation Location
- **No direct equivalent** - closest is `sample_dict` structure in `get_image_paths()`

### Drift Analysis

#### 🔴 **CRITICAL: Wrong Directory Structure**

**Assumed Structure** (Lines 88-90):
```python
# Directory structure:
# built_image_data/{exp}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{time_int:04d}.tif
```

**Actual Working Structure** (build01A:389-397):
```python
# Keyence FF tiles:
ff_dir = BUILT / "FF_images" / exp_name
all_paths = [
    Path(ff_dir) / f"ff_{w}_t{t:04}" / f"im_p{p:04}.jpg"  # Multi-tile structure
]

# Stitched images:
stitch_root / f"{well}_t{t:04}_stitch.jpg"  # Flat structure
```

**Impact**: 🔴 **CRITICAL** - File scanning will fail completely

#### 🔴 **Architecture Mismatch**

**Problem 1: Scans stitched images, not FF tiles**
- Refactored scans `stitched_ff_images/` (line 100)
- Build01 creates both FF tiles (`ff_{well}_t{time}/im_p{pos}.jpg`) AND stitched (`{well}_t{time}_stitch.jpg`)
- SAM2 pipeline needs **stitched** images, not individual tiles
- **Impact**: 🔴 **CRITICAL** - Looking in wrong location

**Problem 2: Channel organization**
- Refactored assumes `{well}/{channel}/` subdirectories
- Working code has flat structure with channel in filename
- **Impact**: 🔴 **CRITICAL** - Wrong directory structure

**Problem 3: Multi-tile metadata lost**
- Refactored treats each image independently
- Working code tracks which tiles belong together
- **Impact**: 🟡 **HIGH** - Can't reconstruct tile groupings

#### ⚠️ **Functional Issues**

1. **Frame Scanning** (Lines 130-143)
   - Filename parsing assumes `{well}_{channel}_t{time}.tif`
   - Working code uses `{well}_t{time}_stitch.jpg`
   - **Impact**: High - won't parse filenames correctly

2. **Image Dimensions** (Lines 145-148)
   - Placeholder implementation (None)
   - Should read from metadata CSV or image files
   - **Impact**: Medium - missing required data

### Recommended Actions
1. **Priority: CRITICAL** - Fix directory structure to scan actual stitched images:
   - Path should be: `built_image_data/stitched_FF_images/{exp}/{well}_t{time}_stitch.jpg`
2. **Priority: CRITICAL** - Update filename parsing for actual structure
3. **Priority: HIGH** - Populate image dimensions from scope metadata
4. **Priority: MEDIUM** - Consider if manifest is needed (working code doesn't use it)
5. **Priority: MEDIUM** - Clarify relationship to sample_dict structure

---

## Cross-Module Issues

### 1. Data Flow Incompatibility

**Working Pipeline**:
```
Raw Data → get_image_paths() → sample_dict (grouped tiles)
         → build_ff_from_keyence(sample_list) → FF projection
         → stitch_ff_from_keyence() → stitched images
```

**Refactored Pipeline** (assumed):
```
Raw Data → extract_keyence_scope_metadata() → scope_metadata.csv (flat)
         → map_series_to_wells() → series_mapping.csv
         → align_scope_and_plate() → unified_metadata.csv
         → generate_image_manifest() → manifest.json
```

**Incompatibility**: Refactored loses tile grouping structure needed for FF projection.

### 2. Schema Assumptions

**Issue**: Refactored modules assume schemas exist and are compatible, but:
- Working code doesn't use schema validation
- Column names may differ
- Data types may differ

**Risk**: Schema validation may reject valid working data.

### 3. Output Locations

**Working Code**:
- Metadata: `metadata/built_metadata_files/{exp}_metadata.csv`
- FF images: `built_image_data/Keyence/FF_images/{exp}/ff_{well}_t{time}/im_p{pos}.jpg`
- Stitched: `built_image_data/stitched_FF_images/{exp}/{well}_t{time}_stitch.jpg`

**Refactored Assumes**:
- Metadata: `experiment_metadata/{exp}/scope_metadata.csv`, `plate_metadata.csv`, etc.
- Images: `built_image_data/{exp}/stitched_ff_images/{well}/{channel}/...`

**Risk**: Path mismatches will break integration.

---

## Priority Matrix

| Module | Priority | Effort | Risk |
|--------|----------|--------|------|
| Keyence Scope Metadata | 🔴 CRITICAL | High | High |
| YX1 Scope Metadata (well mapping) | 🔴 CRITICAL | Medium | High |
| Image Manifest (path structure) | 🔴 CRITICAL | High | High |
| YX1 Series Mapper (Excel sheet) | 🔴 CRITICAL | Low | High |
| YX1 QC Validation | 🟡 HIGH | Medium | Medium |
| Keyence Series Mapper | 🟡 MEDIUM | Low | Low |
| Plate Processing (errors) | 🟢 MEDIUM | Low | Low |
| Align Scope Plate | 🟢 LOW | Low | Low |

---

## Recommendations

### Immediate Actions (Before Integration)

1. **Stop**: Do not integrate current refactored code into Build01 pipeline
2. **Fix Critical Path Issues**:
   - Redesign Keyence scope metadata to preserve tile grouping
   - Fix YX1 well mapping to read series_number_map Excel sheet
   - Correct image manifest directory structure
3. **Add Missing Validation**:
   - YX1 stage position QC
   - Per-well frame availability checks

### Architectural Decisions Needed

1. **Question**: Should metadata extraction produce DataFrames or sample_dict structures?
   - **Current**: Refactored uses DataFrames, working uses dicts
   - **Impact**: Affects all downstream processing
   - **Recommendation**: Need clarity on target architecture

2. **Question**: Is image manifest necessary?
   - **Current**: Refactored adds it, working code doesn't have it
   - **Impact**: Extra pipeline stage
   - **Recommendation**: Clarify if this is for SAM2 integration or general use

3. **Question**: Should series mapping be separate stage?
   - **Current**: Refactored separates, working code does inline
   - **Impact**: More modular but more complex
   - **Recommendation**: Acceptable if well-tested

### Testing Requirements

Before any integration:
1. **Unit tests**: Each extractor against known-good data
2. **Integration tests**: Full pipeline on 2-3 representative experiments
3. **Regression tests**: Compare outputs with working implementation
4. **Edge case tests**: Cytometer data, multi-position, missing frames

### Documentation Needs

1. Data flow diagrams showing:
   - Input → Output for each module
   - How modules connect
   - Where data structures transform
2. Schema documentation:
   - All column names and types
   - Required vs optional fields
   - Relationship to working implementation columns
3. Migration guide:
   - How to run refactored code
   - How to validate against working code
   - Rollback procedures

---

## Appendix: Column Name Mapping

### Working Implementation Columns

**Keyence** (`build01A` output):
- `well`, `time_string`, `time_int`, `Time (s)`, `Time Rel (s)`
- `Width (px)`, `Height (px)`, `Width (um)`, `Height (um)`
- `Objective`, `Channel`, `experiment_date`

**YX1** (`build01B` output):
- `well`, `nd2_series_num`, `time_int`, `Time (s)`, `Time Rel (s)`
- `Height (um)`, `Width (um)`, `Height (px)`, `Width (px)`
- `Objective`, `BF Channel`, `microscope`, `experiment_date`

### Refactored Schemas

**scope_metadata.csv**:
- `experiment_id`, `well_id`, `well_index`, `image_id`, `time_int`, `frame_index`
- `micrometers_per_pixel`, `image_width_px`, `image_height_px`
- `objective_magnification`, `frame_interval_s`, `experiment_time_s`
- `microscope_id`, `channel`, `z_position`

**plate_metadata.csv**:
- `experiment_id`, `well_id`, `well_index`
- `genotype`, `treatment`, `start_age_hpf`, `embryos_per_well`, `temperature`

**Mapping Notes**:
- `well` → `well_index`
- `Time (s)` → `experiment_time_s`
- `Width (px)` → `image_width_px`
- `experiment_date` → `experiment_id`
- New: `well_id`, `image_id`, `frame_index`

---

## Sign-off

**Audit Conducted By**: Claude Code (Sonnet 4.5)  
**Review Status**: ⚠️ **NOT READY FOR INTEGRATION**  
**Next Steps**: Address critical issues before proceeding

