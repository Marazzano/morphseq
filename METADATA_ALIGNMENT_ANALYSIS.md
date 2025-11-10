# Current Plate-Scope Metadata Alignment Architecture

## Overview
The working implementation joins plate metadata (from Excel sheets) with scope metadata (from microscope systems) on a simple identifier: `well` + `experiment_date`.

---

## Data Flow

```
┌─────────────────────────────┐
│  SCOPE METADATA             │
│  (from microscope systems)  │
└──────────────┬──────────────┘
               │
        ┌──────▼──────────────────┐
        │ Keyence:                │
        │ • scrape_keyence_metadata()
        │ • get_image_paths()     │
        │ → meta_rows DataFrame   │
        │                         │
        │ YX1:                    │
        │ • Parse ND2 file        │
        │ • Read series_number_map│
        │ → well_df DataFrame     │
        └──────┬─────────────────┘
               │
        ┌──────▼─────────────────────────────┐
        │ build_experiment_metadata()         │
        │ • Merges on [well, experiment_date]│
        │ • Creates well_id identifier       │
        │ • Validates join success           │
        └──────┬─────────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ COMBINED METADATA               │
        │ • Plate design info             │
        │ • Microscope settings           │
        │ • Spatial/temporal coordinates  │
        └──────────────────────────────────┘
```

---

## 1. SCOPE METADATA EXTRACTION

### Keyence Metadata (build01A_compile_keyence_torch.py)

**Source**: `get_image_paths()` function, lines 101-160

**Key Identifier Mapping**:
- **Well ID**: `well_conv = sorted(well_dir.glob("_*"))[0].name[-3:]`
  - Extracted from Keyence folder structure (e.g., "WA01a" → "A01")
  - Parsed from directory naming convention

- **Time Index**: `t_idx = int(time_dir.name[1:]) if time_dir.stem.startswith("T") else 0`
  - Extracted from "T*" directory structure
  - Named `time_int` in metadata (line 138)
  - Named `time_id` in sample_dict (line 154) - **NOTE: INCONSISTENCY**

**Extracted Metadata Columns** (via `scrape_keyence_metadata()`):
```python
{
    'Time (s)': float,              # Converted from 100-nanosecond units
    'Objective': str,               # Lens name
    'Channel': str,                 # Observation type
    'Width (px)': int,              # Image width in pixels
    'Height (px)': int,             # Image height in pixels
    'Width (um)': float,            # Image width in micrometers
    'Height (um)': float,           # Image height in micrometers
    'well': str,                    # Added manually (e.g., 'A01')
    'time_string': str,             # Added manually (e.g., 'T0000')
    'time_int': int,                # Added manually (time index)
}
```

**Output**: DataFrame with one row per well-timepoint
```
   Time (s)  Objective  Channel  Width (px)  ...  well  time_string  time_int
0    123.45      20x      BF         2560  ...   A01       T0000         0
1    234.56      20x      BF         2560  ...   A01       T0001         1
2    345.67      20x      BF         2560  ...   A02       T0000         0
...
```

---

### YX1 Metadata (build01B_compile_yx1_images_torch.py)

**Source**: ND2 file + series_number_map Excel sheet

**Key Identifier Mapping**: Series → Well via Excel sheet (lines 382-423)
```python
# series_number_map Excel sheet: 8 rows × 12 columns (A01-H12 layout)
# Each cell contains a 1-based series index in the ND2 file
# Example:
#        Col1  Col2  Col3  ...  Col12
# Row 0:   1     2     3   ...   12    (well A01, A02, A03, ..., A12)
# Row 1:  13    14    15   ...   24    (well B01, B02, B03, ..., B12)
# ...
```

**Mapping Validation** (lines 409-419):
```python
if series_idx_1b < 1 or series_idx_1b > n_w:
    # Skip out-of-range series indices
    
if series_idx_1b in used_series:
    # Skip duplicate mappings
```

**Extracted Metadata Columns** (lines 534-549):
```python
well_df = pd.DataFrame({
    'well': str,                    # e.g., 'A01'
    'nd2_series_num': int,          # 1-based series index
    'microscope': str,              # 'YX1'
    'time_int': int,                # 0..n_t-1 for each selected well
    'Height (um)': float,           # Y dimension in micrometers
    'Width (um)': float,            # X dimension in micrometers
    'Height (px)': int,             # Y dimension in pixels
    'Width (px)': int,              # X dimension in pixels
    'BF Channel': int,              # Brightfield channel index
    'Objective': str,               # Objective name
    'Time (s)': float,              # Frame timestamp
})
```

**Output**: DataFrame with one row per well-timepoint
```
   well  nd2_series_num  microscope  time_int  Height (um)  ...  Time (s)
0   A01               1       YX1          0         1024.0  ...     0.0
1   A01               1       YX1          1         1024.0  ...    10.5
2   A02               2       YX1          0         1024.0  ...     0.0
...
```

---

## 2. PLATE METADATA EXTRACTION

**Source**: Excel file at `metadata/well_metadata/{exp_name}_well_metadata.xlsx`

**Required Sheets** (lines 115, 130-149):
```python
well_sheets = [
    "medium",                   # Culture medium conditions
    "genotype",                 # Genotype information
    "chem_perturbation",       # Chemical treatment/perturbation
    "start_age_hpf",           # Starting age in hours post-fertilization
    "embryos_per_well",        # Number of embryos per well
    "temperature"              # Incubation temperature
]
```

**Sheet Format**:
- Each sheet is an 8 rows × 12 columns plate layout (A-H rows, 1-12 columns)
- Header row is skipped (or auto-detected)
- Data is extracted from `[0:8, 1:13]` slice (8 rows, 12 columns)
- Empty cells are filled with NaN or empty string

**Assembled Plate DataFrame** (lines 125-153):
```python
plate_df = pd.DataFrame({
    'well': ['A01', 'A02', ..., 'H12'],  # All 96 wells
    'experiment_date': exp_name,          # Experiment identifier
    'medium': [str],                      # From medium sheet
    'genotype': [str],                    # From genotype sheet
    'chem_perturbation': [str],          # From chem_perturbation sheet
    'start_age_hpf': [str or float],     # From start_age_hpf sheet
    'embryos_per_well': [float],         # From embryos_per_well sheet
    'temperature': [float],              # From temperature sheet
})
```

**Output**: DataFrame with one row per well (96 rows for 96-well plate)
```
   well  experiment_date medium     genotype  ...  start_age_hpf  embryos_per_well  temperature
0   A01      20250101_exp    E3      wild_type ...           24.0                1           28.0
1   A02      20250101_exp    E3      wild_type ...           24.0                1           28.0
...
96  H12      20250101_exp    E3      wild_type ...           24.0                1           28.0
```

**Filtering** (line 153):
- Rows where `start_age_hpf` is empty are dropped
- This naturally filters to only wells with experiments

---

## 3. METADATA JOINING / ALIGNMENT

**Location**: `build_experiment_metadata()` function (lines 206-218)

**Join Operation**:
```python
meta_df = meta_df.merge(
    plate_df,
    on=["well", "experiment_date"],
    how="left",
    indicator=True
)
```

**Join Keys**:
1. `well`: String identifier for plate position (e.g., "A01", "H12")
2. `experiment_date`: Experiment name/date string

**Join Type**: LEFT JOIN
- Keeps all rows from scope metadata (left)
- Adds matching rows from plate metadata (right)
- Unmatched scope rows get NaN for plate columns

**Validation** (lines 209-212):
```python
if not (meta_df["_merge"] == "both").all():
    missing = meta_df.loc[meta_df["_merge"] != "both", ["well", "experiment_date"]]
    raise ValueError(f"Missing well metadata for:\n{missing}")
meta_df = meta_df.drop(columns=["_merge"])
```

**Invariants**:
- ✅ STRICT: ALL scope metadata rows must have matching plate metadata
- ✅ STRICT: No unmatched rows allowed (raises ValueError)
- ✅ Assumes 1:many relationship (1 plate row → many scope rows for different timepoints)

---

## 4. WELL_ID CREATION

**Location**: Line 214 of export_utils.py

**Creation Logic**:
```python
meta_df["well_id"] = meta_df["experiment_date"] + "_" + meta_df["well"]
```

**Examples**:
- `20250101_exp_A01`
- `20250101_exp_H12`

**Purpose**: Unique identifier for (experiment, well) pair across all timepoints

**Column Ordering** (lines 217-218):
- `well_id` is moved to the first column (leftmost)
- All other columns follow

---

## 5. SERIES → WELL MAPPING (YX1-Specific)

**Location**: build01B_compile_yx1_images_torch.py, lines 381-423

**Mapping Process**:

1. **Read series_number_map from Excel**:
   ```python
   sm_raw = plate_map_xl.parse("series_number_map", header=None)
   series_map = sm_raw.iloc[:, 1:13]  # 8x12 grid
   ```

2. **Parse 1-based series indices**:
   ```python
   for c in range(12):  # columns
       for r in range(8):   # rows
           val = series_map.iloc[r, c]
           series_idx_1b = int(val)  # 1-based index
           well_name = row_letter_list[r] + f"{col_id_list[c]:02}"
   ```

3. **Validate**:
   ```python
   # Check range: 1 ≤ series_idx ≤ n_w
   if series_idx_1b < 1 or series_idx_1b > n_w:
       log.warning("Skipping out-of-range series...")
       continue
   
   # Check no duplicates
   if series_idx_1b in used_series:
       log.warning("Duplicate series index...")
       continue
   ```

4. **Build mapping lists**:
   ```python
   well_name_list = ['A01', 'A02', 'B01', ...]      # Selected wells only
   well_ind_list = [1, 2, 13, ...]                  # Corresponding 1-based series indices
   ```

5. **Convert to 0-based for ND2 indexing**:
   ```python
   nd2_well_idx = well_series - 1  # Convert 1-based to 0-based
   ```

**Key Assumptions**:
- Series numbers are 1-based in Excel sheet
- ND2 file series are 0-based (converted internally)
- One well = one unique series
- Not all plate wells need to be mapped (sparse mapping OK)

---

## 6. FINAL OUTPUT STRUCTURE

**CSV Output**: `{exp_name}_metadata.csv` in `metadata/built_metadata_files/`

**All Columns** (after join):
```
well_id, well, experiment_date,              # Identifiers
Time (s), Objective, Channel,                # Keyence-extracted
Width (px), Height (px), Width (um), Height (um),  # Spatial
time_int, time_string,                       # Time indices
medium, genotype, chem_perturbation,         # Plate design
start_age_hpf, embryos_per_well, temperature  # Experimental conditions
[YX1-only: nd2_series_num, microscope, BF Channel]
```

**One Row Per**: Well × Timepoint

**Example** (3 wells × 2 timepoints = 6 rows):
```
well_id          well  experiment_date  Time (s)  Objective  ...  genotype  temperature
20250101_exp_A01  A01   20250101_exp    0.0      20x        ...  wt         28.0
20250101_exp_A01  A01   20250101_exp   10.5      20x        ...  wt         28.0
20250101_exp_A02  A02   20250101_exp    0.0      20x        ...  mutant     28.0
20250101_exp_A02  A02   20250101_exp   10.5      20x        ...  mutant     28.0
20250101_exp_B01  B01   20250101_exp    0.0      20x        ...  wt         28.0
20250101_exp_B01  B01   20250101_exp   10.5      20x        ...  wt         28.0
```

---

## 7. KEY INVARIANTS & ASSUMPTIONS

### Data Consistency
1. **Well naming must match exactly**: 
   - Plate metadata wells: "A01", "A02", ..., "H12" (uppercase, 3 chars)
   - Scope metadata wells: Must exactly match plate wells
   - Mismatch → left join fails → ValueError raised

2. **Every scope metadata row must have plate metadata**:
   - Left join with validation ensures this
   - If scope has well not in plate → ValueError

3. **Plate metadata can be sparse**:
   - Only wells with `start_age_hpf` non-empty are included
   - OK if only subset of 96 wells are used

4. **Time series alignment**:
   - Keyence: Uses folder structure (T0000, T0001, ...)
   - YX1: Uses ND2 frame timestamps
   - Both stored in `Time (s)` column

### Identifier Relationships

```
experiment_date (from config/date)
    ↓
    + well (from folder structure or Excel sheet)
    ↓
    = well_id (unique key for well across timepoints)
    ↓
    + time_int (from folder T* or ND2 frame order)
    ↓
    = unique row identifier (well_id, time_int)
```

### Critical Dependencies

1. **Excel file must exist**: 
   - Path: `metadata/well_metadata/{exp_name}_well_metadata.xlsx`
   - All required sheets must be present
   - Raises FileNotFoundError if missing

2. **Excel format must be correct**:
   - All sheets must be 8 rows × 12 columns (or padding filled)
   - Non-empty cells are read; empty cells become NaN
   - `start_age_hpf` being empty causes row drop

3. **YX1 series_number_map must be valid**:
   - Series indices must be 1-based
   - Series indices must be ≤ n_w (number of series in ND2)
   - Duplicate mappings are rejected

---

## 8. ERROR HANDLING & VALIDATION

### Pre-Join Validation (Keyence)
- None: `get_image_paths()` returns what it finds

### Pre-Join Validation (YX1)
- Series index range check (lines 410-412)
- Duplicate series detection (lines 414-418)
- Out-of-order series warnings

### Post-Join Validation
```python
if not (meta_df["_merge"] == "both").all():
    missing = meta_df.loc[meta_df["_merge"] != "both", ["well", "experiment_date"]]
    raise ValueError(f"Missing well metadata for:\n{missing}")
```

### File Existence Validation
```python
try:
    with pd.ExcelFile(plate_meta_path) as xlf:
        # Process sheets
except FileNotFoundError:
    raise FileNotFoundError(
        f"Well metadata Excel file not found: {plate_meta_path}\n"
        f"Expected file should contain these sheets: medium, genotype, ..."
    )
except Exception as e:
    raise RuntimeError(
        f"Error reading well metadata Excel file: {plate_meta_path}\n"
        f"File exists but cannot be processed."
    )
```

---

## 9. KNOWN ISSUES & INCONSISTENCIES

### Issue 1: time_int vs time_id naming
- **Location**: build01A_compile_keyence_torch.py
- **Problem**: 
  - Metadata row has `time_int` (line 138)
  - Sample dict has `time_id` (line 154)
  - No clear reason for divergence
- **Impact**: Minor - both refer to same value
- **Recommendation**: Standardize on single name

### Issue 2: Missing validation in Keyence get_image_paths()
- **Problem**: No validation that wells extracted from folders match plate metadata
- **Impact**: Silent failures if folder structure doesn't match expectations
- **Recommendation**: Add well name validation

### Issue 3: YX1 series_number_map validation happens early
- **Problem**: Duplicate/out-of-range series only warned, not made errors
- **Impact**: May proceed with incorrect mappings
- **Recommendation**: Make series validation stricter (error on duplicates)

---

## 10. SUMMARY TABLE

| Aspect | Current Implementation |
|--------|------------------------|
| **Join Type** | LEFT JOIN on (well, experiment_date) |
| **Join Keys** | well + experiment_date (string concat) |
| **Well Identifier Format** | "A01", "A02", ..., "H12" (uppercase, 3 chars) |
| **Experiment ID Source** | Folder name / config parameter |
| **Time Index Source** | Folder structure (Keyence) or ND2 frame order (YX1) |
| **Series Mapping (YX1)** | Excel sheet with 1-based series numbers |
| **Validation Strategy** | Merge indicator + exception on unmatched rows |
| **Output Granularity** | One row per well × timepoint |
| **Missing Data Handling** | Raises ValueError (no nulls permitted) |

