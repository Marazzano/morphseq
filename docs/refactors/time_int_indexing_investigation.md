# Time Indexing Investigation - Frame Matching Issue

## Problem Statement
SAM2 pipeline integration is causing "Legacy yolk mask not found" warnings due to potential frame indexing mismatch between:
- SAM2's zero-indexed `frame_index`
- Legacy Build02 masks with one-indexed naming (e.g., `A03_t0060_stitch.jpg.jpg`)

## Code Analysis Findings

### Current Issue Location
- **File**: `src/build/build03A_process_images.py`
- **Line 870**: `exp_df['time_int'] = exp_df['frame_index']` (copies SAM2's zero-indexed values)
- **Line 261**: `stub = f"{well}_t{time_int:04}*"` (searches for legacy masks using zero-indexed value)

### Legacy Mask Naming Evidence
- Legacy Build02 masks are one-indexed: `A03_t0060_stitch.jpg.jpg`, `A03_t0061_stitch.jpg.jpg`
- Located in: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/segmentation/yolk_v1_0050_predictions/20230525/`

### Investigation Plan
1. ✅ Confirmed off-by-one issue exists in current code
2. 🔄 Check raw stitched image naming conventions (YX1 vs Keyence)
3. ⏳ Examine `build01B_compile_yx1_images_torch.py` indexing logic
4. ⏳ Check Keyence image indexing in build01B
5. ⏳ Trace SAM2 time indexing preservation from image_id
6. ⏳ Verify all downstream time sources use image-derived indexing

## Key Question
**Are raw stitched images zero-indexed or one-indexed?** This determines whether SAM2 should preserve original indexing or the legacy masks need adjustment.

### Build01B YX1 Processing Findings
- **File**: `src/build/build01B_compile_yx1_images_torch.py`
- **Line 539**: `time_int_list = np.tile(np.arange(0, n_t, dtype=int), n_selected_wells)`
- **Line 540**: `well_df["time_int"] = time_int_list`
- **🚨 KEY FINDING**: YX1 build01B explicitly creates **ZERO-INDEXED** time_int (0, 1, 2, ..., n_t-1)

### Line 501 Analysis
- `time_id_array[iter_i] = t` where `t` ranges from 0 to n_t-1 in the ND2 loop
- This confirms YX1 processes frames with zero-indexed time values

### Keyence Processing Findings
- **File**: `src/build/build01A_compile_keyence_torch.py`
- **Line 122**: `t_idx = int(time_dir.name[1:])` - extracts time from directory name (e.g., "T0001" → 1)
- **Line 138**: `"time_int": t_idx` - assigns this extracted time to time_int
- **Line 328**: `f"{well}_t{t:04}_stitch.jpg"` - creates stitched images with this time index
- **🚨 CRITICAL FINDING**: Keyence directories start at T0001, so `t_idx` is **ONE-INDEXED** (1, 2, 3, ...)

### Indexing Mismatch Discovered
- **YX1**: Zero-indexed time_int (0, 1, 2, ...)
- **Keyence**: One-indexed time_int (1, 2, 3, ...) from directory names T0001, T0002, etc.
- **Legacy Build02 masks**: One-indexed (t0060, t0061, ...) - likely built from Keyence data

### SAM2 CSV Exporter Issue - ROOT CAUSE FOUND
- **File**: `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`
- **Line 552**: `frame_index + 1` - ALWAYS adds 1 to frame_index for time_int
- **Line 553**: `f"T{frame_index + 1:04d}"` - ALWAYS adds 1 to frame_index for time_string
- **🚨 FUNDAMENTAL PROBLEM**: SAM2 exporter assumes all source images are zero-indexed and forces one-indexing

### The Issue
1. **Keyence images**: Actually one-indexed (A03_t0060) → SAM2 frame_index=59 → time_int=60 ✅ (accidentally correct)
2. **YX1 images**: Actually zero-indexed (A01_t0338) → SAM2 frame_index=338 → time_int=339 ❌ (wrong!)
3. **Legacy Build02 masks**: Built from Keyence data, so one-indexed (t0060)

### Solution Required
SAM2 exporter must extract time values directly from image_id, not reindex frame_index:
- Parse time suffix from image_id (e.g., "A03_t0060" → time_int=60, time_string="T0060")
- Preserve original time indexing from source images
- Stop assuming all sources are zero-indexed

## Next Steps
- ✅ Found root cause: SAM2 exporter force-reindexes regardless of source
- ⏳ Fix SAM2 exporter to parse time from image_id
- ⏳ Verify all downstream time sources use image-derived indexing