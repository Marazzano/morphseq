# Audit Log: generate_image_manifest.py

**Module:** `src/data_pipeline/metadata_ingest/manifests/generate_image_manifest.py`
**Audit Date:** 2025-11-10
**Status:** ✅ COMPLETE

---

## Audit Notes / Scratchpad

**NEW MODULE** - No direct working equivalent in legacy pipeline.

Working implementation approach:
- Image inventory built implicitly during processing
- No explicit manifest file generated
- Each processing step discovers files independently
- Frame ordering handled per-step (not centralized)

Refactor approach:
- **Single source of truth** for image inventory
- Explicit JSON manifest with hierarchical structure
- Frame ordering validated and enforced
- Channel normalization validated
- BF channel presence required

This is a **DESIGN INNOVATION**, not a drift from existing code.

---

## Role / Intended Function

**Summary:** Build experiment-level image manifest with validated frame ordering and channel normalization

**Reference docs:**
- `processing_files_pipeline_structure_and_plan.md` - Lines 133-135 (may relocate note)
- `snakemake_rules_data_flow.md` - Lines 99-118 (rule generate_image_manifest)

**Intended behavior:**
1. Read `scope_and_plate_metadata.csv`
2. Scan `built_image_data/{exp}/stitched_ff_images/` directory
3. Build hierarchical inventory: experiment → wells → channels → frames
4. Validate channel normalization (all channels in VALID_CHANNEL_NAMES)
5. Require BF channel presence (critical for SAM2)
6. Sort frames by time_int (required for SAM2 temporal ordering)
7. Validate against image_manifest schema
8. Output: `experiment_metadata/{exp}/experiment_image_manifest.json`

**Critical design principle:**
> "The experiment image manifest is the single source of truth for per-well, per-channel frame ordering; all segmentation rules consume experiment_image_manifest.json"
> — processing_files_pipeline_structure_and_plan.md, line 57

---

## Working Implementation

**NO DIRECT EQUIVALENT**

The working implementation does NOT create a centralized image manifest. Instead:

1. **Keyence:** `build01A_compile_keyence_torch.py:get_image_paths()`
   - Discovers images during FF building
   - Returns list of dicts with tile paths
   - No persistent manifest

2. **YX1:** `build01B_compile_yx1_images_torch.py`
   - Reads ND2 file structure directly
   - No separate image inventory

3. **Segmentation:** `build03A_process_images.py`
   - Discovers images independently via glob
   - No guaranteed frame ordering
   - Each well processed separately

**Problems with working approach:**
- ❌ No single source of truth for image inventory
- ❌ Frame ordering not validated
- ❌ Channel names not normalized
- ❌ Each step must re-discover images (inefficient)
- ❌ No validation that BF channel exists before segmentation

---

## Drift Summary

**Type(s) of drift:**
- [x] New feature (entirely new module)
- [x] Docs vs code mismatch (design doc says this is critical, but no working equivalent)

**Details:**

### 1. NEW DESIGN FEATURE (Not Drift)
- **Working:** No centralized image manifest
- **Refactor:** Explicit JSON manifest as single source of truth
- **Impact:**
  - Ensures consistent frame ordering for SAM2
  - Validates channel normalization upfront
  - Prevents downstream failures (missing BF channel)
  - Improves efficiency (discover once, use many times)
- **Rationale:** Design docs explicitly call this out as critical (line 57)

### 2. Frame Ordering Validation (NEW)
- **Working:** Frame ordering implicit, not validated
- **Refactor:** Explicit time_int sorting and validation
- **Impact:**
  - SAM2 requires temporal ordering for propagation
  - Catches missing/duplicate frames early
  - Prevents propagation failures

### 3. Channel Normalization Validation (NEW)
- **Working:** No validation of channel names
- **Refactor:** Validates all channels against VALID_CHANNEL_NAMES
- **Impact:**
  - Catches unmapped channel names early
  - Prevents downstream channel selection failures

### 4. BF Channel Requirement (NEW)
- **Working:** No upfront BF validation
- **Refactor:** Fails fast if BF channel missing
- **Impact:**
  - SAM2 segmentation requires BF channel
  - Prevents expensive processing on invalid data

### 5. Hierarchical Structure (NEW)
```json
{
  "experiment_id": "20240509_24hpf",
  "wells": {
    "A01": {
      "channels": {
        "BF": {
          "frames": [
            {"frame_index": 0, "time_int": 0, "file_path": "..."},
            {"frame_index": 1, "time_int": 1, "file_path": "..."}
          ]
        }
      }
    }
  }
}
```
- **Impact:** Efficient lookup, clear hierarchy, easy validation

### 6. Single Source of Truth (NEW)
- **Working:** Each processing step discovers images independently
- **Refactor:** Manifest generated once, consumed by all downstream steps
- **Impact:**
  - Consistency across pipeline
  - Efficiency (no repeated file I/O)
  - Auditability (manifest shows exactly what was processed)

---

## Recommended Actions

### CRITICAL: Validate manifest generation
- [ ] Test on Keyence experiments
  - Verify file discovery in `stitched_ff_images/`
  - Check well/channel/frame hierarchy
  - Validate time_int sorting
- [ ] Test on YX1 experiments
  - Same validations as Keyence

### HIGH PRIORITY: Validate channel normalization
- [ ] Verify all channels map to VALID_CHANNEL_NAMES
- [ ] Test error handling for unmapped channels
- [ ] Confirm BF channel presence check works

### HIGH PRIORITY: Validate frame ordering
- [ ] Verify frames sorted by time_int
- [ ] Test with non-sequential frame indices
- [ ] Check handling of missing frames (gaps in time_int)

### MEDIUM PRIORITY: Test schema validation
- [ ] Verify JSON output matches image_manifest schema
- [ ] Test REQUIRED_EXPERIMENT_FIELDS validation
- [ ] Test REQUIRED_WELL_FIELDS validation
- [ ] Test REQUIRED_CHANNEL_FIELDS validation
- [ ] Test REQUIRED_FRAME_FIELDS validation

### INTEGRATION: Validate downstream consumption
- [ ] Verify segmentation modules can read manifest
- [ ] Test frame ordering is preserved through pipeline
- [ ] Confirm per-well processing uses manifest correctly

---

## Priority

**HIGH** - This is a foundational change to pipeline architecture. Not a drift from working code, but a **design innovation**.

---

## Status

**Reviewed:** 2025-11-10
**Changes implemented:** No
**Blocker:** None (ready for testing, but needs integration validation)

---

## Conclusion

`generate_image_manifest.py` is **NOT A DRIFT** but a **DESIGN INNOVATION**:
- ✅ No working equivalent to compare against
- ✅ Design docs explicitly call this out as critical (line 57)
- ✅ Solves real problems in working implementation:
  - Inconsistent frame ordering
  - No channel normalization validation
  - Repeated file discovery overhead
  - No BF channel validation before segmentation
- ✅ Aligns with "single source of truth" design principle

**Recommendation:** Implement and test thoroughly. This is a **critical architectural improvement**, not a regression.

**Key validation points:**
1. Frame ordering must be correct for SAM2 temporal propagation
2. Channel normalization must catch unmapped channels
3. BF channel must be present for all wells going to segmentation
4. Manifest format must match schema exactly
5. Downstream modules must consume manifest correctly

**No drift detected** - this is new functionality that improves upon working implementation's implicit approach.
