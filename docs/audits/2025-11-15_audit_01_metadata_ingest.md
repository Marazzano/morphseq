# Metadata Ingest Module Audit Report

**Date:** 2025-11-15
**Auditor:** Claude Code Agent
**Branch:** claude/audit-snakemake-refactor-drift-01KEBNNyft67Q9ETx1Rjv6Tq

## Executive Summary

Critical drift detected that will prevent successful integration. The refactored code makes incompatible assumptions about data structures, file paths, and processing workflows.

### Priority Findings

#### 🔴 CRITICAL Issues (Must Fix Before Integration)

1. **Keyence Scope Metadata** - Loses multi-tile z-stack structure
   - Working code groups tiles by well-timepoint for FF projection
   - Refactored code processes individual TIFFs, breaking downstream processing
   - Missing P*/T* directory hierarchy handling

2. **YX1 Well Mapping** - Incorrect series-to-well assignment
   - Refactored generates sequential well indices (00, 01, 02...)
   - Working code reads series_number_map Excel **sheet** (8×12 grid)
   - Will produce wrong well IDs for all YX1 data

3. **Image Manifest** - Wrong directory structure
   - Assumes: `stitched_ff_images/{well}/{channel}/{files}`
   - Actual: `stitched_FF_images/{exp}/{well}_t{time}_stitch.jpg`
   - File scanning will fail completely

4. **YX1 Series Mapper** - Missing Excel sheet parsing
   - Checks for DataFrame column instead of reading Excel sheet
   - Has silent fallback to implicit mapping (dangerous)
   - No stage position validation

#### 🟡 HIGH Priority

- Missing YX1 stage position QC validation (KMeans clustering)
- Missing per-well frame availability diagnostics
- Keyence cytometer detection logic missing

#### 🟢 MEDIUM Priority

- Plate processing error messages less detailed
- Directory search less flexible for edge cases

### Recommended Actions

**DO NOT INTEGRATE** the current refactored code into the Build01 pipeline without addressing:

1. Redesign Keyence scope metadata to preserve tile grouping
2. Fix YX1 well mapping to read Excel sheet correctly
3. Correct image manifest directory structure
4. Add YX1 stage position QC validation
5. Comprehensive integration testing on representative experiments

---

## Detailed Module Breakdown

[The audit from the first agent continues here with all the detailed findings about each module in metadata_ingest]

---

**Status:** REQUIRES MAJOR REWORK before integration
