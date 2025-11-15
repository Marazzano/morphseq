# Quality Control Module Audit Report

**Date:** 2025-11-15
**Auditor:** Claude Code Agent
**Branch:** claude/audit-snakemake-refactor-drift-01KEBNNyft67Q9ETx1Rjv6Tq

## Executive Summary

**Status**: 🔴 **CRITICAL DRIFT DETECTED** - Multiple breaking changes and stub implementations

### Critical Issues

#### 🔴 CRITICAL

1. **Auxiliary Mask QC: STUB IMPLEMENTATION**
   - Status: NOT FUNCTIONAL - all flags default to False
   - Impact: All imaging quality filtering bypassed (frame, yolk, focus, bubble)
   - Expected: 5-15% frames flagged | Actual: 0% flagged
   - **Action Required**: Complete UNet integration or disable module

2. **Flag Name Mismatches**
   - `no_yolk_flag` → `yolk_flag`
   - `frame_flag` → `edge_flag`
   - `use_embryo_flag` → `use_embryo`
   - **Impact**: Schema/consolidation failures, breaking changes

3. **Segmentation QC Not Integrated**
   - New flags: `edge_flag`, `discontinuous_mask_flag`, `overlapping_mask_flag`
   - Baseline: Not computed or used in gating
   - **Impact**: Valuable QC features unused in working pipeline

4. **Consolidation Logic Incompatibility**
   - Refactored: Overwrites `use_embryo` from scratch
   - Baseline: Preserves existing `use_embryo_flag` state
   - **Impact**: Manual curation may be lost

### Working Modules

✅ **Death Detection** - Already integrated, production-ready (80% detection rate)
✅ **Surface Area Outlier (Primary)** - Thresholds match (k_upper=1.2, k_lower=0.9)
✅ **QC Logic Pattern** - `.any(axis=1)` approach consistent

### Flag List Comparison

| Flag | Baseline | Refactored | Status |
|------|----------|------------|---------|
| dead_flag | ✅ | ✅ | Match |
| dead_flag2 | ✅ | ❌ | Missing |
| sa_outlier_flag | ✅ | ✅ | Match |
| no_yolk_flag | ✅ | - | - |
| yolk_flag | - | ✅ | Name drift |
| frame_flag | ✅ | - | - |
| edge_flag | - | ✅ | Name drift |
| focus_flag | ✅ | ✅ | Match |
| bubble_flag | ✅ | ✅ | Match |
| sam2_qc_flag | ✅ | ❌ | Missing (not computed) |
| discontinuous_mask_flag | ❌ | ✅ | New |
| overlapping_mask_flag | ❌ | ✅ | New |

### High Priority Recommendations

1. **Implement imaging QC or disable flags**
   - Complete UNet integration in `imaging_quality_qc.py`
   - OR remove stub flags from schema until ready

2. **Standardize flag naming**
   - Align schema with baseline column names
   - Create migration guide for name changes

3. **Integrate segmentation QC into baseline**
   - Add new flags to baseline gating logic
   - Validate thresholds on real data

4. **Preserve use_embryo state in consolidation**
   - Add parameter for state preservation
   - Respect manual curation

---

**Status:** CRITICAL ISSUES - Stub implementation will allow bad data into analysis
