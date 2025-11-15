# Embeddings and Analysis-Ready Module Audit Report

**Date:** 2025-11-15
**Auditor:** Claude Code Agent
**Branch:** claude/audit-snakemake-refactor-drift-01KEBNNyft67Q9ETx1Rjv6Tq

## Executive Summary

**CRITICAL DRIFT DETECTED**: The refactored embeddings and analysis_ready modules are severely incomplete.

### Status Overview

- **Embeddings Module**: ⛔ **EMPTY** - Only contains blank `__init__.py`
- **Analysis-Ready Module**: ⚠️ **PARTIAL** - Assembly logic exists but has drift
- **Snakemake Integration**: ❌ **MISSING** - No workflow rules

**Overall Risk Level: 🔴 HIGH**

### Critical Gaps

#### Embeddings Module (EMPTY)

**Missing Components** (all from working implementation):

1. **Manifest Preparation** - QC filtering (`use_embryo=True`)
2. **VAE Model Loading** - Python version handling, device selection
3. **Python 3.9 Subprocess** - Required for legacy VAE models
4. **File Validation** - Path resolution, coverage checking
5. **Latent CSV Format** - Column specifications
6. **Incremental Processing** - Detect missing embeddings
7. **CLI Interface** - User-facing commands

**Working Implementation:** `src/analyze/gen_embeddings/` (742 lines total)

#### Analysis-Ready Module (PARTIAL)

**Implemented**: ✅ Core assembly function, schema validation, QC flag handling

**Critical Drift**:

1. **Embedding Column Format Mismatch**
   - Working: `z_mu_*` prefix
   - Refactored: `z*` prefix
   - **Impact**: ⛔ INCOMPATIBLE - will not detect embeddings from working pipeline

2. **Missing QC Filtering**
   - Working: Filters to `use_embryo_flag=True` before merge
   - Refactored: No filtering logic
   - **Impact**: ⚠️ Missing critical QC gate

3. **No snip_id Normalization**
   - Working: Handles format variations (_s#### vs _####)
   - Refactored: No normalization
   - **Impact**: ⚠️ Potential join failures

### Snakemake Integration (MISSING)

**Required Rules** (none implemented):
- `rule generate_embeddings` - Run VAE inference with Python 3.9
- `rule assemble_analysis_ready` - Merge features + QC + embeddings
- `rule validate_embedding_coverage` - Check 95% threshold

### Immediate Actions Required

1. **Populate Embeddings Module** (CRITICAL)
   - Copy core logic from `src/analyze/gen_embeddings/`
   - Preserve Python 3.9 subprocess handling
   - Keep file validation logic

2. **Fix Column Format** (CRITICAL)
   - Standardize on `z_mu_*` prefix
   - Update detection logic in analysis_ready

3. **Add Snakemake Rules** (CRITICAL)
   - Create embedding generation rule
   - Create analysis-ready assembly rule
   - Define dependency chain: QC → embeddings → analysis-ready

### Risk Assessment

| Risk | Severity | Likelihood |
|------|----------|------------|
| Embeddings module non-functional | Critical | 100% |
| Incompatible column formats | High | 90% |
| Missing Python 3.9 handling | Critical | 80% |
| No Snakemake integration | Critical | 100% |
| Missing use_embryo filter | High | 70% |

### Recommendation

**Pause further refactoring** until:
- Embeddings module is implemented
- Column format is standardized
- Snakemake rules are created
- Integration testing is complete

The working implementation in `src/analyze/gen_embeddings/` should be treated as canonical reference for migration.

---

**Status:** UNUSABLE - Requires complete implementation before production
