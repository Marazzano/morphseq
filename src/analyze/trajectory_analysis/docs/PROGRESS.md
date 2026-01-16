# Trajectory Analysis Reorganization - Progress Tracker

**Last Updated**: 2026-01-16 12:00 UTC
**Current Phase**: Phase 3 - QC Consolidation (NEXT)
**Branch**: feat/traj-reorg

## What We're Doing

Reorganizing the trajectory_analysis module from 27 flat files into functional subpackages (distance, utilities, io, qc, clustering, viz) to improve discoverability and maintainability. See `docs/IMPLEMENTATION_PLAN.md` for full details.

## Completed Phases ✓

### Phase 0: Directory Structure ✓
- Created all subpackage directories: distance/, qc/, clustering/, io/, utilities/, viz/plotting/, docs/, tests/
- Created empty `__init__.py` files for each subpackage
- **Commit**: Initial setup (directories created)

### Phase 1: Config Consolidation ✓
- **Merged** `plot_config.py` constants into `config.py`
- **Updated imports** in 4 files:
  - `genotype_styling.py` line 12: `from .config import ...`
  - `facetted_plotting.py` line 26: `from .config import ...`
  - `plotting_3d.py` line 16: `from .config import ...`
  - `_archived/faceted_plotting_legacy.py` line 25: `from .config import ...`
- **Converted** `plot_config.py` to deprecation wrapper (imports from config + warning)
- **Tested**: Config imports work, genotype_styling imports work
- **Commit**: 44fb6b62 "Phase 1: Consolidate config files"

### Phase 2: Distance/Utilities/IO Subpackages ✓
- **Moved files**:
  - `dtw_distance.py` → `distance/`
  - `dba.py` → `distance/`
  - `trajectory_utils.py` → `utilities/`
  - `pca_embedding.py` → `utilities/pca.py`
  - `correlation_analysis.py` → `utilities/correlation.py`
  - `data_loading.py` → `io/`
  - `phenotype_io.py` → `io/`
- **Created `__init__.py`** files for distance/, utilities/, io/
- **Updated imports**:
  - `utilities/trajectory_utils.py`: `from ..config import ...`
  - `io/data_loading.py`: `from ..distance.dtw_distance import ...`
  - `facetted_plotting.py`: `from .utilities.trajectory_utils import compute_trend_line`
  - `__init__.py`: Updated to import from new subpackages
- **Test Results**: All 4 tests passed (distance, utilities, io, cross-imports)
  ```
  ============================================================
  Phase 2 Import Tests
  ============================================================
  Testing distance imports...
  ✓ Distance imports OK
  Testing utilities imports...
  ✓ Utilities imports OK
  Testing io imports...
  ✓ I/O imports OK
  Testing cross-subpackage imports...
  ✓ Cross-subpackage imports OK

  ============================================================
  Summary:
    distance: ✓ PASS
    utilities: ✓ PASS
    io: ✓ PASS
    cross-imports: ✓ PASS
  ============================================================
  All Phase 2 tests PASSED!
  ```
- **Commit**: fb592d23 "Phase 2: Create distance, utilities, and io subpackages"

### Phase 2 Follow-up Fixes ✓
- **Fixed broken deferred imports**:
  - `distance/dtw_distance.py:272`: Changed `from .trajectory_utils` → `from ..utilities.trajectory_utils`
  - `pair_analysis/data_utils.py:121`: Changed `from ..trajectory_utils` → `from ..utilities.trajectory_utils`
- **Strengthened tests**: Added runtime import tests for `prepare_multivariate_array` and `compute_binned_mean`
- **Created backward-compat shims** with deprecation warnings:
  - `dtw_distance.py` → re-exports from `distance.dtw_distance`
  - `dba.py` → re-exports from `distance.dba`
  - `trajectory_utils.py` → re-exports from `utilities.trajectory_utils`
  - `pca_embedding.py` → re-exports from `utilities.pca`
  - `correlation_analysis.py` → re-exports from `utilities.correlation`
  - `data_loading.py` → re-exports from `io.data_loading`
- **Updated docs**: Added reorganization note to README, fixed example imports in dba.py and trajectory_utils.py
- **Commit**: (pending - Phase 2 follow-up fixes)

## Current Phase: Phase 3 - QC Consolidation

### What Needs to Be Done

#### Step 3.1: Create qc/quality_control.py
Manually merge all functions from:
- `outliers.py`: `identify_outliers()`, `remove_outliers_from_distance_matrix()`
- `distance_filtering.py`: `identify_embryo_outliers_iqr()`, `filter_data_and_ids()`, `identify_cluster_outliers_combined()`

#### Step 3.2: Create qc/__init__.py
Export all 5 functions from `quality_control.py`

#### Step 3.3: Update consensus_pipeline.py imports
Lines 35-40: Change to `from .qc.quality_control import (identify_outliers, identify_embryo_outliers_iqr, ...)`

#### Step 3.4: Convert originals to deprecation wrappers
Replace `outliers.py` and `distance_filtering.py` contents with:
- Import from `qc.quality_control`
- Emit `DeprecationWarning`
- Re-export all functions

#### Step 3.5: Test Phase 3
Create `test_phase3.py` testing new QC imports + backward compatibility warnings.

## Remaining Phases (Not Started)

### Phase 4: Clustering Subpackage
- Move 6 clustering files: bootstrap_clustering.py, consensus_pipeline.py, cluster_posteriors.py, cluster_classification.py, cluster_extraction.py, k_selection.py
- Update all relative imports (many!)

### Phase 5: Viz Restructure (Priority 2 - HIGH)
- Move dendrogram.py → viz/
- Move plotting files → viz/plotting/ with renames
- Move genotype_styling.py → viz/
- Update pair_analysis/__init__.py

### Phase 6: Documentation
- Create docs/WORKFLOW.md
- Create docs/EXAMPLES.md

### Phase 7: Update Main __init__.py
- Rewrite to re-export from all new subpackages
- Bump version to 0.3.0

### Phase 8: Testing
- Create comprehensive test_all_imports.py
- Verify everything works

## Key Files That Need Import Updates

**Files already updated (Phase 1-2)**:
- ✓ genotype_styling.py
- ✓ facetted_plotting.py (from .config + from .utilities.trajectory_utils)
- ✓ plotting_3d.py
- ✓ _archived/faceted_plotting_legacy.py
- ✓ utilities/trajectory_utils.py (from ..config)
- ✓ io/data_loading.py (from ..distance.dtw_distance)
- ✓ __init__.py (partial - distance, utilities, io imports updated)

**Files that will need updates in Phase 3**:
- consensus_pipeline.py (imports from outliers + distance_filtering)

**Files that will need updates in Phase 4**:
- All 6 clustering files (change `.config` → `..config`, etc.)

**Files that will need updates in Phase 5**:
- viz/dendrogram.py
- viz/plotting/*.py (4 files with complex imports)
- pair_analysis/__init__.py

## Git Strategy

Using `git mv` to preserve history. One commit per phase with clear message.

**Commits so far**:
1. 44fb6b62 - Phase 1: Consolidate config files
2. (pending) - Phase 2: Create distance, utilities, and io subpackages

## How to Resume

If starting fresh, the next agent should:
1. Read this file (PROGRESS.md)
2. Read docs/IMPLEMENTATION_PLAN.md for full details
3. Continue from Phase 3: QC Consolidation
4. Follow the plan step-by-step, testing after each phase
5. Update this PROGRESS.md file as phases complete

## Working Directory

```
/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/trajectory_analysis
```

Current git branch: `feat/traj-reorg`
