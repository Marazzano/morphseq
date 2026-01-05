# Trajectory Analysis Reorganization - Progress Tracker

**Last Updated**: 2026-01-05 05:58 UTC
**Current Phase**: Phase 2 - Distance/Utilities/IO subpackages (IN PROGRESS)
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

## Current Phase: Phase 2 - Distance/Utilities/IO Subpackages

### What's Been Done (Partial)
- ✓ Moved `dtw_distance.py` → `distance/` (git mv completed)
- ✓ Moved `dba.py` → `distance/` (git mv completed)

### What Needs to Be Done Next

#### Step 2.1: Complete distance/ subpackage
1. **Create `distance/__init__.py`** with exports:
```python
"""Distance Computation Algorithms"""
from .dtw_distance import (
    compute_dtw_distance,
    compute_dtw_distance_matrix,
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
    compute_trajectory_distances,
)
from .dba import dba

__all__ = [
    'compute_dtw_distance', 'compute_dtw_distance_matrix',
    'prepare_multivariate_array', 'compute_md_dtw_distance_matrix',
    'compute_trajectory_distances', 'dba',
]
```

2. **Update import in `data_loading.py`** (line 21):
   - OLD: `from .dtw_distance import compute_dtw_distance_matrix`
   - NEW: `from .distance.dtw_distance import compute_dtw_distance_matrix`

#### Step 2.2: Move utilities/ files
```bash
git mv trajectory_utils.py utilities/
git mv pca_embedding.py utilities/pca.py
git mv correlation_analysis.py utilities/correlation.py
```

Then:
1. **Update imports IN utilities files themselves**:
   - `utilities/trajectory_utils.py` line ~30: Change `from .config` → `from ..config` (add `..`)

2. **Create `utilities/__init__.py`** exporting from all 3 files

3. **Update import in `facetted_plotting.py`** (line 24):
   - OLD: `from .trajectory_utils import compute_trend_line`
   - NEW: `from .utilities.trajectory_utils import compute_trend_line`

#### Step 2.3: Move io/ files
```bash
git mv data_loading.py io/
git mv phenotype_io.py io/
```

Then:
1. **Update import in `io/data_loading.py`** (line 21):
   - OLD: `from .distance.dtw_distance import ...` (already updated in 2.1)
   - NEW: `from ..distance.dtw_distance import ...` (add second `..`)

2. **Create `io/__init__.py`** exporting:
   - `compute_dtw_distance_from_df` (from data_loading)
   - `load_phenotype_file`, `save_phenotype_file` (from phenotype_io)

#### Step 2.4: Test Phase 2
Create `test_phase2.py`:
```python
#!/usr/bin/env python3
print("Testing distance imports...")
from analyze.trajectory_analysis.distance import compute_dtw_distance_matrix, dba
print("✓ Distance imports OK")

print("Testing utilities imports...")
from analyze.trajectory_analysis.utilities import extract_trajectories_df, fit_pca_on_embeddings
print("✓ Utilities imports OK")

print("Testing io imports...")
from analyze.trajectory_analysis.io import compute_dtw_distance_from_df, load_phenotype_file
print("✓ I/O imports OK")
```

Run: `cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/src && python3 test_phase2.py`

#### Step 2.5: Commit Phase 2
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/trajectory_analysis
git add distance/ utilities/ io/ data_loading.py facetted_plotting.py
git commit -m "Phase 2: Create distance, utilities, and io subpackages"
```

## Remaining Phases (Not Started)

### Phase 3: QC Consolidation (Priority 1 - HIGH)
- Consolidate `outliers.py` + `distance_filtering.py` → `qc/quality_control.py`
- Update `consensus_pipeline.py` imports
- Create deprecation wrappers

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

**Files already updated**:
- ✓ genotype_styling.py
- ✓ facetted_plotting.py
- ✓ plotting_3d.py
- ✓ _archived/faceted_plotting_legacy.py

**Files that will need updates in Phase 2**:
- data_loading.py (imports from dtw_distance)
- facetted_plotting.py (imports from trajectory_utils) - needs second update
- utilities/trajectory_utils.py (imports from config)

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

**Next commit**:
2. Phase 2: Create distance, utilities, and io subpackages

## How to Resume

If starting fresh, the next agent should:
1. Read this file (PROGRESS.md)
2. Read docs/IMPLEMENTATION_PLAN.md for full details
3. Continue from "Step 2.1: Complete distance/ subpackage" above
4. Follow the plan step-by-step, testing after each phase
5. Update this PROGRESS.md file as phases complete

## Working Directory

```
/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/trajectory_analysis
```

Current git branch: `feat/traj-reorg`
