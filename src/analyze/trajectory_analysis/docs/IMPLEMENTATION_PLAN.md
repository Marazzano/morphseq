# Trajectory Analysis Module Reorganization - Implementation Plan

## Overview

This plan reorganizes the `trajectory_analysis` module from a flat 27-file structure into functional subpackages (distance, utilities, io, qc, clustering, viz) while maintaining backward compatibility through re-exports in `__init__.py`.

**Scope**: All 5 priorities from REORGANIZATION.md
**Method**: Git mv to preserve history, update imports immediately
**Testing**: Simple import tests for each submodule after each phase

## Key Constraints

1. **Dependency order**: Move files bottom-up (distance → utilities/io → qc → clustering → viz)
2. **Import updates**: Update all imports immediately after each move within the same phase
3. **No breaking changes**: Maintain backward compatibility via `__init__.py` re-exports
4. **Git history**: Use `git mv` to preserve file history

## Implementation Phases

### Phase 0: Setup Directory Structure

Create all new subpackage directories with empty `__init__.py` files:

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/trajectory_analysis
mkdir -p distance qc clustering io utilities viz/plotting docs tests
touch distance/__init__.py qc/__init__.py clustering/__init__.py \
      io/__init__.py utilities/__init__.py viz/__init__.py viz/plotting/__init__.py
```

---

### Phase 1: Config Consolidation (Priority 5)

**Goal**: Merge `plot_config.py` into `config.py` to create single config source.

**Why first**: Many files import from config/plot_config. Consolidating now prevents double updates later.

#### Step 1.1: Expand config.py
Append all constants from `plot_config.py` to `config.py`:
- `GENOTYPE_SUFFIX_COLORS`, `GENOTYPE_SUFFIX_ORDER`
- `PHENOTYPE_COLORS`, `PHENOTYPE_ORDER`
- Matplotlib constants (ALPHA, LINEWIDTH, FONTSIZE, GRID)
- Plotly constants (HEIGHT, WIDTH, HOVER_TEMPLATE)
- Faceted sizing constants

#### Step 1.2: Update imports in 4 files
**Files affected**: `genotype_styling.py`, `facetted_plotting.py`, `plotting_3d.py`, `_archived/faceted_plotting_legacy.py`

Change: `from .plot_config import X` → `from .config import X`

#### Step 1.3: Convert plot_config.py to deprecation wrapper
Replace contents with imports from `config.py` + deprecation warning

#### Test Phase 1
```bash
python3 -c "from analyze.trajectory_analysis.config import GENOTYPE_SUFFIX_COLORS; print('✓')"
python3 -c "from analyze.trajectory_analysis.genotype_styling import get_color_for_genotype; print('✓')"
```

---

### Phase 2: Distance + Utilities + I/O Subpackages (Priority 3)

**Goal**: Move low-level algorithmic code (bottom of dependency tree).

**Why second**: These have minimal dependencies. Moving them first means we won't update them again.

#### Step 2.1: Move distance/ files
```bash
git mv dtw_distance.py distance/
git mv dba.py distance/
```

Create `distance/__init__.py` exporting:
- `compute_dtw_distance`, `compute_dtw_distance_matrix`, `prepare_multivariate_array`, `compute_md_dtw_distance_matrix`, `compute_trajectory_distances`, `dba`

**Update imports**:
- `data_loading.py` line 21: `from .distance.dtw_distance import compute_dtw_distance_matrix`

#### Step 2.2: Move utilities/ files
```bash
git mv trajectory_utils.py utilities/
git mv pca_embedding.py utilities/pca.py
git mv correlation_analysis.py utilities/correlation.py
```

Create `utilities/__init__.py` exporting functions from all 3 files.

**Update imports**:
- `trajectory_utils.py` line 30: `from ..config import DEFAULT_EMBRYO_ID_COL, ...` (add `..`)
- `facetted_plotting.py` line 24: `from .utilities.trajectory_utils import compute_trend_line`

#### Step 2.3: Move io/ files
```bash
git mv data_loading.py io/
git mv phenotype_io.py io/
```

Create `io/__init__.py` exporting:
- `compute_dtw_distance_from_df`, `load_phenotype_file`, `save_phenotype_file`

**Update imports**:
- `io/data_loading.py` line 21: `from ..distance.dtw_distance import compute_dtw_distance_matrix`

#### Test Phase 2
Create `test_phase2.py` testing imports from `distance`, `utilities`, `io` subpackages.

---

### Phase 3: QC Consolidation (Priority 1 - HIGH)

**Goal**: Consolidate `outliers.py` + `distance_filtering.py` → `qc/quality_control.py`

**Why third**: QC depends on distance (already moved). Clustering depends on QC (move next).

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

#### Test Phase 3
Create `test_phase3.py` testing new QC imports + backward compatibility warnings.

---

### Phase 4: Clustering Subpackage (Priority 1 continued)

**Goal**: Move clustering files into `clustering/` subpackage.

**Why fourth**: Clustering depends on distance + qc (both moved). Viz depends on clustering (move last).

#### Step 4.1: Move clustering files
```bash
git mv bootstrap_clustering.py clustering/
git mv consensus_pipeline.py clustering/
git mv cluster_posteriors.py clustering/
git mv cluster_classification.py clustering/
git mv cluster_extraction.py clustering/
git mv k_selection.py clustering/
```

#### Step 4.2: Update imports within clustering/ files
All files need relative imports updated:
- `from .config import X` → `from ..config import X`
- `from .qc.quality_control import X` → `from ..qc.quality_control import X`
- `from .bootstrap_clustering import X` → `from .bootstrap_clustering import X` (same level)

**Critical file**: `consensus_pipeline.py` has 6+ import statements to update.

#### Step 4.3: Create clustering/__init__.py
Export all functions from all 6 clustering modules (bootstrap, posteriors, classification, extraction, consensus_pipeline, k_selection).

#### Test Phase 4
Create `test_phase4.py` testing clustering imports.

---

### Phase 5: Viz Restructure (Priority 2 - HIGH)

**Goal**: Create `viz/plotting/` subpackage with organized plotting modules.

**Why fifth**: Viz imports from everything else. Move last to avoid circular dependencies.

#### Step 5.1: Move dendrogram to viz/
```bash
git mv dendrogram.py viz/
```

**Update imports**:
- `viz/dendrogram.py` line 31: `from ..genotype_styling import get_color_for_genotype` (will update after genotype_styling moves)
- `clustering/consensus_pipeline.py` line ~46: `from ..viz.dendrogram import generate_dendrograms`

#### Step 5.2: Move plotting files to viz/plotting/
```bash
git mv plotting.py viz/plotting/cluster_plots.py
git mv facetted_plotting.py viz/plotting/core.py
git mv plotting_3d.py viz/plotting/plots_3d.py
git mv pair_analysis/plotting.py viz/plotting/pair_plots.py
```

#### Step 5.3: Update imports in moved plotting files
All 4 files need relative imports updated to use `../../` or `../../../`:
- `viz/plotting/cluster_plots.py`: `from ...config import X`
- `viz/plotting/core.py`: `from ...utilities.trajectory_utils import compute_trend_line`, `from ...pair_analysis.data_utils import X`
- `viz/plotting/plots_3d.py`: `from ...config import X`
- `viz/plotting/pair_plots.py`: `from .core import plot_trajectories_faceted`, `from ...genotype_styling import X`

#### Step 5.4: Update pair_analysis/__init__.py
Lines 22-29: Change to `from ..viz.plotting.pair_plots import (plot_pairs_overview, ...)`

#### Step 5.5: Move genotype_styling to viz/
```bash
git mv genotype_styling.py viz/
```

**Update imports**:
- `viz/genotype_styling.py` line 12: `from ..config import GENOTYPE_SUFFIX_COLORS, GENOTYPE_SUFFIX_ORDER`
- `viz/dendrogram.py` line 31: `from .genotype_styling import get_color_for_genotype`
- `viz/plotting/core.py` line 25: `from ..genotype_styling import get_color_for_genotype`
- `viz/plotting/pair_plots.py` line 14: `from ..genotype_styling import build_genotype_style_config, sort_genotypes_by_suffix`

#### Step 5.6: Create viz/__init__.py
Export from dendrogram, genotype_styling, and use **lazy loading** for plotting functions (to avoid circular imports). Pattern from `pair_analysis/__init__.py`.

#### Step 5.7: Create viz/plotting/__init__.py
Export from core, cluster_plots, plots_3d, pair_plots.

#### Test Phase 5
Create `test_phase5.py` testing viz and viz.plotting imports.

---

### Phase 6: Documentation (Priority 4)

#### Step 6.1: Create docs/WORKFLOW.md
Document 3 common workflows:
1. Cluster new data (compute distances → consensus → extract → plot)
2. Outlier detection only
3. Faceted trajectory plots

Include module organization diagram.

#### Step 6.2: Create docs/EXAMPLES.md
Show import patterns:
- New organized imports (recommended)
- Backward compatible imports (still work)
- Full pipeline example with code

---

### Phase 7: Update Main __init__.py

**Goal**: Rewrite `trajectory_analysis/__init__.py` to re-export from new subpackages.

#### Step 7.1: Rewrite __init__.py
Structure:
1. Module docstring explaining reorganization
2. `__version__ = '0.3.0'` (bump version)
3. Re-export from each subpackage:
   - `from .distance import ...`
   - `from .utilities import ...`
   - `from .io import ...`
   - `from .qc import ...`
   - `from .clustering import ...`
   - `from .viz import ...`
   - `from .pair_analysis import ...`
4. `__all__` list with all 60+ exported names

**Critical**: Maintain exact same public API - all functions that were importable from top level must still be importable.

---

### Phase 8: Testing and Verification

#### Step 8.1: Create comprehensive import test
`test_all_imports.py` that tests:
1. All subpackage imports work
2. Top-level imports still work (backward compatibility)
3. No circular import errors
4. Deprecation warnings shown for old wrappers

#### Step 8.2: Create simple per-module tests
`tests/test_imports.py` with one test function per submodule.

#### Step 8.3: Verify file structure
```bash
tree -L 3 trajectory_analysis > tests/final_structure.txt
```

Compare with expected structure in REORGANIZATION.md.

---

## Critical Files to Modify

These are the most complex files requiring careful updates:

1. **`__init__.py`** (362 lines) - Complete rewrite to re-export from new locations
2. **`consensus_pipeline.py`** (23K) - 6+ import statements to update in Phase 3-4
3. **`facetted_plotting.py`** → **`viz/plotting/core.py`** (52K) - Complex imports from 4+ modules
4. **`config.py`** (3.7K) - Expand with plot_config.py contents in Phase 1
5. **`pair_analysis/__init__.py`** - Update to import from viz/plotting/pair_plots.py

## Git Commit Strategy

Create 8 separate commits (one per phase) with clear messages:
- Phase 1: Config consolidation
- Phase 2: Distance, utilities, io subpackages
- Phase 3: QC consolidation
- Phase 4: Clustering subpackage
- Phase 5: Viz restructure
- Phase 6: Documentation
- Phase 7: Update main __init__.py
- Phase 8: Tests

Use `git mv` in all commits to preserve file history.

## Expected Final Structure

```
trajectory_analysis/
├── __init__.py (rewritten with re-exports)
├── config.py (consolidated)
├── plot_config.py (deprecation wrapper)
├── outliers.py (deprecation wrapper)
├── distance_filtering.py (deprecation wrapper)
├── distance/
│   ├── __init__.py
│   ├── dtw_distance.py
│   └── dba.py
├── utilities/
│   ├── __init__.py
│   ├── trajectory_utils.py
│   ├── pca.py (was pca_embedding.py)
│   └── correlation.py (was correlation_analysis.py)
├── io/
│   ├── __init__.py
│   ├── data_loading.py
│   └── phenotype_io.py
├── qc/
│   ├── __init__.py
│   └── quality_control.py (merged outliers + distance_filtering)
├── clustering/
│   ├── __init__.py
│   ├── bootstrap_clustering.py
│   ├── consensus_pipeline.py
│   ├── cluster_posteriors.py
│   ├── cluster_classification.py
│   ├── cluster_extraction.py
│   └── k_selection.py
├── viz/
│   ├── __init__.py (with lazy loading)
│   ├── dendrogram.py
│   ├── genotype_styling.py
│   └── plotting/
│       ├── __init__.py
│       ├── core.py (was facetted_plotting.py)
│       ├── cluster_plots.py (was plotting.py)
│       ├── pair_plots.py (was pair_analysis/plotting.py)
│       └── plots_3d.py (was plotting_3d.py)
├── pair_analysis/
│   ├── __init__.py (updated imports)
│   └── data_utils.py
├── docs/
│   ├── REORGANIZATION.md (existing)
│   ├── WORKFLOW.md (new)
│   └── EXAMPLES.md (new)
└── tests/
    ├── test_all_imports.py
    ├── test_imports.py
    └── final_structure.txt
```

## Success Criteria

1. All imports work from new subpackage locations
2. All imports work from top-level (backward compatibility maintained)
3. Deprecation warnings shown for old modules (plot_config, outliers, distance_filtering)
4. No circular import errors
5. Git history preserved for all moved files
6. File count reduced from 27 scattered files to 6 organized subpackages
7. `test_all_imports.py` passes with all green checkmarks

## Estimated Effort

- Phase 0-1: 30 minutes (setup + config consolidation)
- Phase 2: 45 minutes (distance/utilities/io moves + import updates)
- Phase 3: 45 minutes (QC consolidation + merge logic)
- Phase 4: 45 minutes (clustering moves + many import updates)
- Phase 5: 90 minutes (viz restructure - most complex)
- Phase 6: 30 minutes (documentation)
- Phase 7: 45 minutes (rewrite main __init__.py)
- Phase 8: 30 minutes (comprehensive testing)

**Total: ~5-6 hours** (less than the 8-10 hour estimate in REORGANIZATION.md because we're being aggressive with import updates rather than maintaining dual imports)
