# Trajectory Analysis Module Reorganization Plan

## Current Issues

The `trajectory_analysis/` folder has grown to 27 Python files with some organizational challenges:

1. **Outlier/Filtering Logic Sprawl**: 3 files (`outliers.py`, `distance_filtering.py`, `consensus_pipeline.py`) handle related filtering functionality
2. **Plotting Sprawl**: 4 files (`plotting.py`, `facetted_plotting.py`, `plotting_3d.py`, `pair_analysis/plotting.py`) with overlapping concerns
3. **Unclear Boundaries**: High-level vs low-level functions mixed together
4. **Large `__init__.py`**: 362 lines, 60+ exported functions - impossible to know what's important

The code quality is **good** (well-documented, robust functions), but the **organization** needs improvement.

---

## Recommended Reorganization

This proposal follows a **functionality-first** organization (similar to how analysis packages like Seurat/Monocle present “what you can do”), with a small set of explicit dependency rules to keep the codebase maintainable as it grows.

### Guiding Principles (Option 1: Functional Subpackages)

**Functional grouping wins for analysis packages**: users should be able to find code by intent (distance computation vs clustering vs QC vs visualization) without having to understand architectural layers.

**Recent plotting lessons (from multiclass AUROC workflows)**:
1. Keep **plotting functions pure** (accept fully-prepared DataFrames) and move data munging into explicit preprocessing helpers.
2. Use **consistent significance annotations** (null bands + p-value circles) across all AUROC-like plots to avoid visual drift.
3. Centralize plotting style and markers so downstream scripts don't re-implement legends or thresholds.

**Hard boundaries (dependency rules):**

1. `distance/` is algorithmic: it **must not import** from `clustering/`, `qc/`, or `viz/`.
2. `clustering/` may import from `distance/` and `qc/`, but not from `viz/`.
3. `qc/` may import from `distance/` (e.g., distance-based outlier metrics), but should not import from `clustering/`.
4. `viz/` may import from anywhere; nothing should import from `viz/`.
5. Keep pandas-heavy “dataframe glue” in `io/` and `utilities/` so algorithms stay testable and reusable.

### Priority 1: Quality Control Consolidation (HIGH IMPACT)

**Problem**: Filtering logic scattered across 3 files makes it hard to understand the filtering pipeline.

**Solution**: Create a `qc/` subpackage and consolidate filtering logic into `qc/quality_control.py`.

```
qc/quality_control.py (NEW - consolidates 3 files)
├── Level 1: Generic outlier detection
│   ├── identify_outliers_generic()
│   │   ├── method='median_distance'
│   │   ├── method='percentile'
│   │   ├── method='iqr'
│   │   └── method='mad'
│   └── [from outliers.py]
│
├── Level 2: Stage-specific filtering
│   ├── identify_stage1_outliers_iqr()  [from distance_filtering.py]
│   ├── identify_stage2_outliers_combined()  [from distance_filtering.py]
│   └── filter_data_and_ids()
│
└── Level 3: Backward compatibility wrappers
    └── identify_embryo_outliers_iqr()  [legacy API]

clustering/consensus_pipeline.py
├── run_consensus_pipeline()  [orchestrates qc/quality_control.py]
└── REMOVE internal filtering logic (delegate to qc/quality_control.py)
```

**Implementation Steps**:
1. Create `quality_control.py` with all functions from `outliers.py` and `distance_filtering.py`
2. Update `consensus_pipeline.py` to import from `quality_control.py`
3. Deprecate `outliers.py` and `distance_filtering.py` (add deprecation warnings)
4. Update `__init__.py` imports
5. Run tests to ensure backward compatibility

**Files Affected**:
- Create: `qc/quality_control.py`, `qc/__init__.py`
- Modify: `clustering/consensus_pipeline.py`, `__init__.py`
- Deprecate: `outliers.py`, `distance_filtering.py` (keep as thin wrappers)

---

### Priority 2: Plotting Restructure (HIGH IMPACT)

**Problem**: 4 plotting modules with unclear separation of concerns.

**Solution**: Create `viz/plotting/` subpackage with a clear split between generic plotting primitives and analysis-specific plots, plus a small preprocessing helper layer for plot-ready inputs (mirroring the AUROC workflow used in multiclass analyses).

```
viz/plotting/ (NEW subpackage)
├── __init__.py (public API - exports high-level functions)
│
├── core.py (Level 1 - Generic plotting primitives)
│   ├── TraceData, SubplotData, FigureData (IR pattern)
│   ├── Generic faceted plotting logic
│   └── [from facetted_plotting.py]
│
├── prep.py (NEW - plot-data preparation helpers)
│   ├── prepare_*_data() helpers (e.g., add bin centers, sig flags)
│   └── NO plotting logic
│
├── cluster_plots.py (Level 2 - Clustering-specific)
│   ├── plot_cluster_trajectories_df()
│   ├── plot_membership_trajectories_df()
│   ├── plot_membership_vs_k()
│   ├── plot_posterior_heatmap()
│   ├── plot_2d_scatter()
│   └── [from plotting.py]
│
├── pair_plots.py (Level 2 - Pair/group-specific)
│   ├── plot_pairs_overview()
│   ├── plot_genotypes_by_pair()
│   └── [from pair_analysis/plotting.py]
│
├── 3d_plots.py (Level 2 - 3D specialized)
│   ├── plot_3d_scatter()
│   └── [from plotting_3d.py]
│
└── config.py (stays as-is - styling constants)

pair_analysis/
├── __init__.py
└── data_utils.py (keep - data extraction helpers)

dendrogram.py (stays top-level - specialized module)
```

**Implementation Steps**:
1. Create `plotting/` directory
2. Create `plotting/core.py` with IR classes from `facetted_plotting.py`
3. Refactor `plotting.py` → `plotting/cluster_plots.py`
4. Refactor `plotting_3d.py` → `plotting/3d_plots.py`
5. Move `pair_analysis/plotting.py` → `plotting/pair_plots.py`
6. Update all imports in dependent code
7. Update `__init__.py` to expose clean API

**Files Affected**:
- Create: `viz/plotting/__init__.py`, `viz/plotting/core.py`, `viz/plotting/cluster_plots.py`, `viz/plotting/pair_plots.py`, `viz/plotting/3d_plots.py`
- Move: `plot_config.py` → `viz/plotting/config.py`
- Deprecate: `plotting.py`, `facetted_plotting.py`, `plotting_3d.py`, `pair_analysis/plotting.py` (keep as wrappers)
- Modify: All files that import plotting functions

---

### Priority 3: Utilities + I/O Subpackages (MEDIUM IMPACT)

**Problem**: Utility + I/O modules scattered at top level make the directory cluttered and blur boundaries between algorithms and dataframe glue.

**Solution**: Create two small subpackages:

- `io/` for data loading / saving / phenotype I/O
- `utilities/` for general helpers (e.g., PCA embedding, correlation helpers)

```
io/ (NEW)
├── __init__.py
├── data_loading.py (move)
└── phenotype_io.py (move)

utilities/ (NEW)
├── __init__.py
├── pca.py (rename from pca_embedding.py)
└── correlation.py (rename from correlation_analysis.py)
```

**Implementation Steps**:
1. Create `utilities/` directory with `__init__.py`
2. Move and rename utility modules
3. Update imports in dependent code
4. Update `__init__.py`

**Files Affected**:
- Create: `io/__init__.py`, `utilities/__init__.py`
- Move: `data_loading.py` → `io/data_loading.py`
- Move: `phenotype_io.py` → `io/phenotype_io.py`
- Move/Rename: `pca_embedding.py` → `utilities/pca.py`
- Move/Rename: `correlation_analysis.py` → `utilities/correlation.py`

---

### Priority 4: Documentation (MEDIUM IMPACT)

**Problem**: No clear workflow documentation for typical analyses.

**Solution**: Add `WORKFLOW.md` showing common analysis patterns.

```markdown
# Trajectory Analysis Workflows

## Workflow 1: Cluster New Data

1. Load data: `load_experiment_dataframe()`
2. Interpolate: `interpolate_to_common_grid_df()`
3. Compute distances: `compute_trajectory_distances()`
4. Run consensus: `run_consensus_pipeline()`
5. Extract results: `extract_cluster_embryos()`

## Workflow 2: Project onto Existing Clusters

1. Load reference clusters
2. Prepare arrays: `prepare_multivariate_array(time_grid=...)`
3. Compute cross-DTW: `compute_cross_dtw_distance_matrix()`
4. Assign clusters: `assign_clusters_nearest_neighbor()`
5. Compare frequencies
```

**Files to Create**:
- `WORKFLOW.md` - Step-by-step analysis workflows
- `EXAMPLES.md` - Code examples for common tasks

---

### Priority 5: Config Consolidation (LOW IMPACT)

**Problem**: Config constants split across `config.py` and `plot_config.py`.

**Solution**: Merge into single `config.py`.

```python
# config.py (EXPANDED)
# Bootstrap parameters
DEFAULT_N_BOOTSTRAP = 100

# DTW parameters
DEFAULT_SAKOE_CHIBA_RADIUS = 3

# Plotting colors & sizes (from plot_config.py)
DEFAULT_PLOTLY_HEIGHT = 600
INDIVIDUAL_TRACE_ALPHA = 0.3

# File paths
DEFAULT_DATA_DIR = Path("...")
```

**Implementation Steps**:
1. Move constants from `plot_config.py` to `config.py`
2. Create `plot_config.py` as import-only wrapper for backward compatibility
3. Update imports

**Files Affected**:
- Modify: `config.py` (expand)
- Deprecate: `plot_config.py` (become wrapper)

---

## Proposed Final Structure (Option 1: Functional)

```
trajectory_analysis/
├── __init__.py (clean public API)
├── config.py
├── docs/
│   ├── WORKFLOW.md (NEW)
│   ├── EXAMPLES.md (NEW)
│   └── REORGANIZATION.md (this file)
│
├── distance/ (ALGORITHMS: reusable, no clustering imports)
│   ├── __init__.py
│   ├── dtw_distance.py
│   └── dba.py
│
├── qc/ (FILTERING / OUTLIERS)
│   ├── __init__.py
│   └── quality_control.py (NEW - consolidates outliers.py + distance_filtering.py)
│
├── clustering/ (PIPELINES + CLUSTER-SPECIFIC LOGIC)
│   ├── __init__.py
│   ├── bootstrap_clustering.py
│   ├── consensus_pipeline.py
│   ├── cluster_posteriors.py
│   ├── cluster_classification.py
│   ├── cluster_extraction.py
│   ├── k_selection.py
│   └── cluster_projection.py (NEW - from this analysis)
│
├── io/ (DATA LOADING / SAVING)
│   ├── __init__.py
│   ├── data_loading.py
│   └── phenotype_io.py
│
├── utilities/ (GENERAL HELPERS)
│   ├── __init__.py
│   ├── trajectory_utils.py
│   ├── correlation.py
│   └── pca.py
│
├── viz/
│   ├── __init__.py
│   ├── dendrogram.py
│   └── plotting/
│       ├── __init__.py
│       ├── core.py
│       ├── cluster_plots.py
│       ├── pair_plots.py
│       ├── 3d_plots.py
│       └── config.py
│
├── genotype_styling.py
├── pair_analysis/
│   ├── __init__.py
│   └── data_utils.py
│
└── _archived/
    └── faceted_plotting_legacy.py
```

---

## Implementation Timeline

### Phase 1 (High Impact, ~2-3 hours)
1. Create `quality_control.py` (consolidate filtering)
2. Update `consensus_pipeline.py`
3. Run tests for backward compatibility

### Phase 2 (High Impact, ~4-5 hours)
1. Create `plotting/` subpackage
2. Refactor 4 plotting modules
3. Update all imports
4. Run tests

### Phase 3 (Medium Impact, ~1 hour)
1. Create `utilities/` subpackage
2. Move utility modules
3. Update imports

### Phase 4 (Documentation, ~1 hour)
1. Write `WORKFLOW.md`
2. Write `EXAMPLES.md`
3. Update `README.md`

### Phase 5 (Polish, ~30 min)
1. Consolidate config
2. Clean up `__init__.py`
3. Final testing

**Total Estimated Time**: 8-10 hours

---

## Testing Strategy

For each phase:
1. **Unit tests**: Ensure individual functions still work
2. **Integration tests**: Test full pipelines (load → cluster → plot)
3. **Backward compatibility**: Old imports should still work (with deprecation warnings)
4. **Documentation**: Update docstrings and examples

---

## Backward Compatibility

All reorganization should maintain backward compatibility:

```python
# Old code (still works with deprecation warning)
from src.analyze.trajectory_analysis import identify_embryo_outliers_iqr

# New code (preferred)
from src.analyze.trajectory_analysis.qc.quality_control import identify_outliers_generic
```

Deprecation timeline:
- **Phase 1**: Add deprecation warnings to old imports
- **Phase 2** (3 months later): Remove deprecated modules

---

## Benefits of Reorganization

1. **Discoverability**: “What do you want to do?” maps to a folder (`distance/`, `clustering/`, `qc/`, `viz/`).
2. **Maintainability**: Hard boundaries prevent algorithm code (DTW/DBA) from tangling with clustering pipelines.
3. **Testability**: `distance/` stays small and dependency-light.
4. **Extensibility**: New functionality has an obvious home (e.g., new distances in `distance/`, new selection metrics in `clustering/`).
5. **Cleaner public API**: `__init__.py` can re-export a curated set of user-facing functions.

---

## Questions Before Starting

1. **Backward compatibility window**: How long to maintain deprecated imports?
2. **Testing coverage**: Do we have sufficient tests to validate refactoring?
3. **User communication**: How to notify users of reorganization?
4. **Migration guide**: Should we provide a migration script?

---

## Notes

- This reorganization does NOT change algorithm logic - only file organization
- All public APIs remain accessible (possibly with different import paths)
- Focus on **gradual migration** with deprecation warnings, not breaking changes
- The 27 files aren't the problem - the **grouping and presentation** is

---

**Last Updated**: 2026-01-04
**Status**: Proposal (not yet implemented)
