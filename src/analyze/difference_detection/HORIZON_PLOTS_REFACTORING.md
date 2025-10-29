# Horizon Plots Refactoring: From Results to src/analyze

**Date**: October 29, 2025
**Status**: Structure created, skeleton implementation ready for population

## Overview

Extracted horizon plot functionality into reusable library modules under `src/analyze/difference_detection/`. This enables both the model comparison pipeline (20251020) and the new curvature analysis (20251029) to use the same plotting utilities without code duplication.

## New Structure

### Library Modules (src/analyze/)

```
src/analyze/difference_detection/
├── horizon_plots.py           [NEW] Reusable horizon plot visualization
├── time_matrix.py             [NEW] Time matrix data reshaping
├── __init__.py                [UPDATED] Export new modules
├── classification/
├── distribution/
├── plotting.py
├── metrics.py
└── pipelines.py
```

### Thin Wrapper Scripts (results/)

```
results/mcolon/20251020/
└── compare_3models_full_time_matrix_wrapper.py  [NEW] CLI orchestrator

results/mcolon/20251029_curvature_temporal_analysis/
├── 02_horizon_plots.py        [NEW] Curvature-specific usage
└── (other analysis scripts...)
```

## What Was Created

### 1. **src/analyze/difference_detection/horizon_plots.py**
Empty skeleton with docstrings and function signatures:
- `plot_horizon_grid()` - Main function for N×M grid of heatmaps
- `plot_single_horizon()` - Single heatmap with customization
- `plot_best_condition_map()` - Show which condition performs best per cell
- `compute_shared_colorscale()` - Compute color bounds with percentile clipping
- Internal helpers for formatting, highlights, etc.

### 2. **src/analyze/difference_detection/time_matrix.py**
Empty skeleton with docstrings and function signatures:
- `load_time_matrix_results()` - Load CSVs with condition grouping
- `build_metric_matrices()` - Reshape long-form data to 2D matrices
- `align_matrix_times()` - Ensure matrices share time indices
- `compute_matrix_statistics()` - Summary stats across matrices
- `filter_matrices_by_time_range()` - Temporal windowing
- `interpolate_missing_times()` - Fill sparse time grids

### 3. **src/analyze/difference_detection/__init__.py**
Updated to export new modules and convenience functions

### 4. **results/mcolon/20251020/compare_3models_full_time_matrix_wrapper.py**
Thin wrapper that:
- Imports horizon_plots/time_matrix from analyze package
- Contains fallback plotting logic (original code) for now
- Will use new utilities once they're implemented
- Maintains backward compatibility

### 5. **results/mcolon/20251029_curvature_temporal_analysis/02_horizon_plots.py**
Curvature-specific script that demonstrates usage:
- Loads curvature data from metadata
- Computes timepoint × timepoint correlation matrices
- Uses `plot_horizon_grid()` once implemented
- Shows intended API usage pattern

## Next Steps to Complete Implementation

### Phase 1: Implement Core Utilities
1. **horizon_plots.py**
   - Port `plot_3model_comparison()` → `plot_horizon_grid()`
   - Port `create_best_model_heatmap()` → `plot_best_condition_map()`
   - Extract helper functions for color scaling, formatting, LOEO highlights

2. **time_matrix.py**
   - Port matrix creation logic from compare_3models script
   - Add alignment logic for shared time indices
   - Generalize column/index naming

### Phase 2: Integration
1. Update `compare_3models_full_time_matrix_wrapper.py`
   - Remove fallback plotting functions
   - Call `plot_horizon_grid()` from analyze module

2. Run `02_horizon_plots.py` for curvature analysis
   - Should work once time_matrix/horizon_plots are populated

### Phase 3: Migration & Cleanup
1. Keep original `compare_3models_full_time_matrix.py` for provenance
2. Update any notebooks importing from results/mcolon/20251016/utils
   - Change to: `from analyze.utils import ...`
   - Change to: `from analyze.difference_detection import ...`

## API Contract

The reusable utilities should be parameterized to work with any data that has:
- A 2D metric value (start_time × target_time)
- Conditions/groups to compare
- Optional metadata (genotype labels, LOEO indicators, etc.)

### Example: Model Comparisons
```python
from analyze.difference_detection import plot_horizon_grid

matrices = {'WT': {'WT_test': df1, 'Het_test': df2, ...},
            'Het': {...},
            'Homo': {...}}

plot_horizon_grid(
    matrices,
    row_labels=['WT Model', 'Het Model', 'Homo Model'],
    col_labels=['WT Test', 'Het Test', 'Homo Test'],
    metric='mae',
    loeo_highlight={
        'WT': 'WT_test',
        'Het': 'Het_test',
        'Homo': 'Homo_test'
    }
)
```

### Example: Curvature Analysis
```python
from analyze.difference_detection import plot_horizon_grid

# Correlation matrices per genotype
matrices = {'WT': correlation_matrix_df,
            'Het': correlation_matrix_df,
            'Homo': correlation_matrix_df}

plot_horizon_grid(
    matrices,
    row_labels=['Wildtype', 'Heterozygous', 'Homozygous'],
    col_labels=['Arc Length Ratio'],
    cmap='RdBu_r'
)
```

## Files Modified/Created

| File | Status | Notes |
|------|--------|-------|
| src/analyze/difference_detection/horizon_plots.py | NEW | Skeleton with docstrings |
| src/analyze/difference_detection/time_matrix.py | NEW | Skeleton with docstrings |
| src/analyze/difference_detection/__init__.py | UPDATED | Added exports |
| results/mcolon/20251020/compare_3models_full_time_matrix_wrapper.py | NEW | Thin wrapper + fallback |
| results/mcolon/20251029_curvature_temporal_analysis/02_horizon_plots.py | NEW | Curvature usage example |
| results/mcolon/20251020/compare_3models_full_time_matrix.py | UNCHANGED | Kept for provenance |

## Notes

- All new files are **ready to use** but contain placeholder implementations
- Docstrings are comprehensive and include usage examples
- Existing fallback logic keeps scripts runnable immediately
- The skeleton structure forces clear API boundaries before implementation
- Tests should be written before filling in implementations
