# Implementation Plan: Feature-First Plotting API

## Summary

Implement the feature-first naming convention from PLOTTING_REFACTOR_PLAN.md by:
1. Renaming functions to `plot_feature_over_time` / `plot_feature_over_time_faceted`
2. Renaming parameter `metric` → `feature`
3. Aligning single-panel API with faceted API (use `trend_method` instead of `use_dba`)
4. Adding deprecated wrappers for old names
5. Cleaning up temp files

---

## Current State Analysis

### Faceted version (ALREADY MOSTLY COMPLIANT)
`viz/plotting/faceted/time_series.py`:
```python
def plot_time_series_faceted(
    metric='metric_value',      # → rename to feature
    trend_method='mean',        # ✅ already correct!
    ...
)
```

### Single-panel version (NEEDS MORE WORK)
`viz/plotting/time_series.py`:
```python
def plot_time_series_by_group(
    metric='metric_value',      # → rename to feature
    show_mean=True,             # → rename to show_trend
    use_dba=True,               # → replace with trend_method
    ...
)
```

---

## Implementation Steps

### Step 1: Update single-panel `time_series.py`

**File:** `src/analyze/viz/plotting/time_series.py`

1. **Rename function**: `plot_time_series_by_group` → `plot_feature_over_time`

2. **Update parameters**:
   - `metric` → `feature` (with default `'metric_value'`)
   - `show_mean` → `show_trend` (default `True`)
   - Remove `use_dba: bool`
   - Add `trend_method: Optional[str] = 'mean'` (values: `'mean'`, `'dba'`, `'median'`, `None`)

3. **Update internal logic**:
   - Change `show_mean` references to `show_trend`
   - Change `use_dba` logic to `trend_method == 'dba'`
   - Support `trend_method='median'` (aligns with faceted)

4. **Add deprecated wrapper**:
```python
def plot_time_series_by_group(
    df, metric='metric_value', ..., show_mean=True, use_dba=True, ...
) -> plt.Figure:
    """Deprecated: Use plot_feature_over_time instead."""
    warnings.warn(
        "plot_time_series_by_group is deprecated. "
        "Use plot_feature_over_time instead.",
        DeprecationWarning, stacklevel=2
    )
    # Convert old params to new
    trend_method = 'dba' if use_dba else 'mean' if show_mean else None
    return plot_feature_over_time(
        df, feature=metric, ..., trend_method=trend_method, ...
    )
```

5. **Update `plot_embryos_metric_over_time`** wrapper to point to new function

### Step 2: Update faceted `time_series.py`

**File:** `src/analyze/viz/plotting/faceted/time_series.py`

1. **Rename function**: `plot_time_series_faceted` → `plot_feature_over_time_faceted`

2. **Update parameters**:
   - `metric` → `feature` (keep default `'metric_value'`)
   - Already has `trend_method` ✅

3. **Update helper function**: `_plot_single_group_on_axis`
   - Rename `metric` parameter to `feature`

4. **Add deprecated wrapper**:
```python
def plot_time_series_faceted(...) -> plt.Figure:
    """Deprecated: Use plot_feature_over_time_faceted instead."""
    warnings.warn(...)
    return plot_feature_over_time_faceted(...)
```

5. **Update `plot_embryos_metric_over_time_faceted`** wrapper to point to new function

### Step 3: Update `__init__.py` exports

**File:** `src/analyze/viz/plotting/__init__.py`

Update to export:
```python
# New canonical API
from .time_series import (
    plot_feature_over_time,
    plot_time_series_by_group,  # deprecated
    plot_embryos_metric_over_time,  # deprecated
    get_membership_category_colors,
)
from .faceted.time_series import (
    plot_feature_over_time_faceted,
    plot_time_series_faceted,  # deprecated
    plot_embryos_metric_over_time_faceted,  # deprecated
)

__all__ = [
    # Canonical API
    'plot_feature_over_time',
    'plot_feature_over_time_faceted',
    'get_membership_category_colors',
    # Deprecated (backward compat)
    'plot_time_series_by_group',
    'plot_time_series_faceted',
    'plot_embryos_metric_over_time',
    'plot_embryos_metric_over_time_faceted',
]
```

### Step 4: Fix faceted.py shim exports

**File:** `src/analyze/trajectory_analysis/viz/plotting/faceted.py`

Add missing color helper names to `__all__`:
```python
# Add these to __all__:
'create_color_lookup',
'create_color_lookup_from_column',
'create_color_state',
'get_color_from_state',
```

### Step 5: Cleanup temp files

Delete:
- `src/analyze/trajectory_analysis/viz/plotting/faceted.py.bak`
- `CONTINUE_PHASE3_PLAN.md`
- `src/analyze/PLOTTING_REFACTOR_PLAN.md` (plan is now implemented)
- `test_stratification_implementation.py` (test script)

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/analyze/viz/plotting/time_series.py` | Rename function, update params, add wrapper |
| `src/analyze/viz/plotting/faceted/time_series.py` | Rename function, update metric→feature, add wrapper |
| `src/analyze/viz/plotting/__init__.py` | Update exports |
| `src/analyze/trajectory_analysis/viz/plotting/faceted.py` | Fix `__all__` exports |

## Files to Delete

| File | Reason |
|------|--------|
| `faceted.py.bak` | Backup no longer needed |
| `CONTINUE_PHASE3_PLAN.md` | Phase complete |
| `PLOTTING_REFACTOR_PLAN.md` | Plan implemented |
| `test_stratification_implementation.py` | Test script |

---

## Parameter Mapping Reference

### Single-panel (New → Old)
| New Name | Old Name | Notes |
|----------|----------|-------|
| `feature` | `metric` | Rename only |
| `show_trend` | `show_mean` | Rename only |
| `trend_method` | `use_dba` | `'dba'` if True, `'mean'` if False + show_mean |

### Faceted (New → Old)
| New Name | Old Name | Notes |
|----------|----------|-------|
| `feature` | `metric` | Rename only |
| `trend_method` | `trend_method` | Already aligned ✅ |

---

## Verification

After implementation, run:

```bash
# 1. Test new API imports
python -c "from src.analyze.viz.plotting import plot_feature_over_time, plot_feature_over_time_faceted; print('✅ New API imports OK')"

# 2. Test deprecated imports still work
python -c "
import warnings
warnings.filterwarnings('always')
from src.analyze.viz.plotting import plot_time_series_by_group
" 2>&1 | grep -q "DeprecationWarning" && echo "✅ Deprecation warning emitted"

# 3. Test backward compat wrappers
python -c "
import warnings
warnings.filterwarnings('ignore')
from src.analyze.viz.plotting import plot_embryos_metric_over_time
print('✅ Backward compat wrapper OK')
"

# 4. Run existing tests
python -m unittest discover -s src/analyze/difference_detection/tests -v

# 5. Check no circular imports
python -c "
import src.analyze.utils.timeseries
import src.analyze.viz
import src.analyze.trajectory_analysis
print('✅ No circular imports')
"
```

---

## Commit Message (Draft)

```
Implement feature-first plotting API naming convention

Rename plotting functions to use bioinformatics-standard vocabulary:
- plot_time_series_by_group → plot_feature_over_time
- plot_time_series_faceted → plot_feature_over_time_faceted
- Parameter: metric → feature
- Parameter: show_mean/use_dba → trend_method (string: 'mean', 'dba', 'median')

All old names preserved as deprecated wrappers with helpful warnings.
This completes the PLOTTING_REFACTOR_PLAN.md specification.

## Rationale
- "feature" is standard in Seurat/Scanpy bioinformatics tooling
- trend_method as string is more flexible than separate bool params
- Consistent naming enables better autocomplete and documentation
```
