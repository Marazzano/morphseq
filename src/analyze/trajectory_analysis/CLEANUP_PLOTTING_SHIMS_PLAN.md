# Plan: Remove Confusing Plotting Shims

## Problem Summary

There are **two separate plotting systems** causing confusion:

1. **Generic Plotting** (`src/analyze/viz/plotting/`) - Domain-agnostic, uses `plot_feature_over_time` ✅ Already refactored
2. **Trajectory Plotting** (`src/analyze/trajectory_analysis/viz/plotting/`) - Trajectory-specific, uses `plot_trajectories_faceted`

The confusion stems from an **unnecessary shim** file that's already deprecated.

---

## Files to Remove

### 1. **Delete the Shim**
```
src/analyze/trajectory_analysis/viz/plotting/faceted.py
```

**Reason:** This is a deprecated module-level shim (already has deprecation warning) that just re-exports from the `faceted/` submodule. It's causing import confusion and serves no purpose now that the refactor is complete.

### 2. **Delete the Backup**
```
src/analyze/trajectory_analysis/viz/plotting/faceted.py.bak
```

**Reason:** No longer needed.

---

## Files Importing from the Old Shim

Based on grep search, **only 1 file** imports from the deprecated shim:

### `src/analyze/trajectory_analysis/pair_analysis/plotting.py`

**Current import:**
```python
# Uses the shim indirectly via __init__.py
from ..facetted_plotting import plot_trajectories_faceted
```

**Action Required:** ✅ **No change needed**
- This import is fine - it imports from `facetted_plotting.py` (the real module, note double-t)
- The shim we're deleting is `faceted.py` (single-t, the directory shim)

---

## Files NOT Affected (Already Use Correct Paths)

All other files correctly import from either:
- `src.analyze.trajectory_analysis.facetted_plotting` (the real module with double-t)
- `src.analyze.viz.plotting` (generic plotting)
- Direct submodule imports from `faceted/` subdirectory

---

## Implementation Steps

### Step 1: Verify No Direct Imports of the Shim
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
grep -r "from.*viz.plotting.faceted import" src/ --include="*.py"
# Expected: No results (imports go through __init__.py)
```

### Step 2: Delete Files
```bash
# Delete the shim
rm src/analyze/trajectory_analysis/viz/plotting/faceted.py

# Delete the backup
rm src/analyze/trajectory_analysis/viz/plotting/faceted.py.bak
```

### Step 3: Update `__init__.py` Exports (if needed)

Check `src/analyze/trajectory_analysis/viz/plotting/__init__.py`:
- If it exports from `faceted`, update to import from `faceted/` subdirectory
- Most likely this is already correct

### Step 4: Create Clear Documentation

**Add:** `src/analyze/viz/plotting/README.md`

```markdown
# Plotting Systems Guide

## Two Independent Systems

### 1. Generic Plotting (`src/analyze/viz/plotting/`)
**Use for:** Domain-agnostic time series visualization

**Functions:**
- `plot_feature_over_time()` - Single-panel overlay plot
- `plot_feature_over_time()` - Multi-panel faceted plot via `row_by` / `col_by`

**Key Features:**
- Generic: Works with any time series data
- Uses `feature=` parameter (bioinformatics standard)
- No domain-specific logic (no genotypes, pairs, etc.)

**Example:**
```python
from src.analyze.viz.plotting import plot_feature_over_time

fig = plot_feature_over_time(
    df,
    feature='metric_value',
    time_col='hpf',
    id_col='entity_id',
    color_by='group'
)
```

### 2. Trajectory Analysis Plotting (`src/analyze/trajectory_analysis/viz/plotting/`)
**Use for:** Trajectory-specific visualizations with genotype styling

**Functions:**
- `plot_trajectories_faceted()` - Faceted trajectory plots
- `plot_multimetric_trajectories()` - Multi-metric comparison
- `plot_pairs_overview()` - Pair-specific analysis
- `plot_genotypes_by_pair()` - Genotype comparison

**Key Features:**
- Trajectory-specific: Genotype coloring, pair analysis
- Uses `y_col=` parameter (metric/feature column name)
- Domain logic: Automatic genotype colors, pair grouping

**Example:**
```python
from src.analyze.trajectory_analysis import plot_trajectories_faceted

fig = plot_trajectories_faceted(
    df,
    x_col='predicted_stage_hpf',
    y_col='baseline_deviation_normalized',
    line_by='embryo_id',
    col_by='genotype',
    color_by_grouping='genotype'
)
```

## When to Use Which?

| Need | Use | System |
|------|-----|--------|
| Generic time series plot | `plot_feature_over_time()` | viz/plotting |
| Genotype-aware trajectories | `plot_trajectories_faceted()` | trajectory_analysis |
| Multi-experiment faceting | `plot_feature_over_time()` (row_by/col_by) | viz/plotting |
| Pair analysis | `plot_pairs_overview()` | trajectory_analysis |
```

---

## Verification After Changes

```bash
# 1. Check no broken imports
python -c "from src.analyze.trajectory_analysis import plot_trajectories_faceted; print('✅ OK')"

# 2. Check generic plotting still works
python -c "from src.analyze.viz.plotting import plot_feature_over_time; print('✅ OK')"

# 3. Run tests
python -m pytest src/analyze/trajectory_analysis/tests/ -v
```

---

## Summary of Changes

| Action | File | Reason |
|--------|------|--------|
| DELETE | `viz/plotting/faceted.py` | Deprecated shim, causes confusion |
| DELETE | `viz/plotting/faceted.py.bak` | Backup no longer needed |
| CREATE | `viz/plotting/README.md` | Clear documentation of two systems |
| UPDATE | (none) | No imports need updating! |

---

## Why This Fixes the Confusion

1. **Removes ambiguity** - No more deprecated shim file
2. **Clear separation** - Two distinct systems with clear purposes
3. **Better documentation** - README explains when to use each
4. **Cleaner structure** - Only real modules remain

The shim was a transitional artifact from a previous refactor. Removing it completes the cleanup.
