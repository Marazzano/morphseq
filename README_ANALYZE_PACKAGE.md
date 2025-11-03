# Analyze Package: Difference Detection & Utilities

Comprehensive Python package for morphological phenotype analysis, including classification-based and distribution-based difference detection, time-series visualization, and data utilities.

## Installation

The package is organized under `src/analyze/` and should be added to your Python path:

```python
import sys
sys.path.insert(0, 'src')
from analyze.utils import *
from analyze.difference_detection import *
```

## Quick Start

### 1. Data Loading & Preparation

```python
import sys
sys.path.insert(0, 'src')
from analyze.utils import load_experiments, bin_embryos_by_time, filter_binned_data

# Load experiment data from build06 outputs
df = load_experiments(
    experiment_ids=['20230615', '20230620'],
    build_dir='/path/to/build06/output'
)

# Bin VAE embeddings by developmental time
df_binned = bin_embryos_by_time(
    df,
    time_col='predicted_stage_hpf',
    bin_width=2.0
)

# Filter to embryos with enough time points
df_filtered = filter_binned_data(
    df_binned,
    min_time_bins=3,
    min_embryos=5
)
```

### 2. Classification-Based Phenotype Detection

```python
from analyze.difference_detection import (
    predictive_signal_test,
    compute_embryo_penetrance,
    summarize_penetrance
)

# Test whether genotype can be predicted from morphology
df_time_results, df_embryo_probs = predictive_signal_test(
    df_filtered,
    group_col='genotype',
    time_col='time_bin',
    n_cv_splits=5,
    n_permutations=100,
    use_class_weights=True  # Handle class imbalance
)

# Examine time-resolved results
print(df_time_results)
#   time_bin  AUROC_obs  AUROC_null_mean  pval
#       10.0       0.52            0.495   0.45
#       12.0       0.61            0.502   0.08
#       14.0       0.78            0.506   0.01  ← significant!

# Compute per-embryo penetrance metrics
df_penetrance = compute_embryo_penetrance(
    df_embryo_probs,
    confidence_threshold=0.1
)

# Get summary statistics by genotype
summary = summarize_penetrance(df_penetrance, group_col='true_label')
print(summary)
```

**Interpretation:**
- **AUROC_obs**: Area under ROC curve for predictions at this time bin
- **pval**: Significance from permutation test (reject null if p < 0.05)
- **mean_confidence**: Average prediction certainty across embryos
- **temporal_consistency**: Fraction of time bins correctly classified per embryo
- **penetrance_category**: Categorical label (low/medium/high confidence)

---

### 3. Distribution-Based Difference Detection

```python
from analyze.difference_detection import (
    compute_energy_distance,
    permutation_test_energy,
    compute_mmd,
    permutation_test_mmd,
    hotellings_t2_test
)

# Extract feature matrix for a time bin
df_t12 = df_filtered[df_filtered['time_bin'] == 12.0]
X_wt = df_t12[df_t12['genotype'] == 'wildtype'][[col for col in df_t12.columns if 'z_mu' in col]].values
X_mut = df_t12[df_t12['genotype'] == 'homozygous'][[col for col in df_t12.columns if 'z_mu' in col]].values

# Method 1: Energy Distance
result_energy = permutation_test_energy(X_wt, X_mut, n_permutations=1000)
print(f"Energy distance: {result_energy['energy']:.4f}")
print(f"p-value: {result_energy['pvalue']:.4f}")

# Method 2: Maximum Mean Discrepancy (MMD)
result_mmd = permutation_test_mmd(X_wt, X_mut, n_permutations=1000)
print(f"MMD: {result_mmd['mmd']:.4f}")
print(f"p-value: {result_mmd['pvalue']:.4f}")

# Method 3: Hotelling's T² (parametric, assumes normality)
result_t2 = hotellings_t2_test(X_wt, X_mut)
print(f"T² statistic: {result_t2['t2_statistic']:.4f}")
print(f"p-value: {result_t2['pvalue']:.4f}")
```

**When to use each method:**
- **Energy Distance**: Non-parametric, robust to outliers, good for comparing overall distributional differences
- **MMD**: Kernel-based, flexible, good for detecting differences in any moment of the distribution
- **Hotelling's T²**: Parametric, more powerful if data is normally distributed, focuses on mean differences

---

### 4. Horizon Plots: Time-Series Visualization

```python
from analyze.difference_detection import (
    load_time_matrix_results,
    plot_horizon_grid,
    build_metric_matrices
)

# Load time-matrix comparison results (e.g., from model training)
bundles = load_time_matrix_results(
    root='/path/to/results',
    conditions=['WT', 'Het', 'Homo'],
    filename='full_time_matrix_metrics.csv'
)

# Extract matrices for a specific metric
matrices = {}
for model, bundle in bundles.items():
    matrices[model] = build_metric_matrices(
        bundle.data,
        metric='mae'
    )

# Create horizon plot grid
fig = plot_horizon_grid(
    matrices,
    row_labels=['WT Model', 'Het Model', 'Homo Model'],
    col_labels=['WT Test', 'Het Test', 'Homo Test'],
    metric='mae',
    cmap='RdYlGn_r',
    clip_percentiles=(5, 95),
    loeo_highlight={
        'WT': 'WT_test',
        'Het': 'Het_test',
        'Homo': 'Homo_test'
    },
    title='Model Comparison: MAE across Start/Target Time',
    save_path='horizon_plots.png'
)
```

---

### 5. Complete Workflow Example

```python
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from pathlib import Path

from analyze.utils import (
    load_experiments,
    bin_embryos_by_time,
    filter_binned_data,
    make_safe_comparison_name,
    get_plot_path,
    save_dataframe
)
from analyze.difference_detection import (
    predictive_signal_test,
    compute_embryo_penetrance,
    compute_energy_distance,
    permutation_test_energy,
    plot_horizon_grid
)

# === STEP 1: Load and Prepare Data ===
print("Loading data...")
df = load_experiments(
    experiment_ids=['exp_001', 'exp_002', 'exp_003'],
    build_dir='/net/morphoseq/build06_output'
)

# Bin embeddings by time
df_binned = bin_embryos_by_time(df, bin_width=2.0)
df_filtered = filter_binned_data(df_binned, min_time_bins=3, min_embryos=5)

# === STEP 2: Classification-Based Test ===
print("Running classification test...")
df_time_results, df_embryo_probs = predictive_signal_test(
    df_filtered,
    n_cv_splits=5,
    n_permutations=100
)

# Find onset time (first significant time bin)
significant = df_time_results[df_time_results['pval'] < 0.05]
if not significant.empty:
    onset_time = significant['time_bin'].min()
    print(f"Phenotype onset: {onset_time} hpf")
else:
    print("No significant time bins detected")

# Penetrance analysis
df_penetrance = compute_embryo_penetrance(df_embryo_probs)
high_pen = df_penetrance[df_penetrance['penetrance_category'] == 'high']
print(f"High-penetrance embryos: {len(high_pen)} / {len(df_penetrance)}")

# === STEP 3: Distribution-Based Test at Onset ===
if not significant.empty:
    df_onset = df_filtered[df_filtered['time_bin'] == onset_time]

    # Get embeddings by genotype
    X_wt = df_onset[df_onset['genotype'] == 'wildtype'][
        [c for c in df_onset.columns if 'z_mu_b' in c]
    ].values
    X_mut = df_onset[df_onset['genotype'] == 'homozygous'][
        [c for c in df_onset.columns if 'z_mu_b' in c]
    ].values

    # Energy distance test
    result = permutation_test_energy(X_wt, X_mut, n_permutations=1000)
    print(f"Energy distance p-value: {result['pvalue']:.4f}")

# === STEP 4: Save Results ===
output_dir = Path('/results/morphoseq_analysis')
save_dataframe(df_time_results, output_dir / 'time_results.csv')
save_dataframe(df_penetrance, output_dir / 'penetrance.csv')
print(f"Results saved to {output_dir}")
```

---

## Module Reference

### `analyze.utils`
General-purpose utilities for data handling and file I/O.

| Function | Purpose |
|----------|---------|
| `load_experiment(id, dir)` | Load single experiment from build06 |
| `load_experiments(ids, dir)` | Load and combine multiple experiments |
| `bin_embryos_by_time(df, bin_width)` | Aggregate VAE embeddings by time bins |
| `filter_binned_data(df, min_bins, min_embryos)` | Filter sparse embryos |
| `make_safe_comparison_name(g1, g2)` | Generate safe filenames |
| `get_plot_path(dir, gene, type)` | Generate standardized plot paths |
| `get_data_path(dir, gene, type)` | Generate standardized data paths |
| `save_dataframe(df, path)` | Save with logging |

### `analyze.difference_detection.classification`
Predictive classification and penetrance analysis.

| Function | Purpose |
|----------|---------|
| `predictive_signal_test(df, ...)` | Logistic regression with permutation test |
| `compute_embryo_penetrance(df_probs)` | Per-embryo penetrance metrics |
| `summarize_penetrance(df_pen)` | Summary statistics by group |
| `get_high_penetrance_embryos(df_pen, threshold)` | Filter to high-confidence embryos |

### `analyze.difference_detection.distribution`
Non-parametric distribution tests.

| Function | Purpose |
|----------|---------|
| `compute_energy_distance(X1, X2)` | Energy distance between samples |
| `permutation_test_energy(X1, X2, n_perm)` | Significance test via permutation |
| `compute_mmd(X1, X2, bandwidth)` | Maximum Mean Discrepancy |
| `permutation_test_mmd(X1, X2, n_perm)` | MMD significance test |
| `hotellings_t2_test(X1, X2)` | Multivariate mean comparison |
| `compute_mahalanobis_distance(X, mu, cov)` | Mahalanobis distance |
| `compute_euclidean_distance(X, mu)` | Euclidean distance |

### `analyze.difference_detection.horizon_plots`
Time-series visualization utilities.

| Function | Purpose |
|----------|---------|
| `plot_horizon_grid(matrices, ...)` | N×M grid of heatmaps |
| `plot_single_horizon(matrix, ...)` | Single heatmap |
| `plot_best_condition_map(matrices, ...)` | Winner-per-cell visualization |
| `compute_shared_colorscale(matrices)` | Shared color bounds |

---

## API Design Philosophy

### Flexible Grouping
Most functions accept `group_col` and `time_col` parameters, allowing reuse with different metadata:

```python
# By genotype
predictive_signal_test(df, group_col='genotype', time_col='time_bin')

# By treatment
predictive_signal_test(df, group_col='treatment', time_col='stage_hpf')

# Custom grouping
df['my_group'] = (df['genotype'] + '_' + df['condition'].astype(str))
predictive_signal_test(df, group_col='my_group', time_col='time_bin')
```

### Auto-Detection
Many functions auto-detect columns if not specified:

```python
# Auto-detect latent columns (those containing 'z_mu_b')
predictive_signal_test(df_binned)  # z_cols detected automatically

# Auto-detect time column
bin_embryos_by_time(df)  # uses 'predicted_stage_hpf' if present
```

### Backward Compatibility
Old imports still work (with deprecation warning):

```python
# Old way (deprecated)
from results.mcolon.20251016.utils import load_experiments

# New way (recommended)
from analyze.utils import load_experiments
```

---

## Performance Notes

- **Classification**: ~5-10s for 1000 embryos × 10 time bins × 100 permutations
- **Distribution tests**: ~2-5s per time bin with 1000 permutations
- **Horizon plots**: ~1-2s for 3×3 grid with 5000-point heatmaps
- All computations are parallelizable (future improvement)

---

## Citations & References

- Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: a new approach. *Journal of Statistical Planning and Inference*, 143(8), 1249-1272.
- Gretton, A., et al. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13(1), 723-773.
- Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979). *Multivariate Analysis*. Academic Press.

---

## Contributing

To add new methods:

1. Add functions to appropriate module under `src/analyze/`
2. Update module `__init__.py` with exports
3. Update main `difference_detection/__init__.py`
4. Add docstring examples
5. Update this README

---

## License

Part of the morphseq project. See repository for license details.
