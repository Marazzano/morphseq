# difference_detection: Statistical Testing Framework

A flexible, reusable permutation testing framework for detecting phenotypic differences in developmental data.

## Overview

This module provides statistical tools for comparing genotypes and phenotypes using two complementary approaches:

1. **Distribution Tests** (Pool Shuffle): "Are these populations different?"
2. **Classification Tests** (Label Shuffle): "Can I predict individual outcomes?"

Built on a unified permutation testing infrastructure, enabling consistent methodology across different test statistics while remaining simple and extensible.

## Quick Startx

```python
# Distribution test: Are populations different?
from analyze.difference_detection import permutation_test_energy
result = permutation_test_energy(X_groupA, X_groupB, n_permutations=1000)
print(result)  # PermutationResult(energy=0.2134, p=0.0099)

# Classification test: Can we predict labels?
from analyze.difference_detection import predictive_signal_test
df_binned = bin_embryos_by_time(df, bin_width=2.0)
results, probs = predictive_signal_test(
    df_binned,
    group1="wildtype",
    group2="mutant"
)
```

## Core Concepts: Pool vs Label Shuffle

### The Critical Distinction

| Test | Question | Failure Means... | Success Means... |
|------|----------|------------------|------------------|
| **Pool Shuffle** | Are they different? | No biological effect | Populations have different means (Scientific Truth) |
| **Label Shuffle** | Can I sort them? | Noise too high; groups overlap | Distributions distinct; can predict individuals (Predictive Utility) |

### Pool Shuffle (Distribution Tests)

Combines both groups, shuffles, then splits back.

```
[WT₁, WT₂, Mut₁, Mut₂] → shuffle → [WT₁, Mut₁, WT₂, Mut₂]
                                      ↓
                               Split and compare
```

**Tests:** "Are these samples from the same distribution?"
**Example:** `permutation_test_energy()`, `permutation_test_mmd()`

### Label Shuffle (Classification Tests)

Features stay fixed, labels get shuffled.

```
Features: [X₁, X₂, X₃, X₄]  (unchanged)
Labels:   [WT, WT, Mut, Mut] → [Mut, WT, WT, Mut] (shuffled)
                                 ↓
                           Can classifier recover?
```

**Tests:** "Are labels predictable from features?"
**Example:** `predictive_signal_test()`

### Why Both Matter: A Subtle But Important Distinction

Pool shuffle can succeed while label shuffle fails. This happens when distributions are shifted but overlap too much for individual prediction.

**Example: Height by Sex**
- Pool Shuffle: ✓ Men are taller than women on average (p < 0.001)
- Label Shuffle: ✗ Given height 5'7" (170cm), can't confidently predict sex
- **Why?** Real biological effect exists, but variance within each group drowns out the signal for individuals.

**In morphseq terms:**
- Pool Shuffle Success = "Mutants develop differently than WT" (the biology is real)
- Label Shuffle Failure = "But variance is too high to predict individual embryo fates" (can't use clinically or for individual screening)

## Standard Workflow

```
Raw Data
   ↓
Bin by Time (e.g., 2-hour windows)
   ↓
Choose Test Type
   ├─ Distribution Test → "Are populations different?"
   └─ Classification Test → "Can we predict individuals?"
   ↓
Interpret Results
```

### Distribution Test Path

```python
from analyze.difference_detection import permutation_test_energy

# Extract features by group
X_wt = df_binned[df_binned['genotype']=='wildtype'][z_cols].values
X_mut = df_binned[df_binned['genotype']=='mutant'][z_cols].values

# Test if distributions differ
result = permutation_test_energy(X_wt, X_mut, n_permutations=1000)
if result.pvalue < 0.05:
    print(f"Populations differ: p={result.pvalue:.4f}")
```

### Classification Test Path

```python
from analyze.difference_detection import predictive_signal_test

# Test if labels are predictable from features
df_results, df_embryo_probs = predictive_signal_test(
    df_binned,
    group1="wildtype",
    group2="mutant",
    n_splits=5,
    n_perm=100
)

# Time-resolved results
print(df_results[['time_bin', 'AUROC_obs', 'pval']])
```

## Data Contracts

### Input: df_binned

Standard binned DataFrame with time-resolved embryo data.

**Required columns:**
- `embryo_id`: str, unique identifier
- `time_bin`: int/float, time point (e.g., 0, 2, 4, 6 hpf)
- `genotype`: str, class label
- `z_mu_b*_binned`: float, latent features (VAE embeddings)

**Constraints:**
- Minimum 5 samples per class per time bin
- At least 2 genotypes for binary classification
- Time bins should have consistent spacing

### Output: PermutationResult

Standardized result container for all permutation tests.

```python
result.observed          # Test statistic on real data
result.pvalue            # Permutation p-value
result.null_distribution # Full null distribution (array)
result.null_mean         # Mean of null distribution
result.null_std          # Std of null distribution

# Methods
result.is_significant(alpha=0.05)  # True/False
result.to_dict()                   # Convert to dictionary
str(result)  # "energy: 0.2134 (p=0.0099, null: 0.0821±0.0234)"
```

## Common Use Cases

### Comparing Phenotypes

Test differences between any two groups (not just WT vs mutant).

```python
# WT vs Mutant
result_wt_mut = permutation_test_energy(X_wt, X_mut)

# Mutant1 vs Mutant2
result_mut1_mut2 = permutation_test_energy(X_mut1, X_mut2)
```

### Penetrance Analysis

Measures individual consistency relative to group prediction. One analysis option among many.

```python
from analyze.difference_detection import compute_embryo_penetrance

df_penetrance = compute_embryo_penetrance(
    df_embryo_probs,
    confidence_threshold=0.1
)

# Can compare any phenotypes (not just vs WT)
summary = df_penetrance.groupby('true_label').agg({
    'mean_confidence': 'mean',
    'mean_signed_margin': 'mean'
})
```

### Time-Resolved Detection

Test at each time point to detect onset of significant differences.

```python
# df_results from predictive_signal_test contains one row per time_bin
significant_bins = df_results[df_results['pval'] < 0.05]
onset_time = significant_bins['time_bin'].min()
print(f"Significant signal onset at {onset_time} hpf")
```

## Module Architecture

```
difference_detection/
├── permutation_utils.py       # Core utilities
│   ├── compute_pvalue()       # P-value calculation
│   ├── pool_shuffle()         # Distribution test shuffling
│   ├── label_shuffle()        # Classification test shuffling
│   └── PermutationResult      # Standardized result class
│
├── distance_metrics.py        # Pluggable distance metrics
│   ├── compute_energy_distance()
│   ├── compute_mmd()
│   └── compute_mean_distance()
│
├── distribution_test.py       # Distribution-based tests
│   ├── permutation_test_distribution()
│   ├── permutation_test_energy()
│   └── permutation_test_mmd()
│
├── classification/            # Classification-based tests
│   ├── predictive_test.py     # AUROC with label shuffle
│   ├── penetrance.py          # Embryo-level metrics
│
└── horizon_plots/             # Visualization utilities
```

### Import Paths

```python
# New API (recommended)
from analyze.difference_detection import (
    permutation_test_energy,
    permutation_test_mmd,
    predictive_signal_test,
    compute_pvalue,
    PermutationResult
)

# Old API (still works, backwards compatible)
from analyze.difference_detection.classification import (
    predictive_signal_test,
    compute_embryo_penetrance
)
```

## Extensibility: Adding Custom Test Statistics

The framework is designed for easy extension. To add a new test:

**Step 1:** Add function to `distance_metrics.py`

```python
def compute_my_metric(X1, X2):
    """Compute your test statistic."""
    return some_scalar_value
```

**Step 2:** Use with permutation framework

```python
from analyze.difference_detection import permutation_test_distribution

result = permutation_test_distribution(
    X1, X2,
    statistic="my_metric",
    n_permutations=1000
)
```

That's it. Framework handles shuffling, p-value calculation, and standardized output.

## Key Functions Reference

- `permutation_test_energy(X1, X2, ...)` — Energy distance permutation test
- `permutation_test_mmd(X1, X2, ...)` — Maximum Mean Discrepancy test
- `permutation_test_distribution(X1, X2, statistic=..., ...)` — Generic distribution test
- `predictive_signal_test(df_binned, ...)` — AUROC classification test with time resolution
- `compute_embryo_penetrance(df_probs, ...)` — Individual-level consistency metrics
- `compute_pvalue(observed, null_dist, ...)` — Core p-value utility for custom tests

See docstrings in source files for detailed parameters and return values.

## Notes

- **Data Contracts:** Schemas are evolving as the pipeline matures. Check docstrings for current specifications.
- **P-values:** Uses pseudo-count method (k+1)/(n+1) by default for conservative estimates.
- **Sample Size:** Recommend ≥1000 permutations for robust estimates. Minimum 5 samples per class per time bin.
- **Computation:** Most expensive step is usually classifier training or distance computation. Permutation test loop is parallelizable for large datasets.
