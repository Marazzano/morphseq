# DTW Time-Trend Analysis Package Extraction

## Overview

Successfully extracted DTW clustering and trajectory processing utilities from `results/mcolon/20251029_curvature_temporal_analysis/07b_dtw_clustering_analysis.py` into a new reusable package: **`src/analyze/dtw_time_trend_analysis/`**

## Package Structure

```
src/analyze/dtw_time_trend_analysis/
├── __init__.py                    # Package initialization and API exposure
├── dtw_distance.py                # DTW distance computation (198 lines)
├── trajectory_utils.py            # Trajectory processing and alignment (372 lines)
└── trajectory_statistics.py       # Statistical tests for trajectory patterns (165 lines)

Total: 809 lines of well-documented, reusable code
```

## Extracted Modules

### 1. **dtw_distance.py** - DTW Computation
Core DTW implementation with Sakoe-Chiba band constraint.

**Functions:**
- `compute_dtw_distance(seq1, seq2, window=3, normalize=False)` → float
  - Compute DTW distance between two sequences
  - Supports band-constrained warping and normalization
  - Original source: 07b_dtw_clustering_analysis.py:244-292

- `compute_dtw_distance_matrix(trajectories, window=3, verbose=False)` → np.ndarray
  - Compute pairwise DTW distances for multiple trajectories
  - Returns symmetric (n × n) distance matrix
  - Original source: 07b_dtw_clustering_analysis.py:299-349

**Key Features:**
- Numeric type handling to prevent LAPACK errors
- Sakoe-Chiba band constraint for computational efficiency
- Validation checks (symmetry, diagonal, inf/nan detection)
- Comprehensive docstrings with examples

---

### 2. **trajectory_utils.py** - Trajectory Processing
Functions for extracting, interpolating, and aligning variable-length trajectories.

**Functions:**
- `extract_trajectories(df, genotype_filter=None, metric_name='...', min_timepoints=3, verbose=False)`
  - Extract per-embryo trajectories from long-format data
  - Optional genotype filtering and minimum timepoint threshold
  - Original source: 07b_dtw_clustering_analysis.py:69-114
  - Returns: trajectories, embryo_ids, df_long

- `interpolate_trajectories(df_long, verbose=False)` → pd.DataFrame
  - Handle missing values via linear interpolation within trajectories
  - Keeps interpolation within observed timepoint range
  - Original source: 07b_dtw_clustering_analysis.py:121-163

- `interpolate_to_common_grid(df_long, grid_step=0.5, verbose=False)`
  - Align variable-length trajectories to common timepoint grid
  - Truncates to observed range (no edge padding)
  - Original source: 07b_dtw_clustering_analysis.py:170-237
  - Returns: interpolated_trajectories, embryo_ids_ordered, original_lengths, common_grid

- `extract_early_late_means(df_long, embryo_ids, early_window, late_window, verbose=False)`
  - Extract mean metric values in temporal windows
  - Returns windowed averages for correlation analysis
  - Original source: 07b_dtw_clustering_analysis.py:438-489
  - Returns: early_means_arr, late_means_arr

**Key Features:**
- Handles variable-length sequences robustly
- Linear and cubic spline interpolation options
- Temporal window extraction for early/late analysis
- Full NaN handling and validation

---

### 3. **trajectory_statistics.py** - Statistical Testing
Functions for testing trajectory pattern hypotheses.

**Functions:**
- `test_anticorrelation(cluster_assignments, early_means, late_means, embryo_ids, correlation_threshold=0.3, n_permutations=1000, verbose=False)`
  - Test for anti-correlated patterns between early and late phases
  - Uses Pearson correlation with permutation testing
  - Classifies patterns as "Anti-correlated", "Correlated", or "Uncorrelated"
  - Original source: 07b_dtw_clustering_analysis.py:496-567
  - Returns: dict with per-cluster correlation statistics

**Key Features:**
- Non-parametric permutation testing (1000 permutations)
- Cluster-wise correlation analysis
- Configurable correlation thresholds
- Reproducible randomization (seed=42)

---

## Usage Examples

### Example 1: Basic DTW Distance Computation

```python
import numpy as np
from src.analyze.dtw_time_trend_analysis import compute_dtw_distance

# Compute DTW between two trajectory snippets
trajectory_a = np.array([1.0, 2.0, 3.0, 4.0])
trajectory_b = np.array([1.5, 2.5, 3.5])

distance = compute_dtw_distance(trajectory_a, trajectory_b, window=3)
print(f"DTW Distance: {distance:.3f}")

# Normalized distance (length-independent)
distance_norm = compute_dtw_distance(trajectory_a, trajectory_b, normalize=True)
```

### Example 2: Complete Trajectory Analysis Pipeline

```python
import pandas as pd
from src.analyze.dtw_time_trend_analysis import (
    extract_trajectories,
    interpolate_trajectories,
    interpolate_to_common_grid,
    compute_dtw_distance_matrix,
    extract_early_late_means,
    test_anticorrelation
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load data (assume df has embryo_id, predicted_stage_hpf, normalized_baseline_deviation columns)
df = pd.read_csv('curvature_data.csv')

# STEP 1: Extract per-embryo trajectories
trajectories, embryo_ids, df_long = extract_trajectories(
    df,
    genotype_filter='cep290_homozygous',
    metric_name='normalized_baseline_deviation',
    min_timepoints=3,
    verbose=True
)

# STEP 2: Interpolate missing data
df_long = interpolate_trajectories(df_long, verbose=True)

# STEP 3: Align to common grid
interpolated_trajs, embryo_ids_aligned, orig_lengths, common_grid = interpolate_to_common_grid(
    df_long, grid_step=0.5, verbose=True
)

# STEP 4: Compute DTW distance matrix
distance_matrix = compute_dtw_distance_matrix(
    interpolated_trajs, window=3, verbose=True
)

# STEP 5: Perform hierarchical clustering
clusterer = AgglomerativeClustering(
    n_clusters=3,
    metric='precomputed',
    linkage='average'
)
cluster_assignments = clusterer.fit_predict(distance_matrix)

# STEP 6: Test for early/late anticorrelation
early_means, late_means = extract_early_late_means(
    df_long,
    embryo_ids_aligned,
    early_window=(44, 50),
    late_window=(80, 100),
    verbose=True
)

anticorr_results = test_anticorrelation(
    cluster_assignments,
    early_means,
    late_means,
    embryo_ids_aligned,
    correlation_threshold=0.3,
    verbose=True
)

# Print results
for cluster_id, stats in anticorr_results.items():
    print(f"Cluster {cluster_id}:")
    print(f"  n_embryos: {stats['n_embryos']}")
    print(f"  Interpretation: {stats['interpretation']}")
    print(f"  Pearson r: {stats['pearson_r']:.4f}")
    print(f"  P-value: {stats['p_value']:.4f}")
```

### Example 3: DTW Distance Matrix for Pairwise Analysis

```python
import numpy as np
from src.analyze.dtw_time_trend_analysis import compute_dtw_distance_matrix
import matplotlib.pyplot as plt

# Assuming you have interpolated_trajs from trajectory processing
dist_matrix = compute_dtw_distance_matrix(interpolated_trajs, window=3)

# Visualize distance matrix
plt.figure(figsize=(10, 10))
plt.imshow(dist_matrix, cmap='viridis')
plt.colorbar(label='DTW Distance')
plt.xlabel('Embryo Index')
plt.ylabel('Embryo Index')
plt.title('Pairwise DTW Distances')
plt.tight_layout()
plt.savefig('dtw_distance_matrix.png', dpi=200)

# Check matrix properties
print(f"Symmetric: {np.allclose(dist_matrix, dist_matrix.T)}")
print(f"Zero diagonal: {np.allclose(np.diag(dist_matrix), 0)}")
print(f"Distance range: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")
```

## Migration from Original Script

### Old Code (07b_dtw_clustering_analysis.py)
```python
from pathlib import Path
sys.path.insert(0, str(PROJECT_ROOT))

# DTW distance computation
distance_matrix = compute_dtw_distance_matrix(interpolated_trajs, window=3)
```

### New Code
```python
from src.analyze.dtw_time_trend_analysis import compute_dtw_distance_matrix

# DTW distance computation
distance_matrix = compute_dtw_distance_matrix(interpolated_trajs, window=3, verbose=True)
```

**Benefits:**
- ✓ No script-specific imports or path manipulation
- ✓ Reusable across projects and contexts
- ✓ Better integrated into src/analyze ecosystem
- ✓ Comprehensive docstrings and type hints
- ✓ Optional verbose reporting (instead of required print statements)

## Documentation Quality

All functions include:
- ✓ Comprehensive docstrings (NumPy format)
- ✓ Type hints for parameters and returns
- ✓ Parameter descriptions with defaults
- ✓ Return value descriptions and shapes
- ✓ Extended Notes sections with algorithm details
- ✓ References to original papers (where applicable)
- ✓ Practical code examples
- ✓ Edge case handling documentation

## Testing

Functional tests verify:
- ✓ DTW distance computation accuracy
- ✓ Distance matrix symmetry and zero diagonal
- ✓ Anticorrelation test statistical validity
- ✓ All imports and API exposure

```
✓ DTW distance computed: 2.000
✓ DTW distance matrix shape: (3, 3)
✓ Diagonal check (should be all zeros): True
✓ Symmetry check: True
✓ Anticorrelation test completed for 2 clusters
✓ All functional tests passed!
```

## Next Steps (Optional Enhancements)

1. **Additional Temporal Utilities** - Extract more utilities from scripts 01-07:
   - Time binning utilities
   - Trajectory imputation methods (07a)
   - Penetrance analysis extensions (06)
   - Predictive modeling utilities (04)

2. **Expand Statistical Tests** - Add more trajectory analysis functions:
   - Onset time detection
   - Slope change detection
   - Multi-modal trajectory classification

3. **Integration Examples** - Create analysis notebooks:
   - DTW-based embryo clustering workflow
   - Temporal trend comparison
   - Genotype-specific trajectory patterns

## Dependencies

- **numpy** - Array operations
- **pandas** - Data manipulation
- **scipy** - Interpolation, statistics
- **sklearn** - Clustering (if using provided examples)

All dependencies are standard in the morphseq environment.

## References

- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
- Original analysis: `results/mcolon/20251029_curvature_temporal_analysis/07b_dtw_clustering_analysis.py`
- Package location: `src/analyze/dtw_time_trend_analysis/`

---

**Package created:** 2025-11-03
**Total lines of code:** 809
**Test status:** ✓ All functional tests passed
