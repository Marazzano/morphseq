# Trajectory Analysis Package

**Location:** `src/analyze/trajectory_analysis/`
**Status:** In Development (Migration from `dtw_time_trend_analysis` + consensus clustering)
**Version:** 0.2.0

---

## Overview

The `trajectory_analysis` package provides a comprehensive framework for analyzing temporal trajectories using Dynamic Time Warping (DTW), bootstrap-based consensus clustering, and probabilistic quality assessment.

### Scope

This package unifies:
1. **DTW-based distance computation** (from `dtw_time_trend_analysis`)
2. **Trajectory processing utilities** (from `dtw_time_trend_analysis`)
3. **Statistical correlation analysis** (from `dtw_time_trend_analysis`)
4. **Bootstrap consensus clustering** (from `results/mcolon/20251106_robust_clustering/`)
5. **Posterior probability analysis** (from `results/mcolon/20251106_robust_clustering/`)
6. **Membership quality classification** (from `results/mcolon/20251106_robust_clustering/`)
7. **Trajectory visualization** (integrated plotting)

---

## Package Structure

```
src/analyze/trajectory_analysis/
├── README.md                    # This file
├── __init__.py                  # Public API exports
├── config.py                    # Centralized configuration defaults
│
├── dtw_distance.py              # DTW distance computation
├── dba.py                       # DTW Barycenter Averaging
├── trajectory_utils.py          # Data extraction, interpolation, padding
├── correlation_analysis.py      # Temporal pattern correlation tests
│
├── bootstrap_clustering.py      # Bootstrap resampling for clustering
├── cluster_posteriors.py        # Posterior probability computation
├── cluster_classification.py    # Core/uncertain/outlier classification
│
└── plotting.py                  # Visualization functions
```

---

## Module Responsibilities

### Core DTW & Trajectories

#### `dtw_distance.py`
- Compute DTW distance between two sequences
- Compute full DTW distance matrix
- Sakoe-Chiba band constraint support

#### `dba.py` (DTW Barycenter Averaging)
- Compute consensus trajectory using DBA algorithm
- Iterative refinement with DTW alignment
- Optional Gaussian smoothing

#### `trajectory_utils.py`
- Extract trajectories from long-format DataFrame
- Interpolate to common time grid
- Pad trajectories for plotting
- Extract early/late window means

#### `correlation_analysis.py`
- Test for anticorrelation between early and late patterns
- Permutation-based statistical testing
- Correlation classification

---

### Bootstrap Clustering & Quality Assessment

#### `bootstrap_clustering.py`
- Bootstrap hierarchical clustering with resampling
- Bootstrap k-medoids clustering
- Label alignment across bootstrap iterations
- Silhouette score computation

#### `cluster_posteriors.py`
- **Label alignment via Hungarian algorithm**: Align arbitrary bootstrap cluster IDs to reference labels by maximizing overlap
- Compute posterior probabilities p_i(c) from bootstrap with frequency normalization
- Calculate entropy (Shannon information)
- Calculate log-odds gap between top clusters
- Compute maximum assignment probability
- Modal cluster assignment

**Key Implementation Detail - Hungarian Alignment:**
Bootstrap iterations produce arbitrary cluster labels (cluster "0" in iteration 1 may correspond to cluster "2" in iteration 2). Before computing posteriors, we align each bootstrap iteration's labels to the reference labels using the Hungarian algorithm (linear_sum_assignment from scipy). This ensures cluster identities are consistent across all iterations, enabling accurate posterior probability computation.

#### `cluster_classification.py`
- 2D gating classification (max_p × log_odds_gap)
- Adaptive per-cluster thresholds
- Core/uncertain/outlier categorization
- Classification summary statistics

---

### Visualization

#### `plotting.py`
- Posterior probability heatmaps
- 2D scatter plots (max_p vs log_odds_gap)
- Trajectory plots colored by cluster
- Trajectory plots colored by membership category
- Membership proportion plots across k values

---

## Configuration

### `config.py`

Centralized defaults for all modules:

```python
# Bootstrap parameters
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42

# DTW parameters
DTW_WINDOW = 5
GRID_STEP = 0.5

# Data processing
MIN_TIMEPOINTS = 3
DEFAULT_EMBRYO_ID_COL = 'embryo_id'
DEFAULT_METRIC_COL = 'normalized_baseline_deviation'
DEFAULT_TIME_COL = 'predicted_stage_hpf'
DEFAULT_GENOTYPE_COL = 'genotype'

# Classification thresholds
THRESHOLD_MAX_P = 0.8
THRESHOLD_LOG_ODDS_GAP = 0.7
THRESHOLD_OUTLIER_MAX_P = 0.5
```

---

## Function Workflow & Chaining

### Complete Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LOADING & PREPROCESSING                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  trajectory_utils.extract_trajectories()
                    - Load long-format DataFrame
                    - Filter by genotype
                    - Extract per-embryo trajectories
                              │
                              ▼
              trajectory_utils.interpolate_to_common_grid()
                    - Interpolate to uniform time grid
                    - Handle missing values
                    - Returns: trajectories, embryo_ids, common_grid
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DTW DISTANCE MATRIX                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  dtw_distance.compute_dtw_distance_matrix()
                    - Pairwise DTW distances
                    - Sakoe-Chiba band constraint
                    - Returns: D (n × n distance matrix)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BOOTSTRAP CLUSTERING                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            bootstrap_clustering.run_bootstrap_hierarchical()
                    - Resample 80% of embryos (n_bootstrap times)
                    - Hierarchical clustering on each subsample
                    - Align labels across iterations
                    - Returns: bootstrap_results_dict
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   POSTERIOR PROBABILITY ANALYSIS                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            cluster_posteriors.analyze_bootstrap_results()

            Step 1: Label Alignment (Hungarian Algorithm)
                    - For each bootstrap iteration:
                      * Build contingency table: C[i,j] = count of embryos
                        with bootstrap_label=i and reference_label=j
                      * Use scipy.optimize.linear_sum_assignment to find
                        optimal mapping that maximizes overlap
                      * Remap bootstrap cluster IDs to reference IDs
                    - Ensures cluster "0" always means the same cluster

            Step 2: Posterior Computation
                    - For each embryo i and cluster c:
                      p_i(c) = count(i assigned to c) / count(i sampled)
                    - Normalize by per-embryo sample frequency
                    - Result: p_matrix (n_embryos × n_clusters)

            Step 3: Quality Metrics
                    - max_p: max(p_i) - confidence in top cluster
                    - entropy: H(p_i) - overall uncertainty
                    - log_odds_gap: log(p_top1/p_top2) - disambiguation
                    - Returns: posterior_analysis
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MEMBERSHIP CLASSIFICATION                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            cluster_classification.classify_membership_2d()
                    - 2D gating: max_p × log_odds_gap
                    - Threshold-based categorization
                    - Returns: core/uncertain/outlier labels
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         VISUALIZATION                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   plotting.plot_posterior_heatmap()
                   plotting.plot_2d_scatter()
                   plotting.plot_cluster_trajectories()
```

---

## Key Algorithmic Details

### Hungarian Algorithm for Label Alignment

**Problem:** Bootstrap iterations produce arbitrary cluster labels. Cluster "0" in iteration 1 may correspond to cluster "2" in iteration 2, making direct comparison impossible.

**Solution:** Use the Hungarian algorithm (optimal assignment problem) to align each bootstrap iteration's labels to a reference labeling.

**Algorithm:**
```python
# For each bootstrap iteration:

# 1. Build contingency table
#    C[i, j] = number of embryos with bootstrap_label=i and reference_label=j
C = np.zeros((n_bootstrap_clusters, n_reference_clusters))
for embryo_idx in sampled_indices:
    boot_label = labels_bootstrap[embryo_idx]
    ref_label = labels_reference[embryo_idx]
    C[boot_label, ref_label] += 1

# 2. Find optimal mapping using Hungarian algorithm
#    Maximize overlap = minimize negative overlap
cost_matrix = -C  # Negate because algorithm minimizes cost
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 3. Create mapping dictionary
mapping = {boot_id: ref_id for boot_id, ref_id in zip(row_ind, col_ind)}

# 4. Remap all bootstrap labels
labels_aligned = np.array([mapping.get(label, -1) for label in labels_bootstrap])
```

**Why It Matters:**
- Without alignment, embryo X assigned to cluster 0 in iteration 1 and cluster 2 in iteration 2 would appear to switch clusters, when actually the cluster IDs just differ.
- With alignment, we correctly recognize that both assignments refer to the same biological cluster.
- This enables accurate posterior probability computation: p_i(c) measures true assignment stability, not label numbering artifacts.

**Fallback:** If Hungarian algorithm fails (e.g., unequal cluster counts), a greedy argmax approach is used: for each reference cluster, pick the bootstrap cluster with maximum overlap.

---

## Key Data Structures

### 1. Bootstrap Results Dictionary

```python
bootstrap_results_dict = {
    'embryo_ids': List[str],             # Embryo identifiers (required for tracking)
    'reference_labels': np.ndarray,      # Shape: (n_embryos,), consensus labels
    'bootstrap_results': [               # List of length n_bootstrap
        {
            'labels': np.ndarray,        # Shape: (n_embryos,), -1 for unsampled
            'indices': np.ndarray,       # Sampled embryo indices
            'silhouette': float          # Silhouette score for this iteration
        },
        ...
    ],
    'coassoc': np.ndarray,               # Shape: (n_embryos, n_embryos), optional
    'mean_ari': float                    # Mean adjusted Rand index
}

# Access by embryo ID
embryo_id = 'cep290_wt_run1_embryo_05'
idx = bootstrap_results_dict['embryo_ids'].index(embryo_id)
label = bootstrap_results_dict['reference_labels'][idx]
```

### 2. Posterior Analysis Results

```python
posterior_analysis = {
    'embryo_ids': List[str],             # Embryo identifiers (same order as arrays)
    'p_matrix': np.ndarray,              # Shape: (n_embryos, n_clusters)
    'sample_counts': np.ndarray,         # Shape: (n_embryos,)
    'max_p': np.ndarray,                 # Shape: (n_embryos,)
    'entropy': np.ndarray,               # Shape: (n_embryos,)
    'log_odds_gap': np.ndarray,          # Shape: (n_embryos,)
    'modal_cluster': np.ndarray,         # Shape: (n_embryos,), dtype=int
    'second_best_cluster': np.ndarray    # Shape: (n_embryos,), dtype=int
}

# Access by embryo ID
idx = posterior_analysis['embryo_ids'].index('cep290_wt_run1_embryo_05')
max_prob = posterior_analysis['max_p'][idx]
cluster = posterior_analysis['modal_cluster'][idx]
```

### 3. Classification Results

```python
classification = {
    'embryo_ids': List[str],             # Embryo identifiers (same order as arrays)
    'category': np.ndarray,              # Shape: (n_embryos,), dtype=str
                                         # Values: 'core', 'uncertain', 'outlier'
    'cluster': np.ndarray,               # Shape: (n_embryos,), dtype=int
    'max_p': np.ndarray,                 # Shape: (n_embryos,)
    'log_odds_gap': np.ndarray,          # Shape: (n_embryos,)
    'thresholds': {                      # Thresholds used
        'max_p': float,
        'log_odds_gap': float,
        'outlier_max_p': float
    }
}

# Filter core members
core_mask = classification['category'] == 'core'
core_embryo_ids = [eid for eid, is_core in zip(classification['embryo_ids'], core_mask) if is_core]
```

**Design Decision:** All data structures include `embryo_ids` as a list of strings. This ensures:
- Explicit tracking across experiments (e.g., 'cep290_wt_run1_embryo_05')
- Self-documenting data structures
- Robust merging of results from different runs
- Clear lookup: `embryo_ids[i]` corresponds to all arrays at index `i`

---

## Data Loading

The data loading layer is intentionally simple and explicit. Every step is visible for debugging and modification. No magic - just clear, composable functions.

### Data Loading Pipeline

Four simple steps from CSV to analysis-ready arrays:

```
Step 1: load_experiment_dataframe()
        ↓ df_raw (DataFrame with all columns)

Step 2: extract_trajectory_dataframe()
        ↓ df_traj (Long-format: embryo_id, time, metric_value, metadata)

Step 3: dataframe_to_trajectories()
        ↓ time_arrays, metric_arrays (Lists of np.ndarray, variable length)

Step 4: interpolate_trajectories()
        ↓ time_grid, traj_grid (Common grid, uniform length)
```

### Step 1: Load Raw Data

```python
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# Load all data from experiment
df_raw = load_experiment_dataframe('cep290_run1')

# Inspect what we got
print(df_raw.shape)  # (5000 rows, 281 columns)
print(df_raw.columns)  # All original columns
```

### Step 2: Extract Trajectories (Long Format)

```python
from src.analyze.trajectory_analysis.data_loading import extract_trajectory_dataframe

# Extract trajectories with filtering
df_traj = extract_trajectory_dataframe(
    df_raw,
    filter_dict={'genotype': 'wildtype'},
    keep_cols=['genotype', 'predicted_stage_hpf', 'replicate'],  # Optional metadata
    min_timepoints=3
)

# Now in long format
print(df_traj.columns)  # ['embryo_id', 'time', 'metric_value', 'genotype', ...]
print(len(df_traj))  # ~1200 rows (24 timepoints × 50 embryos)

# Can inspect raw trajectories here for QC
import matplotlib.pyplot as plt
for embryo_id in df_traj['embryo_id'].unique()[:5]:
    embryo_df = df_traj[df_traj['embryo_id'] == embryo_id]
    plt.plot(embryo_df['time'], embryo_df['metric_value'], alpha=0.3, label=embryo_id)
plt.legend()
plt.title("Raw trajectories (pre-interpolation)")
```

### Step 3: Convert to Arrays (Raw, No Interpolation)

```python
from src.analyze.trajectory_analysis.data_loading import dataframe_to_trajectories

# Extract per-embryo arrays
time_arrays, metric_arrays, embryo_ids = dataframe_to_trajectories(df_traj)

# Variable length per embryo (different # of timepoints)
print(f"Embryo 0: {len(time_arrays[0])} timepoints")
print(f"Embryo 1: {len(time_arrays[1])} timepoints")

# All aligned
assert len(time_arrays) == len(metric_arrays) == len(embryo_ids)
```

### Step 4: Interpolate to Common Grid

```python
from src.analyze.trajectory_analysis.data_loading import interpolate_trajectories

# Interpolate to common time grid
time_grid, traj_grid, embryo_ids = interpolate_trajectories(
    time_arrays, metric_arrays, embryo_ids, grid_step=0.5
)

# Now all same length
assert all(len(t) == len(time_grid) for t in traj_grid)
print(f"Grid: {len(time_grid)} timepoints")
print(f"All trajectories: {len(traj_grid)} embryos × {len(time_grid)} timepoints each")
```

### Complete Data Loading Example

```python
from src.analyze.trajectory_analysis.data_loading import (
    load_experiment_dataframe,
    extract_trajectory_dataframe,
    dataframe_to_trajectories,
    interpolate_trajectories
)

# Step 1: Load
df_raw = load_experiment_dataframe('cep290_run1')

# Step 2: Extract
df_traj = extract_trajectory_dataframe(
    df_raw,
    filter_dict={'genotype': 'wildtype'},
    keep_cols=['genotype', 'predicted_stage_hpf']
)

# Step 3: To arrays
time_arrays, metric_arrays, embryo_ids = dataframe_to_trajectories(df_traj)

# Step 4: Interpolate
time_grid, traj_grid, embryo_ids = interpolate_trajectories(
    time_arrays, metric_arrays, embryo_ids, grid_step=0.5
)

# Now ready for analysis
print(f"Loaded {len(embryo_ids)} embryos")
print(f"Time: {time_grid[0]:.1f} - {time_grid[-1]:.1f} HPF")
print(f"Ready for DTW clustering!")
```

### Design Notes

- **Explicit**: Every step is a separate function call - no hidden magic
- **Debuggable**: You can inspect `df_traj` before/after each step
- **Flexible**: Easy to add new filtering, metadata columns, or QC steps
- **Format-Isolated**: If CSV structure changes, only update internal `_load_df*_format()` functions
- **MVP**: Simple, not over-engineered - room to grow as needs change

---

## Example Usage

### Complete End-to-End Pipeline

```python
from src.analyze.trajectory_analysis import (
    extract_trajectories,
    interpolate_to_common_grid,
    compute_dtw_distance_matrix,
    run_bootstrap_hierarchical,
    analyze_bootstrap_results,
    classify_membership_2d,
    plot_posterior_heatmap,
    plot_2d_scatter
)

# 1. Load and process trajectories
df_long = load_my_data()  # Your data loading function
trajectories, embryo_ids, orig_lens = extract_trajectories(
    df_long,
    genotype='wildtype',
    min_timepoints=3
)
# embryo_ids is now ['cep290_wt_run1_emb_01', 'cep290_wt_run1_emb_02', ...]

trajs_interp, embryo_ids, common_grid = interpolate_to_common_grid(
    df_long,
    grid_step=0.5
)

# 2. Compute DTW distance matrix
D = compute_dtw_distance_matrix(trajs_interp, window=5, verbose=True)

# 3. Bootstrap clustering (with embryo_ids for tracking)
k = 3
bootstrap_results = run_bootstrap_hierarchical(
    D,
    k=k,
    embryo_ids=embryo_ids,  # REQUIRED: for explicit tracking
    n_bootstrap=100,
    frac=0.8,
    random_state=42
)
# bootstrap_results now includes 'embryo_ids' key

# 4. Analyze posterior probabilities
posterior_analysis = analyze_bootstrap_results(bootstrap_results)
# posterior_analysis includes 'embryo_ids' (copied from bootstrap_results)

# 5. Classify membership quality
classification = classify_membership_2d(
    max_p=posterior_analysis['max_p'],
    log_odds_gap=posterior_analysis['log_odds_gap'],
    modal_cluster=posterior_analysis['modal_cluster'],
    threshold_max_p=0.8,
    threshold_log_odds=0.7
)
# classification includes 'embryo_ids' (copied from posterior_analysis)

# 6. Visualize results
fig1 = plot_posterior_heatmap(posterior_analysis)  # embryo_ids used for labeling
fig2 = plot_2d_scatter(classification)             # embryo_ids used for labeling

# 7. Examine results - look up by embryo ID
embryo_id = 'cep290_wt_run1_emb_05'
idx = classification['embryo_ids'].index(embryo_id)
print(f"{embryo_id}:")
print(f"  Category: {classification['category'][idx]}")
print(f"  Cluster: {classification['cluster'][idx]}")
print(f"  Max probability: {classification['max_p'][idx]:.3f}")

# 8. Filter and analyze
core_mask = classification['category'] == 'core'
core_embryo_ids = [eid for eid, is_core in zip(classification['embryo_ids'], core_mask) if is_core]
print(f"\nCore members: {len(core_embryo_ids)}/{len(classification['embryo_ids'])}")
print(f"Core embryos: {core_embryo_ids}")

# 9. Cross-experiment comparison
# Results automatically track which experiment each embryo came from
for result in [bootstrap_results, posterior_analysis, classification]:
    for embryo_id in result['embryo_ids']:
        print(f"{embryo_id} → experiment/run encoded in ID")
```

**Key Points:**
- All functions accept `embryo_ids` as a list of strings
- All results include `embryo_ids` in the same order as array indices
- Embryo IDs should encode experiment info: `'cep290_wt_run1_embryo_05'`
- Lookup by ID: `idx = result['embryo_ids'].index(embryo_id)`
- No ambiguity about which embryo is which index

---

## Migration Plan

### From `dtw_time_trend_analysis/`

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `dtw_time_trend_analysis/dtw_distance.py` | `trajectory_analysis/dtw_distance.py` | To migrate |
| `dtw_time_trend_analysis/dtw_clustering.py` | `trajectory_analysis/dba.py` | To migrate + rename |
| `dtw_time_trend_analysis/trajectory_utils.py` | `trajectory_analysis/trajectory_utils.py` | To migrate |
| `dtw_time_trend_analysis/trajectory_statistics.py` | `trajectory_analysis/correlation_analysis.py` | To migrate + rename |

### From `results/mcolon/20251106_robust_clustering/`

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `bootstrap_posteriors.py` | `cluster_posteriors.py` | To migrate |
| `adaptive_classification.py` | `cluster_classification.py` | To migrate |
| `consensus_clustering_plotting.py` | `plotting.py` | To migrate |
| `run_hierarchical_posterior_analysis.py::run_bootstrap_hierarchical()` | `bootstrap_clustering.py` | Extract function |
| `utils/bootstrap_alignment.py` | Internal to `cluster_posteriors.py` | To consolidate |
| `utils/posterior_metrics.py` | Internal to `cluster_posteriors.py` | To consolidate |

### API Standardization

All functions will use consistent parameter naming:

```python
# Standard column names
embryo_id_col: str = 'embryo_id'
metric_col: str = 'normalized_baseline_deviation'
time_col: str = 'predicted_stage_hpf'
genotype_col: str = 'genotype'

# Standard processing parameters
grid_step: float = 0.5
min_timepoints: int = 3
window: int = 5  # DTW Sakoe-Chiba band

# Standard bootstrap parameters
n_bootstrap: int = 100
frac: float = 0.8
random_state: int = 42

# Standard classification parameters
threshold_max_p: float = 0.8
threshold_log_odds: float = 0.7
threshold_outlier_max_p: float = 0.5
```

---

## Dependencies

### Internal
- `numpy` - Array operations
- `scipy` - Optimization (Hungarian algorithm), statistics
- `pandas` - DataFrame manipulation
- `sklearn` - Clustering algorithms (AgglomerativeClustering)
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

### Cross-Module
- `src.analyze.utils.plotting` - Shared plotting utilities
- May import from other `src.analyze` modules as needed

---

## Development Status

### Completed
- ✅ Package structure defined
- ✅ Migration plan documented
- ✅ Function workflow mapped

### In Progress
- 🔄 Creating placeholder files
- 🔄 Migrating core functions
- 🔄 API standardization
- 🔄 Adding type hints

### To Do
- ⏳ Comprehensive docstrings with examples
- ⏳ Unit tests for each module
- ⏳ Integration tests for full pipeline
- ⏳ Update project documentation

---

## References

### Theory
- **DTW:** Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition.
- **DBA:** Petitjean, F., Ketterlin, A., & Gançarski, P. (2011). A global averaging method for dynamic time warping.
- **Bootstrap Clustering:** Fang, Y., & Wang, J. (2012). Selection of the number of clusters via the bootstrap method.
- **Hungarian Algorithm:** Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly.
  - Used for optimal label alignment across bootstrap iterations
  - Implementation: `scipy.optimize.linear_sum_assignment`

### Implementation
- Original DTW implementation: `src/analyze/dtw_time_trend_analysis/`
- Consensus clustering prototypes: `results/mcolon/20251106_robust_clustering/`
- Migration notes: `results/mcolon/20251106_robust_clustering/MIGRATION_NOTES.md`

---

## Contact

For questions about this package:
- See original implementation logs in `docs/refactors/streamline-snakemake/`
- Refer to `results/mcolon/20251106_robust_clustering/cluster_assignment_quality.md` for methodology

---

**Last Updated:** 2025-11-07
