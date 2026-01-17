# Trajectory Analysis Package

**Location:** `src/analyze/trajectory_analysis/`
**Status:** Beta (v0.2.0 - DataFrame-centric API with time-axis alignment fixes)
**Version:** 0.2.0

> **Note:** This package is being reorganized into subpackages. The old flat structure
> is deprecated. Please use the new organized imports:
> - `from src.analyze.trajectory_analysis.distance import compute_dtw_distance_matrix, dba`
> - `from src.analyze.trajectory_analysis.utilities import extract_trajectories_df, fit_pca_on_embeddings`
> - `from src.analyze.trajectory_analysis.io import load_phenotype_file`
> - `from src.analyze.trajectory_analysis.qc import identify_outliers` (coming soon)
> - `from src.analyze.trajectory_analysis.clustering import run_bootstrap_hierarchical` (coming soon)
> - `from src.analyze.trajectory_analysis.viz import plot_trajectories_faceted` (coming soon)
>
> Top-level imports still work but emit deprecation warnings. See `docs/PROGRESS.md`
> for reorganization status.

---

## Overview

The `trajectory_analysis` package provides a comprehensive framework for analyzing temporal trajectories using Dynamic Time Warping (DTW), bootstrap-based consensus clustering, and probabilistic quality assessment.

**v0.2.0 Major Change:** The package now uses a **DataFrame-centric API** where the time column (hpf - hours post fertilization) travels with trajectory data throughout the entire pipeline. This eliminates the time-axis misalignment bugs present in v0.1.x where array indices were assumed to correspond to a shared time grid.

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
├── k_selection.py               # K-selection pipeline
├── distance_filtering.py        # IQR and k-NN distance filtering
├── consensus_pipeline.py        # Two-stage outlier filtering
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

#### `k_selection.py`
- Evaluate multiple k values before filtering
- Quality metrics comparison across k range
- K-selection visualization
- Two-phase pipeline (k-selection + consensus clustering)

#### `distance_filtering.py`
- IQR distance-based outlier filtering (default)
- k-NN distance-based filtering (legacy)
- Stage 1 and Stage 2 filtering

#### `consensus_pipeline.py`
- Two-stage outlier filtering workflow
- Consensus clustering with filtering
- Integration of bootstrap + filtering + classification

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

### Choosing Between Clustering Methods

Two clustering methods are available: **hierarchical** and **k-medoids**. Both are equal alternatives with different strengths. Users should be aware of the trade-offs and ideally use both methods to compare results.

#### K-medoids Clustering (`run_bootstrap_kmedoids`)

**Strengths:**
- **Much less sensitive to noise** - does not get dominated by noisy data
- Better for cross-experiment comparisons
- Robust to outliers and extreme values
- Uses actual data points as cluster centers (medoids), providing interpretable representatives

**Requirements:**
- Requires `scikit-learn-extra` package: `pip install scikit-learn-extra`

**Usage:**
```python
from src.analyze.trajectory_analysis import run_bootstrap_kmedoids

# Use k-medoids instead of hierarchical
bootstrap_results = run_bootstrap_kmedoids(
    D, k=3, embryo_ids=embryo_ids, n_bootstrap=100
)
medoid_indices = bootstrap_results['medoid_indices']  # Representative embryos
```

#### Hierarchical Clustering (`run_bootstrap_hierarchical`)

**Strengths:**
- Better at detecting subtle trends within groups
- More granular at splitting trends
- Standard method, widely understood
- No additional dependencies

**Limitations:**
- Can be dominated by noise in the data
- Less robust to outliers

**Usage:**
```python
from src.analyze.trajectory_analysis import run_bootstrap_hierarchical

# Standard hierarchical clustering
bootstrap_results = run_bootstrap_hierarchical(
    D, k=3, embryo_ids=embryo_ids, n_bootstrap=100
)
```

#### Choosing the Right Method

**Recommendation: Use both methods** when possible to compare results.

Choose based on your data characteristics:
- **Noisy data or cross-experiment analysis** → Use k-medoids
- **Clean data with subtle within-group trends** → Use hierarchical
- **Best practice:** Run both methods and compare cluster assignments and quality metrics

**Comparison Example:**
```python
# Compare both methods
results_hier = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids)
results_kmed = run_bootstrap_kmedoids(D, k=3, embryo_ids=embryo_ids)

# Then compare cluster assignments and quality metrics
posteriors_hier = analyze_bootstrap_results(results_hier)
posteriors_kmed = analyze_bootstrap_results(results_kmed)
```

**Output Compatibility:**
Both methods return identical data structures, making them fully interchangeable in downstream analysis. All posterior probability and quality assessment functions work with both methods.

#### Advanced Workflow: Manual Selection and Subclustering

For deeper investigation of cluster structure, **manual cluster selection and subclustering** is recommended:

**Workflow:**
1. **Initial clustering** - Run k-medoids or hierarchical to get preliminary clusters
2. **Interactive exploration** - Use interactive plots (e.g., Plotly) to visualize trajectories by cluster
3. **Manual selection** - Identify clusters that show heterogeneity or mixed trends
4. **Subclustering** - Re-run clustering on selected cluster members only
5. **Validation** - Confirm trends with interactive plots before accepting subcluster structure

**Why Manual Selection?**
- Automated k-selection may miss subtle within-cluster structure
- Interactive plots reveal patterns not obvious in static visualizations
- Domain expertise helps identify biologically meaningful subgroups
- Allows iterative refinement based on visual inspection

**Example Subclustering Workflow:**
```python
# 1. Initial clustering
results = run_bootstrap_kmedoids(D, k=3, embryo_ids=embryo_ids)
posteriors = analyze_bootstrap_results(results)

# 2. Interactive exploration - identify cluster 1 has mixed trends
# (Use Plotly or similar for interactive visualization)

# 3. Manual selection - extract cluster 1 members
cluster_1_mask = posteriors['modal_cluster'] == 1
cluster_1_embryo_ids = [eid for eid, mask in zip(embryo_ids, cluster_1_mask) if mask]
cluster_1_indices = [i for i, mask in enumerate(cluster_1_mask) if mask]

# 4. Subclustering - re-cluster cluster 1 only
D_subcluster = D[np.ix_(cluster_1_indices, cluster_1_indices)]
subcluster_results = run_bootstrap_hierarchical(
    D_subcluster, k=2, embryo_ids=cluster_1_embryo_ids
)

# 5. Validation - plot subclusters with interactive viewer
# Confirm trends are consistent within each subcluster
```

**Best Practices:**
- Use interactive plots (Plotly, Bokeh) to explore trajectories dynamically
- Look for heterogeneity in trajectory shapes, timing, or amplitude
- Subcluster only when visual inspection confirms mixed trends
- Document selection criteria for reproducibility
- Re-run quality assessment on subclusters to confirm coherence

---

### K-selection Pipeline

The k-selection pipeline helps you choose the optimal number of clusters (k) before running the full consensus clustering workflow. This addresses the challenge of deciding k upfront when filtering might remove embryos that would form good clusters.

#### Workflow: Two-Phase Approach

**Phase 1: K Selection (No Filtering)**
1. Compute distance matrix on all embryos (no filtering)
2. Run bootstrap clustering for k in [2, 3, 4, 5, 6, ...]
3. For each k, compute quality metrics:
   - % Core (high confidence assignments)
   - % Outlier (low confidence)
   - Mean max_p (confidence)
   - Mean entropy (uncertainty)
   - Silhouette score
4. Plot comparison and choose optimal k

**Phase 2: Final Clustering (With Filtering)**
5. Run consensus clustering with chosen k + IQR/posterior filtering
6. Generate final cluster assignments and quality metrics

#### Functions

**`evaluate_k_range(D, embryo_ids, k_range=[2,3,4,5,6], method='hierarchical')`**
- Evaluates multiple k values without filtering
- Supports `method='hierarchical'` or `method='kmedoids'`
- Returns comprehensive quality metrics for each k

**Usage:**
```python
from src.analyze.trajectory_analysis import evaluate_k_range

# Evaluate k range with k-medoids (recommended for noisy data)
results = evaluate_k_range(
    D, embryo_ids,
    k_range=[2, 3, 4, 5],
    method='kmedoids',
    n_bootstrap=100
)

# Access results
summary_df = results['summary_df']  # Quality metrics by k
best_k = results['best_k']  # Recommended k based on metrics
```

**`plot_k_selection(results, output_path=None)`**
- Creates 2x2 grid plot comparing metrics across k values:
  - Membership % vs k (core/uncertain/outlier)
  - Mean max_p vs k
  - Mean entropy vs k
  - Silhouette score vs k

**`run_k_selection_with_plots(D, embryo_ids, df_interpolated, ...)`**
- Complete pipeline with trajectory visualization
- Generates individual membership trajectory plots for each k
- Creates summary comparison plot
- Saves metrics CSV, cluster assignments CSV, and full results pickle

**`run_two_phase_pipeline(D, embryo_ids, df_interpolated, ...)`**
- Combined k-selection + consensus clustering
- Phase 1: Picks optimal k without filtering
- Phase 2: Runs full consensus clustering with chosen k and filtering
- Returns both k-selection results and final clustering results

#### When to Use K-selection

**Use k-selection when:**
- You don't know the optimal number of clusters upfront
- You want data-driven k selection
- You're working with a new dataset or biological system

**Skip k-selection when:**
- k is known from biology (e.g., number of genotypes)
- You're replicating a previous analysis with established k
- You want to try multiple k values manually

#### Quality Metrics

The k-selection pipeline evaluates these metrics for each k:

- **% Core members**: Fraction of embryos with high-confidence assignments (max_p > 0.8)
- **% Outlier members**: Fraction with low confidence (max_p < 0.5)
- **Mean max_p**: Average confidence in top cluster assignment
- **Mean entropy**: Average uncertainty across all clusters
- **Silhouette score**: Cluster separation quality (higher is better)

**Interpretation:**
- Higher % Core → More confident cluster structure
- Lower % Outlier → Fewer ambiguous assignments
- Higher Mean max_p → Stronger cluster memberships
- Lower Mean entropy → Less uncertainty
- Higher Silhouette → Better-separated clusters

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

### Two Analysis Pathways

The package supports two main workflows:

**Option A: Direct Clustering** (when k is known)
- Use when the number of clusters is predetermined (e.g., by biology)
- See "Complete Analysis Pipeline" below

**Option B: K-selection Pipeline** (when k is uncertain)
- Use when you need data-driven k selection
- See "K-selection Workflow" below

---

### Complete Analysis Pipeline (Option A)

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
│                   (Choose hierarchical or k-medoids)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         run_bootstrap_hierarchical()  OR  run_bootstrap_kmedoids()
           - Standard method               - Robust to noise
           - Better for subtle trends      - Better for cross-experiment
           - No extra dependencies         - Requires scikit-learn-extra
                    │                       │
                    └─────────┬─────────────┘
                              ▼
                    - Resample 80% of embryos (n_bootstrap times)
                    - Clustering on each subsample
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

### K-selection Workflow (Option B)

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LOADING & PREPROCESSING                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  [Same as Option A: extract + interpolate + DTW]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 1: K-SELECTION (NO FILTERING)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            k_selection.evaluate_k_range(D, embryo_ids,
                                          k_range=[2,3,4,5,6],
                                          method='hierarchical' or 'kmedoids')
                    - For each k:
                      * Run bootstrap clustering
                      * Compute quality metrics:
                        - % Core / % Outlier
                        - Mean max_p, mean entropy
                        - Silhouette score
                    - Returns: quality metrics by k
                              │
                              ▼
            k_selection.plot_k_selection(results)
                    - 2x2 comparison plot
                    - Visualize metrics vs k
                              │
                              ▼
                    USER DECISION: Choose optimal k
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          PHASE 2: FINAL CLUSTERING (WITH FILTERING)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            consensus_pipeline.run_consensus_clustering(
                D, embryo_ids, k=chosen_k, ...)
                    - Stage 1: IQR distance filtering
                    - Stage 2: Posterior probability filtering
                    - Bootstrap clustering with chosen k
                    - Quality assessment
                    - Returns: final cluster assignments
                              │
                              ▼
                         VISUALIZATION
```

**Shortcut Function:**
```python
# Run both phases automatically
results = run_two_phase_pipeline(D, embryo_ids, df_interpolated, ...)
# Phase 1: Picks optimal k based on quality metrics
# Phase 2: Runs consensus clustering with chosen k and filtering
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

### Shortcut: DataFrame → DTW Distance Matrix

```python
from src.analyze.trajectory_analysis import compute_dtw_distance_from_df

# One-liner wrapper (optionally interpolates first)
D, embryo_ids, time_grid, trajectories = compute_dtw_distance_from_df(
    df_traj,
    window=5,
    interpolate=True,   # set False to use raw (ragged) trajectories
    grid_step=0.5,
    return_arrays=True
)
```

### Design Notes

- **Explicit**: Every step is a separate function call - no hidden magic
- **Debuggable**: You can inspect `df_traj` before/after each step
- **Flexible**: Easy to add new filtering, metadata columns, or QC steps
- **Format-Isolated**: If CSV structure changes, only update internal `_load_df*_format()` functions
- **MVP**: Simple, not over-engineered - room to grow as needs change

---

## Example Usage (v0.2.0 - Current API)

### Complete End-to-End Pipeline with DataFrame-Centric API

```python
from src.analyze.trajectory_analysis import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    df_to_trajectories,
    compute_dtw_distance_matrix,
    run_bootstrap_hierarchical,
    analyze_bootstrap_results,
    classify_membership_2d,
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df,
    plot_posterior_heatmap,
    plot_2d_scatter
)

# 1. Load and process trajectories (DataFrame-centric)
df_raw = load_my_data()  # Your data loading function → DataFrame

df_filtered = extract_trajectories_df(
    df_raw,
    genotype_filter='wildtype',
    min_timepoints=3
)
# df_filtered has columns: [embryo_id, hpf, metric_value, ...]

df_interpolated = interpolate_to_common_grid_df(
    df_filtered,
    grid_step=0.5
)
# df_interpolated: same columns, all embryos aligned to common hpf grid

# 2. Convert to arrays for DTW (using helper function)
trajectories, embryo_ids, common_grid = df_to_trajectories(df_interpolated)

# 3. Compute DTW distance matrix
D = compute_dtw_distance_matrix(trajectories, window=5, verbose=True)

# 4. Bootstrap clustering (with embryo_ids for tracking)
k = 3
bootstrap_results = run_bootstrap_hierarchical(
    D,
    k=k,
    embryo_ids=embryo_ids,
    n_bootstrap=100,
    frac=0.8,
    random_state=42
)

# 5. Analyze posterior probabilities
posterior_analysis = analyze_bootstrap_results(bootstrap_results)

# 6. Classify membership quality
classification = classify_membership_2d(
    max_p=posterior_analysis['max_p'],
    log_odds_gap=posterior_analysis['log_odds_gap'],
    modal_cluster=posterior_analysis['modal_cluster'],
    embryo_ids=posterior_analysis['embryo_ids']
)

# 7. Visualize results using DataFrame (preserves time alignment)
fig1 = plot_cluster_trajectories_df(
    df_interpolated,
    posterior_analysis['modal_cluster'],
    embryo_ids=posterior_analysis['embryo_ids'],
    show_mean=True
)

fig2 = plot_membership_trajectories_df(
    df_interpolated,
    classification,
    embryo_ids=classification['embryo_ids'],
    per_cluster=True
)

fig3 = plot_posterior_heatmap(posterior_analysis)
fig4 = plot_2d_scatter(classification)

# 8. Examine results - look up by embryo ID
embryo_id = 'cep290_wt_run1_emb_05'
idx = classification['embryo_ids'].index(embryo_id)
print(f"{embryo_id}:")
print(f"  Category: {classification['category'][idx]}")
print(f"  Cluster: {classification['cluster'][idx]}")
print(f"  Max probability: {classification['max_p'][idx]:.3f}")

# 9. Filter and analyze
core_mask = classification['category'] == 'core'
core_embryo_ids = [eid for eid, is_core in zip(classification['embryo_ids'], core_mask) if is_core]
print(f"\nCore members: {len(core_embryo_ids)}/{len(classification['embryo_ids'])}")
```

**Key Points:**
- Use `extract_trajectories_df()` and `interpolate_to_common_grid_df()` to keep data in DataFrame format
- Use `df_to_trajectories()` helper for one-line conversion to arrays when needed for DTW
- Use `plot_*_df()` versions which accept `df_interpolated` directly, preserving time alignment
- Time column (hpf) is always explicit and never lost during transformations
- No more worrying about grid index slicing - the DataFrame handles alignment automatically

---

## Migration Guide (v0.1.x → v0.2.0)

### What Changed

v0.2.0 introduces a **DataFrame-centric API** to fix time-axis misalignment bugs:

| Issue | v0.1.x Problem | v0.2.0 Solution |
|-------|----------------|-----------------|
| **Time axis alignment** | Arrays assumed shared grid; late-start embryos plotted against wrong times | Keep DataFrame with explicit hpf column throughout pipeline |
| **Grid index tracking** | `pad_trajectories_for_plotting()` trims arrays, losing time info | Use DataFrame groupby for natural time alignment |
| **API clarity** | Function names ambiguous; easy to mix array/DataFrame stages | Clear naming: `extract_trajectories_df()`, `interpolate_to_common_grid_df()` |
| **Parameter confusion** | Unused `consensus_method` parameter in `compute_consensus_labels()` | Parameter removed; function simplified |
| **Single-cluster edge case** | IndexError when k=1 in `compute_quality_metrics()` | Added k=1 guard for `second_best_cluster` |

### Updating Your Code

#### Old API (v0.1.x) - Array-based
```python
from src.analyze.trajectory_analysis import (
    extract_trajectories,
    interpolate_to_common_grid,
    plot_cluster_trajectories
)

# Returns separate arrays and grid
trajectories, embryo_ids, orig_lens = extract_trajectories(df, genotype='wildtype')
trajs_interp, _, common_grid = interpolate_to_common_grid(trajectories)

# Plotting requires manual grid slicing (error-prone!)
fig = plot_cluster_trajectories(trajs_interp, common_grid, labels)
```

#### New API (v0.2.0) - DataFrame-centric
```python
from src.analyze.trajectory_analysis import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    plot_cluster_trajectories_df
)

# Data stays in DataFrame, time never lost
df_filtered = extract_trajectories_df(df, genotype_filter='wildtype')
df_interpolated = interpolate_to_common_grid_df(df_filtered)

# Plotting uses DataFrame directly, time alignment automatic
fig = plot_cluster_trajectories_df(df_interpolated, labels, embryo_ids=ids)
```

### Migration Steps

**If you're using array-based functions:**

1. **For data processing:**
   - Replace `extract_trajectories()` → `extract_trajectories_df()`
   - Replace `interpolate_to_common_grid()` → `interpolate_to_common_grid_df()`
   - Keep data in DataFrame format as long as possible

2. **For DTW computation:**
   - Use new `df_to_trajectories()` helper: `trajectories, ids, grid = df_to_trajectories(df_interpolated)`
   - This handles the array conversion cleanly in one line

3. **For plotting:**
   - Replace `plot_cluster_trajectories()` → `plot_cluster_trajectories_df(df_interpolated, labels, ...)`
   - Replace `plot_membership_trajectories()` → `plot_membership_trajectories_df(df_interpolated, classification, ...)`
   - No need to pass `common_grid` separately; it's in the DataFrame

4. **For bootstrap:**
   - `run_bootstrap_hierarchical()` signature unchanged
   - `compute_consensus_labels()` no longer accepts `consensus_method` parameter (removed unused param)

### Backward Compatibility

Old array-based functions are **still available but deprecated**:
- `extract_trajectories()` → DeprecationWarning (use `extract_trajectories_df()`)
- `interpolate_to_common_grid()` → DeprecationWarning (use `interpolate_to_common_grid_df()`)
- `pad_trajectories_for_plotting()` → DeprecationWarning (use DataFrame directly)
- `plot_cluster_trajectories()` → DeprecationWarning (use `plot_cluster_trajectories_df()`)
- `plot_membership_trajectories()` → DeprecationWarning (use `plot_membership_trajectories_df()`)

**Important:** Deprecated functions have time-axis alignment bugs. Use new API for correct results.

### Example Migration

**Before (v0.1.x):**
```python
trajectories, embryo_ids, orig_lens = extract_trajectories(df, genotype='wildtype')
trajs_interp, _, grid = interpolate_to_common_grid(trajectories)
D = compute_dtw_distance_matrix(trajs_interp, window=5)
bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids, n_bootstrap=100)
posteriors = analyze_bootstrap_results(bootstrap_results)
fig = plot_cluster_trajectories(trajs_interp, grid, posteriors['modal_cluster'])  # BUG: wrong times!
```

**After (v0.2.0):**
```python
df_filtered = extract_trajectories_df(df, genotype_filter='wildtype')
df_interpolated = interpolate_to_common_grid_df(df_filtered)
trajectories, embryo_ids, grid = df_to_trajectories(df_interpolated)
D = compute_dtw_distance_matrix(trajectories, window=5)
bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids, n_bootstrap=100)
posteriors = analyze_bootstrap_results(bootstrap_results)
fig = plot_cluster_trajectories_df(df_interpolated, posteriors['modal_cluster'])  # CORRECT!
```

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

### Completed (v0.2.0)
- ✅ DataFrame-centric API implementation
- ✅ Time-axis alignment bug fixes
- ✅ Deprecation warnings for v0.1.x functions
- ✅ Migration guide and documentation
- ✅ Edge case fixes (k=1 single-cluster support)
- ✅ Parameter cleanup (removed unused `consensus_method`)

### Tested & Working
- ✅ DTW distance computation with Sakoe-Chiba band
- ✅ DBA (DTW Barycenter Averaging)
- ✅ Bootstrap hierarchical clustering
- ✅ **Bootstrap k-medoids clustering** (added Dec 26, 2025)
- ✅ **K-selection pipeline with quality metrics** (added Dec 23, 2025)
- ✅ **Two-phase consensus clustering** (k-selection + filtering)
- ✅ **IQR distance-based filtering** (default, replaces k-NN)
- ✅ Posterior probability computation with Hungarian alignment
- ✅ Membership classification (2D gating and adaptive)
- ✅ Trajectory visualization (DataFrame-based)

### Known Limitations (Not Planned for Future)
- ❌ Coassociation matrix computation (deprecated in favor of posteriors)
- ❌ Mean ARI metric (replaced by per-embryo posterior metrics)
- ❌ Greedy Hungarian fallback (direct assignment always used now)

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

**Last Updated:** 2025-12-26
