# Cluster Assignment Utility Design - Three Versions

## Context
We need a utility function `get_cluster_assignments()` that runs clustering for multiple k values and returns a simple DataFrame with embryo_id and cluster assignments.

**Existing functions** in `src/analyze/trajectory_analysis/`:
- `compute_dtw_distance_matrix()` - from dtw_distance.py
- `run_bootstrap_hierarchical()` - from bootstrap_clustering.py
- `analyze_bootstrap_results()` - from cluster_posteriors.py
- `classify_membership_2d()` - from cluster_classification.py

---

## Version 1: Minimal Wrapper (Add to Existing Module)

**Location**: Add to `src/analyze/trajectory_analysis/bootstrap_clustering.py`

**New Functions**:
```python
def get_cluster_assignments(
    distance_matrix,
    embryo_ids,
    k_values=[3, 4, 5, 6],
    n_bootstrap=100,
    bootstrap_frac=0.8
):
    """Run clustering for multiple k values.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    embryo_ids : list
        List of embryo identifiers
    k_values : list of int
        K values to test
    n_bootstrap : int
        Number of bootstrap iterations
    bootstrap_frac : float
        Fraction to sample per iteration

    Returns
    -------
    df_assignments : DataFrame
        Columns: embryo_id, cluster_k3, cluster_k4, cluster_k5, cluster_k6,
                 max_p_k3, max_p_k4, max_p_k5, max_p_k6
    all_results : dict
        Full results dict keyed by k: {3: {...}, 4: {...}, ...}
    """
    import pandas as pd
    from .cluster_posteriors import analyze_bootstrap_results

    all_results = {}
    assignment_dfs = []

    for k in k_values:
        # Run bootstrap clustering
        bootstrap_results = run_bootstrap_hierarchical(
            distance_matrix, k, embryo_ids,
            n_bootstrap=n_bootstrap,
            frac=bootstrap_frac
        )

        # Compute posteriors
        posteriors = analyze_bootstrap_results(bootstrap_results)

        # Store full results
        all_results[k] = {
            'bootstrap_results': bootstrap_results,
            'posteriors': posteriors
        }

        # Extract assignments for this k
        df_k = pd.DataFrame({
            'embryo_id': posteriors['embryo_ids'],
            f'cluster_k{k}': posteriors['modal_cluster'],
            f'max_p_k{k}': posteriors['max_p']
        })
        assignment_dfs.append(df_k)

    # Merge all k values
    df_assignments = assignment_dfs[0]
    for df_k in assignment_dfs[1:]:
        df_assignments = df_assignments.merge(df_k, on='embryo_id')

    return df_assignments, all_results
```

**Pros**:
- Minimal new code (~40 lines)
- Uses existing functions directly
- No new files/modules
- Easy to maintain

**Cons**:
- Doesn't include helper functions for WT identification
- Still requires writing cluster characteristics logic in analysis scripts

---

## Version 2: Standalone Utility Module (Recommended)

**Location**: New file `src/analyze/trajectory_analysis/cluster_assignment_utils.py`

**New Functions**:
```python
"""
High-level cluster assignment utilities for multi-k consensus clustering.

This module provides convenience functions for common clustering workflows,
wrapping the core bootstrap clustering and posterior analysis functions.
"""

import numpy as np
import pandas as pd
from .dtw_distance import compute_dtw_distance_matrix
from .bootstrap_clustering import run_bootstrap_hierarchical
from .cluster_posteriors import analyze_bootstrap_results
from .cluster_classification import classify_membership_2d


def get_cluster_assignments(
    distance_matrix,
    embryo_ids,
    k_values=[3, 4, 5, 6],
    n_bootstrap=100,
    bootstrap_frac=0.8,
    random_seed=42
):
    """Run clustering for multiple k values.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    embryo_ids : list
        List of embryo identifiers
    k_values : list of int
        K values to test
    n_bootstrap : int
        Number of bootstrap iterations
    bootstrap_frac : float
        Fraction to sample per iteration
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    df_assignments : DataFrame
        Columns: embryo_id, cluster_k3, cluster_k4, cluster_k5, cluster_k6,
                 max_p_k3, max_p_k4, max_p_k5, max_p_k6,
                 entropy_k3, entropy_k4, ...
    all_results : dict
        Full results dict keyed by k: {3: {...}, 4: {...}, ...}
    """
    all_results = {}
    assignment_dfs = []

    for k in k_values:
        # Run bootstrap clustering
        bootstrap_results = run_bootstrap_hierarchical(
            distance_matrix, k, embryo_ids,
            n_bootstrap=n_bootstrap,
            frac=bootstrap_frac,
            random_state=random_seed
        )

        # Compute posteriors
        posteriors = analyze_bootstrap_results(bootstrap_results)

        # Store full results
        all_results[k] = {
            'bootstrap_results': bootstrap_results,
            'posteriors': posteriors
        }

        # Extract assignments for this k
        df_k = pd.DataFrame({
            'embryo_id': posteriors['embryo_ids'],
            f'cluster_k{k}': posteriors['modal_cluster'],
            f'max_p_k{k}': posteriors['max_p'],
            f'entropy_k{k}': posteriors['entropy']
        })
        assignment_dfs.append(df_k)

    # Merge all k values
    df_assignments = assignment_dfs[0]
    for df_k in assignment_dfs[1:]:
        df_assignments = df_assignments.merge(df_k, on='embryo_id')

    return df_assignments, all_results


def compute_cluster_characteristics(
    df_interpolated,
    cluster_assignments,
    k,
    metric_col='metric_value',
    time_col='hpf'
):
    """Compute per-cluster statistics.

    Parameters
    ----------
    df_interpolated : DataFrame
        Interpolated trajectory data with columns [embryo_id, hpf, metric_value]
    cluster_assignments : DataFrame
        Output from get_cluster_assignments()
    k : int
        Which k value to analyze
    metric_col : str
        Name of metric column
    time_col : str
        Name of time column

    Returns
    -------
    df_characteristics : DataFrame
        Columns: cluster_id, n_embryos, mean_value, std_value, embryo_ids
    """
    cluster_col = f'cluster_k{k}'

    if cluster_col not in cluster_assignments.columns:
        raise ValueError(f"cluster_k{k} not found in assignments")

    # Merge cluster assignments with trajectory data
    df_merged = df_interpolated.merge(
        cluster_assignments[['embryo_id', cluster_col]],
        on='embryo_id'
    )

    rows = []

    for cluster_id in sorted(df_merged[cluster_col].unique()):
        df_cluster = df_merged[df_merged[cluster_col] == cluster_id]

        # Get unique embryos
        cluster_embryos = df_cluster['embryo_id'].unique()

        # Compute cluster mean trajectory (binned mean)
        cluster_mean_traj = df_cluster.groupby(time_col)[metric_col].mean()

        # Overall mean across all timepoints
        mean_val = cluster_mean_traj.mean()
        std_val = cluster_mean_traj.std()

        rows.append({
            'cluster_id': cluster_id,
            'n_embryos': len(cluster_embryos),
            'mean_value': mean_val,
            'std_value': std_val,
            'embryo_ids': ';'.join(cluster_embryos)
        })

    return pd.DataFrame(rows)


def identify_wt_like_clusters(
    cluster_characteristics,
    threshold=0.05
):
    """Identify WT-like clusters based on mean curvature threshold.

    Parameters
    ----------
    cluster_characteristics : DataFrame
        Output from compute_cluster_characteristics()
    threshold : float
        Clusters with mean_value < threshold are WT-like

    Returns
    -------
    df_classified : DataFrame
        Input dataframe with added 'is_wt_like' boolean column
    """
    df_classified = cluster_characteristics.copy()
    df_classified['is_wt_like'] = df_classified['mean_value'] < threshold
    return df_classified
```

**Update `__init__.py`** to export these:
```python
# Add to imports
from .cluster_assignment_utils import (
    get_cluster_assignments,
    compute_cluster_characteristics,
    identify_wt_like_clusters
)

# Add to __all__
```

**Pros**:
- Clean separation of high-level vs low-level functions
- Reusable across multiple analyses
- Includes helpers for cluster characterization
- Still uses existing core functions

**Cons**:
- One new file to maintain
- Slightly more overhead than Version 1

---

## Version 3: Full Pipeline Subpackage

**Location**: New subpackage `src/analyze/trajectory_analysis/clustering/`

**Structure**:
```
src/analyze/trajectory_analysis/clustering/
├── __init__.py
├── assignment.py       # get_cluster_assignments()
├── characteristics.py  # compute_cluster_characteristics(), identify_wt_like_clusters()
└── pipeline.py         # run_full_clustering_pipeline() - end-to-end
```

**clustering/__init__.py**:
```python
from .assignment import get_cluster_assignments
from .characteristics import (
    compute_cluster_characteristics,
    identify_wt_like_clusters
)
from .pipeline import run_full_clustering_pipeline

__all__ = [
    'get_cluster_assignments',
    'compute_cluster_characteristics',
    'identify_wt_like_clusters',
    'run_full_clustering_pipeline'
]
```

**clustering/assignment.py**: Same as Version 2

**clustering/characteristics.py**: Same as Version 2

**clustering/pipeline.py**:
```python
"""End-to-end clustering pipeline for per-group analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from ..dtw_distance import compute_dtw_distance_matrix
from .assignment import get_cluster_assignments
from .characteristics import compute_cluster_characteristics, identify_wt_like_clusters


def run_full_clustering_pipeline(
    df_interpolated,
    k_values=[3, 4, 5, 6],
    n_bootstrap=100,
    wt_threshold=0.05,
    output_dir=None
):
    """Run complete clustering pipeline from trajectories to cluster assignments.

    Parameters
    ----------
    df_interpolated : DataFrame
        Interpolated trajectory data with columns [embryo_id, hpf, metric_value]
    k_values : list of int
        K values to test
    n_bootstrap : int
        Number of bootstrap iterations
    wt_threshold : float
        Threshold for WT-like classification
    output_dir : Path or None
        If provided, save results to this directory

    Returns
    -------
    results : dict
        - 'distance_matrix': np.ndarray
        - 'cluster_assignments': DataFrame
        - 'all_results': dict (keyed by k)
        - 'cluster_characteristics': dict (keyed by k)
    """
    # Extract trajectories
    embryo_ids = df_interpolated['embryo_id'].unique()
    trajectories = []

    for eid in embryo_ids:
        traj = df_interpolated[df_interpolated['embryo_id'] == eid].sort_values('hpf')
        trajectories.append(traj['metric_value'].values)

    # Compute DTW distances
    D = compute_dtw_distance_matrix(trajectories, window=3)

    # Run clustering
    df_assignments, all_results = get_cluster_assignments(
        D, list(embryo_ids),
        k_values=k_values,
        n_bootstrap=n_bootstrap
    )

    # Compute characteristics for each k
    cluster_chars = {}
    for k in k_values:
        chars = compute_cluster_characteristics(df_interpolated, df_assignments, k)
        chars = identify_wt_like_clusters(chars, threshold=wt_threshold)
        cluster_chars[k] = chars

    results = {
        'distance_matrix': D,
        'cluster_assignments': df_assignments,
        'all_results': all_results,
        'cluster_characteristics': cluster_chars
    }

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / 'distance_matrix.npy', D)
        df_assignments.to_csv(output_dir / 'cluster_assignments.csv', index=False)

        for k in k_values:
            cluster_chars[k].to_csv(
                output_dir / f'cluster_characteristics_k{k}.csv',
                index=False
            )

    return results
```

**Pros**:
- Most organized and modular
- End-to-end pipeline function
- Easy to extend with new features
- Clear separation of concerns

**Cons**:
- Most new code
- Might be over-engineering for current need
- More files to maintain

---

## Recommendation

**Version 2** is the sweet spot:
- Adds useful high-level functions without over-engineering
- Keeps code organized in existing structure
- Easy to use in analysis scripts
- Minimal maintenance burden

The key function `get_cluster_assignments()` is present in all versions, so we satisfy the requirement.
