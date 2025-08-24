# Task 02: Time-Weighted Spectral Clustering

## Overview
Create spectral clustering that respects both morphological similarity and temporal proximity, providing meaningful neighborhoods for minibatch UOT sampling.

## Mathematical Foundation

### Affinity Matrix Construction
For points i and j with embeddings z_i, z_j and times t_i, t_j:

```
W_ij = exp(-||z_i - z_j||² / σ_z²) × exp(-(t_i - t_j)² / σ_t²)
```

This creates clusters that are:
- **Morphologically similar** (first term)
- **Temporally close** (second term)

The time weighting ensures we don't compare early embryos with late ones, respecting developmental progression.

## Implementation Guide

### Step 1: Core Clustering Function

```python
# File: src/flux/utils/spectral_time_clustering.py

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Union, Tuple

def time_weighted_spectral_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    time_col: str = "predicted_stage_hpf",
    experiment_ids: Optional[List[str]] = None,
    embryo_ids: Optional[List[str]] = None,
    n_clusters: int = 10,
    sigma_space: Optional[float] = None,
    sigma_time: Optional[float] = None,
    time_weight: bool = True,
    affinity: str = "precomputed",
    return_affinity: bool = False,
    normalize_features: bool = True,
    knn_sparsify: Optional[int] = None,
    verbose: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform spectral clustering with time-weighted affinity.
    
    This function creates clusters that respect both morphological similarity
    and temporal proximity, essential for meaningful developmental analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing embeddings and time information
    feature_cols : List[str], optional
        Columns to use for morphology distance. If None, automatically detects:
        1. First tries biological components (bc_*)
        2. Falls back to VAE latents (z_*)
    time_col : str
        Column containing time information (default: predicted_stage_hpf)
    experiment_ids : List[str], optional
        Filter to specific experiments
    embryo_ids : List[str], optional
        Filter to specific embryos
    n_clusters : int
        Number of clusters to create
    sigma_space : float, optional
        Bandwidth for morphology similarity (auto-computed if None)
    sigma_time : float, optional
        Bandwidth for time similarity (auto-computed if None)
    time_weight : bool
        If True, multiply spatial affinity by time affinity
    affinity : str
        For sklearn compatibility (always "precomputed")
    return_affinity : bool
        If True, also return the affinity matrix
    normalize_features : bool
        If True, standardize features before computing distances
    knn_sparsify : int, optional
        If provided, keep only k nearest neighbors (for efficiency)
    verbose : bool
        Print progress information
    
    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each row
    affinity : np.ndarray (optional)
        The computed affinity matrix if return_affinity=True
    
    Examples
    --------
    >>> df = pd.read_csv('embryo_data.csv')
    >>> labels = time_weighted_spectral_clustering(df, n_clusters=20)
    >>> 
    >>> # With custom parameters and affinity matrix
    >>> labels, W = time_weighted_spectral_clustering(
    ...     df, 
    ...     n_clusters=15,
    ...     sigma_space=2.0,
    ...     sigma_time=1.5,
    ...     return_affinity=True
    ... )
    """
    # Step 1: Filter data if needed
    df_filtered = df.copy()
    
    if experiment_ids is not None:
        if 'experiment_id' in df.columns:
            df_filtered = df_filtered[df_filtered['experiment_id'].isin(experiment_ids)]
        else:
            raise ValueError("experiment_ids specified but no experiment_id column found")
    
    if embryo_ids is not None:
        if 'embryo_id' in df.columns:
            df_filtered = df_filtered[df_filtered['embryo_id'].isin(embryo_ids)]
        else:
            raise ValueError("embryo_ids specified but no embryo_id column found")
    
    if verbose:
        print(f"Working with {len(df_filtered)} points after filtering")
    
    # Step 2: Auto-detect feature columns if not specified
    if feature_cols is None:
        # Try biological components first
        bc_cols = [col for col in df_filtered.columns if col.startswith('bc_')]
        if len(bc_cols) > 0:
            feature_cols = bc_cols
            if verbose:
                print(f"Using {len(bc_cols)} biological component columns")
        else:
            # Fall back to VAE latents
            z_cols = [col for col in df_filtered.columns if col.startswith('z_')]
            if len(z_cols) > 0:
                feature_cols = z_cols
                if verbose:
                    print(f"Using {len(z_cols)} VAE latent columns")
            else:
                raise ValueError("No feature columns found (looked for bc_* and z_*)")
    
    # Step 3: Extract features and time
    features = df_filtered[feature_cols].values
    
    if time_col not in df_filtered.columns:
        raise ValueError(f"Time column '{time_col}' not found")
    times = df_filtered[time_col].values.reshape(-1, 1)
    
    # Step 4: Normalize features if requested
    if normalize_features:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Step 5: Build affinity matrix
    W = build_time_weighted_affinity(
        features, times,
        sigma_space=sigma_space,
        sigma_time=sigma_time,
        time_weight=time_weight,
        knn_sparsify=knn_sparsify,
        verbose=verbose
    )
    
    # Step 6: Run spectral clustering
    if verbose:
        print(f"Running spectral clustering with {n_clusters} clusters...")
    
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1
    )
    labels = clustering.fit_predict(W)
    
    if verbose:
        print(f"Clustering complete. Cluster sizes: {np.bincount(labels)}")
    
    if return_affinity:
        return labels, W
    return labels


def build_time_weighted_affinity(
    features: np.ndarray,
    times: np.ndarray,
    sigma_space: Optional[float] = None,
    sigma_time: Optional[float] = None,
    time_weight: bool = True,
    knn_sparsify: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Build affinity matrix with morphology and time components.
    
    Parameters
    ----------
    features : np.ndarray (n, d)
        Morphological features
    times : np.ndarray (n, 1)
        Time values
    sigma_space : float, optional
        Spatial bandwidth (auto-computed using median heuristic if None)
    sigma_time : float, optional
        Temporal bandwidth (auto-computed if None)
    time_weight : bool
        If True, multiply spatial by temporal affinity
    knn_sparsify : int, optional
        Keep only k nearest neighbors for efficiency
    verbose : bool
        Print progress
        
    Returns
    -------
    W : np.ndarray (n, n)
        Affinity matrix
    """
    n = len(features)
    
    # Auto-compute bandwidths if not provided
    if sigma_space is None:
        # Median heuristic: use median of k-nearest neighbor distances
        k_for_median = min(20, n // 10)
        dists = cdist(features, features, metric='euclidean')
        knn_dists = np.sort(dists, axis=1)[:, 1:k_for_median+1]  # exclude self
        sigma_space = np.median(knn_dists)
        if verbose:
            print(f"Auto-computed σ_space = {sigma_space:.3f}")
    
    if sigma_time is None and time_weight:
        # Use 10% of time range
        time_range = times.max() - times.min()
        sigma_time = 0.1 * time_range
        if verbose:
            print(f"Auto-computed σ_time = {sigma_time:.3f}")
    
    # Compute spatial affinity
    if verbose:
        print("Computing spatial affinity...")
    spatial_dists_sq = cdist(features, features, metric='sqeuclidean')
    W_spatial = np.exp(-spatial_dists_sq / (sigma_space ** 2))
    
    # Compute temporal affinity if requested
    if time_weight:
        if verbose:
            print("Computing temporal affinity...")
        time_dists_sq = cdist(times, times, metric='sqeuclidean')
        W_temporal = np.exp(-time_dists_sq / (sigma_time ** 2))
        W = W_spatial * W_temporal
    else:
        W = W_spatial
    
    # Optional: sparsify using k-NN
    if knn_sparsify is not None:
        if verbose:
            print(f"Sparsifying to {knn_sparsify}-NN graph...")
        W = knn_sparsify_affinity(W, k=knn_sparsify)
    
    # Ensure symmetry (important for spectral clustering)
    W = (W + W.T) / 2
    
    return W


def knn_sparsify_affinity(W: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only k nearest neighbors in affinity matrix.
    
    Parameters
    ----------
    W : np.ndarray (n, n)
        Dense affinity matrix
    k : int
        Number of neighbors to keep
        
    Returns
    -------
    W_sparse : np.ndarray (n, n)
        Sparse affinity matrix
    """
    n = W.shape[0]
    W_sparse = np.zeros_like(W)
    
    for i in range(n):
        # Find k nearest neighbors (excluding self)
        row = W[i].copy()
        row[i] = 0  # exclude self
        knn_indices = np.argpartition(row, -k)[-k:]
        
        # Keep only k-NN connections
        W_sparse[i, knn_indices] = W[i, knn_indices]
    
    # Make symmetric (mutual k-NN)
    W_sparse = np.maximum(W_sparse, W_sparse.T)
    
    return W_sparse


### Step 2: Analysis Utilities

```python
def analyze_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    time_col: str = "predicted_stage_hpf"
) -> pd.DataFrame:
    """
    Analyze cluster properties for validation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original data
    labels : np.ndarray
        Cluster assignments
    time_col : str
        Time column name
        
    Returns
    -------
    pd.DataFrame
        Summary statistics per cluster
    """
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    summary = []
    for cluster_id in np.unique(labels):
        cluster_data = df_with_labels[df_with_labels['cluster'] == cluster_id]
        
        summary.append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'mean_time': cluster_data[time_col].mean(),
            'std_time': cluster_data[time_col].std(),
            'min_time': cluster_data[time_col].min(),
            'max_time': cluster_data[time_col].max(),
            'n_experiments': cluster_data['experiment_id'].nunique() if 'experiment_id' in cluster_data else 0,
            'n_embryos': cluster_data['embryo_id'].nunique() if 'embryo_id' in cluster_data else 0
        })
    
    return pd.DataFrame(summary).sort_values('mean_time')


def find_temporal_neighbors(
    labels: np.ndarray,
    times: np.ndarray,
    max_time_gap: float = 2.0
) -> dict:
    """
    Find which clusters are temporal neighbors (for minibatch sampling).
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster assignments
    times : np.ndarray
        Time values for each point
    max_time_gap : float
        Maximum time difference to consider clusters as neighbors
        
    Returns
    -------
    dict
        Mapping from cluster_id to list of neighboring cluster_ids
    """
    unique_clusters = np.unique(labels)
    cluster_times = {}
    
    # Compute mean time per cluster
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_times[cluster_id] = times[mask].mean()
    
    # Find neighbors
    neighbors = {c: [] for c in unique_clusters}
    
    for c1 in unique_clusters:
        for c2 in unique_clusters:
            if c1 != c2:
                time_diff = cluster_times[c2] - cluster_times[c1]
                if 0 < time_diff <= max_time_gap:
                    neighbors[c1].append(c2)
    
    return neighbors


def visualize_affinity_matrix(W: np.ndarray, labels: np.ndarray = None):
    """
    Visualize the affinity matrix (useful for debugging).
    
    Parameters
    ----------
    W : np.ndarray
        Affinity matrix
    labels : np.ndarray, optional
        Cluster labels for reordering
    """
    import matplotlib.pyplot as plt
    
    # Reorder by clusters if labels provided
    if labels is not None:
        idx = np.argsort(labels)
        W_reordered = W[idx][:, idx]
        labels_reordered = labels[idx]
    else:
        W_reordered = W
        labels_reordered = None
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(W_reordered, cmap='hot', interpolation='nearest')
    
    if labels_reordered is not None:
        # Add cluster boundaries
        boundaries = np.where(np.diff(labels_reordered))[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color='blue', linewidth=0.5)
            ax.axvline(b, color='blue', linewidth=0.5)
    
    ax.set_title("Time-Weighted Affinity Matrix")
    ax.set_xlabel("Point index")
    ax.set_ylabel("Point index")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


### Step 3: Integration with UOT Minibatching

```python
class ClusteredMinibatchSampler:
    """
    Sampler that uses spectral clusters for UOT minibatch generation.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        times: np.ndarray,
        embryo_ids: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 256,
        time_gap_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False
    ):
        """
        Initialize the sampler with clustered data.
        
        Parameters
        ----------
        embeddings : np.ndarray (n, d)
            Feature vectors
        times : np.ndarray (n,)
            Time values
        embryo_ids : np.ndarray (n,)
            Embryo identifiers for tracking
        labels : np.ndarray (n,)
            Cluster assignments from spectral clustering
        batch_size : int
            Maximum points per minibatch
        time_gap_range : Tuple[float, float]
            Range of time gaps to sample
        """
        self.embeddings = embeddings
        self.times = times
        self.embryo_ids = embryo_ids
        self.labels = labels
        self.batch_size = batch_size
        self.time_gap_range = time_gap_range
        self.verbose = verbose
        
        # Precompute cluster statistics
        self.cluster_ids = np.unique(labels)
        self.cluster_mean_times = {}
        self.cluster_indices = {}
        
        for c in self.cluster_ids:
            mask = labels == c
            self.cluster_indices[c] = np.where(mask)[0]
            self.cluster_mean_times[c] = times[mask].mean()
        
        # Find temporal neighbors
        self.neighbors = find_temporal_neighbors(
            labels, times, max_time_gap=time_gap_range[1]
        )
    
    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Sample a source-target minibatch pair.
        
        Returns
        -------
        X : np.ndarray (m, d)
            Source embeddings
        Y : np.ndarray (n, d)
            Target embeddings
        self_indices : np.ndarray (m,)
            Tracking indices (-1 for no track)
        dt : float
            Time gap
        """
        # Sample time gap
        dt = np.random.uniform(*self.time_gap_range)
        
        # Sample source cluster
        source_cluster = np.random.choice(self.cluster_ids)
        
        # Find valid target clusters
        valid_targets = []
        for target_cluster in self.neighbors.get(source_cluster, []):
            time_diff = (self.cluster_mean_times[target_cluster] - 
                        self.cluster_mean_times[source_cluster])
            if abs(time_diff - dt) < 0.5:  # within tolerance
                valid_targets.append(target_cluster)
        
        if len(valid_targets) == 0:
            # No valid targets, sample randomly from later clusters
            later_clusters = [c for c in self.cluster_ids 
                            if self.cluster_mean_times[c] > self.cluster_mean_times[source_cluster]]
            if len(later_clusters) == 0:
                # Edge case: source is latest cluster
                return np.empty((0, self.embeddings.shape[1])), \
                       np.empty((0, self.embeddings.shape[1])), \
                       np.array([]), dt
            target_cluster = np.random.choice(later_clusters)
        else:
            target_cluster = np.random.choice(valid_targets)
        
        # Sample points from clusters
        source_indices = self.cluster_indices[source_cluster]
        target_indices = self.cluster_indices[target_cluster]
        
        # Subsample if needed
        if len(source_indices) > self.batch_size:
            source_indices = np.random.choice(source_indices, self.batch_size, replace=False)
        if len(target_indices) > self.batch_size:
            target_indices = np.random.choice(target_indices, self.batch_size, replace=False)
        
        X = self.embeddings[source_indices]
        Y = self.embeddings[target_indices]
        
        # Build tracking indices
        self_indices = np.full(len(source_indices), -1, dtype=int)
        source_embryos = self.embryo_ids[source_indices]
        target_embryos = self.embryo_ids[target_indices]
        
        for i, emb_id in enumerate(source_embryos):
            matches = np.where(target_embryos == emb_id)[0]
            if len(matches) > 0:
                self_indices[i] = matches[0]
        
        if self.verbose:
            n_tracked = (self_indices >= 0).sum()
            print(f"Sampled batch: {len(X)} → {len(Y)} points, "
                  f"{n_tracked}/{len(X)} tracked, dt={dt:.2f}")
        
        return X, Y, self_indices, dt


## Hyperparameter Selection Guide

### Key Parameters to Tune

1. **Number of clusters (n_clusters)**
   - Rule of thumb: sqrt(n_samples) / 10
   - For 130k points: try 50-200 clusters
   - More clusters = finer temporal resolution

2. **Spatial bandwidth (σ_space)**
   - Controls morphological similarity weight
   - Auto-computed using median k-NN distance
   - Decrease for tighter morphological groups

3. **Temporal bandwidth (σ_time)**
   - Controls temporal proximity weight  
   - Auto-computed as 10% of time range
   - Decrease to prevent mixing of developmental stages

4. **k-NN sparsification**
   - For 130k points, consider k=100-500
   - Reduces memory from O(n²) to O(nk)
   - Too small k can disconnect the graph

### Parameter Selection Strategy

```python
def tune_clustering_parameters(df_sample, param_grid):
    """
    Grid search for optimal clustering parameters.
    
    Evaluates based on:
    - Temporal coherence (low time variance within clusters)
    - Spatial coherence (low feature variance within clusters)
    - Cluster balance (avoid tiny/huge clusters)
    """
    results = []
    
    for params in param_grid:
        labels = time_weighted_spectral_clustering(df_sample, **params)
        
        # Compute quality metrics
        temporal_coherence = compute_temporal_coherence(df_sample, labels)
        spatial_coherence = compute_spatial_coherence(df_sample, labels)
        balance_score = compute_balance_score(labels)
        
        results.append({
            'params': params,
            'temporal_coherence': temporal_coherence,
            'spatial_coherence': spatial_coherence,
            'balance_score': balance_score,
            'combined_score': temporal_coherence * spatial_coherence * balance_score
        })
    
    return pd.DataFrame(results).sort_values('combined_score', ascending=False)
```

## Common Issues and Solutions

### Issue 1: Clusters span too much time
**Symptom**: Clusters contain both early and late embryos  
**Solution**: Decrease σ_time or increase n_clusters

### Issue 2: Disconnected affinity graph
**Symptom**: Spectral clustering fails or gives poor results  
**Solution**: Increase σ_space or k for k-NN

### Issue 3: Memory overflow with 130k points
**Symptom**: OOM when building affinity matrix  
**Solution**: Use k-NN sparsification or process in batches

### Issue 4: Imbalanced clusters
**Symptom**: Some clusters have 1 point, others have thousands  
**Solution**: Adjust σ_space/σ_time or use balanced spectral clustering

## Testing Suite

```python
# File: tests/test_spectral_clustering.py

import numpy as np
import pandas as pd
from src.flux.utils.spectral_time_clustering import (
    time_weighted_spectral_clustering,
    build_time_weighted_affinity,
    analyze_clusters
)

def test_synthetic_progression():
    """Test on synthetic data with clear temporal progression."""
    # Create synthetic embryo data
    n_timepoints = 10
    n_per_time = 100
    n_features = 5
    
    data = []
    for t in range(n_timepoints):
        # Features drift over time
        features = np.random.randn(n_per_time, n_features) + t * 0.5
        times = np.full(n_per_time, t)
        
        for i in range(n_per_time):
            row = {'predicted_stage_hpf': times[i]}
            for j in range(n_features):
                row[f'z_{j}'] = features[i, j]
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Cluster with time weighting
    labels = time_weighted_spectral_clustering(
        df, n_clusters=n_timepoints, time_weight=True
    )
    
    # Check temporal coherence
    summary = analyze_clusters(df, labels)
    assert summary['std_time'].max() < 1.0, "Clusters not temporally coherent"
    
def test_no_time_weighting():
    """Test that disabling time weight gives different results."""
    df = create_test_data()
    
    labels_with_time = time_weighted_spectral_clustering(
        df, time_weight=True
    )
    labels_without_time = time_weighted_spectral_clustering(
        df, time_weight=False
    )
    
    # Results should differ
    assert not np.array_equal(labels_with_time, labels_without_time)

def test_feature_auto_detection():
    """Test automatic detection of bc_* vs z_* columns."""
    # Create data with both types
    df = pd.DataFrame({
        'bc_0': np.random.randn(100),
        'bc_1': np.random.randn(100),
        'z_0': np.random.randn(100),
        'z_1': np.random.randn(100),
        'predicted_stage_hpf': np.random.uniform(0, 10, 100)
    })
    
    # Should use bc_* by default
    labels = time_weighted_spectral_clustering(df, verbose=True)
    # Check in verbose output that bc_* was used
    
def test_sparse_affinity():
    """Test k-NN sparsification."""
    df = create_test_data(n=1000)
    
    # Dense affinity
    labels_dense, W_dense = time_weighted_spectral_clustering(
        df, knn_sparsify=None, return_affinity=True
    )
    
    # Sparse affinity
    labels_sparse, W_sparse = time_weighted_spectral_clustering(
        df, knn_sparsify=50, return_affinity=True
    )
    
    # Sparse should have many zeros
    sparsity = (W_sparse == 0).sum() / W_sparse.size
    assert sparsity > 0.8, f"Matrix not sparse enough: {sparsity:.2f}"
    
    # But results should be similar
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_dense, labels_sparse)
    assert ari > 0.7, f"Sparse/dense results too different: ARI={ari:.2f}"

def test_minibatch_sampler():
    """Test the clustered minibatch sampler."""
    df = create_test_data(n=1000)
    
    # Cluster
    labels = time_weighted_spectral_clustering(df, n_clusters=10)
    
    # Create sampler
    sampler = ClusteredMinibatchSampler(
        embeddings=df[[f'z_{i}' for i in range(5)]].values,
        times=df['predicted_stage_hpf'].values,
        embryo_ids=df['embryo_id'].values if 'embryo_id' in df else np.arange(len(df)),
        labels=labels,
        batch_size=64
    )
    
    # Sample batches
    for _ in range(10):
        X, Y, self_indices, dt = sampler.sample_batch()
        
        # Basic sanity checks
        assert len(X) <= 64
        assert len(Y) <= 64
        assert len(self_indices) == len(X)
        assert dt > 0
```

## Usage Examples

### Basic Usage
```python
# Load data
df = pd.read_csv('embryo_data.csv')

# Run clustering with defaults
labels = time_weighted_spectral_clustering(df, n_clusters=100)

# Analyze results
summary = analyze_clusters(df, labels)
print(summary)
```

### Advanced Usage with UOT
```python
# Cluster with custom parameters
labels, W = time_weighted_spectral_clustering(
    df,
    n_clusters=150,
    sigma_space=2.0,
    sigma_time=1.0,
    knn_sparsify=200,
    return_affinity=True,
    verbose=True
)

# Create minibatch sampler
sampler = ClusteredMinibatchSampler(
    embeddings=df[feature_cols].values,
    times=df['predicted_stage_hpf'].values,
    embryo_ids=df['embryo_id'].values,
    labels=labels,
    batch_size=256
)

# Use in training loop
for epoch in range(num_epochs):
    X, Y, self_indices, dt = sampler.sample_batch()
    
    # Run UOT
    uot_result = compute_biased_uot(
        X, Y, dt, self_indices=self_indices,
        bias_mode='lower_bound'
    )
    
    # Train ODE with velocities
    # ...
```

## Performance Considerations

For 130k points:
- Dense affinity: ~67GB memory (float32)
- With k=200 sparse: ~200MB memory
- Computation time: ~5-10 minutes (dense), ~1-2 minutes (sparse)

Consider:
1. Use k-NN sparsification for large datasets
2. Process in chunks if needed
3. Cache affinity matrix if reusing
4. Use float32 instead of float64

## Next Steps

1. Run discovery phase to understand data schema
2. Test on small subset (1000 points)
3. Tune parameters on medium subset (10k points)  
4. Scale to full dataset with optimizations
5. Integrate with minibatch UOT (Task 01)