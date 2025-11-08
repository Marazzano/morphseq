"""
Bootstrap Clustering

Bootstrap resampling methods for consensus clustering with quality assessment.

Functions
---------
- run_bootstrap_hierarchical: Bootstrap hierarchical clustering with consensus labels
- compute_consensus_labels: Compute consensus cluster labels from bootstrap iterations
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from .config import N_BOOTSTRAP, BOOTSTRAP_FRAC, RANDOM_SEED


def run_bootstrap_hierarchical(
    D: np.ndarray,
    k: int,
    embryo_ids: List[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    random_state: int = RANDOM_SEED,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Bootstrap hierarchical clustering with label alignment.

    Performs repeated hierarchical clustering on random subsamples of the data,
    storing the resulting cluster labels for posterior probability computation.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    k : int
        Number of clusters
    embryo_ids : list of str
        Embryo identifiers (required for tracking). Should encode experiment/run info.
        Example: ['cep290_wt_run1_emb_01', 'cep290_wt_run1_emb_02', ...]
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    frac : float, default=0.8
        Fraction of samples per bootstrap
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress

    Returns
    -------
    bootstrap_results_dict : dict
        - 'embryo_ids': list of str (copy of input)
        - 'reference_labels': np.ndarray, consensus labels from full data
        - 'bootstrap_results': list of dicts
            - 'labels': np.ndarray (-1 for unsampled)
            - 'indices': np.ndarray of sampled indices
            - 'silhouette': float
        - 'n_clusters': int

    Examples
    --------
    >>> D = compute_dtw_distance_matrix(trajectories)
    >>> embryo_ids = ['emb_01', 'emb_02', 'emb_03', ...]
    >>> results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids, n_bootstrap=100)
    >>> reference_labels = results['reference_labels']
    >>> # Lookup embryo by ID
    >>> idx = results['embryo_ids'].index('emb_02')
    >>> label = results['reference_labels'][idx]
    """
    np.random.seed(random_state)
    n_samples = len(D)

    # Compute reference labels from full data
    clusterer = AgglomerativeClustering(
        n_clusters=k,
        linkage='average',
        metric='precomputed'
    )
    reference_labels = clusterer.fit_predict(D)

    # Bootstrap iterations
    bootstrap_results = []
    n_to_sample = max(int(np.ceil(frac * n_samples)), 1)

    if verbose:
        print(f"Running {n_bootstrap} bootstrap iterations...")
        print(f"  Sampling {n_to_sample}/{n_samples} samples per iteration")

    for iter_idx in range(n_bootstrap):
        if verbose and (iter_idx + 1) % 10 == 0:
            print(f"  Progress: {iter_idx + 1}/{n_bootstrap}")

        # Random sample
        sampled_indices = np.random.choice(n_samples, size=n_to_sample, replace=False)
        sampled_indices = np.sort(sampled_indices)

        # Create submatrix
        D_subset = D[np.ix_(sampled_indices, sampled_indices)]

        # Cluster subset
        try:
            clusterer_boot = AgglomerativeClustering(
                n_clusters=k,
                linkage='average',
                metric='precomputed'
            )
            labels_subset = clusterer_boot.fit_predict(D_subset)

            # Create full-size label array with -1 for unsampled
            labels_full = np.full(n_samples, -1, dtype=int)
            labels_full[sampled_indices] = labels_subset

            # Compute silhouette if possible
            try:
                silhouette = silhouette_score(D_subset, labels_subset, metric='precomputed')
            except:
                silhouette = np.nan

            bootstrap_results.append({
                'labels': labels_full,
                'indices': sampled_indices,
                'silhouette': silhouette
            })
        except Exception as e:
            if verbose:
                print(f"    Warning: Bootstrap iteration {iter_idx} failed: {e}")
            continue

    if verbose:
        print(f"\nCompleted {len(bootstrap_results)} successful bootstrap iterations")

    return {
        'embryo_ids': list(embryo_ids),
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results,
        'n_clusters': k,
        'distance_matrix': D,
        'n_samples': n_samples
    }


def compute_consensus_labels(
    bootstrap_results: Dict[str, Any],
    consensus_method: str = 'majority'
) -> np.ndarray:
    """
    Compute consensus labels from bootstrap results.

    Parameters
    ----------
    bootstrap_results : dict
        Output from run_bootstrap_hierarchical()
    consensus_method : str, default='majority'
        Method for consensus: 'majority', 'mode', or 'mean'

    Returns
    -------
    consensus_labels : np.ndarray
        Consensus cluster assignments
    """
    n_samples = bootstrap_results['n_samples']
    bootstrap_iter_results = bootstrap_results['bootstrap_results']

    # Count cluster assignments
    assignment_matrix = np.full((n_samples, bootstrap_results['n_clusters']), 0, dtype=int)

    for boot_result in bootstrap_iter_results:
        labels = boot_result['labels']
        for i in range(n_samples):
            if labels[i] >= 0:
                assignment_matrix[i, labels[i]] += 1

    # Compute consensus: most frequent cluster per sample
    consensus_labels = np.argmax(assignment_matrix, axis=1)

    return consensus_labels
