"""
Two-Stage Outlier Filtering for Trajectory Analysis

Provides robust outlier detection for distance matrices using k-NN and
within-cluster IQR methods, designed to protect rare phenotypes while
removing true outliers.

Stage 1: k-NN IQR filtering (before clustering)
    - Uses k-nearest neighbor distances instead of global mean
    - Protects small clusters of mutants from being flagged as outliers
    - Conservative threshold: Q3 + 4.0×IQR

Stage 2: Within-cluster IQR + posterior filtering (after clustering)
    - Removes embryos that are outliers within their assigned cluster
    - Also removes embryos with low posterior confidence
    - Combined filter: EITHER condition triggers removal

Functions
=========
- identify_embryo_outliers_iqr : Stage 1 k-NN IQR filtering
- filter_data_and_ids : Safe filtering maintaining index alignment
- identify_cluster_outliers_combined : Stage 2 cluster + posterior filtering

Created: 2025-12-22
Purpose: Support consensus clustering pipeline with robust outlier filtering
"""

import numpy as np
from typing import Dict, List, Any, Tuple


def identify_embryo_outliers_iqr(
    D: np.ndarray,
    embryo_ids: List[str],
    *,
    iqr_multiplier: float = 4.0,
    k_neighbors: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Stage 1: Identify outliers using k-NN distances with IQR threshold.

    CRITICAL: Uses k-Nearest Neighbors (not global mean) to protect rare phenotypes.
    A small cluster of mutants won't be flagged as outliers because they have low
    k-NN distances to each other.

    An embryo is flagged as outlier if:
        knn_mean_distance > Q3 + iqr_multiplier × IQR

    This identifies embryos that have no nearby neighbors - true outliers that
    would create singleton clusters or inflate the optimal k in clustering.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n_embryos × n_embryos), symmetric with zero diagonal
    embryo_ids : List[str]
        Embryo identifiers (same order as D rows/columns)
    iqr_multiplier : float, default=4.0
        IQR multiplier for outlier threshold (4.0 is conservative)
        Higher values = fewer outliers removed
    k_neighbors : int, default=5
        Number of nearest neighbors to consider
        Automatically capped at n-1 if dataset is small

    verbose : bool, default=True
        Print filtering diagnostics

    Returns
    -------
    results : Dict[str, Any]
        - 'outlier_indices': np.ndarray of outlier indices
        - 'outlier_ids': List[str] of outlier embryo IDs
        - 'kept_indices': np.ndarray of kept indices
        - 'kept_ids': List[str] of kept embryo IDs
        - 'knn_distances': np.ndarray of k-NN mean distances per embryo
        - 'threshold': float, the computed threshold
        - 'q1': float, first quartile of k-NN distances
        - 'q3': float, third quartile of k-NN distances
        - 'iqr': float, interquartile range

    Examples
    --------
    >>> # Detect outliers before clustering
    >>> results = identify_embryo_outliers_iqr(D, embryo_ids, iqr_multiplier=4.0)
    >>> print(f"Removed {len(results['outlier_ids'])} outliers")
    >>>
    >>> # Filter distance matrix and IDs
    >>> D_filtered, ids_filtered = filter_data_and_ids(
    ...     D, embryo_ids, results['kept_indices']
    ... )

    Notes
    -----
    - k-NN approach protects rare phenotypes: a small mutant cluster has low
      k-NN distances because mutants are near each other
    - Global mean would incorrectly flag mutants as outliers
    - Default k=5 neighbors balances robustness vs sensitivity
    - Default multiplier=4.0 is conservative (matches existing outlier detection)
    """
    n = len(D)
    k = min(k_neighbors, n - 1)  # Cap k at n-1 for small datasets

    if k < 1:
        raise ValueError(f"Need at least 2 embryos for k-NN filtering, got {n}")

    # Compute k-NN mean distance per embryo
    # For each embryo, find k nearest neighbors (excluding self)
    sorted_D = np.sort(D, axis=1)

    # sorted_D[:, 0] is always 0 (distance to self)
    # sorted_D[:, 1:k+1] are the k nearest neighbors
    knn_distances = sorted_D[:, 1:k+1].mean(axis=1)

    # IQR threshold
    q1 = np.percentile(knn_distances, 25)
    q3 = np.percentile(knn_distances, 75)
    iqr = q3 - q1
    threshold = q3 + iqr_multiplier * iqr

    # Identify outliers
    outlier_mask = knn_distances > threshold
    outlier_indices = np.where(outlier_mask)[0]
    kept_indices = np.where(~outlier_mask)[0]

    outlier_ids = [embryo_ids[i] for i in outlier_indices]
    kept_ids = [embryo_ids[i] for i in kept_indices]

    if verbose:
        print(f"\nStage 1: k-NN IQR Filtering")
        print(f"  k-NN distances (k={k}):")
        print(f"    Q1 = {q1:.3f}, Q3 = {q3:.3f}, IQR = {iqr:.3f}")
        print(f"    Threshold: {threshold:.3f} (Q3 + {iqr_multiplier}×IQR)")
        print(f"  Outliers removed: {len(outlier_ids)} / {n}")
        if len(outlier_ids) > 0 and len(outlier_ids) <= 10:
            print(f"  Outlier IDs: {outlier_ids}")
        elif len(outlier_ids) > 10:
            print(f"  Outlier IDs (first 10): {outlier_ids[:10]}")

    return {
        'outlier_indices': outlier_indices,
        'outlier_ids': outlier_ids,
        'kept_indices': kept_indices,
        'kept_ids': kept_ids,
        'knn_distances': knn_distances,
        'threshold': threshold,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }


def filter_data_and_ids(
    D: np.ndarray,
    embryo_ids: List[str],
    indices_to_keep: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    CRITICAL: Safely filter distance matrix AND embryo IDs together.

    This is the "single source of truth" for all filtering operations.
    Never manually slice D and embryo_ids separately - always use this function
    to prevent index drift bugs.

    Index drift is a common bug where:
    - You filter D to remove some rows/columns
    - You forget to update embryo_ids in parallel
    - Now D[i,j] refers to different embryos than embryo_ids[i] and embryo_ids[j]
    - Downstream analysis silently produces wrong results

    This function prevents that by:
    1. Filtering both D and embryo_ids atomically
    2. Validating that filtered lengths match
    3. Raising an error if index drift is detected

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    embryo_ids : List[str]
        Embryo identifiers (length n)
    indices_to_keep : np.ndarray
        Indices to keep (subset of 0..n-1)

    Returns
    -------
    D_filtered : np.ndarray
        Filtered distance matrix (m × m) where m = len(indices_to_keep)
    ids_filtered : List[str]
        Filtered embryo IDs (length m)

    Raises
    ------
    AssertionError
        If filtered matrix and IDs have different lengths (index drift detected)

    Examples
    --------
    >>> # After Stage 1 filtering
    >>> D_filtered, ids_filtered = filter_data_and_ids(
    ...     D, embryo_ids, stage1_results['kept_indices']
    ... )
    >>>
    >>> # After Stage 2 filtering
    >>> D_final, ids_final = filter_data_and_ids(
    ...     D_filtered, ids_filtered, stage2_results['kept_indices']
    ... )

    Notes
    -----
    - Uses np.ix_ for correct 2D subsetting of distance matrix
    - Validates index alignment with assertion
    - This is the ONLY safe way to filter data in the pipeline
    """
    # Filter distance matrix (2D subsetting)
    D_filtered = D[np.ix_(indices_to_keep, indices_to_keep)]

    # Filter embryo IDs (1D subsetting)
    ids_filtered = [embryo_ids[i] for i in indices_to_keep]

    # CRITICAL: Validate index alignment
    assert len(ids_filtered) == len(D_filtered), \
        f"Index drift detected! IDs ({len(ids_filtered)}) != Matrix ({len(D_filtered)})"

    return D_filtered, ids_filtered


def identify_cluster_outliers_combined(
    D: np.ndarray,
    cluster_labels: np.ndarray,
    posterior_results: Dict[str, Any],
    embryo_ids: List[str],
    *,
    iqr_multiplier: float = 4.0,
    posterior_threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Stage 2: Remove embryos if EITHER condition is met:
    1. Within-cluster mean distance > Q3 + iqr_multiplier × IQR (per cluster)
    2. Posterior max_p < posterior_threshold (low confidence)

    This removes:
    - Embryos that are outliers within their assigned cluster (condition 1)
    - Embryos with ambiguous cluster membership (condition 2)

    The IQR filtering is done per-cluster to handle clusters of different sizes
    and densities. An embryo that's an outlier in a tight cluster might be normal
    in a loose cluster.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n), already filtered from Stage 1
    cluster_labels : np.ndarray
        Cluster assignments (n,), values in {0, 1, ..., k-1}
        Typically from posterior_results['modal_cluster']
    posterior_results : Dict[str, Any]
        Output from analyze_bootstrap_results()
        Must contain 'max_p' key with posterior probabilities
    embryo_ids : List[str]
        Embryo identifiers (length n)
    iqr_multiplier : float, default=4.0
        IQR multiplier for within-cluster outlier threshold
    posterior_threshold : float, default=0.5
        Minimum max_p to keep (embryos below this are removed)
    verbose : bool, default=True
        Print filtering diagnostics

    Returns
    -------
    results : Dict[str, Any]
        - 'outlier_indices': np.ndarray of outlier indices
        - 'outlier_ids': List[str] of outlier embryo IDs
        - 'kept_indices': np.ndarray of kept indices
        - 'kept_ids': List[str] of kept embryo IDs
        - 'outlier_reason': Dict[str, str] mapping embryo_id to reason
          ('iqr', 'posterior', or 'both')
        - 'within_cluster_mean_distances': np.ndarray of mean distances

    Examples
    --------
    >>> # After initial clustering and posterior analysis
    >>> stage2_results = identify_cluster_outliers_combined(
    ...     D_filtered,
    ...     posteriors['modal_cluster'],
    ...     posteriors,
    ...     embryo_ids_filtered,
    ...     iqr_multiplier=4.0,
    ...     posterior_threshold=0.5
    ... )
    >>>
    >>> # Filter for final analysis
    >>> D_final, ids_final = filter_data_and_ids(
    ...     D_filtered, embryo_ids_filtered, stage2_results['kept_indices']
    ... )

    Notes
    -----
    - IQR filtering is per-cluster (handles different cluster densities)
    - Posterior filtering uses max_p from bootstrap consensus
    - EITHER condition triggers removal (not both required)
    - Small clusters (≤3 members) skip IQR filtering
    """
    n = len(D)

    # Compute within-cluster mean distances per embryo
    within_cluster_mean = np.zeros(n)
    for i in range(n):
        cluster_i = cluster_labels[i]
        cluster_members = np.where(cluster_labels == cluster_i)[0]
        cluster_members = cluster_members[cluster_members != i]  # Exclude self

        if len(cluster_members) > 0:
            within_cluster_mean[i] = D[i, cluster_members].mean()
        else:
            within_cluster_mean[i] = 0.0  # Singleton cluster

    # IQR threshold per cluster
    outlier_iqr = set()
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_dists = within_cluster_mean[cluster_mask]

        # Only apply IQR filtering if cluster has >3 members
        if len(cluster_dists) > 3:
            q1 = np.percentile(cluster_dists, 25)
            q3 = np.percentile(cluster_dists, 75)
            iqr = q3 - q1
            threshold = q3 + iqr_multiplier * iqr

            cluster_indices = np.where(cluster_mask)[0]
            for idx in cluster_indices:
                if within_cluster_mean[idx] > threshold:
                    outlier_iqr.add(idx)

    # Posterior threshold
    max_p_array = posterior_results['max_p']
    outlier_posterior = set(np.where(max_p_array < posterior_threshold)[0])

    # Combined: EITHER condition
    outlier_indices = np.array(sorted(outlier_iqr | outlier_posterior))
    kept_indices = np.array([i for i in range(n) if i not in outlier_indices])

    outlier_ids = [embryo_ids[i] for i in outlier_indices]
    kept_ids = [embryo_ids[i] for i in kept_indices]

    # Track reason for each outlier
    outlier_reason = {}
    for idx in outlier_indices:
        in_iqr = idx in outlier_iqr
        in_post = idx in outlier_posterior
        if in_iqr and in_post:
            outlier_reason[embryo_ids[idx]] = 'both'
        elif in_iqr:
            outlier_reason[embryo_ids[idx]] = 'iqr'
        else:
            outlier_reason[embryo_ids[idx]] = 'posterior'

    if verbose:
        print(f"\nStage 2: Combined Filtering (IQR + Posterior)")
        print(f"  IQR outliers (within-cluster): {len(outlier_iqr)}")
        print(f"  Posterior outliers (max_p < {posterior_threshold}): {len(outlier_posterior)}")
        print(f"  Total removed: {len(outlier_ids)} / {n}")
        if len(outlier_ids) > 0 and len(outlier_ids) <= 10:
            reasons_summary = {}
            for reason in outlier_reason.values():
                reasons_summary[reason] = reasons_summary.get(reason, 0) + 1
            print(f"  Removal reasons: {reasons_summary}")

    return {
        'outlier_indices': outlier_indices,
        'outlier_ids': outlier_ids,
        'kept_indices': kept_indices,
        'kept_ids': kept_ids,
        'outlier_reason': outlier_reason,
        'within_cluster_mean_distances': within_cluster_mean
    }
