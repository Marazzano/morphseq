"""
Outlier Detection for Distance Matrices

Provides outlier detection functions that work on any distance matrix,
useful for identifying embryos that are consistently far from all others
(which often create singleton clusters in hierarchical clustering).

Functions
=========
- identify_outliers : Detect outlier embryos based on median distance
- remove_outliers_from_distance_matrix : Convenience wrapper for removal

Created: 2025-12-19
Migrated from: results/mcolon/20251218_MD-DTW-morphseq_analysis/md_dtw_prototype.py
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any


def identify_outliers(
    D: np.ndarray,
    embryo_ids: List[str],
    method: str = 'median_distance',
    threshold: Optional[float] = None,
    percentile: float = 95,
    verbose: bool = True,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Identify outlier embryos based on distance matrix.

    Outliers are embryos that are consistently far from all other embryos,
    which often create singleton clusters that inflate k in hierarchical clustering.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n_embryos, n_embryos)
    embryo_ids : List[str]
        List of embryo identifiers (same order as D rows)
    method : str, default='median_distance'
        Outlier detection method:
        - 'median_distance': Flag embryos with median distance > threshold
        - 'percentile': Flag embryos with median distance > percentile of all medians
        - 'iqr': Interquartile Range (Q3 + k×IQR, extreme outlier detection)
        - 'mad': Median Absolute Deviation (robust to outliers)
    threshold : float, optional
        Manual threshold for median distance (used with 'median_distance' method)
    percentile : float, default=95
        Percentile cutoff for 'percentile' method
    verbose : bool, default=True
        Print diagnostic information

    Returns
    -------
    outlier_ids : List[str]
        List of outlier embryo IDs
    inlier_ids : List[str]
        List of non-outlier embryo IDs
    info : Dict[str, Any]
        Dict with diagnostic information:
        - 'median_distances': Median distance for each embryo
        - 'threshold': Threshold used for outlier detection
        - 'outlier_indices': Indices of outliers in original array
        - 'inlier_indices': Indices of inliers in original array

    Examples
    --------
    >>> # Detect outliers using 95th percentile
    >>> outliers, inliers, info = identify_outliers(
    ...     D, embryo_ids, method='percentile', percentile=95
    ... )
    >>> print(f"Found {len(outliers)} outliers: {outliers}")

    >>> # Remove outliers and re-cluster
    >>> D_clean = D[np.ix_(info['inlier_indices'], info['inlier_indices'])]
    >>> embryo_ids_clean = inliers

    Notes
    -----
    - median_distance: Good when you know approximate scale of your data
    - percentile: Adaptive to your data distribution (recommended)
    - mad: Most robust to extreme outliers, but can be conservative
    """
    n = len(embryo_ids)

    if verbose:
        print(f"\nIdentifying outliers using '{method}' method...")
        print(f"  Total embryos: {n}")

    # Compute median distance for each embryo (to all others)
    # Exclude diagonal (distance to self = 0)
    median_distances = np.zeros(n)
    for i in range(n):
        # Get distances to all other embryos (exclude self)
        dists_to_others = np.concatenate([D[i, :i], D[i, i+1:]])
        median_distances[i] = np.median(dists_to_others)

    # Determine threshold based on method
    if method == 'median_distance':
        if threshold is None:
            raise ValueError("threshold must be provided for 'median_distance' method")
        thresh = threshold

    elif method == 'percentile':
        thresh = np.percentile(median_distances, percentile)
        if verbose:
            print(f"  {percentile}th percentile of median distances: {thresh:.3f}")

    elif method == 'iqr':
        # Interquartile Range (IQR) method: Q3 + k×IQR
        q1, q3 = np.percentile(median_distances, [25, 75])
        iqr = q3 - q1
        iqr_multiplier = threshold if threshold is not None else 4.0  # Default: 4.0× (extreme outliers)
        thresh = q3 + iqr_multiplier * iqr
        if verbose:
            print(f"  Q1 (25th percentile): {q1:.3f}")
            print(f"  Q3 (75th percentile): {q3:.3f}")
            print(f"  IQR: {iqr:.3f}")
            print(f"  IQR multiplier: {iqr_multiplier:.1f}×")
            print(f"  Threshold (Q3 + {iqr_multiplier:.1f}×IQR): {thresh:.3f}")

    elif method == 'mad':
        # Median Absolute Deviation (MAD)
        median_of_medians = np.median(median_distances)
        mad = np.median(np.abs(median_distances - median_of_medians))
        # Use 3 * MAD as threshold (robust outlier detection)
        thresh = median_of_medians + 3 * mad
        if verbose:
            print(f"  Median of median distances: {median_of_medians:.3f}")
            print(f"  MAD: {mad:.3f}")
            print(f"  Threshold (median + 3*MAD): {thresh:.3f}")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'median_distance', 'percentile', 'iqr', or 'mad'")

    # Identify outliers
    outlier_mask = median_distances > thresh
    outlier_indices = np.where(outlier_mask)[0]
    inlier_indices = np.where(~outlier_mask)[0]

    outlier_ids = [embryo_ids[i] for i in outlier_indices]
    inlier_ids = [embryo_ids[i] for i in inlier_indices]

    if verbose:
        print(f"  Threshold: {thresh:.3f}")
        print(f"  Outliers detected: {len(outlier_ids)}")
        print(f"  Inliers retained: {len(inlier_ids)}")

        if len(outlier_ids) > 0:
            print(f"\n  Outlier embryos:")
            for embryo_id, med_dist in zip(
                [embryo_ids[i] for i in outlier_indices],
                median_distances[outlier_indices]
            ):
                print(f"    {embryo_id}: median_dist = {med_dist:.3f}")

    # Package info
    info = {
        'median_distances': median_distances,
        'threshold': thresh,
        'outlier_indices': outlier_indices,
        'inlier_indices': inlier_indices,
        'method': method,
    }

    return outlier_ids, inlier_ids, info


def remove_outliers_from_distance_matrix(
    D: np.ndarray,
    embryo_ids: List[str],
    outlier_detection_method: str = 'percentile',
    outlier_threshold: Optional[float] = None,
    outlier_percentile: float = 95,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Remove outlier embryos from distance matrix.

    Convenience wrapper around identify_outliers() that returns a cleaned
    distance matrix ready for clustering.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n_embryos, n_embryos)
    embryo_ids : List[str]
        List of embryo identifiers
    outlier_detection_method : str, default='percentile'
        Method for outlier detection ('median_distance', 'percentile', 'iqr', 'mad')
    outlier_threshold : float, optional
        Manual threshold (for 'median_distance' method)
    outlier_percentile : float, default=95
        Percentile cutoff (for 'percentile' method)
    verbose : bool, default=True
        Print diagnostic information

    Returns
    -------
    D_clean : np.ndarray
        Distance matrix with outliers removed
    embryo_ids_clean : List[str]
        List of non-outlier embryo IDs
    info : Dict[str, Any]
        Dict with outlier detection information (from identify_outliers)

    Examples
    --------
    >>> D_clean, embryo_ids_clean, info = remove_outliers_from_distance_matrix(
    ...     D, embryo_ids, method='percentile', percentile=95
    ... )
    >>> print(f"Removed {len(info['outlier_indices'])} outliers")
    >>> print(f"Clean distance matrix shape: {D_clean.shape}")
    """
    # Identify outliers
    outlier_ids, inlier_ids, info = identify_outliers(
        D,
        embryo_ids,
        method=outlier_detection_method,
        threshold=outlier_threshold,
        percentile=outlier_percentile,
        verbose=verbose,
    )

    # Extract clean distance matrix (inliers only)
    inlier_idx = info['inlier_indices']
    D_clean = D[np.ix_(inlier_idx, inlier_idx)]

    if verbose:
        print(f"\n✓ Outliers removed")
        print(f"  Original size: {D.shape}")
        print(f"  Clean size: {D_clean.shape}")

    return D_clean, inlier_ids, info
