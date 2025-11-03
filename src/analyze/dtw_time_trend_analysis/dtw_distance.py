"""
DTW Distance Computation

Dynamic Time Warping (DTW) implementation with Sakoe-Chiba band constraint.

This module provides functions for computing pairwise DTW distances between
variable-length temporal sequences, suitable for clustering and comparison of
embryonic trajectory data.

Functions
=========
- compute_dtw_distance : Compute DTW distance between two sequences
- compute_dtw_distance_matrix : Compute pairwise DTW distances for multiple sequences
"""

import numpy as np
from typing import Tuple, Optional


def compute_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: int = 3,
    normalize: bool = False
) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.

    Uses the Sakoe-Chiba band constraint to limit the warping path and
    improve computational efficiency. Band width is automatically expanded
    to handle sequences of different lengths.

    Parameters
    ----------
    seq1 : array-like
        First input sequence (1D array)
    seq2 : array-like
        Second input sequence (1D array)
    window : int, default=3
        Sakoe-Chiba band width constraint. Window is automatically expanded
        to max(window, abs(len(seq1) - len(seq2))) to handle length differences.
    normalize : bool, default=False
        If True, return per-step normalized distance (length-independent metric).
        Normalization divides by path length (n + m).

    Returns
    -------
    float
        DTW distance between sequences. If normalize=True, returns per-step distance.
        Returns inf if computation fails (e.g., all NaN values).

    Notes
    -----
    The DTW distance represents the minimum cumulative cost of aligning two sequences,
    where cost is defined as the absolute difference between values at each pair of
    timepoints. The Sakoe-Chiba band constraint restricts the warping path to stay
    within a diagonal band, improving speed while limiting pathological alignments.

    References
    ----------
    - Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization
      for spoken word recognition. IEEE Transactions on Acoustics, Speech, and
      Signal Processing, 26(1), 43-49.

    Examples
    --------
    >>> seq1 = np.array([1.0, 2.0, 3.0, 4.0])
    >>> seq2 = np.array([1.5, 2.5, 3.5])
    >>> dist = compute_dtw_distance(seq1, seq2)
    >>> print(f"DTW distance: {dist:.3f}")

    >>> # Normalized distance (length-independent)
    >>> dist_norm = compute_dtw_distance(seq1, seq2, normalize=True)
    >>> print(f"Normalized DTW: {dist_norm:.3f}")
    """
    # Convert to numeric arrays (prevents LAPACK errors with object dtypes)
    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)
    n, m = len(seq1), len(seq2)

    # Expand window to handle sequence length differences
    w = max(window, abs(n - m))

    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill the cost matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)
        for j in range(j_start, j_end):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    distance = dtw_matrix[n, m]

    # Normalize by path length if requested (length-independent metric)
    if normalize and not np.isinf(distance):
        distance = distance / (n + m)

    return float(distance)


def compute_dtw_distance_matrix(
    trajectories: list,
    window: int = 3,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute pairwise DTW distances for multiple trajectories.

    Computes a symmetric distance matrix where each element (i, j) represents
    the DTW distance between trajectories i and j. Diagonal elements are zero.

    Parameters
    ----------
    trajectories : list of array-like
        List of trajectories (variable-length 1D arrays)
    window : int, default=3
        Sakoe-Chiba band width for DTW computation
    verbose : bool, default=False
        If True, print progress updates every 10 trajectories

    Returns
    -------
    distance_matrix : np.ndarray
        Symmetric (n_trajectories × n_trajectories) distance matrix.
        Distance matrix properties:
        - Symmetric: distance_matrix[i,j] == distance_matrix[j,i]
        - Zero diagonal: np.allclose(np.diag(distance_matrix), 0)
        - Non-negative: all values >= 0 (unless inf from computation errors)

    Raises
    ------
    Warning (printed, not raised)
        If DTW computation fails for a pair, that pair receives inf distance
        and a warning is printed (if verbose=True).

    Examples
    --------
    >>> trajectories = [
    ...     np.array([1.0, 2.0, 3.0, 4.0]),
    ...     np.array([1.5, 2.5, 3.5]),
    ...     np.array([0.9, 2.1, 3.1, 3.9, 4.1])
    ... ]
    >>> dist_matrix = compute_dtw_distance_matrix(trajectories, window=3)
    >>> print(dist_matrix.shape)
    (3, 3)
    >>> print(np.diag(dist_matrix))  # Should be [0, 0, 0]
    [0. 0. 0.]
    >>> print(dist_matrix[0, 1], dist_matrix[1, 0])  # Should be symmetric
    """
    n_trajectories = len(trajectories)
    distance_matrix = np.zeros((n_trajectories, n_trajectories))

    if verbose:
        print(f"\n  Computing pairwise DTW distances (window={window})...")

    for i in range(n_trajectories):
        if verbose and (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{n_trajectories}")

        for j in range(i + 1, n_trajectories):
            try:
                dist = compute_dtw_distance(
                    trajectories[i],
                    trajectories[j],
                    window=window
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
            except Exception as e:
                if verbose:
                    print(f"    Warning: DTW computation failed for pair ({i}, {j}): {e}")
                distance_matrix[i, j] = np.inf
                distance_matrix[j, i] = np.inf

    if verbose:
        # Validation stats
        nan_count = np.isnan(distance_matrix).sum()
        inf_count = np.isinf(distance_matrix).sum()
        diag_check = np.allclose(np.diag(distance_matrix), 0)

        print(f"\n  Validation:")
        print(f"    NaN count: {nan_count}")
        print(f"    Inf count: {inf_count}")
        print(f"    Diagonal ≈ 0: {diag_check}")
        print(f"    Distance stats: min={np.nanmin(distance_matrix):.3f}, "
              f"max={np.nanmax(distance_matrix):.3f}, "
              f"mean={np.nanmean(distance_matrix):.3f}")
        print(f"    Matrix shape: {distance_matrix.shape}")

    return distance_matrix
