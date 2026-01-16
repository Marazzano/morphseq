"""
DTW Distance Computation

Dynamic Time Warping (DTW) implementation with Sakoe-Chiba band constraint.

This module provides functions for computing pairwise DTW distances between
variable-length temporal sequences, suitable for clustering and comparison of
embryonic trajectory data.

Functions
=========
Univariate DTW:
- compute_dtw_distance : Compute DTW distance between two 1D sequences
- compute_dtw_distance_matrix : Compute pairwise DTW distances for multiple 1D sequences

Multivariate DTW (MD-DTW):
- prepare_multivariate_array : Convert DataFrame to 3D array for MD-DTW
- compute_md_dtw_distance_matrix : Compute pairwise MD-DTW distances
- _dtw_multivariate_pair : Helper for pairwise multivariate DTW (internal)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional


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


# ============================================================================
# Multivariate DTW (MD-DTW) Functions
# ============================================================================


def prepare_multivariate_array(
    df: pd.DataFrame,
    metrics: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_grid: Optional[np.ndarray] = None,
    normalize: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Convert long-format DataFrame to 3D array for multivariate DTW.

    Takes a DataFrame with multiple metric columns and converts to a 3D numpy array
    suitable for multivariate DTW computation. Handles interpolation to common time
    grid and optional Z-score normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns [embryo_id, time_col, metric1, metric2, ...]
    metrics : List[str]
        List of metric column names (e.g., ['baseline_deviation_normalized', 'total_length_um'])
    time_col : str, default='predicted_stage_hpf'
        Name of time column
    embryo_id_col : str, default='embryo_id'
        Name of embryo ID column
    time_grid : np.ndarray, optional
        Optional pre-defined time grid. If None, auto-computed from data
    normalize : bool, default=True
        Whether to Z-score normalize each metric globally
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    X : np.ndarray
        Array with shape (n_embryos, n_timepoints, n_metrics)
    embryo_ids : List[str]
        List of embryo identifiers (same order as X rows)
    time_grid : np.ndarray
        Time values (same for all embryos)

    Examples
    --------
    >>> df = load_experiment_dataframe('20251121')
    >>> X, embryo_ids, time_grid = prepare_multivariate_array(
    ...     df,
    ...     metrics=['baseline_deviation_normalized', 'total_length_um']
    ... )
    >>> print(X.shape)  # (n_embryos, n_timepoints, 2)

    Notes
    -----
    - All embryos are interpolated to the same common time grid
    - Missing values (NaN) are handled by linear interpolation
    - Z-score normalization ensures equal weight for all metrics in DTW
    """
    from ..utilities.trajectory_utils import interpolate_to_common_grid_multi_df
    from ..config import GRID_STEP

    if verbose:
        print(f"Preparing multivariate array for {len(metrics)} metrics...")
        print(f"  Metrics: {metrics}")
        print(f"  Normalization: {normalize}")

    # Step 1: Get embryo IDs in sorted order (for consistency)
    embryo_ids = sorted(df[embryo_id_col].unique())
    n_embryos = len(embryo_ids)
    n_metrics = len(metrics)

    if verbose:
        print(f"  Embryos: {n_embryos}")

    # Step 2: Interpolate each metric for all embryos using the trajectory utility
    # If time_grid is provided, pass it through; otherwise let the utility derive the grid
    provided_time_grid = None
    if time_grid is not None:
        provided_time_grid = np.asarray(time_grid, dtype=float)
        if provided_time_grid.ndim != 1:
            raise ValueError(f"time_grid must be 1D, got shape {provided_time_grid.shape}")
        if len(provided_time_grid) == 0:
            raise ValueError("time_grid must be non-empty")
        # Ensure strictly increasing, unique grid (dedupe protects against float replication)
        provided_time_grid = np.unique(provided_time_grid)
        if len(provided_time_grid) > 1 and not np.all(np.diff(provided_time_grid) > 0):
            provided_time_grid = np.sort(provided_time_grid)

    df_interp = interpolate_to_common_grid_multi_df(
        df,
        metrics,
        grid_step=(provided_time_grid[1] - provided_time_grid[0]) if provided_time_grid is not None and len(provided_time_grid) > 1 else GRID_STEP,
        time_col=time_col,
        time_grid=provided_time_grid,
        fill_edges=False,
        verbose=verbose,
    )

    # If a grid was provided, keep it exactly (critical for cross-dataset comparisons).
    # Otherwise derive grid from interpolation output.
    if provided_time_grid is not None:
        time_grid = provided_time_grid
    else:
        time_grid = np.sort(df_interp[time_col].unique())
    n_timepoints = len(time_grid)

    if verbose:
        print(f"  Time points: {n_timepoints} ({time_grid.min():.1f} - {time_grid.max():.1f} hpf)")

    # Step 3: Initialize 3D array
    X = np.zeros((n_embryos, n_timepoints, n_metrics))

    for i, embryo_id in enumerate(embryo_ids):
        emb_df = df_interp[df_interp[embryo_id_col] == embryo_id].set_index(time_col)

        if emb_df.empty:
            if verbose:
                print(f"  WARNING: Embryo {embryo_id} has no interpolated rows, using zeros")
            continue

        for j, metric in enumerate(metrics):
            # Reindex onto full grid and fill missing values
            ser = emb_df[metric].reindex(time_grid)
            # Fill interior gaps only; keep out-of-range edges as NaN then set to 0.
            ser = ser.interpolate(limit_area='inside').fillna(0)
            X[i, :, j] = ser.values

    # Step 5: Handle remaining NaNs (e.g., at edges due to interpolation bounds)
    mask = np.isnan(X)
    if mask.any():
        for i in range(n_embryos):
            for j in range(n_metrics):
                series = X[i, :, j]
                nans = np.isnan(series)

                if nans.all():
                    # All NaNs - set to 0
                    X[i, :, j] = 0
                else:
                    # Use pandas to fill NaNs with interpolation
                    filled = pd.Series(series).interpolate(limit_area='inside').fillna(0)
                    X[i, :, j] = filled.values

    if verbose:
        print(f"  Array shape: {X.shape}")
        print(f"  Before normalization:")
        for j, metric in enumerate(metrics):
            print(f"    {metric}: mean={X[:, :, j].mean():.3f}, std={X[:, :, j].std():.3f}")

    # Step 6: Global Z-score normalization (if enabled)
    if normalize:
        original_shape = X.shape
        X_reshaped = X.reshape(-1, n_metrics)

        # Z-score each metric globally
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_reshaped)

        X = X_normalized.reshape(original_shape)

        if verbose:
            print(f"  After normalization:")
            for j, metric in enumerate(metrics):
                print(f"    {metric}: mean={X[:, :, j].mean():.6f}, std={X[:, :, j].std():.6f}")

    if verbose:
        print(f"✓ Multivariate array prepared successfully")

    return X, embryo_ids, time_grid


def _dtw_multivariate_pair(
    ts_a: np.ndarray,
    ts_b: np.ndarray,
    window: Optional[int] = 3,
) -> float:
    """
    Compute DTW distance between two multivariate time series.

    Uses Euclidean distance in feature space as the local distance metric.
    Implements dynamic programming with optional Sakoe-Chiba band constraint.

    Parameters
    ----------
    ts_a : np.ndarray
        2D array with shape (T_a, n_features) - time series A
    ts_b : np.ndarray
        2D array with shape (T_b, n_features) - time series B
    window : int, optional
        Sakoe-Chiba band width (None for unconstrained DTW)

    Returns
    -------
    float
        DTW distance between ts_a and ts_b

    Notes
    -----
    - The "multivariate" part is handled by computing Euclidean distance
      between feature vectors at each timepoint pair
    - ts_a[i] and ts_b[j] are vectors in feature space
    - Distance is computed as sqrt(sum((ts_a[i] - ts_b[j])^2))
    """
    # Step 1: Compute local cost matrix
    # dist_matrix[i, j] = Euclidean distance between ts_a[i] and ts_b[j]
    # This naturally handles multivariate data
    dist_matrix = cdist(ts_a, ts_b, metric='euclidean')

    n, m = dist_matrix.shape

    # Step 2: Initialize accumulated cost matrix
    # Set all to infinity initially
    acc_cost = np.full((n + 1, m + 1), np.inf)
    acc_cost[0, 0] = 0

    # Step 3: Dynamic Programming with Sakoe-Chiba Constraint
    # The window parameter limits how far we can warp in time
    if window is None:
        # Unconstrained DTW - compute all pairs
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i - 1, j - 1]
                acc_cost[i, j] = cost + min(
                    acc_cost[i - 1, j],      # Insertion (skip in ts_a)
                    acc_cost[i, j - 1],      # Deletion (skip in ts_b)
                    acc_cost[i - 1, j - 1]   # Match
                )
    else:
        # Constrained DTW - only compute within band
        # The band width is determined by the window parameter
        w = max(window, abs(n - m))

        for i in range(1, n + 1):
            # Determine valid range for j
            start_j = max(1, i - w)
            end_j = min(m + 1, i + w + 1)

            for j in range(start_j, end_j):
                cost = dist_matrix[i - 1, j - 1]
                acc_cost[i, j] = cost + min(
                    acc_cost[i - 1, j],
                    acc_cost[i, j - 1],
                    acc_cost[i - 1, j - 1]
                )

    return acc_cost[n, m]


def compute_md_dtw_distance_matrix(
    X: np.ndarray,
    sakoe_chiba_radius: Optional[int] = 3,
    n_jobs: int = -1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute multivariate DTW distance matrix.

    Computes pairwise multivariate DTW distances between all embryos using
    pure Python/NumPy implementation with parallel processing.

    Parameters
    ----------
    X : np.ndarray
        3D array with shape (n_embryos, n_timepoints, n_metrics)
    sakoe_chiba_radius : int, optional
        Sakoe-Chiba band constraint width.
        Limits warping to within ±radius timepoints.
        None for unconstrained DTW (slower).
        Default: 3 (good balance of flexibility and speed)
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available CPUs.
        1 means single-threaded (no parallelization).
    verbose : bool, default=True
        Print progress and diagnostics

    Returns
    -------
    distance_matrix : np.ndarray
        Array with shape (n_embryos, n_embryos).
        Symmetric matrix where D[i,j] = DTW distance between embryo i and j

    Examples
    --------
    >>> X, embryo_ids, time_grid = prepare_multivariate_array(df, metrics=['curvature', 'length'])
    >>> D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3)
    >>> print(D.shape)  # (n_embryos, n_embryos)
    >>> # Use D for hierarchical clustering
    >>> from src.analyze.trajectory_analysis import run_bootstrap_hierarchical
    >>> results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids)

    Notes
    -----
    - This is a pure Python implementation using NumPy/SciPy
    - Parallelized with joblib for multi-core speedup
    - Time complexity: O(N^2 * T^2 / n_jobs) where N=embryos, T=timepoints
    - Output is symmetric by construction: D[i,j] == D[j,i]
    - Diagonal is zero: D[i,i] == 0
    """
    from joblib import Parallel, delayed, cpu_count

    n_embryos = X.shape[0]

    # Determine actual number of jobs
    if n_jobs == -1:
        actual_jobs = cpu_count()
    else:
        actual_jobs = min(n_jobs, cpu_count())

    if verbose:
        print(f"Computing MD-DTW distance matrix...")
        print(f"  Embryos: {n_embryos}")
        print(f"  Array shape: {X.shape}")
        print(f"  Sakoe-Chiba radius: {sakoe_chiba_radius}")
        print(f"  Parallel jobs: {actual_jobs} (of {cpu_count()} CPUs available)")

    # Generate all unique pairs (i, j) where i < j
    pairs = [(i, j) for i in range(n_embryos) for j in range(i + 1, n_embryos)]
    total_pairs = len(pairs)

    if verbose:
        print(f"  Computing {total_pairs} pairwise distances...")

    # Parallel computation of all pairwise distances
    if actual_jobs == 1:
        # Single-threaded fallback
        results = []
        for idx, (i, j) in enumerate(pairs):
            dist = _dtw_multivariate_pair(X[i], X[j], window=sakoe_chiba_radius)
            results.append(dist)
            if verbose and (idx + 1) % max(1, total_pairs // 10) == 0:
                print(f"  Progress: {idx + 1}/{total_pairs} ({100*(idx+1)//total_pairs}%)", end='\r')
    else:
        # Parallel computation
        results = Parallel(n_jobs=actual_jobs, verbose=0)(
            delayed(_dtw_multivariate_pair)(X[i], X[j], window=sakoe_chiba_radius)
            for i, j in pairs
        )

    # Build symmetric distance matrix from results
    D = np.zeros((n_embryos, n_embryos))
    for (i, j), dist in zip(pairs, results):
        D[i, j] = dist
        D[j, i] = dist

    if verbose:
        # Verify properties
        diagonal_max = np.max(np.abs(np.diag(D)))
        asymmetry = np.max(np.abs(D - D.T))

        print(f"\n✓ Distance matrix computed")
        print(f"  Shape: {D.shape}")
        print(f"  Distance range: [{D[D > 0].min():.4f}, {D.max():.4f}]")
        print(f"  Max diagonal value: {diagonal_max:.2e} (should be ~0)")
        print(f"  Max asymmetry: {asymmetry:.2e} (should be ~0)")

        if diagonal_max > 1e-10:
            print(f"  WARNING: Diagonal not zero (max={diagonal_max:.2e})")
        if asymmetry > 1e-10:
            print(f"  WARNING: Matrix not symmetric (max asymmetry={asymmetry:.2e})")

    return D


def compute_trajectory_distances(
    df: pd.DataFrame,
    metrics: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_window: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    sakoe_chiba_radius: int = 3,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compute MD-DTW distance matrix from trajectory DataFrame.

    This is the PRIMARY convenience function for converting trajectory data into
    a distance matrix for clustering analysis. Handles time filtering, array
    preparation, and distance computation in one step.

    This function combines three steps:
    1. Optional time window filtering
    2. Multivariate array preparation (prepare_multivariate_array)
    3. MD-DTW distance computation (compute_md_dtw_distance_matrix)

    Parameters
    ----------
    df : pd.DataFrame
        Long-format trajectory data with columns for time, embryo_id, and metrics.
    metrics : List[str]
        Names of columns to use as features (e.g., ['curvature', 'length']).
    time_col : str, default='predicted_stage_hpf'
        Column name for time values.
    embryo_id_col : str, default='embryo_id'
        Column identifying unique embryos/trajectories.
    time_window : Optional[Tuple[float, float]], default=None
        If provided, filters to (min_time, max_time) before computing distances.
        Example: (30, 60) to analyze only 30-60 hpf.
    normalize : bool, default=True
        If True, z-score normalize each metric across all embryos.
    sakoe_chiba_radius : int, default=3
        Warping window constraint for DTW (3 is a good default).
    n_jobs : int, default=-1
        Number of parallel jobs for distance computation.
        -1 means use all available CPUs (auto-detect).
        1 means single-threaded (no parallelization).
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    D : np.ndarray
        Distance matrix (n_embryos × n_embryos), symmetric.
    embryo_ids : List[str]
        Ordered list of embryo IDs corresponding to distance matrix rows/cols.
    time_grid : np.ndarray
        Common time grid used for interpolation.

    Examples
    --------
    >>> # Compute distances on full time range
    >>> D, embryo_ids, time_grid = compute_trajectory_distances(
    ...     df,
    ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
    ... )

    >>> # Compute distances on specific time window (e.g., 30-60 hpf)
    >>> D, embryo_ids, time_grid = compute_trajectory_distances(
    ...     df,
    ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
    ...     time_window=(30, 60),
    ... )

    >>> # Then cluster
    >>> from src.analyze.trajectory_analysis import run_k_selection_with_plots
    >>> results = run_k_selection_with_plots(
    ...     df=df,
    ...     D=D,
    ...     embryo_ids=embryo_ids,
    ...     output_dir=Path('results/clustering'),
    ...     k_range=[2, 3, 4, 5],
    ... )

    Notes
    -----
    - This is a convenience wrapper that simplifies the common workflow
    - For advanced use cases, you can still use prepare_multivariate_array()
      and compute_md_dtw_distance_matrix() separately
    - Time filtering happens BEFORE interpolation, which may result in fewer
      embryos if some have no data in the specified window
    """
    if verbose:
        print("="*70)
        print("COMPUTE TRAJECTORY DISTANCES")
        print("="*70)

    # Step 1: Filter by time window if specified
    if time_window is not None:
        min_time, max_time = time_window
        if verbose:
            print(f"\n1. Filtering to time window: [{min_time}, {max_time}] {time_col}")

        df_filtered = df[
            (df[time_col] >= min_time) &
            (df[time_col] <= max_time)
        ].copy()

        if verbose:
            n_embryos_before = df[embryo_id_col].nunique()
            n_embryos_after = df_filtered[embryo_id_col].nunique()
            print(f"   Embryos before: {n_embryos_before}")
            print(f"   Embryos after: {n_embryos_after}")

            if n_embryos_after < n_embryos_before:
                print(f"   WARNING: Lost {n_embryos_before - n_embryos_after} embryos")
                print("            (no data in time window)")
    else:
        df_filtered = df.copy()
        if verbose:
            print("\n1. Using full time range")

    # Step 2: Prepare multivariate array
    if verbose:
        print(f"\n2. Preparing multivariate array")
        print(f"   Metrics: {metrics}")
        print(f"   Normalize: {normalize}")

    X, embryo_ids, time_grid = prepare_multivariate_array(
        df_filtered,
        metrics=metrics,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        normalize=normalize,
        verbose=verbose,
    )

    if verbose:
        print(f"\n   Array shape: {X.shape} (embryos × timepoints × metrics)")
        print(f"   Time grid: [{time_grid[0]:.1f}, {time_grid[-1]:.1f}] ({len(time_grid)} points)")

    # Step 3: Compute MD-DTW distance matrix
    if verbose:
        print(f"\n3. Computing MD-DTW distances")
        print(f"   Sakoe-Chiba radius: {sakoe_chiba_radius}")

    D = compute_md_dtw_distance_matrix(
        X,
        sakoe_chiba_radius=sakoe_chiba_radius,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if verbose:
        print(f"\nDistance matrix: {D.shape}")
        print(f"  Range: [{D[D > 0].min():.3f}, {D.max():.3f}]")
        print("="*70)

    return D, embryo_ids, time_grid
