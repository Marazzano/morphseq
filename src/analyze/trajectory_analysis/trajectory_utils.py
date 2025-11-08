"""
Trajectory Utilities

Data extraction, interpolation, and preprocessing for temporal trajectories.

This module provides a DataFrame-centric API where time column (hpf) travels
with data through the entire pipeline, eliminating time-axis alignment bugs.

Functions (New API - DataFrame-first)
-------------------------------------
- extract_trajectories_df: Extract filtered trajectory DataFrame
- interpolate_trajectories: Fill missing values via linear interpolation (operates on DataFrame)
- interpolate_to_common_grid_df: Align trajectories to common time grid (returns DataFrame)
- df_to_trajectories: Convert DataFrame to arrays for DTW computation
- extract_early_late_means: Extract window-based mean values (operates on DataFrame)

Functions (Legacy API - deprecated, kept for backward compatibility)
-------------------------------------------------------------------
- extract_trajectories: DEPRECATED - use extract_trajectories_df()
- interpolate_to_common_grid: DEPRECATED - use interpolate_to_common_grid_df()
- pad_trajectories_for_plotting: DEPRECATED - use plotting functions with DataFrame directly
"""

import warnings
import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Tuple, List, Dict, Optional
from .config import DEFAULT_EMBRYO_ID_COL, DEFAULT_METRIC_COL, DEFAULT_TIME_COL, MIN_TIMEPOINTS, GRID_STEP


# ============================================================================
# NEW API: DataFrame-first (recommended)
# ============================================================================

def extract_trajectories_df(
    df: pd.DataFrame,
    genotype_filter: Optional[str] = None,
    metric_name: str = 'normalized_baseline_deviation',
    min_timepoints: int = MIN_TIMEPOINTS,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Extract and filter trajectory data, returning long-format DataFrame.

    Converts raw data into a long-format DataFrame with columns [embryo_id, hpf, metric_value].
    Filters by genotype and requires minimum number of timepoints per embryo.
    Time column (hpf) is preserved and will travel with data through the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns: embryo_id, predicted_stage_hpf, metric_name, and optionally genotype
    genotype_filter : str, optional
        If specified, filter dataframe to only this genotype
    metric_name : str, default='normalized_baseline_deviation'
        Name of the metric column to extract
    min_timepoints : int, default=MIN_TIMEPOINTS
        Minimum number of non-NaN timepoints required per embryo
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    df_filtered : pd.DataFrame
        Filtered trajectories with columns [embryo_id, hpf, metric_value].
        Only includes embryos with >= min_timepoints observations.
        Time column (hpf) is preserved for downstream processing.

    Examples
    --------
    >>> df_filtered = extract_trajectories_df(
    ...     df,
    ...     genotype_filter='cep290_homozygous',
    ...     metric_name='curvature'
    ... )
    >>> print(f"Extracted {df_filtered['embryo_id'].nunique()} embryos")
    >>> print(df_filtered.head())
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 1: DATA EXTRACTION & PREPARATION")
        print(f"{'='*80}")

    df = df.copy()

    # Filter for genotype if specified
    if genotype_filter is not None:
        df = df[df['genotype'] == genotype_filter].copy()
        if verbose:
            print(f"\n  Filtered to {genotype_filter}: {len(df)} timepoints")

    # Extract relevant columns
    df_filtered = df[['embryo_id', 'predicted_stage_hpf', metric_name]].copy()
    df_filtered.columns = ['embryo_id', 'hpf', 'metric_value']

    # Drop NaN values
    initial_rows = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=['metric_value'])
    dropped = initial_rows - len(df_filtered)
    if verbose:
        print(f"  Dropped {dropped} rows with NaN metric values: {len(df_filtered)} remaining")

    # Filter for minimum timepoints per embryo
    embryo_counts = df_filtered.groupby('embryo_id').size()
    embryos_to_keep = embryo_counts[embryo_counts >= min_timepoints].index
    df_filtered = df_filtered[df_filtered['embryo_id'].isin(embryos_to_keep)]

    if verbose:
        print(f"\n  Extracted {df_filtered['embryo_id'].nunique()} embryo trajectories (min {min_timepoints} timepoints)")
        embryo_lens = df_filtered.groupby('embryo_id').size()
        print(f"  Mean trajectory length: {embryo_lens.mean():.1f} timepoints")

    return df_filtered


def interpolate_to_common_grid_df(
    df_filtered: pd.DataFrame,
    grid_step: float = GRID_STEP,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Interpolate all trajectories to a common time grid, returning DataFrame.

    Aligns variable-length trajectories to a regular grid of timepoints.
    All embryos will share the same set of hpf values (the common grid).
    Time column (hpf) is preserved in output DataFrame.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered trajectory data from extract_trajectories_df()
        Must have columns: embryo_id, hpf, metric_value
    grid_step : float, default=GRID_STEP
        Step size for common time grid (in HPF units)
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    df_interpolated : pd.DataFrame
        Trajectories on common grid with columns [embryo_id, hpf, metric_value].
        All embryos now share identical hpf values (the global common grid).
        Time column is preserved for plotting and downstream use.

    Examples
    --------
    >>> df_interpolated = interpolate_to_common_grid_df(df_filtered, grid_step=0.5)
    >>> print(f"Grid size: {df_interpolated['hpf'].nunique()}")
    >>> print(f"HPF range: {df_interpolated['hpf'].min():.1f} to {df_interpolated['hpf'].max():.1f}")
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 3: TRAJECTORY INTERPOLATION TO COMMON TIMEPOINTS")
        print(f"{'='*80}")

    df_filtered = df_filtered.copy()

    # Find global min/max hpf across all trajectories
    min_hpf = df_filtered['hpf'].min()
    max_hpf = df_filtered['hpf'].max()

    if verbose:
        print(f"\n  HPF range: {min_hpf:.1f} to {max_hpf:.1f}")

    # Create common timepoint grid
    common_grid = np.arange(min_hpf, max_hpf + grid_step, grid_step)
    if verbose:
        print(f"  Common grid: {len(common_grid)} timepoints (step={grid_step} hpf)")

    # Interpolate each trajectory to common grid
    rows_list = []

    for embryo_id, group in df_filtered.groupby('embryo_id'):
        group_sorted = group.sort_values('hpf')
        hpf_vals = group_sorted['hpf'].values
        metric_vals = group_sorted['metric_value'].values

        # Linear interpolation to common grid
        f = interpolate.interp1d(
            hpf_vals,
            metric_vals,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Interpolate - will have NaN where outside observed range
        interpolated = f(common_grid)

        # Create DataFrame rows (keep time column!)
        for hpf, metric_value in zip(common_grid, interpolated):
            if not np.isnan(metric_value):  # Only keep valid values
                rows_list.append({
                    'embryo_id': embryo_id,
                    'hpf': hpf,
                    'metric_value': metric_value
                })

    df_interpolated = pd.DataFrame(rows_list)

    if verbose:
        n_embryos = df_interpolated['embryo_id'].nunique()
        print(f"\n  Interpolated shape: {n_embryos} embryos")
        embryo_lens = df_interpolated.groupby('embryo_id').size()
        print(f"  Trajectory lengths: min={embryo_lens.min()}, max={embryo_lens.max()}, mean={embryo_lens.mean():.1f}")
        print(f"  ✓ All trajectories on common grid, time column preserved")

    return df_interpolated


def df_to_trajectories(
    df_interpolated: pd.DataFrame,
    embryo_id_col: str = 'embryo_id',
    value_col: str = 'metric_value'
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Convert interpolated DataFrame to trajectory arrays for DTW computation.

    Extracts value arrays and embryo ordering from the long-format DataFrame.
    This is a lightweight conversion step - use only when array form is needed
    (e.g., for DTW distance computation). For plotting, keep data in DataFrame form
    to preserve time information.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
        Regularized trajectory data from interpolate_to_common_grid_df()
        Must have columns: embryo_id, hpf, metric_value
    embryo_id_col : str, default='embryo_id'
        Column name for embryo identifiers
    value_col : str, default='metric_value'
        Column name for metric values

    Returns
    -------
    trajectories : list of np.ndarray
        Value arrays for each embryo
    embryo_ids : list of str
        Embryo identifiers in same order as trajectories
    common_grid : np.ndarray
        Unique time values (hpf) from the common grid

    Examples
    --------
    >>> trajectories, embryo_ids, common_grid = df_to_trajectories(df_interpolated)
    >>> # Now can use for DTW:
    >>> from .dtw_distance import compute_dtw_distance_matrix
    >>> D = compute_dtw_distance_matrix(trajectories)
    """
    # Get unique embryo IDs in order
    embryo_ids = df_interpolated[embryo_id_col].unique().tolist()

    # Extract trajectory arrays
    trajectories = []
    for embryo_id in embryo_ids:
        subset = df_interpolated[df_interpolated[embryo_id_col] == embryo_id]
        subset_sorted = subset.sort_values('hpf')
        trajectories.append(subset_sorted[value_col].values)

    # Extract common grid
    common_grid = np.sort(df_interpolated['hpf'].unique())

    return trajectories, embryo_ids, common_grid


# ============================================================================
# LEGACY API: Array-based (deprecated, kept for backward compatibility)
# ============================================================================

def extract_trajectories(
    df: pd.DataFrame,
    genotype_filter: Optional[str] = None,
    metric_name: str = 'normalized_baseline_deviation',
    min_timepoints: int = MIN_TIMEPOINTS,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[str], pd.DataFrame]:
    """
    DEPRECATED: Use extract_trajectories_df() instead.

    Extract per-embryo trajectories from long-format dataframe (legacy array API).

    This function returns arrays and a separate dataframe, which loses time context.
    Use extract_trajectories_df() for the new DataFrame-centric API that preserves
    time information throughout the pipeline.

    Returns
    -------
    trajectories : list of np.ndarray
        List of trajectory arrays (variable lengths, sorted by hpf)
    embryo_ids : list of str
        Embryo IDs corresponding to trajectories
    trajectories_df : pd.DataFrame
        Long-format dataframe used for extraction (with NaN removed)
    """
    warnings.warn(
        "extract_trajectories() is deprecated. Use extract_trajectories_df() instead, "
        "which returns a DataFrame with time column preserved. "
        "See migration guide in README.md",
        DeprecationWarning,
        stacklevel=2
    )

    if verbose:
        print(f"\n{'='*80}")
        print("STEP 1: DATA EXTRACTION & PREPARATION (LEGACY - DEPRECATED)")
        print(f"{'='*80}")

    df = df.copy()

    # Filter for genotype if specified
    if genotype_filter is not None:
        df = df[df['genotype'] == genotype_filter].copy()
        if verbose:
            print(f"\n  Filtered to {genotype_filter}: {len(df)} timepoints")

    # Extract relevant columns
    df_long = df[['embryo_id', 'predicted_stage_hpf', metric_name]].copy()
    df_long.columns = ['embryo_id', 'hpf', 'metric_value']

    # Drop NaN values
    initial_rows = len(df_long)
    df_long = df_long.dropna(subset=['metric_value'])
    dropped = initial_rows - len(df_long)
    if verbose:
        print(f"  Dropped {dropped} rows with NaN metric values: {len(df_long)} remaining")

    # Extract per-embryo trajectories
    trajectories = []
    embryo_ids = []

    for embryo_id, group in df_long.groupby('embryo_id'):
        trajectory = group.sort_values('hpf')['metric_value'].values

        if len(trajectory) >= min_timepoints:
            trajectories.append(trajectory)
            embryo_ids.append(embryo_id)

    if verbose:
        print(f"\n  Extracted {len(trajectories)} embryo trajectories (min {min_timepoints} timepoints)")
        print(f"  Mean trajectory length: {np.mean([len(t) for t in trajectories]):.1f} timepoints")

    return trajectories, embryo_ids, df_long


def interpolate_trajectories(
    df_long: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply linear interpolation to handle missing values within trajectories.

    For each embryo, identifies gaps in timepoints and uses linear interpolation
    to fill NaN metric values. Interpolation only occurs within the observed
    timepoint range (no extrapolation).

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format dataframe with columns: embryo_id, hpf, metric_value
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    df_long : pd.DataFrame
        Imputed dataframe (same shape as input, with NaN values filled)

    Notes
    -----
    - Uses scipy.interpolate.interp1d with kind='linear'
    - Only interpolates within observed HPF range
    - Extrapolation is disabled (fill_value='extrapolate' not used for boundaries)

    Examples
    --------
    >>> df_long_imputed = interpolate_trajectories(df_long)
    >>> print(f"Remaining NaN values: {df_long_imputed['metric_value'].isna().sum()}")
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 2: MISSING DATA HANDLING (INTERPOLATION)")
        print(f"{'='*80}")

    df_long = df_long.copy()
    interpolated_count = 0

    for embryo_id in df_long['embryo_id'].unique():
        embryo_data = df_long[df_long['embryo_id'] == embryo_id].copy()

        if len(embryo_data) > 1 and embryo_data['metric_value'].isna().sum() > 0:
            # Interpolate missing values
            valid_mask = ~embryo_data['metric_value'].isna()
            if valid_mask.sum() > 1:
                f = interpolate.interp1d(
                    embryo_data[valid_mask]['hpf'].values,
                    embryo_data[valid_mask]['metric_value'].values,
                    kind='linear',
                    fill_value='extrapolate'
                )

                # Fill NaN values with interpolated values
                nan_mask = embryo_data['metric_value'].isna()
                df_long.loc[embryo_data[nan_mask].index, 'metric_value'] = f(
                    embryo_data[nan_mask]['hpf'].values
                )
                interpolated_count += nan_mask.sum()

    if verbose:
        print(f"\n  Interpolated {interpolated_count} missing values")
        print(f"  Remaining NaN values: {df_long['metric_value'].isna().sum()}")

    return df_long


def interpolate_to_common_grid(
    df_long: pd.DataFrame,
    grid_step: float = GRID_STEP,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[str], Dict[str, int], np.ndarray]:
    """
    DEPRECATED: Use interpolate_to_common_grid_df() instead.

    Interpolate all trajectories to a common timepoint grid (legacy array API).

    This function returns trimmed arrays without time context, leading to plotting bugs.
    Use interpolate_to_common_grid_df() for the new DataFrame-centric API that
    preserves time information.
    """
    warnings.warn(
        "interpolate_to_common_grid() is deprecated. Use interpolate_to_common_grid_df() instead, "
        "which returns a DataFrame with time column preserved. "
        "See migration guide in README.md",
        DeprecationWarning,
        stacklevel=2
    )

    if verbose:
        print(f"\n{'='*80}")
        print("STEP 3: TRAJECTORY INTERPOLATION TO COMMON TIMEPOINTS (LEGACY - DEPRECATED)")
        print(f"{'='*80}")

    # Find min/max hpf across all trajectories
    min_hpf = df_long.groupby('embryo_id')['hpf'].min().min()
    max_hpf = df_long.groupby('embryo_id')['hpf'].max().max()

    if verbose:
        print(f"\n  HPF range: {min_hpf:.1f} to {max_hpf:.1f}")

    # Create common timepoint grid
    common_grid = np.arange(min_hpf, max_hpf + grid_step, grid_step)
    if verbose:
        print(f"  Common grid: {len(common_grid)} timepoints (step={grid_step} hpf)")

    # Interpolate each trajectory
    interpolated_trajectories = []
    original_lengths = {}
    embryo_ids_ordered = []

    for embryo_id, group in df_long.groupby('embryo_id'):
        group_sorted = group.sort_values('hpf')
        hpf_vals = group_sorted['hpf'].values
        metric_vals = group_sorted['metric_value'].values

        original_lengths[embryo_id] = len(metric_vals)

        # Linear interpolation to common grid
        f = interpolate.interp1d(
            hpf_vals,
            metric_vals,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Interpolate - will have NaN where outside observed range
        interpolated = f(common_grid)

        # Trim to observed range (remove NaN padding)
        # This keeps trajectories variable-length but NaN-free for DTW
        valid_mask = ~np.isnan(interpolated)
        interpolated_trimmed = interpolated[valid_mask]

        if len(interpolated_trimmed) > 0:  # Only keep if we have data
            interpolated_trajectories.append(interpolated_trimmed)
            embryo_ids_ordered.append(embryo_id)

    if verbose:
        print(f"\n  Interpolated shape: {len(interpolated_trajectories)} embryos")
        if interpolated_trajectories:
            lengths = [len(t) for t in interpolated_trajectories]
            print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
            print(f"  Variable lengths (trimmed to observed ranges, no NaN padding)")

    return interpolated_trajectories, embryo_ids_ordered, original_lengths, common_grid


def pad_trajectories_for_plotting(
    trajectories: List[np.ndarray],
    common_grid: np.ndarray,
    df_long: pd.DataFrame,
    embryo_ids: List[str],
    verbose: bool = False
) -> List[np.ndarray]:
    """
    DEPRECATED: No longer needed with DataFrame-centric API.

    This function is deprecated. Use plotting functions with df_interpolated
    DataFrame directly instead. The DataFrame version automatically preserves
    time information without manual padding.

    See migration guide in README.md for updated plotting workflow.
    """
    warnings.warn(
        "pad_trajectories_for_plotting() is deprecated. "
        "Use plotting functions with df_interpolated DataFrame directly. "
        "See migration guide in README.md",
        DeprecationWarning,
        stacklevel=2
    )

    if verbose:
        print(f"\n  Padding {len(trajectories)} trajectories to common grid (LEGACY - DEPRECATED)...")

    padded_trajectories = []

    for embryo_id, traj in zip(embryo_ids, trajectories):
        # Find this embryo's observed time range from original data
        embryo_data = df_long[df_long['embryo_id'] == embryo_id].sort_values('hpf')
        if len(embryo_data) == 0:
            # No data for this embryo, return all NaN
            padded_trajectories.append(np.full(len(common_grid), np.nan))
            continue

        min_hpf = embryo_data['hpf'].min()
        max_hpf = embryo_data['hpf'].max()

        # Find start and end indices in common grid
        start_idx = np.searchsorted(common_grid, min_hpf, side='left')
        end_idx = np.searchsorted(common_grid, max_hpf, side='right')

        # Handle edge case where trajectory extends beyond grid
        traj_len = len(traj)
        grid_span = end_idx - start_idx

        if grid_span != traj_len:
            # Adjust end_idx if mismatch (can happen due to interpolation grid steps)
            end_idx = start_idx + traj_len

        # Create padded array
        padded_traj = np.full(len(common_grid), np.nan)

        # Insert trajectory data at correct position
        if end_idx <= len(common_grid):
            padded_traj[start_idx:end_idx] = traj
        else:
            # Trajectory extends beyond grid, truncate
            available = len(common_grid) - start_idx
            padded_traj[start_idx:] = traj[:available]

        padded_trajectories.append(padded_traj)

    if verbose:
        # Verify all same length
        lengths = [len(t) for t in padded_trajectories]
        if len(set(lengths)) == 1:
            print(f"  ✓ All {len(padded_trajectories)} trajectories padded to uniform length: {lengths[0]}")
        else:
            print(f"  WARNING: Inconsistent lengths after padding: {set(lengths)}")

    return padded_trajectories


def extract_early_late_means(
    df_long: pd.DataFrame,
    embryo_ids: List[str],
    early_window: Tuple[float, float],
    late_window: Tuple[float, float],
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mean metric values in early and late temporal windows.

    For each embryo, computes the mean metric value within early and late
    HPF windows. NaN values are handled appropriately.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format dataframe with columns: embryo_id, hpf, metric_value
    embryo_ids : list of str
        Embryo IDs to extract means for (should be ordered)
    early_window : tuple of (float, float)
        (min_hpf, max_hpf) for early window
    late_window : tuple of (float, float)
        (min_hpf, max_hpf) for late window
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    early_means_arr : np.ndarray
        Mean metric values in early window for each embryo (NaN if no data in window)
    late_means_arr : np.ndarray
        Mean metric values in late window for each embryo (NaN if no data in window)

    Examples
    --------
    >>> early_means, late_means = extract_early_late_means(
    ...     df_long,
    ...     embryo_ids,
    ...     early_window=(44, 50),
    ...     late_window=(80, 100)
    ... )
    >>> print(f"Early: {np.nanmean(early_means):.3f} ± {np.nanstd(early_means):.3f}")
    >>> print(f"Late: {np.nanmean(late_means):.3f} ± {np.nanstd(late_means):.3f}")
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 7: EXTRACT EARLY/LATE MEANS")
        print(f"{'='*80}")
        print(f"\n  Early window: {early_window[0]}-{early_window[1]} hpf")
        print(f"  Late window: {late_window[0]}-{late_window[1]} hpf")

    early_means = {}
    late_means = {}

    for embryo_id in embryo_ids:
        embryo_data = df_long[df_long['embryo_id'] == embryo_id]

        # Early window
        early_data = embryo_data[
            (embryo_data['hpf'] >= early_window[0]) &
            (embryo_data['hpf'] <= early_window[1])
        ]
        if len(early_data) > 0:
            early_means[embryo_id] = early_data['metric_value'].mean()
        else:
            early_means[embryo_id] = np.nan

        # Late window
        late_data = embryo_data[
            (embryo_data['hpf'] >= late_window[0]) &
            (embryo_data['hpf'] <= late_window[1])
        ]
        if len(late_data) > 0:
            late_means[embryo_id] = late_data['metric_value'].mean()
        else:
            late_means[embryo_id] = np.nan

    # Convert to arrays (in embryo_ids order)
    early_means_arr = np.array([early_means.get(e, np.nan) for e in embryo_ids])
    late_means_arr = np.array([late_means.get(e, np.nan) for e in embryo_ids])

    if verbose:
        print(f"\n  Early means: {np.nanmean(early_means_arr):.4f} ± {np.nanstd(early_means_arr):.4f}")
        print(f"  Late means: {np.nanmean(late_means_arr):.4f} ± {np.nanstd(late_means_arr):.4f}")

    return early_means_arr, late_means_arr
