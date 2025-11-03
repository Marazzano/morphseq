"""
Trajectory Processing Utilities

Functions for extracting, interpolating, and aligning temporal trajectories.

This module provides utilities for converting long-format temporal data into
per-embryo trajectories, handling missing values, and aligning trajectories
to common timepoint grids for subsequent analysis (e.g., DTW clustering).

Functions
=========
- extract_trajectories : Extract per-embryo trajectories from long-format data
- interpolate_trajectories : Handle missing values via linear interpolation
- interpolate_to_common_grid : Align trajectories to common timepoint grid
- extract_early_late_means : Extract mean values in temporal windows
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Tuple, List, Dict, Optional


def extract_trajectories(
    df: pd.DataFrame,
    genotype_filter: Optional[str] = None,
    metric_name: str = 'normalized_baseline_deviation',
    min_timepoints: int = 3,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[str], pd.DataFrame]:
    """
    Extract per-embryo trajectories from long-format dataframe.

    Converts long-format data (embryo_id, predicted_stage_hpf, metric_value)
    into a list of per-embryo trajectories. Optionally filters by genotype
    and requires minimum number of timepoints per embryo.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: embryo_id, predicted_stage_hpf, metric
    genotype_filter : str, optional
        If specified, filter dataframe to only this genotype
    metric_name : str, default='normalized_baseline_deviation'
        Name of the metric column to extract as trajectory values
    min_timepoints : int, default=3
        Minimum number of non-NaN timepoints required per embryo
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    trajectories : list of np.ndarray
        List of trajectory arrays (variable lengths, sorted by hpf)
    embryo_ids : list of str
        Embryo IDs corresponding to trajectories
    trajectories_df : pd.DataFrame
        Long-format dataframe used for extraction (with NaN removed)
        Columns: embryo_id, hpf, metric_value

    Examples
    --------
    >>> trajectories, embryo_ids, df_long = extract_trajectories(
    ...     df,
    ...     genotype_filter='cep290_homozygous',
    ...     metric_name='curvature'
    ... )
    >>> print(f"Extracted {len(trajectories)} embryos")
    >>> print(f"Mean trajectory length: {np.mean([len(t) for t in trajectories]):.1f}")
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
    grid_step: float = 0.5,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[str], Dict[str, int], np.ndarray]:
    """
    Interpolate all trajectories to a common timepoint grid.

    Aligns variable-length trajectories to a regular grid of timepoints.
    Trajectories are truncated to their observed range (no edge padding).

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format dataframe with columns: embryo_id, hpf, metric_value
    grid_step : float, default=0.5
        Step size for timepoint grid (in HPF units)
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    interpolated_trajectories : list of np.ndarray
        Trajectories at common timepoint grid (variable lengths due to truncation)
    embryo_ids_ordered : list of str
        Embryo IDs in same order as interpolated_trajectories
    original_lengths : dict
        Original trajectory lengths {embryo_id: n_timepoints}
    common_grid : np.ndarray
        Common timepoint grid (HPF values)

    Notes
    -----
    - Common grid spans from global min_hpf to max_hpf across all embryos
    - Each trajectory is truncated to its observed range
    - Trajectories with no valid data are excluded

    Examples
    --------
    >>> interp_trajs, embryo_ids, orig_lens, grid = interpolate_to_common_grid(
    ...     df_long, grid_step=0.5
    ... )
    >>> print(f"Grid size: {len(grid)}")
    >>> print(f"HPF range: {grid[0]:.1f} to {grid[-1]:.1f}")
    """
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 3: TRAJECTORY INTERPOLATION TO COMMON TIMEPOINTS")
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

        # Only keep timepoints within observed range (no padding)
        valid_mask = ~np.isnan(interpolated)
        interpolated_trimmed = interpolated[valid_mask]

        if len(interpolated_trimmed) > 0:  # Only keep if we have data
            interpolated_trajectories.append(interpolated_trimmed)
            embryo_ids_ordered.append(embryo_id)

    if verbose:
        print(f"\n  Interpolated shape: {len(interpolated_trajectories)} embryos")
        if interpolated_trajectories:
            print(f"  Interpolated lengths: min={min([len(t) for t in interpolated_trajectories])}, "
                  f"max={max([len(t) for t in interpolated_trajectories])}, "
                  f"mean={np.mean([len(t) for t in interpolated_trajectories]):.1f}")

    return interpolated_trajectories, embryo_ids_ordered, original_lengths, common_grid


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
