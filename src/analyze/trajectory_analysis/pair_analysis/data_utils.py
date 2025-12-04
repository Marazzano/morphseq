"""
Data utilities for pair analysis.

Functions for extracting trajectories and computing statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.ndimage import gaussian_filter1d


def get_trajectories_for_group(
    df: pd.DataFrame,
    filter_dict: Dict[str, Any],
    time_col: str = 'predicted_stage_hpf',
    metric_col: str = 'baseline_deviation_normalized',
    embryo_id_col: str = 'embryo_id',
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[Dict]], Optional[np.ndarray], int]:
    """Extract trajectories for a specific group defined by filter conditions.

    Args:
        df: DataFrame with trajectory data
        filter_dict: Dict of {column: value} pairs to filter on
        time_col: Column name for time values
        metric_col: Column name for metric values
        embryo_id_col: Column name for embryo IDs
        smooth_method: Smoothing method ('gaussian' or None for no smoothing).
            Default: 'gaussian'
        smooth_params: Parameters for smoothing. Defaults:
            - gaussian: {'sigma': 1.5}
            - None: no smoothing applied (raw data)

    Returns:
        Tuple of (trajectories, embryo_ids, n_embryos)
        - trajectories: List of dicts with 'embryo_id', 'times', 'metrics'
        - embryo_ids: Array of unique embryo IDs
        - n_embryos: Number of embryos

    Example:
        trajectories, ids, n = get_trajectories_for_group(
            df,
            {'pair': 'cep290_pair_1', 'genotype': 'cep290_homozygous'},
            time_col='predicted_stage_hpf',
            metric_col='baseline_deviation_normalized',
            smooth_method='gaussian',
            smooth_params={'sigma': 1.5}
        )
    """
    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filter_dict.items():
        mask &= (df[col] == val)

    filtered = df[mask].copy()

    if len(filtered) == 0:
        return None, None, 0

    # Set default smoothing parameters
    if smooth_params is None:
        if smooth_method == 'gaussian':
            smooth_params = {'sigma': 1.5}
        else:
            smooth_params = {}

    # Group by embryo and get trajectories
    embryo_ids = filtered[embryo_id_col].unique()
    trajectories = []

    for embryo_id in embryo_ids:
        embryo_data = filtered[filtered[embryo_id_col] == embryo_id].sort_values(time_col)
        if len(embryo_data) > 1:
            times = embryo_data[time_col].values
            metrics = embryo_data[metric_col].values

            # Apply Gaussian smoothing if requested
            if smooth_method == 'gaussian':
                sigma = smooth_params.get('sigma', 1.5)
                metrics = gaussian_filter1d(metrics, sigma=sigma)
            # else: use raw data (no smoothing)

            trajectories.append({
                'embryo_id': embryo_id,
                'times': times,
                'metrics': metrics,
            })

    return trajectories, embryo_ids, len(trajectories)


def compute_binned_mean(
    times: np.ndarray,
    values: np.ndarray,
    bin_width: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """Compute binned mean of values over time.

    Args:
        times: Array of time values
        values: Array of metric values
        bin_width: Width of time bins (default 0.5 hpf)

    Returns:
        Tuple of (bin_times, bin_means) as lists
    """
    if len(times) == 0 or len(values) == 0:
        return [], []

    time_bins = np.arange(np.floor(times.min()), np.ceil(times.max()) + bin_width, bin_width)
    bin_means = []
    bin_times = []

    for i in range(len(time_bins) - 1):
        mask = (times >= time_bins[i]) & (times < time_bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(values[mask].mean())
            bin_times.append((time_bins[i] + time_bins[i + 1]) / 2)

    return bin_times, bin_means


def get_global_axis_ranges(
    all_trajectories: List[List[Dict]],
    padding_fraction: float = 0.1,
) -> Tuple[float, float, float, float]:
    """Compute global axis ranges from multiple trajectory lists.

    Args:
        all_trajectories: List of trajectory lists (each from get_trajectories_for_group)
        padding_fraction: Fraction of range to add as padding

    Returns:
        Tuple of (time_min, time_max, metric_min, metric_max)
    """
    time_min, time_max = float('inf'), float('-inf')
    metric_min, metric_max = float('inf'), float('-inf')

    for trajectories in all_trajectories:
        if trajectories is None:
            continue
        for traj in trajectories:
            time_min = min(time_min, traj['times'].min())
            time_max = max(time_max, traj['times'].max())
            metric_min = min(metric_min, traj['metrics'].min())
            metric_max = max(metric_max, traj['metrics'].max())

    # Add padding
    if metric_max > metric_min:
        padding = (metric_max - metric_min) * padding_fraction
        metric_min -= padding
        metric_max += padding

    return time_min, time_max, metric_min, metric_max
