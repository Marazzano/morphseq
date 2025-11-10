"""
Data Loading Module

MVP data loading layer that is:
- Simple and explicit (no magic)
- Every step visible and debuggable
- Format-isolated (CSV changes only affect internal functions)
- Returns DataFrames until final step

Workflow:
  1. load_experiment_dataframe() - Load raw CSVs
  2. extract_trajectory_dataframe() - Filter and extract long-format trajectories
  3. dataframe_to_trajectories() - Convert to arrays (raw, no interpolation)
  4. interpolate_trajectories() - Interpolate to common time grid
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


# =============================================================================
# INTERNAL: Format-Specific Loaders (Isolation Layer)
# =============================================================================

def _load_df03_format(experiment_id: str) -> pd.DataFrame:
    """
    Load df03 format CSVs (current format).

    Internal function. Isolates format-specific logic.
    Searches standard locations for curvature and metadata files.

    If CSV structure changes, only update this function.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., '20251017_combined')

    Returns
    -------
    pd.DataFrame
        Raw merged data with all original columns

    Raises
    ------
    FileNotFoundError
        If curvature or metadata files cannot be found
    """
    # Standard file locations (search order)
    project_root = Path(__file__).resolve().parents[3]

    # Try standard locations
    search_paths = [
        (project_root / 'morphseq_playground' / 'metadata' / 'body_axis' / 'summary',
         project_root / 'morphseq_playground' / 'metadata' / 'build06_output'),
        (project_root / 'morphseq_playground' / 'metadata' / 'body_axis' / 'summary',
         project_root / 'morphseq_playground' / 'metadata' / 'body_axis' / 'summary'),
    ]

    curv_path = None
    meta_path = None

    # Search for curvature file
    curv_patterns = [
        f'curvature_metrics_{experiment_id}.csv',
        f'{experiment_id}_curvature.csv',
    ]

    # Search for metadata file
    meta_patterns = [
        f'df03_final_output_with_latents_{experiment_id}.csv',
        f'{experiment_id}_metadata.csv',
    ]

    # Try to find files in search paths
    for curv_dir, meta_dir in search_paths:
        if curv_path is None:
            for pattern in curv_patterns:
                candidate = curv_dir / pattern
                if candidate.exists():
                    curv_path = candidate
                    break

        if meta_path is None:
            for pattern in meta_patterns:
                candidate = meta_dir / pattern
                if candidate.exists():
                    meta_path = candidate
                    break

        if curv_path and meta_path:
            break

    # If still not found, raise error with helpful message
    if not curv_path:
        available_curv = curv_patterns
        search_locs = [p[0] for p in search_paths]
        raise FileNotFoundError(
            f"Curvature data not found for experiment '{experiment_id}'\n"
            f"  Searched for: {available_curv}\n"
            f"  Searched in: {search_locs}\n"
            f"  Tip: Use load_experiment_dataframe(experiment_id) and provide full paths if needed"
        )

    if not meta_path:
        available_meta = meta_patterns
        search_locs = [p[1] for p in search_paths]
        raise FileNotFoundError(
            f"Metadata not found for experiment '{experiment_id}'\n"
            f"  Searched for: {available_meta}\n"
            f"  Searched in: {search_locs}\n"
            f"  Tip: Use load_experiment_dataframe(experiment_id) and provide full paths if needed"
        )

    print(f"  Loading curvature from: {curv_path}")
    print(f"  Loading metadata from: {meta_path}")

    df_curv = pd.read_csv(curv_path)
    df_meta = pd.read_csv(meta_path)

    print(f"    Curvature: {len(df_curv)} rows")
    print(f"    Metadata: {len(df_meta)} rows")

    # Merge on snip_id (common key in this data structure)
    if 'snip_id' in df_curv.columns and 'snip_id' in df_meta.columns:
        df_merged = df_curv.merge(df_meta, on='snip_id', how='inner')
        print(f"    Merged on 'snip_id': {len(df_merged)} rows")
    elif 'embryo_id' in df_curv.columns and 'embryo_id' in df_meta.columns:
        df_merged = df_curv.merge(df_meta, on='embryo_id', how='inner')
        print(f"    Merged on 'embryo_id': {len(df_merged)} rows")
    else:
        raise ValueError(
            f"Could not find common merge key. "
            f"Curvature columns: {list(df_curv.columns)} "
            f"Metadata columns: {list(df_meta.columns)}"
        )

    if len(df_merged) == 0:
        raise ValueError("Merge resulted in empty dataframe - no matching records")

    return df_merged


def _load_df04_format(experiment_id: str) -> pd.DataFrame:
    """
    Load df04 format CSVs (future format placeholder).

    Internal function. Isolates format-specific logic for future formats.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier

    Returns
    -------
    pd.DataFrame
        Raw merged data with all original columns
    """
    # TODO: Implement when df04 format is available
    raise NotImplementedError(
        "df04 format not yet implemented. "
        "Please implement _load_df04_format() when df04 data is available."
    )


def _load_raw_data(experiment_id: str, format_version: str) -> pd.DataFrame:
    """
    Route to format-specific loader.

    This is the only place that knows about different data formats.
    All format-specific logic is delegated to _load_df*_format() functions.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier
    format_version : str
        Data format version ('df03', 'df04', etc.)

    Returns
    -------
    pd.DataFrame
        Raw data from specified format
    """
    if format_version == 'df03':
        return _load_df03_format(experiment_id)
    elif format_version == 'df04':
        return _load_df04_format(experiment_id)
    else:
        raise ValueError(
            f"Unknown format: {format_version}. "
            f"Supported formats: 'df03', 'df04'"
        )


# =============================================================================
# PUBLIC API
# =============================================================================

def load_experiment_dataframe(
    experiment_id: str,
    format_version: str = 'df03'
) -> pd.DataFrame:
    """
    Load raw experiment data from CSV files.

    Loads all original columns without any filtering or processing.
    This is Step 1 of the data loading pipeline.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., 'cep290_run1')
    format_version : str, default='df03'
        Data format version to load ('df03', 'df04', etc.)

    Returns
    -------
    pd.DataFrame
        Raw experiment data with all original columns.
        Typical columns: embryo_id, predicted_stage_hpf,
        normalized_baseline_deviation, genotype, etc.

    Examples
    --------
    >>> df_raw = load_experiment_dataframe('cep290_run1')
    >>> print(df_raw.shape)
    (5000, 281)
    >>> print(df_raw.columns)
    Index(['embryo_id', 'predicted_stage_hpf', 'normalized_baseline_deviation', ...])
    """
    return _load_raw_data(experiment_id, format_version)


def extract_trajectory_dataframe(
    df: pd.DataFrame,
    embryo_id_col: str = 'embryo_id',
    time_col: str = 'predicted_stage_hpf',
    metric_col: str = 'normalized_baseline_deviation',
    keep_cols: Optional[List[str]] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    min_timepoints: int = 3
) -> pd.DataFrame:
    """
    Extract trajectory data in long format from raw DataFrame.

    Filters data, selects columns, and returns long-format trajectory data
    ready for downstream analysis. This is Step 2 of the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment DataFrame (from load_experiment_dataframe)
    embryo_id_col : str, default='embryo_id'
        Column name for embryo identifiers
    time_col : str, default='predicted_stage_hpf'
        Column name for time points
    metric_col : str, default='normalized_baseline_deviation'
        Column name for metric values to extract
    keep_cols : list of str, optional
        Additional metadata columns to keep in output.
        Default: ['genotype', 'predicted_stage_hpf']
        Example: ['genotype', 'replicate', 'well_id']
    filter_dict : dict, optional
        Filtering criteria. Example: {'genotype': 'wildtype', 'stage': 'early'}
        If None, no filtering is applied.
    min_timepoints : int, default=3
        Minimum number of timepoints required per embryo.
        Embryos with fewer timepoints are excluded.

    Returns
    -------
    pd.DataFrame
        Long-format trajectory data with columns:
        - embryo_id: Embryo identifier
        - time: Time points (renamed from time_col)
        - metric_value: Metric values (renamed from metric_col)
        - [additional columns from keep_cols]

        All NaN values are retained (for later handling during interpolation).
        Data is NOT interpolated yet.

    Examples
    --------
    >>> df_raw = load_experiment_dataframe('cep290_run1')
    >>> df_traj = extract_trajectory_dataframe(
    ...     df_raw,
    ...     filter_dict={'genotype': 'wildtype'}
    ... )
    >>> print(df_traj.columns)
    Index(['embryo_id', 'time', 'metric_value', 'genotype', 'predicted_stage_hpf'])
    >>> print(len(df_traj))
    1200  # ~24 timepoints × 50 embryos

    >>> # With custom metadata columns
    >>> df_traj = extract_trajectory_dataframe(
    ...     df_raw,
    ...     filter_dict={'genotype': 'wildtype'},
    ...     keep_cols=['genotype', 'replicate', 'well_id']
    ... )
    """
    # Default metadata columns
    if keep_cols is None:
        keep_cols = ['genotype', 'predicted_stage_hpf']

    # Apply filters
    df_filtered = df.copy()
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in df_filtered.columns:
                raise ValueError(f"Filter column '{col}' not found in data")
            df_filtered = df_filtered[df_filtered[col] == val]

    # Select columns: required + optional
    cols_to_keep = [embryo_id_col, time_col, metric_col]
    for col in keep_cols:
        if col not in cols_to_keep and col in df_filtered.columns:
            cols_to_keep.append(col)

    df_subset = df_filtered[cols_to_keep].copy()

    # Rename core columns for consistency
    df_subset = df_subset.rename(columns={
        time_col: 'time',
        metric_col: 'metric_value'
    })

    # Filter embryos with sufficient timepoints
    counts = df_subset.groupby(embryo_id_col).size()
    valid_embryos = counts[counts >= min_timepoints].index
    df_subset = df_subset[df_subset[embryo_id_col].isin(valid_embryos)].copy()

    # Reset index for cleanliness
    df_subset = df_subset.reset_index(drop=True)

    return df_subset


def dataframe_to_trajectories(
    df_traj: pd.DataFrame,
    embryo_id_col: str = 'embryo_id',
    sort_by: str = 'time'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Convert long-format trajectory DataFrame to arrays.

    Extracts per-embryo time points and metric values as separate arrays.
    NO interpolation is performed - returns raw data as-is.
    This is Step 3 of the pipeline.

    Parameters
    ----------
    df_traj : pd.DataFrame
        Long-format trajectory data (from extract_trajectory_dataframe)
        Must contain: embryo_id, time, metric_value columns
    embryo_id_col : str, default='embryo_id'
        Column name for embryo identifiers
    sort_by : str, default='time'
        Column to sort by before extracting arrays

    Returns
    -------
    time_arrays : list of np.ndarray
        Per-embryo time points. Variable length (one array per embryo).
        Example: [array([10.0, 10.5, 11.0, ...]), array([10.0, 10.5, ...])]
    metric_arrays : list of np.ndarray
        Per-embryo metric values. Variable length, aligned with time_arrays.
        Example: [array([0.23, 0.25, 0.27, ...]), array([0.15, 0.16, ...])]
    embryo_ids : list of str
        Embryo identifiers in same order as arrays.
        Example: ['embryo_01', 'embryo_02', ...]

    Notes
    -----
    - Arrays have variable lengths (different embryos may have different numbers of timepoints)
    - Missing values (NaN) are preserved in metric_arrays
    - Order is preserved: time_arrays[i] and metric_arrays[i] correspond to embryo_ids[i]

    Examples
    --------
    >>> df_traj = extract_trajectory_dataframe(df_raw)
    >>> time_arrays, metric_arrays, embryo_ids = dataframe_to_trajectories(df_traj)
    >>> # Each embryo has variable-length arrays
    >>> len(time_arrays[0])  # 23 timepoints for embryo 0
    23
    >>> len(time_arrays[1])  # 22 timepoints for embryo 1
    22
    >>> # All three lists aligned
    >>> len(time_arrays) == len(metric_arrays) == len(embryo_ids)
    True
    """
    # Get unique embryo IDs in order
    embryo_ids = df_traj[embryo_id_col].unique().tolist()

    time_arrays = []
    metric_arrays = []

    for embryo_id in embryo_ids:
        # Extract data for this embryo
        embryo_df = df_traj[df_traj[embryo_id_col] == embryo_id].copy()

        # Sort by specified column
        embryo_df = embryo_df.sort_values(sort_by)

        # Extract as numpy arrays
        time_arr = embryo_df['time'].values
        metric_arr = embryo_df['metric_value'].values

        time_arrays.append(time_arr)
        metric_arrays.append(metric_arr)

    return time_arrays, metric_arrays, embryo_ids


def interpolate_trajectories(
    time_arrays: List[np.ndarray],
    metric_arrays: List[np.ndarray],
    embryo_ids: List[str],
    grid_step: float = 0.5
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Interpolate trajectories to common time grid.

    Takes variable-length per-embryo trajectories and interpolates them to a
    common time grid so all trajectories have the same length and alignment.
    This is Step 4 of the pipeline.

    Parameters
    ----------
    time_arrays : list of np.ndarray
        Per-embryo time points (variable length, from dataframe_to_trajectories)
    metric_arrays : list of np.ndarray
        Per-embryo metric values (variable length, must align with time_arrays)
    embryo_ids : list of str
        Embryo identifiers (in same order as arrays)
    grid_step : float, default=0.5
        Step size for common time grid (in same units as time_arrays)
        Example: 0.5 HPF means grid points at [10.0, 10.5, 11.0, ...]

    Returns
    -------
    time_grid : np.ndarray
        Common time grid spanning min to max of all input time points
        Example: array([10.0, 10.5, 11.0, ..., 23.5, 24.0])
    traj_grid : list of np.ndarray
        Trajectories interpolated to time_grid (all same length now)
        Missing/extrapolated values are NaN
    embryo_ids : list of str
        Embryo IDs in same order (returned unchanged)

    Notes
    -----
    - Linear interpolation used
    - Values outside time range set to NaN (extrapolation not performed)
    - All returned trajectories have same length (len(time_grid))

    Examples
    --------
    >>> time_arrays, metric_arrays, embryo_ids = dataframe_to_trajectories(df_traj)
    >>> time_grid, traj_grid, embryo_ids = interpolate_trajectories(
    ...     time_arrays, metric_arrays, embryo_ids, grid_step=0.5
    ... )
    >>> # Now all trajectories aligned to same grid
    >>> all(len(t) == len(time_grid) for t in traj_grid)
    True
    >>> # Passing to DTW analysis
    >>> D = compute_dtw_distance_matrix(traj_grid)
    """
    # Find global time range across all trajectories
    all_times = np.concatenate(time_arrays)
    min_time = np.nanmin(all_times)
    max_time = np.nanmax(all_times)

    # Create common time grid
    time_grid = np.arange(min_time, max_time + grid_step, grid_step)

    # Interpolate each trajectory to common grid
    traj_grid = []
    for time_arr, metric_arr in zip(time_arrays, metric_arrays):
        # Linear interpolation
        # NaN outside of time range (left=np.nan, right=np.nan)
        interpolated = np.interp(
            time_grid,
            time_arr,
            metric_arr,
            left=np.nan,
            right=np.nan
        )
        traj_grid.append(interpolated)

    return time_grid, traj_grid, embryo_ids
