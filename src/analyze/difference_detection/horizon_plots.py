"""
Horizon plot utilities for visualizing time-series correlation and comparison matrices.

This module provides reusable functions for creating "horizon plots" — grids of heatmaps
that compare metrics across multiple conditions, timepoints, or model predictions.
Originally extracted from results/mcolon/20251020/compare_3models_full_time_matrix.py
to support both model comparisons and curvature temporal analysis.

Key Functions
=============
- plot_horizon_grid() : Create N×M grid of synchronized heatmaps
- plot_single_horizon() : Create a single customized heatmap
- plot_best_condition_map() : Show which condition performs best per cell
- compute_shared_colorscale() : Determine min/max values with optional percentile clipping

Example Usage
=============
>>> from analyze.difference_detection.horizon_plots import plot_horizon_grid
>>> matrices = {'WT': {'metric': df1}, 'Het': {'metric': df2}}
>>> plot_horizon_grid(
...     matrices,
...     row_labels=['WT Model', 'Het Model'],
...     col_labels=['WT Test', 'Het Test'],
...     metric='mae',
...     cmap='viridis'
... )

Notes
-----
- Supports shared colorscales across multiple heatmaps for better visual comparison
- Can highlight special cells (e.g., LOEO cross-validation with red borders)
- Percentile clipping improves dynamic range when data has outliers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def compute_shared_colorscale(
    matrices: Dict,
    metric: Optional[str] = None,
    clip_percentiles: Optional[Tuple[float, float]] = (5, 95),
    valid_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    Compute min/max colorscale across all matrices.

    Useful for ensuring consistent color representation when comparing multiple
    heatmaps in a grid.

    Parameters
    ----------
    matrices : dict
        Nested dict structure where leaves are pandas DataFrames or arrays
    metric : str, optional
        If provided, only aggregate values from this metric column
    clip_percentiles : tuple of float, optional
        (lower, upper) percentiles to use for clipping. Default (5, 95) clips outliers.
        Set to None to use full data range.
    valid_range : tuple of float, optional
        If provided, constrain vmin/vmax to [lower, upper]. Useful for
        metrics like R² that should stay in [-1, 1].

    Returns
    -------
    vmin, vmax : float
        Recommended colorscale bounds
    """
    # Placeholder: Logic to be implemented
    pass


def plot_horizon_grid(
    matrices: Dict[str, Dict[str, pd.DataFrame]],
    row_labels: List[str],
    col_labels: List[str],
    metric: str = 'mae',
    cmap: str = 'viridis',
    clip_percentiles: Optional[Tuple[float, float]] = (5, 95),
    annotate: bool = False,
    loeo_highlight: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> Figure:
    """
    Create N×M grid of heatmaps comparing multiple conditions or models.

    This is the main high-level plotting function. Automatically:
    - Computes shared colorscale across all panels
    - Aligns axes labels and ticks
    - Adds optional LOEO (Leave-One-Embryo-Out) highlighting
    - Handles missing data gracefully

    Parameters
    ----------
    matrices : dict of dict of DataFrame
        {row_key: {col_key: matrix_df}}
        Each matrix_df should be 2D with time indices (start_time × target_time)
    row_labels : list of str
        Labels for rows (len must match number of row_keys)
    col_labels : list of str
        Labels for columns (len must match number of col_keys)
    metric : str
        Which metric to visualize ('mae', 'r2', 'error_std', etc.)
    cmap : str
        Matplotlib colormap name ('viridis', 'RdYlGn', etc.)
    clip_percentiles : tuple or None
        (lower, upper) percentiles for dynamic range clipping. None = use full range.
    annotate : bool
        If True, show numerical values in cells (slow for large matrices)
    loeo_highlight : dict, optional
        {row_key: col_key} pairs to highlight with red borders
        Indicates which cells use LOEO cross-validation
    title : str, optional
        Main title for the figure
    figsize : tuple, optional
        (width, height) in inches
    save_path : str or Path, optional
        If provided, save figure to this path
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure

    Raises
    ------
    ValueError
        If dimensions don't match (len(row_labels) != num rows, etc.)
    """
    # Placeholder: Logic to be implemented
    pass


def plot_single_horizon(
    matrix: pd.DataFrame,
    metric: str = 'mae',
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = 'Target Time (hpf)',
    ylabel: str = 'Start Time (hpf)',
    annotate: bool = False
) -> Axes:
    """
    Create a single heatmap with customization options.

    Parameters
    ----------
    matrix : pd.DataFrame
        2D matrix to visualize
    metric : str
        Metric name for colorbar label
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale bounds. If None, auto-compute from data.
    ax : Axes, optional
        Existing axes to draw on. If None, creates new figure.
    title : str, optional
        Heatmap title
    xlabel, ylabel : str
        Axis labels
    annotate : bool
        Show numerical values in cells

    Returns
    -------
    ax : Axes
    """
    # Placeholder: Logic to be implemented
    pass


def plot_best_condition_map(
    matrices: Dict[str, Dict[str, pd.DataFrame]],
    row_labels: List[str],
    col_labels: List[str],
    metric: str = 'mae',
    mode: str = 'min',
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> Figure:
    """
    Show which row condition performs best at each cell.

    For example, given model predictions (WT, Het, Homo) on different test
    genotypes, show which model has the lowest MAE (or highest R²) at each
    timepoint pair.

    Parameters
    ----------
    matrices : dict of dict of DataFrame
        {row_key: {col_key: matrix_df}}
    row_labels : list of str
        Names of conditions/models to compare
    col_labels : list of str
        Names of test sets/genotypes
    metric : str
        Which metric column ('mae', 'r2', etc.)
    mode : {'min', 'max'}
        'min' = lowest is best (MAE, error), 'max' = highest is best (R²)
    title : str, optional
    figsize : tuple, optional
    save_path : str or Path, optional
    dpi : int

    Returns
    -------
    fig : Figure
        Contains one subplot per column (col_label)

    Notes
    -----
    Each cell in output matrix shows an index (0, 1, 2, ...) indicating which
    row condition is best at that cell.
    """
    # Placeholder: Logic to be implemented
    pass


# ============================================================================
# Helper functions (internal use, not exported)
# ============================================================================

def _determine_color_scale(
    all_values: np.ndarray,
    metric: str,
    clip_percentiles: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    Determine appropriate vmin/vmax for a metric.

    Handles metric-specific logic:
    - R² is constrained to [-1, 1]
    - MAE/error_std are constrained to [0, ∞)
    - Percentile clipping can improve dynamic range

    Parameters
    ----------
    all_values : np.ndarray
        Flattened array of metric values (may contain NaN)
    metric : str
        Metric name ('mae', 'r2', 'error_std')
    clip_percentiles : tuple or None
        (lower, upper) percentiles to use

    Returns
    -------
    vmin, vmax : float
    """
    # Placeholder: Logic to be implemented
    pass


def _add_loeo_highlight(ax: Axes, is_loeo: bool, linewidth: float = 3) -> None:
    """
    Add red border to axes if this cell uses LOEO cross-validation.

    Parameters
    ----------
    ax : Axes
        The axes to modify
    is_loeo : bool
        Whether this cell should be highlighted
    linewidth : float
        Border width in points
    """
    # Placeholder: Logic to be implemented
    pass


def _format_tick_labels(
    ax: Axes,
    is_bottom_row: bool,
    is_left_col: bool
) -> None:
    """
    Format axis labels and ticks for consistent appearance across grid.

    Shows axis labels only on borders to reduce clutter.

    Parameters
    ----------
    ax : Axes
    is_bottom_row : bool
    is_left_col : bool
    """
    # Placeholder: Logic to be implemented
    pass


if __name__ == '__main__':
    # Demonstration / testing goes here
    pass
