"""
Trajectory plotting functions for faceted layouts.

Contains:
- plot_trajectories_faceted: Standard faceted plot for single metric
- plot_multimetric_trajectories: Multi-metric plot (rows=metrics, cols=groups)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

from ....pair_analysis.data_utils import (
    get_trajectories_for_group,
    get_global_axis_ranges,
)
from ....config import (
    DEFAULT_PLOTLY_HEIGHT,
    DEFAULT_PLOTLY_WIDTH,
    HEIGHT_PER_ROW,
    WIDTH_PER_COL,
)

from .shared import (
    FigureData,
    SubplotData,
    create_color_state,
    validate_error_type,
    _compile_subplot_traces,
    _render_figure,
)


def plot_trajectories_faceted(
    df: pd.DataFrame,
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    color_by_grouping: Optional[str] = None,
    color_palette: Optional[List[str]] = None,
    facet_order: Optional[Dict[str, List]] = None,
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: str = 'Time (hpf)',
    y_label: str = 'Value',
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    trend_statistic: str = 'median',
    trend_smooth_sigma: Optional[float] = 1.5,
    time_col: Optional[str] = None,  # Compat
    metric_col: Optional[str] = None,  # Compat
    # Error band options
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    error_band_alpha: float = 0.2,
    # Linear fit option
    show_linear_fit: bool = False,
) -> Any:
    """
    Standard faceted plot (rows=row_by, cols=col_by) for a SINGLE metric.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing trajectories
    x_col : str
        Column for x-axis (time)
    y_col : str
        Column for y-axis (metric)
    line_by : str
        Column identifying individual trajectories (e.g., 'embryo_id')
    row_by : Optional[str]
        Column to facet by rows
    col_by : Optional[str]
        Column to facet by columns
    color_by_grouping : Optional[str]
        Column to color trend lines by
    color_palette : Optional[List[str]]
        Custom color palette (list of hex colors). If None, uses STANDARD_PALETTE.
    show_individual : bool, default=True
        Whether to show individual trajectory traces
    show_error_band : bool, default=False
        Whether to show error band around trend line
    error_type : str, default='iqr'
        Error measure: 'sd'/'se' for mean, 'iqr'/'mad' for median
    error_band_alpha : float, default=0.2
        Transparency of error band
    show_linear_fit : bool, default=False
        Whether to overlay linear regression on trend line

    Returns
    -------
    Figure
        Plotly or matplotlib figure
    """
    x_col = time_col or x_col
    y_col = metric_col or y_col

    # Validate error_type if error band is enabled
    if show_error_band:
        validate_error_type(trend_statistic, error_type)

    def _sorted_unique(values: pd.Series) -> list:
        # Robust sort for mixed object dtypes (e.g., strings + NaN floats).
        uniques = list(values.unique())
        return sorted(uniques, key=lambda v: (pd.isna(v), str(v)))

    row_values = _sorted_unique(df[row_by]) if row_by else [None]
    col_values = _sorted_unique(df[col_by]) if col_by else [None]

    if facet_order:
        if row_by and row_by in facet_order:
            row_values = [v for v in facet_order[row_by] if v in row_values]
        if col_by and col_by in facet_order:
            col_values = [v for v in facet_order[col_by] if v in col_values]

    # Global ranges
    all_trajs, _, _ = get_trajectories_for_group(df, {}, x_col, y_col, line_by, smooth_method, smooth_params)
    t_min, t_max, m_min, m_max = get_global_axis_ranges([all_trajs] if all_trajs else [])

    subplots = []
    color_state = create_color_state(df, col_by, color_by_grouping, line_by, color_palette)
    legend_tracker = set()

    for r_idx, row_val in enumerate(row_values):
        for c_idx, col_val in enumerate(col_values):
            filter_dict = {}
            if row_by: filter_dict[row_by] = row_val
            if col_by: filter_dict[col_by] = col_val

            traces = _compile_subplot_traces(
                df, filter_dict, x_col, y_col, line_by, color_state,
                col_val, row_val, color_by_grouping, col_by, row_by,
                bin_width, smooth_method, smooth_params, y_label, legend_tracker,
                trend_statistic, trend_smooth_sigma,
                show_individual=show_individual,
                show_error_band=show_error_band,
                error_type=error_type,
                error_band_alpha=error_band_alpha,
                show_linear_fit=show_linear_fit,
            )

            subplots.append(SubplotData(
                row=r_idx + 1,
                col=c_idx + 1,
                traces=traces,
                xlim=(t_min, t_max),
                ylim=(m_min, m_max),
                title=str(col_val) if (r_idx == 0 and col_by) else None,
                x_label=x_label if (r_idx == len(row_values) - 1) else None,
                y_label=y_label if (c_idx == 0) else None
            ))

    final_title = title or f"Trajectories {f'by {row_by}' if row_by else ''} {f'vs {col_by}' if col_by else ''}"
    fig_data = FigureData(
        title=final_title,
        n_rows=len(row_values),
        n_cols=len(col_values),
        height=max(DEFAULT_PLOTLY_HEIGHT, len(row_values) * height_per_row),
        width=max(DEFAULT_PLOTLY_WIDTH, len(col_values) * width_per_col),
        subplots=subplots,
        row_labels=[str(r) for r in row_values] if row_by else [],
        color_by_grouping=color_by_grouping,
    )

    return _render_figure(fig_data, backend, output_path)

def plot_multimetric_trajectories(
    df: pd.DataFrame,
    metrics: List[str],
    col_by: str,
    x_col: str = 'predicted_stage_hpf',
    line_by: str = 'embryo_id',
    color_by_grouping: Optional[str] = None,
    color_palette: Optional[List[str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    col_order: Optional[List] = None,
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: str = 'Time (hpf)',
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    trend_statistic: str = 'median',
    trend_smooth_sigma: Optional[float] = 1.5,
    # Error band options
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    error_band_alpha: float = 0.2,
    # Linear fit option
    show_linear_fit: bool = False,
) -> Any:
    """
    Multi-metric plot: Rows = Metrics, Cols = Groups.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing trajectories
    metrics : List[str]
        List of metric columns to plot (one per row)
    col_by : str
        Column to facet by columns
    color_palette : Optional[List[str]]
        Custom color palette (list of hex colors). If None, uses STANDARD_PALETTE.
    show_individual : bool, default=True
        Whether to show individual trajectory traces
    show_error_band : bool, default=False
        Whether to show error band around trend line
    error_type : str, default='iqr'
        Error measure: 'sd'/'se' for mean, 'iqr'/'mad' for median
    error_band_alpha : float, default=0.2
        Transparency of error band
    show_linear_fit : bool, default=False
        Whether to overlay linear regression on trend line

    Returns
    -------
    Figure
        Plotly or matplotlib figure
    """
    # Validate error_type if error band is enabled
    if show_error_band:
        validate_error_type(trend_statistic, error_type)

    row_values = metrics
    col_values = sorted(df[col_by].unique())
    if col_order:
        col_values = [c for c in col_order if c in col_values]

    metric_ranges = {}
    x_range = (df[x_col].min(), df[x_col].max())

    for m in metrics:
        trajs, _, _ = get_trajectories_for_group(df, {}, x_col, m, line_by, smooth_method, smooth_params)
        _, _, m_min, m_max = get_global_axis_ranges([trajs] if trajs else [])
        metric_ranges[m] = (m_min, m_max)

    subplots = []
    color_state = create_color_state(df, col_by, color_by_grouping, line_by, color_palette)
    legend_tracker = set()

    for r_idx, metric in enumerate(row_values):
        current_y_label = metric_labels.get(metric, metric) if metric_labels else metric

        for c_idx, col_val in enumerate(col_values):
            filter_dict = {col_by: col_val}

            traces = _compile_subplot_traces(
                df, filter_dict, x_col, metric, line_by, color_state,
                col_val, metric, color_by_grouping, col_by, None,
                bin_width, smooth_method, smooth_params, current_y_label, legend_tracker,
                trend_statistic, trend_smooth_sigma,
                show_individual=show_individual,
                show_error_band=show_error_band,
                error_type=error_type,
                error_band_alpha=error_band_alpha,
                show_linear_fit=show_linear_fit,
            )

            subplots.append(SubplotData(
                row=r_idx + 1,
                col=c_idx + 1,
                traces=traces,
                xlim=x_range,
                ylim=metric_ranges[metric],
                title=str(col_val) if r_idx == 0 else None,
                x_label=x_label if (r_idx == len(row_values) - 1) else None,
                y_label=current_y_label if (c_idx == 0) else None
            ))

    final_title = title or f"Multi-Metric Trajectories by {col_by}"
    fig_data = FigureData(
        title=final_title,
        n_rows=len(row_values),
        n_cols=len(col_values),
        height=max(DEFAULT_PLOTLY_HEIGHT, len(row_values) * height_per_row),
        width=max(DEFAULT_PLOTLY_WIDTH, len(col_values) * width_per_col),
        subplots=subplots,
        row_labels=[(metric_labels.get(m, m) if metric_labels else m) for m in metrics],
        color_by_grouping=color_by_grouping,
    )

    return _render_figure(fig_data, backend, output_path)
