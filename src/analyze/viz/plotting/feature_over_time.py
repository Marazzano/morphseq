"""
Plot feature over time with optional faceting.

100% DOMAIN-AGNOSTIC: No trajectory_analysis imports.
Caller provides color_lookup with domain-specific logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Any, Union, Dict, Set, Tuple

# Generic imports ONLY
from src.analyze.utils.data_processing import get_trajectories_for_group, get_global_axis_ranges
from src.analyze.utils.stats import compute_trend_line

# Engine imports
from .faceting_engine import (
    FigureData, SubplotData, TraceData, TraceStyle,
    FacetSpec, StyleSpec, render, default_style,
    iter_facet_cells, create_color_lookup, compute_error_band, STANDARD_PALETTE,
)


def _build_color_lookup(
    df: pd.DataFrame,
    color_by: Optional[str],
    color_lookup: Optional[Dict[Any, str]] = None,
    palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Build or use provided color lookup (NO domain logic).
    
    Private helper. If color_lookup provided, use it.
    Otherwise, auto-assign from palette.
    """
    if color_lookup is not None:
        return color_lookup
    
    if color_by is None or color_by not in df.columns:
        return {}
    
    unique_vals = list(df[color_by].dropna().unique())
    return create_color_lookup(unique_vals, palette or STANDARD_PALETTE)


def _build_subplot_ir(
    df: pd.DataFrame,
    filter_dict: Dict[str, Any],
    x_col: str,
    y_col: str,
    line_by: str,
    color_lookup: Dict[Any, str],
    color_by: Optional[str],
    subplot_key: Tuple[Any, Any],
    legend_tracker: Set[str],
    *,
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    trend_statistic: str = 'median',
    trend_smooth_sigma: float = 1.5,
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
) -> SubplotData:
    """Build SubplotData (IR) for one facet cell.
    
    Private helper. Lives here because it's trajectory-specific compilation.
    """
    # Determine color groups
    if color_by:
        mask = pd.Series(True, index=df.index)
        for k, v in filter_dict.items():
            mask &= (df[k] == v)
        groups = sorted(df.loc[mask, color_by].dropna().unique())
    else:
        groups = [None]
    
    traces: List[TraceData] = []
    
    for group_val in groups:
        group_filter = filter_dict.copy()
        if color_by:
            group_filter[color_by] = group_val
        
        trajectories, _, _ = get_trajectories_for_group(
            df, group_filter,
            time_col=x_col, metric_col=y_col, embryo_id_col=line_by,
            smooth_method=smooth_method, smooth_params=smooth_params,
        )
        
        if not trajectories:
            continue
        
        color = color_lookup.get(group_val, STANDARD_PALETTE[0])
        
        # Individual traces
        if show_individual:
            for traj in trajectories:
                traces.append(TraceData(
                    x=traj['times'], y=traj['metrics'],
                    style=TraceStyle(
                        color=color,
                        alpha=0.2,
                        width=0.8,
                        zorder=2,
                    ),
                    show_legend=False,
                    hover_meta={'header': f"ID: {traj['embryo_id']}", 'detail': f"{y_col}: {{y:.3f}}"},
                ))
        
        # Aggregated data
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])
        
        # Legend: show once per group across all subplots
        label = str(group_val) if group_val else trend_statistic
        legend_key = f"{y_col}_{group_val}"
        show_legend = legend_key not in legend_tracker
        if show_legend:
            legend_tracker.add(legend_key)
        
        # Error band
        if show_error_band:
            band_t, band_c, band_e = compute_error_band(
                all_times, all_metrics, bin_width,
                statistic=trend_statistic, error_type=error_type,
            )
            if band_t is not None:
                traces.append(TraceData(
                    x=band_t, y=band_c,
                    band_lower=band_c - band_e,
                    band_upper=band_c + band_e,
                    style=TraceStyle(color=color, alpha=0.2, width=0, zorder=3),
                    render_as='band',
                    show_legend=False,
                ))
        
        # Trend line
        trend_t, trend_v = compute_trend_line(
            all_times, all_metrics, bin_width,
            statistic=trend_statistic, smooth_sigma=trend_smooth_sigma,
        )
        if trend_t is not None and len(trend_t) > 0:
            traces.append(TraceData(
                x=np.array(trend_t), y=np.array(trend_v),
                style=TraceStyle(color=color, alpha=1.0, width=2.2, zorder=5),
                label=label,
                legend_group=legend_key,
                show_legend=show_legend,
                hover_meta={'header': f"{trend_statistic.capitalize()}: {label}", 'detail': f"{y_col}: {{y:.3f}}"},
            ))
    
    return SubplotData(
        key=subplot_key,
        traces=traces,
        x_label=x_col,
        y_label=y_col,
    )


def plot_feature_over_time(
    df: pd.DataFrame,
    feature: str,
    time_col: str = 'predicted_stage_hpf',
    id_col: str = 'embryo_id',
    color_by: Optional[str] = None,
    color_lookup: Optional[Dict[Any, str]] = None,  # ← USER PROVIDES domain-specific colors
    # Faceting (consistent API)
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    layout: Optional[FacetSpec] = None,
    # Display
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    trend_statistic: str = 'median',
    trend_smooth_sigma: float = 1.5,
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    # Output
    backend: str = 'plotly',
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    style: Optional[StyleSpec] = None,
    color_palette: Optional[List[str]] = None,  # Generic fallback
) -> Any:
    """Plot a feature over time, optionally faceted.
    
    100% DOMAIN-AGNOSTIC: Caller provides color_lookup for domain-specific coloring.
    If color_lookup=None, auto-assigns colors from palette.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time-series data
    feature : str
        Column name for y-axis metric
    time_col : str, default='predicted_stage_hpf'
        Column name for x-axis (time)
    id_col : str, default='embryo_id'
        Column name for trajectory identifiers
    color_by : str, optional
        Column to color traces by
    color_lookup : Dict[Any, str], optional
        Pre-built mapping from values in color_by column to hex colors.
        Use this to inject domain-specific coloring (e.g., genotype colors).
    facet_row : str, optional
        Column to facet by rows
    facet_col : str, optional
        Column to facet by columns
    layout : FacetSpec, optional
        Layout specification (row_order, col_order, sharex, sharey, etc.)
    show_individual : bool, default=True
        Whether to show individual trajectory lines
    show_error_band : bool, default=False
        Whether to show error band around trend line
    error_type : str, default='iqr'
        Error measure ('sd'/'se' for mean, 'iqr'/'mad' for median)
    trend_statistic : str, default='median'
        Central tendency statistic ('mean' or 'median')
    trend_smooth_sigma : float, default=1.5
        Gaussian smoothing sigma for trend line
    bin_width : float, default=0.5
        Bin width for aggregating trend line
    smooth_method : str, optional, default='gaussian'
        Smoothing method for individual trajectories
    smooth_params : dict, optional
        Parameters for smoothing method
    backend : str, default='plotly'
        Rendering backend ('plotly', 'matplotlib', or 'both')
    output_path : str or Path, optional
        Path to save figure
    title : str, optional
        Figure title
    style : StyleSpec, optional
        Style specification
    color_palette : List[str], optional
        Fallback palette if color_lookup not provided. Uses STANDARD_PALETTE if None.
    
    Returns
    -------
    Figure
        Plotly or matplotlib figure (or dict if backend='both')
    """
    layout = layout or FacetSpec(row_order=None, col_order=None)
    style = style or default_style()
    
    # Build color lookup (generic or user-provided)
    color_lookup = _build_color_lookup(df, color_by, color_lookup, color_palette)
    
    # Determine facet values
    row_vals = (layout.row_order if layout.row_order 
                else sorted(df[facet_row].dropna().unique()) if facet_row 
                else [None])
    col_vals = (layout.col_order if layout.col_order 
                else sorted(df[facet_col].dropna().unique()) if facet_col 
                else [None])
    
    # Compile subplots using engine's grid iterator
    legend_tracker: Set[str] = set()
    subplots = []
    
    for row_val, col_val, filter_dict, subplot_key in iter_facet_cells(
        facet_row, facet_col, row_vals, col_vals
    ):
        subplot = _build_subplot_ir(
            df=df,
            filter_dict=filter_dict,
            x_col=time_col,
            y_col=feature,
            line_by=id_col,
            color_lookup=color_lookup,
            color_by=color_by,
            subplot_key=subplot_key,
            legend_tracker=legend_tracker,
            show_individual=show_individual,
            show_error_band=show_error_band,
            error_type=error_type,
            trend_statistic=trend_statistic,
            trend_smooth_sigma=trend_smooth_sigma,
            bin_width=bin_width,
            smooth_method=smooth_method,
            smooth_params=smooth_params,
        )
        subplots.append(subplot)
    
    # Assemble FigureData
    fig_data = FigureData(
        title=title or f"{feature} over {time_col}",
        subplots=subplots,
        legend_title=color_by,
    )
    
    # Render via engine
    return render(fig_data, backend=backend, facet=layout, style=style, output_path=output_path)
