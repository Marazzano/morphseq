"""
Faceted plotting utilities for trajectory analysis.

Generic faceted plots that can group by ANY column - no pair-specific logic.
Supports both Plotly (interactive HTML with hover) and Matplotlib (static PNG).

Level 1: Generic group-by plotting (called by pair_analysis.plotting.py Level 2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .pair_analysis.data_utils import (
    get_trajectories_for_group,
    compute_binned_mean,
    get_global_axis_ranges,
)
from .genotype_styling import get_color_for_genotype, build_genotype_style_config
from .plot_config import (
    DEFAULT_PLOTLY_HEIGHT,
    DEFAULT_PLOTLY_WIDTH,
    HEIGHT_PER_ROW,
    WIDTH_PER_COL,
    INDIVIDUAL_TRACE_ALPHA,
    INDIVIDUAL_TRACE_LINEWIDTH,
    MEAN_TRACE_LINEWIDTH,
    OVERLAY_ALPHA,
)

# Standard qualitative color palette - used when color_by doesn't match genotypes
STANDARD_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Grey
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


def make_color_lookup(values: pd.Series) -> Dict:
    """Build a color lookup dict mapping unique values to palette colors.
    
    Works with any data type (ints, strings, etc.) - simple palette assignment.
    """
    unique_vals = list(pd.unique(values.dropna()))
    return {v: STANDARD_PALETTE[i % len(STANDARD_PALETTE)] for i, v in enumerate(unique_vals)}


def get_color_for_value(value: Any, color_by_column: Optional[str] = None) -> str:
    """Get color for a value, using genotype coloring if applicable, otherwise standard palette.
    
    Args:
        value: The value to get a color for (could be genotype, cluster ID, pair name, etc.)
        color_by_column: Optional column name to help determine if this is genotype data
        
    Returns:
        Hex color string
    """
    value_str = str(value)
    
    # Check if this looks like genotype data
    is_genotype = (
        color_by_column in ['genotype', 'Genotype', 'geno'] or
        any(keyword in value_str.lower() for keyword in ['wt', 'het', 'homo', 'wild', 'mutant'])
    )
    
    if is_genotype:
        # Use genotype-specific coloring
        return get_color_for_genotype(value_str)
    else:
        # Use standard palette - hash the string to get consistent color assignment
        unique_vals = [value]  # Will be expanded in actual usage
        color_idx = abs(hash(value_str)) % len(STANDARD_PALETTE)
        return STANDARD_PALETTE[color_idx]


def build_color_lookup_smart(df: pd.DataFrame, column: str) -> Dict:
    """Build color lookup that uses genotype colors for genotypes, standard palette otherwise.
    
    Args:
        df: DataFrame containing the data
        column: Column name to build colors for
        
    Returns:
        Dict mapping values to colors
    """
    unique_vals = df[column].dropna().unique()
    color_lookup = {}
    
    # Check if this is a genotype column
    is_genotype_col = (
        column in ['genotype', 'Genotype', 'geno'] or
        any(any(keyword in str(v).lower() for keyword in ['wt', 'het', 'homo', 'wild', 'mutant']) 
            for v in unique_vals)
    )
    
    if is_genotype_col:
        # Use genotype coloring
        for val in unique_vals:
            color_lookup[val] = get_color_for_genotype(str(val))
    else:
        # Use standard palette
        for i, val in enumerate(sorted(unique_vals, key=str)):
            color_lookup[val] = STANDARD_PALETTE[i % len(STANDARD_PALETTE)]
    
    return color_lookup


def _validate_facet_params(row_by: Optional[str], col_by: Optional[str], overlay: Optional[str]):
    """Validate facet parameters for potential issues.

    Issues warning (not error) if facet params have duplicates, since user might
    have a valid edge case. Exception: col_by == overlay is explicitly handled.
    """
    params = [row_by, col_by, overlay]
    params_named = {'row_by': row_by, 'col_by': col_by, 'overlay': overlay}

    # Check for None values
    non_none = [p for p in params if p is not None]

    # Check for duplicates (but col_by == overlay is explicitly handled, so skip warning)
    if len(non_none) != len(set(non_none)):
        duplicates = [p for p in non_none if non_none.count(p) > 1]
        
        # If the only duplicate is col_by == overlay, this is intentional and handled
        if not (len(duplicates) == 1 and col_by == overlay and col_by is not None):
            warnings.warn(
                f"Facet parameters have duplicates: {duplicates}. "
                "Each of row_by, col_by, overlay should ideally be unique.",
                UserWarning
            )


def _prepare_facet_grid_data(
    df: pd.DataFrame,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    overlay: Optional[str] = None,
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    time_col: Optional[str] = None,  # Backward compatibility
    metric_col: Optional[str] = None,  # Backward compatibility
) -> Tuple[Dict, List[str], List[str], Tuple[float, float, float, float]]:
    """Prepare data for faceted plotting - shared between backends.

    Args:
        df: DataFrame with trajectory data
        row_by: Column for row facets
        col_by: Column for column facets
        overlay: Column for overlay grouping
        x_col: X-axis column name
        y_col: Y-axis column name
        line_by: Column defining individual lines
        smooth_method: Gaussian smoothing or None
        smooth_params: Smoothing parameters

    Returns:
        Tuple of:
        - grid_data: {(row_val, col_val, overlay_val): {'trajectories': [...], 'n_embryos': N}}
        - row_order: Ordered row values
        - col_order: Ordered column values
        - (time_min, time_max, metric_min, metric_max): Global axis ranges
    """
    # Backward compatibility: handle old parameter names
    if time_col is not None and x_col == 'predicted_stage_hpf':
        x_col = time_col
    if metric_col is not None and y_col == 'baseline_deviation_normalized':
        y_col = metric_col

    # Validate parameters
    _validate_facet_params(row_by, col_by, overlay)

    # Get unique values for each facet dimension
    row_values = sorted(df[row_by].unique()) if row_by else [None]
    col_values = sorted(df[col_by].unique()) if col_by else [None]
    overlay_values = sorted(df[overlay].unique()) if overlay else [None]

    # Collect all data and trajectories for axis range calculation
    grid_data = {}
    all_trajectories_list = []

    for row_val in row_values:
        for col_val in col_values:
            for overlay_val in overlay_values:
                # Build filter dict
                filter_dict = {}
                if row_by:
                    filter_dict[row_by] = row_val
                if col_by:
                    filter_dict[col_by] = col_val
                if overlay:
                    filter_dict[overlay] = overlay_val

                # Extract trajectories for this cell
                trajectories, embryo_ids, n_embryos = get_trajectories_for_group(
                    df,
                    filter_dict,
                    time_col=x_col,
                    metric_col=y_col,
                    embryo_id_col=line_by,
                    smooth_method=smooth_method,
                    smooth_params=smooth_params,
                )

                key = (row_val, col_val, overlay_val)
                grid_data[key] = {
                    'trajectories': trajectories,
                    'embryo_ids': embryo_ids,
                    'n_embryos': n_embryos,
                    'filter': filter_dict,
                }

                if trajectories:
                    all_trajectories_list.append(trajectories)

    # Compute global axis ranges
    axis_ranges = get_global_axis_ranges(all_trajectories_list)

    return grid_data, row_values, col_values, axis_ranges


def plot_trajectories_faceted(
    df: pd.DataFrame,
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    overlay: Optional[str] = None,
    color_by: Optional[str] = None,
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
) -> Any:
    """Create faceted trajectory plots with flexible grouping.

    Generic Level 1 function that can group by any column(s).

    Args:
        df: DataFrame with trajectory data
        x_col: X-axis column name
        y_col: Y-axis column name
        line_by: Column defining individual lines (default: embryo_id)
        row_by: Column for row facets (optional)
        col_by: Column for column facets (optional)
        overlay: Column for overlay grouping within subplots (optional)
        color_by: Column for determining trace color (auto-colors genotypes)
        facet_order: Dict with custom ordering {'col_name': [ordered_values]}
        height_per_row: Plotly height per row (pixels)
        width_per_col: Plotly width per column (pixels)
        backend: 'plotly', 'matplotlib', or 'both'
        output_path: Path for saving figure (include extension for format)
        title: Figure title
        x_label: X-axis label
        y_label: Y-axis label
        bin_width: Bin width for mean trajectory calculation
        smooth_method: 'gaussian' or None
        smooth_params: Smoothing parameters dict

    Returns:
        - If backend='plotly': Plotly Figure
        - If backend='matplotlib': Matplotlib Figure
        - If backend='both': {'plotly': plotly_fig, 'matplotlib': mpl_fig}
    """
    # Prepare shared data
    grid_data, row_values, col_values, axis_ranges = _prepare_facet_grid_data(
        df,
        row_by=row_by,
        col_by=col_by,
        overlay=overlay,
        x_col=x_col,
        y_col=y_col,
        line_by=line_by,
        smooth_method=smooth_method,
        smooth_params=smooth_params,
    )

    time_min, time_max, metric_min, metric_max = axis_ranges

    # Apply custom ordering if provided
    if facet_order:
        if row_by and row_by in facet_order:
            row_values = [v for v in facet_order[row_by] if v in row_values]
        if col_by and col_by in facet_order:
            col_values = [v for v in facet_order[col_by] if v in col_values]

    # Set default title
    if title is None:
        parts = []
        if row_by:
            parts.append(f"by {row_by}")
        if col_by:
            parts.append(f"vs {col_by}")
        if overlay:
            parts.append(f"({overlay} overlaid)")
        title = "Trajectories " + " ".join(parts) if parts else "Trajectories"

    # Create backend-specific figures
    if backend in ['plotly', 'both']:
        fig_plotly = _plot_faceted_plotly(
            grid_data, row_values, col_values, x_col, y_col,
            row_by, col_by, overlay, color_by,
            time_min, time_max, metric_min, metric_max,
            height_per_row, width_per_col,
            title, x_label, y_label, bin_width,
        )
        if backend == 'plotly':
            if output_path:
                fig_plotly.write_html(str(output_path))
            return fig_plotly

    if backend in ['matplotlib', 'both']:
        fig_mpl = _plot_faceted_matplotlib(
            grid_data, row_values, col_values, x_col, y_col,
            row_by, col_by, overlay, color_by,
            time_min, time_max, metric_min, metric_max,
            title, x_label, y_label, bin_width,
        )
        if backend == 'matplotlib':
            if output_path:
                fig_mpl.savefig(str(output_path), dpi=150, bbox_inches='tight')
            return fig_mpl

    # backend == 'both'
    if output_path:
        output_path = Path(output_path)
        html_path = output_path.with_suffix('.html')
        png_path = output_path.with_suffix('.png')
        fig_plotly.write_html(str(html_path))
        fig_mpl.savefig(str(png_path), dpi=150, bbox_inches='tight')

    return {'plotly': fig_plotly, 'matplotlib': fig_mpl}


def _plot_faceted_plotly(
    grid_data: Dict,
    row_values: List,
    col_values: List,
    x_col: str,
    y_col: str,
    row_by: Optional[str],
    col_by: Optional[str],
    overlay: Optional[str],
    color_by: Optional[str],
    time_min: float,
    time_max: float,
    metric_min: float,
    metric_max: float,
    height_per_row: int,
    width_per_col: int,
    title: str,
    x_label: str,
    y_label: str,
    bin_width: float,
) -> go.Figure:
    """Plotly backend for faceted plotting with embryo hover."""
    n_rows = len(row_values)
    n_cols = len(col_values)

    height = max(DEFAULT_PLOTLY_HEIGHT, n_rows * height_per_row)
    width = max(DEFAULT_PLOTLY_WIDTH, n_cols * width_per_col)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[str(c) for c in col_values] if col_by else None,
    )

    # Track which legend groups we've already shown
    shown_legend_groups = set()

    for row_idx, row_val in enumerate(row_values, start=1):
        for col_idx, col_val in enumerate(col_values, start=1):
            for overlay_val in grid_data.keys():
                if overlay_val[0] != row_val or overlay_val[1] != col_val:
                    continue

                overlay_group = overlay_val[2] if overlay else None
                cell_data = grid_data[overlay_val]
                trajectories = cell_data['trajectories']

                if trajectories is None or cell_data['n_embryos'] == 0:
                    continue

                # Determine color
                # Priority order: overlay (most specific), col_by, row_by
                color_value = None
                if color_by:
                    if color_by == overlay:
                        color_value = overlay_group
                    elif color_by == col_by:
                        color_value = col_val
                    elif color_by == row_by:
                        color_value = row_val

                if color_value is not None:
                    color = get_color_for_genotype(str(color_value))
                else:
                    color = '#1f77b4'  # Default blue

                # Plot individual trajectories with embryo hover
                for traj in trajectories:
                    times = traj['times']
                    metrics = traj['metrics']
                    embryo_id = traj['embryo_id']

                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=metrics,
                            mode='lines',
                            line=dict(color=color, width=0.8),
                            opacity=INDIVIDUAL_TRACE_ALPHA,
                            hovertemplate=(
                                f'<b>Embryo:</b> {embryo_id}<br>'
                                '<b>Time:</b> %{x:.2f} hpf<br>'
                                f'<b>{y_label}:</b> %{{y:.4f}}<br>'
                                '<extra></extra>'
                            ),
                            showlegend=False,
                        ),
                        row=row_idx, col=col_idx
                    )

                # Plot mean trajectory
                all_times = np.concatenate([t['times'] for t in trajectories])
                all_metrics = np.concatenate([t['metrics'] for t in trajectories])
                bin_times, bin_means = compute_binned_mean(all_times, all_metrics, bin_width)

                if bin_times:
                    label_text = str(overlay_group) if overlay else y_label
                    legend_group = label_text
                    show_legend = legend_group not in shown_legend_groups
                    if show_legend:
                        shown_legend_groups.add(legend_group)

                    fig.add_trace(
                        go.Scatter(
                            x=bin_times,
                            y=bin_means,
                            mode='lines',
                            line=dict(color=color, width=MEAN_TRACE_LINEWIDTH),
                            name=label_text,
                            legendgroup=legend_group,
                            showlegend=show_legend,
                        ),
                        row=row_idx, col=col_idx
                    )

            # Set axis limits for this subplot
            fig.update_xaxes(range=[time_min, time_max], row=row_idx, col=col_idx)
            fig.update_yaxes(range=[metric_min, metric_max], row=row_idx, col=col_idx)
            fig.update_xaxes(title_text=x_label, row=row_idx, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text=y_label, row=row_idx, col=col_idx)

    # Add row labels on the left side if row_by is specified
    if row_by and n_rows > 1:
        for row_idx, row_val in enumerate(row_values, start=1):
            # Calculate vertical position (center of row)
            y_position = 1 - (row_idx - 0.5) / n_rows
            fig.add_annotation(
                text=f"<b>{row_val}</b>",
                xref="paper",
                yref="paper",
                x=-0.05,
                y=y_position,
                xanchor="right",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12),
            )

    fig.update_layout(
        title_text=title,
        height=height,
        width=width,
        hovermode='closest',
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    return fig


def _plot_faceted_matplotlib(
    grid_data: Dict,
    row_values: List,
    col_values: List,
    x_col: str,
    y_col: str,
    row_by: Optional[str],
    col_by: Optional[str],
    overlay: Optional[str],
    color_by: Optional[str],
    time_min: float,
    time_max: float,
    metric_min: float,
    metric_max: float,
    title: str,
    x_label: str,
    y_label: str,
    bin_width: float,
) -> plt.Figure:
    """Matplotlib backend for faceted plotting."""
    n_rows = len(row_values)
    n_cols = len(col_values)

    figsize = (5 * n_cols, 4.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax = axes[row_idx][col_idx]

            for overlay_val in grid_data.keys():
                if overlay_val[0] != row_val or overlay_val[1] != col_val:
                    continue

                overlay_group = overlay_val[2] if overlay else None
                cell_data = grid_data[overlay_val]
                trajectories = cell_data['trajectories']

                if trajectories is None or cell_data['n_embryos'] == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10, color='lightgray')
                    continue

                # Determine color
                # Priority order: overlay (most specific), col_by, row_by
                color_value = None
                if color_by:
                    if color_by == overlay:
                        color_value = overlay_group
                    elif color_by == col_by:
                        color_value = col_val
                    elif color_by == row_by:
                        color_value = row_val

                if color_value is not None:
                    color = get_color_for_genotype(str(color_value))
                else:
                    color = '#1f77b4'  # Default blue

                # Plot individual trajectories
                for traj in trajectories:
                    ax.plot(traj['times'], traj['metrics'],
                           color=color, alpha=INDIVIDUAL_TRACE_ALPHA, linewidth=INDIVIDUAL_TRACE_LINEWIDTH)

                # Plot mean trajectory
                all_times = np.concatenate([t['times'] for t in trajectories])
                all_metrics = np.concatenate([t['metrics'] for t in trajectories])
                bin_times, bin_means = compute_binned_mean(all_times, all_metrics, bin_width)

                if bin_times:
                    label_text = str(overlay_group) if overlay else y_label
                    ax.plot(bin_times, bin_means, color=color, linewidth=MEAN_TRACE_LINEWIDTH,
                           label=label_text, zorder=5)

            # Set axis properties
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f'{col_val}', fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_xlim(time_min, time_max)
            ax.set_ylim(metric_min, metric_max)
            ax.tick_params(labelsize=9)
            if ax.get_legend_handles_labels()[0]:  # Has legend
                ax.legend(fontsize=9, loc='upper right')

    # Add row labels on the left side if row_by is specified
    if row_by and n_rows > 1:
        for row_idx, row_val in enumerate(row_values):
            # Calculate vertical position (center of row in figure coordinates)
            y_position = 1 - (row_idx + 0.5) / n_rows
            fig.text(
                -0.02, y_position,
                str(row_val),
                rotation=90,
                verticalalignment='center',
                horizontalalignment='right',
                fontsize=12,
                fontweight='bold',
                transform=fig.transFigure
            )

    plt.tight_layout()
    return fig


# TODO (refactor ideas for later):
# - Pull shared helpers out of both backends (resolve_color, compute_mean_trace, metric_label)
# - Precompute per-cell trajectories once and reuse in Plotly/Matplotlib
# - Centralize subplot label/x/y-title logic to trim inline branching
# - Keep color_by literal: build a color lookup from the requested column and avoid special cases


def plot_multimetric_trajectories(
    df: pd.DataFrame,
    metrics: List[str],
    col_by: str,
    x_col: str = 'predicted_stage_hpf',
    line_by: str = 'embryo_id',
    overlay: Optional[str] = None,
    color_by: Optional[str] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    col_order: Optional[List] = None,
    share_y: str = 'row',
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    x_label: str = 'Time (hpf)',
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
) -> Any:
    """Create multi-metric time-series plots with metrics in rows, clusters in columns.

    This function extends the faceted plotting system to support multiple metrics
    with independent y-axis scales. Each metric gets its own row, and clusters are
    displayed as columns, with shared x-axis (time) for temporal alignment.

    Args:
        df: DataFrame with trajectory data
        metrics: List of column names for metrics to plot (e.g., ['curvature', 'length'])
        col_by: Column name for column facets (e.g., 'cluster', 'genotype', 'pair')
        x_col: X-axis column name (time)
        line_by: Column defining individual lines (default: embryo_id)
        overlay: Column for overlay grouping within columns (creates separate mean trajectories per group)
        color_by: Column for determining trace color (auto-colors genotypes if not specified)
        metric_labels: Optional display names for metrics {metric_col: label}
        col_order: Custom ordering for column facets
        share_y: Y-axis sharing: 'row' (per-metric), 'all', 'none' (default: 'row')
        height_per_row: Plotly height per row (pixels)
        width_per_col: Plotly width per column (pixels)
        backend: 'plotly', 'matplotlib', or 'both'
        output_path: Path for saving figure
        title: Figure title
        x_label: X-axis label
        bin_width: Bin width for mean trajectory calculation
        smooth_method: 'gaussian' or None
        smooth_params: Smoothing parameters dict

    Returns:
        - If backend='plotly': Plotly Figure
        - If backend='matplotlib': Matplotlib Figure
        - If backend='both': {'plotly': plotly_fig, 'matplotlib': mpl_fig}

    Example:
        >>> # Color by cluster to see cluster identity
        >>> fig = plot_multimetric_trajectories(
        ...     df,
        ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
        ...     col_by='cluster',
        ...     color_by='cluster',
        ...     metric_labels={
        ...         'baseline_deviation_normalized': 'Curvature (Z-score)',
        ...         'total_length_um': 'Length (μm)'
        ...     },
        ...     backend='plotly'
        ... )

        >>> # Color by genotype to validate clusters separate genotypes
        >>> fig = plot_multimetric_trajectories(
        ...     df,
        ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
        ...     col_by='cluster',
        ...     color_by='genotype',
        ...     backend='plotly'
        ... )
    """
    # Step 1: Prepare data for each metric
    metric_data = {}
    all_col_values = set()

    for metric in metrics:
        grid_data, row_vals, col_vals, (t_min, t_max, m_min, m_max) = _prepare_facet_grid_data(
            df,
            row_by=None,
            col_by=col_by,
            overlay=overlay,
            x_col=x_col,
            y_col=metric,
            line_by=line_by,
            smooth_method=smooth_method,
            smooth_params=smooth_params,
        )

        metric_data[metric] = {
            'grid_data': grid_data,
            'col_values': col_vals,
            'time_range': (t_min, t_max),
            'metric_range': (m_min, m_max),
        }
        all_col_values.update(col_vals)

    # Sort column values for consistent ordering
    if col_order:
        col_values = [c for c in col_order if c in all_col_values]
    else:
        col_values = sorted(all_col_values)

    # Set default title
    if title is None:
        title = f"Multi-Metric Trajectories by {col_by}"

    # Create backend-specific figures
    if backend in ['plotly', 'both']:
        fig_plotly = _plot_multimetric_plotly(
            df, metric_data, metrics, col_values, col_by,
            x_col, overlay, color_by, line_by, metric_labels,
            height_per_row, width_per_col,
            title, x_label, bin_width,
        )
        if backend == 'plotly':
            if output_path:
                fig_plotly.write_html(str(output_path))
            return fig_plotly

    if backend in ['matplotlib', 'both']:
        fig_mpl = _plot_multimetric_matplotlib(
            df, metric_data, metrics, col_values, col_by,
            x_col, overlay, color_by, line_by, metric_labels,
            title, x_label, bin_width,
        )
        if backend == 'matplotlib':
            if output_path:
                fig_mpl.savefig(str(output_path), dpi=150, bbox_inches='tight')
            return fig_mpl

    # backend == 'both'
    if output_path:
        output_path = Path(output_path)
        html_path = output_path.with_suffix('.html')
        png_path = output_path.with_suffix('.png')
        fig_plotly.write_html(str(html_path))
        fig_mpl.savefig(str(png_path), dpi=150, bbox_inches='tight')

    return {'plotly': fig_plotly, 'matplotlib': fig_mpl}


def _plot_multimetric_plotly(
    df: pd.DataFrame,
    metric_data: Dict,
    metrics: List[str],
    col_values: List,
    col_by: str,
    x_col: str,
    overlay: Optional[str],
    color_by: Optional[str],
    line_by: str,
    metric_labels: Optional[Dict[str, str]],
    height_per_row: int,
    width_per_col: int,
    title: str,
    x_label: str,
    bin_width: float,
) -> go.Figure:
    """Plotly backend for multi-metric faceted plotting."""
    n_metrics = len(metrics)
    n_cols = len(col_values)

    height = max(DEFAULT_PLOTLY_HEIGHT, n_metrics * height_per_row)
    width = max(DEFAULT_PLOTLY_WIDTH, n_cols * width_per_col)

    fig = make_subplots(
        rows=n_metrics,
        cols=n_cols,
        subplot_titles=[str(c) for c in col_values] if n_metrics == 1 else None,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Track legend groups shown (across all metrics to avoid duplicates)
    shown_legend_groups = set()
    
    # Build smart color lookup for overlay/color_by column
    color_column = color_by if color_by else (overlay if overlay else col_by)
    color_lookup = build_color_lookup_smart(df, color_column)

    for metric_idx, metric in enumerate(metrics, start=1):
        time_min, time_max = metric_data[metric]['time_range']
        metric_min, metric_max = metric_data[metric]['metric_range']

        for col_idx, col_val in enumerate(col_values, start=1):
            # Iterate over all overlay groups for this (metric, col) cell
            grid_data = metric_data[metric]['grid_data']
            
            for key, cell_data in grid_data.items():
                row_val, key_col_val, overlay_val = key
                
                # Only process entries matching this column
                if key_col_val != col_val:
                    continue
                
                # When col_by == overlay, only show matching overlay groups
                # (e.g., cluster 0 column should only show overlay cluster 0)
                if overlay and overlay == col_by and overlay_val != col_val:
                    continue
                
                trajectories = cell_data.get('trajectories', [])
                if not trajectories or cell_data.get('n_embryos', 0) == 0:
                    continue

                # Determine color for this overlay group
                # Priority: color_by (if specified), else overlay, else col_by
                if color_by:
                    if color_by == overlay:
                        color_value = overlay_val
                    elif color_by == col_by:
                        color_value = col_val
                    else:
                        # color_by is some other column - use first value in this group
                        # (all trajectories in this group should have same color_by value)
                        color_value = overlay_val if overlay else col_val
                else:
                    # Default: color by overlay if present, else by column
                    color_value = overlay_val if overlay else col_val
                
                # Get color from smart lookup
                color = color_lookup.get(color_value, STANDARD_PALETTE[0])

                y_label_text = metric_labels.get(metric, metric) if metric_labels else metric

                # Plot individual trajectories (all same color for this overlay group)
                for traj in trajectories:
                    times = traj['times']
                    metric_vals = traj['metrics']
                    embryo_id = traj['embryo_id']

                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=metric_vals,
                            mode='lines',
                            line=dict(color=color, width=INDIVIDUAL_TRACE_LINEWIDTH),
                            opacity=INDIVIDUAL_TRACE_ALPHA,
                            hovertemplate=(
                                f'<b>Embryo:</b> {embryo_id}<br>'
                                '<b>Time:</b> %{x:.2f} hpf<br>'
                                f'<b>{y_label_text}:</b> %{{y:.4f}}<br>'
                                '<extra></extra>'
                            ),
                            showlegend=False,
                        ),
                        row=metric_idx, col=col_idx
                    )

                # Plot mean trajectory for this overlay group
                all_times = np.concatenate([t['times'] for t in trajectories])
                all_metric_vals = np.concatenate([t['metrics'] for t in trajectories])
                bin_times, bin_means = compute_binned_mean(all_times, all_metric_vals, bin_width)

                if bin_times:
                    # Label: show overlay value if present, else column value
                    if overlay:
                        label_text = str(overlay_val)
                        legend_group = str(overlay_val)
                    else:
                        label_text = f"{col_by}={col_val}" if col_by else "mean"
                        legend_group = f"{col_val}"
                    
                    # Show legend only once per overlay group (first occurrence)
                    show_legend = legend_group not in shown_legend_groups
                    if show_legend:
                        shown_legend_groups.add(legend_group)

                    fig.add_trace(
                        go.Scatter(
                            x=bin_times,
                            y=bin_means,
                            mode='lines',
                            line=dict(color=color, width=MEAN_TRACE_LINEWIDTH),
                            name=label_text,
                            legendgroup=legend_group,
                            showlegend=show_legend,
                        ),
                        row=metric_idx, col=col_idx
                    )

        # Set axis ranges for this metric row
        for col_idx in range(1, n_cols + 1):
            fig.update_xaxes(range=[time_min, time_max], row=metric_idx, col=col_idx)
            fig.update_yaxes(range=[metric_min, metric_max], row=metric_idx, col=col_idx)

            # X-label on bottom row only; hide duplicate x tick labels on upper rows
            if metric_idx == n_metrics:
                fig.update_xaxes(title_text=x_label, row=metric_idx, col=col_idx)
            else:
                fig.update_xaxes(showticklabels=False, title_text=None, row=metric_idx, col=col_idx)

            # Y-label on leftmost column only
            if col_idx == 1:
                y_label_text = metric_labels.get(metric, metric) if metric_labels else metric
                fig.update_yaxes(title_text=y_label_text, row=metric_idx, col=col_idx)

    # Add column titles at top
    if n_metrics > 1:
        for col_idx, col_val in enumerate(col_values, start=1):
            x_position = (col_idx - 0.5) / n_cols
            fig.add_annotation(
                text=f"<b>{col_val}</b>",
                xref="paper",
                yref="paper",
                x=x_position,
                y=1.02,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=12),
            )
    
    # Add metric row labels on the left side
    if n_metrics > 1:
        for metric_idx, metric in enumerate(metrics, start=1):
            y_position = 1 - (metric_idx - 0.5) / n_metrics
            metric_label = metric_labels.get(metric, metric) if metric_labels else metric

            fig.add_annotation(
                text=f"<b>{metric_label}</b>",
                xref="paper",
                yref="paper",
                x=-0.08,
                y=y_position,
                xanchor="right",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12),
            )

    fig.update_layout(
        title_text=title,
        height=height,
        width=width,
        hovermode='closest',
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    return fig


def _plot_multimetric_matplotlib(
    df: pd.DataFrame,
    metric_data: Dict,
    metrics: List[str],
    col_values: List,
    col_by: str,
    x_col: str,
    overlay: Optional[str],
    color_by: Optional[str],
    line_by: str,
    metric_labels: Optional[Dict[str, str]],
    title: str,
    x_label: str,
    bin_width: float,
) -> plt.Figure:
    """Matplotlib backend for multi-metric faceted plotting."""
    n_metrics = len(metrics)
    n_cols = len(col_values)

    figsize = (5 * n_cols, 4.5 * n_metrics)
    fig, axes = plt.subplots(n_metrics, n_cols, figsize=figsize, sharex=True)

    # Handle single row or column cases
    if n_metrics == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_metrics == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Build smart color lookup for overlay/color_by column
    color_column = color_by if color_by else (overlay if overlay else col_by)
    color_lookup = build_color_lookup_smart(df, color_column)

    for metric_idx, metric in enumerate(metrics):
        time_min, time_max = metric_data[metric]['time_range']
        metric_min, metric_max = metric_data[metric]['metric_range']

        for col_idx, col_val in enumerate(col_values):
            ax = axes[metric_idx][col_idx]

            # Iterate over all overlay groups for this (metric, col) cell
            grid_data = metric_data[metric]['grid_data']
            has_data = False
            
            for key, cell_data in grid_data.items():
                row_val, key_col_val, overlay_val = key
                
                # Only process entries matching this column
                if key_col_val != col_val:
                    continue
                
                # When col_by == overlay, only show matching overlay groups
                # (e.g., cluster 0 column should only show overlay cluster 0)
                if overlay and overlay == col_by and overlay_val != col_val:
                    continue
                
                trajectories = cell_data.get('trajectories', [])
                if not trajectories or cell_data.get('n_embryos', 0) == 0:
                    continue
                
                has_data = True

                # Determine color for this overlay group (same logic as plotly)
                if color_by:
                    if color_by == overlay:
                        color_value = overlay_val
                    elif color_by == col_by:
                        color_value = col_val
                    else:
                        color_value = overlay_val if overlay else col_val
                else:
                    color_value = overlay_val if overlay else col_val
                
                # Get color from smart lookup
                color = color_lookup.get(color_value, STANDARD_PALETTE[0])

                # Plot individual trajectories (all same color for this overlay group)
                for traj in trajectories:
                    ax.plot(traj['times'], traj['metrics'],
                           color=color, alpha=INDIVIDUAL_TRACE_ALPHA,
                           linewidth=INDIVIDUAL_TRACE_LINEWIDTH)

                # Plot mean trajectory for this overlay group
                all_times = np.concatenate([t['times'] for t in trajectories])
                all_metric_vals = np.concatenate([t['metrics'] for t in trajectories])
                bin_times, bin_means = compute_binned_mean(all_times, all_metric_vals, bin_width)

                if bin_times:
                    if overlay:
                        label_text = str(overlay_val)
                    else:
                        label_text = f"{col_by}={col_val}" if col_by else "mean"
                    ax.plot(bin_times, bin_means, color=color,
                           linewidth=MEAN_TRACE_LINEWIDTH, label=label_text, zorder=5)
            
            if not has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='lightgray')
                continue

            # Set axis properties
            ax.set_xlim(time_min, time_max)
            ax.set_ylim(metric_min, metric_max)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=9)

            # Labels
            if metric_idx == n_metrics - 1:
                ax.set_xlabel(x_label, fontsize=10)
            if col_idx == 0:
                y_label_text = metric_labels.get(metric, metric) if metric_labels else metric
                ax.set_ylabel(y_label_text, fontsize=10)

            # Title (col name) on top row
            if metric_idx == 0:
                ax.set_title(str(col_val), fontweight='bold', fontsize=11)

            # Legend
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=9, loc='upper right')

    # Add metric row labels on the left side
    if n_metrics > 1:
        for metric_idx, metric in enumerate(metrics):
            y_position = 1 - (metric_idx + 0.5) / n_metrics
            metric_label = metric_labels.get(metric, metric) if metric_labels else metric

            fig.text(
                -0.02, y_position,
                metric_label,
                rotation=90,
                verticalalignment='center',
                horizontalalignment='right',
                fontsize=12,
                fontweight='bold',
                transform=fig.transFigure
            )

    plt.tight_layout()
    return fig


# ==============================================================================
# TODO: Future refactoring opportunities (code deduplication)
# ==============================================================================
#
# The Plotly and Matplotlib backends share significant logic. Suggested refactors:
#
# 1. Extract shared helpers (used by both backends):
#    - resolve_color(df, col_val, col_by, color_by, color_lookup) 
#      Returns (color_value, color) so the "mode per col" logic lives once
#    - compute_mean_trace(trajectories, bin_width) 
#      Centralize binned mean calculation instead of repeating in both backends
#    - get_metric_label(metric, metric_labels) 
#      Avoid repeated ternaries like "metric_labels.get(metric, metric) if metric_labels else metric"
#    - prepare_metric_grid(df, metric, col_by, ...) 
#      Centralize per-metric data preparation loop
#
# 2. Reduce backend divergence:
#    - Build per-cell trajectory data once: Cell = {times, values, embryo_id}
#    - Reuse across both Plotly and Matplotlib instead of extracting separately
#
# 3. Palette handling:
#    - Already good: precompute color_lookup once
#    - Simplify: when color_by == col_by, just use color_lookup[col_val]
#      (no need for separate branch since lookup was built on color_by)
#
# 4. Layout tweaks:
#    - Matplotlib: call axes[i][j].label_outer() to auto-hide duplicate labels
#    - Plotly: already uses showticklabels=False on non-bottom rows
#    - Centralize title/label placement into tiny helper functions
#
# Benefits: ~30-40% less code, easier to maintain, same outputs
