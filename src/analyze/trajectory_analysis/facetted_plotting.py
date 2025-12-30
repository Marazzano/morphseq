"""
Faceted plotting utilities for trajectory analysis.

Generic faceted plots that can group by ANY column - no pair-specific logic.
Supports both Plotly (interactive HTML with hover) and Matplotlib (static PNG).
Refactored to use an Intermediate Representation (IR) pattern for stability.

Level 1: Generic group-by plotting (called by pair_analysis.plotting.py Level 2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .pair_analysis.data_utils import (
    get_trajectories_for_group,
    get_global_axis_ranges,
)
from .trajectory_utils import compute_trend_line
from .genotype_styling import get_color_for_genotype
from .plot_config import (
    DEFAULT_PLOTLY_HEIGHT,
    DEFAULT_PLOTLY_WIDTH,
    HEIGHT_PER_ROW,
    WIDTH_PER_COL,
    INDIVIDUAL_TRACE_ALPHA,
    INDIVIDUAL_TRACE_LINEWIDTH,
    MEAN_TRACE_LINEWIDTH,
)

# Standard qualitative color palette
STANDARD_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# ==============================================================================
# 1. Intermediate Representation (The "IR")
# ==============================================================================

@dataclass
class TraceData:
    """Represents a single line on a plot."""
    x: np.ndarray
    y: np.ndarray
    color: str
    width: float
    alpha: float
    label: Optional[str] = None
    hover_header: Optional[str] = None
    hover_detail: Optional[str] = None
    legend_group: Optional[str] = None
    show_legend: bool = False
    zorder: int = 1

@dataclass
class SubplotData:
    """Represents a single grid cell (one set of axes)."""
    row: int  # 1-based index
    col: int  # 1-based index
    traces: List[TraceData]
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None

@dataclass
class FigureData:
    """Represents the complete figure, agnostic of backend."""
    title: str
    n_rows: int
    n_cols: int
    height: int
    width: int
    subplots: List[SubplotData]
    row_labels: List[str] = field(default_factory=list)
    col_labels: List[str] = field(default_factory=list)
    color_by_grouping: Optional[str] = None

# ==============================================================================
# 2. Color & Data Helpers (Shared Logic)
# ==============================================================================

def make_color_lookup(values: pd.Series) -> Dict:
    unique_vals = list(pd.unique(values.dropna()))
    return {v: STANDARD_PALETTE[i % len(STANDARD_PALETTE)] for i, v in enumerate(unique_vals)}

def build_color_lookup_for_column(df: pd.DataFrame, column: Optional[str]) -> Dict:
    if column is None or column not in df.columns:
        return {}
    if column == 'genotype':
        unique_vals = list(pd.unique(df[column].dropna()))
        return {v: get_color_for_genotype(str(v)) for v in unique_vals}
    return make_color_lookup(df[column])

def build_color_state(
    df: pd.DataFrame,
    col_by: Optional[str],
    color_by_grouping: Optional[str],
    line_by: str
) -> Dict[str, Dict]:

    primary_column = color_by_grouping or col_by

    return {
        'primary_lookup': build_color_lookup_for_column(df, primary_column),
        'col_lookup': build_color_lookup_for_column(df, col_by),
        'grouping_lookup': build_color_lookup_for_column(df, color_by_grouping),
    }

def resolve_color_value(
    color_val_raw: Any,
    color_state: Dict[str, Dict],
    fallback_val: Any = None,
    lookup_type: str = 'primary_lookup'
) -> str:
    """Helper to safely lookup hex color from state."""
    if color_val_raw is not None:
        c = color_state[lookup_type].get(color_val_raw)
        if c: return c

    # Fallback
    if fallback_val is not None:
        # Try finding the fallback in any of the lookups
        c = (color_state['primary_lookup'].get(fallback_val) or
             color_state['grouping_lookup'].get(fallback_val) or
             color_state['col_lookup'].get(fallback_val))
        if c: return c

    return STANDARD_PALETTE[0]

def _build_traces_for_cell(
    df: pd.DataFrame,
    filter_dict: Dict,
    x_col: str,
    y_col: str,
    line_by: str,
    color_state: Dict,
    col_val: Any,
    row_val: Any,
    color_by_grouping: Optional[str],
    col_by: Optional[str],
    row_by: Optional[str],
    bin_width: float,
    smooth_method: Optional[str],
    smooth_params: Optional[Dict],
    y_label_text: str,
    legend_tracker: set,
    trend_statistic: str = 'median',
    trend_smooth_sigma: Optional[float] = 1.5,
) -> List[TraceData]:
    """Generates all traces (individual + means) for a specific grid cell."""

    # Determine which groupings exist in this specific cell data to avoid empty loops
    if color_by_grouping:
        subset_mask = pd.Series(True, index=df.index)
        for k, v in filter_dict.items():
            subset_mask &= (df[k] == v)
        existing_groups = sorted(df[subset_mask][color_by_grouping].unique())
    else:
        existing_groups = [None]

    cell_traces = []

    for grouping_val in existing_groups:
        group_filter = filter_dict.copy()
        if color_by_grouping:
            group_filter[color_by_grouping] = grouping_val

        trajectories, _, n_embryos = get_trajectories_for_group(
            df, group_filter, time_col=x_col, metric_col=y_col, 
            embryo_id_col=line_by, smooth_method=smooth_method, smooth_params=smooth_params
        )

        if not trajectories:
            continue

        # 1. Individual Traces
        # Track the colors seen in this group to check for homogeneity
        seen_color_vals = set()

        for traj in trajectories:
            embryo_id = traj['embryo_id']

            # Simplified color resolution: use grouping value or column value
            line_color_val = grouping_val if color_by_grouping else col_val

            seen_color_vals.add(line_color_val)
            color_hex = resolve_color_value(line_color_val, color_state, fallback_val=col_val)
            
            cell_traces.append(TraceData(
                x=traj['times'],
                y=traj['metrics'],
                color=color_hex,
                width=INDIVIDUAL_TRACE_LINEWIDTH,
                alpha=INDIVIDUAL_TRACE_ALPHA,
                hover_header=f"Embryo: {embryo_id}",
                hover_detail=f"{y_label_text}: %{{y:.4f}}",
                show_legend=False,
                zorder=2
            ))

        # 2. Trend Line Trace Logic
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])

        # TODO(morphseq): We have observed cases where the trend line does not render all the
        # way to the smallest x values (earliest timepoints), even when per-facet/group
        # `min(time)` matches the global minimum (i.e. not a simple data-availability issue).
        # Suspects: bin edge inclusion/exclusion in `compute_trend_line` (e.g. lowest-edge
        # handling), NaN dropping after binning, or smoothing behavior near boundaries.
        bin_times, bin_stats = compute_trend_line(
            all_times, all_metrics, bin_width,
            statistic=trend_statistic,
            smooth_sigma=trend_smooth_sigma
        )

        if bin_times is not None and len(bin_times) > 0:
            # --- STRICT HOMOGENEITY CHECK ---
            trend_color_val = None

            # If every line in this group has the exact same color value,
            # we promote that color to the trend line.
            if len(seen_color_vals) == 1:
                trend_color_val = list(seen_color_vals)[0]
            else:
                # Heterogeneous group! Fallback to the grouping identity.
                # e.g. If "Pair 1" has mixed Genotypes, color it by "Pair 1" color, not Genotype.
                trend_color_val = grouping_val if color_by_grouping else col_val

            trend_color_hex = resolve_color_value(trend_color_val, color_state, fallback_val=col_val)

            # Label Logic
            if color_by_grouping:
                label = str(grouping_val)
                group_key = f"{y_col}_{grouping_val}"
            else:
                label = f"{col_by}={col_val}" if col_by else trend_statistic
                group_key = f"{y_col}_{col_val}"

            show_legend = group_key not in legend_tracker
            if show_legend:
                legend_tracker.add(group_key)

            # Hover text reflects the statistic type (median or mean)
            statistic_label = trend_statistic.capitalize()

            cell_traces.append(TraceData(
                x=bin_times,
                y=bin_stats,
                color=trend_color_hex,
                width=MEAN_TRACE_LINEWIDTH,
                alpha=1.0,
                label=label,
                legend_group=group_key,
                show_legend=show_legend,
                zorder=5,
                hover_header=f"{statistic_label}: {label}",
                hover_detail=f"{y_label_text}: %{{y:.4f}}",
            ))

    return cell_traces

# ==============================================================================
# 3. Public API Functions
# ==============================================================================

def plot_trajectories_faceted(
    df: pd.DataFrame,
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    color_by_grouping: Optional[str] = None,
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
    time_col: Optional[str] = None, # Compat
    metric_col: Optional[str] = None, # Compat
) -> Any:
    """Standard faceted plot (rows=row_by, cols=col_by) for a SINGLE metric."""
    x_col = time_col or x_col
    y_col = metric_col or y_col

    row_values = sorted(df[row_by].unique()) if row_by else [None]
    col_values = sorted(df[col_by].unique()) if col_by else [None]
    
    if facet_order:
        if row_by and row_by in facet_order:
            row_values = [v for v in facet_order[row_by] if v in row_values]
        if col_by and col_by in facet_order:
            col_values = [v for v in facet_order[col_by] if v in col_values]

    # Global ranges
    all_trajs, _, _ = get_trajectories_for_group(df, {}, x_col, y_col, line_by, smooth_method, smooth_params)
    t_min, t_max, m_min, m_max = get_global_axis_ranges([all_trajs] if all_trajs else [])

    subplots = []
    color_state = build_color_state(df, col_by, color_by_grouping, line_by)
    legend_tracker = set()

    for r_idx, row_val in enumerate(row_values):
        for c_idx, col_val in enumerate(col_values):
            filter_dict = {}
            if row_by: filter_dict[row_by] = row_val
            if col_by: filter_dict[col_by] = col_val

            traces = _build_traces_for_cell(
                df, filter_dict, x_col, y_col, line_by, color_state,
                col_val, row_val, color_by_grouping, col_by, row_by,
                bin_width, smooth_method, smooth_params, y_label, legend_tracker,
                trend_statistic, trend_smooth_sigma
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

    return _dispatch_renderers(fig_data, backend, output_path)

def plot_multimetric_trajectories(
    df: pd.DataFrame,
    metrics: List[str],
    col_by: str,
    x_col: str = 'predicted_stage_hpf',
    line_by: str = 'embryo_id',
    color_by_grouping: Optional[str] = None,
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
    share_y: str = 'row',
    trend_statistic: str = 'median',
    trend_smooth_sigma: Optional[float] = 1.5,
) -> Any:
    """Multi-metric plot: Rows = Metrics, Cols = Groups."""
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
    color_state = build_color_state(df, col_by, color_by_grouping, line_by)
    legend_tracker = set()

    for r_idx, metric in enumerate(row_values):
        current_y_label = metric_labels.get(metric, metric) if metric_labels else metric

        for c_idx, col_val in enumerate(col_values):
            filter_dict = {col_by: col_val}

            traces = _build_traces_for_cell(
                df, filter_dict, x_col, metric, line_by, color_state,
                col_val, metric, color_by_grouping, col_by, None,
                bin_width, smooth_method, smooth_params, current_y_label, legend_tracker,
                trend_statistic, trend_smooth_sigma
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

    return _dispatch_renderers(fig_data, backend, output_path)

# ==============================================================================
# 4. Renderers (Dumb & Backend Specific)
# ==============================================================================

def _dispatch_renderers(fig_data: FigureData, backend: str, output_path: Optional[Path]) -> Any:
    results = {}
    
    if backend in ['plotly', 'both']:
        fig_plotly = _render_plotly(fig_data)
        if output_path and backend == 'plotly':
            fig_plotly.write_html(str(output_path))
        results['plotly'] = fig_plotly

    if backend in ['matplotlib', 'both']:
        fig_mpl = _render_matplotlib(fig_data)
        if output_path and backend == 'matplotlib':
            fig_mpl.savefig(str(output_path), dpi=150, bbox_inches='tight')
        results['matplotlib'] = fig_mpl

    if backend == 'both' and output_path:
        path = Path(output_path)
        results['plotly'].write_html(str(path.with_suffix('.html')))
        results['matplotlib'].savefig(str(path.with_suffix('.png')), dpi=150, bbox_inches='tight')

    if backend == 'both': return results
    return results[backend]

def _render_plotly(data: FigureData) -> go.Figure:
    fig = make_subplots(
        rows=data.n_rows, cols=data.n_cols,
        subplot_titles=[s.title for s in data.subplots if s.title],
        vertical_spacing=0.08, horizontal_spacing=0.05
    )

    for sub in data.subplots:
        for trace in sub.traces:
            fig.add_trace(
                go.Scatter(
                    x=trace.x, y=trace.y,
                    mode='lines',
                    line=dict(color=trace.color, width=trace.width),
                    opacity=trace.alpha,
                    name=trace.label,
                    legendgroup=trace.legend_group,
                    showlegend=trace.show_legend,
                    hovertemplate=(
                        f'<b>{trace.hover_header}</b><br>' +
                        f'<b>Time:</b> %{{x:.2f}}<br>' +
                        f'<b>{trace.hover_detail}</b><br><extra></extra>'
                    ) if trace.hover_header else None
                ),
                row=sub.row, col=sub.col
            )
        
        fig.update_xaxes(range=sub.xlim, title_text=sub.x_label, row=sub.row, col=sub.col)
        fig.update_yaxes(range=sub.ylim, title_text=sub.y_label, row=sub.row, col=sub.col)

    if data.row_labels and data.n_rows > 1:
        for idx, label in enumerate(data.row_labels, start=1):
            y_pos = 1 - (idx - 0.5) / data.n_rows
            fig.add_annotation(
                text=f"<b>{label}</b>", xref="paper", yref="paper",
                x=-0.06, y=y_pos, showarrow=False, xanchor="right", font=dict(size=12)
            )

    # Set legend title from color_by_grouping parameter
    legend_config = dict(x=1.02, y=1)
    if data.color_by_grouping:
        legend_config['title'] = dict(text=data.color_by_grouping)
    
    fig.update_layout(
        title_text=data.title, height=data.height, width=data.width,
        hovermode='closest', template="plotly_white", legend=legend_config
    )
    return fig

def _render_matplotlib(data: FigureData) -> plt.Figure:
    from matplotlib.lines import Line2D
    
    figsize = (5 * data.n_cols, 4.5 * data.n_rows)
    fig, axes = plt.subplots(data.n_rows, data.n_cols, figsize=figsize, squeeze=False)
    
    # Collect all unique legend entries across ALL subplots
    # Use dict to preserve order while ensuring uniqueness (label -> color)
    legend_entries = {}
    
    for sub in data.subplots:
        ax = axes[sub.row - 1][sub.col - 1]
        
        has_data = False
        for trace in sub.traces:
            has_data = True
            ax.plot(
                trace.x, trace.y, color=trace.color, alpha=trace.alpha,
                linewidth=trace.width,
                zorder=trace.zorder
            )
            
            # Collect legend entries from trend lines (thick lines with labels)
            if trace.label and trace.width >= MEAN_TRACE_LINEWIDTH:
                if trace.label not in legend_entries:
                    legend_entries[trace.label] = trace.color

        if not has_data:
             ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                     transform=ax.transAxes, fontsize=10, color='lightgray')

        ax.set_xlim(sub.xlim)
        ax.set_ylim(sub.ylim)
        if sub.title: ax.set_title(sub.title, fontweight='bold', fontsize=11)
        if sub.x_label: ax.set_xlabel(sub.x_label, fontsize=10)
        if sub.y_label: ax.set_ylabel(sub.y_label, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Create unified figure-level legend with all groups
    # Anchor relative to the rightmost facet so it truly borders the subplot grid.
    if legend_entries:
        legend_handles = [
            Line2D([0], [0], color=color, linewidth=MEAN_TRACE_LINEWIDTH, label=label)
            for label, color in legend_entries.items()
        ]

        rightmost_ax = axes[0, -1]
        fig.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            bbox_transform=rightmost_ax.transAxes,
            borderaxespad=0.0,
            fontsize=14,
            frameon=True,
            framealpha=0.9,
        )

        # Make space for the legend just to the right of the facets.
        plt.subplots_adjust(right=0.82)

    if data.row_labels and data.n_rows > 1:
        for idx, label in enumerate(data.row_labels):
            y_pos = 1 - (idx + 0.5) / data.n_rows
            fig.text(0.02, y_pos, label, rotation=90, va='center', ha='right',
                     fontsize=12, fontweight='bold', transform=fig.transFigure)

    fig.suptitle(data.title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 0.82 if legend_entries else 1, 0.96])
    return fig
    
# ==============================================================================
# 5. Proportion Grid Plot (Categorical Breakdown)
# ==============================================================================

def plot_proportion_grid(
    df: pd.DataFrame,
    col_by: str,
    row_by: List[str],
    count_by: str = 'embryo_id',
    col_order: Optional[List] = None,
    row_labels: Optional[Dict[str, str]] = None,
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',
    color_palette: Optional[Dict[str, Dict[str, str]]] = None,
) -> plt.Figure:
    """
    Plot proportion breakdown of categorical variables across groups.

    Creates a grid where:
    - Each column = one value of col_by (e.g., cluster)
    - Each row = one categorical variable from row_by list
    - Each cell = bar chart showing proportions of that categorical

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with categorical columns.
    col_by : str
        Column defining the grid columns (e.g., 'cluster').
    row_by : List[str]
        List of categorical column names for rows (e.g., ['genotype', 'pair']).
    count_by : str
        Column to count unique values of (default: 'embryo_id').
    col_order : Optional[List]
        Explicit ordering for col_by values.
    row_labels : Optional[Dict[str, str]]
        Display labels for row_by columns (e.g., {'experiment_id': 'Experiment'}).
    height_per_row : int
        Height in pixels per row.
    width_per_col : int
        Width in pixels per column.
    output_path : Optional[Path]
        If provided, saves figure to this path.
    title : Optional[str]
        Figure title.
    normalize : bool
        If True, show proportions (0-1). If False, show raw counts.
    bar_mode : str
        'stacked' for single stacked bar, 'grouped' for side-by-side bars.
    color_palette : Optional[Dict[str, Dict[str, str]]]
        Nested dict of {row_category: {value: color}}. If None, uses defaults.

    Returns
    -------
    plt.Figure
        The matplotlib figure.

    Example
    -------
    >>> fig = plot_proportion_grid(
    ...     df,
    ...     col_by='cluster',
    ...     row_by=['genotype', 'pair', 'experiment_id'],
    ...     count_by='embryo_id',
    ...     bar_mode='grouped',  # or 'stacked'
    ...     title='Cluster Composition Breakdown',
    ... )
    """
    # Determine column values (clusters)
    col_values = sorted(df[col_by].unique())
    if col_order:
        col_values = [c for c in col_order if c in col_values]

    n_rows = len(row_by)
    n_cols = len(col_values)

    # Build color palettes for each categorical
    palettes = {}
    for cat in row_by:
        if color_palette and cat in color_palette:
            palettes[cat] = color_palette[cat]
        elif cat == 'genotype':
            # Use genotype-specific colors
            unique_vals = sorted(df[cat].dropna().unique())
            palettes[cat] = {v: get_color_for_genotype(str(v)) for v in unique_vals}
        else:
            unique_vals = sorted(df[cat].dropna().unique())
            palettes[cat] = {v: STANDARD_PALETTE[i % len(STANDARD_PALETTE)]
                           for i, v in enumerate(unique_vals)}

    # Create figure
    fig_width = max(6, n_cols * (width_per_col / 100))
    fig_height = max(4, n_rows * (height_per_row / 100))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    # Track legend entries per row (categorical)
    row_legend_entries = {cat: {} for cat in row_by}

    for r_idx, cat in enumerate(row_by):
        cat_values = sorted(df[cat].dropna().unique())

        for c_idx, col_val in enumerate(col_values):
            ax = axes[r_idx, c_idx]

            # Filter to this column value
            subset = df[df[col_by] == col_val]

            # Count unique count_by values per category
            counts = subset.groupby(cat)[count_by].nunique()

            # Reindex to ensure all category values present
            counts = counts.reindex(cat_values, fill_value=0)

            if normalize and counts.sum() > 0:
                proportions = counts / counts.sum()
            else:
                proportions = counts

            # Draw bars based on mode
            n_bars = len(cat_values)

            if bar_mode == 'grouped':
                # Side-by-side bars
                bar_width = 0.8 / n_bars
                x_positions = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, n_bars)

                for i, val in enumerate(cat_values):
                    height = proportions.get(val, 0)
                    color = palettes[cat].get(val, STANDARD_PALETTE[0])
                    ax.bar(x_positions[i], height, width=bar_width,
                           color=color, edgecolor='white', linewidth=0.5)

                    # Track for legend
                    if val not in row_legend_entries[cat]:
                        row_legend_entries[cat][val] = color

                # Y-axis for grouped mode
                max_val = proportions.max() if len(proportions) > 0 else 1
                ax.set_ylim(0, 1.05 if normalize else max_val * 1.1)
            else:
                # Stacked bar (default)
                bottom = 0
                bar_width = 0.6
                for val in cat_values:
                    height = proportions.get(val, 0)
                    if height > 0:
                        color = palettes[cat].get(val, STANDARD_PALETTE[0])
                        ax.bar(0, height, bottom=bottom, width=bar_width,
                               color=color, edgecolor='white', linewidth=0.5)

                        # Track for legend
                        if val not in row_legend_entries[cat]:
                            row_legend_entries[cat][val] = color

                        bottom += height

                ax.set_ylim(0, 1 if normalize else counts.sum() * 1.05)

            # Styling
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])

            # Column titles on top row
            if r_idx == 0:
                ax.set_title(f'{col_by}={col_val}', fontsize=10, fontweight='bold')

            # Y-axis labels on left column only
            if c_idx == 0:
                label = row_labels.get(cat, cat) if row_labels else cat
                ax.set_ylabel(label, fontsize=10, fontweight='bold')
                if normalize:
                    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            else:
                ax.set_yticks([])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    # Add legends for each row on the right side
    for r_idx, cat in enumerate(row_by):
        if row_legend_entries[cat]:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5)
                for val, color in row_legend_entries[cat].items()
            ]
            legend_labels = list(row_legend_entries[cat].keys())

            # Position legend to the right of this row
            ax_right = axes[r_idx, -1]
            ax_right.legend(
                legend_handles, legend_labels,
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),
                fontsize=8,
                frameon=True,
                framealpha=0.9,
            )

    # Title
    final_title = title or f'Proportion Breakdown by {col_by}'
    fig.suptitle(final_title, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')

    return fig


# ==============================================================================
# 5. Proportion Faceted Plot (Consistent API with plot_trajectories_faceted)
# ==============================================================================

def plot_proportion_faceted(
    df: pd.DataFrame,
    color_by_grouping: str,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    count_by: str = 'embryo_id',
    facet_order: Optional[Dict[str, List]] = None,
    color_order: Optional[List] = None,
    color_palette: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',
    height_per_row: int = HEIGHT_PER_ROW,
    width_per_col: int = WIDTH_PER_COL,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_counts: bool = True,
) -> plt.Figure:
    """
    Plot proportion breakdown with faceted grid structure.

    API is consistent with plot_trajectories_faceted:
    - row_by/col_by define facet grid by column VALUES (not variable names)
    - color_by_grouping defines the categorical for bar colors

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with categorical columns.
    color_by_grouping : str
        Column name for bar coloring (e.g., 'phenotype').
    row_by : Optional[str]
        Column defining grid rows (e.g., 'genotype'). None = single row.
    col_by : Optional[str]
        Column defining grid columns (e.g., 'pair'). None = single column.
    count_by : str
        Column to count unique values of (default: 'embryo_id').
    facet_order : Optional[Dict[str, List]]
        Ordering for facet values. Keys: 'row_by', 'col_by'.
        Example: {'row_by': ['wildtype', 'het', 'hom'], 'col_by': ['pair_2', 'pair_7']}
    color_order : Optional[List]
        Order of color_by_grouping values (bar order and legend order).
    color_palette : Optional[Dict[str, str]]
        Color mapping {value: hex_color}. Uses STANDARD_PALETTE if not provided.
    normalize : bool
        If True, show proportions (0-1). If False, show raw counts.
    bar_mode : str
        'grouped' for side-by-side bars, 'stacked' for stacked bars.
    height_per_row : int
        Height in pixels per row.
    width_per_col : int
        Width in pixels per column.
    output_path : Optional[Path]
        If provided, saves figure to this path.
    title : Optional[str]
        Figure title.
    show_counts : bool
        If True, show count annotations on bars.

    Returns
    -------
    plt.Figure
        The matplotlib figure.

    Example
    -------
    >>> fig = plot_proportion_faceted(
    ...     df,
    ...     color_by_grouping='phenotype',
    ...     row_by='genotype_suffix',
    ...     col_by='pair',
    ...     color_palette={
    ...         'CE': '#9467BD',
    ...         'HTA': '#17BECF',
    ...         'BA_rescue': '#E377C2',
    ...         'non_penetrant': '#7F7F7F',
    ...     },
    ...     facet_order={
    ...         'row_by': ['wildtype', 'heterozygous', 'homozygous'],
    ...         'col_by': ['pair_2', 'pair_7', 'pair_8'],
    ...     },
    ...     title='Phenotype Distribution by Pair and Genotype',
    ... )
    """
    # Determine row and column values
    if row_by is not None:
        row_values = sorted(df[row_by].dropna().unique())
        if facet_order and 'row_by' in facet_order:
            row_values = [v for v in facet_order['row_by'] if v in row_values]
    else:
        row_values = [None]

    if col_by is not None:
        col_values = sorted(df[col_by].dropna().unique())
        if facet_order and 'col_by' in facet_order:
            col_values = [v for v in facet_order['col_by'] if v in col_values]
    else:
        col_values = [None]

    # Determine color_by_grouping values
    color_values = sorted(df[color_by_grouping].dropna().unique())
    if color_order:
        color_values = [v for v in color_order if v in color_values]

    # Build color palette
    if color_palette is None:
        color_palette = {v: STANDARD_PALETTE[i % len(STANDARD_PALETTE)]
                        for i, v in enumerate(color_values)}

    n_rows = len(row_values)
    n_cols = len(col_values)

    # Figure sizing
    fig_width = max(6, n_cols * (width_per_col / 100))
    fig_height = max(4, n_rows * (height_per_row / 100))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    # Legend tracking
    legend_handles = []
    legend_labels = []

    for r_idx, row_val in enumerate(row_values):
        for c_idx, col_val in enumerate(col_values):
            ax = axes[r_idx, c_idx]

            # Filter to this cell
            subset = df.copy()
            if row_by is not None and row_val is not None:
                subset = subset[subset[row_by] == row_val]
            if col_by is not None and col_val is not None:
                subset = subset[subset[col_by] == col_val]

            # Count unique count_by values per color_by_grouping category
            counts = subset.groupby(color_by_grouping)[count_by].nunique()
            counts = counts.reindex(color_values, fill_value=0)

            total = counts.sum()
            if normalize and total > 0:
                proportions = counts / total
            else:
                proportions = counts

            # Draw bars
            n_bars = len(color_values)

            if bar_mode == 'grouped':
                bar_width = 0.8 / max(n_bars, 1)
                x_positions = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, n_bars) if n_bars > 0 else []

                for i, val in enumerate(color_values):
                    height = proportions.get(val, 0)
                    color = color_palette.get(val, STANDARD_PALETTE[0])

                    bar = ax.bar(x_positions[i], height, width=bar_width,
                                color=color, edgecolor='white', linewidth=0.5)

                    # Add count annotation if requested
                    if show_counts and counts.get(val, 0) > 0:
                        ax.annotate(f'{int(counts.get(val, 0))}',
                                   xy=(x_positions[i], height),
                                   ha='center', va='bottom',
                                   fontsize=7, color='black')

                    # Track for legend (only once)
                    if r_idx == 0 and c_idx == 0:
                        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                                                           facecolor=color,
                                                           edgecolor='white',
                                                           linewidth=0.5))
                        legend_labels.append(val)

                # Y-axis limits for grouped mode
                max_val = proportions.max() if len(proportions) > 0 else 1
                ax.set_ylim(0, 1.15 if normalize else max_val * 1.2)

            else:  # stacked
                bottom = 0
                bar_width = 0.6

                for val in color_values:
                    height = proportions.get(val, 0)
                    if height > 0:
                        color = color_palette.get(val, STANDARD_PALETTE[0])
                        ax.bar(0, height, bottom=bottom, width=bar_width,
                              color=color, edgecolor='white', linewidth=0.5)

                        # Track for legend (only once)
                        if r_idx == 0 and c_idx == 0 and val not in legend_labels:
                            legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                                                               facecolor=color,
                                                               edgecolor='white',
                                                               linewidth=0.5))
                            legend_labels.append(val)

                        bottom += height

                ax.set_ylim(0, 1.05 if normalize else total * 1.1)

            # Styling - tighter bar spacing
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])

            # Column titles on top row
            if r_idx == 0 and col_by is not None and col_val is not None:
                ax.set_title(f'{col_val}', fontsize=10, fontweight='bold')

            # Row labels on left column
            if c_idx == 0 and row_by is not None and row_val is not None:
                ax.set_ylabel(f'{row_val}', fontsize=10, fontweight='bold')

            # Y-axis formatting - set ticks on ALL facets for grid alignment
            if normalize:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                if c_idx == 0:
                    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
                else:
                    ax.set_yticklabels([])  # Hide labels but keep ticks for grid
            else:
                if c_idx != 0:
                    ax.set_yticklabels([])

            # Add horizontal grid lines across all facets
            ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
            ax.set_axisbelow(True)

            # Clean up spines - keep bottom for baseline
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('gray')

    # Add single legend for all panels
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='center right',
            bbox_to_anchor=(0.98, 0.5),
            fontsize=9,
            frameon=True,
            framealpha=0.9,
            title=color_by_grouping,
            title_fontsize=10,
        )

    # Title
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    else:
        parts = []
        if col_by:
            parts.append(f'by {col_by}')
        if row_by:
            parts.append(f'and {row_by}')
        default_title = f'{color_by_grouping} Distribution {" ".join(parts)}'
        fig.suptitle(default_title, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.82, 0.96])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')

    return fig

