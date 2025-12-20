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
    compute_binned_mean,
    get_global_axis_ranges,
)
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

        # 2. Mean Trace Logic
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])
        bin_times, bin_means = compute_binned_mean(all_times, all_metrics, bin_width)

        if bin_times is not None:
            # --- STRICT HOMOGENEITY CHECK ---
            mean_color_val = None

            # If every line in this group has the exact same color value,
            # we promote that color to the mean line.
            if len(seen_color_vals) == 1:
                mean_color_val = list(seen_color_vals)[0]
            else:
                # Heterogeneous group! Fallback to the grouping identity.
                # e.g. If "Pair 1" has mixed Genotypes, color it by "Pair 1" color, not Genotype.
                mean_color_val = grouping_val if color_by_grouping else col_val

            mean_color_hex = resolve_color_value(mean_color_val, color_state, fallback_val=col_val)

            # Label Logic
            if color_by_grouping:
                label = str(grouping_val)
                group_key = f"{y_col}_{grouping_val}"
            else:
                label = f"{col_by}={col_val}" if col_by else "mean"
                group_key = f"{y_col}_{col_val}"

            show_legend = group_key not in legend_tracker
            if show_legend:
                legend_tracker.add(group_key)

            cell_traces.append(TraceData(
                x=bin_times,
                y=bin_means,
                color=mean_color_hex,
                width=MEAN_TRACE_LINEWIDTH,
                alpha=1.0,
                label=label,
                legend_group=group_key,
                show_legend=show_legend,
                zorder=5,
                hover_header=f"Mean: {label}",
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
                bin_width, smooth_method, smooth_params, y_label, legend_tracker
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
                bin_width, smooth_method, smooth_params, current_y_label, legend_tracker
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

    fig.update_layout(
        title_text=data.title, height=data.height, width=data.width,
        hovermode='closest', template="plotly_white", legend=dict(x=1.02, y=1)
    )
    return fig

def _render_matplotlib(data: FigureData) -> plt.Figure:
    figsize = (5 * data.n_cols, 4.5 * data.n_rows)
    fig, axes = plt.subplots(data.n_rows, data.n_cols, figsize=figsize, squeeze=False)
    
    for sub in data.subplots:
        ax = axes[sub.row - 1][sub.col - 1]
        
        has_data = False
        for trace in sub.traces:
            has_data = True
            ax.plot(
                trace.x, trace.y, color=trace.color, alpha=trace.alpha,
                linewidth=trace.width, label=trace.label if trace.show_legend else None,
                zorder=trace.zorder
            )

        if not has_data:
             ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                     transform=ax.transAxes, fontsize=10, color='lightgray')

        ax.set_xlim(sub.xlim)
        ax.set_ylim(sub.ylim)
        if sub.title: ax.set_title(sub.title, fontweight='bold', fontsize=11)
        if sub.x_label: ax.set_xlabel(sub.x_label, fontsize=10)
        if sub.y_label: ax.set_ylabel(sub.y_label, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        if any(t.show_legend for t in sub.traces):
            ax.legend(fontsize=9, loc='upper right')

    if data.row_labels and data.n_rows > 1:
        for idx, label in enumerate(data.row_labels):
            y_pos = 1 - (idx + 0.5) / data.n_rows
            fig.text(0.02, y_pos, label, rotation=90, va='center', ha='right',
                     fontsize=12, fontweight='bold', transform=fig.transFigure)

    fig.suptitle(data.title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    return fig