"""
Shared utilities for faceted plotting.

Contains:
- Intermediate Representation (IR) dataclasses
- Color helpers
- Error band validation and computation
- Renderers (Plotly and Matplotlib)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from src.analyze.utils.data_processing import (
    get_trajectories_for_group,
    get_global_axis_ranges,
)
from src.analyze.utils.stats import compute_trend_line
from ...styling import get_color_for_genotype
from ....config import (
    DEFAULT_PLOTLY_HEIGHT,
    DEFAULT_PLOTLY_WIDTH,
    HEIGHT_PER_ROW,
    WIDTH_PER_COL,
    INDIVIDUAL_TRACE_ALPHA,
    INDIVIDUAL_TRACE_LINEWIDTH,
    MEAN_TRACE_LINEWIDTH,
)
from src.analyze.viz.styling import (
    GENOTYPE_SUFFIX_COLORS,
    GENOTYPE_SUFFIX_ORDER,
    PHENOTYPE_COLORS,
    PHENOTYPE_ORDER,
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
    """Represents a single line or band on a plot."""
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
    # Band support (for error bands / fill_between)
    band_lower: Optional[np.ndarray] = None
    band_upper: Optional[np.ndarray] = None
    render_as: str = 'line'  # 'line' or 'band'
    linestyle: str = '-'  # '-' for solid, '--' for dashed

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

def _normalize_color(color) -> str:
    """Convert any color format to hex string for Plotly compatibility.

    Handles:
    - Hex strings (pass through)
    - RGB/RGBA tuples (matplotlib format)
    - Named colors
    - rgb()/rgba() strings
    """
    import matplotlib.colors as mcolors

    # Already a valid hex string
    if isinstance(color, str):
        if color.startswith('#'):
            return color
        if color.startswith('rgb'):
            return color  # Plotly accepts rgb() strings
        # Try to convert named color to hex
        try:
            return mcolors.to_hex(color)
        except ValueError:
            return color  # Return as-is if conversion fails

    # Tuple (RGB or RGBA from matplotlib colormaps)
    if isinstance(color, (tuple, list)):
        try:
            return mcolors.to_hex(color)
        except ValueError:
            # Fallback: format as rgba string
            if len(color) == 4:
                r, g, b, a = color
                return f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})'
            elif len(color) == 3:
                r, g, b = color
                return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'

    return str(color)  # Last resort


def create_color_lookup(
    values: pd.Series,
    color_palette: Optional[List[str]] = None
) -> Dict:
    """Create color lookup dict, respecting Categorical ordering if present.

    Parameters
    ----------
    values : pd.Series
        Values to create color mapping for. If this is an ordered Categorical,
        the category order is preserved. Otherwise, order-of-first-occurrence
        is used.
    color_palette : Optional[List[str]]
        Custom color palette. Colors can be hex strings, RGB tuples,
        or any format matplotlib accepts. If None, uses STANDARD_PALETTE.

    Returns
    -------
    Dict mapping values to hex colors
    """
    palette = color_palette if color_palette is not None else STANDARD_PALETTE
    # Normalize all colors to hex for Plotly compatibility
    palette = [_normalize_color(c) for c in palette]

    # Respect Categorical ordering if present
    if hasattr(values, 'cat') and values.cat.ordered:
        present_cats = set(values.dropna().unique())
        unique_vals = [c for c in values.cat.categories if c in present_cats]
    else:
        unique_vals = list(pd.unique(values.dropna()))

    return {v: palette[i % len(palette)] for i, v in enumerate(unique_vals)}

def create_color_lookup_from_column(
    df: pd.DataFrame,
    column: Optional[str],
    color_palette: Optional[List[str]] = None
) -> Dict:
    """Build color lookup for a column, optionally using custom palette.

    Parameters
    ----------
    color_palette : Optional[List[str] or Dict]
        Can be a list of hex colors (converted to dict) or a dict mapping values to colors.
        If a list, colors are applied to values in categorical order if present.
    """
    if column is None or column not in df.columns:
        return {}

    # Auto-detect based on column name (if no custom palette)
    if column == 'genotype' and color_palette is None:
        # Use genotype suffix colors - extract suffix and map
        unique_vals = list(pd.unique(df[column].dropna()))
        lookup = {}
        for val in unique_vals:
            val_str = str(val)
            # Try to match genotype suffix (e.g., 'b9d2_homozygous' -> 'homozygous')
            matched = False
            for suffix in GENOTYPE_SUFFIX_ORDER:
                if val_str.endswith('_' + suffix) or val_str == suffix:
                    lookup[val] = GENOTYPE_SUFFIX_COLORS[suffix]
                    matched = True
                    break
            if not matched:
                # Fallback to standard palette
                lookup[val] = STANDARD_PALETTE[len(lookup) % len(STANDARD_PALETTE)]
        return lookup

    # Handle dict-based color_palette (pass through to make_color_lookup)
    if isinstance(color_palette, dict):
        return color_palette

    # Handle list-based color_palette (convert to dict like plot_proportions does)
    if isinstance(color_palette, (list, tuple)):
        col_series = df[column]
        # Respect Categorical ordering if present
        if hasattr(col_series, 'cat') and col_series.cat.ordered:
            present_cats = set(col_series.dropna().unique())
            unique_vals = [c for c in col_series.cat.categories if c in present_cats]
        else:
            unique_vals = list(pd.unique(col_series.dropna()))

        # Normalize colors to hex
        normalized_palette = [_normalize_color(c) for c in color_palette]
        return {v: normalized_palette[i % len(normalized_palette)]
                for i, v in enumerate(unique_vals)}

    return create_color_lookup(df[column], color_palette)

def create_color_state(
    df: pd.DataFrame,
    col_by: Optional[str],
    color_by_grouping: Optional[str],
    line_by: str,
    color_palette: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Build color state for all relevant columns.

    Parameters
    ----------
    color_palette : Optional[List[str]]
        Custom color palette. Applied to color_by_grouping if set,
        otherwise to col_by.
    """
    primary_column = color_by_grouping or col_by

    # Apply custom palette to the primary coloring column
    return {
        'primary_lookup': create_color_lookup_from_column(df, primary_column, color_palette),
        'col_lookup': create_color_lookup_from_column(df, col_by, color_palette if not color_by_grouping else None),
        'grouping_lookup': create_color_lookup_from_column(df, color_by_grouping, color_palette),
    }

def get_color_from_state(
    color_val_raw: Any,
    color_state: Dict[str, Dict],
    fallback_val: Any = None,
    mapping_type: str = 'primary_lookup'
) -> str:
    """Helper to safely lookup hex color from state."""
    if color_val_raw is not None:
        c = color_state[mapping_type].get(color_val_raw)
        if c: return c

    # Fallback
    if fallback_val is not None:
        # Try finding the fallback in any of the lookups
        c = (color_state['primary_lookup'].get(fallback_val) or
             color_state['grouping_lookup'].get(fallback_val) or
             color_state['col_lookup'].get(fallback_val))
        if c: return c

    return STANDARD_PALETTE[0]


# ==============================================================================
# Error Band Validation & Computation
# ==============================================================================

VALID_ERROR_TYPES = {
    'mean': ['sd', 'se'],
    'median': ['iqr', 'mad'],
}
"""Valid error_type values for each trend_statistic."""


def validate_error_type(trend_statistic: str, error_type: str) -> None:
    """
    Validate that error_type is compatible with trend_statistic.

    Parameters
    ----------
    trend_statistic : str
        Central tendency measure ('mean' or 'median')
    error_type : str
        Error measure ('sd', 'se' for mean; 'iqr', 'mad' for median)

    Raises
    ------
    ValueError
        If error_type is incompatible with trend_statistic
    """
    valid = VALID_ERROR_TYPES.get(trend_statistic, [])
    if error_type not in valid:
        raise ValueError(
            f"error_type='{error_type}' is incompatible with trend_statistic='{trend_statistic}'. "
            f"Valid options for '{trend_statistic}': {valid}"
        )


def compute_error_band(
    times: np.ndarray,
    metrics: np.ndarray,
    bin_width: float,
    statistic: str = 'median',
    error_type: str = 'iqr',
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute binned central tendency ± error for trajectories.

    Parameters
    ----------
    times : np.ndarray
        Time values (concatenated from all trajectories)
    metrics : np.ndarray
        Metric values (concatenated from all trajectories)
    bin_width : float
        Width of time bins for aggregation
    statistic : str
        Central tendency: 'mean' or 'median'
    error_type : str
        Error measure: 'sd'/'se' for mean, 'iqr'/'mad' for median

    Returns
    -------
    bin_times : np.ndarray or None
        Bin center times
    central_values : np.ndarray or None
        Mean or median per bin
    error_values : np.ndarray or None
        Error measure per bin (SD, SE, IQR/2, or MAD)
    """
    from scipy import stats as scipy_stats

    if len(times) == 0 or len(metrics) == 0:
        return None, None, None

    # Remove NaNs
    mask = ~(np.isnan(times) | np.isnan(metrics))
    times = times[mask]
    metrics = metrics[mask]

    if len(times) == 0:
        return None, None, None

    # Create bins
    t_min, t_max = times.min(), times.max()
    bins = np.arange(t_min, t_max + bin_width, bin_width)

    if len(bins) < 2:
        return None, None, None

    # Assign each point to a bin
    bin_indices = np.digitize(times, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    # Compute statistics per bin
    bin_times_list = []
    central_list = []
    error_list = []

    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        bin_values = metrics[bin_mask]

        if len(bin_values) < 2:
            continue

        bin_center = (bins[i] + bins[i + 1]) / 2
        bin_times_list.append(bin_center)

        if statistic == 'mean':
            central = np.mean(bin_values)
            if error_type == 'sd':
                error = np.std(bin_values, ddof=1)
            else:  # 'se'
                error = np.std(bin_values, ddof=1) / np.sqrt(len(bin_values))
        else:  # 'median'
            central = np.median(bin_values)
            if error_type == 'iqr':
                q75, q25 = np.percentile(bin_values, [75, 25])
                error = (q75 - q25) / 2  # Half IQR for symmetric band
            else:  # 'mad'
                error = np.median(np.abs(bin_values - central))

        central_list.append(central)
        error_list.append(error)

    if len(bin_times_list) == 0:
        return None, None, None

    return np.array(bin_times_list), np.array(central_list), np.array(error_list)


def compute_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Compute linear regression fit.

    Parameters
    ----------
    x : np.ndarray
        X values (typically bin times)
    y : np.ndarray
        Y values (typically trend line values)

    Returns
    -------
    x_fit : np.ndarray or None
        X values for fit line
    y_fit : np.ndarray or None
        Fitted Y values
    r_squared : float or None
        Coefficient of determination (R²)
    """
    from scipy.stats import linregress

    if x is None or y is None or len(x) < 2:
        return None, None, None

    # Remove NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return None, None, None

    try:
        result = linregress(x_clean, y_clean)
        y_fit = result.slope * x_clean + result.intercept
        r_squared = result.rvalue ** 2
        return x_clean, y_fit, r_squared
    except Exception:
        return None, None, None


def _compile_subplot_traces(
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
    # New parameters for error bands and linear fits
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    error_band_alpha: float = 0.2,
    show_linear_fit: bool = False,
) -> List[TraceData]:
    """Generates all traces (individual + means + error bands + fits) for a specific grid cell."""

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

        # Track the colors seen in this group to check for homogeneity
        seen_color_vals = set()

        # 1. Individual Traces (if enabled)
        if show_individual:
            for traj in trajectories:
                embryo_id = traj['embryo_id']

                # Simplified color resolution: use grouping value or column value
                line_color_val = grouping_val if color_by_grouping else col_val

                seen_color_vals.add(line_color_val)
                color_hex = get_color_from_state(line_color_val, color_state, fallback_val=col_val)

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
        else:
            # Still need to determine color even if not plotting individuals
            line_color_val = grouping_val if color_by_grouping else col_val
            seen_color_vals.add(line_color_val)

        # Concatenate all trajectory data for aggregate statistics
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])

        # Determine trend color
        if len(seen_color_vals) == 1:
            trend_color_val = list(seen_color_vals)[0]
        else:
            trend_color_val = grouping_val if color_by_grouping else col_val
        trend_color_hex = get_color_from_state(trend_color_val, color_state, fallback_val=col_val)

        # Label Logic
        if color_by_grouping:
            label = str(grouping_val)
            group_key = f"{y_col}_{grouping_val}"
        else:
            label = f"{col_by}={col_val}" if col_by else trend_statistic
            group_key = f"{y_col}_{col_val}"

        show_legend_for_group = group_key not in legend_tracker
        if show_legend_for_group:
            legend_tracker.add(group_key)

        # 2. Error Band (if enabled) - render BEFORE trend line so it's behind
        if show_error_band:
            band_times, central_vals, error_vals = compute_error_band(
                all_times, all_metrics, bin_width,
                statistic=trend_statistic,
                error_type=error_type
            )
            if band_times is not None and len(band_times) > 0:
                cell_traces.append(TraceData(
                    x=band_times,
                    y=central_vals,  # y is the center (for hover, etc.)
                    band_lower=central_vals - error_vals,
                    band_upper=central_vals + error_vals,
                    color=trend_color_hex,
                    width=0,  # No line width for band
                    alpha=error_band_alpha,
                    render_as='band',
                    show_legend=False,
                    zorder=3,
                ))

        # 3. Trend Line
        bin_times, bin_stats = compute_trend_line(
            all_times, all_metrics, bin_width,
            statistic=trend_statistic,
            smooth_sigma=trend_smooth_sigma
        )

        if bin_times is not None and len(bin_times) > 0:
            statistic_label = trend_statistic.capitalize()

            cell_traces.append(TraceData(
                x=bin_times,
                y=bin_stats,
                color=trend_color_hex,
                width=MEAN_TRACE_LINEWIDTH,
                alpha=1.0,
                label=label,
                legend_group=group_key,
                show_legend=show_legend_for_group,
                zorder=5,
                hover_header=f"{statistic_label}: {label}",
                hover_detail=f"{y_label_text}: %{{y:.4f}}",
            ))

            # 4. Linear Fit (if enabled)
            if show_linear_fit:
                x_fit, y_fit, r_squared = compute_linear_fit(bin_times, bin_stats)
                if x_fit is not None:
                    fit_label = f"{label} fit (R²={r_squared:.2f})"
                    cell_traces.append(TraceData(
                        x=x_fit,
                        y=y_fit,
                        color=trend_color_hex,
                        width=1.5,
                        alpha=0.8,
                        linestyle='--',
                        label=fit_label,
                        show_legend=False,  # Don't clutter legend with fit lines
                        zorder=6,
                        hover_header=f"Linear Fit: {label}",
                        hover_detail=f"R²={r_squared:.3f}",
                    ))

    return cell_traces

# ==============================================================================
# Renderers (Dumb & Backend Specific)
# ==============================================================================

def _render_figure(fig_data: FigureData, backend: str, output_path: Optional[Path]) -> Any:
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
            # Handle band (fill_between) vs line rendering
            if trace.render_as == 'band' and trace.band_lower is not None and trace.band_upper is not None:
                # For Plotly bands, we need upper line + lower line with fill='tonexty'
                # Add upper bound (invisible line)
                fig.add_trace(
                    go.Scatter(
                        x=trace.x, y=trace.band_upper,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=sub.row, col=sub.col
                )
                # Add lower bound with fill to upper
                # Convert hex color to rgba for fill
                hex_color = trace.color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                fill_color = f'rgba({r},{g},{b},{trace.alpha})'
                fig.add_trace(
                    go.Scatter(
                        x=trace.x, y=trace.band_lower,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=fill_color,
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=sub.row, col=sub.col
                )
            else:
                # Regular line trace
                line_dash = 'dash' if trace.linestyle == '--' else 'solid'
                fig.add_trace(
                    go.Scatter(
                        x=trace.x, y=trace.y,
                        mode='lines',
                        line=dict(color=trace.color, width=trace.width, dash=line_dash),
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

    # Add row labels on the left (rotated vertically)
    if data.row_labels and data.n_rows > 1:
        for idx, label in enumerate(data.row_labels, start=1):
            y_pos = 1 - (idx - 0.5) / data.n_rows
            fig.add_annotation(
                text=f"<b>{label}</b>",
                xref="paper", yref="paper",
                x=-0.06, y=y_pos,
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                textangle=-90,  # Rotate 90° counterclockwise
                font=dict(size=13)
            )

    # Add column labels on top (centered above each column)
    if data.col_labels and data.n_cols > 1:
        for idx, label in enumerate(data.col_labels, start=1):
            x_pos = (idx - 0.5) / data.n_cols
            fig.add_annotation(
                text=f"<b>{label}</b>",
                xref="paper", yref="paper",
                x=x_pos, y=1.02,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=13)
            )

    # Set legend title from color_by_grouping parameter
    legend_config = dict(x=1.02, y=1)
    if data.color_by_grouping:
        legend_config['title'] = dict(text=data.color_by_grouping)

    fig.update_layout(
        title_text=data.title, height=data.height, width=data.width,
        hovermode='closest', template="plotly_white", legend=legend_config,
        # Ensure row labels (left, rotated) and column labels (top) are visible
        margin=dict(l=140, r=140, t=100, b=70),
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

            # Handle band (fill_between) vs line rendering
            if trace.render_as == 'band' and trace.band_lower is not None and trace.band_upper is not None:
                ax.fill_between(
                    trace.x, trace.band_lower, trace.band_upper,
                    color=trace.color, alpha=trace.alpha,
                    zorder=trace.zorder
                )
            else:
                ax.plot(
                    trace.x, trace.y, color=trace.color, alpha=trace.alpha,
                    linewidth=trace.width, linestyle=trace.linestyle,
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
# Backward Compatibility (Deprecated Aliases)
# ==============================================================================

import warnings

def make_color_lookup(*args, **kwargs):
    """Deprecated: Use create_color_lookup instead."""
    warnings.warn(
        "make_color_lookup is deprecated. Use create_color_lookup instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_color_lookup(*args, **kwargs)

def build_color_lookup_for_column(*args, **kwargs):
    """Deprecated: Use create_color_lookup_from_column instead."""
    warnings.warn(
        "build_color_lookup_for_column is deprecated. Use create_color_lookup_from_column instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_color_lookup_from_column(*args, **kwargs)

def build_color_state(*args, **kwargs):
    """Deprecated: Use create_color_state instead."""
    warnings.warn(
        "build_color_state is deprecated. Use create_color_state instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_color_state(*args, **kwargs)

def resolve_color_value(*args, **kwargs):
    """Deprecated: Use get_color_from_state instead."""
    # Handle old parameter name
    if 'lookup_type' in kwargs:
        kwargs['mapping_type'] = kwargs.pop('lookup_type')
    warnings.warn(
        "resolve_color_value is deprecated. Use get_color_from_state instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_color_from_state(*args, **kwargs)
