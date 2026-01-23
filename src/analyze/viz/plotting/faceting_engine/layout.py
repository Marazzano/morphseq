"""
Facet layout planning.

FacetSpec = layout behavior only (wrap, sharex, sharey, ordering, drop_empty)
LayoutPlan = computed grid positions + label suppression
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from .ir import FigureData, SubplotKey


@dataclass
class FacetSpec:
    """Layout behavior specification (NOT data slicing).
    
    NOTE: row_order/col_order are VALUES, not column names.
    Plot module determines which values exist and passes them here.
    """
    row_order: Optional[List[Any]] = None
    col_order: Optional[List[Any]] = None
    wrap: Optional[int] = None  # facet_wrap mode
    sharex: bool = True
    sharey: bool = True
    drop_empty: bool = False


@dataclass
class SubplotPosition:
    """Position info for a subplot."""
    row: int  # 1-based
    col: int  # 1-based
    show_x_label: bool
    show_y_label: bool


@dataclass
class LayoutPlan:
    """Computed layout for rendering."""
    n_rows: int
    n_cols: int
    positions: Dict[int, SubplotPosition]  # subplot_index → position
    row_labels: List[str]
    col_labels: List[str]


def plan_layout(
    fig_data: FigureData,
    facet: Optional[FacetSpec] = None,
) -> LayoutPlan:
    """Compute layout from figure data and facet spec.
    
    NOTE: Does NOT compute figure size. That's compute_figure_size().
    """
    facet = facet or FacetSpec()
    
    # Collect unique row/col values from subplots
    row_vals = []
    col_vals = []
    for sub in fig_data.subplots:
        row_val, col_val = sub.key
        if row_val is not None and row_val not in row_vals:
            row_vals.append(row_val)
        if col_val is not None and col_val not in col_vals:
            col_vals.append(col_val)
    
    # Apply ordering if specified
    if facet.row_order:
        row_vals = [v for v in facet.row_order if v in row_vals]
    if facet.col_order:
        col_vals = [v for v in facet.col_order if v in col_vals]
    
    # Handle single subplot case
    if not row_vals:
        row_vals = [None]
    if not col_vals:
        col_vals = [None]
    
    n_rows = len(row_vals)
    n_cols = len(col_vals)
    
    # Compute positions by subplot INDEX (not key)
    positions: Dict[int, SubplotPosition] = {}
    for idx, sub in enumerate(fig_data.subplots):
        row_val, col_val = sub.key
        row_idx = (row_vals.index(row_val) + 1) if row_val in row_vals else 1
        col_idx = (col_vals.index(col_val) + 1) if col_val in col_vals else 1
        
        # Label suppression
        show_x = (row_idx == n_rows) if facet.sharex else True
        show_y = (col_idx == 1) if facet.sharey else True
        
        positions[idx] = SubplotPosition(
            row=row_idx, col=col_idx,
            show_x_label=show_x, show_y_label=show_y,
        )
    
    # Row/col labels for facet strips
    row_labels = fig_data.row_labels or [str(v) for v in row_vals if v is not None]
    col_labels = fig_data.col_labels or [str(v) for v in col_vals if v is not None]
    
    return LayoutPlan(
        n_rows=n_rows,
        n_cols=n_cols,
        positions=positions,
        row_labels=row_labels,
        col_labels=col_labels,
    )


def compute_figure_size(
    n_rows: int,
    n_cols: int,
    style: "StyleSpec",
) -> tuple[int, int]:
    """Compute (height, width) from grid dimensions and style."""
    from .style.defaults import StyleSpec  # avoid circular
    height = max(style.min_height, n_rows * style.height_per_row)
    width = max(style.min_width, n_cols * style.width_per_col)
    return height, width
