"""
Color utilities for faceted plotting.

NO pandas imports. Engine stays pure.
"""

from typing import Sequence, Dict, Any, Optional, List
import matplotlib.colors as mcolors


STANDARD_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def normalize_color(color: Any) -> str:
    """Convert any color format to hex string."""
    try:
        return mcolors.to_hex(color)
    except (ValueError, TypeError):
        return str(color)


def to_rgba_string(color: Any, alpha: float = 1.0) -> str:
    """Convert any color to rgba() string for Plotly fill.
    
    Uses mcolors.to_rgba for robust parsing of all formats.
    """
    try:
        r, g, b, _ = mcolors.to_rgba(color)
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"
    except (ValueError, TypeError):
        return f"rgba(128,128,128,{alpha})"


def create_color_lookup(
    unique_values: Sequence[Any],
    palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Create value→color mapping from unique values.
    
    NOTE: Caller is responsible for extracting unique values
    and ordering. This function just assigns colors.
    """
    palette = palette or STANDARD_PALETTE
    return {v: palette[i % len(palette)] for i, v in enumerate(unique_values)}
