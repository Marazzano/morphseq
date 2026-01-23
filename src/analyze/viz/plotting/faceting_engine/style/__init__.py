"""Style module for faceted plotting."""

from .colors import STANDARD_PALETTE, normalize_color, to_rgba_string, create_color_lookup
from .defaults import StyleSpec, default_style, paper_style

__all__ = [
    'STANDARD_PALETTE', 'normalize_color', 'to_rgba_string', 'create_color_lookup',
    'StyleSpec', 'default_style', 'paper_style',
]
