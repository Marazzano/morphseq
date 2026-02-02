"""
Visualization styling utilities and shared color palettes.
"""

from .lookup import ColorLookup, build_suffix_color_lookup
from .color_mapping_config import (
    GENOTYPE_SUFFIX_COLORS,
    GENOTYPE_SUFFIX_ORDER,
    GENOTYPE_COLORS,
    PHENOTYPE_COLORS,
    PHENOTYPE_ORDER,
)

__all__ = [
    'ColorLookup',
    'build_suffix_color_lookup',
    'GENOTYPE_SUFFIX_COLORS',
    'GENOTYPE_SUFFIX_ORDER',
    'GENOTYPE_COLORS',
    'PHENOTYPE_COLORS',
    'PHENOTYPE_ORDER',
]
