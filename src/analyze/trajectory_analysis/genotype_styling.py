"""
DEPRECATED: This module has moved to trajectory_analysis.viz.styling

This file is a backward-compatibility shim. Please update your imports to use:
    from trajectory_analysis.viz.styling import (
        extract_genotype_suffix,
        extract_genotype_prefix,
        get_color_for_genotype,
        sort_genotypes_by_suffix,
        build_genotype_style_config,
        format_genotype_label,
    )
"""
import warnings

warnings.warn(
    "trajectory_analysis.genotype_styling is deprecated. "
    "Import from trajectory_analysis.viz.styling instead.",
    DeprecationWarning,
    stacklevel=2
)

from .viz.styling import (
    extract_genotype_suffix,
    extract_genotype_prefix,
    get_color_for_genotype,
    sort_genotypes_by_suffix,
    build_genotype_style_config,
    format_genotype_label,
)

__all__ = [
    'extract_genotype_suffix',
    'extract_genotype_prefix',
    'get_color_for_genotype',
    'sort_genotypes_by_suffix',
    'build_genotype_style_config',
    'format_genotype_label',
]
