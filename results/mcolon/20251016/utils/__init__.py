"""
DEPRECATED: Utils module has been migrated to src.analyze.utils

This module provides backward compatibility by re-exporting all functions
from the new location. Please update your imports to use the new path:

    from analyze.utils import (
        bin_embryos_by_time,
        filter_binned_data,
        load_experiment,
        load_experiments,
        filter_by_genotypes,
        get_genotype_summary,
        validate_required_columns,
        make_safe_comparison_name,
        get_plot_path,
        get_data_path,
        save_dataframe,
        extract_genotype_short_names,
    )

Notes
-----
This module will be removed in a future version. Migration to the new
location is recommended to take advantage of improved type hints,
documentation, and pathlib integration.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "results.mcolon.20251016.utils is deprecated. "
    "Please use analyze.utils instead. "
    "See docstring for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from analyze.utils import (
    bin_embryos_by_time,
    filter_binned_data,
    load_experiment,
    load_experiments,
    filter_by_genotypes,
    get_genotype_summary,
    validate_required_columns,
    make_safe_comparison_name,
    get_plot_path,
    get_data_path,
    save_dataframe,
    extract_genotype_short_names,
)

__all__ = [
    'bin_embryos_by_time',
    'filter_binned_data',
    'load_experiment',
    'load_experiments',
    'filter_by_genotypes',
    'get_genotype_summary',
    'validate_required_columns',
    'make_safe_comparison_name',
    'get_plot_path',
    'get_data_path',
    'save_dataframe',
    'extract_genotype_short_names',
]
