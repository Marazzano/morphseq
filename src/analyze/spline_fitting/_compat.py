"""Backwards compatibility shims for old import paths.

This module provides deprecation warnings and re-exports for code that
imports from the old scattered locations. All new code should import from
src.analyze.spline_fitting instead.

Deprecated Import Paths:
    from src.functions.spline_fitting_v2 import LocalPrincipalCurve, spline_fit_wrapper
    from src.functions.spline_morph_spline_metrics import LocalPrincipalCurve
    from src.functions.embryo_df_performance_metrics import LocalPrincipalCurve

New Import Path:
    from src.analyze.spline_fitting import LocalPrincipalCurve, spline_fit_wrapper

This module is not part of the public API and should not be imported directly.
"""

import warnings


def _deprecation_warning(old_path, new_path):
    """Issue deprecation warning for old import paths."""
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Use 'from src.analyze.spline_fitting import {new_path}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Shims will be added to old files via modifications, not here.
# This module serves as documentation of the deprecation strategy.

__all__ = []  # Nothing exported directly
