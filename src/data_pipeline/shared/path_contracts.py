"""Deprecated path-value helper module.

Do not route paths through this module. Pipeline roots come from
``pipeline_orchestrator/env.yaml`` and pipeline artifact paths come from
``pipeline_orchestrator/orchestration/paths.py``.

This file is intentionally left as a migration tripwire. Existing import sites should be fixed to
receive concrete paths, or to use the configured root from the Snakefile/tasks layer explicitly.
"""

from __future__ import annotations

import warnings


DEPRECATION_MESSAGE = (
    "data_pipeline.shared.path_contracts is deprecated and intentionally no longer resolves paths. "
    "Use pipeline_orchestrator/env.yaml for roots and "
    "pipeline_orchestrator/orchestration/paths.py for artifact paths. "
    "Pass concrete paths into loaders instead of using a shared hidden default."
)

warnings.warn(DEPRECATION_MESSAGE, RuntimeWarning, stacklevel=2)


def resolve_data_root_relative_path(*args: object, **kwargs: object) -> object:
    """Fail loudly; callers must use configured roots instead of hidden defaults."""
    raise RuntimeError(DEPRECATION_MESSAGE)


def require_existing_path(
    *args: object,
    **kwargs: object,
) -> object:
    """Fail loudly; callers must validate concrete configured paths."""
    raise RuntimeError(DEPRECATION_MESSAGE)
