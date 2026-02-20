"""Compatibility package for legacy imports.

Historically modules lived under `src/functions/*.py` and were imported as:
    from src.functions.utilities import path_leaf

After refactoring, canonical implementations moved to `src/core/functions/*.py`.
This package keeps old import paths working by extending the package search path
to include `src/core/functions`.
"""

from __future__ import annotations

from pathlib import Path


_legacy_pkg_dir = Path(__file__).resolve().parent
_core_functions_dir = _legacy_pkg_dir.parent / "core" / "functions"

# Ensure `import src.functions.<module>` can find modules under src/core/functions.
if _core_functions_dir.is_dir():
    __path__.append(str(_core_functions_dir))  # type: ignore[name-defined]

