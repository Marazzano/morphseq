"""Compatibility shims for legacy ``src.functions`` imports.

These modules were moved under ``src.core.functions``. Keep this package so
older build/pipeline code continues to import successfully.

Only ``utilities`` is eagerly imported; other submodules are available via
direct import (e.g. ``from src.functions.image_utils import ...``) but are
not bulk-loaded here to avoid pulling in heavy/unrelated dependencies.
"""

from src.functions.utilities import *  # noqa: F401,F403
