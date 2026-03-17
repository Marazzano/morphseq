"""Backward-compatibility aliases for legacy module paths.

Checkpoints and config files saved before the src/ → src/core/ refactor
embed old module paths (e.g. ``src.lightning.train_config``).  Importing
this module registers ``sys.modules`` aliases so that ``pickle.load`` /
``torch.load`` can resolve those paths transparently.

Usage
-----
Just import this module once before loading any legacy checkpoint::

    import src.core.run.compat  # noqa: F401
"""

import importlib
import sys

# Map of old module path → new module path.
# Add entries here as new mismatches are discovered.
_LEGACY_MODULE_MAP = {
    "src.lightning.train_config": "src.core.lightning.train_config",
}


def _register_aliases():
    for old, new in _LEGACY_MODULE_MAP.items():
        if old not in sys.modules:
            try:
                sys.modules[old] = importlib.import_module(new)
            except ImportError:
                pass  # silently skip if target doesn't exist


_register_aliases()
