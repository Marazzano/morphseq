"""Make the shared synthetic-mask builders importable from these tests.

This directory deliberately has NO ``__init__.py`` -- one here would shadow the real
``embryo_geometry`` package and break every import in it (``tests/image_geometry/``
follows the same rule). Without a package, ``import synthetic`` needs this directory on
``sys.path`` explicitly.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
