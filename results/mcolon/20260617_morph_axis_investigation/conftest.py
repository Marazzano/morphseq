"""Project-wide pytest path setup for this investigation directory.

Mirrors the sys.path setup every entry-point script in this directory does by
hand (RUN_DIR + PROJECT_ROOT/src), so `morphseq_investigation.core.*` imports
(e.g. `analyze.utils.resampling` in `core/_resample_adapters.py`) resolve
under pytest without requiring an externally-exported PYTHONPATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))
