"""frame_detections report — STUB (retired for now).

The detections-per-well-over-time curtain was tried and retired: kept-detections-per-frame spans
only ~1–3, so 95 wells overplot onto 3 integer levels with no dynamic range to tell wells apart —
renderer A (grouped traces) needs a continuous value (like fraction_alive over 0–1) to be legible.

General lesson: a per-well/per-embryo trace ensemble is only useful when the plotted value has
enough distinct levels. Low-cardinality integer COUNTS (detections/frame, embryos/well) belong in a
distribution (renderer B / a histogram) or a heatmap, not a trace curtain.

Kept as a stub so the co-located slot exists. A future frame_detections report is more likely a
confidence histogram (renderer C, split by is_kept) than a per-well count trace — left for later.
"""

from __future__ import annotations

from pathlib import Path


def build(output_dir: Path) -> list[Path]:
    return []
