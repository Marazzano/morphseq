"""Resolve an inventory's stored raw-source path, tolerant of the input tree having moved.

Acquisition inventories store the FULL absolute path of each raw source (the path as it existed at
ingest time). That is simple and exact at write time, but a stored absolute path goes stale if the
input tree is later moved/renamed/re-mounted. ``resolve_under_input_root`` bridges that: by default it
trusts the stored absolute path; with ``full_root_fallback=True`` it re-anchors a stale path onto the
CURRENT ``input_root`` by pivoting on the well-known ``raw_image_data/`` layout segment.

The pipeline organizes all input data under one ``input_root`` (raw images at
``raw_image_data/{scope}/{experiment}/...``), so the ``raw_image_data/`` segment is the stable pivot:
everything after it is the input-relative tail, independent of whatever root prefix was recorded.
"""

from __future__ import annotations

from pathlib import Path

# The layout segment every raw-image path contains; the pivot for re-anchoring onto input_root.
_RAW_IMAGE_DATA_SEGMENT = "raw_image_data"


def _input_relative_tail(stored: Path) -> Path | None:
    """Return the path from the ``raw_image_data/`` segment onward, or ``None`` if absent."""
    parts = stored.parts
    for i, part in enumerate(parts):
        if part == _RAW_IMAGE_DATA_SEGMENT:
            return Path(*parts[i:])
    return None


def resolve_under_input_root(
    stored: str | Path,
    *,
    input_root: str | Path | None,
    scope_label: str,
    full_root_fallback: bool = False,
) -> Path:
    """Resolve a stored raw-source path to an absolute ``Path`` for disk access.

    Default (``full_root_fallback=False``): return the stored path as-is (it is written absolute).

    ``full_root_fallback=True``: if the stored absolute path does not exist on disk, re-anchor it onto
    ``input_root`` by pivoting on the ``raw_image_data/`` segment
    (``input_root / raw_image_data/{scope}/{experiment}/...``) — so an inventory written before the
    input tree moved still resolves. Raises if the stored path is missing, fallback is on, but the
    path lacks a ``raw_image_data/`` segment (nothing to pivot on) or no ``input_root`` was given.

    ``scope_label`` names the calling contract so errors stay attributable.
    """
    path = Path(str(stored))

    if not full_root_fallback:
        return path
    if path.exists():
        return path

    # Stored path is stale — re-anchor its input-relative tail onto the current input_root.
    tail = _input_relative_tail(path)
    if tail is None:
        raise ValueError(
            f"[{scope_label}] stored source path {str(stored)!r} does not exist and has no "
            f"'{_RAW_IMAGE_DATA_SEGMENT}/' segment to re-anchor on. Cannot resolve under input_root."
        )
    if input_root is None:
        raise ValueError(
            f"[{scope_label}] stored source path {str(stored)!r} does not exist and full_root_fallback "
            "is on, but no input_root was provided to re-anchor against."
        )
    return (Path(input_root) / tail).resolve()
