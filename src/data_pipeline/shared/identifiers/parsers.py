"""Canonical identifier parsers.

Parsing is the inverse of minting: it DECOMPOSES an existing identifier back into
its parts. This is the only sanctioned place for that — it replaces ad-hoc
``.split("_")`` scattered through the call sites.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations

import re


_LOCAL_ID_RE = re.compile(r"(\d+)$")


def normalize_embryo_local_track_id(value: object) -> int:
    """Normalize tracker-native embryo IDs like ``embryo_0`` to an integer local track id."""
    if isinstance(value, bool):
        raise ValueError("embryo local track id cannot be boolean")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    match = _LOCAL_ID_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse embryo local track id from {value!r}")
    return int(match.group(1))


_WELL_INDEX_RE = re.compile(r"^[A-Za-z]\d{1,3}$")


def split_well_id(well_id: str) -> tuple[str, str]:
    """Decompose a GLOBAL ``well_id`` into ``(experiment_id, well_index)``.

    Inverse of ``build_well_id``: ``"20240418_A01" -> ("20240418", "A01")``. The
    well_index is the final underscore-delimited token (a local plate label like
    ``A01``); everything before it is the experiment id. This replaces the ad-hoc
    ``.rsplit("_", 1)`` / ``re.match(r"^(.+)_([A-H]\\d{2})$", …)`` scattered through
    the per-well boundaries and the legacy ``video_id`` parsing.
    """
    text = str(well_id).strip()
    if "_" not in text:
        raise ValueError(
            f"well_id {well_id!r} is not global ({{experiment_id}}_{{well_index}}); "
            "it looks like a bare local well_index."
        )
    experiment_id, well_index = text.rsplit("_", 1)
    if not experiment_id or not well_index:
        raise ValueError(f"Could not split well_id {well_id!r} into (experiment_id, well_index)")
    return experiment_id, well_index
