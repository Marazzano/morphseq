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


def split_well_id(well_id: str) -> tuple[str, str]:
    """Decompose a GLOBAL ``well_id`` into ``(experiment_id, well)``.

    Scope-2 only. The CURRENT (Scope 1) ``well_id`` is the plate-LOCAL label
    (``A01``) — a single token with no ``experiment_id`` to split out — so this
    is intentionally not implemented yet. It activates when ``build_well_id``
    flips to ``(experiment_id, well)`` and ``well_id`` becomes
    ``{experiment_id}_{well}``. See
    target/well_id_throughline_refactor_plan.md (Scope 2).
    """
    raise NotImplementedError(
        "split_well_id requires GLOBAL well_id ({experiment_id}_{well}); "
        "activated in Scope 2. Today's well_id is the local label 'A01'."
    )
