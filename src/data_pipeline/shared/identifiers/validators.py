"""Canonical identifier validators.

Validators are the guard rails: they FAIL LOUDLY when an identifier does not
match the contract, so malformed/stale data cannot silently flow downstream.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations

import re


# A local well label (e.g. A01, B12). A *bare* match is exactly what a global
# well_id must NOT be — it means an un-promoted local label leaked downstream.
_WELL_INDEX_RE = re.compile(r"^[A-Za-z]\d{1,3}$")


def validate_well_id(well_id: str) -> str:
    """Assert ``well_id`` is a well-formed GLOBAL well id; return it unchanged.

    The guard rail that stops a stale local ``A01`` from silently flowing once
    ``well_id`` is global (``{experiment_id}_{well_index}``). Fails loudly on a
    bare local label so an un-promoted id is caught at the boundary, not at a
    silently-empty join.
    """
    text = str(well_id).strip()
    if not text:
        raise ValueError("well_id is empty")
    if _WELL_INDEX_RE.match(text):
        raise ValueError(
            f"well_id {well_id!r} is a bare LOCAL well_index, not a global well_id "
            "({experiment_id}_{well_index}). A local label leaked past promotion."
        )
    if "_" not in text:
        raise ValueError(
            f"well_id {well_id!r} is not global: expected {{experiment_id}}_{{well_index}}."
        )
    return text
