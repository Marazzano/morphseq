"""Canonical identifier validators.

Validators are the guard rails: they FAIL LOUDLY when an identifier does not
match the contract, so malformed/stale data cannot silently flow downstream.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations

import re

from .parsers import parse_physical_embryo_id


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


def validate_physical_embryo_id(physical_embryo_id: str, *, well_id: str | None = None) -> str:
    """Assert ``physical_embryo_id`` is well-formed; return it unchanged.

    The reusable STRING-level guard rail every snip/embryo-grain product leans on
    (registry, snips, features, QC). It answers exactly one question — "is this one
    identifier a valid ``{well_id}_e{local_embryo_index:02d}`` (one-based, ≥ 1)?" —
    and owns it for the whole pipeline so nobody re-regexes the grammar locally.

    Decomposes via ``parse_physical_embryo_id`` (which already fails loud on a bad
    shape or on ``_e00``/index < 1), then runs ``validate_well_id`` on the embedded
    well component so a leaked LOCAL well label is also caught here. When ``well_id``
    is supplied, cross-checks that the embedded well matches — the round-trip a table
    validator uses to prove ``physical_embryo_id`` agrees with its ``well_id`` column.
    """
    text = str(physical_embryo_id).strip()
    if not text:
        raise ValueError("physical_embryo_id is empty")
    embedded_well_id, _local_embryo_index = parse_physical_embryo_id(text)
    validate_well_id(embedded_well_id)
    if well_id is not None and embedded_well_id != str(well_id).strip():
        raise ValueError(
            f"physical_embryo_id {physical_embryo_id!r} encodes well_id {embedded_well_id!r} "
            f"but the supplied well_id is {well_id!r}. The embedded well_id must match the "
            "well_id column. A disagreement means the identity was minted against the wrong well."
        )
    return text
