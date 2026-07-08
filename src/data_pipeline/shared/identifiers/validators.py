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

# Canonical well_index: exactly one letter A–H + exactly two digits 01–12.
_WELL_INDEX_CANONICAL_RE = re.compile(r"^([A-Ha-h])(\d{2})$")
_VALID_ROWS = frozenset("ABCDEFGHabcdefgh")
_VALID_COLS = frozenset(range(1, 13))


def normalize_well_index(row: str, col: int) -> str:
    """Return the canonical ``well_index`` string (e.g. ``"A01"``) for a grid cell.

    Validates the cell is inside the 8×12 plate range (rows A–H, columns 1–12)
    and returns the zero-padded canonical form.  Raises ``ValueError`` for any
    cell outside that range so callers never silently mint out-of-range labels.

    Args:
        row: Plate row letter — must be A–H (case-insensitive).
        col: Plate column number — must be 1–12 (inclusive).
    """
    row_upper = str(row).strip().upper()
    if row_upper not in _VALID_ROWS:
        raise ValueError(
            f"normalize_well_index: row {row!r} is out of range. "
            "Plate rows must be A–H (case-insensitive). "
            "Check whether the sheet has non-standard row labels."
        )
    col_int = int(col)
    if col_int not in _VALID_COLS:
        raise ValueError(
            f"normalize_well_index: column {col!r} is out of range. "
            "Plate columns must be 1–12. "
            "Check whether the sheet has non-standard column numbers."
        )
    return f"{row_upper}{col_int:02d}"


def validate_well_index(label: str) -> str:
    """Assert ``label`` is a canonical well_index (e.g. ``"A01"``); return it unchanged.

    A canonical well_index is exactly one letter A–H (upper-case) followed by a
    two-digit column 01–12.  Raises ``ValueError`` for anything outside this grammar
    so malformed labels are caught at the ingest boundary.
    """
    text = str(label).strip()
    m = _WELL_INDEX_CANONICAL_RE.match(text)
    if not m:
        raise ValueError(
            f"validate_well_index: {label!r} is not a canonical well_index. "
            "Expected format is one letter A–H (upper-case) followed by two digits, e.g. 'A01'. "
            "Use normalize_well_index(row, col) to mint a canonical label."
        )
    row_upper = m.group(1).upper()
    col_int = int(m.group(2))
    if row_upper not in _VALID_ROWS:
        raise ValueError(
            f"validate_well_index: row {row_upper!r} in {label!r} is outside A–H."
        )
    if col_int not in _VALID_COLS:
        raise ValueError(
            f"validate_well_index: column {col_int} in {label!r} is outside 1–12."
        )
    return f"{row_upper}{col_int:02d}"


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
