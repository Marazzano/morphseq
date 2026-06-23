"""Plate metadata loader (L1 ingest).

Reads an Excel workbook of plate-design sheets (genotype, medium, temperature, …)
and assembles a canonical long table — one row per ``well_index``, one column per
accepted page — without minting identity, enforcing the required-field schema, or
writing any side files.

Public surface::

    @dataclass(frozen=True)
    class PlatePages:
        table: pd.DataFrame              # long table, one row per well_index
        accepted: list[str]              # page names that became columns
        rejected: list[tuple[str, str]]  # (page_name, human reason)
        page_methods: dict[str, str]     # page_name -> "grid" | "long"

    def load_plate_metadata_pages(input_file: Path) -> PlatePages: ...

Design decisions (locked 2026-06-22):
- ``well_index`` is minted by ``shared/identifiers.normalize_well_index`` — never
  rolled inline.
- Provenance lives in ``PlatePages.page_methods``, NOT as a row-level column.
- Grid ingester is MVP; long ingester is recognized (classified) but rejected with
  a visible "not implemented in MVP" reason.
- Malformed grid pages raise; unrecognized pages are rejected (soft).
- No output-column collision across accepted pages (fail loud).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_pipeline.shared.identifiers import normalize_well_index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLATE_ROWS: tuple[str, ...] = tuple("ABCDEFGH")
_PLATE_COLS: tuple[int, ...] = tuple(range(1, 13))
_N_ROWS = len(_PLATE_ROWS)
_N_COLS = len(_PLATE_COLS)

_LONG_INGEST_NOT_IMPLEMENTED_REASON = (
    "long-table ingest is not implemented in MVP. "
    "To ingest this page, reformat it as a standard 8×12 grid "
    "(rows A–H as the first column, columns 1–12 as subsequent columns)."
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlatePages:
    """Result of loading all plate metadata pages from one workbook.

    ``table`` is the canonical long table: one row per ``well_index`` (e.g. A01),
    one column per accepted page name (normalized).  ``accepted`` lists every
    page that contributed a column; ``rejected`` is the list of
    ``(page_name, human_reason)`` pairs for pages that were skipped.
    ``page_methods`` records which ingest mode produced each accepted page.

    The ``table`` has NO row-level ``ingest_format`` column — provenance lives
    here in ``page_methods``, where it can differ per page.
    """

    table: pd.DataFrame
    accepted: list[str]
    rejected: list[tuple[str, str]]
    page_methods: dict[str, str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_plate_metadata_pages(input_file: Path) -> PlatePages:
    """Open ``input_file`` and return a ``PlatePages`` for all discovered sheets.

    Iterates every sheet in the workbook, classifies each one, dispatches to the
    appropriate ingester, outer-merges accepted pages by ``well_index``, and
    returns the result.  Non-plate pages are rejected (soft); malformed grid
    pages raise immediately (hard).

    Raises:
        FileNotFoundError: if ``input_file`` does not exist.
        ValueError: for hard failures (malformed grid, ambiguous format,
            output-column collision across accepted pages).
    """
    if not input_file.exists():
        raise FileNotFoundError(f"[plate_metadata_loader] input file not found: {input_file}")

    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []
    page_methods: dict[str, str] = {}
    page_frames: list[pd.DataFrame] = []

    with pd.ExcelFile(input_file) as xlf:
        for sheet_name in xlf.sheet_names:
            raw_df = _read_sheet_safely(xlf, sheet_name)
            if raw_df is None:
                rejected.append((sheet_name, "could not be read as a DataFrame"))
                continue

            page_class, reason = classify_plate_page(raw_df)

            if page_class == "grid":
                long_df = ingest_plate_grid_sheet_to_long(raw_df, page_name=sheet_name)
                col_name = _normalize_page_name(sheet_name)
                _assert_no_column_collision(col_name, accepted, sheet_name)
                accepted.append(col_name)
                page_methods[col_name] = "grid"
                page_frames.append(long_df.rename(columns={sheet_name: col_name}))

            elif page_class == "long":
                rejected.append((sheet_name, _LONG_INGEST_NOT_IMPLEMENTED_REASON))

            else:
                rejected.append((sheet_name, reason or "not a recognized plate page"))

    if not page_frames:
        table = _empty_well_index_table()
    elif len(page_frames) == 1:
        table = page_frames[0].copy()
    else:
        table = page_frames[0]
        for right in page_frames[1:]:
            table = table.merge(right, on="well_index", how="outer")
        table = table.sort_values("well_index").reset_index(drop=True)

    return PlatePages(
        table=table,
        accepted=accepted,
        rejected=rejected,
        page_methods=page_methods,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_plate_page(df: pd.DataFrame) -> tuple[str, str]:
    """Classify ``df`` as ``"grid"``, ``"long"``, or ``"rejected"``.

    Returns ``(classification, reason)`` where ``reason`` is an empty string for
    ``"grid"`` and ``"long"`` and a human explanation for ``"rejected"``.

    Raises:
        ValueError: if the page is readable as **both** grid and long (ambiguous).
    """
    looks_like_grid = _could_be_grid(df)
    looks_like_long = _could_be_long(df)

    if looks_like_grid and looks_like_long:
        raise ValueError(
            "classify_plate_page: sheet is readable as both a grid (8×12) and a "
            "long table (has well_index column). Ambiguous format — cannot pick one. "
            "Rename one column or restructure the sheet so only one format matches."
        )
    if looks_like_grid:
        return "grid", ""
    if looks_like_long:
        return "long", ""
    return "rejected", (
        "sheet does not match the 8×12 grid format (rows A–H × cols 1–12) "
        "or the long-table format (a derivable well_index column). "
        "Non-plate sheets (e.g. 'Export Summary') are skipped."
    )


def _could_be_grid(df: pd.DataFrame) -> bool:
    """Return True if ``df`` has ≥8 rows and ≥13 columns (room for row-labels + 1–12)."""
    return df.shape[0] >= _N_ROWS and df.shape[1] >= _N_COLS + 1


def _could_be_long(df: pd.DataFrame) -> bool:
    """Return True if ``df`` has a column whose name looks like a well identifier column."""
    if df.empty:
        return False
    lower_cols = {str(c).strip().lower() for c in df.columns}
    return bool(lower_cols & {"well_index", "well", "well_name"})


# ---------------------------------------------------------------------------
# Grid ingester (MVP)
# ---------------------------------------------------------------------------

def ingest_plate_grid_sheet_to_long(raw_df: pd.DataFrame, *, page_name: str) -> pd.DataFrame:
    """Flatten one 8×12 plate grid sheet into a long DataFrame.

    The first column is treated as row labels (A–H); the next 12 columns are
    treated as plate columns 1–12.  Validates that the grid is complete and
    sensibly laid out before minting any ``well_index`` values.

    Returns a two-column DataFrame: ``["well_index", page_name]``.

    Raises:
        ValueError: if rows are not A–H in sequence, columns are not 1–12 in
            sequence, the span is incomplete, or duplicate ``well_index`` values
            would result.
    """
    _assert_grid_has_enough_shape(raw_df, page_name)

    row_labels_raw = raw_df.iloc[:_N_ROWS, 0].tolist()
    _assert_rows_are_A_to_H_in_sequence(row_labels_raw, page_name)

    col_headers_raw = list(raw_df.columns[1: _N_COLS + 1])
    _assert_cols_are_1_to_12_in_sequence(col_headers_raw, page_name)

    value_block = raw_df.iloc[:_N_ROWS, 1: _N_COLS + 1]

    records = []
    for row_idx, row_letter in enumerate(_PLATE_ROWS):
        for col_idx, col_num in enumerate(_PLATE_COLS):
            well_idx = normalize_well_index(row_letter, col_num)
            value = value_block.iloc[row_idx, col_idx]
            records.append({"well_index": well_idx, page_name: value})

    long_df = pd.DataFrame(records)

    dup_mask = long_df.duplicated(subset=["well_index"], keep=False)
    if dup_mask.any():
        dups = long_df.loc[dup_mask, "well_index"].unique().tolist()
        raise ValueError(
            f"[plate_metadata_loader] grid page '{page_name}' produced duplicate "
            f"well_index values: {dups}. Each grid cell must map to a unique well."
        )

    return long_df


# ---------------------------------------------------------------------------
# Grid validation helpers
# ---------------------------------------------------------------------------

def _assert_grid_has_enough_shape(df: pd.DataFrame, page_name: str) -> None:
    if df.shape[0] < _N_ROWS or df.shape[1] < _N_COLS + 1:
        raise ValueError(
            f"[plate_metadata_loader] grid page '{page_name}' has shape {df.shape} "
            f"but needs at least ({_N_ROWS}, {_N_COLS + 1}) "
            "(8 data rows + row-label column + 12 value columns). "
            "Verify the sheet is a standard 8×12 plate layout."
        )


def _assert_rows_are_A_to_H_in_sequence(row_labels_raw: list, page_name: str) -> None:
    """Fail if the first-column labels are not exactly A–H in order (case-insensitive)."""
    seen = [str(r).strip().upper() for r in row_labels_raw]
    expected = list(_PLATE_ROWS)
    if seen != expected:
        raise ValueError(
            f"[plate_metadata_loader] grid page '{page_name}': "
            f"row labels are {seen!r} but must be exactly {expected!r} in sequence. "
            "Check for extra, missing, or out-of-order rows in the sheet."
        )


def _assert_cols_are_1_to_12_in_sequence(col_headers_raw: list, page_name: str) -> None:
    """Fail if the column headers are not exactly 1–12 in order."""
    try:
        seen = [int(c) for c in col_headers_raw]
    except (ValueError, TypeError):
        raise ValueError(
            f"[plate_metadata_loader] grid page '{page_name}': "
            f"column headers {col_headers_raw!r} cannot all be parsed as integers. "
            "Plate column headers must be integers 1–12."
        )
    expected = list(_PLATE_COLS)
    if seen != expected:
        raise ValueError(
            f"[plate_metadata_loader] grid page '{page_name}': "
            f"column headers are {seen!r} but must be exactly {expected!r} in sequence. "
            "Columns must be 1–12 with no gaps, extras, or re-ordering."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_sheet_safely(xlf: pd.ExcelFile, sheet_name: str) -> pd.DataFrame | None:
    try:
        return xlf.parse(sheet_name, header=0)
    except Exception:
        return None


def _normalize_page_name(sheet_name: str) -> str:
    """Lower-case, strip, replace spaces/hyphens with underscores."""
    return str(sheet_name).strip().lower().replace(" ", "_").replace("-", "_")


def _assert_no_column_collision(
    col_name: str, already_accepted: list[str], sheet_name: str
) -> None:
    if col_name in already_accepted:
        raise ValueError(
            f"[plate_metadata_loader] normalized page name '{col_name}' "
            f"(from sheet '{sheet_name}') collides with a previously accepted page. "
            "Each sheet must produce a unique output column. "
            "Rename one of the conflicting sheets in the workbook."
        )


def _empty_well_index_table() -> pd.DataFrame:
    """Return a table with only the 96 canonical well_index values and no value columns."""
    return pd.DataFrame({
        "well_index": [
            normalize_well_index(r, c) for r in _PLATE_ROWS for c in _PLATE_COLS
        ]
    })
