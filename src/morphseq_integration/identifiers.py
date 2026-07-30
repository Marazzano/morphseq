"""Sequencing-side identifier grammar — the one place the seq key is minted.

The sci-PLEX embryo key is ``{sci_expt}_P{plate:02d}_{hash_well}``, where ``hash_well`` is
NOT zero-padded — e.g. ``GENE7_P01_A1``. morphseq's own canonical well form IS zero-padded
(``A01``), so exactly one boundary conversion is needed and it lives here.

Two normalizations that previously lived inline in notebooks (and drifted between them):

- Plate sheets carry hash wells however the curator typed them (``A1``, ``a01``, ``A01``).
  ``normalize_hash_well`` routes every spelling through the pipeline's own well grammar, so a
  ``well_index`` minted here is byte-identical to one minted by the plate loader.
- Newer sequencing corpora suffix ``embryo_ID`` with an RT block (``GENE7_P18_A1_Bl1``) while the
  sample-level key drops it. ``strip_rt_block`` is the single place that difference is absorbed.

Every function fails loud rather than guessing: a hash well outside A-H x 1-12, or a plate number
that is not a positive integer, is a data-entry error worth surfacing at build time.
"""

from __future__ import annotations

import re

import pandas as pd

from data_pipeline.shared.identifiers import normalize_well_index

# A free-form well label: row letter + 1-2 digit column. Range is enforced by normalize_well_index.
_WELL_LABEL_RE = re.compile(r"^([A-Za-z])\s*0*(\d{1,2})$")

# A hash plate as authored: bare number (1, 18) or already-formatted (P1, P01).
_PLATE_RE = re.compile(r"^[Pp]?0*(\d{1,3})$")

# Trailing RT-block suffix on a sequencing embryo_ID: GENE7_P18_A1_Bl1 -> GENE7_P18_A1
_RT_BLOCK_RE = re.compile(r"_[Bb][Ll]\d+$")

PLATE_PREFIX = "P"
_PLATE_WIDTH = 2


def normalize_hash_well(value: object) -> str:
    """Return the canonical zero-padded morphseq well form (``A1`` / ``a01`` -> ``A01``).

    Routes through ``data_pipeline.shared.identifiers.normalize_well_index`` so the result is
    byte-identical to a ``well_index`` minted anywhere else in the pipeline.

    Raises:
        ValueError: if ``value`` is not a recognizable well label, or is out of range.
    """
    text = str(value).strip()
    match = _WELL_LABEL_RE.match(text)
    if match is None:
        raise ValueError(
            f"[morphseq_integration] hash well {value!r} is not a recognizable well label "
            "(expected a letter A-H followed by a column number, e.g. 'A1' or 'A01')."
        )
    return normalize_well_index(match.group(1), int(match.group(2)))


def to_sequencing_well(well_index: object) -> str:
    """Convert a canonical padded well (``A01``) to the unpadded sequencing form (``A1``).

    Accepts any spelling ``normalize_hash_well`` accepts, so this is safe to call on raw sheet
    values as well as on already-canonical ones.
    """
    canonical = normalize_hash_well(well_index)
    return f"{canonical[0]}{int(canonical[1:])}"


def format_hash_plate(value: object) -> str:
    """Normalize an authored hash plate to the sequencing form (``1`` / ``P1`` / ``P01`` -> ``P01``).

    Raises:
        ValueError: if ``value`` is not a positive integer plate number, with or without a
            ``P`` prefix.
    """
    text = str(value).strip()
    # Tolerate float-ish spellings that Excel/pandas produce for integer cells (18 -> '18.0').
    if text.endswith(".0"):
        text = text[:-2]
    match = _PLATE_RE.match(text)
    if match is None:
        raise ValueError(
            f"[morphseq_integration] hash plate {value!r} is not a recognizable plate number "
            "(expected a positive integer, optionally 'P'-prefixed, e.g. 1, 18, 'P01')."
        )
    number = int(match.group(1))
    if number < 1:
        raise ValueError(
            f"[morphseq_integration] hash plate {value!r} resolves to {number}; "
            "plate numbers are 1-based."
        )
    return f"{PLATE_PREFIX}{number:0{_PLATE_WIDTH}d}"


def build_seq_sample_id(sci_expt: str, hash_plate: object, hash_well: object) -> str:
    """Mint the canonical sequencing sample key: ``{sci_expt}_P{plate:02d}_{unpadded_well}``.

    Example: ``build_seq_sample_id("GENE7", 1, "A01") == "GENE7_P01_A1"``.

    Raises:
        ValueError: on an empty ``sci_expt``, or an unparseable plate or well.
    """
    expt = str(sci_expt).strip()
    if not expt:
        raise ValueError("[morphseq_integration] sci_expt must be non-empty to mint a sample id.")
    return f"{expt}_{format_hash_plate(hash_plate)}_{to_sequencing_well(hash_well)}"


def strip_rt_block(embryo_id: object) -> str:
    """Drop a trailing RT-block suffix so a sequencing ``embryo_ID`` reduces to its sample key.

    ``GENE7_P18_A1_Bl1`` -> ``GENE7_P18_A1``. Ids without the suffix pass through unchanged, so
    this is safe to apply uniformly across sequencing corpus generations.
    """
    return _RT_BLOCK_RE.sub("", str(embryo_id).strip())


def is_blank(value: object) -> bool:
    """True if a sheet cell carries no authored value (NaN, None, or whitespace-only)."""
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))
