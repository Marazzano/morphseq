"""YX1 acquisition inventory — the maximal per-coordinate record of the ND2 tensor.

YX1 is a tensor: ``dask_arr[time, position, Z, channel, Y, X]``. A frame's address is a tuple of
indices into ONE ND2 file (Y/X are pixels). This module is the YX1 twin of the Keyence acquisition
concept file (``target/acquisition_inventory_flow.md``): it owns the YX1 inventory MODEL as pure
functions over rows, and the YX1 validator that DECLARES the schema + cell key and calls the shared
check primitives (``scope/shared/acquisition_checks.py``) for the mechanics.

Key decisions (see ``target/recompose_yx1_front_end.md`` Phase 1C):
  - **Record everything; collapse nothing.** One row per full tensor coordinate
    ``(position_index, z_index, channel_index, time_index)`` — Z is EXPLODED even though stitch
    LoG-projects it today, because the inventory is the system of record for a future per-Z pass.
  - **No per-plane file path** (``source_nd2_path`` only) — P/Z/C are array axes inside the ND2.
  - **The channel mapping lives here, once:** ``channel_index ↔ raw_channel_name ↔ channel``.
  - **No collision is possible** for YX1 (one ND2 cell per coordinate) — the uniqueness check is a
    defensive "is this scope really clean?" assertion, the YX1 analogue of the Keyence collision key.

Import direction: this module MAY import the shared check primitives + identifiers; it MUST NOT
import stages, Snakemake/tasks, stitch, or Keyence logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import nd2
import pandas as pd

from data_pipeline.metadata_ingest.scope.acquisition_inventory_contract import (
    REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS,
)
from data_pipeline.metadata_ingest.scope.shared.acquisition_checks import (
    assert_channel_mapping_consistent,
    assert_columns_present,
    assert_positive_column,
    assert_unique_on_key,
)
from data_pipeline.metadata_ingest.time_helpers import add_elapsed_time_columns

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Contract — schema + identity (the columns + the tensor cell key this scope produces)
# ─────────────────────────────────────────────────────────────────────────────────────────────

# YX1-specific Tier-2 columns (the tensor address + ND2 facts unique to YX1 — see the schema policy).
# These are ALLOWED + carried but are NOT the shared core; the shared core is hard-checked for every
# scope and lives in ``acquisition_inventory_contract.py``. ``acquisition_time_s`` is the YX1 raw time
# atom kept for audit/re-derivation; the CONVERGED ``elapsed_time_s`` is core (derived from it).
YX1_ACQUISITION_INVENTORY_SCOPE_COLUMNS: tuple[str, ...] = (
    "raw_position_label",       # ND2 P-index as string (PRE-mapping — no well_id at ingest)
    "z_index",                  # tensor Z axis (0..n_z-1) — exploded, never collapsed
    "channel_index",            # tensor C axis (numeric index into the ND2 channel list)
    "acquisition_time_s",       # raw per-frame ND2 timestamp (the atom elapsed_time_s is derived from)
    "x_um",                     # stage position (provenance; enables the join)
    "y_um",
    "objective_magnification",
    "n_z",                      # full Z depth of this acquisition (provenance)
    "source_nd2_path",          # the ONE ND2 (no per-plane path)
)

# The maximal per-coordinate schema = the SHARED Tier-1 core + the YX1 Tier-2 extras. Standardized
# ``*_index`` axis vocabulary — the inventory is a NEW artifact, born with target names (the legacy
# scope_metadata keeps time_int/z_position until the Scope-2 collapse).
YX1_ACQUISITION_INVENTORY_COLUMNS: tuple[str, ...] = (
    *REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS,
    *YX1_ACQUISITION_INVENTORY_SCOPE_COLUMNS,
)

# The tensor cell key — exactly one raw unit may occupy each cell. YX1 is clean by construction.
YX1_ACQUISITION_CELL_KEY: tuple[str, ...] = (
    "position_index",
    "z_index",
    "channel_index",
    "time_index",
)

_SCOPE_LABEL = "YX1 acquisition inventory"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Validation — fail-loud guards over the contract (schema/calibration/cell-key + source readability)
# ─────────────────────────────────────────────────────────────────────────────────────────────


def assert_acquisition_sources_readable(df: pd.DataFrame, *, scope_label: str) -> None:
    """Fail loud unless every ``source_nd2_path`` in the inventory still exists AND opens.

    The disk-touching half of the YX1 acquisition contract. ``source_nd2_path`` is a YX1
    acquisition-inventory field, so the readability check lives here (with the contract), not in the
    materializer that consumes it. It is intentionally NOT a ``scope/shared/acquisition_checks.py``
    primitive — those are pure DataFrame functions with no disk knowledge.

    Each UNIQUE path is checked once (cost is O(#ND2s), not O(#rows) — for YX1 that is one file).
    The check is "exists + ``nd2.ND2File`` opens"; it opens then closes, it does NOT read the tensor.
    A failure means the inventory CSV is fine but its raw source moved / was deleted / is corrupt
    since extraction — surfaced HERE as a named contract error instead of a deep backend traceback.
    """
    if "source_nd2_path" not in df.columns:
        raise ValueError(
            f"{scope_label}: cannot check source readability — column 'source_nd2_path' is missing "
            f"(present columns: {list(df.columns)})."
        )

    for raw_path in df["source_nd2_path"].dropna().unique():
        path = Path(str(raw_path))
        if not path.exists():
            raise ValueError(
                f"{scope_label}: source_nd2_path {str(path)!r} does not exist. The acquisition "
                "inventory points at a raw ND2 that has moved or been deleted since extraction — "
                "re-run scope ingest for this experiment, or restore the ND2 at that path."
            )
        try:
            nd2.ND2File(path).close()
        except Exception as exc:  # noqa: BLE001 — any open failure is a contract violation here
            raise ValueError(
                f"{scope_label}: source_nd2_path {str(path)!r} exists but failed to open as an ND2 "
                f"({type(exc).__name__}: {exc}). The raw source is unreadable/corrupt — restore a "
                "good ND2 at that path, or re-run scope ingest for this experiment."
            ) from exc


def validate_yx1_acquisition_inventory(df: pd.DataFrame, *, check_sources: bool = False) -> None:
    """Fail loud unless the YX1 inventory is schema-complete, calibrated, and a clean tensor.

    YX1 declares WHAT to check (its schema + cell key); the shared primitives do HOW. The same
    primitives back Keyence with its own (colliding) key — here they are a defensive assertion that
    YX1 is clean by construction.

    One contract, two modes by lifecycle moment. At BUILD time (default ``check_sources=False``) this
    validates declared facts only — the ND2 was just opened to build the inventory, so re-opening it
    would be tautological. At the CONSUME boundary (``check_sources=True``, called by the YX1
    materialize backend just before it reads the ND2) it ADDITIONALLY asserts each ``source_nd2_path``
    still exists/opens — the moment the "did the raw source survive?" risk actually appears.
    """
    # Hard-check the shared core first (every scope must satisfy it), then the full YX1 schema.
    assert_columns_present(df, REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS, scope_label=_SCOPE_LABEL)
    assert_columns_present(df, YX1_ACQUISITION_INVENTORY_COLUMNS, scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "micrometers_per_pixel", scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "image_width_px", scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "image_height_px", scope_label=_SCOPE_LABEL)
    assert_channel_mapping_consistent(df, normalized_column="channel_id", scope_label=_SCOPE_LABEL)
    assert_unique_on_key(df, YX1_ACQUISITION_CELL_KEY, scope_label=_SCOPE_LABEL)
    assert_elapsed_time_valid(df, scope_label=_SCOPE_LABEL)
    if check_sources:
        assert_acquisition_sources_readable(df, scope_label=_SCOPE_LABEL)


def assert_elapsed_time_valid(df: pd.DataFrame, *, scope_label: str) -> None:
    """Fail loud unless ``elapsed_time_s`` is finite and non-negative on every row.

    ``elapsed_time_s`` is rebased per position to that position's first frame, so the minimum is 0
    (not > 0) — a NaN or negative value means the raw ``acquisition_time_s`` was missing/unsorted.
    """
    elapsed = pd.to_numeric(df["elapsed_time_s"], errors="coerce")
    bad = elapsed.isna() | (elapsed < 0)
    if bad.any():
        raise ValueError(
            f"{scope_label}: 'elapsed_time_s' must be finite and non-negative; "
            f"{int(bad.sum())} row(s) violate this (sample: {elapsed[bad].head(5).tolist()})."
        )


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Builder — assemble the inventory from the facts the ONE raw read gathered (pure; no ND2 access)
# ─────────────────────────────────────────────────────────────────────────────────────────────


def build_yx1_acquisition_inventory_rows(
    *,
    experiment_id: str,
    n_t: int,
    n_z: int,
    timestamps: Sequence[float],
    channels: Sequence[tuple[int, str, str]],
    stage_xy: Mapping[int, tuple[float, float]],
    micrometers_per_pixel: float,
    image_width_px: int,
    image_height_px: int,
    objective_magnification: str,
    source_nd2_path: Path | str,
) -> list[dict]:
    """Build the un-collapsed inventory rows — one per ``(position, z, channel, time)`` coordinate.

    Pure function over the facts the ONE raw read already gathered (no ND2 access here). The caller
    (``extract_yx1_scope_metadata``) passes in-memory values; no CSV roundtrip at the raw read.

    Args:
        channels: one ``(channel_index, normalized_channel, raw_channel_name)`` triple per ND2
            channel — the full channel mapping, recorded once.
        stage_xy: ``{position_index: (x_um, y_um)}`` from the ND2 frame metadata at T=0.
        timestamps: per-T acquisition time in seconds (length ``n_t``).
    """
    source_nd2_path = str(source_nd2_path)
    rows: list[dict] = []

    for position_index in sorted(stage_xy):
        raw_position_label = str(position_index)
        x_um, y_um = stage_xy.get(position_index, (float("nan"), float("nan")))

        for time_index in range(n_t):
            acquisition_time_s = float(timestamps[time_index])

            for z_index in range(n_z):
                for channel_index, channel, raw_channel_name in channels:
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "raw_position_label": raw_position_label,
                            "position_index": int(position_index),
                            "z_index": int(z_index),
                            "channel_index": int(channel_index),
                            "channel_id": channel,
                            "raw_channel_name": raw_channel_name,
                            "time_index": int(time_index),
                            "acquisition_time_s": acquisition_time_s,
                            "x_um": x_um,
                            "y_um": y_um,
                            "micrometers_per_pixel": micrometers_per_pixel,
                            "image_width_px": int(image_width_px),
                            "image_height_px": int(image_height_px),
                            "objective_magnification": objective_magnification,
                            "microscope_id": "YX1",
                            "n_z": int(n_z),
                            "source_nd2_path": source_nd2_path,
                        }
                    )

    return rows


def _derive_elapsed_time_s(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``elapsed_time_s`` (seconds since each position's first frame) via the shared helper.

    Reuses ``time_helpers.add_elapsed_time_columns`` (the one scope-neutral time derivation) pointed at
    the YX1 raw atom ``acquisition_time_s`` and grouped per ``position_index`` (YX1 position ≡ well,
    1:1). The helper sorts on an internal ``time_int`` and also emits min/hr columns; we alias
    ``time_index`` → ``time_int`` for it and keep only the canonical ``elapsed_time_s``.
    """
    work = df.copy()
    work["time_int"] = work["time_index"]
    # The helper sorts rows internally but preserves the original index labels, so reindexing back
    # onto df.index restores row order while carrying each row's derived value.
    work = add_elapsed_time_columns(
        work,
        group_cols=["position_index"],
        experiment_time_col="acquisition_time_s",
    )
    out = df.copy()
    out["elapsed_time_s"] = work["elapsed_time_s"].reindex(df.index)
    return out


def build_yx1_acquisition_inventory(**kwargs) -> pd.DataFrame:
    """Build + validate the YX1 acquisition inventory DataFrame (the one entry point the stage calls)."""
    rows = build_yx1_acquisition_inventory_rows(**kwargs)
    df = pd.DataFrame(rows)
    df = _derive_elapsed_time_s(df)
    df = df.reindex(columns=list(YX1_ACQUISITION_INVENTORY_COLUMNS))
    validate_yx1_acquisition_inventory(df)
    return df
