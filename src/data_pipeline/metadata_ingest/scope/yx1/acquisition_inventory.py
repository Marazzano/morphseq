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

import pandas as pd

from data_pipeline.metadata_ingest.scope.shared.acquisition_checks import (
    assert_channel_mapping_consistent,
    assert_columns_present,
    assert_positive_column,
    assert_unique_on_key,
)

# The maximal per-coordinate schema (the tensor address + all relevant ND2 facts). Standardized
# ``*_index`` axis vocabulary — the inventory is a NEW artifact, so it is born with target names
# (the legacy scope_metadata keeps time_int/z_position until the Scope-2 collapse).
YX1_ACQUISITION_INVENTORY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "raw_position_label",       # ND2 P-index as string (PRE-mapping — no well_id at ingest)
    "position_index",           # tensor P axis (int)
    "z_index",                  # tensor Z axis (0..n_z-1) — exploded, never collapsed
    "channel_index",            # tensor C axis (numeric index into the ND2 channel list)
    "channel",                  # normalized token (BF/GFP/RFP)
    "raw_channel_name",         # raw ND2 channel string (e.g. "EYES - Dia")
    "time_index",               # tensor T axis (standardized; legacy time_int)
    "acquisition_time_s",       # per-frame ND2 timestamp
    "x_um",                     # stage position (provenance; enables the join)
    "y_um",
    "micrometers_per_pixel",    # calibration — validated > 0
    "image_width_px",
    "image_height_px",
    "objective_magnification",
    "microscope_id",
    "n_z",                      # full Z depth of this acquisition (provenance)
    "source_nd2_path",          # the ONE ND2 (no per-plane path)
)

# The tensor cell key — exactly one raw unit may occupy each cell. YX1 is clean by construction.
YX1_ACQUISITION_CELL_KEY: tuple[str, ...] = (
    "position_index",
    "z_index",
    "channel_index",
    "time_index",
)

_SCOPE_LABEL = "YX1 acquisition inventory"


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
                            "channel": channel,
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


def validate_yx1_acquisition_inventory(df: pd.DataFrame) -> None:
    """Fail loud unless the YX1 inventory is schema-complete, calibrated, and a clean tensor.

    YX1 declares WHAT to check (its schema + cell key); the shared primitives do HOW. The same
    primitives back Keyence with its own (colliding) key — here they are a defensive assertion that
    YX1 is clean by construction.
    """
    assert_columns_present(df, YX1_ACQUISITION_INVENTORY_COLUMNS, scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "micrometers_per_pixel", scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "image_width_px", scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "image_height_px", scope_label=_SCOPE_LABEL)
    assert_channel_mapping_consistent(df, scope_label=_SCOPE_LABEL)
    assert_unique_on_key(df, YX1_ACQUISITION_CELL_KEY, scope_label=_SCOPE_LABEL)


def build_yx1_acquisition_inventory(**kwargs) -> pd.DataFrame:
    """Build + validate the YX1 acquisition inventory DataFrame (the one entry point the stage calls)."""
    rows = build_yx1_acquisition_inventory_rows(**kwargs)
    df = pd.DataFrame(rows, columns=list(YX1_ACQUISITION_INVENTORY_COLUMNS))
    validate_yx1_acquisition_inventory(df)
    return df
