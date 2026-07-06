"""Keyence acquisition inventory — the maximal per-coordinate record of the raw BZ-X TIFF planes.

Keyence is NOT a tensor file. Each acquired plane is a separate TIFF named
``...XY##_NNNNN_Z###_CH#.tif``; a frame's address — ``(well, tile, z, channel, time)`` — lives in
the path + filename. This module is the Keyence twin of ``scope/yx1/acquisition_inventory.py``: it
owns the Keyence inventory MODEL as pure functions over rows, and the Keyence validator that
DECLARES the schema + cell key and calls the shared check primitives
(``scope/shared/acquisition_checks.py``) for the mechanics.

Key decisions (see ``target/specs/front_end/keyence_wire_through.md`` Stage A + §8.3):
  - **Record everything; collapse nothing.** One row per raw plane
    ``(well, tile, z_index, channel_index, time_index)`` — Z is EXPLODED (raw Keyence data IS
    per-Z-plane; the legacy materializer focus-stacks on the fly). No ``drop_duplicates`` ever.
  - **One ``source_tiff_path`` PER ROW** (vs YX1's one ``source_nd2_path`` per well) — Keyence is
    many files per frame.
  - **Channel is MAPPED, never defaulted.** Keyence BZ-X embeds proprietary XML readable mainly
    inside Keyence's own software, so the scraped channel NAME is unreliable; the filename ``CH#``
    index is the reliable anchor. ``channel_id`` is resolved by mapping ``channel_index`` through
    ``KEYENCE_CHANNEL_INDEX_MAP`` and FAILS LOUD on an unmapped index (no silent BF default — that
    would mislabel a real fluorescence plane). ``channel_index`` stays the faithful on-disk ``CH#``.
  - **Collisions CAN happen** (re-acquisition) — unlike YX1. Stage A asserts cell-key uniqueness
    FAIL-LOUD; resolution/quarantine is Stage E. A clean experiment passes; a re-acquired well fails
    here until Stage E exists. The cell key separates tiles (``position_index`` is tile-unique) and
    channels (``channel_index``), so a normal multi-tile/multi-Z/multi-channel well is NOT a collision.

Import direction: this module MAY import the shared check primitives, the raw-plane parser, the
channel mapper, and identifiers; it MUST NOT import stages, Snakemake/tasks, stitch, or YX1 logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.scope.acquisition_inventory_contract import (
    REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS,
)
from data_pipeline.acquisition.metadata_ingest.scope.keyence.channel_map import KEYENCE_CHANNEL_INDEX_MAP
from data_pipeline.acquisition.metadata_ingest.scope.keyence.raw_plane_parsing import (
    _extract_keyence_well_and_tile,
    _parse_keyence_time_z_channel,
)
from data_pipeline.acquisition.metadata_ingest.scope.shared.acquisition_checks import (
    assert_channel_mapping_consistent,
    assert_columns_present,
    assert_positive_column,
    assert_unique_on_key,
)
from data_pipeline.acquisition.metadata_ingest.time_helpers import add_elapsed_time_columns
from data_pipeline.shared.channel_vocabulary import validate_channel_id
from data_pipeline.shared.identifiers import build_well_id

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Contract — schema + identity (the columns + the raw-plane cell key this scope produces)
# ─────────────────────────────────────────────────────────────────────────────────────────────

# Keyence-specific Tier-2 columns (the per-plane address + per-tile facts unique to Keyence). These
# are ALLOWED + carried but are NOT the shared core; the shared core lives in
# ``acquisition_inventory_contract.py`` and is hard-checked for every scope. ``acquisition_time_s`` is
# the Keyence raw time atom kept for audit/re-derivation; the converged ``elapsed_time_s`` is core.
KEYENCE_ACQUISITION_INVENTORY_SCOPE_COLUMNS: tuple[str, ...] = (
    "well_index",                  # raw well label parsed from the path (PRE-well_id join: e.g. "B04")
    "well_id",                     # composed global id (build_well_id) — Keyence resolves well at ingest
    "tile_id",                     # raster tile index within the well (drives TileSpec.tile_id at stitch)
    "position_index_within_well",  # tile site within the well (== tile_id) — local position
    "n_tiles_in_well",             # modal tile count for this well (Stage E compares against this)
    "z_index",                     # raw Z plane (exploded, never collapsed)
    "channel_index",               # the on-disk CH# integer (the reliable channel anchor)
    "time_index_claimed",          # the parsed timepoint as the filename CLAIMS it (Stage E may dispute)
    "acquisition_time_s",          # raw per-frame timestamp (the atom elapsed_time_s is derived from)
    "objective_magnification",     # scraped lens (provenance)
    "orientation",                 # tile raster orientation (feeds stitch config; "unknown" if unknown)
    "source_tiff_path",            # the ONE raw TIFF for THIS plane (per-row, not per-well)
)

# The maximal per-coordinate schema = the SHARED Tier-1 core + the Keyence Tier-2 extras.
KEYENCE_ACQUISITION_INVENTORY_COLUMNS: tuple[str, ...] = (
    *REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS,
    *KEYENCE_ACQUISITION_INVENTORY_SCOPE_COLUMNS,
)

# The raw acquisition-cell key (per ``run_well_schema.md``). Exactly one raw plane may occupy each
# cell. ``position_index`` is tile-unique (see the builder), so this key genuinely separates tiles;
# ``channel_index`` separates channels; ``z_index`` separates Z. A duplicate cell = a re-acquisition.
KEYENCE_ACQUISITION_CELL_KEY: tuple[str, ...] = (
    "well_id",
    "position_index",
    "z_index",
    "channel_index",
    "time_index_claimed",
)

_SCOPE_LABEL = "Keyence acquisition inventory"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Validation — fail-loud guards over the contract (schema/calibration/cell-key + source readability)
# ─────────────────────────────────────────────────────────────────────────────────────────────


def assert_keyence_acquisition_sources_readable(df: pd.DataFrame, *, scope_label: str) -> None:
    """Fail loud unless every ``source_tiff_path`` in the inventory still exists AND opens.

    The disk-touching half of the Keyence acquisition contract — the twin of YX1's ND2 readability
    check. ``source_tiff_path`` is a Keyence acquisition-inventory field, so the readability check
    lives here (with the contract), not in the materializer that consumes it. It is intentionally NOT
    a ``scope/shared/acquisition_checks.py`` primitive — those are pure DataFrame functions with no
    disk knowledge.

    Each UNIQUE path is checked once (cost is O(#TIFFs), not O(#rows)). The check is "exists +
    ``skimage.io.imread`` opens"; a failure means the inventory CSV is fine but its raw source moved /
    was deleted / is corrupt since extraction — surfaced HERE as a named contract error.
    """
    import skimage.io as skio  # local import: the check is disk-time, not import-time

    if "source_tiff_path" not in df.columns:
        raise ValueError(
            f"{scope_label}: cannot check source readability — column 'source_tiff_path' is missing "
            f"(present columns: {list(df.columns)})."
        )

    for raw_path in df["source_tiff_path"].dropna().unique():
        path = Path(str(raw_path))
        if not path.exists():
            raise ValueError(
                f"{scope_label}: source_tiff_path {str(path)!r} does not exist. The acquisition "
                "inventory points at a raw TIFF that has moved or been deleted since extraction — "
                "re-run scope ingest for this experiment, or restore the TIFF at that path."
            )
        try:
            skio.imread(path)
        except Exception as exc:  # noqa: BLE001 — any read failure is a contract violation here
            raise ValueError(
                f"{scope_label}: source_tiff_path {str(path)!r} exists but failed to open as a TIFF "
                f"({type(exc).__name__}: {exc}). The raw source is unreadable/corrupt — restore a "
                "good TIFF at that path, or re-run scope ingest for this experiment."
            ) from exc


def assert_channel_id_in_vocabulary(df: pd.DataFrame, *, scope_label: str) -> None:
    """Fail loud unless every ``channel_id`` is in the controlled vocabulary ``VALID_CHANNEL_NAMES``.

    The contract-time net at the system of record (delegates the membership rule to the vocabulary
    owner ``validate_channel_id``). The builder already mints canonical ``channel_id`` via the mapper;
    this catches a hand-edited / stale table.
    """
    for channel_id in df["channel_id"].astype(str).unique():
        validate_channel_id(channel_id)


def assert_elapsed_time_valid(df: pd.DataFrame, *, scope_label: str) -> None:
    """Fail loud unless ``elapsed_time_s`` is finite and non-negative on every row.

    ``elapsed_time_s`` is rebased per well to that well's first frame, so the minimum is 0 (not > 0);
    a NaN or negative value means the raw ``acquisition_time_s`` was missing/unsorted.
    """
    elapsed = pd.to_numeric(df["elapsed_time_s"], errors="coerce")
    bad = elapsed.isna() | (elapsed < 0)
    if bad.any():
        raise ValueError(
            f"{scope_label}: 'elapsed_time_s' must be finite and non-negative; "
            f"{int(bad.sum())} row(s) violate this (sample: {elapsed[bad].head(5).tolist()})."
        )


def validate_keyence_acquisition_inventory(df: pd.DataFrame, *, check_sources: bool = False) -> None:
    """Fail loud unless the Keyence inventory is schema-complete, calibrated, and collision-free.

    Keyence declares WHAT to check (its schema + raw-plane cell key); the shared primitives do HOW.
    Unlike YX1, Keyence CAN collide — a duplicate cell key here is a real re-acquisition, surfaced
    fail-loud (resolution/quarantine is Stage E; until then a clean experiment passes and a re-acquired
    well fails here, which is the intended bridge behaviour).

    One contract, two modes by lifecycle moment (mirrors YX1's ``check_sources``). At BUILD time
    (default ``check_sources=False``) this validates declared facts only. At the CONSUME boundary
    (``check_sources=True``, called by the Keyence materialize backend before reading tiles) it
    ADDITIONALLY asserts each ``source_tiff_path`` still exists/opens.
    """
    assert_columns_present(df, REQUIRED_ACQUISITION_INVENTORY_CORE_COLUMNS, scope_label=_SCOPE_LABEL)
    assert_columns_present(df, KEYENCE_ACQUISITION_INVENTORY_COLUMNS, scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "micrometers_per_pixel", scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "image_width_px", scope_label=_SCOPE_LABEL)
    assert_positive_column(df, "image_height_px", scope_label=_SCOPE_LABEL)
    assert_channel_mapping_consistent(
        df,
        index_column="channel_index",
        raw_column="raw_channel_name",
        normalized_column="channel_id",
        scope_label=_SCOPE_LABEL,
    )
    assert_channel_id_in_vocabulary(df, scope_label=_SCOPE_LABEL)
    # Fail loud on duplicate cell key (re-acquisition). Stage E will add the eligibility/quarantine
    # resolver (resolve_keyence_acquisitions.py) to handle real re-acquired wells gracefully.
    # Until then: clean experiments pass, re-acquired wells fail here — the intended bridge behaviour.
    assert_unique_on_key(df, KEYENCE_ACQUISITION_CELL_KEY, scope_label=_SCOPE_LABEL)
    assert_elapsed_time_valid(df, scope_label=_SCOPE_LABEL)
    if check_sources:
        assert_keyence_acquisition_sources_readable(df, scope_label=_SCOPE_LABEL)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Builder — discover raw planes, scrape per-plane XML, explode one row per plane (fail-loud channel)
# ─────────────────────────────────────────────────────────────────────────────────────────────


def _channel_id_for_index(channel_index: int) -> str:
    """Resolve canonical ``channel_id`` from the reliable ``CH#`` index — fail loud if unmapped.

    Keyence channel NAME metadata is proprietary/unreliable, so the on-disk ``CH#`` index is the
    anchor. An unmapped index raises (naming the index) via the shared applier — the fix is to add the
    real channel to ``KEYENCE_CHANNEL_INDEX_MAP``, never to default it to BF.
    """
    return KEYENCE_CHANNEL_INDEX_MAP.to_canonical(channel_index)


def build_keyence_acquisition_inventory_rows(
    *,
    experiment_id: str,
    raw_data_dir: Path,
    scrape_plane_metadata,
) -> list[dict]:
    """Discover every raw ``*CH*.tif`` plane and build one inventory row per plane (no collapse).

    Args:
        scrape_plane_metadata: callable ``(tiff_path) -> dict`` returning the scraped per-plane XML
            facts (``micrometers_per_pixel``, ``image_width_px``, ``image_height_px``,
            ``objective_magnification``, ``acquisition_time_s``, and optional ``raw_channel_name``).
            Injected so tests can stub the disk-touching XML scrape. ``extract_scope_metadata`` passes
            the real Keyence scraper.

    ``position_index`` is assigned as a GLOBAL enumeration over the sorted ``(well_index, tile_id)``
    pairs present, so it is unique per tile per well — the cell key (which already carries ``well_id``)
    then genuinely separates tiles. ``position_index_within_well`` carries the local ``tile_id``.
    """
    experiment_id = str(experiment_id).strip()
    raw_data_dir = Path(raw_data_dir)

    # Pass 1: parse the filename grammar of every plane (skip non-Keyence-plane files).
    parsed_planes: list[dict] = []
    for tiff_path in sorted(raw_data_dir.rglob("*CH*.tif")):
        well_index, tile_id = _extract_keyence_well_and_tile(tiff_path)
        if well_index is None:
            continue
        tzc = _parse_keyence_time_z_channel(tiff_path)
        if tzc is None:
            continue
        time_index_claimed, z_index, channel_index = tzc
        parsed_planes.append(
            {
                "well_index": well_index,
                "tile_id": int(tile_id),
                "z_index": int(z_index),
                "channel_index": int(channel_index),
                "time_index_claimed": int(time_index_claimed),
                "source_tiff_path": str(tiff_path),
            }
        )

    if not parsed_planes:
        raise ValueError(
            f"{_SCOPE_LABEL}: no parseable Keyence '*CH*.tif' planes under {raw_data_dir}. "
            "Expected files named like '...XY##_NNNNN_Z###_CH#.tif'."
        )

    # Global tile-unique position_index over (well_index, tile_id), and per-well tile counts.
    well_tile_pairs = sorted({(p["well_index"], p["tile_id"]) for p in parsed_planes})
    position_index_by_pair = {pair: idx for idx, pair in enumerate(well_tile_pairs)}
    n_tiles_in_well = (
        pd.DataFrame(well_tile_pairs, columns=["well_index", "tile_id"])
        .groupby("well_index")["tile_id"]
        .nunique()
        .to_dict()
    )

    rows: list[dict] = []
    for plane in parsed_planes:
        well_index = plane["well_index"]
        tile_id = plane["tile_id"]
        channel_index = plane["channel_index"]
        well_id = build_well_id(experiment_id, well_index)
        channel_id = _channel_id_for_index(channel_index)

        meta = scrape_plane_metadata(Path(plane["source_tiff_path"]))
        raw_channel_name = meta.get("raw_channel_name")
        if raw_channel_name in (None, ""):
            # Proprietary metadata had no channel name — record the reliable CH# token as provenance.
            raw_channel_name = f"CH{channel_index}"

        rows.append(
            {
                # Tier-1 shared core
                "experiment_id": experiment_id,
                "position_index": int(position_index_by_pair[(well_index, tile_id)]),
                "channel_id": channel_id,
                "raw_channel_name": str(raw_channel_name),
                "time_index": int(plane["time_index_claimed"]),
                "micrometers_per_pixel": float(meta["micrometers_per_pixel"]),
                "image_width_px": int(meta["image_width_px"]),
                "image_height_px": int(meta["image_height_px"]),
                "microscope_id": "Keyence",
                # Keyence Tier-2
                "well_index": well_index,
                "well_id": well_id,
                "tile_id": int(tile_id),
                "position_index_within_well": int(tile_id),
                "n_tiles_in_well": int(n_tiles_in_well[well_index]),
                "z_index": int(plane["z_index"]),
                "channel_index": int(channel_index),
                "time_index_claimed": int(plane["time_index_claimed"]),
                "acquisition_time_s": float(meta["acquisition_time_s"]),
                "objective_magnification": meta.get("objective_magnification", "unknown"),
                "orientation": meta.get("orientation", "unknown"),
                "source_tiff_path": plane["source_tiff_path"],
            }
        )

    return rows


def _derive_elapsed_time_s(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``elapsed_time_s`` (seconds since each well's first frame) via the shared helper.

    Reuses ``time_helpers.add_elapsed_time_columns`` pointed at the Keyence raw atom
    ``acquisition_time_s`` and grouped per ``well_id``. The helper sorts on an internal ``time_int``
    and emits min/hr columns; we alias ``time_index_claimed`` → ``time_int`` and keep only the
    canonical ``elapsed_time_s``.
    """
    work = df.copy()
    work["time_int"] = work["time_index_claimed"]
    work = add_elapsed_time_columns(
        work,
        group_cols=["well_id"],
        experiment_time_col="acquisition_time_s",
    )
    out = df.copy()
    out["elapsed_time_s"] = work["elapsed_time_s"].reindex(df.index)
    return out


def build_keyence_acquisition_inventory(
    *,
    experiment_id: str,
    raw_data_dir: Path,
    scrape_plane_metadata,
) -> pd.DataFrame:
    """Build + validate the Keyence acquisition inventory DataFrame (the one entry point the stage calls)."""
    rows = build_keyence_acquisition_inventory_rows(
        experiment_id=experiment_id,
        raw_data_dir=raw_data_dir,
        scrape_plane_metadata=scrape_plane_metadata,
    )
    df = pd.DataFrame(rows)
    df = _derive_elapsed_time_s(df)
    df = df.reindex(columns=list(KEYENCE_ACQUISITION_INVENTORY_COLUMNS))
    validate_keyence_acquisition_inventory(df)
    log.info(
        "Built Keyence acquisition inventory: %d planes, %d wells, tiles/well=%s",
        len(df),
        df["well_id"].nunique(),
        sorted(df["n_tiles_in_well"].unique().tolist()),
    )
    return df
