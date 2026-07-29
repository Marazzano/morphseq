"""Collection acquisition ingest — wire the union into the acquisition stage.

The DAG runs on ``experiment_id``. For a MERGED collection plate
(``{collection}_{plate_token}``), one experiment is built from N raw source children
(the ``{date}_{plate_token}[_{event_label}]`` folders/files under the ``_coll`` dir).
This module is the acquisition-stage entry point that:

  1. READS the plate's sources from the collection-provenance artifact (it never globs the
     ``_coll`` dir — discovery does that ONCE, when the artifact is built),
  2. reads EACH source ONCE via the real per-scope acquisition-inventory builder (the
     layer that mints ``channel_id`` / ``elapsed_time_s`` — NOT the raw scope extractor),
  3. UNIONs them into one acquisition inventory keyed by ``{collection}_{plate}``
     (``union_collection_acquisition_inventories`` — Merge-A identity, one-read-per-source).

It owns ONLY the read_source dispatch (per scope) + orchestration; the union and the
grammar are imported, never re-implemented (DRY). See docs/EXPERIMENT_GROUP_PLATE_MODEL.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import (
    SourceChild,
    union_collection_acquisition_inventories,
)
from data_pipeline.acquisition.metadata_ingest.collection_merge_primitives import (
    rebuild_well_id_for_plate,
)
from data_pipeline.shared.identifiers import (
    build_image_id,
    parse_collection_name_from_plate_id,
)

# ─────────────────────────────────────────────────────────────────────────────────────
# Per-scope read_source — read ONE source child into a validated acquisition inventory
# ─────────────────────────────────────────────────────────────────────────────────────

def _read_keyence_source(child_dir: Path, child_experiment_id: str) -> pd.DataFrame:
    from data_pipeline.acquisition.metadata_ingest.scope.keyence.acquisition_inventory import (
        build_keyence_acquisition_inventory,
    )
    from data_pipeline.acquisition.metadata_ingest.scope.keyence.extract_scope_metadata import (
        make_keyence_plane_scraper,
    )

    return build_keyence_acquisition_inventory(
        experiment_id=child_experiment_id,
        raw_data_dir=child_dir,
        scrape_plane_metadata=make_keyence_plane_scraper(),
    )


# ─────────────────────────────────────────────────────────────────────────────────────
# Per-scope scope-metadata readers (Step 1) — the OTHER artifact, same per-source read
# ─────────────────────────────────────────────────────────────────────────────────────
# scope_metadata and acquisition_inventory are NOT interchangeable: scope_metadata carries the
# per-position GEOMETRY (raw_position_label, x_um, y_um) + channel that map_positions/apply ingest.
# Writing the acquisition inventory into the scope_csv slot is what crashed `apply` on `channel`.

def _read_keyence_scope_metadata(
    source_path: Path, source_id: str, output_csv: Path
) -> pd.DataFrame:
    from data_pipeline.acquisition.metadata_ingest.scope.keyence.extract_scope_metadata import (
        extract_keyence_scope_metadata,
    )

    # The Keyence extractor resolves its wells under `raw_data_dir / experiment_id`, so it takes the
    # collection dir + the source child name. The PLATE re-stamp happens in the union.
    return extract_keyence_scope_metadata(
        raw_data_dir=source_path.parent,
        experiment_id=source_id,
        output_csv=output_csv,
    )


def _read_yx1_scope_metadata(
    source_path: Path, source_id: str, output_csv: Path
) -> pd.DataFrame:
    from data_pipeline.acquisition.metadata_ingest.scope.yx1.extract_yx1_scope_metadata import (
        extract_yx1_scope_metadata,
    )

    # Point at THIS source's ND2 FILE, not the collection dir: a _coll dir holds one ND2 per source
    # (the same plate at several ages), so a directory would be ambiguous. `source_path` is the
    # artifact's recorded raw_path. A YX1 source is an .nd2 FILE; tolerate a
    # suffix-less record (older artifacts) but never probe disk to choose between candidates.
    nd2_path = source_path if source_path.suffix.lower() == ".nd2" else source_path.with_suffix(".nd2")
    return extract_yx1_scope_metadata(
        raw_data_dir=nd2_path,
        experiment_id=source_id,
        output_csv=output_csv,
    )


_SCOPE_METADATA_READERS = {
    "Keyence": _read_keyence_scope_metadata,
    "YX1": _read_yx1_scope_metadata,
}


# ─────────────────────────────────────────────────────────────────────────────────────
# Per-scope PLATE RE-KEY — rebind whatever identity a scope minted against the source id
# ─────────────────────────────────────────────────────────────────────────────────────
# Scopes differ in WHAT identity they resolve at ingest, so this is a per-scope concern and does
# NOT belong in the scope-agnostic union:
#
#   Keyence resolves the well at ingest (from the XY##/_A01 folder marker) and mints well_index,
#           well_id and image_id from the experiment_id it is HANDED. We hand it the SOURCE name
#           (its raw dir is `raw_data_dir / experiment_id`), so all three come out source-bound and
#           must be rebound to the plate.
#   YX1     resolves nothing at ingest — no well_index/well_id/image_id at all (well_id is attached
#           later by apply_position_to_well_mapping). Nothing to re-key.

def _rekey_keyence_scope_metadata_to_plate(
    block: pd.DataFrame, experiment_id: str
) -> pd.DataFrame:
    """Rebind Keyence's ingest-minted ids from the source id to the PLATE id.

    The ``well_id`` half is the SHARED ``rebuild_well_id_for_plate`` (the acquisition union calls the
    same helper, so the two cannot diverge on how a source-bound well_id becomes plate-bound). The
    only Keyence-specific part is ``image_id``: scope metadata carries it, the acquisition inventory
    does not, so it exists to rebuild only here.

    ``image_id`` is later OVERWRITTEN by ``apply_position_to_well_mapping``, but leaving a stale
    source-keyed value in the unioned artifact would make it internally inconsistent for anything
    that reads scope metadata before ``apply``.
    """
    if "well_index" not in block.columns:
        raise ValueError(
            "_rekey_keyence_scope_metadata_to_plate: Keyence scope metadata is missing "
            "'well_index'. It is the source-independent raw well label the plate-keyed well_id is "
            "rebuilt from; without it the merged plate cannot share wells across sources."
        )

    out = rebuild_well_id_for_plate(block, experiment_id)

    # image_id = (well_id, channel_id, time_index). Keyence scope metadata names the channel token
    # `channel` (the laggard name for channel_id — the pipeline's canonical token).
    if "image_id" in out.columns:
        channel_col = "channel_id" if "channel_id" in out.columns else "channel"
        missing_for_image_id = [c for c in (channel_col, "time_index") if c not in out.columns]
        if missing_for_image_id:
            raise ValueError(
                "_rekey_keyence_scope_metadata_to_plate: cannot rebuild 'image_id' — missing "
                f"{missing_for_image_id}. image_id is (well_id, channel_id, time_index); leaving "
                "the source-keyed value would make the unioned scope metadata inconsistent."
            )
        out["image_id"] = [
            build_image_id(well_id, str(channel), int(time_index))
            for well_id, channel, time_index in zip(
                out["well_id"], out[channel_col], out["time_index"]
            )
        ]
    return out


# Keyence has real work; YX1 legitimately has none. Same dispatch shape as the reader dicts.
_SCOPE_METADATA_REKEYS = {
    "Keyence": _rekey_keyence_scope_metadata_to_plate,
    "YX1": None,
}


def ingest_collection_scope_metadata(
    *,
    experiment_id: str,
    sources: list[dict],
    microscope: str,
    output_csv: Path,
    scratch_dir: Path | None = None,
) -> pd.DataFrame:
    """Build ONE experiment-level scope_metadata for a collection plate (Step 1) and write it.

    Reads each source's scope metadata ONCE (its own geometry/calibration/timing), then delegates the
    per-source-lossless concat + source-identity stamping to ``union_collection_scope_metadata``.
    ``sources`` comes from the collection-classify artifact — nothing is re-globbed.
    """
    from data_pipeline.acquisition.metadata_ingest.collection_scope_union import (
        union_collection_scope_metadata,
    )

    if microscope not in _SCOPE_METADATA_READERS:
        raise ValueError(
            f"ingest_collection_scope_metadata: unsupported microscope {microscope!r}; expected one "
            f"of {sorted(_SCOPE_METADATA_READERS)}."
        )
    read_scope = _SCOPE_METADATA_READERS[microscope]

    scratch = Path(scratch_dir) if scratch_dir else Path(output_csv).parent / "_per_source"
    scratch.mkdir(parents=True, exist_ok=True)

    def read_source(record: dict, _experiment_id: str) -> pd.DataFrame:
        source_id = str(record["file"])
        return read_scope(
            Path(record["raw_path"]),
            source_id,
            scratch / f"scope_metadata_raw__{source_id}.csv",
        )

    unioned = union_collection_scope_metadata(
        experiment_id=experiment_id,
        sources=sources,
        read_source=read_source,
        # Per-scope: rebind ids this scope minted against the source id (Keyence) — or nothing
        # (YX1, which mints no well identity at ingest).
        rekey_to_plate=_SCOPE_METADATA_REKEYS[microscope],
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    unioned.to_csv(output_csv, index=False)
    return unioned


def _read_yx1_source(child_file_or_dir: Path, child_experiment_id: str) -> pd.DataFrame:
    """Read ONE YX1 source into its acquisition inventory.

    ``build_yx1_acquisition_inventory`` is a PURE row-builder over facts already gathered from the
    ND2 — it does no file I/O — so the inventory is obtained the way the single-experiment path gets
    it: from ``extract_yx1_scope_metadata``, which emits it as a side output of the SAME one read.

    The path handed over is this source's ND2 FILE, not the ``_coll`` directory: that dir holds one
    ND2 per source, so a directory would be ambiguous.
    """
    import tempfile

    from data_pipeline.acquisition.metadata_ingest.scope.yx1.extract_yx1_scope_metadata import (
        extract_yx1_scope_metadata,
    )

    nd2_path = (
        child_file_or_dir
        if child_file_or_dir.suffix.lower() == ".nd2"
        else child_file_or_dir.with_suffix(".nd2")
    )

    # The scope CSV is a required output of the extractor but the acquisition union does not consume
    # it here (the scope union produces the real one from its own read); write both to scratch and
    # return the inventory.
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        inventory_csv = scratch_dir / "acquisition_inventory.csv"
        extract_yx1_scope_metadata(
            raw_data_dir=nd2_path,
            experiment_id=child_experiment_id,
            output_csv=scratch_dir / "scope_metadata.csv",
            acquisition_inventory_csv=inventory_csv,
        )
        return pd.read_csv(inventory_csv)


_SCOPE_READERS = {"Keyence": _read_keyence_source, "YX1": _read_yx1_source}


# ─────────────────────────────────────────────────────────────────────────────────────
# Orchestration — find sources, union, write
# ─────────────────────────────────────────────────────────────────────────────────────

# NOTE (Step 2, docs/COLLECTION_STEP_BY_STEP.md): the position→well mapping is NO LONGER derived
# from the unioned inventory. Deriving it assumed the well↔position relation is shared across
# sources, which is true for Keyence (the well is a folder marker) but FALSE for YX1, where the well
# comes from stage x/y matched to a reference grid and every source has its own stage frame (the
# plate is re-seated across the gap). The mapping now lives in `collection_position_mapping`, which
# runs the existing per-scope map function once per source and concatenates the blocks.


def ingest_collection_acquisition_inventory(
    *,
    experiment_id: str,
    sources: list[dict],
    microscope: str,
    output_csv: Path,
) -> pd.DataFrame:
    """Build ONE acquisition inventory for a merged collection plate and write it.

    Args:
        experiment_id: the merged ``{collection}_{plate_token}`` id.
        sources: the collection-provenance artifact's ``sources`` records (``file`` / ``raw_path`` /
            ``source_ordinal``). The artifact is the SINGLE source of truth for which sources exist
            and where they are — this function never globs the ``_coll`` dir.
        microscope: "Keyence" | "YX1" — selects the per-source reader.
        output_csv: destination for the unioned acquisition inventory CSV.

    Returns the unioned inventory (also written to ``output_csv``).

    KEYSTONE RULE: discovery globs ``_coll`` exactly ONCE (when the provenance artifact is built);
    every disk-touching step downstream READS that artifact. This function used to re-glob and even
    re-probe disk for a ``.nd2`` suffix, which made the source manifest advisory rather than
    authoritative — the two ingest paths could disagree if files were added, removed, or renamed
    between steps. Both the scope-metadata union and this one now consume the same ``sources``.

    The position→well mapping is NOT produced here — it is Step 2's own artifact, built by
    ``collection_position_mapping`` from a per-source map run (see the note above).
    """
    if microscope not in _SCOPE_READERS:
        raise ValueError(
            f"ingest_collection_acquisition_inventory: unsupported microscope {microscope!r}; "
            f"expected one of {sorted(_SCOPE_READERS)}."
        )
    if not sources:
        raise ValueError(
            f"ingest_collection_acquisition_inventory: no sources for {experiment_id!r}. Read them "
            "from the collection-provenance artifact's 'sources'."
        )
    read_one = _SCOPE_READERS[microscope]

    # The collection dir name is a pure parse of the plate id — no filesystem probe.
    collection_name = parse_collection_name_from_plate_id(experiment_id)

    # `raw_path` from the artifact IS the resolved on-disk location (a Keyence dir or a YX1 .nd2),
    # recorded at discovery. No suffix guessing, no existence probing to pick a path.
    raw_path_by_source_id = {str(rec["file"]): Path(rec["raw_path"]) for rec in sources}

    def read_source(source: SourceChild, _child_experiment_id: str) -> pd.DataFrame:
        return read_one(raw_path_by_source_id[source.child_name], source.child_name)

    ordered_source_ids = [
        str(rec["file"]) for rec in sorted(sources, key=lambda r: int(r["source_ordinal"]))
    ]
    source_children = [
        SourceChild(child_name=name, scope=microscope) for name in ordered_source_ids
    ]
    unioned = union_collection_acquisition_inventories(
        collection_name=collection_name, sources=source_children, read_source=read_source
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    unioned.to_csv(output_csv, index=False)
    return unioned
