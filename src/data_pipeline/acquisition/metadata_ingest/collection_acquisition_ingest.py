"""Collection acquisition ingest — wire the union into the acquisition stage.

The DAG runs on ``experiment_id``. For a MERGED collection plate
(``{collection}_{plate_token}``), one experiment is built from N raw source children
(the ``{date}_{plate_token}[_{event_label}]`` folders/files under the ``_coll`` dir).
This module is the acquisition-stage entry point that:

  1. finds the plate's source children (``find_collection_plate_sources`` — the inverse
     of the id composer),
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
from data_pipeline.acquisition.metadata_ingest.experiment_collection import (
    find_collection_plate_sources,
)
from data_pipeline.acquisition.metadata_ingest.position_well_mapping.position_well_mapping_contract import (
    REQUIRED_POSITION_WELL_MAPPING_COLUMNS,
    validate_position_well_mapping,
)

# The union already resolves well_id↔position (each source's acquisition-inventory builder minted
# well_id), so the collection's position mapping is a DERIVATION, not a re-map — recorded as this
# mapping_method so downstream can see it came from the union, not a scope position solve.
_COLLECTION_MAPPING_METHOD = "collection_union"


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


def _read_yx1_source(child_file_or_dir: Path, child_experiment_id: str) -> pd.DataFrame:
    from data_pipeline.acquisition.metadata_ingest.scope.yx1.acquisition_inventory import (
        build_yx1_acquisition_inventory,
    )

    return build_yx1_acquisition_inventory(
        experiment_id=child_experiment_id,
        raw_data_dir=child_file_or_dir,
    )


_SCOPE_READERS = {"Keyence": _read_keyence_source, "YX1": _read_yx1_source}


# ─────────────────────────────────────────────────────────────────────────────────────
# Orchestration — find sources, union, write
# ─────────────────────────────────────────────────────────────────────────────────────

def derive_position_well_mapping(unioned: pd.DataFrame) -> pd.DataFrame:
    """Derive the canonical position→well mapping from the unioned inventory.

    The union already carries ``experiment_id``, ``position_index``, ``well_index`` and the
    ``well_id`` its per-source builder minted, so the collection's mapping is a SELECT+dedup —
    NOT a re-map. ``mapping_method`` records that it came from the union. The result satisfies
    the same ``validate_position_well_mapping`` contract the native materializer consumes, so
    the collection reuses the native path unchanged.
    """
    needed = ["experiment_id", "position_index", "well_index", "well_id"]
    missing = [c for c in needed if c not in unioned.columns]
    if missing:
        raise ValueError(
            f"derive_position_well_mapping: unioned inventory missing {missing}; cannot derive "
            "the position mapping. (Expected the per-source acquisition-inventory builder to mint "
            "well_id + position_index.)"
        )
    mapping = unioned[needed].drop_duplicates().reset_index(drop=True)
    mapping["mapping_method"] = _COLLECTION_MAPPING_METHOD
    mapping = mapping[list(REQUIRED_POSITION_WELL_MAPPING_COLUMNS)]
    validate_position_well_mapping(mapping)
    return mapping


def ingest_collection_acquisition_inventory(
    *,
    experiment_id: str,
    raw_root: Path,
    microscope: str,
    output_csv: Path,
    position_well_mapping_csv: Path | None = None,
) -> pd.DataFrame:
    """Build ONE acquisition inventory for a merged collection plate and write it.

    Args:
        experiment_id: the merged ``{collection}_{plate_token}`` id.
        raw_root: raw image root containing the ``_coll`` dir (scope-anchored by the caller).
        microscope: "Keyence" | "YX1" — selects the per-source reader.
        output_csv: destination for the unioned acquisition inventory CSV.
        position_well_mapping_csv: if given, also derive + write the canonical position→well
            mapping (the second artifact the native materializer needs). The mapping is derived
            from the union (well_id already resolved), NOT re-solved from raw positions.

    Returns the unioned inventory (also written to ``output_csv``).
    """
    if microscope not in _SCOPE_READERS:
        raise ValueError(
            f"ingest_collection_acquisition_inventory: unsupported microscope {microscope!r}; "
            f"expected one of {sorted(_SCOPE_READERS)}."
        )
    read_one = _SCOPE_READERS[microscope]

    collection_name, child_names = find_collection_plate_sources(experiment_id, raw_root)
    collection_dir = raw_root / collection_name

    def read_source(source: SourceChild, _child_experiment_id: str) -> pd.DataFrame:
        # Keyence children are dirs (name); YX1 children are .nd2 files (stem). The child
        # path is the collection child; read it into a per-source acquisition inventory.
        child_path = collection_dir / source.child_name
        if not child_path.exists():
            child_path = collection_dir / f"{source.child_name}.nd2"
        return read_one(child_path, source.child_name)

    sources = [SourceChild(child_name=name, scope=microscope) for name in child_names]
    unioned = union_collection_acquisition_inventories(
        collection_name=collection_name, sources=sources, read_source=read_source
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    unioned.to_csv(output_csv, index=False)

    if position_well_mapping_csv is not None:
        mapping = derive_position_well_mapping(unioned)
        position_well_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(position_well_mapping_csv, index=False)

    return unioned
