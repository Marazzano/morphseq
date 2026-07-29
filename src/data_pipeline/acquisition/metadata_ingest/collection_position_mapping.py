"""Collection POSITION MAPPING — map each source file independently, concat into ONE artifact.

This module owns **Step 2** of the collection worklist (see ``docs/COLLECTION_STEP_BY_STEP.md``).

**Why per-file, not map-once.** For Keyence the well is a folder marker (``XY01/_A01``) — fixed on
disk, the same in every source. But for **YX1 the well is derived from stage ``x_um``/``y_um`` matched
to a reference grid, and each source is its OWN ND2 with its OWN stage frame** (the plate is
re-seated across the gap). So source-t28's ``position_index=1`` and source-t52's ``position_index=1``
are DIFFERENT physical acquisitions. You cannot map once and reuse — **each source must be mapped
independently**, against its own geometry.

**The existing per-scope map functions are reused UNCHANGED.** A single source file is
single-experiment-shaped (one raw dir / one scope-metadata block), which is exactly what those
functions already handle. The collection logic here is only: the per-source loop + the concat,
driven by the collection artifact. No mapping rules are re-implemented.

**Output shape — ONE artifact, per-source blocks concatenated.** Separate per-source mapping files
would make the DAG variable-arity (checkpoints / dynamic fan-in — the complexity the pipeline
avoids). Instead ONE ``position_well_mapping.csv`` per experiment with N sources as ROWS, which is
also how the acquisition inventory already holds N sources. The DAG artifact count is unchanged and
downstream reads one file::

    position_well_mapping.csv  (ONE artifact per experiment)
      experiment_id         position_index  well_index  well_id                   time_index  source_file
      chem28c_coll_plate01  1               A01         chem28c_coll_plate01_A01  0           20250622_plate01_t28hpf
      chem28c_coll_plate01  1               A01         chem28c_coll_plate01_A01  1           20250623_plate01_t52hpf

``well_id`` is PLATE-keyed (``chem28c_coll_plate01_A01``) so A01@t28 and A01@t52 are the same well —
that is the whole point of merging the plate. ``time_index`` (REQUIRED, the key) tags which source
block a row belongs to and EQUALS the artifact's ``sources[].time_index``; ``source_file`` is
human-readable provenance.

Import direction: MAY import the per-scope map functions (it orchestrates them) + shared identifiers;
MUST NOT import Snakemake/tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.position_well_mapping import (
    validate_position_well_mapping,
)
from data_pipeline.shared.identifiers import build_well_id

# The two columns the collection concat ADDS to the per-source mapping. Both are always present
# (a single experiment gets one block: time_index=0 + its own raw as source_file).
COLLECTION_MAPPING_COLUMNS: tuple[str, ...] = ("time_index", "source_file")


def _rekey_well_id_to_plate(mapping: pd.DataFrame, *, experiment_id: str) -> pd.DataFrame:
    """Re-stamp ``experiment_id`` and REBUILD ``well_id`` off the PLATE id.

    The per-scope map functions compose ``well_id = build_well_id(experiment_id, well_index)``. We
    hand them the SOURCE name as the id (Keyence resolves its raw dir from it), so their output is
    source-keyed. Rebuilding off the plate id — using the source-independent ``well_index`` (A01) —
    is what makes A01@t28 and A01@t52 ONE well. The rebuild reuses ``build_well_id`` (the single
    generator), never string surgery.
    """
    out = mapping.copy()
    out["experiment_id"] = experiment_id
    out["well_id"] = out["well_index"].astype(str).map(
        lambda well_index: build_well_id(experiment_id, well_index)
    )
    return out


def map_collection_positions_to_wells(
    *,
    experiment_id: str,
    sources: Sequence[dict],
    scope_metadata_csv: Path,
    output_mapping_csv: Path,
    output_provenance_json: Path,
    microscope: str,
    raw_root: Path,
    ref_xy_csv: Path | None = None,
    scratch_dir: Path | None = None,
) -> pd.DataFrame:
    """Map EACH source independently, then concat into one plate-keyed mapping artifact.

    Args:
        experiment_id: the merged ``{collection}_{plate_token}`` id (the PLATE) — what ``well_id``
            is keyed to.
        sources: the collection artifact's ``sources`` records (``file`` / ``raw_path`` /
            ``time_index``). Read from the artifact; never re-globbed, never re-derived.
        scope_metadata_csv: the UNIONED scope metadata (Step 1). Split back into per-source blocks
            by ``time_index`` — each source is mapped against its OWN geometry.
        output_mapping_csv: destination for the ONE concatenated mapping artifact.
        output_provenance_json: destination for the mapping provenance summary.
        microscope: "Keyence" | "YX1" — selects the per-scope map function.
        raw_root: raw image root containing the ``_coll`` dir (Keyence resolves its well dirs here).
        ref_xy_csv: the YX1 reference plate grid (required for YX1, unused for Keyence).
        scratch_dir: where the per-source map functions write their own (intermediate) outputs.
            Defaults beside ``output_mapping_csv``. These per-source files are NOT DAG artifacts —
            the DAG artifact is the single concatenated CSV.

    Returns:
        The concatenated, validated mapping.
    """
    if not sources:
        raise ValueError(
            f"map_collection_positions_to_wells: no sources for {experiment_id!r}. Read them from "
            "the collection-classify artifact's 'sources'."
        )

    scope_union = pd.read_csv(scope_metadata_csv)
    if "source_ordinal" not in scope_union.columns:
        raise ValueError(
            f"map_collection_positions_to_wells: unioned scope metadata {scope_metadata_csv} has no "
            "'source_ordinal' column — Step 1 must stamp it (it is the source key that lets Step 2 "
            "re-split the union into per-source blocks). NOTE: splitting on time_index would be "
            "wrong — a timelapse source spans MANY time_index values but is ONE source to map."
        )

    scratch = Path(scratch_dir) if scratch_dir else Path(output_mapping_csv).parent / "_per_source"
    scratch.mkdir(parents=True, exist_ok=True)

    # Re-split the union by SOURCE, not by frame. Each block IS that source's own scope metadata
    # (its own x/y, calibration, timing). Nothing is reconstructed — Step 1 concatenated, it did not
    # collapse. Splitting on source_ordinal (rather than time_index) is what makes a timelapse
    # source one mapping unit instead of one-per-frame.
    blocks_by_source_ordinal = {
        int(source_ordinal): block
        for source_ordinal, block in scope_union.groupby("source_ordinal")
    }

    mapped_blocks: list[pd.DataFrame] = []
    for record in sources:
        source_id = str(record["file"])
        source_ordinal = int(record["source_ordinal"])

        block = blocks_by_source_ordinal.get(source_ordinal)
        if block is None or block.empty:
            raise ValueError(
                f"map_collection_positions_to_wells: unioned scope metadata has no rows for "
                f"source_ordinal={source_ordinal} (source {source_id!r}). Step 1 must emit one "
                f"block per source; found blocks for {sorted(blocks_by_source_ordinal)}."
            )

        # Hand the per-scope map function this source's OWN scope metadata block.
        block_csv = scratch / f"scope_metadata__{source_id}.csv"
        block.to_csv(block_csv, index=False)
        source_mapping_csv = scratch / f"position_well_mapping__{source_id}.csv"
        source_provenance_json = scratch / f"mapping_provenance__{source_id}.json"

        per_source = _map_one_source(
            microscope=microscope,
            source_id=source_id,
            scope_block_csv=block_csv,
            output_mapping_csv=source_mapping_csv,
            output_provenance_json=source_provenance_json,
            raw_root=Path(raw_root),
            ref_xy_csv=ref_xy_csv,
        )

        # Re-key to the PLATE, then stamp the source identity onto the block. `source_ordinal` is
        # the JOIN KEY (which source); `source_file` is readable provenance. Deliberately NO
        # time_index: the mapping is per SOURCE, and every frame of that source — however many
        # merged time_index values it spans — inherits the same position→well mapping.
        per_source = _rekey_well_id_to_plate(per_source, experiment_id=experiment_id)
        per_source["source_ordinal"] = source_ordinal
        per_source["source_file"] = source_id
        mapped_blocks.append(per_source)

    mapping = pd.concat(mapped_blocks, ignore_index=True, sort=False)
    validate_position_well_mapping(
        mapping, scope_label=f"collection position_well_mapping ({experiment_id})"
    )

    out_mapping = Path(output_mapping_csv)
    out_mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out_mapping, index=False)

    out_prov = Path(output_provenance_json)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_prov.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "microscope": microscope,
                # Each source was mapped independently by the per-scope map function, then re-keyed
                # to the plate id and concatenated — recorded so downstream can see the provenance.
                "mapping_method": "collection_per_source_map",
                "n_sources": len(sources),
                "n_rows": int(len(mapping)),
                "n_wells": int(mapping["well_index"].nunique()),
                "sources": [
                    {"source_file": str(r["file"]), "time_index": int(r["time_index"])}
                    for r in sources
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return mapping


def _map_one_source(
    *,
    microscope: str,
    source_id: str,
    scope_block_csv: Path,
    output_mapping_csv: Path,
    output_provenance_json: Path,
    raw_root: Path,
    ref_xy_csv: Path | None,
) -> pd.DataFrame:
    """Run the EXISTING per-scope map function on ONE source, unchanged.

    ``experiment_id`` passed to the map function is the SOURCE name, not the plate: Keyence resolves
    its well dirs as ``raw_data_dir / experiment_id``, so it must name the source's own directory.
    The plate re-key happens in the caller (``_rekey_well_id_to_plate``) — that keeps these map
    functions untouched, which is the point of Step 2.
    """
    if microscope == "Keyence":
        from data_pipeline.acquisition.metadata_ingest.scope.keyence.map_keyence_positions_to_wells import (
            map_positions_to_wells_keyence,
        )

        # raw_root here is the COLLECTION dir; `source_id` is the child dir inside it.
        return map_positions_to_wells_keyence(
            raw_data_dir=raw_root,
            scope_metadata_csv=scope_block_csv,
            output_mapping_csv=output_mapping_csv,
            output_provenance_json=output_provenance_json,
            experiment_id=source_id,
        )

    if microscope == "YX1":
        if not ref_xy_csv:
            raise ValueError(
                "map_collection_positions_to_wells: --ref-xy-csv is required for YX1 mapping "
                "(set scope_metadata.yx1.ref_xy_csv in config.yaml). Each YX1 source is matched to "
                "the reference grid against its OWN stage frame."
            )
        from data_pipeline.acquisition.metadata_ingest.scope.yx1.map_yx1_positions_to_wells import (
            map_positions_to_wells_yx1,
        )

        return map_positions_to_wells_yx1(
            scope_metadata_csv=scope_block_csv,
            output_mapping_csv=output_mapping_csv,
            output_provenance_json=output_provenance_json,
            experiment_id=source_id,
            ref_xy_csv=ref_xy_csv,
        )

    raise ValueError(
        f"map_collection_positions_to_wells: unsupported microscope {microscope!r}; expected "
        "'Keyence' or 'YX1'."
    )
