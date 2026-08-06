"""Real YX1 collection end-to-end: Steps 0 -> 1 -> 2 -> 3.

The collection is 20260624_2x_td_bf_pbx_coll — ONE plate (plate01) acquired at t33/t52/t77 hpf,
3 sources, 18G per ND2, multi-channel (fluorescence + BF). This is the path that REQUIRES the
per-source position mapping (YX1 derives the well from stage x_um/y_um, and each source is its own
ND2 with its own stage frame because the plate is re-seated), and the source-keyed apply join
(whose validate="many_to_one" previously raised MergeError for any multi-source YX1 plate).

Run under SGE via yx1_e2e.sh — the ND2 reads are slow and large, so not inline.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_ingest import (
    ingest_collection_acquisition_inventory,
    ingest_collection_scope_metadata,
)
from data_pipeline.acquisition.metadata_ingest.collection_provenance import (
    build_collection_provenance,
)
from data_pipeline.acquisition.metadata_ingest.collection_position_mapping import (
    map_collection_positions_to_wells,
)
from data_pipeline.acquisition.metadata_ingest.collection_discovery import (
    resolve_experiment_ids,
)
from data_pipeline.acquisition.metadata_ingest.scope.shared.apply_position_to_well_mapping import (
    apply_position_to_well_mapping,
)

RAW = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline/input/raw_image_data/YX1"
)
REPO = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/.coll_wt")
REF_XY = REPO / "metadata/reference/yx1/nd2_ref_plate_xy_coordinates.csv"
OUT = REPO / ".collection_test_scratch/e2e_yx1"

FAILURES: list[str] = []


def check(label, condition, got=""):
    status = "  PASS  " if condition else "  FAIL  "
    print(status + label + (f"   [{got}]" if got != "" else ""), flush=True)
    if not condition:
        FAILURES.append(label)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Resolve + Step 0 ─────────────────────────────────────────────────────────────
    print("=== resolve + Step 0 (build collection provenance) ===", flush=True)
    ids = resolve_experiment_ids(
        ["20260624_2x_td_bf_pbx_coll"], raw_root=RAW, microscope="YX1"
    )
    print(f"resolved: {ids}", flush=True)
    check("ONE plate id resolved", len(ids) == 1, str(ids))
    experiment_id = ids[0]
    # The grammar fix: plate01 (not "pbx", which the positional parse produced).
    check("id ends _coll_plate01", experiment_id.endswith("_coll_plate01"), experiment_id)

    payload = build_collection_provenance(experiment_id, RAW, "YX1")
    (OUT / "cls.json").write_text(json.dumps(payload, indent=2) + "\n")
    ordinals = [s["source_ordinal"] for s in payload["sources"]]
    ages = payload["start_age_by_source_ordinal"]
    check("3 sources, ordinals 0..2", ordinals == [0, 1, 2], str(ordinals))
    # The grammar fix again: ages parsed from the descriptive-middle names.
    check("ages parsed t33/t52/t77", ages == {"0": 33, "1": 52, "2": 77}, str(ages))

    # ── Step 1: the TWO artifacts, one read per source ───────────────────────────────
    print("=== Step 1 (scope union + acquisition inventory) ===", flush=True)
    scope = ingest_collection_scope_metadata(
        experiment_id=experiment_id,
        sources=payload["sources"],
        microscope="YX1",
        output_csv=OUT / "scope.csv",
    )
    print(f"scope rows={len(scope)} cols={len(scope.columns)}", flush=True)
    check("scope has stage geometry (YX1 needs x/y)", {"x_um", "y_um"} <= set(scope.columns))
    check("scope has channel token", "channel" in scope.columns)
    check("3 source ordinals present", sorted(scope["source_ordinal"].unique()) == [0, 1, 2])
    check("raw_time_index preserved", "raw_time_index" in scope.columns)
    # YX1 mints NO well identity at ingest — the re-key is legitimately a no-op here.
    check("YX1 scope has NO well_id at ingest", "well_id" not in scope.columns)

    # PER-SOURCE-LOSSLESS: every source's geometry rows survive, tagged by ordinal. NOTE this pilot
    # was NOT re-seated between acquisitions, so the three sources report the SAME stage coordinates
    # — identical values here are the real data, not a collapse. What matters is that each source's
    # rows are present under its own ordinal (checked below), which is what lets Step 2 map each
    # source against its own geometry whether or not the coordinates happen to differ.
    print("distinct stage positions per source:", flush=True)
    print(
        scope.groupby("source_ordinal")
        .apply(lambda g: g[["x_um", "y_um"]].drop_duplicates().shape[0], include_groups=False)
        .to_string(),
        flush=True,
    )
    rows_per_source = scope.groupby("source_ordinal").size()
    check("every source contributes rows", (rows_per_source > 0).all(), str(dict(rows_per_source)))
    check("sources contribute EQUAL row counts (same plate, same layout)",
          rows_per_source.nunique() == 1, str(dict(rows_per_source)))

    # Artifact-driven (keystone rule): sources come from the provenance payload, never a re-glob.
    inventory = ingest_collection_acquisition_inventory(
        experiment_id=experiment_id,
        sources=payload["sources"],
        microscope="YX1",
        output_csv=OUT / "inv.csv",
    )
    print(f"inventory rows={len(inventory)}", flush=True)

    # ── Keystone: both artifacts agree on the source key ─────────────────────────────
    print("=== keystone: cross-artifact agreement ===", flush=True)
    def relation(df):
        return sorted(set(zip(df["source_ordinal"], df["raw_time_index"], df["time_index"])))

    scope_rel, inv_rel = relation(scope), relation(inventory)
    # Print a SUMMARY, not 288 tuples: the per-source merged block and its raw span.
    def blocks(df):
        grouped = df.groupby("source_ordinal")
        return {
            int(o): (
                int(g["raw_time_index"].min()), int(g["raw_time_index"].max()),
                int(g["time_index"].min()), int(g["time_index"].max()),
            )
            for o, g in grouped
        }

    print(f"scope  ordinal -> (raw_min,raw_max, merged_min,merged_max): {blocks(scope)}", flush=True)
    print(f"inv    ordinal -> (raw_min,raw_max, merged_min,merged_max): {blocks(inventory)}", flush=True)
    check("(ordinal, raw_time_index) -> time_index IDENTICAL", scope_rel == inv_rel)
    check("same source ordinals", set(scope["source_ordinal"]) == set(inventory["source_ordinal"]))

    # These sources are SNAPSHOTS (no T axis in the ND2 -> one timepoint each), so each ordinal owns
    # exactly one merged time_index: 0, 1, 2 — matching the declared t33/t52/t77 ages.
    scope_blocks = blocks(scope)
    check("each source owns a contiguous merged block", all(
        (hi - lo) == (raw_hi - raw_lo) for raw_lo, raw_hi, lo, hi in scope_blocks.values()
    ), str(scope_blocks))
    check("blocks are disjoint and consecutive", [
        (lo, hi) for _raw_lo, _raw_hi, lo, hi in scope_blocks.values()
    ] == sorted({
        (lo, hi) for _raw_lo, _raw_hi, lo, hi in scope_blocks.values()
    }) and all(
        scope_blocks[o][2] == scope_blocks[o - 1][3] + 1 for o in sorted(scope_blocks) if o > 0
    ))
    check("every source restarts its OWN numbering at 0", all(
        raw_lo == 0 for raw_lo, _raw_hi, _lo, _hi in scope_blocks.values()
    ))

    # ── Step 2: per-source mapping (the reason YX1 cannot map once) ──────────────────
    print("=== Step 2 (per-source position mapping) ===", flush=True)
    mapping = map_collection_positions_to_wells(
        experiment_id=experiment_id,
        sources=payload["sources"],
        scope_metadata_csv=OUT / "scope.csv",
        output_mapping_csv=OUT / "map.csv",
        output_provenance_json=OUT / "prov.json",
        microscope="YX1",
        raw_root=RAW / "20260624_2x_td_bf_pbx_coll",
        ref_xy_csv=REF_XY,
    )
    print(f"mapping rows={len(mapping)}", flush=True)
    print(mapping.head(12).to_string(), flush=True)
    check("mapping has source_ordinal", "source_ordinal" in mapping.columns)
    check("one block per source", sorted(mapping["source_ordinal"].unique()) == [0, 1, 2])
    check("well_id PLATE-keyed", mapping["well_id"].astype(str).str.startswith(experiment_id).all())
    # The collision that broke the old contract: position_index recurs per source.
    per_source_positions = mapping.groupby("source_ordinal")["position_index"].nunique()
    check("positions mapped in every source", (per_source_positions > 0).all(), str(dict(per_source_positions)))

    # ── Step 3: apply — the MergeError blocker ───────────────────────────────────────
    print("=== Step 3 (apply — the many_to_one blocker) ===", flush=True)
    mapped = apply_position_to_well_mapping(
        scope_metadata_csv=OUT / "scope.csv",
        mapping_csv=OUT / "map.csv",
        output_csv=OUT / "mapped.csv",
        experiment_id=experiment_id,
    )
    print(f"mapped rows={len(mapped)}", flush=True)
    check("apply did not drop rows", len(mapped) == len(scope), f"{len(mapped)} vs {len(scope)}")
    check("every row got a well_id", not mapped["well_id"].isna().any())

    # Snapshot sources -> the merged axis has ONE timepoint per source.
    n_timepoints = mapped["time_index"].nunique()
    print(f"merged timepoints: {n_timepoints}", flush=True)
    expected_timepoints = int(scope["time_index"].nunique())
    check("merged timepoints preserved through apply", n_timepoints == expected_timepoints,
          f"{n_timepoints} vs {expected_timepoints}")

    n_wells = mapped["well_id"].nunique()
    print(f"distinct wells: {n_wells}", flush=True)
    check("wells resolved", n_wells > 0, str(n_wells))

    # THE MERGE WORKING: one physical well appears in every source's block. That is what makes the
    # collection one plate rather than three unrelated experiments.
    ordinals_per_well = mapped.groupby("well_id")["source_ordinal"].nunique()
    n_sources = len(payload["sources"])
    shared = int((ordinals_per_well == n_sources).sum())
    print(f"wells present in all {n_sources} sources: {shared}/{n_wells}", flush=True)
    check(f"wells shared across all {n_sources} sources", shared == n_wells, f"{shared}/{n_wells}")

    print(flush=True)
    print("RESULT: " + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"),
          flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("RESULT: EXCEPTION", flush=True)
        sys.exit(2)
