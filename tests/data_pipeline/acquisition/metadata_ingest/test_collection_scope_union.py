"""Tests for the collection SCOPE METADATA union (Step 1 of the collection worklist).

Pins the contract of ``union_collection_scope_metadata``:

  * PER-SOURCE-LOSSLESS: each source keeps its OWN geometry / calibration / timing. The union
    concatenates and TAGS; it must never dedup or collapse values as if they were shared (the
    plate is re-seated between sources, so x/y legitimately differ).
  * source identity stamped per row (source_ordinal / source_id / source_path)
  * merged time axis via the shared remapper (raw_time_index preserved)
  * the PER-SCOPE re-key seam: Keyence's ingest-minted ids get rebound to the plate; YX1 has
    nothing to rebind. Crucially the re-key runs AFTER the time remap, so composite ids are built
    from the MERGED coordinate.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_scope_union.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_ingest import (
    _rekey_keyence_scope_metadata_to_plate,
)
from data_pipeline.acquisition.metadata_ingest.collection_scope_union import (
    SOURCE_IDENTITY_COLUMNS,
    union_collection_scope_metadata,
)
from data_pipeline.acquisition.metadata_ingest.scope.shared.acquisition_channels import (
    CHANNEL_ID_COLUMN,
)

PLATE = "chem28c_coll_plate01"


def _sources(n=2):
    return [
        {
            "file": f"2025062{i + 2}_plate01_t{28 + i * 24}hpf",
            "raw_path": f"/raw/chem28c_coll/2025062{i + 2}_plate01_t{28 + i * 24}hpf",
            "declared_hpf": 28 + i * 24,
            "source_ordinal": i,
            "time_index": i,
        }
        for i in range(n)
    ]


def _keyence_block(source_index, *, x_um, n_wells=2, n_frames=1):
    """A Keyence-shaped per-source scope frame: well identity ALREADY minted, source-bound."""
    rows = []
    for well_number in range(1, n_wells + 1):
        well_index = f"A0{well_number}"
        for time_index in range(n_frames):
            rows.append(
                {
                    # Source-bound: minted against the SOURCE name, not the plate.
                    "experiment_id": f"source{source_index}",
                    "well_index": well_index,
                    "well_id": f"source{source_index}_{well_index}",
                    "image_id": f"source{source_index}_{well_index}_BF_t{time_index:04d}",
                    "raw_position_label": str(well_number),
                    "time_index": time_index,
                    CHANNEL_ID_COLUMN: "BF",
                    # Per-source acquisition facts — MUST survive distinct.
                    "x_um": x_um + well_number,
                    "micrometers_per_pixel": 0.5 + source_index,
                }
            )
    return pd.DataFrame(rows)


def _yx1_block(source_index, *, x_um, n_positions=2):
    """A YX1-shaped per-source scope frame: NO well identity (attached later at apply)."""
    return pd.DataFrame(
        {
            "experiment_id": [f"source{source_index}"] * n_positions,
            "raw_position_label": [str(p) for p in range(n_positions)],
            "time_index": [0] * n_positions,
            CHANNEL_ID_COLUMN: ["BF"] * n_positions,
            "x_um": [x_um + p for p in range(n_positions)],
        }
    )


# ── Per-source-lossless: distinct acquisition facts must all survive ──────────────────


def test_each_source_keeps_its_own_geometry():
    blocks = {0: _keyence_block(0, x_um=1000.0), 1: _keyence_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
    )
    # Two DISTINCT stage frames — the plate was re-seated. Nothing may be deduped/collapsed.
    by_ordinal = unioned.groupby("source_ordinal")["x_um"].apply(set)
    assert by_ordinal.loc[0] == {1001.0, 1002.0}
    assert by_ordinal.loc[1] == {9001.0, 9002.0}
    # Per-source calibration also differs and survives.
    assert set(unioned.groupby("source_ordinal")["micrometers_per_pixel"].first()) == {0.5, 1.5}


def test_all_rows_are_kept():
    blocks = {0: _keyence_block(0, x_um=1000.0), 1: _keyence_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
    )
    assert len(unioned) == 4  # 2 sources x 2 wells


# ── Source identity + merged time axis ────────────────────────────────────────────────


def test_source_identity_columns_come_from_the_contract():
    """The union must stamp exactly what SOURCE_IDENTITY_COLUMNS declares — imported, not restated."""
    blocks = {0: _keyence_block(0, x_um=1000.0), 1: _keyence_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
    )
    missing = [c for c in SOURCE_IDENTITY_COLUMNS if c not in unioned.columns]
    assert not missing, f"union did not stamp contract columns {missing}"


def test_source_identity_is_stamped():
    blocks = {0: _keyence_block(0, x_um=1000.0), 1: _keyence_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
    )
    assert sorted(unioned["source_ordinal"].unique()) == [0, 1]
    assert unioned["experiment_id"].unique().tolist() == [PLATE]
    assert set(unioned["source_id"]) == {
        "20250622_plate01_t28hpf",
        "20250623_plate01_t52hpf",
    }


def test_two_snapshot_sources_get_distinct_merged_time_index():
    blocks = {0: _keyence_block(0, x_um=1000.0), 1: _keyence_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
    )
    # Both sources natively call their frame 0; merged they must be 0 and 1.
    assert sorted(unioned["time_index"].unique()) == [0, 1]
    assert unioned["raw_time_index"].unique().tolist() == [0]
    assert set(zip(unioned["source_ordinal"], unioned["time_index"])) == {(0, 0), (1, 1)}


def test_timelapse_source_spans_a_range_of_time_index():
    # The worklist requires this: one source_ordinal may span MANY time_index values.
    blocks = {
        0: _keyence_block(0, x_um=1000.0, n_frames=3),
        1: _keyence_block(1, x_um=9000.0, n_frames=1),
    }
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
    )
    by_ordinal = unioned.groupby("source_ordinal")["time_index"].apply(set)
    assert by_ordinal.loc[0] == {0, 1, 2}
    assert by_ordinal.loc[1] == {3}


# ── The PER-SCOPE re-key seam ─────────────────────────────────────────────────────────


def test_keyence_ids_are_rebound_to_the_plate():
    blocks = {0: _keyence_block(0, x_um=1000.0), 1: _keyence_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
        rekey_to_plate=_rekey_keyence_scope_metadata_to_plate,
    )
    # A01 is ONE well across both sources — the point of merging the plate.
    assert set(unioned["well_id"]) == {f"{PLATE}_A01", f"{PLATE}_A02"}
    assert not any(wid.startswith("source") for wid in unioned["well_id"])


def test_rekey_builds_image_id_from_the_MERGED_time_index():
    """Regression: the re-key must run AFTER the time remap.

    image_id is (well_id, channel_id, time_index). Re-keying BEFORE the remap stamps each source's
    own pre-merge time_index, so two snapshot sources both produce ..._t0000 and collide.
    """
    blocks = {0: _keyence_block(0, x_um=1000.0), 1: _keyence_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
        rekey_to_plate=_rekey_keyence_scope_metadata_to_plate,
    )
    assert set(unioned["image_id"]) == {
        f"{PLATE}_A01_BF_t0000",
        f"{PLATE}_A02_BF_t0000",
        f"{PLATE}_A01_BF_t0001",
        f"{PLATE}_A02_BF_t0001",
    }
    # 2 wells x 2 timepoints, all distinct — no collision.
    assert unioned["image_id"].nunique() == 4


def test_yx1_needs_no_rekey():
    # YX1 mints no well identity at ingest, so rekey_to_plate is legitimately None.
    blocks = {0: _yx1_block(0, x_um=1000.0), 1: _yx1_block(1, x_um=9000.0)}
    unioned = union_collection_scope_metadata(
        experiment_id=PLATE,
        sources=_sources(),
        read_source=lambda rec, _exp: blocks[rec["source_ordinal"]],
        rekey_to_plate=None,
    )
    assert "well_id" not in unioned.columns
    assert unioned["experiment_id"].unique().tolist() == [PLATE]
    assert sorted(unioned["time_index"].unique()) == [0, 1]


def test_keyence_rekey_without_well_index_fails_loud():
    block = _keyence_block(0, x_um=1000.0).drop(columns=["well_index"])
    with pytest.raises(ValueError, match="missing 'well_index'"):
        _rekey_keyence_scope_metadata_to_plate(block, PLATE)


# ── Fail loud ────────────────────────────────────────────────────────────────────────


def test_no_sources_fails_loud():
    with pytest.raises(ValueError, match="no sources given"):
        union_collection_scope_metadata(
            experiment_id=PLATE, sources=[], read_source=lambda rec, _exp: pd.DataFrame()
        )


def test_duplicate_source_ordinal_fails_loud():
    duped = _sources()
    duped[1]["source_ordinal"] = 0
    with pytest.raises(ValueError, match="both claim source_ordinal"):
        union_collection_scope_metadata(
            experiment_id=PLATE,
            sources=duped,
            read_source=lambda rec, _exp: _keyence_block(0, x_um=1.0),
        )


def test_empty_per_source_read_fails_loud():
    with pytest.raises(ValueError, match="returned empty scope"):
        union_collection_scope_metadata(
            experiment_id=PLATE,
            sources=_sources(1),
            read_source=lambda rec, _exp: pd.DataFrame(),
        )


def test_missing_record_key_fails_loud():
    bad = [{"file": "f", "raw_path": "/r"}]  # no source_ordinal
    with pytest.raises(ValueError, match="source_ordinal"):
        union_collection_scope_metadata(
            experiment_id=PLATE,
            sources=bad,
            read_source=lambda rec, _exp: _keyence_block(0, x_um=1.0),
        )
