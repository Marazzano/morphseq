"""Tests for the SOURCE-AWARE join in apply_position_to_well_mapping.

Covers the YX1-shaped merge branch (scope metadata carries NO well identity, so identity is joined
in from the mapping). This is the branch a collection plate previously could not use at all:

    merge(on=["experiment_id", "position_index"], validate="many_to_one")

A collection mapping holds one block per raw source, so position_index recurs once per source and
the right side is not unique → pandas MergeError. `source_ordinal` is what disambiguates it.

The key is the SOURCE, not the frame: one source can span MANY merged time_index values and every
frame of that source inherits the same position→well mapping. These tests pin that, plus the
single-experiment path staying byte-identical.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_apply_source_keyed_join.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.scope.shared.acquisition_channels import (
    CHANNEL_ID_COLUMN,
)

from data_pipeline.acquisition.metadata_ingest.scope.shared.apply_position_to_well_mapping import (
    apply_position_to_well_mapping,
)

PLATE = "chem28c_coll_plate01"
SINGLE = "20250912"


def _write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _yx1_scope_rows(experiment_id, *, source_ordinal, positions, time_indices):
    """YX1-shaped scope metadata: raw_position_label + geometry, NO well_index/well_id."""
    rows = []
    for position in positions:
        for time_index in time_indices:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "raw_position_label": position,
                    "time_index": time_index,
                    "source_ordinal": source_ordinal,
                    CHANNEL_ID_COLUMN: "BF",
                    "x_um": 1000.0 + position + 5000 * source_ordinal,
                    "y_um": 2000.0,
                }
            )
    return rows


def _collection_mapping(n_sources=2, positions=(0, 1)):
    """One mapping block per source — position_index REPEATS across blocks by design."""
    rows = []
    for source_ordinal in range(n_sources):
        for position in positions:
            well_index = f"A0{position + 1}"
            rows.append(
                {
                    "experiment_id": PLATE,
                    "position_index": position,
                    "well_index": well_index,
                    "well_id": f"{PLATE}_{well_index}",
                    "mapping_method": "collection_per_source_map",
                    "source_ordinal": source_ordinal,
                    "source_file": f"source{source_ordinal}",
                }
            )
    return pd.DataFrame(rows)


# ── The blocker: a collection mapping through the merge branch ─────────────────────────


def test_collection_mapping_joins_by_source_ordinal(tmp_path):
    scope = pd.DataFrame(
        _yx1_scope_rows(PLATE, source_ordinal=0, positions=(0, 1), time_indices=(0,))
        + _yx1_scope_rows(PLATE, source_ordinal=1, positions=(0, 1), time_indices=(1,))
    )
    out = tmp_path / "mapped.csv"
    result = apply_position_to_well_mapping(
        scope_metadata_csv=_write(tmp_path, "scope.csv", scope),
        mapping_csv=_write(tmp_path, "map.csv", _collection_mapping()),
        output_csv=out,
        experiment_id=PLATE,
    )
    assert len(result) == 4
    assert set(result["well_id"]) == {f"{PLATE}_A01", f"{PLATE}_A02"}
    # Each source's position 0 resolved to A01 within its OWN block.
    got = set(zip(result["source_ordinal"], result["position_index"], result["well_id"]))
    assert got == {
        (0, 0, f"{PLATE}_A01"),
        (0, 1, f"{PLATE}_A02"),
        (1, 0, f"{PLATE}_A01"),
        (1, 1, f"{PLATE}_A02"),
    }


def test_per_source_mapping_can_differ_across_sources(tmp_path):
    """The whole reason YX1 must map per source: the plate is re-seated between acquisitions.

    Source 0 has position 0 → A01; source 1 has position 0 → A02 (a different physical well at the
    same stage index). A single shared mapping could not express this.
    """
    mapping = pd.DataFrame(
        [
            {"experiment_id": PLATE, "position_index": 0, "well_index": "A01",
             "well_id": f"{PLATE}_A01", "mapping_method": "m", "source_ordinal": 0},
            {"experiment_id": PLATE, "position_index": 0, "well_index": "A02",
             "well_id": f"{PLATE}_A02", "mapping_method": "m", "source_ordinal": 1},
        ]
    )
    scope = pd.DataFrame(
        _yx1_scope_rows(PLATE, source_ordinal=0, positions=(0,), time_indices=(0,))
        + _yx1_scope_rows(PLATE, source_ordinal=1, positions=(0,), time_indices=(1,))
    )
    result = apply_position_to_well_mapping(
        scope_metadata_csv=_write(tmp_path, "scope.csv", scope),
        mapping_csv=_write(tmp_path, "map.csv", mapping),
        output_csv=tmp_path / "mapped.csv",
        experiment_id=PLATE,
    )
    by_source = dict(zip(result["source_ordinal"], result["well_id"]))
    assert by_source == {0: f"{PLATE}_A01", 1: f"{PLATE}_A02"}


def test_timelapse_source_all_frames_inherit_one_mapping(tmp_path):
    """One source_ordinal spanning MANY time_index values still needs only ONE mapping row.

    This is why the join key is the source, not the frame.
    """
    scope = pd.DataFrame(
        # Source 0 is a timelapse: merged time_index 0,1,2 — all one source.
        _yx1_scope_rows(PLATE, source_ordinal=0, positions=(0, 1), time_indices=(0, 1, 2))
        + _yx1_scope_rows(PLATE, source_ordinal=1, positions=(0, 1), time_indices=(3,))
    )
    result = apply_position_to_well_mapping(
        scope_metadata_csv=_write(tmp_path, "scope.csv", scope),
        mapping_csv=_write(tmp_path, "map.csv", _collection_mapping()),
        output_csv=tmp_path / "mapped.csv",
        experiment_id=PLATE,
    )
    assert len(result) == 8  # (2 positions x 3 frames) + (2 positions x 1 frame)
    assert not result["well_id"].isna().any()
    # Every frame of source 0 at position 0 got the same well.
    src0_pos0 = result[(result["source_ordinal"] == 0) & (result["position_index"] == 0)]
    assert set(src0_pos0["well_id"]) == {f"{PLATE}_A01"}
    assert sorted(src0_pos0["time_index"]) == [0, 1, 2]


def test_mapping_with_source_ordinal_but_scope_without_fails_loud(tmp_path):
    scope = pd.DataFrame(
        _yx1_scope_rows(PLATE, source_ordinal=0, positions=(0, 1), time_indices=(0,))
    ).drop(columns=["source_ordinal"])
    with pytest.raises(ValueError, match="same source key"):
        apply_position_to_well_mapping(
            scope_metadata_csv=_write(tmp_path, "scope.csv", scope),
            mapping_csv=_write(tmp_path, "map.csv", _collection_mapping()),
            output_csv=tmp_path / "mapped.csv",
            experiment_id=PLATE,
        )


def test_uncovered_position_error_names_the_full_join_key(tmp_path):
    # Source 1 has no mapping block at all.
    mapping = _collection_mapping(n_sources=1)
    scope = pd.DataFrame(
        _yx1_scope_rows(PLATE, source_ordinal=0, positions=(0, 1), time_indices=(0,))
        + _yx1_scope_rows(PLATE, source_ordinal=1, positions=(0, 1), time_indices=(1,))
    )
    with pytest.raises(ValueError, match="source_ordinal"):
        apply_position_to_well_mapping(
            scope_metadata_csv=_write(tmp_path, "scope.csv", scope),
            mapping_csv=_write(tmp_path, "map.csv", mapping),
            output_csv=tmp_path / "mapped.csv",
            experiment_id=PLATE,
        )


# ── The single-experiment path must be untouched ──────────────────────────────────────


def test_single_experiment_join_is_unchanged(tmp_path):
    """No source_ordinal anywhere → joins on exactly the original keys."""
    scope = pd.DataFrame(
        [
            {"experiment_id": SINGLE, "raw_position_label": 0, "time_index": t,
             CHANNEL_ID_COLUMN: "BF", "x_um": 1.0, "y_um": 2.0}
            for t in (0, 1, 2)
        ]
    )
    mapping = pd.DataFrame(
        [{"experiment_id": SINGLE, "position_index": 0, "well_index": "B01",
          "well_id": f"{SINGLE}_B01", "mapping_method": "yx1_xy_match"}]
    )
    result = apply_position_to_well_mapping(
        scope_metadata_csv=_write(tmp_path, "scope.csv", scope),
        mapping_csv=_write(tmp_path, "map.csv", mapping),
        output_csv=tmp_path / "mapped.csv",
        experiment_id=SINGLE,
    )
    assert len(result) == 3
    assert set(result["well_id"]) == {f"{SINGLE}_B01"}
    assert set(result["image_id"]) == {
        f"{SINGLE}_B01_BF_t0000",
        f"{SINGLE}_B01_BF_t0001",
        f"{SINGLE}_B01_BF_t0002",
    }


def test_string_typed_source_ordinal_still_matches(tmp_path):
    """A CSV round-trip can type one side as object; coercion must happen before the merge."""
    scope = pd.DataFrame(
        _yx1_scope_rows(PLATE, source_ordinal=0, positions=(0,), time_indices=(0,))
    )
    scope["source_ordinal"] = scope["source_ordinal"].astype(str)
    mapping = _collection_mapping(n_sources=1, positions=(0,))
    result = apply_position_to_well_mapping(
        scope_metadata_csv=_write(tmp_path, "scope.csv", scope),
        mapping_csv=_write(tmp_path, "map.csv", mapping),
        output_csv=tmp_path / "mapped.csv",
        experiment_id=PLATE,
    )
    assert set(result["well_id"]) == {f"{PLATE}_A01"}
