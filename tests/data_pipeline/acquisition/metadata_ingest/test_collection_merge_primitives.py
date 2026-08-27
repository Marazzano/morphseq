"""Tests for the collection time axis — the raw→merged time_index contract.

Pins the invariants of ``remap_source_time_indices`` (the ONE helper both collection unions
use, so they cannot drift on what ``time_index`` means):

  * raw_time_index preserves the source-native value; time_index is the merged coordinate
  * rows sharing a raw_time_index share a time_index (a frame is many rows)
  * the merged axis is CONTIGUOUS — sparse / 1-based source numbering is renumbered, not carried
  * the returned offset makes consecutive sources occupy disjoint, adjacent blocks
  * fail-loud on a missing / null / non-numeric source time_index

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_merge_primitives.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.collection_merge_primitives import (
    remap_source_time_indices,
)


def _frame(time_indices, **extra):
    """One source's rows: a time_index per row, plus any extra columns."""
    data = {"time_index": list(time_indices)}
    data.update({k: [v] * len(data["time_index"]) for k, v in extra.items()})
    return pd.DataFrame(data)


# ── The snapshot case (today's real collections) ──────────────────────────────────────


def test_two_snapshot_sources_get_adjacent_single_frames():
    # Each snapshot source calls its only frame 0; merged they must become 0 and 1.
    first, offset = remap_source_time_indices(_frame([0]), 0)
    second, offset = remap_source_time_indices(_frame([0]), offset)

    assert first["time_index"].tolist() == [0]
    assert second["time_index"].tolist() == [1]
    assert offset == 2
    # Provenance: both sources still know they were their own frame 0.
    assert first["raw_time_index"].tolist() == [0]
    assert second["raw_time_index"].tolist() == [0]


def test_three_snapshot_sources_are_contiguous():
    offset = 0
    merged = []
    for _ in range(3):
        out, offset = remap_source_time_indices(_frame([0]), offset)
        merged.extend(out["time_index"].tolist())
    assert merged == [0, 1, 2]
    assert offset == 3


# ── The timelapse case (one source spans a RANGE of time_index) ───────────────────────


def test_timelapse_source_keeps_a_contiguous_block():
    out, offset = remap_source_time_indices(_frame([0, 1, 2]), 0)
    assert out["time_index"].tolist() == [0, 1, 2]
    assert offset == 3


def test_timelapse_then_snapshot_do_not_collide():
    timelapse, offset = remap_source_time_indices(_frame([0, 1, 2]), 0)
    snapshot, offset = remap_source_time_indices(_frame([0]), offset)
    assert timelapse["time_index"].tolist() == [0, 1, 2]
    assert snapshot["time_index"].tolist() == [3]
    assert offset == 4
    # Disjointness is the whole point: no merged timepoint is claimed twice.
    assert not set(timelapse["time_index"]) & set(snapshot["time_index"])


# ── Sparse / non-zero-based source numbering (the trapdoor naive addition falls into) ─


def test_sparse_source_indices_are_renumbered_densely():
    # Keyence reads T#### from disk and subtracts 1, so a partial/resumed acquisition can
    # yield 2, 4. Naive `time_index + offset` would keep the gap AND mis-width the block.
    out, offset = remap_source_time_indices(_frame([2, 4]), 0)
    assert out["time_index"].tolist() == [0, 1]
    assert out["raw_time_index"].tolist() == [2, 4]
    assert offset == 2  # block width is the DISTINCT count, not max+1


def test_one_based_source_indices_are_renumbered():
    out, offset = remap_source_time_indices(_frame([1, 2, 3]), 0)
    assert out["time_index"].tolist() == [0, 1, 2]
    assert out["raw_time_index"].tolist() == [1, 2, 3]
    assert offset == 3


def test_sparse_first_source_does_not_leave_a_hole_for_the_second():
    first, offset = remap_source_time_indices(_frame([5, 9]), 0)
    second, _ = remap_source_time_indices(_frame([0]), offset)
    assert first["time_index"].tolist() == [0, 1]
    # Naive max+1 offsetting would have started the second source at 10.
    assert second["time_index"].tolist() == [2]


def test_unsorted_source_rows_are_ordered_by_native_value():
    out, _ = remap_source_time_indices(_frame([2, 0, 1]), 0)
    # Row order is preserved, but the mapping follows native ordering.
    assert out["raw_time_index"].tolist() == [2, 0, 1]
    assert out["time_index"].tolist() == [2, 0, 1]


def test_numeric_ordering_not_lexical():
    # "10" must sort AFTER "9" — lexical ordering would invert them.
    out, _ = remap_source_time_indices(_frame(["9", "10"]), 0)
    assert out["raw_time_index"].tolist() == [9, 10]
    assert out["time_index"].tolist() == [0, 1]


# ── A frame is many rows: z-planes / tiles / channels must not split a timepoint ──────


def test_rows_sharing_a_raw_time_index_share_a_merged_time_index():
    # 3 z-planes x 2 timepoints, interleaved.
    out, offset = remap_source_time_indices(_frame([0, 0, 0, 1, 1, 1]), 0)
    assert out["time_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert offset == 2  # two timepoints, not six


def test_many_rows_per_frame_with_sparse_indices():
    out, offset = remap_source_time_indices(_frame([4, 4, 7, 7]), 10)
    assert out["time_index"].tolist() == [10, 10, 11, 11]
    assert offset == 12


def test_other_columns_ride_through_untouched():
    out, _ = remap_source_time_indices(_frame([0, 1], well_index="A01"), 0)
    assert out["well_index"].tolist() == ["A01", "A01"]


def test_input_frame_is_not_mutated():
    original = _frame([3, 5])
    remap_source_time_indices(original, 0)
    assert original["time_index"].tolist() == [3, 5]
    assert "raw_time_index" not in original.columns


# ── Fail loud ────────────────────────────────────────────────────────────────────────


def test_missing_time_index_column_fails_loud():
    with pytest.raises(ValueError, match="expected a source-native 'time_index' column"):
        remap_source_time_indices(pd.DataFrame({"well_index": ["A01"]}), 0)


def test_null_time_index_fails_loud():
    with pytest.raises(ValueError, match="missing value"):
        remap_source_time_indices(pd.DataFrame({"time_index": [0, None]}), 0)


def test_non_numeric_time_index_fails_loud():
    with pytest.raises(ValueError, match="integer-valued"):
        remap_source_time_indices(pd.DataFrame({"time_index": ["first", "second"]}), 0)
