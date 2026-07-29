"""Tests for the named-axis ND2 reader.

Pins the contract that replaces positional ``nd.shape`` unpacking. The real failure this guards:
a pbx pilot is ``(P, Z, C, Y, X)`` — 96 positions, 9 focal planes, 2 channels, NO time axis — and
reading ``shape[:3]`` as ``(T, W, Z)`` turned 96 wells into 96 timepoints, 9 focal planes into 9
positions, and 2 channels into 2 planes, yielding 1728 plausible-looking (and wrong) rows.

``read_nd2_axes`` takes the ``sizes`` MAPPING, so every real layout is testable without an ND2.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/scope/yx1/test_nd2_axes.py
"""

from __future__ import annotations

import pytest

from data_pipeline.acquisition.metadata_ingest.scope.yx1.nd2_axes import read_nd2_axes

# The real pbx collection layout — the file that exposed the bug.
PBX_PILOT = {"P": 96, "Z": 9, "C": 2, "Y": 2304, "X": 2304}
# A timelapse layout (T present).
TIMELAPSE = {"T": 12, "P": 96, "Z": 5, "C": 1, "Y": 2048, "X": 2048}


# ── Sizes are read by NAME, never by position ─────────────────────────────────────────


def test_real_pbx_pilot_axes_are_read_correctly():
    axes = read_nd2_axes(PBX_PILOT)
    assert axes.n_p == 96, "96 is the POSITION count (was misread as timepoints)"
    assert axes.n_z == 9, "9 is the FOCAL PLANE count (was misread as positions)"
    assert axes.n_c == 2, "2 is the CHANNEL count (was misread as z-planes)"
    assert axes.n_t == 1, "no T axis => a single-timepoint snapshot"
    assert (axes.height_px, axes.width_px) == (2304, 2304)


def test_missing_t_axis_is_a_snapshot_not_an_error():
    axes = read_nd2_axes(PBX_PILOT)
    assert axes.is_snapshot is True
    assert axes.n_t == 1


def test_timelapse_axes():
    axes = read_nd2_axes(TIMELAPSE)
    assert (axes.n_t, axes.n_p, axes.n_z, axes.n_c) == (12, 96, 5, 1)
    assert axes.is_snapshot is False


def test_legacy_W_is_the_position_axis():
    axes = read_nd2_axes({"T": 3, "W": 24, "Z": 4, "Y": 512, "X": 512})
    assert axes.n_p == 24
    assert axes.n_c == 1  # absent axis => 1


def test_absent_axes_default_to_one():
    axes = read_nd2_axes({"Y": 64, "X": 64})
    assert (axes.n_t, axes.n_p, axes.n_z, axes.n_c) == (1, 1, 1, 1)
    assert axes.n_rows == 1
    assert axes.n_frames == 1


def test_row_grain_includes_channels():
    assert read_nd2_axes(PBX_PILOT).n_rows == 96 * 9 * 2
    assert read_nd2_axes(TIMELAPSE).n_rows == 12 * 96 * 5 * 1


# ── Flat frame indexing follows the file's ACTUAL axis order ───────────────────────────


def test_C_is_NOT_a_sequence_axis():
    """The core distinction: one frame carries all its channels.

    Verified against the real file: sizes give P*Z*C = 1728 metadata rows, but the ND2 declares only
    864 frames = P*Z. Counting C as addressable overshoots by 2x and runs off the end of the file.
    """
    axes = read_nd2_axes(PBX_PILOT, ["XYPosLoop", "ZStackLoop"])
    assert axes.sequence_order == ("P", "Z")
    assert axes.n_frames == 96 * 9 == 864
    assert axes.n_rows == 96 * 9 * 2 == 1728


def test_frame_index_for_the_pbx_layout():
    axes = read_nd2_axes(PBX_PILOT, ["XYPosLoop", "ZStackLoop"])
    assert axes.frame_index(position=0, z=0) == 0
    assert axes.frame_index(position=0, z=1) == 1
    assert axes.frame_index(position=1, z=0) == 9
    assert axes.frame_index(position=95, z=8) == axes.n_frames - 1


def test_loop_order_drives_the_strides():
    """Identical sizes, different LOOP order => different strides."""
    p_outer = read_nd2_axes(PBX_PILOT, ["XYPosLoop", "ZStackLoop"])
    z_outer = read_nd2_axes(PBX_PILOT, ["ZStackLoop", "XYPosLoop"])
    assert p_outer.frame_index(position=1) == 9    # stride = Z
    assert z_outer.frame_index(position=1) == 1    # position is innermost
    assert z_outer.frame_index(z=1) == 96          # stride = P


def test_frame_index_with_a_time_axis():
    axes = read_nd2_axes(TIMELAPSE, ["TimeLoop", "XYPosLoop", "ZStackLoop"])
    assert axes.sequence_order == ("T", "P", "Z")
    assert axes.n_frames == 12 * 96 * 5
    assert axes.frame_index(time=0, position=0, z=0) == 0
    assert axes.frame_index(time=1, position=0, z=0) == 96 * 5
    assert axes.frame_index(time=0, position=1, z=0) == 5


def test_frame_index_takes_no_channel_argument():
    axes = read_nd2_axes(PBX_PILOT, ["XYPosLoop", "ZStackLoop"])
    with pytest.raises(TypeError):
        axes.frame_index(position=0, channel=1)


def test_frame_index_rejects_a_coordinate_on_an_absent_axis():
    axes = read_nd2_axes(PBX_PILOT, ["XYPosLoop", "ZStackLoop"])  # no T axis
    with pytest.raises(IndexError, match="T=1 is out of range"):
        axes.frame_index(position=0, time=1)


def test_frame_index_rejects_out_of_range_coordinates():
    axes = read_nd2_axes(PBX_PILOT, ["XYPosLoop", "ZStackLoop"])
    with pytest.raises(IndexError, match="P=96 is out of range"):
        axes.frame_index(position=96)
    with pytest.raises(IndexError, match="Z=9 is out of range"):
        axes.frame_index(z=9)


def test_every_sequence_coordinate_maps_to_a_distinct_frame():
    axes = read_nd2_axes({"P": 4, "Z": 3, "C": 2, "Y": 8, "X": 8}, ["XYPosLoop", "ZStackLoop"])
    seen = {axes.frame_index(position=p, z=z) for p in range(4) for z in range(3)}
    assert seen == set(range(axes.n_frames))


def test_unknown_loop_fails_loud():
    with pytest.raises(ValueError, match="unrecognized ND2 experiment loop"):
        read_nd2_axes(PBX_PILOT, ["XYPosLoop", "SpectralLoop"])


def test_loop_without_a_matching_size_fails_loud():
    # A TimeLoop on a file whose sizes declare no T: the structures disagree.
    with pytest.raises(ValueError, match="no such axis"):
        read_nd2_axes(PBX_PILOT, ["TimeLoop", "XYPosLoop", "ZStackLoop"])


# ── Fail loud ────────────────────────────────────────────────────────────────────────


def test_missing_spatial_axes_fail_loud():
    with pytest.raises(ValueError, match="missing spatial axis"):
        read_nd2_axes({"P": 4, "Z": 2})


def test_unknown_axis_label_fails_loud():
    # Never silently ignore an axis: unindexed frames would be read from the wrong offsets.
    with pytest.raises(ValueError, match="unrecognized ND2 axis"):
        read_nd2_axes({"P": 4, "Q": 7, "Y": 8, "X": 8})


def test_both_P_and_W_is_ambiguous():
    with pytest.raises(ValueError, match="BOTH 'P' and 'W'"):
        read_nd2_axes({"P": 4, "W": 4, "Y": 8, "X": 8})


# ── select_zyx_stack: the PIXEL path (wrong axes here corrupt images, not just metadata) ──


import numpy as np

from data_pipeline.acquisition.metadata_ingest.scope.yx1.nd2_axes import select_zyx_stack


def _labelled_array(order, sizes):
    """An array whose every voxel encodes its own coordinate, so mis-slicing is detectable."""
    shape = tuple(sizes[a] for a in order)
    arr = np.zeros(shape, dtype=np.int64)
    for idx in np.ndindex(*shape):
        coord = dict(zip(order, idx))
        # Encode P/Z/C/T distinctly; Y/X are ignored so every pixel of a plane shares the code.
        arr[idx] = (
            coord.get("P", 0) * 1_000_000
            + coord.get("T", 0) * 10_000
            + coord.get("Z", 0) * 100
            + coord.get("C", 0)
        )
    return arr


def test_snapshot_with_channels_returns_the_REAL_z_stack():
    """The real pbx layout (P, Z, C, Y, X).

    Positional indexing (`arr[t, w, :, :, :]`) returns shape (C, Y, X) here — a 2-plane
    "z-stack" that is actually the CHANNEL axis, so focus-stacking would run over
    [BF, fluorescence] as if they were focal planes. Verified against the real ND2:
    old path gave (2, 2304, 2304); named-axis path gives (9, 2304, 2304).
    """
    order = ("P", "Z", "C", "Y", "X")
    sizes = {"P": 4, "Z": 9, "C": 2, "Y": 3, "X": 3}
    arr = _labelled_array(order, sizes)
    axes = read_nd2_axes(sizes, ["XYPosLoop", "ZStackLoop"])

    stack = select_zyx_stack(arr, axes, order, position=2, channel=1)
    assert stack.shape == (9, 3, 3), "must be the 9 focal planes, not the 2 channels"
    # Every plane carries its own z code, at the requested position and channel.
    for z in range(9):
        assert stack[z, 0, 0] == 2 * 1_000_000 + z * 100 + 1


def test_channels_are_separated_not_stacked():
    order = ("P", "Z", "C", "Y", "X")
    sizes = {"P": 2, "Z": 4, "C": 2, "Y": 2, "X": 2}
    arr = _labelled_array(order, sizes)
    axes = read_nd2_axes(sizes, ["XYPosLoop", "ZStackLoop"])

    bf = select_zyx_stack(arr, axes, order, position=0, channel=0)
    rfp = select_zyx_stack(arr, axes, order, position=0, channel=1)
    assert bf.shape == rfp.shape == (4, 2, 2)
    assert not np.array_equal(bf, rfp), "distinct channels must not return identical pixels"


def test_timelapse_layout_selects_the_right_frame():
    order = ("T", "P", "Z", "C", "Y", "X")
    sizes = {"T": 3, "P": 2, "Z": 5, "C": 2, "Y": 2, "X": 2}
    arr = _labelled_array(order, sizes)
    axes = read_nd2_axes(sizes, ["TimeLoop", "XYPosLoop", "ZStackLoop"])

    stack = select_zyx_stack(arr, axes, order, position=1, time=2, channel=0)
    assert stack.shape == (5, 2, 2)
    assert stack[0, 0, 0] == 1 * 1_000_000 + 2 * 10_000


def test_layout_without_channels_still_works():
    order = ("T", "P", "Z", "Y", "X")
    sizes = {"T": 2, "P": 2, "Z": 3, "Y": 2, "X": 2}
    arr = _labelled_array(order, sizes)
    axes = read_nd2_axes(sizes, ["TimeLoop", "XYPosLoop", "ZStackLoop"])
    stack = select_zyx_stack(arr, axes, order, position=1, time=1)
    assert stack.shape == (3, 2, 2)


def test_unusual_axis_order_is_handled():
    """Z outermost — positional code would be badly wrong; named slicing is unaffected."""
    order = ("Z", "P", "C", "Y", "X")
    sizes = {"Z": 6, "P": 3, "C": 2, "Y": 2, "X": 2}
    arr = _labelled_array(order, sizes)
    axes = read_nd2_axes(sizes, ["ZStackLoop", "XYPosLoop"])
    stack = select_zyx_stack(arr, axes, order, position=2, channel=1)
    assert stack.shape == (6, 2, 2)
    for z in range(6):
        assert stack[z, 0, 0] == 2 * 1_000_000 + z * 100 + 1


def test_missing_z_axis_fails_loud():
    order = ("P", "C", "Y", "X")
    sizes = {"P": 2, "C": 2, "Y": 2, "X": 2}
    arr = _labelled_array(order, sizes)
    axes = read_nd2_axes(sizes, ["XYPosLoop"])
    with pytest.raises(ValueError, match="no 'Z' axis"):
        select_zyx_stack(arr, axes, order, position=0)
