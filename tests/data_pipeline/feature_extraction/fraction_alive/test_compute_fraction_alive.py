"""Tests for the pure compute_fraction_alive — especially the snip-frame alignment that fixed the
full-frame embryo vs snip-resolution via mask mismatch."""

import math

import numpy as np

from data_pipeline.feature_extraction.fraction_alive._legacy_compute import compute_fraction_alive

SNIP_FRAME = (576, 256)


def test_mismatched_resolutions_do_not_crash_with_snip_frame_shape() -> None:
    # The original blocker: full-frame embryo mask vs snip-resolution via mask -> shape error.
    embryo = np.zeros((2189, 2189), dtype=np.uint8)
    embryo[800:1400, 800:1200] = 1
    via = np.zeros(SNIP_FRAME, dtype=np.uint8)
    via[100:200, 50:150] = 1

    frac = compute_fraction_alive(embryo, via, snip_frame_shape=SNIP_FRAME)

    assert 0.0 <= frac <= 1.0


def test_no_dead_tissue_is_fully_alive() -> None:
    embryo = np.zeros(SNIP_FRAME, dtype=np.uint8)
    embryo[100:300, 50:150] = 1
    via = np.zeros(SNIP_FRAME, dtype=np.uint8)  # no dead tissue

    assert compute_fraction_alive(embryo, via, snip_frame_shape=SNIP_FRAME) == 1.0


def test_via_covering_whole_embryo_is_zero_alive() -> None:
    embryo = np.zeros(SNIP_FRAME, dtype=np.uint8)
    embryo[100:300, 50:150] = 1
    via = embryo.copy()  # all embryo tissue flagged dead

    assert compute_fraction_alive(embryo, via, snip_frame_shape=SNIP_FRAME) == 0.0


def test_half_dead_is_half_alive() -> None:
    embryo = np.zeros(SNIP_FRAME, dtype=np.uint8)
    embryo[100:300, 50:150] = 1  # 200 x 100 region
    via = np.zeros(SNIP_FRAME, dtype=np.uint8)
    via[100:200, 50:150] = 1  # exactly the top half of the embryo

    frac = compute_fraction_alive(embryo, via, snip_frame_shape=SNIP_FRAME)

    assert math.isclose(frac, 0.5, abs_tol=1e-6)


def test_empty_embryo_is_null() -> None:
    embryo = np.zeros(SNIP_FRAME, dtype=np.uint8)
    via = np.zeros(SNIP_FRAME, dtype=np.uint8)
    via[10:20, 10:20] = 1

    assert math.isnan(compute_fraction_alive(embryo, via, snip_frame_shape=SNIP_FRAME))


def test_fallback_alignment_without_snip_frame_shape() -> None:
    # Backward-compatible 2-arg form still aligns mismatched shapes defensively.
    embryo = np.zeros((400, 400), dtype=np.uint8)
    embryo[100:300, 100:300] = 1
    via = np.zeros((200, 200), dtype=np.uint8)
    via[50:100, 50:100] = 1

    frac = compute_fraction_alive(embryo, via)

    assert 0.0 <= frac <= 1.0
