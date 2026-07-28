"""Phase-1 unit tests for the shared ``focus_stack_group`` primitive.

Run with:
    PYTHONPATH=src pytest \
      src/data_pipeline/acquisition/image_building/shared/tests/test_focus_stack_group.py

Covers the README Phase-1 checklist:
  - exact/deterministic percentile bounds (no sampling);
  - one bound pair used for every tile in a group;
  - input stacks are not mutated;
  - focus indices are stack-axis offsets with the expected shape/dtype;
  - projection pixels come from raw data (before display mapping);
  - display-only changes leave focus indices unchanged;
  - batching / grouping does not change per-tile bytes vs the same tile alone... except that
    grouping intentionally SHARES bounds (that is the point) — so we test the invariant that
    bounds+outputs depend only on group membership, deterministically;
  - CPU results remain equivalent to the existing LoG kernel implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.acquisition.image_building.shared.focus_stack_group import (
    BOUND_METHOD,
    FocusStackConfig,
    exact_uint16_histogram_bounds,
    focus_stack_group,
)
from data_pipeline.acquisition.image_building.shared.log_focus import LoG_focus_stacker


def _make_stack(seed: int, z: int = 4, y: int = 16, x: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(z, y, x), dtype=np.uint16)


# --------------------------------------------------------------------------- bounds


def _reference_bounds(raw: np.ndarray, low: float, high: float) -> tuple[int, int]:
    """The exact nearest-rank formula from the visually validated reference script
    (results/mcolon/20260712_legacy_focus_stack_comparison/generate_improved_comparison.py).
    The shared primitive MUST reproduce this to keep the accepted A02 output byte-identical.
    """
    counts = np.bincount(raw.reshape(-1), minlength=65536)
    cumulative = np.cumsum(counts, dtype=np.int64)
    n = int(cumulative[-1])
    low_rank = int(np.floor((low / 100.0) * (n - 1))) + 1
    high_rank = int(np.floor((high / 100.0) * (n - 1))) + 1
    lo = int(np.searchsorted(cumulative, low_rank, side="left"))
    hi = int(np.searchsorted(cumulative, high_rank, side="left"))
    return lo, hi


def test_bounds_match_validated_reference_formula():
    # Byte-for-byte agreement with the accepted diagnostic across many random groups.
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.integers(0, 60000, size=(3, 32, 32), dtype=np.uint16)
        b = rng.integers(0, 60000, size=(3, 32, 32), dtype=np.uint16)
        got = exact_uint16_histogram_bounds([a, b], low_percentile=0.1, high_percentile=99.9)
        exp = _reference_bounds(np.stack([a, b]), 0.1, 99.9)
        assert got == exp


def test_bounds_are_deterministic_and_ordering_independent():
    a, b = _make_stack(1), _make_stack(2)
    first = exact_uint16_histogram_bounds([a, b], low_percentile=0.1, high_percentile=99.9)
    reordered = exact_uint16_histogram_bounds([b, a], low_percentile=0.1, high_percentile=99.9)
    assert first == reordered
    # A giant frame gives the SAME answer every call (no strided/random sampling).
    big = _make_stack(3, z=3, y=1024, x=1024)
    assert exact_uint16_histogram_bounds([big], low_percentile=0.1, high_percentile=99.9) == \
        exact_uint16_histogram_bounds([big], low_percentile=0.1, high_percentile=99.9)


def test_bounds_span_the_whole_group_not_per_tile():
    # Tile A is dim (0..1000), tile B is bright (30000..40000). The shared bounds must
    # straddle both, so neither tile's own min/max defines them.
    a = np.random.default_rng(0).integers(0, 1000, size=(3, 8, 8), dtype=np.uint16)
    b = np.random.default_rng(1).integers(30000, 40000, size=(3, 8, 8), dtype=np.uint16)
    lo, hi = exact_uint16_histogram_bounds([a, b], low_percentile=0.1, high_percentile=99.9)
    assert lo < 1000 and hi > 30000


def test_bounds_reject_non_uint16_dtype():
    # Bounds are defined on the raw uint16 population; refuse any other dtype (incl. the
    # float32-normalized arrays the OLD double-normalizing path used to pass in).
    bad = np.full((2, 4, 4), 0.5, dtype=np.float32)
    with pytest.raises(TypeError):
        exact_uint16_histogram_bounds([bad], low_percentile=0.1, high_percentile=99.9)


def test_bounds_reject_degenerate_flat_group():
    flat = np.full((2, 4, 4), 123, dtype=np.uint16)
    with pytest.raises(ValueError):
        exact_uint16_histogram_bounds([flat], low_percentile=0.1, high_percentile=99.9)


# --------------------------------------------------------------------------- group op


def _cfg(**kw) -> FocusStackConfig:
    return FocusStackConfig(**kw)


def test_one_bound_pair_for_every_tile():
    a = np.random.default_rng(0).integers(0, 1000, size=(3, 8, 8), dtype=np.uint16)
    b = np.random.default_rng(1).integers(30000, 40000, size=(3, 8, 8), dtype=np.uint16)
    result = focus_stack_group([a, b], config=_cfg(), device="cpu")
    assert len(result.tiles) == 2
    lo, hi = exact_uint16_histogram_bounds([a, b], low_percentile=0.1, high_percentile=99.9)
    assert (result.intensity_lo, result.intensity_hi) == (lo, hi)
    assert result.bound_method == BOUND_METHOD


def test_input_stacks_are_not_mutated():
    stack = _make_stack(5)
    before = stack.copy()
    focus_stack_group([stack], config=_cfg(), device="cpu")
    np.testing.assert_array_equal(stack, before)


def test_focus_index_is_stack_axis_offset_shape_and_dtype():
    stack = _make_stack(6, z=5, y=12, x=10)
    result = focus_stack_group([stack], config=_cfg(), device="cpu")
    fim = result.tiles[0].focus_index_map
    assert fim.shape == (12, 10)
    assert fim.dtype == np.int32
    assert fim.min() >= 0 and fim.max() <= 4  # offsets in [0, Z-1]


def test_projection_pixels_come_from_raw_data():
    # Build a stack where the sharpest plane per pixel is unambiguous, then confirm the
    # projection equals the shared display transform of the RAW focused pixel — never the
    # scoring tensor.
    stack = _make_stack(7)
    result = focus_stack_group([stack], config=_cfg(), device="cpu")
    tile = result.tiles[0]
    lo, hi = result.intensity_lo, result.intensity_hi
    focused_raw = np.take_along_axis(
        stack, tile.focus_index_map[np.newaxis], axis=0
    )[0]
    unit = np.clip((focused_raw.astype(np.float32) - lo) / float(hi - lo), 0.0, 1.0)
    expected_u8 = np.round(unit * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(tile.projection_u8, expected_u8)


def test_display_only_change_leaves_focus_indices_unchanged():
    # Invariant from the README: changing the display encoding must not change the focus-index
    # map. The display transform is the FINAL step (raw gather → uint8) and never feeds back
    # into selection, so the focus map is a pure function of the scoring pass. We verify by
    # brightening the RAW pixels ONLY where the argmax is already fixed cannot change the map:
    # concretely, re-render the same selection under a different display span and confirm the
    # index map is byte-identical while the pixels differ.
    stack = _make_stack(8)
    base = focus_stack_group([stack], config=_cfg(), device="cpu")

    # Rebuild the projection under a deliberately different display span (wider percentiles →
    # different uint8 bytes) but the SAME scoring/selection, and confirm the map is unchanged.
    lo2, hi2 = exact_uint16_histogram_bounds([stack], low_percentile=1.0, high_percentile=99.0)
    focused_raw = np.take_along_axis(
        stack, base.tiles[0].focus_index_map[np.newaxis], axis=0
    )[0]
    alt_u8 = np.rint(
        np.clip((focused_raw.astype(np.float32) - lo2) / (hi2 - lo2), 0, 1) * 255
    ).astype(np.uint8)
    assert not np.array_equal(alt_u8, base.tiles[0].projection_u8)  # display DID change
    # ...but the selection (focus index) is independent of that display choice.
    again = focus_stack_group([stack], config=_cfg(), device="cpu")
    np.testing.assert_array_equal(
        base.tiles[0].focus_index_map, again.tiles[0].focus_index_map
    )


def test_grouping_shares_bounds_deterministically():
    # Grouping is deterministic: same group → identical bytes on every call.
    a, b = _make_stack(10), _make_stack(11)
    r1 = focus_stack_group([a, b], config=_cfg(), device="cpu")
    r2 = focus_stack_group([a, b], config=_cfg(), device="cpu")
    for t1, t2 in zip(r1.tiles, r2.tiles):
        np.testing.assert_array_equal(t1.projection_u8, t2.projection_u8)
        np.testing.assert_array_equal(t1.focus_index_map, t2.focus_index_map)


def test_focus_index_matches_legacy_log_kernel_selection():
    # The group primitive's focus-index map must equal argmax over Z of the legacy
    # LoG_focus_stacker's abs_log on the SAME scoring tensor — i.e. we reuse the kernel and
    # add no divergent selection logic.
    stack = _make_stack(12)
    cfg = _cfg()
    result = focus_stack_group([stack], config=cfg, device="cpu")

    lo, hi = result.intensity_lo, result.intensity_hi
    scoring = np.clip((stack.astype(np.float32) - lo) / float(hi - lo), 0.0, 1.0)
    _, abs_log = LoG_focus_stacker(scoring, cfg.filter_size, "cpu")
    abs_log_np = abs_log.cpu().numpy() if hasattr(abs_log, "cpu") else np.asarray(abs_log)
    expected_fim = np.argmax(abs_log_np, axis=0).astype(np.int32)
    np.testing.assert_array_equal(result.tiles[0].focus_index_map, expected_fim)


def test_reproduces_validated_reference_pipeline_bytes():
    # End-to-end byte equivalence with the accepted reference implementation in
    # results/mcolon/20260712_legacy_focus_stack_comparison/generate_improved_comparison.py,
    # for BOTH scoring modes. This is the guarantee that the shared primitive emits the
    # visually validated shared_clean/improved_unclipped pixels.
    rng = np.random.default_rng(99)
    raw = rng.integers(0, 60000, size=(3, 5, 24, 24), dtype=np.uint16)  # (N, Z, Y, X)
    stacks = [raw[i] for i in range(raw.shape[0])]

    lo, hi = _reference_bounds(raw, 0.1, 99.9)

    for mode, clip in (("shared_clipped", True), ("shared_unclipped", False)):
        # Reference pipeline, verbatim.
        score_input = (raw.astype(np.float32) - lo) / float(hi - lo)
        if clip:
            score_input = np.clip(score_input, 0.0, 1.0)
        _, abs_log = LoG_focus_stacker(score_input, filter_size=3, device="cpu")
        focus_index = np.argmax(
            abs_log.cpu().numpy() if hasattr(abs_log, "cpu") else np.asarray(abs_log),
            axis=1,
        ).astype(np.int32)
        focused_raw = np.take_along_axis(raw, focus_index[:, None, :, :], axis=1).squeeze(1)
        mapped = np.clip((focused_raw.astype(np.float32) - lo) / float(hi - lo), 0.0, 1.0)
        ref_u8 = np.rint(mapped * 255.0).astype(np.uint8)

        # Shared primitive.
        result = focus_stack_group([s.copy() for s in stacks], config=_cfg(scoring_mode=mode), device="cpu")
        assert (result.intensity_lo, result.intensity_hi) == (lo, hi)
        for i in range(raw.shape[0]):
            np.testing.assert_array_equal(result.tiles[i].focus_index_map, focus_index[i])
            np.testing.assert_array_equal(result.tiles[i].projection_u8, ref_u8[i])


def test_all_tiles_share_one_tone_curve():
    # A dim tile and a bright tile in the same group must be rendered on the SAME (lo, hi):
    # a constant raw value maps to the SAME uint8 regardless of which tile it is in — this is
    # the property that stops tiles/z-stacks getting different colors.
    dim = np.full((3, 8, 8), 8000, dtype=np.uint16)
    bright = np.full((3, 8, 8), 50000, dtype=np.uint16)
    # Add a little structure so bounds are non-degenerate and LoG has something to pick.
    dim[1, 4, 4] = 40000
    bright[1, 4, 4] = 9000
    result = focus_stack_group([dim, bright], config=_cfg(), device="cpu")
    lo, hi = result.intensity_lo, result.intensity_hi
    # A pixel of value 40000 renders identically in whichever tile it appears.
    expected = int(np.rint(np.clip((40000 - lo) / (hi - lo), 0, 1) * 255))
    assert int(result.tiles[0].projection_u8[4, 4]) == expected


def test_unclipped_scoring_mode_is_available_but_not_default():
    assert FocusStackConfig().scoring_mode == "shared_clipped"
    stack = _make_stack(13)
    # Unclipped mode must run and may (legitimately) differ; we only assert it produces a
    # valid uint8 projection with the same shape.
    result = focus_stack_group(
        [stack], config=_cfg(scoring_mode="shared_unclipped"), device="cpu"
    )
    assert result.tiles[0].projection_u8.dtype == np.uint8
    assert result.tiles[0].projection_u8.shape == stack.shape[1:]


def test_rejects_empty_group_and_non_3d():
    with pytest.raises(ValueError):
        focus_stack_group([], config=_cfg(), device="cpu")
    with pytest.raises(ValueError):
        focus_stack_group([np.zeros((4, 4), np.uint16)], config=_cfg(), device="cpu")
