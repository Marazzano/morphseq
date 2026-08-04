"""The two geometry conventions the canonical grid must honor.

These are the ANALYSIS-side twins of ``tests/image_geometry/test_resize_coordinate_convention.py``.
Both defects pinned here were found and fixed on the pipeline side first; this file exists so the
analysis side cannot silently drift back.

DEFECT 1 — half-pixel bias. ``cv2.warpAffine`` implements the naive ``x_out = sf * x_src``;
``cv2.resize`` implements the pixel-center ``x_out = sf * (x_src + 0.5) - 0.5``. A scale fused
into an affine therefore lands ``(sf-1)/2`` px away from where the same scale expressed as a
resize would land. Every embryo shifts by the SAME amount, so no aggregate statistic can see it.

DEFECT 2 — un-anti-aliased downscale. ``cv2.warpAffine`` silently ignores ``INTER_AREA``, so a
fused affine containing a downscale physically cannot prefilter. A downscale must be its own
``cv2.resize`` step.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from analyze.utils.coord.grids.canonical import (
    CanonicalGridConfig,
    _pixel_center_affine,
    _prescale_for_downscale,
    to_canonical_grid_mask,
)

N = 64


def _ramp(n: int = N) -> np.ndarray:
    """v(x) = x, so a resampled pixel's value IS the source coordinate it sampled."""
    return np.tile(np.arange(n, dtype=np.float32), (n, 1))


def _pixel_center_inverse(x_out: float, sf: float) -> float:
    return (x_out + 0.5) / sf - 0.5


def _warp_tol(sf: float) -> float:
    """One ``warpAffine`` coordinate quantum, expressed in SOURCE pixels.

    ``cv2.warpAffine`` evaluates INTER_LINEAR in fixed point with 5 fractional bits, so sampled
    positions snap to multiples of 1/32 in OUTPUT units — an engine property, unrelated to which
    coordinate convention the affine encodes. At sf=0.25/0.5 the exact answer already lies on that
    lattice and the match is exact; at an arbitrary scale like 0.3231 it cannot be. Dividing by sf
    converts the quantum to source units, which is what these assertions measure. Tolerating less
    than one quantum would pin the engine's rounding rather than the convention.
    """
    return 1.0 / (32.0 * sf)


class TestHalfPixelConvention:
    """Defect 1: an affine-fused scale must sample where a resize would sample."""

    @pytest.mark.parametrize("sf", [0.25, 0.5, 0.3231])
    def test_raw_warp_affine_uses_the_naive_rule(self, sf):
        # This is the DEFECT MECHANISM, pinned so the premise of the fix stays verifiable.
        out_n = int(round(N * sf))
        M = np.float32([[sf, 0, 0], [0, sf, 0]])
        sampled = cv2.warpAffine(_ramp(), M, (out_n, out_n), flags=cv2.INTER_LINEAR)[0, :]
        # Naive: output pixel j sampled source j/sf.
        assert sampled[1] == pytest.approx(1.0 / sf, abs=_warp_tol(sf))
        # ...which is NOT the pixel-center answer. The gap is the systematic bias (1-sf)/(2*sf).
        assert abs(sampled[1] - _pixel_center_inverse(1, sf)) == pytest.approx(
            (1 - sf) / (2 * sf), abs=_warp_tol(sf)
        )
        # And the bias must exceed the engine's own rounding, or there would be nothing to fix.
        assert (1 - sf) / (2 * sf) > _warp_tol(sf)

    @pytest.mark.parametrize("sf", [0.25, 0.5, 0.3231])
    def test_corrected_affine_matches_the_resize_convention(self, sf):
        out_n = int(round(N * sf))
        M = _pixel_center_affine(np.float32([[sf, 0, 0], [0, sf, 0]]))
        sampled = cv2.warpAffine(
            _ramp(), M.astype(np.float32), (out_n, out_n), flags=cv2.INTER_LINEAR
        )[0, :]
        for j in (0, 1, 2):
            assert sampled[j] == pytest.approx(_pixel_center_inverse(j, sf), abs=_warp_tol(sf))

    def test_correction_is_identity_for_pure_translation(self):
        # The anchor shift is a pure translation; the correction must not perturb it, or every
        # anchored embryo would move for no reason.
        M = np.float32([[1, 0, 12.5], [0, 1, -3.25]])
        assert np.allclose(_pixel_center_affine(M), M)

    def test_correction_is_nonzero_for_a_scaling_affine(self):
        # Regression guard: a "simplification" that gates the correction on some scale test, or
        # drops it entirely, must fail here.
        M = np.float32([[0.3231, 0, 5.0], [0, 0.3231, 5.0]])
        assert not np.allclose(_pixel_center_affine(M), M)

    def test_correction_round_trips_against_the_forward_rule(self):
        for sf in (0.25, 0.5, 0.3231, 2.0):
            M = _pixel_center_affine(np.array([[sf, 0.0, 0.0], [0.0, sf, 0.0]]))
            for x in (0.0, 7.0, 31.5):
                # forward: x_out = sf*(x_src + 0.5) - 0.5
                assert M[0, 0] * x + M[0, 2] == pytest.approx(sf * (x + 0.5) - 0.5, abs=1e-9)


class TestAntiAliasedDownscale:
    """Defect 2: a downscale must prefilter, which a fused affine cannot do."""

    def test_warp_affine_silently_ignores_inter_area(self):
        # THE mechanism. cv2 accepts the flag and does not area-average — pinned because the whole
        # justification for splitting the resize out of the affine rests on it.
        rng = np.random.default_rng(0)
        tex = rng.random((N, N)).astype(np.float32) * 255
        sf = 0.25
        out_n = int(N * sf)
        M = np.float32([[sf, 0, 0], [0, sf, 0]])
        warped_area = cv2.warpAffine(tex, M, (out_n, out_n), flags=cv2.INTER_AREA)
        warped_nearest = cv2.warpAffine(tex, M, (out_n, out_n), flags=cv2.INTER_NEAREST)
        # INTER_AREA was ignored: it produced the un-prefiltered result.
        assert np.allclose(warped_area, warped_nearest)
        # A real INTER_AREA resize is materially different.
        resized = cv2.resize(tex, (out_n, out_n), interpolation=cv2.INTER_AREA)
        assert not np.allclose(resized, warped_area, atol=1.0)

    def test_prescale_downscale_is_area_filtered_for_images(self):
        rng = np.random.default_rng(1)
        tex = (rng.random((N, N)) * 255).astype(np.float32)
        out, residual = _prescale_for_downscale(tex, 0.25, is_mask=False)
        assert out.shape == (16, 16)
        expected = cv2.resize(tex, (16, 16), interpolation=cv2.INTER_AREA)
        assert np.allclose(out, expected)
        # The residual scale carries whatever the integer output dims could not express.
        assert residual == pytest.approx(1.0, abs=1e-6)

    def test_prescale_uses_nearest_for_masks(self):
        # A categorical raster must never be blended into fractional values.
        m = np.zeros((N, N), np.float32)
        m[16:48, 16:48] = 1.0
        out, _ = _prescale_for_downscale(m, 0.25, is_mask=True)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_prescale_is_a_noop_when_upscaling(self):
        tex = _ramp()
        out, residual = _prescale_for_downscale(tex, 2.0, is_mask=False)
        assert out is tex
        assert residual == 2.0

    def test_prescale_residual_composes_back_to_the_requested_scale(self):
        # The realized ratio is out_n/in_n from the INTEGER dims, never the requested factor.
        # residual * realized must reconstruct the request, or the embryo lands at the wrong size.
        for sf in (0.25, 0.3231, 0.4142, 0.5):
            tex = _ramp()
            out, residual = _prescale_for_downscale(tex, sf, is_mask=False)
            realized = out.shape[0] / N
            assert residual * realized == pytest.approx(sf, abs=1e-9)

    def test_prescale_realized_scale_is_not_the_requested_scale(self):
        # Pinned because using the requested factor to map coordinates reintroduces exactly the
        # systematic sub-pixel bias that TestHalfPixelConvention exists to prevent.
        requested = 0.4142
        out, residual = _prescale_for_downscale(_ramp(), requested, is_mask=False)
        realized = out.shape[0] / N
        assert realized != pytest.approx(requested, abs=1e-3)
        assert residual != pytest.approx(1.0, abs=1e-3)


class TestEndToEndCanonicalization:
    """The convenience verb still produces a valid, well-placed mask after both fixes."""

    @staticmethod
    def _embryo(h=700, w=900):
        yy, xx = np.mgrid[0:h, 0:w]
        t = np.linspace(0, 1, 400)
        mask = np.zeros((h, w), bool)
        for cx, cy, rad in zip(150 + t * 600, 350 + 120 * np.sin(t * np.pi), 55 - 25 * t):
            mask |= ((xx - cx) ** 2 + (yy - cy) ** 2) < rad ** 2
        yolk = (((xx - 200) ** 2 + (yy - 380) ** 2) < 70 ** 2) & mask
        return mask.astype(np.uint8), yolk.astype(np.uint8)

    def test_canonicalization_produces_a_nonempty_interior_mask(self):
        mask, yolk = self._embryo()
        res = to_canonical_grid_mask(mask, um_per_px=3.2308, yolk_mask=yolk)
        out = res.mask
        assert out.shape == (256, 576)
        assert out.sum() > 0
        # _validate_output_mask forbids touching the border; assert the postcondition holds.
        assert not out[0, :].any() and not out[-1, :].any()
        assert not out[:, 0].any() and not out[:, -1].any()

    def test_anti_aliasing_preserves_area_better_than_the_fused_affine(self):
        # The physical claim: prefiltering a 3x downscale retains thin structure that nearest
        # sampling drops. Compare retained_ratio against the ideal scale^2 area.
        mask, yolk = self._embryo()
        res = to_canonical_grid_mask(mask, um_per_px=3.2308, yolk_mask=yolk)
        retained = res.meta["align_meta"]["retained_ratio"]
        assert retained == pytest.approx(1.0, abs=0.10), (
            f"retained_ratio={retained} — anti-aliased downscale should track expected area"
        )

    def test_um_per_px_scaling_is_honored(self):
        # Halving source um/px halves the physical size per pixel, so the canonical footprint
        # must shrink roughly 4x in area. Guards against the residual-scale composition breaking.
        mask, yolk = self._embryo()
        big = to_canonical_grid_mask(mask, um_per_px=3.2308, yolk_mask=yolk).mask.sum()
        small = to_canonical_grid_mask(mask, um_per_px=1.6154, yolk_mask=yolk).mask.sum()
        assert big / small == pytest.approx(4.0, rel=0.15)


class TestBboxRemainsInclusive:
    """``_bbox`` returns INCLUSIVE maxima and feeds shift-clamp arithmetic. Deliberate."""

    def test_bbox_maxima_are_inclusive(self):
        from analyze.utils.coord.grids.canonical import CanonicalAligner

        aligner = CanonicalAligner()
        m = np.zeros((32, 32), np.float32)
        m[5:10, 7:12] = 1.0
        min_y, max_y, min_x, max_x = aligner._bbox(m)
        # Inclusive: last SET row/col, not the half-open end. Converting to half-open would move
        # every clamp bound in _apply_anchor_shift by one pixel.
        assert (min_y, max_y, min_x, max_x) == (5, 9, 7, 11)
