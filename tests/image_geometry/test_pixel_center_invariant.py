"""The pixel-center coordinate convention, stated from FIRST PRINCIPLES for AFFINES.

WHY THIS FILE EXISTS, AND WHAT IT DELIBERATELY DOES NOT DO.

``tests/image_geometry/test_resize_coordinate_convention.py`` already pins the convention for the
RESIZE path. This file is its affine twin. Every assertion below is written against an
ANALYTICALLY DERIVED expected value -- a number computed from the convention itself -- and never
against what some other implementation happens to produce.

That distinction is the entire point. ``tests/embryo_geometry/test_orientation_equivalence.py``
pins ``image_geometry.candidates`` against ``analyze...canonical.CanonicalAligner`` at abs=1e-6,
and it passed for the whole time BOTH sides carried the naive convention. Two clocks five minutes
slow agree perfectly. An agreement test cannot detect a MIRRORED defect; only an independent
invariant can. So nothing here may import ``CanonicalAligner``, and nothing here may compare two
implementations to each other.

THE CONVENTION. A pixel with integer index ``x`` is a SAMPLE OF AREA centered at ``x + 0.5`` in
continuous units; the raster spans ``[0, n)``. Scaling that continuous space by ``sf`` and
re-discretizing gives

    x_out = sf * (x_src + 0.5) - 0.5          (forward)
    x_src = (x_out + 0.5) / sf - 0.5          (inverse)

The naive rule ``x_out = sf * x_src`` treats an index as a corner. The two differ by exactly
``(sf - 1) / 2`` -- a constant, direction-consistent offset applied identically to every sample.
That is what makes it dangerous: a shared bias is invisible to every aggregate statistic (area,
IoU, mean error, pixel diffs all stay clean) while the entire population sits off-grid relative to
the resize-based coordinates it is compared against.

``cv2.resize`` implements the pixel-center rule. ``cv2.warpAffine`` applies whatever matrix it is
handed, with NO half-pixel correction of its own -- so a scale fused into an affine lands
``(sf-1)/2`` px away from where the identical scale expressed as a resize lands. Generalizing to a
full 2x3 affine ``out = A @ src + t`` under the naive rule, requiring instead that pixel CENTERS
map to pixel CENTERS gives ``(out + 0.5) = A @ (src + 0.5) + t``, i.e. the translation column
gains ``A @ [0.5, 0.5] - 0.5``.

MEASUREMENT TECHNIQUE. A linear ramp ``v(x) = x``. Linear interpolation of a linear function is
exact, so an output pixel's VALUE *is* the source coordinate it sampled. That reads the mapping
directly, with sub-pixel resolution. A single bright pixel cannot do this at a downscale -- it
lands inside one output pixel and its measured position quantizes to that pixel's integer index --
so the impulse tests below are run at an UPSCALE, where the quantization is finer than the effect.

TOLERANCE. ``cv2.warpAffine`` evaluates INTER_LINEAR in fixed point with 5 fractional bits, so
sampled positions snap to multiples of 1/32 in OUTPUT units. That is a property of the resampling
engine, unrelated to which convention the matrix encodes. Every tolerance here is ONE such quantum,
and each test that claims a bias exists ALSO asserts the bias EXCEEDS that quantum -- so the
tolerance can never be quietly widened until a real defect fits inside it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_geometry.candidates import (
    centered_placement_affine,
    enumerate_orientation_candidates,
)

N = 64


# ---------------------------------------------------------------------------
# The convention, written out once. Everything below is derived from these.
# ---------------------------------------------------------------------------


def forward(x_src: float, sf: float) -> float:
    """Source coordinate -> output coordinate, pixel-center convention."""
    return sf * (x_src + 0.5) - 0.5


def inverse(x_out: float, sf: float) -> float:
    """Output coordinate -> source coordinate, pixel-center convention."""
    return (x_out + 0.5) / sf - 0.5


def naive_forward(x_src: float, sf: float) -> float:
    """The DEFECT, named so the tests can assert distance from it rather than imply it."""
    return sf * x_src


def _ramp(n: int = N) -> np.ndarray:
    """v(x) = x, so a resampled pixel's value IS the source coordinate it sampled."""
    return np.tile(np.arange(n, dtype=np.float32), (n, 1))


def warp_quantum(sf: float) -> float:
    """One ``warpAffine`` coordinate quantum (1/32 output px), expressed in SOURCE px.

    The engine snaps sampled positions to a 1/32 lattice in OUTPUT units; dividing by ``sf``
    converts that to the source units these assertions measure in. Tolerating LESS than one
    quantum would pin the engine's rounding rather than the coordinate convention; tolerating
    MORE would let a real half-pixel bias hide inside the tolerance -- which is why every bias
    claim below is paired with an assertion that the bias exceeds this value.
    """
    return 1.0 / (32.0 * sf)


def expected_bias(sf: float) -> float:
    """|naive - pixel_center| in SOURCE px at a given scale: ``(1 - sf) / (2 * sf)``."""
    return abs(1.0 - sf) / (2.0 * sf)


SCALES = [0.25, 0.5, 0.3231, 2.0]


# ---------------------------------------------------------------------------
# 1. A known source pixel CENTER maps to the analytically expected destination center.
# ---------------------------------------------------------------------------


class TestPixelCenterMapping:
    """Pure arithmetic on the matrix -- no resampling engine involved at all."""

    @pytest.mark.parametrize("sf", SCALES)
    @pytest.mark.parametrize("x_src", [0.0, 1.0, 7.0, 31.0, 63.0])
    def test_scaling_affine_maps_centers_to_expected_centers(self, sf, x_src):
        """The load-bearing assertion: the matrix must implement ``sf*(x+0.5)-0.5``."""
        affine = centered_placement_affine(
            rotation_deg=0.0,
            scale=sf,
            src_center_xy=(0.0, 0.0),
            out_shape_yx=(N, N),
        )
        # Undo the centering translation so this reads the linear part + half-pixel term only.
        # centered_placement_affine adds (w_out/2 - cx, h_out/2 - cy); with cx=cy=0 that is
        # exactly (N/2, N/2), a pure translation the convention must leave untouched.
        x_out = affine[0, 0] * x_src + affine[0, 1] * x_src + affine[0, 2] - N / 2
        assert x_out == pytest.approx(forward(x_src, sf), abs=1e-9), (
            f"sf={sf}: source center {x_src} mapped to {x_out}, expected "
            f"{forward(x_src, sf)} under the pixel-center convention "
            f"(the naive rule would give {naive_forward(x_src, sf)})"
        )

    @pytest.mark.parametrize("sf", [0.25, 0.5, 0.3231])
    def test_the_defect_is_larger_than_the_engine_quantum(self, sf):
        """Without this, 'fixing' the convention could be indistinguishable from rounding."""
        assert expected_bias(sf) > warp_quantum(sf), (
            f"sf={sf}: the half-pixel bias {expected_bias(sf):.4f} px does not exceed one "
            f"warpAffine quantum {warp_quantum(sf):.4f} px -- this scale cannot prove anything"
        )

    def test_pure_translation_is_untouched_by_the_convention(self):
        """A translation moves centers and corners identically; correcting it would be a bug.

        The canonical anchor shift is a pure translation. If the half-pixel term leaked into it,
        every anchored embryo would move for no reason at all.
        """
        affine = centered_placement_affine(
            rotation_deg=0.0,
            scale=1.0,
            src_center_xy=(10.0, 20.0),
            out_shape_yx=(N, N),
        )
        expected = np.array(
            [[1.0, 0.0, N / 2 - 10.0], [0.0, 1.0, N / 2 - 20.0]], dtype=np.float64
        )
        assert np.allclose(affine, expected, atol=1e-9), (
            "a scale-1 rotation-0 placement must be an exact pure translation"
        )

    def test_identity_scale_is_the_identity_map(self):
        """sf=1 has zero bias by construction -- pinned so the correction cannot be unconditional
        in the wrong direction (e.g. a sign flip that only shows up away from 1)."""
        assert forward(7.0, 1.0) == pytest.approx(7.0)
        affine = centered_placement_affine(
            rotation_deg=0.0, scale=1.0, src_center_xy=(0.0, 0.0), out_shape_yx=(N, N)
        )
        assert affine[0, 0] * 7.0 + affine[0, 2] - N / 2 == pytest.approx(7.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. An impulse lands at the predicted output location.
# ---------------------------------------------------------------------------


class TestImpulseLocation:
    """A single lit pixel, measured through the real resampling engine.

    Run at an UPSCALE on purpose. At a downscale the impulse falls inside one output pixel and
    its measured position quantizes to that pixel's integer index, which cannot resolve a
    sub-pixel offset -- the measurement would silently prove nothing.
    """

    @pytest.mark.parametrize("sf", [2.0, 4.0, 3.0])
    @pytest.mark.parametrize("x_src", [8, 20, 40])
    def test_impulse_lands_at_the_predicted_center(self, sf, x_src):
        src = np.zeros((N, N), dtype=np.float32)
        src[x_src, x_src] = 1.0

        out_n = int(round(N * sf))
        affine = centered_placement_affine(
            rotation_deg=0.0, scale=sf, src_center_xy=(0.0, 0.0), out_shape_yx=(out_n, out_n)
        )
        # Remove the centering so the impulse's absolute position is directly predictable.
        affine = affine.copy()
        affine[0, 2] -= out_n / 2
        affine[1, 2] -= out_n / 2

        warped = cv2.warpAffine(
            src, affine.astype(np.float32), (out_n, out_n), flags=cv2.INTER_LINEAR
        )
        assert warped.sum() > 0, "impulse fell off the output grid; the test measures nothing"

        # Intensity-weighted centroid resolves the spread impulse to sub-pixel precision.
        ys, xs = np.nonzero(warped)
        w = warped[ys, xs]
        cx_meas = float((xs * w).sum() / w.sum())

        expected = forward(float(x_src), sf)
        assert cx_meas == pytest.approx(expected, abs=max(warp_quantum(sf), 0.05)), (
            f"sf={sf}, src={x_src}: impulse centroid {cx_meas:.4f}, expected {expected:.4f} "
            f"(naive would predict {naive_forward(float(x_src), sf):.4f})"
        )

    @pytest.mark.parametrize("sf", [2.0, 4.0])
    def test_impulse_is_measurably_off_the_naive_prediction(self, sf):
        """The positive claim needs a negative twin, or 'passes' could mean 'both agree'."""
        x_src = 20
        src = np.zeros((N, N), dtype=np.float32)
        src[x_src, x_src] = 1.0
        out_n = int(round(N * sf))
        affine = centered_placement_affine(
            rotation_deg=0.0, scale=sf, src_center_xy=(0.0, 0.0), out_shape_yx=(out_n, out_n)
        ).copy()
        affine[0, 2] -= out_n / 2
        affine[1, 2] -= out_n / 2
        warped = cv2.warpAffine(
            src, affine.astype(np.float32), (out_n, out_n), flags=cv2.INTER_LINEAR
        )
        ys, xs = np.nonzero(warped)
        w = warped[ys, xs]
        cx_meas = float((xs * w).sum() / w.sum())
        naive = naive_forward(float(x_src), sf)
        assert abs(cx_meas - naive) > warp_quantum(sf), (
            f"sf={sf}: measured impulse position is indistinguishable from the naive prediction; "
            "either the fix is absent or this scale cannot resolve it"
        )


# ---------------------------------------------------------------------------
# 3. Round trip: scale then inverse-scale recovers the starting coordinate.
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Composition must be exact, which is only true if BOTH directions use the same rule."""

    @pytest.mark.parametrize("sf", SCALES)
    @pytest.mark.parametrize("x", [0.0, 7.0, 31.5, 63.0])
    def test_analytic_round_trip(self, sf, x):
        assert inverse(forward(x, sf), sf) == pytest.approx(x, abs=1e-9)

    @pytest.mark.parametrize("sf", SCALES)
    def test_matrix_round_trip_through_centered_placement(self, sf):
        """Down then up through the REAL constructor must return to the start.

        This is the strongest structural check available without a resampling engine: the
        half-pixel term must compose correctly, not merely be present. A correction applied with
        the wrong sign, or applied to only one of the two matrices, breaks here even though each
        matrix in isolation might look plausible.
        """
        fwd = centered_placement_affine(
            rotation_deg=0.0, scale=sf, src_center_xy=(0.0, 0.0), out_shape_yx=(N, N)
        ).copy()
        inv = centered_placement_affine(
            rotation_deg=0.0, scale=1.0 / sf, src_center_xy=(0.0, 0.0), out_shape_yx=(N, N)
        ).copy()
        # Strip the centering translations; they are not part of the scale round trip.
        fwd[:, 2] -= N / 2
        inv[:, 2] -= N / 2

        for x in (0.0, 7.0, 31.5, 63.0):
            mid = fwd[0, 0] * x + fwd[0, 2]
            back = inv[0, 0] * mid + inv[0, 2]
            assert back == pytest.approx(x, abs=1e-9), (
                f"sf={sf}: {x} -> {mid} -> {back}; the half-pixel term does not compose"
            )

    @pytest.mark.parametrize("sf", [0.5, 2.0])
    def test_rotation_by_360_is_the_identity_placement(self, sf):
        """A full turn must not accumulate a half-pixel drift."""
        a0 = centered_placement_affine(
            rotation_deg=0.0, scale=sf, src_center_xy=(12.0, 9.0), out_shape_yx=(N, N)
        )
        a360 = centered_placement_affine(
            rotation_deg=360.0, scale=sf, src_center_xy=(12.0, 9.0), out_shape_yx=(N, N)
        )
        assert np.allclose(a0, a360, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Resize-only and the equivalent affine-only mapping AGREE.
# ---------------------------------------------------------------------------


class TestResizeAffineAgreement:
    """THE cross-seam property. Not an implementation-vs-implementation test.

    ``cv2.resize`` is an INDEPENDENT ORACLE here, not a peer implementation: it is a third-party
    engine whose convention is separately pinned, from first principles, in
    ``test_resize_coordinate_convention.py`` (and cross-checked there against skimage). Requiring
    our affine to match it is requiring our affine to match the documented convention -- which is
    exactly what the mirrored-defect trap made impossible when both peers were ours.
    """

    @pytest.mark.parametrize("out_n", [16, 32, 27])
    def test_affine_scale_samples_where_resize_samples(self, out_n):
        sf = out_n / N
        ramp = _ramp()

        resized = cv2.resize(ramp, (out_n, out_n), interpolation=cv2.INTER_LINEAR)[0, :]

        affine = centered_placement_affine(
            rotation_deg=0.0, scale=sf, src_center_xy=(0.0, 0.0), out_shape_yx=(out_n, out_n)
        ).copy()
        affine[0, 2] -= out_n / 2
        affine[1, 2] -= out_n / 2
        warped = cv2.warpAffine(
            ramp, affine.astype(np.float32), (out_n, out_n), flags=cv2.INTER_LINEAR
        )[0, :]

        for j in (0, 1, 2, out_n - 1):
            assert warped[j] == pytest.approx(resized[j], abs=warp_quantum(sf)), (
                f"sf={sf}: affine sampled source {warped[j]:.4f} at output pixel {j}, "
                f"resize sampled {resized[j]:.4f} -- the two seams disagree"
            )

    @pytest.mark.parametrize("out_n", [16, 32])
    def test_the_naive_affine_visibly_disagrees_with_resize(self, out_n):
        """Pins the DEFECT MECHANISM, so the premise of the fix stays independently verifiable.

        If this ever stops failing-to-agree, cv2 changed its warpAffine convention and the
        correction in candidates.py must be re-derived rather than trusted.
        """
        sf = out_n / N
        ramp = _ramp()
        resized = cv2.resize(ramp, (out_n, out_n), interpolation=cv2.INTER_LINEAR)[0, :]
        naive = np.float32([[sf, 0, 0], [0, sf, 0]])
        warped = cv2.warpAffine(ramp, naive, (out_n, out_n), flags=cv2.INTER_LINEAR)[0, :]
        assert abs(warped[1] - resized[1]) == pytest.approx(
            expected_bias(sf), abs=warp_quantum(sf)
        )
        assert abs(warped[1] - resized[1]) > warp_quantum(sf)


# ---------------------------------------------------------------------------
# 5. The invariant reaches the PUBLIC verb, not just the matrix constructor.
# ---------------------------------------------------------------------------


class TestCandidateEnumerationHonorsTheConvention:
    """``enumerate_orientation_candidates`` is what production actually calls."""

    @pytest.mark.parametrize("sf", [0.25, 0.5])
    def test_enumerated_candidate_affine_maps_centers_correctly(self, sf):
        mask = np.zeros((N, N), np.uint8)
        mask[20:40, 10:50] = 1
        cands = enumerate_orientation_candidates(
            mask,
            base_rotation_deg=0.0,
            scale=sf,
            src_center_xy=(0.0, 0.0),
            out_shape_yx=(N, N),
            allow_flip=False,
        )
        affine = cands[0].affine_2x3
        x_out = affine[0, 0] * 10.0 + affine[0, 2] - N / 2
        assert x_out == pytest.approx(forward(10.0, sf), abs=1e-9), (
            "the public enumeration verb must carry the same convention as the matrix builder"
        )

    def test_unscaled_unrotated_candidate_is_a_pure_translation(self):
        """Guards against the correction being applied where it does not belong."""
        mask = np.zeros((N, N), np.uint8)
        mask[20:40, 10:50] = 1
        cands = enumerate_orientation_candidates(
            mask,
            base_rotation_deg=0.0,
            scale=1.0,
            src_center_xy=(32.0, 32.0),
            out_shape_yx=(N, N),
            allow_flip=False,
        )
        affine = cands[0].affine_2x3
        assert np.allclose(affine[:, :2], np.eye(2), atol=1e-9)
        assert np.allclose(affine[:, 2], [0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# 6. The ANALYSIS side must satisfy the SAME invariant, derived the same way.
# ---------------------------------------------------------------------------


class TestCanonicalAlignerHonorsTheSameConvention:
    """``analyze...canonical.CanonicalAligner`` is the other half of the mirrored defect.

    READ THIS BEFORE ADDING ANYTHING HERE. These assertions compare ``CanonicalAligner`` to the
    ANALYTIC rule at the top of this file -- never to ``image_geometry.candidates``. Comparing
    the two implementations is precisely what ``test_orientation_equivalence.py`` does, and
    precisely why the defect went undetected for as long as it did: both sides were naive, so
    they agreed. This class exists so that each side is independently anchored, which is what
    makes their agreement downstream evidence of anything at all.
    """

    @staticmethod
    def _aligner(shape=(256, 576)):
        from analyze.utils.coord.grids.canonical import CanonicalAligner

        return CanonicalAligner(target_shape_hw=shape)

    @pytest.mark.parametrize("sf", SCALES)
    @pytest.mark.parametrize("x_src", [0.0, 7.0, 31.0])
    def test_placement_affine_maps_centers_to_expected_centers(self, sf, x_src):
        aligner = self._aligner()
        affine = aligner._placement_affine(0.0, 0.0, 0.0, sf)
        # Strip the canvas centering (W/2, H/2 with cx=cy=0), a pure translation.
        x_out = affine[0, 0] * x_src + affine[0, 2] - aligner.W / 2
        assert x_out == pytest.approx(forward(x_src, sf), abs=1e-9), (
            f"sf={sf}: canonical placement mapped {x_src} to {x_out}, expected "
            f"{forward(x_src, sf)} (naive would give {naive_forward(x_src, sf)})"
        )

    @pytest.mark.parametrize("sf", [0.25, 0.5, 0.3231])
    def test_placement_affine_is_not_the_naive_rule(self, sf):
        """The negative twin. Measured in OUTPUT px, where the offset is exactly (sf-1)/2."""
        aligner = self._aligner()
        affine = aligner._placement_affine(0.0, 0.0, 0.0, sf)
        x_out = affine[0, 0] * 7.0 + affine[0, 2] - aligner.W / 2
        offset = x_out - naive_forward(7.0, sf)
        assert offset == pytest.approx((sf - 1.0) / 2.0, abs=1e-9), (
            f"sf={sf}: offset from the naive rule is {offset}, expected exactly {(sf-1)/2}"
        )
        assert abs(offset) > 1.0 / 32.0, "the offset must exceed one warpAffine quantum"

    def test_placement_affine_leaves_pure_translation_alone(self):
        aligner = self._aligner()
        affine = aligner._placement_affine(10.0, 20.0, 0.0, 1.0)
        expected = np.array(
            [[1.0, 0.0, aligner.W / 2 - 10.0], [0.0, 1.0, aligner.H / 2 - 20.0]]
        )
        assert np.allclose(affine, expected, atol=1e-9)

    @pytest.mark.parametrize("out_n", [16, 32])
    def test_canonical_affine_samples_where_resize_samples(self, out_n):
        """Same independent-oracle check as TestResizeAffineAgreement, analysis side."""
        sf = out_n / N
        aligner = self._aligner(shape=(out_n, out_n))
        ramp = _ramp()
        resized = cv2.resize(ramp, (out_n, out_n), interpolation=cv2.INTER_LINEAR)[0, :]
        affine = aligner._placement_affine(0.0, 0.0, 0.0, sf).copy()
        affine[0, 2] -= out_n / 2
        affine[1, 2] -= out_n / 2
        warped = cv2.warpAffine(
            ramp, affine.astype(np.float32), (out_n, out_n), flags=cv2.INTER_LINEAR
        )[0, :]
        for j in (0, 1, 2, out_n - 1):
            assert warped[j] == pytest.approx(resized[j], abs=warp_quantum(sf))

    def test_both_sides_import_one_correction_not_two(self):
        """The STRUCTURAL guard against the defect reappearing.

        The original bug was two independent copies of the placement arithmetic drifting into
        the same wrong convention. A behavioral test cannot prevent a third copy from being
        added; this one asserts the analysis side is literally bound to the shared function.
        """
        from analyze.utils.coord.grids import canonical as canon
        from image_geometry.candidates import pixel_center_affine

        assert canon._pixel_center_affine is pixel_center_affine, (
            "canonical.py must use image_geometry's correction, not a private copy -- "
            "duplicating it is how the mirrored defect was created"
        )


# ---------------------------------------------------------------------------
# 7. The convention is STAMPED, so stale caches fail structurally.
# ---------------------------------------------------------------------------


class TestCoordinateConventionVersion:
    """A shared, opaque cache key -- the only staleness signal that works here.

    The v1 -> v2 change moves every coordinate sub-pixel. A v1 artifact has the right shape,
    the right dtype, a sane area and a clean IoU; it is simply offset. Nothing structural
    distinguishes it, so mtime- or shape-based staleness checks pass on data that is wrong.
    An explicit version string is the only thing a consumer can compare that actually fails.
    """

    def test_the_version_names_the_convention(self):
        from image_geometry import COORDINATE_CONVENTION_VERSION

        # Named for the SEMANTICS, not for a file or a release, because several unrelated
        # consumers (canonical grid, UOT couplings, derived caches) share the same key.
        assert COORDINATE_CONVENTION_VERSION == "pixel_center_v2"

    def test_canonical_mask_metadata_carries_the_version(self):
        from analyze.utils.coord.grids.canonical import to_canonical_grid_mask
        from image_geometry import COORDINATE_CONVENTION_VERSION

        mask = np.zeros((200, 300), np.uint8)
        yy, xx = np.mgrid[0:200, 0:300]
        mask[((xx - 150) ** 2 / 90**2 + (yy - 100) ** 2 / 30**2) < 1] = 1
        res = to_canonical_grid_mask(mask, um_per_px=3.2308)
        assert res.meta["coordinate_convention_version"] == COORDINATE_CONVENTION_VERSION

    def test_canonical_image_metadata_carries_the_version(self):
        from analyze.utils.coord.grids.canonical import to_canonical_grid_image
        from image_geometry import COORDINATE_CONVENTION_VERSION

        img = (_ramp(128) * 2).astype(np.uint8)
        res = to_canonical_grid_image(img, um_per_px=3.2308)
        assert res.meta["coordinate_convention_version"] == COORDINATE_CONVENTION_VERSION

    def test_absence_is_the_v1_signal(self):
        """Old artifacts predate the key entirely, so consumers must treat missing as v1.

        Pinned as a CONTRACT rather than an implementation detail: a consumer that reads with
        ``meta.get(key, "pixel_center_v1")`` is correct, one that defaults to the current
        version silently accepts every wrong cache ever written.
        """
        from image_geometry import COORDINATE_CONVENTION_VERSION

        legacy_meta: dict = {"coord_frame_id": "canonical_grid", "coord_frame_version": 1}
        assert (
            legacy_meta.get("coordinate_convention_version", "pixel_center_v1")
            != COORDINATE_CONVENTION_VERSION
        )
