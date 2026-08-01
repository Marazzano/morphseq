"""Native-grid intensity evidence.

The load-bearing tests here are the two the design turns on: that histograms pool EXACTLY by
summation (without which a pooled well-level null is not computable from per-embryo rows), and that
the neighbor prefilter catches a fish which does not overlap the target bbox but does overlap its
annulus (the case an unexpanded box misses -- precisely the contaminating one).
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.object_extraction.channel_intensity import (
    HIST_BIN_WIDTH_DN,
    HIST_N_BINS,
    ChannelIntensityError,
    build_annulus,
    expected_annulus_area_px,
    extract_embryo_intensity_evidence,
    histogram_region,
    neighbor_union_mask,
)
from image_geometry import BoxYX

SHAPE = (128, 128)
UM = (2.0, 2.0)
DTYPE_MAX = 65535


def _disc(cy: int, cx: int, radius_px: int, shape=SHAPE) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px**2


def _row(mask_id: str, mask: np.ndarray) -> dict:
    box = BoxYX.from_mask(mask)
    return {
        "mask_id": mask_id,
        "bbox_y_min_px": box.y0,
        "bbox_y_max_px": box.y1,
        "bbox_x_min_px": box.x0,
        "bbox_x_max_px": box.x1,
        "_mask": mask,
    }


def _decode(row):
    return row["_mask"]


class TestHistogramsPoolExactly:
    """The property the whole pooled-null design rests on."""

    def test_summing_two_histograms_equals_histogramming_the_union(self):
        rng = np.random.RandomState(0)
        a = rng.randint(0, 4096, size=500).astype(np.uint16)
        b = rng.randint(0, 4096, size=300).astype(np.uint16)

        pooled = histogram_region(a, dtype_max=DTYPE_MAX).hist_counts.astype(np.int64)
        pooled = pooled + histogram_region(b, dtype_max=DTYPE_MAX).hist_counts.astype(np.int64)
        together = histogram_region(
            np.concatenate([a, b]), dtype_max=DTYPE_MAX
        ).hist_counts.astype(np.int64)

        np.testing.assert_array_equal(pooled, together)

    def test_sums_are_exact_not_binned(self):
        # sum/sumsq ride alongside because they pool by summation too and give EXACT moments, which
        # the binned counts only approximate. If they were derived from bin centers this would fail.
        values = np.array([1, 2, 3, 4000, 65535], dtype=np.uint16)
        ev = histogram_region(values, dtype_max=DTYPE_MAX)
        assert ev.sum_dn == pytest.approx(float(values.astype(np.float64).sum()))
        assert ev.sumsq_dn == pytest.approx(float(np.square(values.astype(np.float64)).sum()))

    def test_saturated_pixels_are_counted_not_dropped(self):
        # A 2-copy embryo is the most likely to saturate, and saturation compresses 2-copy toward
        # 1-copy -- destroying the dosage comparison while looking like clean data.
        values = np.array([10, 20, DTYPE_MAX, DTYPE_MAX], dtype=np.uint16)
        ev = histogram_region(values, dtype_max=DTYPE_MAX)
        assert ev.clipped_px == 2
        assert ev.n_px == 4
        assert ev.hist_counts.sum() == 4, "a saturated pixel must still be binned, not discarded"

    def test_empty_region_is_zero_not_an_error(self):
        ev = histogram_region(np.array([], dtype=np.uint16), dtype_max=DTYPE_MAX)
        assert ev.n_px == 0 and ev.hist_counts.sum() == 0

    def test_bin_index_is_exact_for_the_declared_width(self):
        values = np.array([0, HIST_BIN_WIDTH_DN - 1, HIST_BIN_WIDTH_DN], dtype=np.uint16)
        counts = histogram_region(values, dtype_max=DTYPE_MAX).hist_counts
        assert counts[0] == 2 and counts[1] == 1


class TestAnnulusIsPhysical:
    def test_membership_is_tested_in_micrometers_not_pixels(self):
        # On an ANISOTROPIC grid a physical circle is a pixel-space ellipse -- the correct shape. A
        # circle in pixels would be an ellipse in the specimen.
        mask = _disc(64, 64, 6)
        anisotropic = build_annulus(
            mask, inner_radius_um=10.0, outer_radius_um=20.0, um_per_px_yx=(2.0, 4.0)
        )
        ys, xs = np.where(anisotropic)
        # Half-width in x should be about half the half-height in y, since x pixels are 2x wider.
        assert (xs.max() - xs.min()) < (ys.max() - ys.min())

    def test_inner_radius_is_a_gap_that_excludes_the_embryo(self):
        mask = _disc(64, 64, 6)
        annulus = build_annulus(
            mask, inner_radius_um=8.0, outer_radius_um=16.0, um_per_px_yx=UM
        )
        assert not (annulus & mask).any(), "the ring must not overlap the embryo it rings"

    def test_expected_area_is_rasterized_not_analytic(self):
        # An analytic pi(ro^2-ri^2) disagrees with any rasterization at small radii, so a fraction
        # built from it would drift with radius in a way that looks like data.
        expected = expected_annulus_area_px(
            inner_radius_um=2.0, outer_radius_um=6.0, um_per_px_yx=UM
        )
        analytic = np.pi * (6.0**2 - 2.0**2) / (UM[0] * UM[1])
        assert expected != pytest.approx(analytic, rel=0.01)

    def test_degenerate_radii_fail_loud(self):
        with pytest.raises(ChannelIntensityError, match="outer > inner"):
            build_annulus(_disc(64, 64, 6), inner_radius_um=9.0, outer_radius_um=9.0,
                          um_per_px_yx=UM)

    def test_empty_mask_fails_loud(self):
        with pytest.raises(ChannelIntensityError, match="empty"):
            build_annulus(np.zeros(SHAPE, bool), inner_radius_um=2.0, outer_radius_um=6.0,
                          um_per_px_yx=UM)


class TestCloseFishExclusion:
    def test_neighbor_outside_the_bbox_but_inside_the_annulus_is_excluded(self):
        # THE CASE THE UNEXPANDED PREFILTER MISSES, and it is exactly the contaminating one: a fish
        # whose bbox does not touch the target's, but whose body sits in the target's background.
        target = _disc(64, 40, 6)
        neighbor = _disc(64, 70, 6)
        target_box = BoxYX.from_mask(target)
        neighbor_box = BoxYX.from_mask(neighbor)
        assert not target_box.intersects(neighbor_box), "fixture must have disjoint bboxes"

        annulus = build_annulus(
            target, inner_radius_um=4.0, outer_radius_um=40.0, um_per_px_yx=UM
        )
        assert (annulus & neighbor).any(), "fixture must put the neighbor inside the annulus"

        exclusion, n = neighbor_union_mask(
            target_mask_id="t",
            target_box=target_box,
            neighbors=[_row("t", target), _row("n", neighbor)],
            decode=_decode,
            shape_yx=SHAPE,
            outer_radius_um=40.0,
            exclude_radius_um=4.0,
            um_per_px_yx=UM,
        )
        assert n == 1, "the neighbor must survive the expanded-bbox prefilter"
        assert not (annulus & ~exclusion & neighbor).any()

    def test_the_target_is_never_subtracted_from_its_own_background(self):
        # Excluding by row position rather than mask_id produces a confidently invalid row that
        # looks entirely well-formed.
        target = _disc(64, 64, 6)
        exclusion, n = neighbor_union_mask(
            target_mask_id="t",
            target_box=BoxYX.from_mask(target),
            neighbors=[_row("t", target)],
            decode=_decode,
            shape_yx=SHAPE,
            outer_radius_um=20.0,
            exclude_radius_um=4.0,
            um_per_px_yx=UM,
        )
        assert n == 0 and not exclusion.any()

    def test_invalid_masks_still_exclude_because_they_still_emit_photons(self):
        # is_valid_mask is a tracking/QC judgement, not a photometric one. Filtering on it here
        # would admit contamination from exactly the embryos QC flagged as problematic.
        target = _disc(64, 40, 6)
        neighbor = _row("n", _disc(64, 70, 6))
        neighbor["is_valid_mask"] = False
        _exclusion, n = neighbor_union_mask(
            target_mask_id="t",
            target_box=BoxYX.from_mask(target),
            neighbors=[_row("t", target), neighbor],
            decode=_decode,
            shape_yx=SHAPE,
            outer_radius_um=40.0,
            exclude_radius_um=4.0,
            um_per_px_yx=UM,
        )
        assert n == 1

    def test_a_distant_neighbor_is_prefiltered_away(self):
        target = _disc(20, 20, 6)
        far = _disc(110, 110, 6)
        _exclusion, n = neighbor_union_mask(
            target_mask_id="t",
            target_box=BoxYX.from_mask(target),
            neighbors=[_row("t", target), _row("far", far)],
            decode=_decode,
            shape_yx=SHAPE,
            outer_radius_um=10.0,
            exclude_radius_um=4.0,
            um_per_px_yx=UM,
        )
        assert n == 0, "prefilter must not decode masks that cannot touch the annulus"


class TestEvidenceRow:
    # outer=40um at 2um/px reaches 20px past the mask edge. The neighbor fixture below sits 17px
    # away, so it is genuinely inside the ring -- at outer=30um it would be 15px of reach against a
    # 17px gap, and the prefilter would CORRECTLY reject it.
    def _extract(self, image, target, neighbors, outer_radius_um=40.0):
        return extract_embryo_intensity_evidence(
            image=image,
            target_mask=target,
            target_mask_id="t",
            neighbors=neighbors,
            decode=_decode,
            um_per_px_yx=UM,
            inner_radius_um=4.0,
            outer_radius_um=outer_radius_um,
            exclude_radius_um=4.0,
            dtype_max=DTYPE_MAX,
        )

    def test_uint16_signal_survives(self):
        # The whole point: a 12000-DN embryo must read as 12000, not be compressed into 8 bits.
        target = _disc(64, 40, 6)
        image = np.full(SHAPE, 600, dtype=np.uint16)
        image[target] = 12000
        row = self._extract(image, target, [_row("t", target)])
        assert row["embryo_sum_dn"] == pytest.approx(12000.0 * int(target.sum()))

    def test_exclusion_is_reported_quantitatively(self):
        target = _disc(64, 40, 6)
        neighbor = _disc(64, 70, 6)
        image = np.full(SHAPE, 600, dtype=np.uint16)
        clean = self._extract(image, target, [_row("t", target)])
        crowded = self._extract(image, target, [_row("t", target), _row("n", neighbor)])
        assert crowded["annulus_excluded_px"] > clean["annulus_excluded_px"]
        assert crowded["annulus_area_fraction"] < clean["annulus_area_fraction"]
        assert crowded["annulus_neighbor_count"] == 1

    def test_a_neighbor_just_out_of_reach_is_correctly_ignored(self):
        # The complement of the case above, and the reason the prefilter pads by the OUTER RADIUS
        # rather than something generous: at outer=30um the ring reaches 15px, the neighbor sits
        # 17px away, and excluding it would shrink the background for no physical reason.
        target = _disc(64, 40, 6)
        neighbor = _disc(64, 70, 6)
        image = np.full(SHAPE, 600, dtype=np.uint16)
        row = self._extract(
            image, target, [_row("t", target), _row("n", neighbor)], outer_radius_um=30.0
        )
        assert row["annulus_neighbor_count"] == 0
        assert row["annulus_excluded_px"] == 0

    def test_no_background_is_subtracted_here(self):
        # RAW ONLY. Pooling is a well-grain operation; emitting a correction here would couple
        # "did we measure correctly" to "is our background model current".
        target = _disc(64, 40, 6)
        image = np.full(SHAPE, 600, dtype=np.uint16)
        row = self._extract(image, target, [_row("t", target)])
        assert not any("bgsub" in k or "corrected" in k for k in row)

    def test_mismatched_grids_fail_loud(self):
        with pytest.raises(ChannelIntensityError, match="different grids"):
            self._extract(
                np.zeros((64, 64), np.uint16), _disc(64, 40, 6), [_row("t", _disc(64, 40, 6))]
            )

    def test_row_is_self_describing(self):
        target = _disc(64, 40, 6)
        row = self._extract(np.full(SHAPE, 600, np.uint16), target, [_row("t", target)])
        # A reader must be able to interpret the counts without consulting this module's source.
        assert row["hist_n_bins"] == HIST_N_BINS
        assert row["hist_bin_width_dn"] == HIST_BIN_WIDTH_DN
        assert row["intensity_recipe_version"]
        assert row["image_micrometers_per_pixel_y"] == UM[0]
