"""The pooled background null.

The load-bearing test is `test_the_mode_resists_a_bright_rim_where_the_mean_does_not`: the whole
reason the estimator is a mode rather than a mean is that the annulus distribution is
right-contaminated by rim autofluorescence that neighbor exclusion cannot remove. If that claim were
false, the simpler estimator would be the right one.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.feature_extraction.channel_intensity import (
    NULL_ESTIMATOR,
    ChannelIntensityNullError,
    correct_row,
    estimate_null,
    estimate_well_null,
    pool_histograms,
)
from data_pipeline.object_extraction.channel_intensity import (
    HIST_BIN_WIDTH_DN,
    HIST_N_BINS,
    histogram_region,
)

DTYPE_MAX = 65535


def _hist(values) -> list[int]:
    return histogram_region(
        np.asarray(values, dtype=np.uint16), dtype_max=DTYPE_MAX
    ).hist_counts.tolist()


def _background(n=5000, center=600, spread=20, seed=0):
    rng = np.random.RandomState(seed)
    return np.clip(rng.normal(center, spread, n), 0, DTYPE_MAX).astype(np.uint16)


class TestPoolingIsExact:
    def test_pooling_equals_measuring_everything_at_once(self):
        # The property that makes per-embryo histograms a valid substitute for re-reading pixels.
        a, b = _background(seed=1), _background(n=3000, seed=2)
        pooled = pool_histograms([_hist(a), _hist(b)])
        together = np.asarray(_hist(np.concatenate([a, b])), dtype=np.int64)
        np.testing.assert_array_equal(pooled, together)

    def test_pooling_is_order_independent(self):
        a, b, c = _background(seed=3), _background(seed=4), _background(seed=5)
        np.testing.assert_array_equal(
            pool_histograms([_hist(a), _hist(b), _hist(c)]),
            pool_histograms([_hist(c), _hist(a), _hist(b)]),
        )

    def test_a_foreign_bin_spec_fails_loud(self):
        # Rows written under a different bin spec are not poolable, and summing them would produce
        # a plausible-looking null that means nothing.
        with pytest.raises(ChannelIntensityNullError, match="bins"):
            pool_histograms([[1, 2, 3]])


class TestModeIsTheRightEstimator:
    def test_the_mode_resists_a_bright_rim_where_the_mean_does_not(self):
        # THE JUSTIFICATION FOR THE WHOLE ESTIMATOR CHOICE. The well rim autofluoresces (~1019 DN
        # measured, vs ~575 mid-well) and neighbor exclusion cannot remove it -- the rim is not a
        # mask. Here 15% of the pooled annulus is rim.
        clean = _background(n=8500, center=600, spread=20, seed=6)
        rim = _background(n=1500, center=1019, spread=40, seed=7)
        contaminated = np.concatenate([clean, rim])

        null = estimate_null([_hist(contaminated)])
        naive_mean = float(contaminated.mean())

        assert abs(null.mode_dn - 600) < 2 * HIST_BIN_WIDTH_DN, (
            "the mode must sit on the true background, ignoring the rim lobe"
        )
        assert naive_mean - 600 > 40, "fixture must actually drag the mean"
        assert abs(null.mode_dn - 600) < abs(naive_mean - 600), (
            "if the mean were as good here, the simpler estimator would be correct"
        )

    def test_robust_sigma_is_unscaled(self):
        # (p84-p16)/2 with NO 1.4826 Gaussian-consistency factor: that factor is a distributional
        # assumption and belongs downstream where someone makes it deliberately.
        null = estimate_null([_hist(_background(n=20000, center=600, spread=40, seed=8))])
        assert null.robust_sigma_dn == pytest.approx(40, rel=0.25)

    def test_a_thin_pool_is_flagged_not_silently_returned(self):
        # A null built from a handful of pixels is noise with a name.
        null = estimate_null([_hist(_background(n=50))], minimum_pooled_pixels=1000)
        assert not null.valid and null.pooled_px == 50

    def test_an_empty_pool_does_not_pretend(self):
        null = estimate_null([])
        assert not null.valid and np.isnan(null.mode_dn)

    def test_row_count_survives_a_generator(self):
        # estimate_null takes an iterable; counting after pooling would exhaust a generator and
        # report zero rows while the pixel count looked fine.
        null = estimate_null(_hist(_background(n=2000, seed=i)) for i in range(3))
        assert null.n_rows_pooled == 3


class TestWellNullMeasuresItsOwnAssumption:
    def _rows(self, per_time):
        return [
            {"time_index": t, "annulus_hist_counts": _hist(values)}
            for t, values in per_time.items()
        ]

    def test_drift_is_measured_rather_than_assumed(self):
        # Pooling over time assumes stationarity, which photobleaching and lamp drift violate -- and
        # that drift is confounded with developmental stage. So it is REPORTED, not assumed away.
        drifting = self._rows(
            {t: _background(n=4000, center=600 + 60 * t, seed=10 + t) for t in range(4)}
        )
        out = estimate_well_null(drifting)
        assert out["null_drift_dn"] > 100, "a real drift must be visible in the emitted evidence"
        assert out["null_n_timepoints"] == 4

    def test_a_stationary_well_reports_no_drift(self):
        stable = self._rows({t: _background(n=4000, center=600, seed=20 + t) for t in range(4)})
        out = estimate_well_null(stable)
        assert out["null_drift_dn"] <= HIST_BIN_WIDTH_DN

    def test_estimator_identity_is_carried(self):
        out = estimate_well_null(self._rows({0: _background(n=4000)}))
        assert out["null_estimator"] == NULL_ESTIMATOR

    def test_no_rows_is_an_error_not_a_fabricated_null(self):
        with pytest.raises(ChannelIntensityNullError, match="no rows"):
            estimate_well_null([])


class TestDosageSurvivesEndToEnd:
    """The requirement the whole path exists for, exercised through every stage at once.

    Extraction -> pooled null -> correction, on a simulated well with a bright autofluorescent rim
    in every frame. Unit tests can each pass while the composition still fails to recover dosage,
    so this asserts the thing actually wanted: that a 2-copy embryo reads twice a 1-copy one.
    """

    def test_two_copies_read_as_twice_one_copy(self):
        from data_pipeline.object_extraction.channel_intensity import (
            extract_embryo_intensity_evidence,
        )
        from image_geometry import BoxYX

        shape, um = (160, 160), (2.0, 2.0)
        rng = np.random.RandomState(0)
        background, signal = 600, {"A": 900, "B": 1200}  # A = 1 copy, B = 2 copies

        def disc(cy, cx, r):
            yy, xx = np.ogrid[: shape[0], : shape[1]]
            return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2

        def mask_row(mask_id, mask):
            box = BoxYX.from_mask(mask)
            return {
                "mask_id": mask_id, "bbox_y_min_px": box.y0, "bbox_y_max_px": box.y1,
                "bbox_x_min_px": box.x0, "bbox_x_max_px": box.x1, "_mask": mask,
            }

        a_mask, b_mask = disc(60, 50, 8), disc(60, 110, 8)
        neighbors = [mask_row("A", a_mask), mask_row("B", b_mask)]
        evidence = []
        for time_index in range(3):
            frame = np.clip(rng.normal(background, 20, shape), 0, DTYPE_MAX).astype(np.uint16)
            # The rim autofluoresces and is NOT a mask, so neighbor exclusion cannot remove it.
            yy, xx = np.ogrid[: shape[0], : shape[1]]
            rim = ((yy - 80) ** 2 + (xx - 80) ** 2) > 75**2
            frame[rim] = np.clip(
                rng.normal(1019, 40, int(rim.sum())), 0, DTYPE_MAX
            ).astype(np.uint16)

            for mask_id, mask in (("A", a_mask), ("B", b_mask)):
                lit = frame.copy()
                lit[mask] = np.clip(
                    rng.normal(signal[mask_id], 50, int(mask.sum())), 0, DTYPE_MAX
                ).astype(np.uint16)
                row = extract_embryo_intensity_evidence(
                    image=lit, target_mask=mask, target_mask_id=mask_id,
                    neighbors=neighbors, decode=lambda r: r["_mask"], um_per_px_yx=um,
                    inner_radius_um=6.0, outer_radius_um=30.0, exclude_radius_um=6.0,
                    dtype_max=DTYPE_MAX,
                )
                row["time_index"] = time_index
                row["embryo"] = mask_id
                evidence.append(row)

        null = estimate_well_null(evidence)
        assert null["null_valid"]
        assert abs(null["null_mode_dn"] - background) < 2 * HIST_BIN_WIDTH_DN, (
            "the pooled mode must land on the true background despite the rim"
        )

        def corrected(mask_id):
            return float(np.mean([
                correct_row(r, null)["embryo_mean_bgsub_dn"]
                for r in evidence if r["embryo"] == mask_id
            ]))

        ratio = corrected("B") / corrected("A")
        true_ratio = (signal["B"] - background) / (signal["A"] - background)
        assert ratio == pytest.approx(true_ratio, rel=0.05), (
            f"dosage did not survive: recovered {ratio:.3f} vs true {true_ratio:.3f}"
        )
        assert all(correct_row(r, null)["intensity_dosage_usable"] for r in evidence)


class TestCorrectionKeepsRawRaw:
    def _null(self, mode=600.0, sigma=20.0, valid=True):
        return {
            "null_estimator": NULL_ESTIMATOR,
            "null_mode_dn": mode,
            "null_robust_sigma_dn": sigma,
            "null_valid": valid,
            "null_drift_dn": 0.0,
        }

    def _row(self, mean_dn=1200.0, px=1000, clipped=0, area_fraction=0.9):
        return {
            "embryo_px": px,
            "embryo_sum_dn": mean_dn * px,
            "embryo_clipped_px": clipped,
            "annulus_area_fraction": area_fraction,
        }

    def test_corrected_columns_are_additions_not_overwrites(self):
        out = correct_row(self._row(), self._null())
        assert out["embryo_mean_dn"] == 1200.0, "raw must survive untouched"
        assert out["embryo_mean_bgsub_dn"] == pytest.approx(600.0)

    def test_negative_corrections_are_not_clipped(self):
        # A negative value means the null over-subtracted for this embryo -- exactly the diagnostic
        # needed to judge the estimator. Clipping would make an over-subtracting null look fine.
        out = correct_row(self._row(mean_dn=500.0), self._null(mode=600.0))
        assert out["embryo_mean_bgsub_dn"] < 0

    def test_both_mean_and_integrated_are_emitted(self):
        # Integrated scales with embryo volume, so across timepoints it conflates dosage with
        # growth. Picking one here would make that call for every downstream analysis.
        out = correct_row(self._row(), self._null())
        assert "embryo_mean_bgsub_dn" in out and "embryo_integrated_bgsub_dn" in out
        assert out["embryo_integrated_bgsub_dn"] == pytest.approx(600.0 * 1000)

    def test_saturation_blocks_the_dosage_call(self):
        # Saturation compresses 2-copy toward 1-copy: the comparison dies in the direction that
        # looks like clean data.
        out = correct_row(self._row(clipped=50), self._null())
        assert out["embryo_saturation_concern"]
        assert not out["intensity_dosage_usable"]

    def test_an_invalid_null_blocks_the_dosage_call(self):
        out = correct_row(self._row(), self._null(valid=False))
        assert not out["intensity_dosage_usable"]

    def test_a_mostly_excluded_annulus_blocks_the_dosage_call(self):
        out = correct_row(self._row(area_fraction=0.1), self._null())
        assert not out["intensity_dosage_usable"]

    def test_a_degenerate_null_spread_gives_nan_not_infinity(self):
        out = correct_row(self._row(), self._null(sigma=0.0))
        assert np.isnan(out["embryo_bgsub_z"])

    def test_correction_never_looks_at_genotype(self):
        # The pipeline is genotype-blind and must stay that way: a measurement that depended on its
        # own hypothesis would be worthless. Dosage is joined downstream, at analysis_ready.
        out = correct_row(self._row(), self._null())
        assert not any(
            k in out for k in ("genotype", "copy_number", "zygosity", "dosage")
        )
