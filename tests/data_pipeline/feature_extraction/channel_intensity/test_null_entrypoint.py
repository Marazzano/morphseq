"""The pooling entrypoint's I/O behaviour, which the kernel tests cannot see.

pooling.py and correction.py are pure functions over records and are tested as such. What is only
testable here is what the entrypoint does AROUND them: how it groups, what it refuses, and what
survives a CSV round-trip. Those are exactly where this path can go wrong silently -- a null pooled
across the wrong grouping still produces a plausible number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from data_pipeline.feature_extraction.channel_intensity.entrypoint import (
    run_channel_intensity_null,
)
from data_pipeline.object_extraction.channel_intensity.extraction import (
    HIST_BIN_WIDTH_DN,
    HIST_N_BINS,
)


def _hist(center_dn: int, n_px: int) -> list[int]:
    counts = np.zeros(HIST_N_BINS, dtype=int)
    counts[center_dn // HIST_BIN_WIDTH_DN] = n_px
    return counts.tolist()


def _row(well: str, time_index: int, *, annulus_dn: int, embryo_dn: int, product="RFP__projection__max"):
    embryo_px = 1000
    return {
        "experiment_id": "EXP",
        "well_id": well,
        "time_index": time_index,
        "mask_id": f"{well}_t{time_index}_m0",
        "source_image_product_key": product,
        "annulus_hist_counts": json.dumps(_hist(annulus_dn, 50_000)),
        "embryo_hist_counts": json.dumps(_hist(embryo_dn, embryo_px)),
        "embryo_sum_dn": float(embryo_dn * embryo_px),
        "embryo_sumsq_dn": float(embryo_dn**2 * embryo_px),
        "embryo_px": embryo_px,
        "embryo_clipped_px": 0,
    }


def _write(tmp_path, rows):
    import pandas as pd

    path = tmp_path / "channel_intensity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestTheNullIsPooledPerWellAndProduct:
    def test_each_well_gets_its_own_null(self, tmp_path):
        # Two wells with genuinely different backgrounds. A null pooled across BOTH would land
        # between them and be wrong for each -- and would still look like a perfectly ordinary number.
        source = _write(tmp_path, [
            _row("EXP_A01", 0, annulus_dn=400, embryo_dn=2000),
            _row("EXP_A01", 1, annulus_dn=400, embryo_dn=2000),
            _row("EXP_B01", 0, annulus_dn=1200, embryo_dn=2000),
            _row("EXP_B01", 1, annulus_dn=1200, embryo_dn=2000),
        ])
        null, _ = run_channel_intensity_null(
            channel_intensity_csv=source,
            output_null_csv=tmp_path / "null.csv",
            output_corrected_csv=tmp_path / "corrected.csv",
        )
        modes = null.set_index("well_id")["null_mode_dn"].to_dict()
        assert modes["EXP_A01"] < 500 < modes["EXP_B01"]

    def test_two_source_products_never_share_a_null(self, tmp_path):
        # Intensity off a CLAHE'd raster is a different quantity. Pooling them together produces a
        # null that describes neither, so the product must be part of the grouping key.
        source = _write(tmp_path, [
            _row("EXP_A01", 0, annulus_dn=400, embryo_dn=2000),
            _row("EXP_A01", 0, annulus_dn=1600, embryo_dn=2000, product="BF__projection__focus_stack"),
        ])
        null, _ = run_channel_intensity_null(
            channel_intensity_csv=source,
            output_null_csv=tmp_path / "null.csv",
            output_corrected_csv=tmp_path / "corrected.csv",
        )
        assert len(null) == 2, "one null per (well, source product)"
        assert null["null_mode_dn"].nunique() == 2


class TestCorrectedRowsAreSelfExplaining:
    def test_every_row_carries_its_null_and_the_estimator_name(self, tmp_path):
        # A corrected number whose provenance needs a join back tends to get quoted without it.
        source = _write(tmp_path, [_row("EXP_A01", 0, annulus_dn=400, embryo_dn=2000)])
        _, corrected = run_channel_intensity_null(
            channel_intensity_csv=source,
            output_null_csv=tmp_path / "null.csv",
            output_corrected_csv=tmp_path / "corrected.csv",
        )
        assert "well_null_mode_dn" in corrected.columns
        assert corrected["null_estimator"].iloc[0]

    def test_the_correction_actually_subtracts_the_background(self, tmp_path):
        source = _write(tmp_path, [_row("EXP_A01", 0, annulus_dn=400, embryo_dn=2000)])
        _, corrected = run_channel_intensity_null(
            channel_intensity_csv=source,
            output_null_csv=tmp_path / "null.csv",
            output_corrected_csv=tmp_path / "corrected.csv",
        )
        mode = float(corrected["well_null_mode_dn"].iloc[0])
        assert corrected["embryo_mean_bgsub_dn"].iloc[0] == pytest.approx(2000 - mode, abs=HIST_BIN_WIDTH_DN)

    def test_histograms_are_dropped_from_the_corrected_table(self, tmp_path):
        # Bulky, already persisted in the raw artifact, and nothing downstream of correction reads
        # them. Carrying them would multiply the corrected table's size for no reader.
        source = _write(tmp_path, [_row("EXP_A01", 0, annulus_dn=400, embryo_dn=2000)])
        _, corrected = run_channel_intensity_null(
            channel_intensity_csv=source,
            output_null_csv=tmp_path / "null.csv",
            output_corrected_csv=tmp_path / "corrected.csv",
        )
        assert not [c for c in corrected.columns if c.endswith("_hist_counts")]


class TestItRefusesRatherThanFabricates:
    def test_an_empty_table_raises(self, tmp_path):
        import pandas as pd

        path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["experiment_id", "well_id", "source_image_product_key"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="no rows"):
            run_channel_intensity_null(
                channel_intensity_csv=path,
                output_null_csv=tmp_path / "null.csv",
                output_corrected_csv=tmp_path / "corrected.csv",
            )

    def test_a_missing_grouping_key_raises_rather_than_pooling_everything(self, tmp_path):
        # Without source_image_product_key the groupby would silently pool every product together.
        # Failing loud beats a null that describes a mixture nobody asked for.
        rows = [_row("EXP_A01", 0, annulus_dn=400, embryo_dn=2000)]
        del rows[0]["source_image_product_key"]
        with pytest.raises(ValueError, match="source_image_product_key"):
            run_channel_intensity_null(
                channel_intensity_csv=_write(tmp_path, rows),
                output_null_csv=tmp_path / "null.csv",
                output_corrected_csv=tmp_path / "corrected.csv",
            )
