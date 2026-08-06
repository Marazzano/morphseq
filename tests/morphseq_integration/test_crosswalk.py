"""Crosswalk assembly and its 1-to-1 invariant."""

from __future__ import annotations

import pandas as pd
import pytest

from morphseq_integration.crosswalk import (
    ALL_STATUSES,
    CROSSWALK_COLUMNS,
    STATUS_BLANK_WELL,
    STATUS_NO_HASH_MAP,
    STATUS_PAIRED,
    summarize,
    validate_crosswalk,
)
from morphseq_integration.experiment_key import load_experiment_key


def _row(**overrides) -> dict:
    row = {
        "experiment_id": "20250612_30hpf_ctrl_atf6",
        "well_id": "20250612_30hpf_ctrl_atf6_A01",
        "well_index": "A01",
        "sci_expt": "GENE7",
        "hash_plate": "P18",
        "hash_well": "A01",
        "seq_sample_id": "GENE7_P18_A1",
        "pairing_status": STATUS_PAIRED,
        "hash_map_source": "plate_workbook",
    }
    row.update(overrides)
    return row


class TestValidateCrosswalk:
    def test_accepts_a_one_to_one_table(self):
        crosswalk = pd.DataFrame(
            [
                _row(),
                _row(
                    well_id="20250612_30hpf_ctrl_atf6_A02",
                    well_index="A02",
                    hash_well="A02",
                    seq_sample_id="GENE7_P18_A2",
                ),
            ]
        )
        validate_crosswalk(crosswalk)

    def test_rejects_a_duplicate_well_id(self):
        crosswalk = pd.DataFrame([_row(), _row(seq_sample_id="GENE7_P18_B1")])
        with pytest.raises(ValueError, match="duplicate well_id"):
            validate_crosswalk(crosswalk)

    def test_rejects_two_wells_claiming_one_sample(self):
        """A duplicate seq_sample_id means a bad hash_plate_num or a wrong sci_expt."""
        crosswalk = pd.DataFrame(
            [_row(), _row(well_id="20250612_30hpf_ctrl_atf6_A02", well_index="A02")]
        )
        with pytest.raises(ValueError, match="same seq_sample_id"):
            validate_crosswalk(crosswalk)

    def test_ignores_unpaired_rows(self):
        """Blank wells share an empty sample id by construction; only paired rows must be 1-to-1."""
        crosswalk = pd.DataFrame(
            [
                _row(),
                _row(
                    well_id="20250612_30hpf_ctrl_atf6_A02",
                    well_index="A02",
                    hash_plate="",
                    hash_well="",
                    seq_sample_id="",
                    pairing_status=STATUS_BLANK_WELL,
                ),
                _row(
                    well_id="20250612_30hpf_ctrl_atf6_A03",
                    well_index="A03",
                    hash_plate="",
                    hash_well="",
                    seq_sample_id="",
                    pairing_status=STATUS_BLANK_WELL,
                ),
            ]
        )
        validate_crosswalk(crosswalk)

    def test_rejects_a_missing_column(self):
        with pytest.raises(ValueError, match="missing column"):
            validate_crosswalk(pd.DataFrame([{"experiment_id": "x"}]))


class TestSummarize:
    def test_always_reports_every_status(self):
        """The headline 'paired' count must not vanish when nothing paired — that is the case you
        most need to see."""
        crosswalk = pd.DataFrame(
            [
                _row(
                    hash_plate="",
                    hash_well="",
                    seq_sample_id="",
                    pairing_status=STATUS_NO_HASH_MAP,
                )
            ]
        )
        summary = summarize(crosswalk)
        for status in ALL_STATUSES:
            assert status in summary.columns
        assert summary.loc[0, STATUS_PAIRED] == 0
        assert summary.loc[0, STATUS_NO_HASH_MAP] == 1

    def test_counts_per_experiment(self):
        crosswalk = pd.DataFrame(
            [
                _row(),
                _row(
                    well_id="20250612_30hpf_ctrl_atf6_A02",
                    well_index="A02",
                    hash_well="A02",
                    seq_sample_id="GENE7_P18_A2",
                ),
                _row(
                    experiment_id="20250612_24hpf_ctrl_atf6",
                    well_id="20250612_24hpf_ctrl_atf6_A01",
                    hash_plate="P01",
                    seq_sample_id="GENE7_P01_A1",
                ),
            ]
        )
        summary = summarize(crosswalk).set_index("experiment_id")
        assert summary.loc["20250612_30hpf_ctrl_atf6", STATUS_PAIRED] == 2
        assert summary.loc["20250612_24hpf_ctrl_atf6", STATUS_PAIRED] == 1

    def test_column_order_is_stable(self):
        summary = summarize(pd.DataFrame([_row()]))
        assert list(summary.columns) == ["experiment_id", "sci_expt", *ALL_STATUSES]


class TestExperimentKey:
    def test_shipped_key_loads_and_covers_the_six_gene7_plates(self):
        key = load_experiment_key()
        assert len(key) >= 6
        for timepoint in ("24", "30", "36"):
            for arm in ("ctrl_atf6", "wfs1_ctcf"):
                experiment_id = f"20250612_{timepoint}hpf_{arm}"
                assert key.sci_expt_for(experiment_id) == "GENE7"

    def test_unknown_experiment_returns_none(self):
        assert load_experiment_key().sci_expt_for("not_an_experiment") is None

    def test_rejects_duplicate_rows(self, tmp_path):
        path = tmp_path / "key.csv"
        path.write_text(
            "experiment_id,sci_expt,notes\n20250612_24hpf_ctrl_atf6,GENE7,a\n"
            "20250612_24hpf_ctrl_atf6,GENE8,b\n"
        )
        with pytest.raises(ValueError, match="duplicate experiment_id"):
            load_experiment_key(path)

    def test_rejects_blank_sci_expt_value(self, tmp_path):
        path = tmp_path / "key.csv"
        path.write_text("experiment_id,sci_expt,notes\n20250612_24hpf_ctrl_atf6,,unknown yet\n")
        with pytest.raises(ValueError, match="no sci_expt"):
            load_experiment_key(path)

    def test_rejects_a_missing_column(self, tmp_path):
        path = tmp_path / "key.csv"
        path.write_text("experiment_id,notes\n20250612_24hpf_ctrl_atf6,x\n")
        with pytest.raises(ValueError, match="missing required column"):
            load_experiment_key(path)

    def test_tolerates_trailing_blank_lines(self, tmp_path):
        path = tmp_path / "key.csv"
        path.write_text("experiment_id,sci_expt,notes\n20250612_24hpf_ctrl_atf6,GENE7,x\n,,\n")
        assert len(load_experiment_key(path)) == 1

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_experiment_key(tmp_path / "absent.csv")


def test_crosswalk_columns_are_the_documented_shape():
    assert CROSSWALK_COLUMNS == (
        "experiment_id",
        "well_id",
        "well_index",
        "sci_expt",
        "hash_plate",
        "hash_well",
        "seq_sample_id",
        "pairing_status",
        "hash_map_source",
    )
