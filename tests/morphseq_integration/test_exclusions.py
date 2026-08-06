"""Curated image-QC exclusions."""

from __future__ import annotations

import pandas as pd
import pytest

from morphseq_integration.exclusions import (
    EXCLUSION_COLUMNS,
    apply_exclusions,
    excluded_well_ids,
    load_excluded_wells,
)

HEADER = ",".join(EXCLUSION_COLUMNS)


def _write(tmp_path, *rows):
    path = tmp_path / "excluded_wells.csv"
    path.write_text(HEADER + "\n" + "".join(row + "\n" for row in rows))
    return path


class TestLoadExcludedWells:
    def test_shipped_table_is_valid(self):
        """The real curated table must always load — it gates every analysis."""
        table = load_excluded_wells()
        assert len(table) > 0
        assert table["well_id"].is_unique
        assert set(EXCLUSION_COLUMNS) <= set(table.columns)

    def test_shipped_table_covers_the_curated_gene7_wells(self):
        ids = excluded_well_ids()
        assert "20250612_24hpf_ctrl_atf6_A08" in ids
        assert "20250612_36hpf_wfs1_ctcf_H06" in ids
        assert len(ids) == 14

    def test_rejects_a_well_id_that_disagrees_with_its_spelling(self, tmp_path):
        """The redundant columns are a curator-facing safety net, so drift must fail loud."""
        path = _write(
            tmp_path,
            "20250612,24,ctrl_atf6,A08,20250612_24hpf_ctrl_atf6_Z99,image_qc,nl,2026-07-30",
        )
        with pytest.raises(ValueError, match="disagrees"):
            load_excluded_wells(path)

    def test_rejects_duplicates(self, tmp_path):
        row = "20250612,24,ctrl_atf6,A08,20250612_24hpf_ctrl_atf6_A08,image_qc,nl,2026-07-30"
        with pytest.raises(ValueError, match="duplicate well_id"):
            load_excluded_wells(_write(tmp_path, row, row))

    def test_rejects_a_missing_column(self, tmp_path):
        path = tmp_path / "excluded_wells.csv"
        path.write_text("well_id\n20250612_24hpf_ctrl_atf6_A08\n")
        with pytest.raises(ValueError, match="missing column"):
            load_excluded_wells(path)

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_excluded_wells(tmp_path / "absent.csv")


class TestApplyExclusions:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "well_id": [
                    "20250612_24hpf_ctrl_atf6_A08",  # excluded
                    "20250612_24hpf_ctrl_atf6_A01",  # kept
                    "20250612_36hpf_wfs1_ctcf_H06",  # excluded
                ],
                "experiment_id": ["a", "a", "b"],
            }
        )

    def test_drop_removes_only_excluded_rows(self, frame):
        out = apply_exclusions(frame)
        assert out["well_id"].tolist() == ["20250612_24hpf_ctrl_atf6_A01"]

    def test_flag_keeps_every_row(self, frame):
        out = apply_exclusions(frame, mode="flag")
        assert len(out) == 3
        assert out["curated_excluded"].tolist() == [True, False, True]

    def test_does_not_mutate_the_input(self, frame):
        before = frame.copy()
        apply_exclusions(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_unmatched_exclusions_are_not_an_error(self):
        """A single-plate table legitimately matches only some exclusions."""
        out = apply_exclusions(pd.DataFrame({"well_id": ["20250612_30hpf_wfs1_ctcf_A01"]}))
        assert len(out) == 1

    def test_rejects_an_unknown_mode(self, frame):
        with pytest.raises(ValueError, match="must be 'drop' or 'flag'"):
            apply_exclusions(frame, mode="nonsense")

    def test_rejects_a_frame_without_well_id(self):
        with pytest.raises(ValueError, match="no 'well_id' column"):
            apply_exclusions(pd.DataFrame({"x": [1]}))
