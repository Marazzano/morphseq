"""Tests for plate_metadata_loader (L1 ingest)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipeline.metadata_ingest.plate.plate_metadata_loader import (
    PlatePages,
    classify_plate_page,
    ingest_plate_grid_sheet_to_long,
    load_plate_metadata_pages,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_grid_df() -> pd.DataFrame:
    """Return a minimal valid 8×12 grid DataFrame (row labels in col 0, cols 1–12)."""
    rows = list("ABCDEFGH")
    data = {"row": rows}
    for col in range(1, 13):
        data[col] = [f"{r}{col:02d}" for r in rows]
    return pd.DataFrame(data)


def _make_long_df() -> pd.DataFrame:
    """Return a DataFrame that looks like a long-table format (has well_index column)."""
    return pd.DataFrame({
        "well_index": [f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)],
        "genotype": ["wt"] * 96,
    })


# ---------------------------------------------------------------------------
# classify_plate_page
# ---------------------------------------------------------------------------

class TestClassifyPlatePage:
    def test_grid_classified_as_grid(self):
        df = _make_grid_df()
        cls, _ = classify_plate_page(df)
        assert cls == "grid"

    def test_long_classified_as_long(self):
        df = _make_long_df()
        cls, _ = classify_plate_page(df)
        assert cls == "long"

    def test_tiny_df_is_rejected(self):
        df = pd.DataFrame({"a": [1, 2]})
        cls, reason = classify_plate_page(df)
        assert cls == "rejected"
        assert reason

    def test_ambiguous_raises(self):
        # A grid-shaped df that also has a "well_index" column header.
        df = _make_grid_df()
        df = df.rename(columns={"row": "well_index"})
        with pytest.raises(ValueError, match="readable as both"):
            classify_plate_page(df)


# ---------------------------------------------------------------------------
# ingest_plate_grid_sheet_to_long
# ---------------------------------------------------------------------------

class TestIngestPlateGridSheetToLong:
    def test_valid_grid_produces_96_rows(self):
        df = _make_grid_df()
        long = ingest_plate_grid_sheet_to_long(df, page_name="genotype")
        assert len(long) == 96
        assert "well_index" in long.columns
        assert "genotype" in long.columns

    def test_page_methods_key_matches_page_name(self):
        df = _make_grid_df()
        long = ingest_plate_grid_sheet_to_long(df, page_name="genotype")
        assert long.columns.tolist() == ["well_index", "genotype"]

    def test_well_index_format_is_canonical(self):
        df = _make_grid_df()
        long = ingest_plate_grid_sheet_to_long(df, page_name="genotype")
        # All well_index values should match A01..H12
        import re
        pattern = re.compile(r"^[A-H]\d{2}$")
        bad = [w for w in long["well_index"] if not pattern.match(w)]
        assert bad == [], f"Non-canonical well_index values: {bad}"

    def test_cols_not_1_to_12_raises(self):
        df = _make_grid_df()
        # Rename col 1 to 0 so sequence is 0, 2, 3, …, 12 — breaks the 1-12 rule.
        df = df.rename(columns={1: 0})
        with pytest.raises(ValueError, match="column headers"):
            ingest_plate_grid_sheet_to_long(df, page_name="genotype")

    def test_cols_shifted_range_raises(self):
        df = _make_grid_df()
        # Shift all col headers up by 1 → 2–13 instead of 1–12.
        rename = {c: c + 1 for c in range(1, 13)}
        df = df.rename(columns=rename)
        with pytest.raises(ValueError, match="column headers"):
            ingest_plate_grid_sheet_to_long(df, page_name="genotype")

    def test_rows_not_A_to_H_raises(self):
        df = _make_grid_df()
        rows = df["row"].tolist()
        rows[0] = "X"
        df["row"] = rows
        with pytest.raises(ValueError, match="row labels"):
            ingest_plate_grid_sheet_to_long(df, page_name="genotype")

    def test_wrong_shape_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="shape"):
            ingest_plate_grid_sheet_to_long(df, page_name="genotype")


# ---------------------------------------------------------------------------
# load_plate_metadata_pages (uses mock ExcelFile)
# ---------------------------------------------------------------------------

class TestLoadPlateMetadataPages:
    """Tests that do not hit the filesystem use a patched ExcelFile."""

    def _patch_excel(self, sheet_map: dict[str, pd.DataFrame]):
        """Return a context manager mock that yields an ExcelFile-like object."""
        mock_xlf = MagicMock()
        mock_xlf.__enter__ = lambda s: mock_xlf
        mock_xlf.__exit__ = MagicMock(return_value=False)
        mock_xlf.sheet_names = list(sheet_map.keys())

        def parse_side_effect(name, **kwargs):
            return sheet_map[name]

        mock_xlf.parse = parse_side_effect
        return mock_xlf

    def _run(self, sheet_map: dict[str, pd.DataFrame], tmp_path: Path) -> PlatePages:
        """Write a fake xlsx (actually a zip) OR mock ExcelFile and call the loader."""
        # Write real xlsx via openpyxl if available; otherwise skip.
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            pytest.skip("openpyxl required for loader integration tests")

        with pd.ExcelWriter(tmp_path / "meta.xlsx", engine="openpyxl") as writer:
            for sheet_name, df in sheet_map.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        return load_plate_metadata_pages(tmp_path / "meta.xlsx")

    def test_single_grid_page_accepted(self, tmp_path):
        pages = self._run({"genotype": _make_grid_df()}, tmp_path)
        assert "genotype" in pages.accepted
        assert len(pages.table) == 96
        assert "genotype" in pages.table.columns
        assert pages.page_methods["genotype"] == "grid"

    def test_page_methods_records_grid_for_grid_page(self, tmp_path):
        pages = self._run({"genotype": _make_grid_df()}, tmp_path)
        assert pages.page_methods.get("genotype") == "grid"

    def test_table_has_no_row_level_ingest_format_column(self, tmp_path):
        pages = self._run({"genotype": _make_grid_df()}, tmp_path)
        assert "ingest_format" not in pages.table.columns

    def test_long_page_accepted_and_passes_values_through(self, tmp_path):
        pages = self._run({"well_data": _make_long_df()}, tmp_path)
        # The long ingester is now BUILT — the page's value columns are accepted, not rejected.
        assert "well_data" not in [r[0] for r in pages.rejected]
        assert "genotype" in pages.accepted
        assert "genotype" in pages.table.columns
        assert pages.page_methods.get("genotype") == "long"

    def test_extra_grid_page_sequenced_flows_through(self, tmp_path):
        pages = self._run(
            {"genotype": _make_grid_df(), "sequenced": _make_grid_df()},
            tmp_path,
        )
        assert "genotype" in pages.accepted
        assert "sequenced" in pages.accepted
        assert "genotype" in pages.table.columns
        assert "sequenced" in pages.table.columns

    def test_column_name_collision_raises(self, tmp_path):
        # Two sheets that normalize to the same name.
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            pytest.skip("openpyxl required")

        # "My Genotype" and "my_genotype" both normalize to "my_genotype".
        with pd.ExcelWriter(tmp_path / "meta.xlsx", engine="openpyxl") as writer:
            _make_grid_df().to_excel(writer, sheet_name="my genotype", index=False)
            _make_grid_df().to_excel(writer, sheet_name="my_genotype", index=False)

        with pytest.raises(ValueError, match="collides"):
            load_plate_metadata_pages(tmp_path / "meta.xlsx")

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_plate_metadata_pages(tmp_path / "nonexistent.xlsx")
