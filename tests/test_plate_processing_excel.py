from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.plate.plate_processing import process_plate_layout


def _make_plate_sheet(*, values: list[list[object]], row_labels: list[object]) -> pd.DataFrame:
    """
    Build a DataFrame shaped like the Numbers-exported MorphSeq plate sheets:
    - header row with columns: ['Unnamed: 0', 1..12]
    - 8 data rows A..H, but row label cells can be missing (NaN).
    """
    assert len(values) == 8
    assert all(len(r) == 12 for r in values)
    assert len(row_labels) == 8
    cols = ["Unnamed: 0"] + list(range(1, 13))
    rows = []
    for lab, row in zip(row_labels, values):
        rows.append([lab] + row)
    return pd.DataFrame(rows, columns=cols)


def _write_plate_workbook(xlsx_path: Path, *, row_labels: list[object]) -> None:
    """Write a full synthetic plate workbook, with ``row_labels`` used for the string grids.

    Factored out so the strict-row-label test and the parsing test build byte-identical workbooks
    apart from the one thing under test.
    """
    # genotype: fill with distinct values for a couple wells we assert on.
    geno_vals = [["wt"] * 12 for _ in range(8)]
    geno_vals[7][0] = "hom"  # H01
    geno_vals[7][1] = "het"  # H02

    # chem_perturbation: blank everywhere -> should become treatment="none"
    # Use whitespace (not NaN) so the row is preserved when written to Excel, but still normalizes
    # to "none".
    treat_vals = [[""] * 12 for _ in range(8)]
    treat_vals[7][0] = " "  # keep last row from being entirely empty in Excel

    # start_age_hpf: drop A01 by leaving it empty (NaN), keep all others.
    age_vals = [[13.0] * 12 for _ in range(8)]
    age_vals[0][0] = np.nan  # A01 missing

    temp_vals = [[30.0] * 12 for _ in range(8)]
    medium_vals = [["MC10"] * 12 for _ in range(8)]
    # series_number_map: numeric grid (not strictly required by schema, but required by mapping
    # helpers).
    series_vals = [[float(i + 1 + r * 12) for i in range(12)] for r in range(8)]
    well_formed = ["A", "B", "C", "D", "E", "F", "G", "H"]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        _make_plate_sheet(values=medium_vals, row_labels=row_labels).to_excel(
            w, sheet_name="medium", index=False)
        _make_plate_sheet(values=geno_vals, row_labels=row_labels).to_excel(
            w, sheet_name="genotype", index=False)
        _make_plate_sheet(values=treat_vals, row_labels=row_labels).to_excel(
            w, sheet_name="chem_perturbation", index=False)
        _make_plate_sheet(values=age_vals, row_labels=well_formed).to_excel(
            w, sheet_name="start_age_hpf", index=False)
        _make_plate_sheet(values=temp_vals, row_labels=well_formed).to_excel(
            w, sheet_name="temperature", index=False)
        _make_plate_sheet(values=series_vals, row_labels=well_formed).to_excel(
            w, sheet_name="series_number_map", index=False)


def test_missing_row_label_is_REJECTED_not_inferred(tmp_path: Path) -> None:
    """A grid page whose row labels are not exactly A-H in sequence must fail loud.

    POLICY REVERSAL, pinned deliberately. This test previously asserted the opposite — that a blank
    "H" label cell was tolerated and the row recovered by position. The loader now requires exact
    A-H (_assert_rows_are_A_to_H_in_sequence), and that is the safer contract: inferring a plate row
    from its position is how an entire row of 12 wells silently acquires the wrong genotype when a
    sheet has an extra, missing, or reordered row. A hard failure names the sheet; a silent
    inference does not.
    """
    xlsx_path = tmp_path / "plate.xlsx"
    _write_plate_workbook(xlsx_path, row_labels=["A", "B", "C", "D", "E", "F", "G", np.nan])

    with pytest.raises(ValueError, match="must be exactly"):
        process_plate_layout(xlsx_path, experiment_id="EXP", output_csv=tmp_path / "out.csv")


def test_process_plate_layout_excel_parses_grids_and_defaults_treatment(tmp_path: Path) -> None:
    # Create a synthetic workbook with the minimum required sheets plus series_number_map.
    xlsx_path = tmp_path / "plate.xlsx"

    row_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    _write_plate_workbook(xlsx_path, row_labels=row_labels)

    out_csv = tmp_path / "plate_metadata.csv"
    df = process_plate_layout(xlsx_path, experiment_id="EXP", output_csv=out_csv)

    # ALL 96 WELLS SURVIVE, INCLUDING A01 WITH ITS BLANK start_age_hpf. This test used to assert
    # 95 rows, i.e. that a well missing start_age_hpf was DROPPED here. Parsing no longer discards
    # wells: a blank cell becomes a null the row still carries. That is the better split — dropping
    # a well is a QC judgement, and making it silently at parse time deletes the evidence that the
    # sheet was incomplete. Downstream consumers (e.g. stage_predictions) flag the null explicitly.
    assert len(df) == 96
    a01 = df[df["well_index"].astype(str) == "A01"]
    assert len(a01) == 1
    assert pd.isna(a01["start_age_hpf"].iloc[0])

    # The grid is passed through under its OWN sheet name. Renaming chem_perturbation -> treatment
    # and defaulting blanks to "none" is a later normalization step, not this function's job.
    assert "chem_perturbation" in df.columns
    assert "treatment" not in df.columns

    # The H row parses to the right wells (the grid is read by label, not by position).
    h = df[df["well_index"].astype(str).str.startswith("H")].set_index("well_index")
    assert h.loc["H01", "genotype"] == "hom"
    assert h.loc["H02", "genotype"] == "het"

    # series_number_map should be parsed as a per-well column (float is fine).
    assert "series_number_map" in df.columns
    assert df["series_number_map"].notna().all()
