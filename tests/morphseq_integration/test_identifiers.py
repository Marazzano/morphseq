"""Sequencing identifier grammar — the boundary where morphseq's padded wells meet sci-PLEX's."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphseq_integration.identifiers import (
    build_seq_sample_id,
    format_hash_plate,
    is_blank,
    normalize_hash_well,
    strip_rt_block,
    to_sequencing_well,
)


class TestNormalizeHashWell:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("A1", "A01"),
            ("A01", "A01"),
            ("a1", "A01"),
            ("a01", "A01"),
            (" B7 ", "B07"),
            ("H12", "H12"),
            ("C010", "C10"),
        ],
    )
    def test_canonicalizes_every_spelling(self, raw, expected):
        assert normalize_hash_well(raw) == expected

    @pytest.mark.parametrize("raw", ["I01", "A13", "A0", "AA1", "1A", "", "A", "A1B"])
    def test_rejects_out_of_range_and_malformed(self, raw):
        with pytest.raises(ValueError):
            normalize_hash_well(raw)


class TestToSequencingWell:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("A01", "A1"), ("A1", "A1"), ("H12", "H12"), ("B07", "B7"), ("a09", "A9")],
    )
    def test_drops_the_zero_pad(self, raw, expected):
        assert to_sequencing_well(raw) == expected

    def test_round_trips_through_the_padded_form(self):
        for row in "ABCDEFGH":
            for col in range(1, 13):
                padded = f"{row}{col:02d}"
                assert normalize_hash_well(to_sequencing_well(padded)) == padded


class TestFormatHashPlate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (1, "P01"),
            ("1", "P01"),
            (18, "P18"),
            ("P1", "P01"),
            ("P01", "P01"),
            ("p18", "P18"),
            ("18.0", "P18"),
            (18.0, "P18"),
            (100, "P100"),
        ],
    )
    def test_normalizes_authored_spellings(self, raw, expected):
        """Excel round-trips integer cells as floats, so '18.0' must not be rejected."""
        assert format_hash_plate(raw) == expected

    @pytest.mark.parametrize("raw", [0, "0", "P0", -1, "abc", "", "P", "1.5"])
    def test_rejects_non_positive_and_malformed(self, raw):
        with pytest.raises(ValueError):
            format_hash_plate(raw)


class TestBuildSeqSampleId:
    def test_mints_the_documented_gene7_key(self):
        assert build_seq_sample_id("GENE7", 1, "A01") == "GENE7_P01_A1"
        assert build_seq_sample_id("GENE7", 18, "A01") == "GENE7_P18_A1"
        assert build_seq_sample_id("GENE7", "P04", "H12") == "GENE7_P04_H12"

    def test_is_insensitive_to_input_spelling(self):
        """Every authored spelling of the same well must mint one key."""
        keys = {
            build_seq_sample_id("GENE7", plate, well)
            for plate in (5, "5", "P5", "P05", "5.0")
            for well in ("B7", "B07", "b7", " B07 ")
        }
        assert keys == {"GENE7_P05_B7"}

    def test_rejects_empty_experiment(self):
        with pytest.raises(ValueError):
            build_seq_sample_id("   ", 1, "A01")


class TestStripRtBlock:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("GENE7_P18_A1_Bl1", "GENE7_P18_A1"),
            ("GENE7_P18_A1_Bl12", "GENE7_P18_A1"),
            ("GENE7_P18_A1_bl3", "GENE7_P18_A1"),
            ("GENE7_P18_A1", "GENE7_P18_A1"),
        ],
    )
    def test_reduces_to_the_sample_key(self, raw, expected):
        assert strip_rt_block(raw) == expected

    def test_is_idempotent(self):
        once = strip_rt_block("GENE7_P18_A1_Bl1")
        assert strip_rt_block(once) == once

    def test_leaves_a_non_block_suffix_alone(self):
        """Only _Bl<digits> is an RT block; a similar-looking suffix must survive."""
        assert strip_rt_block("GENE7_P18_A1_Block") == "GENE7_P18_A1_Block"
        assert strip_rt_block("GENE7_P18_A1_Bl") == "GENE7_P18_A1_Bl"


class TestIsBlank:
    @pytest.mark.parametrize("value", [None, float("nan"), np.nan, pd.NA, pd.NaT, "", "   ", "\t"])
    def test_true_for_unauthored_cells(self, value):
        assert is_blank(value) is True

    @pytest.mark.parametrize("value", ["A1", "0", 0, 1, 18.0, "nan"])
    def test_false_for_authored_values(self, value):
        """0 is a real authored value; the string 'nan' is text, not a missing marker."""
        assert is_blank(value) is False
