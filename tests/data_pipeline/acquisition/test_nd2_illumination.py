"""Exposure parsing, pinned against the layout that actually appears in Nikon ND2 text dumps.

THE BUG THIS EXISTS TO PREVENT is not "no exposure" -- it is "the wrong exposure, silently." A
parser that mis-aligns channels reports two differently-exposed channels as identical, which
manufactures agreement exactly where the analysis is looking for a difference. That is strictly
worse than returning nothing, so every test below is about ALIGNMENT, not about extraction.
"""

from __future__ import annotations

import pytest

from data_pipeline.acquisition.metadata_ingest.scope.yx1.nd2_illumination import (
    ChannelIllumination,
    read_channel_illumination,
)


class _FakeND2:
    def __init__(self, blob: str | None):
        self.text_info = None if blob is None else {"description": blob}


# The real layout, reduced from 20260624_pbx_flouresence_bf_pilot_plate01_t33hpf.nd2. The settings
# dump repeats as BLOCKS of n_channels -- BF then tdtomato, the pair repeated -- and NOT as runs of
# one channel at a time. Getting this backwards is the mis-alignment bug.
_TWO_CHANNEL_REAL = """
Camera Settings:   Exposure: 11 ms
Nikon Ti2, Illuminator(DIA) Iris intensity: 7.5
SpectraIII/Celesta, MultiLaser(Celesta):
 Power:  1.0
 Power:  2.0
Camera Settings:   Exposure: 600 ms
Nikon Ti2, Illuminator(DIA) Iris intensity: 18.1
SpectraIII/Celesta, MultiLaser(Celesta):
 Power: 20.0
 Power:  2.0
"""


class TestChannelAlignment:
    def test_each_channel_gets_its_own_exposure(self):
        # THE REGRESSION. A stride-based read would return 11.0 for both channels, hiding a 55x
        # difference -- and on the real pbx files it hid the 600-vs-300 fluorescence change that
        # looked exactly like a 2x dosage effect.
        bf, fluorescence = read_channel_illumination(_FakeND2(_TWO_CHANNEL_REAL), 2)
        assert bf.exposure_ms == 11.0
        assert fluorescence.exposure_ms == 600.0

    def test_the_active_laser_line_wins_over_its_idle_floor(self):
        # Idle lines sit at 1.0; the line that was actually driving the channel is the max.
        _, fluorescence = read_channel_illumination(_FakeND2(_TWO_CHANNEL_REAL), 2)
        assert fluorescence.illumination_power == 20.0

    def test_iris_follows_its_own_channel(self):
        bf, fluorescence = read_channel_illumination(_FakeND2(_TWO_CHANNEL_REAL), 2)
        assert (bf.dia_iris_intensity, fluorescence.dia_iris_intensity) == (7.5, 18.1)


class TestItFailsSoftAndNeverGuesses:
    @pytest.mark.parametrize(
        "blob",
        [None, "", "no settings here at all"],
        ids=["no-text-info", "empty", "unparseable"],
    )
    def test_missing_metadata_yields_empties_not_an_exception(self, blob):
        # A metadata read must degrade an analysis, never take down an ingest.
        assert read_channel_illumination(_FakeND2(blob), 2) == [ChannelIllumination()] * 2

    def test_a_count_that_does_not_divide_by_channels_yields_empties(self):
        # Three exposures across two channels means the format is not what this parser assumes.
        # Returning a guessed alignment here would be the mis-attribution bug wearing a smile.
        blob = "Exposure: 11 ms\nExposure: 600 ms\nExposure: 300 ms"
        assert read_channel_illumination(_FakeND2(blob), 2) == [ChannelIllumination()] * 2

    def test_one_entry_per_channel_always(self):
        for n in (1, 2, 4):
            assert len(read_channel_illumination(_FakeND2(_TWO_CHANNEL_REAL), n)) == n
