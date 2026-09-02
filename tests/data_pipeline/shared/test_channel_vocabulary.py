"""Tests for the canonical channel vocabulary — names, validation, and display colors."""

import re

import pytest

from data_pipeline.shared.channel_vocabulary import (
    BRIGHTFIELD_CHANNELS,
    CHANNEL_ID_COLORS,
    VALID_CHANNEL_NAMES,
    color_for_channel_id,
    validate_channel_id,
)

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


class TestChannelIdColors:
    def test_every_canonical_channel_has_a_color(self):
        """THE test that earns its place: a channel added without a color must fail here.

        Asserted as set equality against the vocabulary rather than by re-listing channel names, so
        this tracks VALID_CHANNEL_NAMES automatically instead of drifting from it.
        """
        assert set(CHANNEL_ID_COLORS) == set(VALID_CHANNEL_NAMES)

    @pytest.mark.parametrize("channel_id", sorted(CHANNEL_ID_COLORS))
    def test_colors_are_six_digit_hex(self, channel_id):
        # Format only — the specific hex values are the contract's to own, not the test's to restate.
        assert _HEX_COLOR.match(CHANNEL_ID_COLORS[channel_id])

    def test_colors_are_distinct(self):
        """Two channels sharing a color would render as one, silently."""
        assert len(set(CHANNEL_ID_COLORS.values())) == len(CHANNEL_ID_COLORS)

    @pytest.mark.parametrize("channel_id", sorted(VALID_CHANNEL_NAMES))
    def test_accessor_returns_the_mapped_color(self, channel_id):
        assert color_for_channel_id(channel_id) == CHANNEL_ID_COLORS[channel_id]

    def test_accessor_rejects_a_non_vocabulary_channel(self):
        with pytest.raises(ValueError, match="not in the canonical vocabulary"):
            color_for_channel_id("NOPE")

    def test_accessor_rejects_a_raw_microscope_name(self):
        """Colors are keyed on canonical channel_id, never a scope's raw dialect.

        'tdtomato' is a real YX1 raw channel name that maps to RFP; asking for its color directly
        must fail rather than silently missing, since the dialect->canonical translation is the
        scope adapter's job.
        """
        with pytest.raises(ValueError, match="not in the canonical vocabulary"):
            color_for_channel_id("tdtomato")

    def test_brightfield_is_neutral_not_a_fluorophore_color(self):
        """BF renders gray: it is transmitted light, not an emission channel.

        Checked structurally (R == G == B) rather than against a literal hex, so the exact shade can
        change without touching this test.
        """
        bf_color = color_for_channel_id(BRIGHTFIELD_CHANNELS[0]).lstrip("#")
        r, g, b = (int(bf_color[i:i + 2], 16) for i in (0, 2, 4))
        assert r == g == b


class TestValidateChannelId:
    @pytest.mark.parametrize("channel_id", sorted(VALID_CHANNEL_NAMES))
    def test_accepts_every_canonical_name(self, channel_id):
        assert validate_channel_id(channel_id) == channel_id

    def test_rejects_unknown_and_names_the_vocabulary(self):
        with pytest.raises(ValueError, match="not in the canonical vocabulary"):
            validate_channel_id("mCherry")
