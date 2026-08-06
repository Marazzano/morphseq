"""Tests for the two-channel display composite."""

import numpy as np
import pytest

from data_pipeline.shared.channel_vocabulary import CHANNEL_ID_COLORS, color_for_channel_id
from data_pipeline.viz.channel_composite import (
    composite_two_channels,
    hex_to_rgb_fractions,
    stretch_to_unit,
    tint_single_channel,
)


class TestStretchToUnit:
    def test_maps_percentile_window_to_unit_range(self):
        img = np.arange(0, 10000, dtype=np.uint16).reshape(100, 100)
        out = stretch_to_unit(img)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_flat_image_does_not_divide_by_zero(self):
        out = stretch_to_unit(np.full((8, 8), 500, dtype=np.uint16))
        assert np.all(out == 0.0)

    def test_rejects_a_non_2d_image(self):
        with pytest.raises(ValueError, match="2D single-channel"):
            stretch_to_unit(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_stretch_is_per_image_not_shared(self):
        """The load-bearing property: a dim channel must still reach full range on its own.

        BF and RFP differ ~22x in the real data; under a shared stretch the dim channel would be
        invisible. Independent stretching is what makes the overlay show anything at all.
        """
        bright = (np.arange(64, dtype=np.float64).reshape(8, 8) * 1000)
        dim = (np.arange(64, dtype=np.float64).reshape(8, 8) * 10)
        assert stretch_to_unit(dim).max() == pytest.approx(stretch_to_unit(bright).max())


class TestHexConversion:
    @pytest.mark.parametrize("channel_id", sorted(CHANNEL_ID_COLORS))
    def test_every_vocabulary_color_converts(self, channel_id):
        r, g, b = hex_to_rgb_fractions(color_for_channel_id(channel_id))
        assert all(0.0 <= v <= 1.0 for v in (r, g, b))

    def test_rejects_malformed_hex(self):
        with pytest.raises(ValueError, match="#RRGGBB"):
            hex_to_rgb_fractions("#F00")


class TestCompositeTwoChannels:
    def _frames(self):
        """A dim base plus an EXTENDED bright region in the overlay.

        The bright region spans ~6% of the frame on purpose. A single-pixel punctum sits above the
        99.5th percentile and is clipped away by the display stretch entirely — see
        ``test_sparser_than_the_stretch_window_is_clipped_away``, which pins that behavior. Real
        fluorescence covers an embryo, not one pixel, so this fixture matches the actual case.

        The base is kept dim (max 0.3 after stretch) so an additive tint has headroom; a bright base
        legitimately saturates to white, which is correct compositing, not a tint failure.
        """
        base = np.tile(np.arange(16, dtype=np.uint16) * 10, (16, 1))
        overlay = np.zeros((16, 16), dtype=np.uint16)
        overlay[6:10, 6:10] = 5000
        return base, overlay

    def test_returns_rgb_uint8(self):
        base, overlay = self._frames()
        out = composite_two_channels(
            base, overlay, base_channel_id="BF", overlay_channel_id="RFP"
        )
        assert out.shape == (16, 16, 3)
        assert out.dtype == np.uint8

    def test_overlay_signal_carries_its_channel_tint(self):
        """Where fluorescence is bright, the composite must lean toward that channel's color."""
        base, overlay = self._frames()
        out = composite_two_channels(
            base, overlay, base_channel_id="BF", overlay_channel_id="RFP"
        ).astype(int)
        r, g, b = out[8, 8]
        assert r > g and r > b, "RFP punctum should render red-dominant"

    def test_base_alone_stays_neutral_gray(self):
        """With no overlay signal, the output must be pure grayscale — no color cast."""
        base, _ = self._frames()
        out = composite_two_channels(
            base, np.zeros_like(base), base_channel_id="BF", overlay_channel_id="RFP"
        ).astype(int)
        assert np.array_equal(out[:, :, 0], out[:, :, 1])
        assert np.array_equal(out[:, :, 1], out[:, :, 2])

    def test_base_gain_dims_the_base(self):
        base, overlay = self._frames()
        full = composite_two_channels(
            base, overlay, base_channel_id="BF", overlay_channel_id="RFP"
        ).astype(int)
        dimmed = composite_two_channels(
            base, overlay, base_channel_id="BF", overlay_channel_id="RFP", base_gain=0.4
        ).astype(int)
        assert dimmed.sum() < full.sum()

    def test_a_bright_base_saturates_toward_white(self):
        """Not a bug: where the base is already bright, adding a tint clips to white.

        Documented because it looks like "the tint stopped working" — the fix is base_gain, which is
        why that knob exists.
        """
        # A gradient whose RIGHT side is bright after stretching (a uniform base would stretch to
        # 0.0, since the stretch is relative to the image's own percentiles, not to dtype max).
        # Sample at column 14, where the base is near 1.0 and the overlay also has signal.
        bright_base = np.tile(np.linspace(0, 60000, 16, dtype=np.uint16), (16, 1))
        overlay = np.zeros((16, 16), dtype=np.uint16)
        overlay[6:10, 12:16] = 5000
        y, x = 8, 14

        saturated = composite_two_channels(
            bright_base, overlay, base_channel_id="BF", overlay_channel_id="RFP"
        ).astype(int)
        assert saturated[y, x, 0] == saturated[y, x, 1] == 255, "bright base clips to white"

        recovered = composite_two_channels(
            bright_base, overlay, base_channel_id="BF", overlay_channel_id="RFP", base_gain=0.2
        ).astype(int)
        r, g, b = recovered[y, x]
        assert r > g and r > b, "dimming the base restores the overlay tint"

    def test_sparser_than_the_stretch_window_is_clipped_away(self):
        """A signal rarer than the stretch's top percentile is DISPLAYED as nothing.

        Real behavior worth knowing before trusting a composite: a handful of isolated bright pixels
        sit above the 99.5th percentile, so the stretch clips them and the overlay contributes zero.
        The stored data still holds them — this is a display limit, not data loss.
        """
        overlay = np.zeros((64, 64), dtype=np.uint16)
        overlay[32, 32] = 60000  # 1 pixel in 4096, far above p99.5
        assert stretch_to_unit(overlay)[32, 32] == 0.0
        # Widening the window to include it recovers the signal.
        assert stretch_to_unit(overlay, (0.0, 100.0))[32, 32] == pytest.approx(1.0)

    def test_mismatched_shapes_fail_loud(self):
        with pytest.raises(ValueError, match="same write policy|share a shape"):
            composite_two_channels(
                np.zeros((16, 16), dtype=np.uint16),
                np.zeros((8, 8), dtype=np.uint16),
                base_channel_id="BF",
                overlay_channel_id="RFP",
            )

    def test_unknown_channel_id_fails_loud(self):
        base, overlay = self._frames()
        with pytest.raises(ValueError, match="not in the canonical vocabulary"):
            composite_two_channels(
                base, overlay, base_channel_id="BF", overlay_channel_id="tdtomato"
            )

    def test_generic_over_channels_not_bf_rfp_specific(self):
        """Any vocabulary pair must composite — GFP will want exactly this view."""
        base, overlay = self._frames()
        out = composite_two_channels(
            base, overlay, base_channel_id="BF", overlay_channel_id="GFP"
        ).astype(int)
        r, g, b = out[8, 8]
        assert g > r and g > b, "GFP punctum should render green-dominant"


class TestTintSingleChannel:
    def test_tints_toward_the_channel_color(self):
        img = np.zeros((8, 8), dtype=np.uint16)
        img[4, 4] = 4000
        out = tint_single_channel(img, channel_id="RFP").astype(int)
        r, g, b = out[4, 4]
        assert r > g and r > b
