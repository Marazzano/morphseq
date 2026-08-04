"""Snip product-key grammar and the recipe registry.

The load-bearing test is `test_two_bf_products_do_not_collide`: it is the reason the key carries the
whole source product key rather than just the channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.acquisition.image_materialization.image_product_keys import (
    build_image_product_key,
)
from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
    DEFAULT_BF_SNIP_PRODUCT_KEY,
    build_snip_product_key,
    channel_id_for_snip_product_key,
    parse_snip_product_key,
    snip_product_key_for_row,
)
from data_pipeline.object_extraction.snip_processing.snip_recipes import (
    CLAHE_BLEND,
    NO_CHANGE,
    SUPPORTED_SNIP_RECIPES,
    SnipRecipeError,
    assert_recipe_is_quantitative,
    assert_source_dtype_is_acceptable,
    recipe_contract,
)

BF_PROJECTION = "BF__projection__focus_stack"
BF_ZSTACK = "BF__z_stack"
RFP_MAX = "RFP__projection__max"


class TestKeyGrammar:
    def test_two_bf_products_do_not_collide(self):
        # THE REASON THE SOURCE PRODUCT IS IN THE KEY. build_image_product_key emits both of these
        # and they share channel_id="BF" -- but they are different pixels at different calibration
        # (BF__z_stack ships downsample_factor 4 while the projection is native, a divergence
        # already live in the shipped config). Under a {channel}__{recipe} grammar both would
        # collapse to "BF__no_change": one key, two materially different products.
        assert build_image_product_key(channel_id="BF", image_product_type="z_stack") == BF_ZSTACK
        a = build_snip_product_key(source_image_product_key=BF_PROJECTION, snip_recipe=NO_CHANGE)
        b = build_snip_product_key(source_image_product_key=BF_ZSTACK, snip_recipe=NO_CHANGE)
        assert a != b
        assert channel_id_for_snip_product_key(a) == channel_id_for_snip_product_key(b) == "BF"

    @pytest.mark.parametrize(
        "source,recipe,expected",
        [
            (BF_PROJECTION, CLAHE_BLEND, "BF__projection__focus_stack__clahe_blend"),
            (RFP_MAX, NO_CHANGE, "RFP__projection__max__no_change"),
            (BF_ZSTACK, NO_CHANGE, "BF__z_stack__no_change"),
        ],
    )
    def test_round_trip(self, source, recipe, expected):
        key = build_snip_product_key(source_image_product_key=source, snip_recipe=recipe)
        assert key == expected
        assert parse_snip_product_key(key) == (source, recipe)

    def test_parse_splits_from_the_right(self):
        # It must: the source key contains '__' itself, so a left split would tear it apart and
        # yield source="BF" for a three-field source.
        source, recipe = parse_snip_product_key("BF__projection__focus_stack__clahe_blend")
        assert source == BF_PROJECTION and recipe == CLAHE_BLEND

    def test_a_recipe_containing_the_separator_is_rejected(self):
        # Otherwise the key becomes ambiguous to parse and the round-trip silently lies.
        with pytest.raises(ValueError, match="may not contain"):
            build_snip_product_key(source_image_product_key=RFP_MAX, snip_recipe="no__change")

    def test_a_non_canonical_source_fails_where_it_can_be_named(self):
        with pytest.raises(ValueError, match="not canonical|cannot parse"):
            build_snip_product_key(source_image_product_key="BF", snip_recipe=NO_CHANGE)

    def test_a_structurally_invalid_key_is_rejected(self):
        # A source that is not itself a canonical image product key (too few fields) must fail here,
        # where the message can name it, rather than downstream where it looks like a missing file.
        with pytest.raises(ValueError, match="not canonical|cannot parse"):
            parse_snip_product_key("BF__no_change")

    def test_case_is_preserved_not_normalized(self):
        # The UPSTREAM grammar is case-preserving: parse_image_product_key accepts
        # "BF__projection__FOCUS_STACK" as a legal (if nonexistent) method. This grammar must not
        # invent a case rule the source keys do not share -- diverging normalization between the two
        # would make a snip key fail to match the product it names.
        key = build_snip_product_key(
            source_image_product_key="BF__projection__FOCUS_STACK", snip_recipe=NO_CHANGE
        )
        assert parse_snip_product_key(key) == ("BF__projection__FOCUS_STACK", NO_CHANGE)

    def test_the_default_bf_key_is_canonical(self):
        # A typo'd constant would silently make the compatibility resolver match nothing.
        assert parse_snip_product_key(DEFAULT_BF_SNIP_PRODUCT_KEY) == (BF_PROJECTION, CLAHE_BLEND)


class TestRowNormalization:
    @pytest.mark.parametrize("absent", [None, float("nan"), "", "nan", "<NA>", "  "])
    def test_csv_round_trip_nulls_read_as_absent(self, absent):
        # These all mean ABSENT depending on writer and inferred dtype; returning them as strings
        # would produce a product key that looks real and matches nothing.
        assert snip_product_key_for_row({"snip_product_key": absent}) is None

    def test_a_real_key_survives(self):
        assert snip_product_key_for_row({"snip_product_key": RFP_MAX + "__no_change"}) == (
            "RFP__projection__max__no_change"
        )

    def test_a_row_without_the_column_is_absent_not_an_error(self):
        assert snip_product_key_for_row({}) is None


class TestRecipeContracts:
    def test_vocabulary_is_derived_from_the_contract_table(self):
        # One source of truth: a recipe with a contract is supported by construction, so "declared"
        # and "implemented" cannot drift.
        assert SUPPORTED_SNIP_RECIPES == {NO_CHANGE, CLAHE_BLEND}

    def test_unknown_recipe_names_what_is_available(self):
        with pytest.raises(SnipRecipeError, match="unknown snip_recipe"):
            recipe_contract("asinh_scaled")

    def test_clahe_blend_requires_uint8(self):
        # It documents uint8 in / uint8 out, and the noise blend is tuned for 8-bit BF.
        with pytest.raises(SnipRecipeError, match="requires dtype"):
            assert_source_dtype_is_acceptable(
                np.zeros((4, 4), np.uint16), snip_recipe=CLAHE_BLEND, source="frame.png"
            )

    def test_no_change_accepts_any_dtype(self):
        for dtype in (np.uint8, np.uint16, np.float32):
            assert_source_dtype_is_acceptable(
                np.zeros((4, 4), dtype), snip_recipe=NO_CHANGE, source="frame.png"
            )

    def test_the_refusal_points_at_the_dtype_preserving_route(self):
        # A message that only says "no" leaves the caller stuck; this one names the way forward.
        with pytest.raises(SnipRecipeError, match="no_change"):
            assert_source_dtype_is_acceptable(
                np.zeros((4, 4), np.uint16), snip_recipe=CLAHE_BLEND, source="frame.png"
            )

    def test_measuring_a_photometric_product_is_blocked(self):
        # Intensity off a CLAHE snip is plausible and meaningless: local adaptive equalization
        # destroys absolute intensity while leaving an image that looks fine.
        with pytest.raises(SnipRecipeError, match="photometric"):
            assert_recipe_is_quantitative(CLAHE_BLEND)

    def test_no_change_is_measurable(self):
        assert_recipe_is_quantitative(NO_CHANGE)


class TestRenderDispatch:
    """The recipe decides photometry; geometry already happened."""

    def test_no_change_preserves_uint16_values_and_dtype(self):
        # THE POINT OF P2. A 12000-DN pixel must read back as 12000, not be compressed into 8 bits.
        from data_pipeline.object_extraction.snip_processing.snip_recipes import render_snip

        source = np.array([[3000, 12000], [0, 65535]], dtype=np.uint16)
        out = render_snip(source, snip_recipe=NO_CHANGE)
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out, source)

    def test_no_change_does_not_copy_or_cast(self):
        # A cast here would be exactly the silent dtype conversion the recipe promises not to do,
        # and it would be invisible in a diff.
        from data_pipeline.object_extraction.snip_processing.snip_recipes import render_snip

        source = np.zeros((4, 4), dtype=np.uint16)
        assert render_snip(source, snip_recipe=NO_CHANGE) is source

    def test_clahe_blend_changes_pixels(self):
        # The complement: if this returned its input unchanged, the dispatcher would be wired to
        # the wrong renderer and every BF snip would silently lose its augmentation.
        from data_pipeline.object_extraction.snip_processing.snip_recipes import render_snip

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, (32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        out = render_snip(
            image, snip_recipe=CLAHE_BLEND, mask=mask,
            background_mean=12.0, background_std=3.0,
            blend_radius_um=20.0, pixel_size_um=7.8,
        )
        assert out.shape == image.shape
        assert not np.array_equal(out, image)

    def test_an_unknown_recipe_cannot_render(self):
        from data_pipeline.object_extraction.snip_processing.snip_recipes import render_snip

        with pytest.raises(SnipRecipeError, match="unknown snip_recipe"):
            render_snip(np.zeros((2, 2), np.uint8), snip_recipe="asinh_scaled")

    def test_every_contract_has_a_renderer(self):
        # Pinned as a test as well as an import-time assert: a recipe with a contract but no
        # renderer would pass plan validation and fail while rendering; a renderer with no contract
        # would run without its dtype precondition ever being checked, which is the hazard-zero
        # failure mode exactly.
        from data_pipeline.object_extraction.snip_processing.snip_recipes import (
            _SNIP_RECIPE_RENDERERS,
            SNIP_RECIPE_CONTRACTS,
        )

        assert set(_SNIP_RECIPE_RENDERERS) == set(SNIP_RECIPE_CONTRACTS)
