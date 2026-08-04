"""Loading the snip product plan from nested config.

The structural claim under test: because a snip nests under the image product it crops, a snip that
disagrees with its source is UNREPRESENTABLE rather than merely rejected.
"""

from __future__ import annotations

import pytest

from data_pipeline.object_extraction.snip_processing.snip_product_keys import (
    DEFAULT_BF_SNIP_PRODUCT_KEY,
)
from data_pipeline.object_extraction.snip_processing.snip_product_plan import (
    UnsupportedSnipRequest,
    load_snip_product_plan,
)


def _config(*products):
    return {"image_materialization": {"products": list(products)}}


BF = {
    "channel_id": "BF",
    "image_product_type": "projection",
    "projection_method": "focus_stack",
}
RFP = {"channel_id": "RFP", "image_product_type": "projection", "projection_method": "max"}


class TestDefaults:
    def test_no_config_renders_todays_bf_product(self):
        # A pre-migration config must keep producing exactly what it produced before this seam
        # existed -- otherwise the seam is a behavior change disguised as a refactor.
        assert load_snip_product_plan(None).snip_product_keys == (DEFAULT_BF_SNIP_PRODUCT_KEY,)

    def test_products_without_snips_render_the_default(self):
        # Config that predates `snips:` entirely.
        plan = load_snip_product_plan(_config(BF, RFP))
        assert plan.snip_product_keys == (DEFAULT_BF_SNIP_PRODUCT_KEY,)

    def test_absent_snips_on_one_product_means_none_of_that_product(self):
        # ABSENT MEANS NONE HERE, not "default". At the image-product level absent means "use the
        # default set" because there is no parent to inherit from; here the parent is explicit, so
        # defaulting would render products nobody asked for.
        plan = load_snip_product_plan(
            _config(BF, {**RFP, "snips": [{"snip_recipe": "no_change"}]})
        )
        assert plan.snip_product_keys == ("RFP__projection__max__no_change",)


class TestNesting:
    def test_the_source_is_the_parent(self):
        plan = load_snip_product_plan(
            _config(
                {**BF, "snips": [{"snip_recipe": "clahe_blend"}]},
                {**RFP, "snips": [{"snip_recipe": "no_change"}]},
            )
        )
        assert plan.snip_product_keys == (
            "BF__projection__focus_stack__clahe_blend",
            "RFP__projection__max__no_change",
        )

    def test_one_source_may_have_several_recipes(self):
        plan = load_snip_product_plan(
            _config({**RFP, "snips": [{"snip_recipe": "no_change"}, {"snip_recipe": "clahe_blend"}]})
        )
        assert plan.snip_product_keys == (
            "RFP__projection__max__no_change",
            "RFP__projection__max__clahe_blend",
        )

    def test_restating_the_source_on_a_snip_is_rejected(self):
        # The mismatch a flat config would have allowed. Rejecting the KEY rather than validating
        # its VALUE is what makes the disagreement unrepresentable: there is no correct way to write
        # a channel here, so there is no way to write a wrong one.
        for stray in ("channel_id", "source_image_product_key"):
            with pytest.raises(UnsupportedSnipRequest, match="may not set"):
                load_snip_product_plan(
                    _config({**RFP, "snips": [{"snip_recipe": "no_change", stray: "BF"}]})
                )

    def test_a_snip_on_an_invalid_source_fails_where_it_can_be_named(self):
        # A snip cannot be rendered from a source that will not be materialized.
        broken = {"channel_id": "BF", "image_product_type": "projection"}  # method missing
        with pytest.raises(UnsupportedSnipRequest, match="not a valid image product"):
            load_snip_product_plan(_config({**broken, "snips": [{"snip_recipe": "no_change"}]}))


class TestThreeWayRule:
    def test_an_explicit_empty_snips_list_is_an_error(self):
        # Same reasoning as products: [] -- an explicit empty list is a mistake, not a request for
        # the default. Silently substituting the default would surprise someone who meant "none".
        with pytest.raises(UnsupportedSnipRequest, match="provided but empty"):
            load_snip_product_plan(_config({**RFP, "snips": []}))

    def test_snips_must_be_a_list(self):
        with pytest.raises(UnsupportedSnipRequest, match="must be a list"):
            load_snip_product_plan(_config({**RFP, "snips": {"snip_recipe": "no_change"}}))

    def test_products_must_be_a_list(self):
        with pytest.raises(UnsupportedSnipRequest, match="must be a list"):
            load_snip_product_plan({"image_materialization": {"products": {"channel_id": "BF"}}})


class TestValidation:
    def test_an_unimplemented_recipe_is_rejected_at_load(self):
        # Fail at plan time with the supported set named, rather than a layer deeper while
        # rendering -- the failure mode the materialization module's grammar/capability split exists
        # to prevent.
        with pytest.raises(UnsupportedSnipRequest, match="not supported"):
            load_snip_product_plan(_config({**RFP, "snips": [{"snip_recipe": "asinh_scaled"}]}))

    def test_a_missing_recipe_is_rejected(self):
        with pytest.raises(UnsupportedSnipRequest, match="not supported"):
            load_snip_product_plan(_config({**RFP, "snips": [{}]}))

    def test_duplicate_products_are_rejected(self):
        # Each key names one output path, so two requests for it would put two jobs on the same
        # files -- which Snakemake cannot model and which silently races.
        with pytest.raises(UnsupportedSnipRequest, match="duplicate snip product"):
            load_snip_product_plan(
                _config({**RFP, "snips": [{"snip_recipe": "no_change"}] * 2})
            )

    def test_the_same_recipe_on_two_sources_is_not_a_duplicate(self):
        # BF__z_stack__no_change and RFP__projection__max__no_change share a recipe and nothing
        # else. This is the collision the channel-only key grammar would have created.
        plan = load_snip_product_plan(
            _config(
                {
                    "channel_id": "BF",
                    "image_product_type": "z_stack",
                    "snips": [{"snip_recipe": "no_change"}],
                },
                {**RFP, "snips": [{"snip_recipe": "no_change"}]},
            )
        )
        assert plan.snip_product_keys == (
            "BF__z_stack__no_change",
            "RFP__projection__max__no_change",
        )
