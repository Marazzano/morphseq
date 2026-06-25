"""Tests for resolved image-product plan JSONs."""

import pytest

from data_pipeline.image_materialization.resolved_product_plans import (
    load_resolved_product_plan_for_well,
    resolve_requested_image_products_for_well,
    write_resolved_product_plan_for_well,
)

EXP = "20250912"
WELL = "20250912_B01"


def test_default_config_resolves_projection_product_key():
    plans = resolve_requested_image_products_for_well(
        experiment_id=EXP,
        well_id=WELL,
        scope_name="yx1",
        config=None,
    )
    assert [p.product_key for p in plans] == ["BF__projection__focus_stack"]
    assert plans[0].product.image_product_type == "projection"


def test_scope_name_is_normalized_to_lowercase():
    plans = resolve_requested_image_products_for_well(
        experiment_id=EXP,
        well_id=WELL,
        scope_name="YX1",
        config=None,
    )
    assert plans[0].scope_name == "yx1"


def test_z_stack_config_writes_and_loads_selected_product_plan(tmp_path):
    config = {
        "image_materialization": {
            "products": [
                {"channel_id": "BF", "image_product_type": "z_stack"},
            ]
        }
    }
    out = tmp_path / "BF__z_stack_resolved_product_plan.json"

    written = write_resolved_product_plan_for_well(
        experiment_id=EXP,
        well_id=WELL,
        scope_name="yx1",
        config=config,
        product_key="BF__z_stack",
        output_json=out,
    )
    loaded = load_resolved_product_plan_for_well(
        out,
        expected_experiment_id=EXP,
        expected_well_id=WELL,
        expected_product_key="BF__z_stack",
    )

    assert written.product_key == "BF__z_stack"
    assert loaded.product.image_product_type == "z_stack"
    assert loaded.product.projection_method is None
    assert loaded.product.xy_composition == "identity"


def test_unknown_requested_product_key_fails_loud(tmp_path):
    with pytest.raises(ValueError, match="Active product keys"):
        write_resolved_product_plan_for_well(
            experiment_id=EXP,
            well_id=WELL,
            scope_name="yx1",
            config=None,
            product_key="BF__z_stack",
            output_json=tmp_path / "out.json",
        )


def test_load_rejects_wrong_expected_well(tmp_path):
    out = tmp_path / "plan.json"
    write_resolved_product_plan_for_well(
        experiment_id=EXP,
        well_id=WELL,
        scope_name="yx1",
        config=None,
        product_key="BF__projection__focus_stack",
        output_json=out,
    )

    with pytest.raises(ValueError, match="well_id"):
        load_resolved_product_plan_for_well(out, expected_well_id="20250912_C01")
