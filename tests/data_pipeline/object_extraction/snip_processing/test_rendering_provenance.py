from __future__ import annotations

import json

from data_pipeline.object_extraction.snip_processing.provenance import (
    build_rendering_contract,
    merge_rendering_sidecars,
    update_rendering_sidecar,
)


def _contract(product_key: str, source_key: str, recipe: str, *, dtype: str):
    return build_rendering_contract(
        snip_product_key=product_key,
        source_image_product_key=source_key,
        snip_recipe=recipe,
        target_micrometers_per_pixel=6.5,
        blend_radius_micrometers=75.0,
        blend_radius_applied=recipe == "clahe_blend",
        frame_shape=(576, 256),
        mask_contract={"artifact": "frame_masks"},
        orientation_contract={"policy": "pca_major_axis_yolk_down"},
        clahe_enabled=recipe == "clahe_blend",
        background_contract={"applied_to_pixels": recipe == "clahe_blend"},
        resampling_contract={"source_calibration_column": "image_micrometers_per_pixel"},
        file_encoding={"processed_snip": {"format": "PNG", "dtype": dtype}},
    )


def test_sidecar_keeps_distinct_contracts_per_snip_product(tmp_path):
    bf_key = "BF__projection__focus_stack__clahe_blend"
    rfp_key = "RFP__projection__max__no_change"
    path = tmp_path / "snip_inventory.csv.provenance.json"

    update_rendering_sidecar(
        path,
        fixed_contract=_contract(
            bf_key, "BF__projection__focus_stack", "clahe_blend", dtype="uint8"
        ),
        snip_product_key=bf_key,
        well_observations={
            "20250912_B01": {
                "mask_sources_and_versions": [{"source": "sam2", "version": "v1"}],
                "source_micrometers_per_pixel_values": [2.17],
                "pixel_dtypes": ["uint8"],
            }
        },
    )
    update_rendering_sidecar(
        path,
        fixed_contract=_contract(
            rfp_key, "RFP__projection__max", "no_change", dtype="source_dtype"
        ),
        snip_product_key=rfp_key,
        well_observations={
            "20250912_B01": {
                "mask_sources_and_versions": [{"source": "sam2", "version": "v1"}],
                "source_micrometers_per_pixel_values": [3.25],
                "pixel_dtypes": ["uint16"],
            }
        },
    )

    document = json.loads(path.read_text())
    assert set(document["products"]) == {bf_key, rfp_key}
    assert document["products"][bf_key]["rendering"]["clahe"]["enabled"] is True
    assert document["products"][rfp_key]["rendering"]["clahe"]["enabled"] is False
    assert document["products"][rfp_key]["observations"]["pixel_dtypes"] == ["uint16"]


def test_merge_sidecars_unions_wells_without_collapsing_product_dimension(tmp_path):
    key = "BF__projection__focus_stack__clahe_blend"
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, well_id, scale in zip(
        paths, ("20250912_B01", "20250912_B02"), (2.17, 3.25)
    ):
        update_rendering_sidecar(
            path,
            fixed_contract=_contract(
                key, "BF__projection__focus_stack", "clahe_blend", dtype="uint8"
            ),
            snip_product_key=key,
            well_observations={
                well_id: {
                    "mask_sources_and_versions": [],
                    "source_micrometers_per_pixel_values": [scale],
                    "pixel_dtypes": ["uint8"],
                }
            },
        )

    merged = tmp_path / "merged.json"
    merge_rendering_sidecars(paths, merged)
    product = json.loads(merged.read_text())["products"][key]
    assert set(product["observations"]["wells"]) == {"20250912_B01", "20250912_B02"}
    assert product["observations"]["source_micrometers_per_pixel_values"] == [2.17, 3.25]
