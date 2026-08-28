"""Small executable observation/asset fixture for manifest contract v2.0."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.data.manifest_types import (
    CohortReport,
    ManifestPolicy,
    PipelineManifestResult,
    QCPolicy,
    SchemaReport,
    SplitPolicy,
)


BF_PRODUCT = "BF__projection__focus_stack__clahe_blend"
RFP_PRODUCT = "RFP__projection__max__no_change"
BF_Z_PRODUCT = "BF__z_stack__no_change"
SOURCE_PRODUCTS = {
    BF_PRODUCT: "BF__projection__focus_stack",
    RFP_PRODUCT: "RFP__projection__max",
    BF_Z_PRODUCT: "BF__z_stack",
}


def synthetic_manifest_v2(asset_root: Path | None = None) -> PipelineManifestResult:
    """Return deterministic tables with projection siblings and ordered z planes.

    ``asset_root`` only anchors declared paths; this fixture does not create pixels.
    The deliberately non-grammatical IDs help consumers prove they treat IDs as
    opaque values rather than parsing or reconstructing them.
    """

    root = Path("/synthetic-assets") if asset_root is None else Path(asset_root)
    observation_rows = [
        _observation("snip::alpha", "animal::alpha", "experiment::one", "well::one", 0, "train"),
        _observation("snip::beta", "animal::beta", "experiment::one", "well::two", 1, "eval"),
        _observation("snip::gamma", "animal::gamma", "experiment::two", "well::three", 0, "test"),
    ]
    observation_table = pd.DataFrame(observation_rows).reset_index(drop=True)

    asset_rows = [
        _asset(root, "snip::alpha", BF_PRODUCT, pd.NA, 0),
        _asset(root, "snip::alpha", RFP_PRODUCT, pd.NA, 1),
        _asset(root, "snip::alpha", BF_Z_PRODUCT, 0, 2),
        _asset(root, "snip::alpha", BF_Z_PRODUCT, 1, 3),
        _asset(root, "snip::alpha", BF_Z_PRODUCT, 2, 4),
        _asset(root, "snip::beta", BF_PRODUCT, pd.NA, 5),
        _asset(root, "snip::gamma", BF_PRODUCT, pd.NA, 6),
    ]
    asset_table = pd.DataFrame(asset_rows).reset_index(drop=True)
    asset_table["z_index"] = pd.array(asset_table["z_index"], dtype="Int64")

    selected_assets = asset_table.loc[
        asset_table["snip_product_key"].eq(BF_PRODUCT) & asset_table["z_index"].isna()
    ].copy()
    resolved_sample_table = observation_table.merge(
        selected_assets,
        on="snip_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_asset"),
        sort=False,
    ).reset_index(drop=True)

    split_assignments = observation_table[["physical_embryo_id", "split"]].copy()
    policy = ManifestPolicy(
        name="test_only_manifest_v2_fixture",
        version="1",
        experiment_ids=("experiment::one", "experiment::two"),
        allowed_product_keys=(BF_PRODUCT, RFP_PRODUCT, BF_Z_PRODUCT),
        selected_product_key=BF_PRODUCT,
        qc=QCPolicy(name="fixture_evaluated_pass", version="1"),
        splits=SplitPolicy(
            tolerance=1.0,
            explicit_test_experiments=("experiment::two",),
        ),
    )
    return PipelineManifestResult(
        observation_table=observation_table,
        asset_table=asset_table,
        resolved_sample_table=resolved_sample_table,
        source_inventory=(),
        schema_report=SchemaReport(),
        cohort_report=CohortReport(),
        split_assignments=split_assignments.reset_index(drop=True),
        policy=policy,
    )


def _observation(
    snip_id: str,
    physical_embryo_id: str,
    experiment_id: str,
    well_id: str,
    time_index: int,
    split: str,
) -> dict[str, object]:
    return {
        "snip_id": snip_id,
        "embryo_id": f"explicit-embryo::{snip_id}",
        "physical_embryo_id": physical_embryo_id,
        "experiment_id": experiment_id,
        "well_id": well_id,
        "image_id": f"explicit-image::{snip_id}",
        "time_index": time_index,
        "channel_id": "BF",
        "elapsed_time_s": float(time_index * 900),
        "elapsed_time_status": "available",
        "incubation_temperature_c": 28.5,
        "temperature_status": "available",
        "start_age_hpf": 24.0,
        "start_age_source": "plate_metadata",
        "genotype": "source-spelling/control",
        "medium": "E3",
        "strain": "AB",
        "chem_perturbation": pd.NA,
        "predicted_stage_hpf": 24.0 + time_index * 0.25,
        "stage_status": "predicted",
        "stage_model_version": "fixture-clock-v1",
        "use_snip": True,
        "qc_fail_reasons": "",
        "sa_outlier_flag": False,
        "sa_qc_applicability": "exclusion",
        "qc_status": "evaluated",
        "qc_schema_version": "fixture-qc-v1",
        "split": split,
    }


def _asset(
    root: Path,
    snip_id: str,
    product_key: str,
    z_index: object,
    source_order: int,
) -> dict[str, object]:
    token = f"asset-{source_order:02d}"
    return {
        "snip_id": snip_id,
        "snip_product_key": product_key,
        "z_index": z_index,
        "processed_snip_path": str(root / f"{token}.png"),
        "embryo_mask_snip_path": str(root / f"{token}-mask.png"),
        "is_valid_snip": True,
        "error_message": pd.NA,
        "source_image_product_key": SOURCE_PRODUCTS[product_key],
        "image_path": str(root / f"{token}-source.png"),
        "snip_transform_id": "fixture-transform-v1",
        "output_grid_id": "fixture-grid-576x256",
        "source_micrometers_per_pixel": 6.5,
        "snip_micrometers_per_pixel": 6.5,
        "source_height_px": 576,
        "source_width_px": 256,
        "output_height_px": 576,
        "output_width_px": 256,
        "orientation_policy": "fixture-explicit",
        "orientation_source": "fixture",
        "centering": "continuous",
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "realized_scale_y": 1.0,
        "realized_scale_x": 1.0,
        "pixel_dtype": "uint8",
        "resolved_transform_chain_json": "[]",
        "source_order": source_order,
    }
