"""Writers for small declared pipeline-source fixtures used by A1 tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core.data.pipeline_manifest import ExperimentSourcePaths


BF_PRODUCT = "BF__projection__focus_stack__clahe_blend"
RFP_PRODUCT = "RFP__projection__max__no_change"


def write_pipeline_source_fixture(
    root: Path,
    experiment_id: str,
    *,
    row_count: int = 4,
    identity_offset: int = 0,
    include_stage: bool = True,
    include_qc: bool = True,
    collection: bool = False,
    booleans_as_strings: bool = False,
) -> ExperimentSourcePaths:
    source_root = Path(root) / experiment_id
    source_root.mkdir(parents=True)

    identity_rows = [
        _identity_row(experiment_id, identity_offset + index) for index in range(row_count)
    ]
    snip_rows: list[dict[str, object]] = []
    for local_index, identity in enumerate(identity_rows):
        source_order = identity_offset * 2 + local_index * 2
        snip_rows.append(_snip_row(source_root, identity, BF_PRODUCT, source_order))
        if local_index == 0:
            snip_rows.append(_snip_row(source_root, identity, RFP_PRODUCT, source_order + 1))
    snips = pd.DataFrame(snip_rows)

    frame_rows: list[dict[str, object]] = []
    for local_index, identity in enumerate(identity_rows):
        index = identity_offset + local_index
        for channel, method in (("BF", "focus_stack"), ("RFP", "max")):
            frame_rows.append(
                {
                    "experiment_id": experiment_id,
                    "well_id": identity["well_id"],
                    "image_id": f"frame-image-{index}-{channel}",
                    "channel_id": channel,
                    "time_index": identity["time_index"],
                    "z_index": pd.NA,
                    "image_product_type": "projection",
                    "projection_method": method,
                    "elapsed_time_s": float(index * 600),
                }
            )
    frames = pd.DataFrame(frame_rows)

    plate_rows = []
    for local_index, identity in enumerate(identity_rows):
        index = identity_offset + local_index
        plate_rows.append(
            {
                "experiment_id": experiment_id,
                "well_id": identity["well_id"],
                "well_index": f"W{index:03d}",
                "genotype": f"source/genotype/{index % 2}",
                "start_age_hpf": 20.0 + index,
                "temperature": 27.5 + index * 0.1,
                "medium": "E3",
                "strain": "AB",
                "chem_perturbation": pd.NA,
            }
        )
    plate = pd.DataFrame(plate_rows)

    stage_rows = []
    qc_rows = []
    for local_index, identity in enumerate(identity_rows):
        index = identity_offset + local_index
        stage_rows.append(
            {
                **identity,
                "predicted_stage_hpf": 20.0 + index + index / 6,
                "model_version": "fixture-stage-v1",
                "stage_prediction_status": "predicted",
            }
        )
        true_value: object = "True" if booleans_as_strings else True
        false_value: object = "False" if booleans_as_strings else False
        qc_rows.append(
            {
                "experiment_id": identity["experiment_id"],
                "well_id": identity["well_id"],
                "physical_embryo_id": identity["physical_embryo_id"],
                "embryo_id": identity["embryo_id"],
                "snip_id": identity["snip_id"],
                "use_snip": true_value,
                "qc_fail_reasons": "",
                "sa_outlier_flag": false_value,
                "surface_area_qc_applicability": "exclusion",
                "focus_flag": false_value,
                "focus_qc_applicability": "exclusion",
            }
        )

    snip_path = source_root / "snip_inventory.csv"
    frame_path = source_root / "frame_inventory.csv"
    plate_path = source_root / "plate_metadata.csv"
    stage_path = source_root / "stage_predictions.csv"
    qc_path = source_root / "snip_qc.parquet"
    collection_path = source_root / "collection_provenance.json"
    acquisition_path = source_root / "acquisition_inventory.csv"
    snips.to_csv(snip_path, index=False)
    frames.to_csv(frame_path, index=False)
    plate.to_csv(plate_path, index=False)
    if include_stage:
        pd.DataFrame(stage_rows).to_csv(stage_path, index=False)
    if include_qc:
        pd.DataFrame(qc_rows).to_parquet(qc_path, index=False)

    if collection:
        age_map = {str(index): 30 + index for index in range(row_count)}
        sources = [
            {
                "file": f"declared-source-{index}",
                "raw_path": str(source_root / f"declared-source-{index}"),
                "declared_hpf": 30 + index,
                "source_ordinal": index,
                "time_index": index,
            }
            for index in range(row_count)
        ]
        provenance = {
            "experiment_id": experiment_id,
            "is_collection": True,
            "sources": sources,
            "start_age_by_source_ordinal": age_map,
            "start_age_by_time_index": age_map,
        }
        acquisition = pd.DataFrame(
            {
                "well_id": [row["well_id"] for row in identity_rows],
                "time_index": [row["time_index"] for row in identity_rows],
                "source_ordinal": list(range(row_count)),
            }
        )
        acquisition.to_csv(acquisition_path, index=False)
    else:
        provenance = {
            "experiment_id": experiment_id,
            "is_collection": False,
            "sources": [
                {
                    "file": "declared-single-source",
                    "raw_path": str(source_root / "declared-single-source"),
                    "declared_hpf": None,
                    "source_ordinal": 0,
                    "time_index": 0,
                }
            ],
            "start_age_by_source_ordinal": {},
            "start_age_by_time_index": {},
        }
    collection_path.write_text(json.dumps(provenance), encoding="utf-8")

    paths = {
        "snip_inventory": snip_path,
        "frame_inventory": frame_path,
        "plate_metadata": plate_path,
        "stage_predictions": stage_path if include_stage else None,
        "snip_qc": qc_path if include_qc else None,
        "collection_provenance": collection_path,
        "acquisition_inventory": acquisition_path if collection else None,
    }
    return ExperimentSourcePaths(
        experiment_id=experiment_id,
        collection_provenance_applicable=collection,
        authorities={name: "tests.core.fixtures.pipeline_source_tables" for name in paths},
        **paths,
    )


def _identity_row(experiment_id: str, index: int) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "well_id": f"well-opaque-{index}",
        "physical_embryo_id": f"animal-opaque-{index}",
        "embryo_id": f"embryo-opaque-{index}",
        "snip_id": f"observation-opaque-{index}",
        "image_id": f"origin-image-opaque-{index}",
        "time_index": index,
        "channel_id": "BF",
    }


def _snip_row(
    source_root: Path,
    identity: dict[str, object],
    product_key: str,
    source_order: int,
) -> dict[str, object]:
    asset_name = f"asset-{source_order:03d}"
    return {
        **identity,
        "processed_snip_path": str(source_root / f"{asset_name}.png"),
        "snip_product_key": product_key,
        "is_valid_snip": True,
        "error_message": pd.NA,
        "image_path": str(source_root / f"{asset_name}-source.png"),
        "embryo_mask_snip_path": str(source_root / f"{asset_name}-mask.png"),
        "source_micrometers_per_pixel": 6.5,
        "snip_micrometers_per_pixel": 6.5,
        "source_image_product_key": (
            "BF__projection__focus_stack"
            if product_key == BF_PRODUCT
            else "RFP__projection__max"
        ),
        "snip_transform_id": "fixture-transform",
        "output_grid_id": "fixture-grid",
        "source_height_px": 576,
        "source_width_px": 256,
        "output_height_px": 576,
        "output_width_px": 256,
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "realized_scale_y": 1.0,
        "realized_scale_x": 1.0,
        "orientation_policy": "fixture-explicit",
        "orientation_source": "fixture",
        "centering": "continuous",
        "pixel_dtype": "uint8",
        "resolved_transform_chain_json": "[]",
    }
