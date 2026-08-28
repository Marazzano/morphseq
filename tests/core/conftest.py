from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.data.manifest_types import (
    AssetSelectionPolicy,
    CovariatePolicy,
    ExperimentTables,
    ManifestPolicy,
    MetricMappingPolicy,
    QCPolicy,
    SplitPolicy,
    StagePolicy,
    ValidityPolicy,
)


BF_PRODUCT = "BF__projection__focus_stack__clahe_blend"
RFP_PRODUCT = "RFP__projection__max__no_change"
Z_PRODUCT = "BF__z_stack__no_change"


def _inventory_row(
    *,
    experiment_id: str,
    well_id: str,
    physical_embryo_id: str,
    embryo_id: str,
    snip_id: str,
    time_index: int,
    product: str,
    z_index,
    source_order: int,
    valid=True,
):
    return {
        "experiment_id": experiment_id,
        "well_id": well_id,
        "physical_embryo_id": physical_embryo_id,
        "embryo_id": embryo_id,
        "snip_id": snip_id,
        "image_id": f"declared-frame-{well_id}-{time_index}",
        "time_index": time_index,
        "channel_id": "BF",
        "mask_id": f"mask-{snip_id}",
        "track_id": f"track-{physical_embryo_id}",
        "snip_product_key": product,
        "z_index": z_index,
        "processed_snip_path": str(Path("/tmp") / f"asset-{source_order}.png"),
        "image_path": str(Path("/tmp") / f"source-{source_order}.tif"),
        "embryo_mask_snip_path": str(Path("/tmp") / f"mask-{source_order}.png"),
        "is_valid_snip": valid,
        "error_message": None,
        "source_image_product_key": product,
        "snip_transform_id": f"transform-{snip_id}",
        "output_grid_id": "grid-576x256",
        "source_micrometers_per_pixel": 1.2,
        "snip_micrometers_per_pixel": 6.5,
        "pixel_dtype": "uint8",
        "resolved_transform_chain_json": "[]",
    }


@pytest.fixture
def manifest_bundle_factory():
    def factory(
        experiment_id: str = "exp-A",
        *,
        include_siblings: bool = True,
        n_observations: int = 3,
        qc: bool = True,
        stage: bool = True,
    ) -> ExperimentTables:
        inventory_rows = []
        frame_rows = []
        plate_rows = []
        stage_rows = []
        qc_rows = []
        source_order = 0
        for index in range(n_observations):
            well_id = f"well-{experiment_id}-{index}"
            physical_id = f"physical::{experiment_id}::{index}"
            embryo_id = f"embryo::{experiment_id}::{index}"
            snip_id = f"observation::{experiment_id}::{index}"
            inventory_rows.append(
                _inventory_row(
                    experiment_id=experiment_id,
                    well_id=well_id,
                    physical_embryo_id=physical_id,
                    embryo_id=embryo_id,
                    snip_id=snip_id,
                    time_index=index,
                    product=BF_PRODUCT,
                    z_index=pd.NA,
                    source_order=source_order,
                )
            )
            source_order += 1
            if index == 0 and include_siblings:
                inventory_rows.append(
                    _inventory_row(
                        experiment_id=experiment_id,
                        well_id=well_id,
                        physical_embryo_id=physical_id,
                        embryo_id=embryo_id,
                        snip_id=snip_id,
                        time_index=index,
                        product=RFP_PRODUCT,
                        z_index=pd.NA,
                        source_order=source_order,
                    )
                )
                source_order += 1
                for z_index in (2, 0, 1):
                    inventory_rows.append(
                        _inventory_row(
                            experiment_id=experiment_id,
                            well_id=well_id,
                            physical_embryo_id=physical_id,
                            embryo_id=embryo_id,
                            snip_id=snip_id,
                            time_index=index,
                            product=Z_PRODUCT,
                            z_index=z_index,
                            source_order=source_order,
                        )
                    )
                    source_order += 1
            frame_rows.extend(
                [
                    {
                        "experiment_id": experiment_id,
                        "well_id": well_id,
                        "time_index": index,
                        "channel_id": channel,
                        "z_index": pd.NA,
                        "image_product_type": "projection",
                        "projection_method": method,
                        "elapsed_time_s": float(index * 60 + 7),
                    }
                    for channel, method in (("BF", "focus_stack"), ("RFP", "max"))
                ]
            )
            plate_rows.append(
                {
                    "experiment_id": experiment_id,
                    "well_id": well_id,
                    "well_index": f"W{index}",
                    "genotype": f"genotype-{index}",
                    "start_age_hpf": float(20 + index),
                    "temperature": float(27 + index / 10),
                    "medium": "E3",
                    "strain": "AB",
                    "chem_perturbation": None,
                }
            )
            stage_rows.append(
                {
                    "experiment_id": experiment_id,
                    "well_id": well_id,
                    "physical_embryo_id": physical_id,
                    "embryo_id": embryo_id,
                    "snip_id": snip_id,
                    "predicted_stage_hpf": float(21 + index),
                    "stage_prediction_status": "predicted",
                    "model_version": "kimmel1995_temp_rate_v1",
                }
            )
            qc_rows.append(
                {
                    "experiment_id": experiment_id,
                    "well_id": well_id,
                    "physical_embryo_id": physical_id,
                    "embryo_id": embryo_id,
                    "snip_id": snip_id,
                    "use_snip": "True",
                    "qc_fail_reasons": "",
                    "sa_outlier_flag": "False",
                    "edge_flag": False,
                    "surface_area_qc_applicability": "exclusion",
                }
            )
        return ExperimentTables(
            experiment_id=experiment_id,
            snip_inventory=pd.DataFrame(inventory_rows),
            frame_inventory=pd.DataFrame(frame_rows),
            plate_metadata=pd.DataFrame(plate_rows),
            stage_predictions=pd.DataFrame(stage_rows) if stage else None,
            snip_qc=pd.DataFrame(qc_rows) if qc else None,
            collection_provenance={
                "experiment_id": experiment_id,
                "is_collection": False,
                "sources": [
                    {
                        "file": experiment_id,
                        "raw_path": f"/raw/{experiment_id}",
                        "declared_hpf": None,
                        "source_ordinal": 0,
                        "time_index": 0,
                    }
                ],
                "start_age_by_source_ordinal": {},
                "start_age_by_time_index": {},
            },
            schema_versions={"snip_qc": "Q-current-test"},
        )

    return factory

@pytest.fixture
def manifest_policy_factory():
    def factory(
        experiment_ids=("exp-A",),
        *,
        qc_enabled=True,
        stage_enabled=True,
        split_policy=None,
        required_covariates=(),
        exclude_flags=(),
        metric_enabled=False,
    ) -> ManifestPolicy:
        return ManifestPolicy(
            pipeline_output_root=Path("/pipeline/output"),
            experiment_ids=tuple(experiment_ids),
            assets=AssetSelectionPolicy(
                (BF_PRODUCT, RFP_PRODUCT, Z_PRODUCT), BF_PRODUCT, "projection_null"
            ),
            validity=ValidityPolicy(True, "require_valid_test"),
            qc=QCPolicy(
                qc_enabled,
                "temporary_smoke_qc",
                "v1",
                ("evaluated",),
                True,
                tuple(exclude_flags),
            ),
            stage=StagePolicy(
                stage_enabled,
                "temporary_smoke_stage",
                ("predicted",),
                True,
            ),
            covariates=CovariatePolicy(
                "temporary_smoke_covariates", tuple(required_covariates)
            ),
            splits=split_policy or SplitPolicy(enabled=False, required_splits=()),
            metric_mapping=(
                MetricMappingPolicy(
                    True,
                    "test_only_single_group",
                    "v1",
                    scientific_policy=False,
                    constant_group="test_group",
                )
                if metric_enabled
                else MetricMappingPolicy(False, "metric_disabled", "v1")
            ),
        )

    return factory
