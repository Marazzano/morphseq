from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.image_materialization.frame_inventory_contract import (
    validate_frame_inventory_identity_contract,
)
from data_pipeline.acquisition.image_materialization.frame_modality import (
    frame_modality_for_image,
)
from data_pipeline.shared.identifiers import build_image_id, build_well_id


def _row() -> dict:
    experiment_id = "20260723_seahub_GENE1_shard001"
    well_index = "A01"
    well_id = build_well_id(experiment_id, well_index)
    image_id = build_image_id(well_id, "BF", 0)
    return {
        "experiment_id": experiment_id,
        "well_index": well_index,
        "well_id": well_id,
        "channel_id": "BF",
        "time_index": 0,
        "z_index": pd.NA,
        "image_id": image_id,
        "image_product_type": "projection",
        "projection_method": "focus_stack",
        "source_scope": "seahub",
        "image_kind": "single_z",
        "z_position": pd.NA,
        "calibration_status": "placeholder",
    }


def test_single_z_can_use_projection_compatibility_slot():
    frame = pd.DataFrame([_row()])
    validate_frame_inventory_identity_contract(frame)
    modality = frame_modality_for_image(
        frame,
        image_id=frame.iloc[0]["image_id"],
        product_key="BF__projection__focus_stack",
    )
    assert modality["image_kind"] == "single_z"
    assert pd.isna(modality["z_position"])


def test_partial_modality_block_fails_loud():
    row = _row()
    row.pop("calibration_status")
    with pytest.raises(ValueError, match="partial frame modality block"):
        validate_frame_inventory_identity_contract(pd.DataFrame([row]))
