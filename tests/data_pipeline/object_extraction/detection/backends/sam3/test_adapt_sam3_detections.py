import pandas as pd

from data_pipeline.object_extraction.detection.backends.sam3.adapt_sam3_detections import (
    adapt_sam3_detections,
)
from data_pipeline.object_extraction.detection.validate_frame_detections import (
    validate_frame_detection_block,
)

EXP = "20250912"
WELL_ID = "20250912_B01"
IMAGE_ID = "20250912_B01_BF_t0000"


def _identity_row() -> dict:
    return {
        "experiment_id": EXP,
        "well_id": WELL_ID,
        "image_id": IMAGE_ID,
        "time_index": 0,
        "z_index": pd.NA,
        "channel_id": "BF",
        "source_image_path": f"images/{IMAGE_ID}.png",
        "image_width_px": 1000,
        "image_height_px": 800,
    }


def test_adapts_normalized_sam3_boxes_to_frame_detections():
    df = adapt_sam3_detections(
        [
            {"box_xyxy": [100, 80, 300, 240], "score": 0.95, "label": "embryo"},
            {"bbox_xyxy": [500, 400, 700, 560], "confidence": 0.25, "is_kept": False},
        ],
        identity_row=_identity_row(),
    )

    validate_frame_detection_block(df)
    assert len(df) == 2
    assert df.loc[0, "detector_backend"] == "sam3"
    assert df.loc[0, "detection_id"] == f"{IMAGE_ID}_det0000"
    assert df.loc[0, "confidence"] == 0.95
    assert df.loc[0, "bbox_format"] == "xyxy_px_abs"
    assert bool(df.loc[0, "is_kept"]) is True
    assert bool(df.loc[1, "is_kept"]) is False


def test_adapts_empty_response_to_placeholder():
    df = adapt_sam3_detections([], identity_row=_identity_row())

    validate_frame_detection_block(df)
    assert len(df) == 1
    assert df.loc[0, "detection_id"] == f"{IMAGE_ID}_det_none"
    assert bool(df.loc[0, "is_kept"]) is False
    assert pd.isna(df.loc[0, "confidence"])
