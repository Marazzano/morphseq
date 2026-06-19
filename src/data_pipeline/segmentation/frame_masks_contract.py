"""Target `frame_masks` contract constants and legacy adapters."""

from __future__ import annotations

import pandas as pd


FRAME_MASKS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "image_id",
    "time_index",
    "z_index",
    "channel_id",
    "source_image_path",
    "image_width_px",
    "image_height_px",
    "mask_id",
    "track_id",
    "seed_id",
    "mask_rle",
    "mask_rle_format",
    "area_px",
    "bbox_x_min_px",
    "bbox_y_min_px",
    "bbox_x_max_px",
    "bbox_y_max_px",
    "centroid_x_px",
    "centroid_y_px",
    "mask_confidence",
    "is_valid_mask",
    "segmentation_backend",
    "segmentation_model_id",
    "tracking_backend",
    "track_id_source",
)

FRAME_MASKS_UNIQUE_KEY: tuple[str, ...] = ("mask_id",)


def empty_frame_masks() -> pd.DataFrame:
    return pd.DataFrame(columns=FRAME_MASKS_REQUIRED_COLUMNS)


def adapt_legacy_mask_rle_to_frame_masks(mask_rle_df: pd.DataFrame) -> pd.DataFrame:
    """Adapt legacy `embryo_mask_rle` rows into the target `frame_masks` shape.

    This is intended for contract testing and strangler comparison, not as the final SAM2 adapter.
    """
    if mask_rle_df.empty:
        return empty_frame_masks()

    rows: list[dict[str, object]] = []
    ordered = mask_rle_df.sort_values(["image_id", "embryo_id"], kind="mergesort")
    for image_id, group in ordered.groupby("image_id", sort=False):
        for local_idx, (_, row) in enumerate(group.iterrows()):
            source_backend = str(row.get("source_backend", "legacy"))
            source_model = str(row.get("source_model", "unknown"))
            model_release = str(row.get("model_release", "unknown"))
            rows.append(
                {
                    "experiment_id": str(row["experiment_id"]),
                    "well_id": str(row["well_id"]),
                    "image_id": str(image_id),
                    "time_index": int(row.get("time_index", row.get("time_int"))),
                    "z_index": row.get("z_index", pd.NA),
                    "channel_id": str(row.get("channel_id", "")),
                    "source_image_path": str(row.get("source_image_path", "")),
                    "image_width_px": row.get("image_width_px", pd.NA),
                    "image_height_px": row.get("image_height_px", pd.NA),
                    "mask_id": f"{image_id}_m{local_idx:04d}",
                    "track_id": str(row.get("embryo_id", row.get("track_id", ""))),
                    "seed_id": row.get("seed_id", pd.NA),
                    "mask_rle": row.get("mask_rle", pd.NA),
                    "mask_rle_format": str(row.get("mask_rle_format", "legacy_json")),
                    "area_px": float(row.get("area_px", 0.0)),
                    "bbox_x_min_px": float(row.get("bbox_x_min_px", row.get("bbox_x_min", 0.0))),
                    "bbox_y_min_px": float(row.get("bbox_y_min_px", row.get("bbox_y_min", 0.0))),
                    "bbox_x_max_px": float(row.get("bbox_x_max_px", row.get("bbox_x_max", 0.0))),
                    "bbox_y_max_px": float(row.get("bbox_y_max_px", row.get("bbox_y_max", 0.0))),
                    "centroid_x_px": float(row.get("centroid_x_px", 0.0)),
                    "centroid_y_px": float(row.get("centroid_y_px", 0.0)),
                    "mask_confidence": float(row.get("mask_confidence", 0.0)),
                    "is_valid_mask": True,
                    "segmentation_backend": source_backend,
                    "segmentation_model_id": f"{source_model}:{model_release}",
                    "tracking_backend": source_backend,
                    "track_id_source": "legacy_embryo_id",
                }
            )
    return pd.DataFrame(rows, columns=FRAME_MASKS_REQUIRED_COLUMNS)
