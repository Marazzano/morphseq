"""
Schema definitions for the segmentation_and_tracking product.

Co-located contract home for the Phase-3 segmentation/tracking sub-contracts:
the final segmentation_tracking table plus its upstream stages
(frame_detections, seed_selection, track_instances, mask_rle). Each stage's
normalizer imports its own ``REQUIRED_COLUMNS_*`` / ``UNIQUE_KEY_*`` pair from
here.
"""

REQUIRED_COLUMNS_SEGMENTATION_TRACKING = [
    # Core IDs
    'experiment_id',
    'well_id',              # Global well id: {experiment_id}_{well_index}
    'well_index',
    'image_id',
    'embryo_id',
    'snip_id',
    'time_int',

    # Mask data
    'mask_rle',             # Compressed mask as RLE string
    'area_px',              # Raw pixel area from SAM2
    'bbox_x_min',
    'bbox_y_min',
    'bbox_x_max',
    'bbox_y_max',
    'mask_confidence',

    # Geometry (will be converted to μm in features)
    'centroid_x_px',
    'centroid_y_px',

    # SAM2 metadata
    'is_seed_frame',        # Boolean - was this a SAM2 seed frame?

    # File references
    'source_image_path',    # Path to original stitched FF image
    'exported_mask_path',   # Path to exported PNG mask
]

# Phase 3 sub-contracts (segmentation_and_tracking)

REQUIRED_COLUMNS_FRAME_DETECTIONS = [
    "experiment_id",
    "well_id",
    "image_id",
    "time_int",
    "detection_index",
    "detection_instance_id",
    "box_x_min_abs",
    "box_y_min_abs",
    "box_x_max_abs",
    "box_y_max_abs",
    "detection_confidence",
    "image_height_px",
    "image_width_px",
    "source_backend",
    "source_model",
    "model_release",
    "run_id",
]

UNIQUE_KEY_FRAME_DETECTIONS = [
    "experiment_id",
    "well_id",
    "image_id",
    "detection_index",
    "source_backend",
    "run_id",
]

REQUIRED_COLUMNS_SEED_SELECTION = [
    "experiment_id",
    "well_id",
    "seed_time_int",
    "seed_image_id",
    "num_detections",
    "avg_confidence",
    "selection_reason",
    "candidate_frames_evaluated",
    "selected_detection_indices",
    "detector_backend",
    "run_id",
]

UNIQUE_KEY_SEED_SELECTION = [
    "experiment_id",
    "well_id",
    "run_id",
]

REQUIRED_COLUMNS_TRACK_INSTANCES = [
    "experiment_id",
    "well_id",
    "well_index",
    "image_id",
    "embryo_id",
    "embryo_local_id",
    "channel_id",
    "instance_id",
    "time_int",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "area_px",
    "mask_confidence",
    "centroid_x_px",
    "centroid_y_px",
    "is_seed_frame",
    "source_backend",
    "source_model",
    "model_release",
    "run_id",
]

UNIQUE_KEY_TRACK_INSTANCES = [
    "experiment_id",
    "well_id",
    "image_id",
    "embryo_id",
    "source_backend",
    "run_id",
]

REQUIRED_COLUMNS_MASK_RLE = [
    "experiment_id",
    "well_id",
    "image_id",
    "embryo_id",
    "embryo_local_id",
    "channel_id",
    "instance_id",
    "snip_id",
    "time_int",
    "mask_type",
    "mask_rle",
    "area_px",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "centroid_x_px",
    "centroid_y_px",
    "mask_confidence",
    "is_seed_frame",
    "source_image_path",
    "exported_mask_path",
    "source_backend",
    "source_model",
    "model_release",
    "run_id",
]

UNIQUE_KEY_MASK_RLE = [
    "experiment_id",
    "well_id",
    "image_id",
    "embryo_id",
    "mask_type",
    "source_backend",
    "run_id",
]

# V2 extension (optional): provenance + mask_type on the final contract.
REQUIRED_COLUMNS_SEGMENTATION_TRACKING_V2 = [
    *REQUIRED_COLUMNS_SEGMENTATION_TRACKING,
    "mask_type",
    "source_backend",
    "source_model",
    "model_release",
    "run_id",
]
