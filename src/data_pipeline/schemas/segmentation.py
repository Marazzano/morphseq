"""
Schema definition for SAM2 segmentation tracking output.

This module defines required columns for the segmentation tracking table,
which contains embryo masks, tracking metadata, and SAM2-specific fields.
"""

REQUIRED_COLUMNS_SEGMENTATION_TRACKING = [
    # Core IDs
    'experiment_id',
    'video_id',
    'well_id',              # Well identifier for grouping
    'well_index',
    'image_id',
    'embryo_id',
    'snip_id',
    'frame_index',
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


# V2 schema: self-contained snapshot so downstream stages never need to join frame manifests.
REQUIRED_COLUMNS_SEGMENTATION_TRACKING_V2 = [
    "schema_version",

    # Core IDs (same as V1)
    *REQUIRED_COLUMNS_SEGMENTATION_TRACKING[:9],

    # Additional identity
    "instance_id",

    # Frame physical snapshot
    "channel_id",
    "source_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
    "frame_snapshot_hash",

    # Canonical mask naming
    "embryo_mask_rle",
    "embryo_mask_path",

    # Rest of V1 fields
    *REQUIRED_COLUMNS_SEGMENTATION_TRACKING[9:],
]
