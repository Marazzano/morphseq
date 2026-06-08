"""
Schema definition for scope (microscope) metadata.

This module defines required columns for microscope-extracted metadata,
including spatial and temporal calibration parameters.
"""

REQUIRED_COLUMNS_SCOPE_METADATA = [
    # Raw acquisition identity (pre-mapping; well_id is minted later at the join)
    'experiment_id',
    'raw_position_label',
    'time_int',

    # Stage XY (µm) — per series, T=0; enables CSV→CSV well mapping downstream
    'x_um',
    'y_um',

    # Spatial calibration
    'micrometers_per_pixel',
    'image_width_px',
    'image_height_px',
    'objective_magnification',

    # Temporal calibration
    'frame_interval_s',
    'absolute_start_time',
    'experiment_time_s',

    # Acquisition metadata
    'microscope_id',
    'channel',
    'z_position',
]
