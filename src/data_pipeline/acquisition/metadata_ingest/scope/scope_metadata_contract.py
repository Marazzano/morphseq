"""
Schema definition for scope (microscope) metadata.

Required columns for microscope-extracted metadata, including spatial and
temporal calibration parameters. Co-located with the scope-extraction product
(``scope/<scope>/extract_*_scope_metadata.py``); this is its authoritative
schema home.

Distinct from ``acquisition_inventory_contract.py``: that guards the tiered
acquisition inventory; this guards the raw per-series scope-metadata table
consumed at the well-mapping join.
"""

REQUIRED_COLUMNS_SCOPE_METADATA = [
    # Raw acquisition identity (pre-mapping; well_id is minted later at the join)
    'experiment_id',
    'raw_position_label',
    'time_index',

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
    'channel_id',
    'z_position',
]
