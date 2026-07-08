"""Validate and merge the live per-well frame_inventory shards emitted by materialize_well.

The validator (``validate_frame_inventory``, the public gate) lives in ``frame_inventory_validation``;
the table merge op lives in ``frame_inventory``. Both are re-exported here as the package surface.
"""

from .frame_inventory import merge_frame_inventory_shards
from .frame_inventory_validation import validate_frame_inventory

__all__ = [
    "merge_frame_inventory_shards",
    "validate_frame_inventory",
]
