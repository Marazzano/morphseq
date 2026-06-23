"""Legacy re-export shim for plate metadata schema constants.

New code should import from:
    data_pipeline.metadata_ingest.plate.plate_metadata_contract

This module is kept so existing ``from data_pipeline.schemas.plate_metadata import
REQUIRED_COLUMNS_PLATE_METADATA`` call sites continue to work unchanged.
"""

from data_pipeline.metadata_ingest.plate.plate_metadata_contract import (
    REQUIRED_PLATE_METADATA_COLUMNS as REQUIRED_COLUMNS_PLATE_METADATA,
)

__all__ = ["REQUIRED_COLUMNS_PLATE_METADATA"]
