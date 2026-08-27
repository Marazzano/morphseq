"""Canonical identifier helpers for the pipeline.

One sacred place that mints, parses, and validates identifiers. Split into
``constructors`` (mint), ``parsers`` (decompose), and ``validators`` (guard).

This ``__init__`` re-exports every public name so existing
``from data_pipeline.shared.identifiers import ...`` call sites keep working
unchanged across the Scope-1 package split.

See docs/data_pipeline/specs/identifier_and_wildcard_contract.md
and target/well_id_throughline_refactor_plan.md (Scope 1 = this split; Scope 2 =
the well_id-first / global-well_id signature flip).
"""

from __future__ import annotations

from .constructors import build_embryo_id
from .constructors import build_image_id
from .constructors import build_mask_id
from .constructors import build_no_mask_id
from .constructors import build_physical_embryo_id
from .constructors import build_snip_id
from .constructors import build_snip_transform_id
from .constructors import build_track_id
from .constructors import build_well_id
from .constructors import sanitize_experiment_id
from .parsers import compose_collection_experiment_id
from .parsers import is_collection
from .parsers import is_collection_plate_id
from .parsers import parse_collection_name_from_plate_id
from .parsers import normalize_embryo_local_track_id  # deprecated alias for parse_embryo_local_track_id
from .parsers import parse_declared_hpf
from .parsers import parse_embryo_id
from .parsers import parse_embryo_local_track_id
from .parsers import parse_image_id
from .parsers import parse_image_id_with_z_index
from .parsers import parse_mask_id
from .parsers import parse_physical_embryo_id
from .parsers import parse_event_label
from .parsers import parse_plate_token
from .parsers import parse_snip_id
from .parsers import parse_track_id
from .parsers import parse_well_row_col
from .parsers import split_well_id
from .parsers import track_index_to_embryo_index
from .validators import normalize_well_index
from .validators import validate_physical_embryo_id
from .validators import validate_well_id
from .validators import validate_well_index

__all__ = [
    # constructors
    "build_embryo_id",
    "build_image_id",
    "build_mask_id",
    "build_no_mask_id",
    "build_physical_embryo_id",
    "build_snip_id",
    "build_snip_transform_id",
    "build_track_id",
    "build_well_id",
    "sanitize_experiment_id",
    # parsers
    "compose_collection_experiment_id",
    "is_collection",
    "is_collection_plate_id",
    "parse_collection_name_from_plate_id",
    "normalize_embryo_local_track_id",  # deprecated; use parse_embryo_local_track_id
    "parse_declared_hpf",
    "parse_event_label",
    "parse_plate_token",
    "parse_embryo_id",
    "parse_embryo_local_track_id",
    "parse_image_id",
    "parse_image_id_with_z_index",
    "parse_mask_id",
    "parse_physical_embryo_id",
    "parse_snip_id",
    "parse_track_id",
    "parse_well_row_col",
    "split_well_id",
    "track_index_to_embryo_index",
    # validators
    "normalize_well_index",
    "validate_physical_embryo_id",
    "validate_well_id",
    "validate_well_index",
]
