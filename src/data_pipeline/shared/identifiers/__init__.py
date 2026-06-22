"""Canonical identifier helpers for the pipeline.

One sacred place that mints, parses, and validates identifiers. Split into
``constructors`` (mint), ``parsers`` (decompose), and ``validators`` (guard).

This ``__init__`` re-exports every public name so existing
``from data_pipeline.shared.identifiers import ...`` call sites keep working
unchanged across the Scope-1 package split.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md
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
from .constructors import build_track_id
from .constructors import build_well_id
from .constructors import sanitize_experiment_id
from .parsers import normalize_embryo_local_track_id  # deprecated alias for parse_embryo_local_track_id
from .parsers import parse_embryo_id
from .parsers import parse_embryo_local_track_id
from .parsers import parse_image_id
from .parsers import parse_mask_id
from .parsers import parse_physical_embryo_id
from .parsers import parse_snip_id
from .parsers import parse_track_id
from .parsers import split_well_id
from .parsers import track_index_to_embryo_index
from .validators import validate_physical_embryo_id
from .validators import validate_well_id

__all__ = [
    # constructors
    "build_embryo_id",
    "build_image_id",
    "build_mask_id",
    "build_no_mask_id",
    "build_physical_embryo_id",
    "build_snip_id",
    "build_track_id",
    "build_well_id",
    "sanitize_experiment_id",
    # parsers
    "normalize_embryo_local_track_id",  # deprecated; use parse_embryo_local_track_id
    "parse_embryo_id",
    "parse_embryo_local_track_id",
    "parse_image_id",
    "parse_mask_id",
    "parse_physical_embryo_id",
    "parse_snip_id",
    "parse_track_id",
    "split_well_id",
    "track_index_to_embryo_index",
    # validators
    "validate_physical_embryo_id",
    "validate_well_id",
]
