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
from .constructors import build_snip_id
from .constructors import build_well_id
from .constructors import sanitize_experiment_id
from .parsers import normalize_embryo_local_track_id
from .parsers import split_well_id
from .validators import validate_well_id

__all__ = [
    "build_embryo_id",
    "build_image_id",
    "build_snip_id",
    "build_well_id",
    "sanitize_experiment_id",
    "normalize_embryo_local_track_id",
    "split_well_id",
    "validate_well_id",
]
