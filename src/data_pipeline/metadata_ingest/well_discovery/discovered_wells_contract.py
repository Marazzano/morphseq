"""Contract for discovered_wells.txt — physical wells present in canonical metadata.

discovered_wells.txt is physical identity, not QC-passed or runnable identity:
  run_wells = discovered_wells ∩ target_wells [ ∩ eligible_wells ]

Validation rejects bare local well_index labels (e.g. "A01") that leaked past
the well_id promotion step in join_series_mapping_to_scope_metadata.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.shared.identifiers.validators import validate_well_id


def validate_discovered_wells(wells: list[str]) -> None:
    """Raise ValueError if any entry is not a valid global well_id."""
    for w in wells:
        validate_well_id(w)


def read_discovered_wells(path: Path) -> list[str]:
    """Return the ordered list of well_ids from a discovered_wells.txt file."""
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def write_discovered_wells(path: Path, wells: list[str]) -> None:
    """Write an ordered list of well_ids to path, one per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(wells) + "\n")
