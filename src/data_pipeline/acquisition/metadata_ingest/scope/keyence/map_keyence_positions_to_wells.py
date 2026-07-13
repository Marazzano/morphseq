"""
Keyence-specific position-to-well mapping.

Maps Keyence microscope acquisition positions to well positions based on file structure.
"""

import argparse
import pandas as pd
import json
import re
from pathlib import Path
import logging
import re

from data_pipeline.acquisition.metadata_ingest.position_well_mapping import validate_position_well_mapping
from data_pipeline.acquisition.metadata_ingest.scope.keyence.raw_plane_parsing import (
    _parse_keyence_xy_position_index,
    _read_keyence_well_marker,
)
from data_pipeline.shared.identifiers import build_well_id

log = logging.getLogger(__name__)

# Position mapping schema
REQUIRED_COLUMNS_POSITION_MAPPING = [
    'experiment_id',
    'position_index',
    'well_index',
    'well_id',
    'mapping_method',
]


<<<<<<< HEAD
_WELL_MARKER_RE = re.compile(r"^_([A-H])(\d{1,2})$")


def _well_index_from_marker(well_dir: Path) -> str | None:
    """Return the well label the microscope itself recorded in ``well_dir``, or None.

    Keyence drops a zero-byte marker file named ``_<WELL>`` (e.g. ``_B12``) into every ``XY##``
    directory. That marker is GROUND TRUTH and must win over any arithmetic on the ``XY`` index:
    the scope images plates in a serpentine (boustrophedon) order, so ``XY13`` is ``B12``, not
    ``B01``. Deriving the well from ``(idx-1)//12`` / ``(idx-1)%12`` column-reverses every even
    row and silently mislabels half the plate.
    """
    for entry in well_dir.iterdir():
        m = _WELL_MARKER_RE.match(entry.name)
        if m:
            return f"{m.group(1)}{int(m.group(2)):02d}"
    return None
=======
def _keyence_position_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"XY(\d+)([A-Za-z]?)", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).lower()
    return 10**9, path.name
>>>>>>> 62691eca2a99c1c3bb7a28444c2d51fc0bf934c4


def _discover_keyence_wells(raw_data_dir: Path, experiment_id: str) -> list:
    """
    Discover Keyence acquisition positions and their well markers.

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        experiment_id: Experiment identifier

    Returns:
        List of (position_index, well_index, position_path) tuples sorted by Keyence position
    """
    exp_dir = raw_data_dir / experiment_id

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    positions = []

    # Check for XY pattern (e.g., XY01, XY16)
    xy_dirs = sorted(
        (p for p in exp_dir.glob("XY*") if p.is_dir()),
        key=_keyence_position_sort_key,
    )
    if xy_dirs:
<<<<<<< HEAD
        for well_dir in xy_dirs:
            # The scope's own `_<WELL>` marker is authoritative — see _well_index_from_marker.
            well_index = _well_index_from_marker(well_dir)
            if well_index is not None:
                wells.append((well_index, well_dir))
                continue

            well_name = well_dir.name  # e.g., "XY01a"
            suffix = well_name[2:]
            if suffix.isdigit():
                xy_idx = int(suffix)
                row = (xy_idx - 1) // 12
                col = (xy_idx - 1) % 12 + 1
                well_index = f"{chr(65 + row)}{col:02d}"
                log.warning(
                    "No _<WELL> marker in %s; falling back to raster arithmetic -> %s. This is "
                    "WRONG for serpentine acquisitions (XY13 is B12, not B01).",
                    well_dir, well_index,
                )
                wells.append((well_index, well_dir))
=======
        for position_dir in xy_dirs:
            position_name = position_dir.name  # e.g., "XY01" or legacy "XY01a"
            suffix = position_name[2:]
            if suffix.isdigit():
                position_index = _parse_keyence_xy_position_index(position_dir)
                if position_index is None:
                    raise ValueError(f"Unexpected Keyence XY position directory: {position_dir}")
                well_index = _read_keyence_well_marker(position_dir)
                positions.append((position_index, well_index, position_dir))
>>>>>>> 62691eca2a99c1c3bb7a28444c2d51fc0bf934c4
                continue

            # Legacy format with trailing row letter (e.g. XY01a)
            well_num = position_name[2:4]
            well_letter = position_name[-1].upper()
            if well_num.isdigit() and well_letter.isalpha():
                well_index = f"{well_letter}{well_num}"
                positions.append((len(positions), well_index, position_dir))

    # Check for W0 pattern (W001 → A01)
    w0_dirs = sorted(exp_dir.glob("W0*"))
    if w0_dirs and not xy_dirs:
<<<<<<< HEAD
        for well_dir in w0_dirs:
            well_index = _well_index_from_marker(well_dir)
            if well_index is not None:
                wells.append((well_index, well_dir))
                continue

            well_name = well_dir.name  # e.g., "W001"
            well_num = int(well_name[1:])
=======
        for position_dir in w0_dirs:
            position_name = position_dir.name  # e.g., "W001"
            well_num = int(position_name[1:])
>>>>>>> 62691eca2a99c1c3bb7a28444c2d51fc0bf934c4
            # Convert to row/col (1-indexed, 12 cols per row)
            row = (well_num - 1) // 12
            col = (well_num - 1) % 12 + 1
            well_index = f"{chr(65 + row)}{col:02d}"
            positions.append((well_num, well_index, position_dir))

    if not positions:
        raise ValueError(f"No Keyence well directories found in {exp_dir}")

    log.info(f"Discovered {len(positions)} Keyence positions in {exp_dir}")
    return sorted(positions, key=lambda x: x[0])


def _count_positions_per_well(well_path: Path) -> int:
    """
    Count number of positions/series in a Keyence well directory.

    Keyence can have:
    - Single position (files directly in well dir)
    - Multiple positions (P* subdirectories)

    Args:
        well_path: Path to well directory

    Returns:
        Number of positions/series in well
    """
    # Check for P* subdirectories
    pos_dirs = sorted(well_path.glob("P*"))
    if pos_dirs:
        return len(pos_dirs)

    # A well's mosaic tiles are NOT separate acquisition positions here: position_index in this
    # mapping lives in scope_metadata's per-XY space (one entry per well). Tiles are resolved
    # downstream from the acquisition inventory's own per-tile rows.
    image_files = list(well_path.glob("*CH*.tif"))
    if image_files:
        return 1

    # No images found
    log.warning(f"No images found in {well_path}")
    return 0


def map_positions_to_wells_keyence(
    raw_data_dir: Path,
    scope_metadata_csv: Path,
    output_mapping_csv: Path,
    output_provenance_json: Path,
    experiment_id: str,
) -> pd.DataFrame:
    """
    Map Keyence acquisition positions to wells (plate-free).

    For Keyence microscopes, the mapping strategy is:
    1. Discover acquisition position directories (XY## or W0##)
    2. Read explicit Keyence well markers under XY## directories (_A01, _B12, ...)
    3. Preserve the Keyence acquisition position index from XY##
    4. Cross-reference with scope metadata to flag missing wells

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        scope_metadata_csv: Path to validated scope_series_metadata_raw.csv
        output_mapping_csv: Path to write position_well_mapping.csv
        output_provenance_json: Path to write mapping_provenance.json
        experiment_id: Experiment identifier (used for composing well_id)

    Returns:
        DataFrame with position-to-well mapping

    Raises:
        FileNotFoundError: If required files not found
        ValueError: If mapping fails validation
    """
    log.info(f"Mapping Keyence positions to wells for {experiment_id}")

    # Load scope metadata to cross-reference
    scope_df = pd.read_csv(scope_metadata_csv)

    # Discover acquisition positions from raw data directory
    positions = _discover_keyence_wells(raw_data_dir, experiment_id)

    # Build mapping
    rows = []
    warnings = []
    scope_wells = (
        set(scope_df['well_index'].astype(str).values)
        if 'well_index' in scope_df.columns
        else set()
    )

    for position_index, well_index, position_path in positions:
        n_positions = _count_positions_per_well(position_path)

        if n_positions == 0:
            warnings.append(f"Well {well_index}: No images found")
            continue

        if well_index not in scope_wells:
            warnings.append(f"Well {well_index}: Not found in scope metadata")

        row = {
            'experiment_id': experiment_id,
            'position_index': int(position_index),
            'well_index': well_index,
            'well_id': build_well_id(experiment_id, well_index),
            'mapping_method': 'keyence_xy_well_marker',
            'source_position_name': position_path.name,
            'n_positions_in_well': n_positions,
            'source_directory': str(position_path),
        }

        rows.append(row)

    if not rows:
        raise ValueError(f"No valid position-to-well mappings found for {experiment_id}")

    # Build DataFrame
    df = pd.DataFrame(rows)

    validate_position_well_mapping(df, scope_label="Keyence position_well_mapping")

    # Write output CSV
    output_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_mapping_csv, index=False)
    log.info(f"Wrote Keyence position-well mapping to {output_mapping_csv}")

    # Write provenance JSON
    provenance = {
        'experiment_id': experiment_id,
        'microscope': 'Keyence',
        'mapping_method': 'keyence_xy_well_marker',
        'n_positions': int(len(df)),
        'n_wells': int(df['well_index'].nunique()),
        'mapping_summary': {
            'total_positions': int(len(df)),
            'total_wells': int(df['well_index'].nunique()),
            'wells_with_multiple_positions': int((df['n_positions_in_well'] > 1).sum()),
            'min_position_index': int(df['position_index'].min()),
            'max_position_index': int(df['position_index'].max()),
        },
        'warnings': warnings if warnings else None,
        'source_files': {
            'scope_metadata': str(scope_metadata_csv),
        },
    }

    output_provenance_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_provenance_json, 'w') as f:
        json.dump(provenance, f, indent=2)
    log.info(f"Wrote mapping provenance to {output_provenance_json}")

    if warnings:
        log.warning(f"Mapping completed with {len(warnings)} warnings")
        for warn in warnings[:5]:  # Show first 5 warnings
            log.warning(f"  {warn}")
        if len(warnings) > 5:
            log.warning(f"  ... and {len(warnings) - 5} more warnings")

    return df


def load_position_well_mapping(mapping_csv: Path) -> pd.DataFrame:
    """
    Load and validate position-to-well mapping.

    Args:
        mapping_csv: Path to position_well_mapping.csv

    Returns:
        Validated DataFrame

    Raises:
        FileNotFoundError: If file not found
        ValueError: If validation fails
    """
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Position mapping not found: {mapping_csv}")

    df = pd.read_csv(mapping_csv)

    validate_position_well_mapping(df, scope_label=str(mapping_csv))

    return df



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-keyence-experiment-dir", type=Path, required=True)
    p.add_argument("--scope-metadata-csv", type=Path, required=True)
    p.add_argument("--output-mapping-csv", type=Path, required=True)
    p.add_argument("--output-provenance-json", type=Path, required=True)
    p.add_argument("--experiment-id", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    map_positions_to_wells_keyence(
        raw_data_dir=args.raw_keyence_experiment_dir,
        scope_metadata_csv=args.scope_metadata_csv,
        output_mapping_csv=args.output_mapping_csv,
        output_provenance_json=args.output_provenance_json,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
