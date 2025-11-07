"""
YX1 series-to-well mapping.

Maps YX1 ND2 series numbers to plate well positions using explicit mapping
from plate metadata or implicit positional mapping.
"""

from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _parse_series_number_map(plate_df: pd.DataFrame) -> dict | None:
    """
    Extract series_number_map from plate metadata if present.

    Args:
        plate_df: Plate metadata DataFrame

    Returns:
        Dictionary mapping series numbers to well names, or None if not found
    """
    # Check if there's a series_number_map column or similar
    if 'series_number_map' in plate_df.columns:
        # Build mapping from series to well
        mapping = {}
        for _, row in plate_df.iterrows():
            series = row.get('series_number_map')
            well = row.get('well')
            if pd.notna(series) and pd.notna(well):
                try:
                    series_num = int(series)
                    mapping[series_num] = well
                except (ValueError, TypeError):
                    continue

        if mapping:
            log.info(f"Found explicit series_number_map with {len(mapping)} entries")
            return mapping

    log.info("No explicit series_number_map found in plate metadata")
    return None


def _build_implicit_mapping(scope_df: pd.DataFrame, plate_df: pd.DataFrame) -> dict:
    """
    Build implicit series-to-well mapping based on positional order.

    For YX1, we assume series numbers map to wells in row-major order
    (A01, A02, ..., A12, B01, B02, ..., H12) unless explicit mapping exists.

    Args:
        scope_df: Scope metadata DataFrame
        plate_df: Plate metadata DataFrame

    Returns:
        Dictionary mapping series numbers (1-based) to well names
    """
    # Get unique wells from scope metadata (these are the ND2 series)
    scope_wells = scope_df['well_index'].unique()
    n_series = len(scope_wells)

    # Get wells from plate metadata (these are the experiment wells)
    plate_wells = sorted(plate_df['well'].unique())

    log.info(f"Building implicit mapping: {n_series} series → {len(plate_wells)} plate wells")

    # Build positional mapping
    mapping = {}

    # YX1 series are typically 1-based indices
    for series_idx, well_index in enumerate(scope_wells):
        # Convert well_index to 1-based series number
        try:
            series_num = int(well_index) + 1  # Convert from 0-based to 1-based
        except ValueError:
            series_num = series_idx + 1

        # Map to plate well by position
        if series_idx < len(plate_wells):
            mapping[series_num] = plate_wells[series_idx]
        else:
            log.warning(f"Series {series_num} has no corresponding plate well")

    return mapping


def map_series_to_wells_yx1(
    plate_metadata_csv: Path,
    scope_metadata_csv: Path,
    output_mapping_csv: Path,
    output_provenance_json: Path
) -> pd.DataFrame:
    """
    Map YX1 ND2 series numbers to well positions.

    Args:
        plate_metadata_csv: Path to plate metadata CSV
        scope_metadata_csv: Path to scope metadata CSV
        output_mapping_csv: Path to output mapping CSV
        output_provenance_json: Path to output provenance JSON

    Returns:
        DataFrame with series-to-well mapping
    """
    log.info("Mapping YX1 series to wells")

    # Load metadata
    plate_df = pd.read_csv(plate_metadata_csv)
    scope_df = pd.read_csv(scope_metadata_csv)

    log.info(f"Loaded plate metadata: {len(plate_df)} rows")
    log.info(f"Loaded scope metadata: {len(scope_df)} rows")

    # Try explicit mapping first
    series_map = _parse_series_number_map(plate_df)
    mapping_method = "explicit"

    # Fall back to implicit mapping
    if series_map is None:
        series_map = _build_implicit_mapping(scope_df, plate_df)
        mapping_method = "implicit_positional"

    log.info(f"Using {mapping_method} mapping method")
    log.info(f"Mapped {len(series_map)} series")

    # Build mapping DataFrame
    rows = []
    for series_num, well_name in sorted(series_map.items()):
        rows.append({
            'series_number': series_num,
            'well_index': well_name,
            'mapping_method': mapping_method
        })

    mapping_df = pd.DataFrame(rows)

    # Write mapping CSV
    output_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_mapping_csv, index=False)
    log.info(f"Wrote series mapping to {output_mapping_csv}")

    # Build provenance
    provenance = {
        'mapping_method': mapping_method,
        'total_series': len(series_map),
        'source_plate_metadata': str(plate_metadata_csv),
        'source_scope_metadata': str(scope_metadata_csv),
        'mapping_summary': {
            'min_series': int(min(series_map.keys())),
            'max_series': int(max(series_map.keys())),
            'wells': sorted(series_map.values())
        },
        'warnings': []
    }

    # Check for gaps in series numbers
    series_nums = sorted(series_map.keys())
    expected_series = list(range(series_nums[0], series_nums[-1] + 1))
    gaps = set(expected_series) - set(series_nums)
    if gaps:
        warning = f"Series number gaps detected: {sorted(gaps)}"
        log.warning(warning)
        provenance['warnings'].append(warning)

    # Check for duplicate wells
    well_counts = pd.Series(list(series_map.values())).value_counts()
    duplicates = well_counts[well_counts > 1]
    if len(duplicates) > 0:
        warning = f"Duplicate well mappings: {duplicates.to_dict()}"
        log.warning(warning)
        provenance['warnings'].append(warning)

    # Write provenance
    with open(output_provenance_json, 'w') as f:
        json.dump(provenance, f, indent=2)
    log.info(f"Wrote provenance to {output_provenance_json}")

    return mapping_df


def load_series_mapping(mapping_csv: Path) -> dict:
    """
    Load series-to-well mapping from CSV.

    Args:
        mapping_csv: Path to mapping CSV

    Returns:
        Dictionary mapping series numbers to well names
    """
    df = pd.read_csv(mapping_csv)
    return dict(zip(df['series_number'], df['well_index']))
