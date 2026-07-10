"""
Keyence microscope metadata extraction.

Extracts scope metadata from Keyence BZ-X TIFF files and validates against schema.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import logging
import re

from data_pipeline.acquisition.metadata_ingest.scope.scope_metadata_contract import REQUIRED_COLUMNS_SCOPE_METADATA
from data_pipeline.acquisition.metadata_ingest.scope.keyence.channel_map import KEYENCE_CHANNEL_INDEX_MAP
from data_pipeline.acquisition.metadata_ingest.scope.keyence.acquisition_inventory import (
    build_keyence_acquisition_inventory,
)
from data_pipeline.acquisition.metadata_ingest.scope.keyence.raw_plane_parsing import (
    _extract_keyence_well_and_tile,
    _parse_keyence_xy_position_index,
    _parse_keyence_time_z_channel,
)
from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.shared.identifiers import build_image_id
from data_pipeline.shared.identifiers import build_well_id

log = logging.getLogger(__name__)


def _extract_time_int_from_path(file_path: Path) -> int:
    """
    Infer Keyence time index with legacy-compatible semantics.

    Preferred source is directory token `T####`; if absent, use filename token
    `_T####_Z...`. Layouts without explicit T are treated as single-timepoint.
    """
    for part in file_path.parts:
        t_match = re.fullmatch(r'T(\d+)', part, flags=re.IGNORECASE)
        if t_match:
            return max(int(t_match.group(1)) - 1, 0)

    name_match = re.search(r'_T(\d+)_Z\d+_CH\d+', file_path.name, flags=re.IGNORECASE)
    if name_match:
        return max(int(name_match.group(1)) - 1, 0)

    return 0


def _scrape_keyence_metadata(tiff_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from Keyence TIFF file.

    Keyence BZ-X microscopes embed XML metadata in TIFF files between <Data> tags.

    Args:
        tiff_path: Path to Keyence TIFF file

    Returns:
        Dictionary with metadata fields
    """
    def _findnth(haystack, needle, n):
        """Find the nth occurrence of needle in haystack."""
        parts = haystack.split(needle, n + 1)
        if len(parts) <= n + 1:
            return -1
        return len(haystack) - len(parts[-1]) - len(needle)

    with open(tiff_path, 'rb') as f:
        fulldata = f.read()

    # Extract XML metadata between <Data> tags
    metadata = fulldata.partition(b'<Data>')[2].partition(b'</Data>')[0].decode()

    meta_dict = {}
    keyword_list = ['ShootingDateTime', 'LensName', 'Observation Type', 'Width', 'Height', 'Width', 'Height']
    outname_list = ['Time (s)', 'Objective', 'Channel', 'Width (px)', 'Height (px)', 'Width (um)', 'Height (um)']

    for k in range(len(keyword_list)):
        param_string = keyword_list[k]
        name = outname_list[k]

        if (param_string == 'Width') or (param_string == 'Height'):
            if 'um' in name:
                ind1 = _findnth(metadata, param_string + ' Type', 2)
                ind2 = _findnth(metadata, '/' + param_string, 2)
            else:
                ind1 = _findnth(metadata, param_string + ' Type', 1)
                ind2 = _findnth(metadata, '/' + param_string, 1)
        else:
            ind1 = metadata.find(param_string)
            ind2 = metadata.find('/' + param_string)

        long_string = metadata[ind1:ind2]
        subind1 = long_string.find(">")
        subind2 = long_string.find("<")
        param_val = long_string[subind1+1:subind2]

        sysind = long_string.find("System.")
        dtype = long_string[sysind+7:subind1-1]
        if 'Int' in dtype:
            param_val = int(param_val)

        if param_string == "ShootingDateTime":
            # Convert from 100 nanoseconds to seconds
            param_val = float(param_val) / 10 / 1000 / 1000
        elif "um" in name:
            # Convert from nanometers to micrometers
            param_val = float(param_val) / 1000

        meta_dict[name] = param_val

    return meta_dict


def _scrape_keyence_plane_metadata(tiff_path: Path) -> Dict[str, Any]:
    """Adapt ``_scrape_keyence_metadata`` to the acquisition-inventory scraper contract.

    The acquisition inventory wants canonical field names + a derived ``micrometers_per_pixel``; the
    raw scraper returns Keyence-labelled keys (``'Width (um)'``, ``'Time (s)'``, ...). This adapter is
    the seam ``build_keyence_acquisition_inventory`` calls per plane (and that tests stub). The channel
    NAME is included only when the proprietary XML actually carried one — the inventory anchors channel
    identity on the filename ``CH#`` index, not this name.
    """
    meta = _scrape_keyence_metadata(tiff_path)
    width_px = meta.get("Width (px)", 1)
    width_um = meta.get("Width (um)", 0)
    micrometers_per_pixel = (width_um / width_px) if width_px else 0.0
    raw_channel_name = meta.get("Channel")
    return {
        "micrometers_per_pixel": micrometers_per_pixel,
        "image_width_px": meta.get("Width (px)", 0),
        "image_height_px": meta.get("Height (px)", 0),
        "objective_magnification": meta.get("Objective", "unknown"),
        "acquisition_time_s": meta.get("Time (s)", 0.0),
        "raw_channel_name": raw_channel_name,
    }


def _channel_index_from_path(tiff_path: Path) -> int:
    """Return the reliable on-disk ``CH#`` channel index for a Keyence plane filename.

    Keyence channel NAME metadata is proprietary/unreliable, so the filename ``CH#`` index is the
    single trustworthy channel signal. Fail loud if a ``*CH*.tif`` lacks a parseable ``CH#`` token.
    """
    parsed = _parse_keyence_time_z_channel(tiff_path)
    if parsed is None:
        raise ValueError(
            f"Keyence: cannot parse a 'CH#' channel index from {tiff_path.name!r}. "
            "Expected a filename like '...XY##_NNNNN_Z###_CH#.tif'."
        )
    _time_index, _z_index, channel_index = parsed
    return channel_index


def _to_channel_id(channel_index: int) -> str:
    """Map a Keyence ``CH#`` channel index to its canonical channel_id (exact-match; fail loud).

    Anchored on the reliable filename index (not the proprietary scraped name) via the single
    ``KEYENCE_CHANNEL_INDEX_MAP``. An unmapped index raises — add the real channel, never default.
    """
    return KEYENCE_CHANNEL_INDEX_MAP.to_canonical(channel_index)


def _discover_keyence_files(raw_data_dir: Path, experiment_id: str) -> List[Path]:
    """
    Discover Keyence TIFF files in raw data directory.

    Keyence file structure can vary:
    - {exp}/XY##/{files}  (multi-well format)
    - {exp}/W0##/{files}  (cytometer format)
    - {exp}/{files}       (flat format)

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        experiment_id: Experiment identifier

    Returns:
        List of TIFF file paths
    """
    exp_dir = raw_data_dir / experiment_id

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Try different Keyence file patterns
    tiff_files = []

    # Pattern 1: Files with CH (channel) indicator
    tiff_files = list(exp_dir.rglob("*CH*.tif"))

    if not tiff_files:
        # Pattern 2: Any TIFF files
        tiff_files = list(exp_dir.rglob("*.tif"))

    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {exp_dir}")

    log.info(f"Discovered {len(tiff_files)} Keyence TIFF files in {exp_dir}")
    return sorted(tiff_files)


def _extract_well_from_path(file_path: Path) -> str:
    """
    Extract well identifier from Keyence file path.

    Common patterns:
    - XY##a/... → "A##" format
    - W0##/... → well index
    - Filename contains well info

    Args:
        file_path: Path to Keyence TIFF file

    Returns:
        Well identifier (e.g., "A01", "B12")
    """
    parsed_well, _tile_id = _extract_keyence_well_and_tile(file_path)
    if parsed_well is not None:
        return parsed_well

    # Check for legacy XY pattern in path (e.g., XY01a)
    for part in file_path.parts:
        if part.startswith('XY'):
            suffix = part[2:]
            # Legacy XY01a format
            well_num = part[2:4]
            well_letter = part[-1].upper()
            if well_num.isdigit() and well_letter.isalpha():
                return f"{well_letter}{well_num}"

    # Check for W0 pattern (W001 → A01)
    for part in file_path.parts:
        if part.startswith('W0'):
            well_idx = int(part[1:])
            # Convert to row/col (1-indexed, 12 cols per row)
            row = (well_idx - 1) // 12
            col = (well_idx - 1) % 12 + 1
            return f"{chr(65 + row)}{col:02d}"

    # Fallback: extract from filename
    filename = file_path.name
    # Look for patterns like "A01", "B12", etc.
    import re
    match = re.search(r'[A-H](0[1-9]|1[0-2])', filename)
    if match:
        return match.group(0)

    log.warning(f"Could not extract well from path: {file_path}")
    return "unknown"


def _extract_position_label_within_well(file_path: Path) -> str:
    """Return the Keyence acquisition-position token within a well directory."""
    for part in file_path.parts:
        if re.fullmatch(r"P\d+", part, flags=re.IGNORECASE):
            return part.upper()
    return "P0"


def extract_keyence_scope_metadata(
    raw_data_dir: Path,
    experiment_id: str,
    output_csv: Path,
    acquisition_inventory_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Extract Keyence scope metadata from raw TIFF files.

    Reads Keyence BZ-X microscope TIFF files, extracts embedded metadata,
    normalizes channel names, and validates against schema.

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        experiment_id: Experiment identifier
        output_csv: Path to write validated scope_series_metadata_raw.csv
        acquisition_inventory_csv: Optional output path for the maximal per-coordinate
            ``acquisition_inventory__keyence.csv`` (one row per raw plane —
            (well, tile, z_index, channel_index, time_index)). Built from a RAW-PLANE scan, NOT from
            the collapsed FF rows this function emits — the inventory is the never-collapsed system of
            record. Nothing downstream consumes it yet (Stage A).

    Returns:
        Validated DataFrame with scope metadata

    Raises:
        FileNotFoundError: If data directory or files not found
        ValueError: If validation fails
    """
    experiment_id = str(experiment_id).strip()
    log.info(f"Extracting Keyence scope metadata for {experiment_id}")

    # Discover TIFF files
    tiff_files = _discover_keyence_files(raw_data_dir, experiment_id)
    # Extract metadata from each file
    rows = []
    for tiff_path in tiff_files:
        try:
            meta = _scrape_keyence_metadata(tiff_path)

            # Extract well from path
            well_index = _extract_well_from_path(tiff_path)
            position_index = _parse_keyence_xy_position_index(tiff_path)
            if position_index is None:
                raise ValueError(
                    f"Could not parse Keyence XY acquisition position from {tiff_path}. "
                    "Expected path to contain a directory like 'XY13'."
                )

            # Resolve channel_id from the reliable filename CH# index (proprietary names unreliable).
            channel_index = _channel_index_from_path(tiff_path)
            normalized_channel = _to_channel_id(channel_index)
            raw_channel = meta.get('Channel') or f"CH{channel_index}"

            # Compute micrometers per pixel
            width_um = meta.get('Width (um)', 0)
            width_px = meta.get('Width (px)', 1)
            micrometers_per_pixel = width_um / width_px if width_px > 0 else 0

            # Legacy-compatible time parsing:
            # - T#### directory or _T####_ token => true timepoint
            # - otherwise single-timepoint acquisition (time_int=0)
            time_int = _extract_time_int_from_path(tiff_path)

            # Build row
            well_id = build_well_id(experiment_id, well_index)
            row = {
                'experiment_id': experiment_id,
                'raw_position_label': str(position_index),
                'position_index': position_index,
                'well_index': well_index,
                'well_id': well_id,
                'time_int': time_int,
                'image_id': build_image_id(well_id, normalized_channel, time_int),

                # Spatial calibration
                'micrometers_per_pixel': micrometers_per_pixel,
                'image_width_px': meta.get('Width (px)', 0),
                'image_height_px': meta.get('Height (px)', 0),
                'objective_magnification': meta.get('Objective', 'unknown'),
                'x_um': np.nan,
                'y_um': np.nan,

                # Temporal calibration
                'absolute_start_time': meta.get('Time (s)', 0),
                'experiment_time_s': meta.get('Time (s)', 0),
                'frame_interval_s': 0,  # Will compute after sorting

                # Acquisition metadata
                'microscope_id': 'Keyence',
                'channel': normalized_channel,
                'z_position': 0,  # Keyence FF images are single Z

                # Provenance
                'raw_channel_name': raw_channel,
                'source_file': str(tiff_path),
            }

            rows.append(row)

        except Exception as e:
            log.warning(f"Failed to extract metadata from {tiff_path}: {e}")
            continue

    if not rows:
        raise ValueError(f"No valid metadata extracted for {experiment_id}")

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Sort by well and time
    df = df.sort_values(['well_index', 'time_int']).reset_index(drop=True)

    # Compute frame_interval_s
    # Group by well and compute time differences
    def compute_intervals(group):
        if len(group) > 1:
            # Compute median interval for this well
            times = group['experiment_time_s'].values
            intervals = np.diff(times)
            median_interval = np.median(intervals) if len(intervals) > 0 else 0
            group['frame_interval_s'] = median_interval
        else:
            group['frame_interval_s'] = 0
        return group

    df = df.groupby('well_index', group_keys=False).apply(compute_intervals)

    # Adjust experiment_time_s to be relative to experiment start
    min_time = df['experiment_time_s'].min()
    df['experiment_time_s'] = df['experiment_time_s'] - min_time

    # Validate against schema.
    # x_um / y_um are NaN on Keyence (BZ-X stage coordinates not exposed in TIFF XML);
    # they are declared in the shared schema for parity and populated in Stage E (if needed).
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_SCOPE_METADATA,
        stage_name="Keyence scope metadata extraction",
        nullable_columns=["x_um", "y_um"],
    )

    # Write output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"Wrote Keyence scope metadata to {output_csv}")

    # Emit the maximal acquisition inventory from a RAW-PLANE scan (record-only; Stage A).
    # One row per raw TIFF plane (well, tile, z, channel, time) — never collapsed. This deliberately
    # re-scans the raw tree (not the FF-collapsed `df` above) because the inventory must keep per-Z
    # grain that the FF rows discard.
    if acquisition_inventory_csv is not None:
        inventory_df = build_keyence_acquisition_inventory(
            experiment_id=experiment_id,
            raw_data_dir=raw_data_dir / experiment_id,
            scrape_plane_metadata=_scrape_keyence_plane_metadata,
        )
        acquisition_inventory_csv = Path(acquisition_inventory_csv)
        acquisition_inventory_csv.parent.mkdir(parents=True, exist_ok=True)
        inventory_df.to_csv(acquisition_inventory_csv, index=False)
        log.info(
            f"Wrote Keyence acquisition inventory ({len(inventory_df)} rows) to "
            f"{acquisition_inventory_csv}"
        )

    return df



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-keyence-experiment-dir", type=Path, required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument(
        "--acquisition-inventory-csv",
        type=Path,
        default=None,
        help="Optional output path for acquisition_inventory__keyence.csv (record-only).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    extract_keyence_scope_metadata(
        raw_data_dir=args.raw_keyence_experiment_dir,
        experiment_id=args.experiment_id,
        output_csv=args.output_csv,
        acquisition_inventory_csv=args.acquisition_inventory_csv,
    )


if __name__ == "__main__":
    main()
