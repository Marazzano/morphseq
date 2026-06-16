"""
YX1 microscope metadata extraction.

Extracts scope metadata from YX1 ND2 files and produces validated scope_series_metadata_raw.csv.
"""

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
import nd2

from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA
from data_pipeline.schemas.channel_normalization import CHANNEL_NORMALIZATION_MAP
from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.metadata_ingest.scope.yx1.acquisition_inventory import (
    build_yx1_acquisition_inventory,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _find_nd2_file(raw_data_dir: Path) -> Path:
    """Find the ND2 file in the raw data directory."""
    nd2_files = list(raw_data_dir.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No ND2 files found in {raw_data_dir}")
    if len(nd2_files) > 1:
        raise RuntimeError(f"Multiple ND2 files found in {raw_data_dir}: {nd2_files}")
    return nd2_files[0]


def _extract_timestamps(nd: nd2.ND2File, n_t: int, n_w: int, n_z: int, n_c: int = 1) -> np.ndarray:
    """
    Extract timestamps from ND2 file with gap imputation.

    Args:
        nd: Open ND2File object
        n_t: Number of timepoints
        n_w: Number of wells
        n_z: Number of Z slices

    Returns:
        Array of timestamps in seconds (length n_t)
    """
    # Extract timestamps from first well
    times = np.full((n_t,), np.nan, dtype=float)

    for t in range(n_t):
        seq = t * n_w * n_z * max(n_c, 1)  # First well, first z-slice, first channel
        try:
            times[t] = nd.frame_metadata(seq).channels[0].time.relativeTimeMs / 1000.0
        except Exception:
            continue  # Leave as NaN

    valid_count = (~np.isnan(times)).sum()
    log.info(f"Extracted {valid_count}/{n_t} valid timestamps from ND2")

    # Calculate cycle time from valid data
    s = pd.Series(times)
    original_valid = s.dropna()

    if len(original_valid) >= 2:
        original_diffs = original_valid.diff().dropna()
        cycle_time = original_diffs.median()
        log.info(f"Calculated cycle time: {cycle_time:.2f}s")
    else:
        cycle_time = 1800.0  # 30 minutes default
        log.info(f"Using default cycle time: {cycle_time:.2f}s")

    # Impute missing values
    if s.isna().any():
        missing_count = s.isna().sum()
        log.info(f"Imputing {missing_count} missing timestamps...")

        if len(original_valid) > 0:
            first_valid_idx = s.first_valid_index()
            last_valid_idx = s.last_valid_index()

            # Fill backwards from first valid
            if first_valid_idx > 0:
                first_time = s.iloc[first_valid_idx]
                for i in range(first_valid_idx - 1, -1, -1):
                    s.iloc[i] = first_time - (first_valid_idx - i) * cycle_time

            # Fill forwards from last valid
            if last_valid_idx < len(s) - 1:
                last_time = s.iloc[last_valid_idx]
                for i in range(last_valid_idx + 1, len(s)):
                    s.iloc[i] = last_time + (i - last_valid_idx) * cycle_time

            # Fill middle gaps
            for i in range(len(s)):
                if pd.isna(s.iloc[i]):
                    s.iloc[i] = s.iloc[0] + i * cycle_time

    # Last resort
    if s.isna().all():
        log.warning("No valid timestamps - using default intervals")
        s = pd.Series(np.arange(n_t, dtype=float) * 1800.0)

    # Ensure monotonic
    s = s.cummax()

    return s.to_numpy()


def _normalize_channel_name(raw_name: str) -> str:
    """
    Normalize YX1 channel name to standard name.

    Args:
        raw_name: Raw channel name from ND2

    Returns:
        Normalized channel name (e.g., 'BF', 'GFP')
    """
    # Try direct mapping first
    if raw_name in CHANNEL_NORMALIZATION_MAP:
        return CHANNEL_NORMALIZATION_MAP[raw_name]

    # Try case-insensitive match for common patterns
    raw_lower = raw_name.lower()

    # Check for BF variations
    if any(x in raw_lower for x in ['dia', 'empty', 'brightfield', 'bf']):
        return 'BF'

    # Check for fluorescence channels
    if 'gfp' in raw_lower:
        return 'GFP'
    if 'rfp' in raw_lower or 'mcherry' in raw_lower:
        return 'RFP'

    # Default: return as-is and warn
    log.warning(f"Unknown channel name '{raw_name}' - using as-is")
    return raw_name


def extract_yx1_scope_metadata(
    raw_data_dir: Path,
    output_csv: Path,
    experiment_id: str,
    acquisition_inventory_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Extract YX1 scope metadata from ND2 file.

    Args:
        raw_data_dir: Directory containing ND2 file
        output_csv: Output path for scope_series_metadata_raw.csv
        experiment_id: Experiment identifier
        acquisition_inventory_csv: Optional output path for the maximal per-coordinate
            ``acquisition_inventory__yx1.csv`` (one row per (position, z, channel, time)). When
            given, it is emitted from the SAME single ND2 read — record-only, nothing downstream
            consumes it yet.

    Returns:
        DataFrame with validated scope metadata
    """
    experiment_id = str(experiment_id).strip()
    log.info(f"Extracting YX1 scope metadata for {experiment_id}")

    # Find and open ND2 file
    nd2_path = _find_nd2_file(raw_data_dir)
    log.info(f"Reading ND2 file: {nd2_path}")

    with nd2.ND2File(nd2_path) as nd:
        # Get dimensions (supports both T,W,Z,C,Y,X and T,W,Z,Y,X layouts)
        shape = nd.shape
        n_t, n_w, n_z = shape[:3]
        n_c = int(shape[3]) if len(shape) >= 6 else 1
        log.info(f"ND2 shape: T={n_t}, W={n_w}, Z={n_z}")

        # Get spatial calibration
        voxel_size = nd.voxel_size()
        micrometers_per_pixel = voxel_size[0]  # X dimension
        image_height_px = shape[-2]  # Y
        image_width_px = shape[-1]   # X

        # Get channel names + the full channel mapping (index ↔ raw name ↔ normalized token).
        # This triple is recorded once in the acquisition inventory instead of being re-derived
        # (and partly lost) downstream at the join and at stitch.
        channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]
        log.info(f"Raw channel names: {channel_names}")
        channel_mapping = [
            (idx, _normalize_channel_name(raw_name), raw_name)
            for idx, raw_name in enumerate(channel_names)
        ]

        # Get objective info
        try:
            objective = nd.frame_metadata(0).channels[0].microscope.objectiveName
        except:
            objective = "Unknown"

        # Extract timestamps
        timestamps = _extract_timestamps(nd, n_t, n_w, n_z, n_c=n_c)

        # Calculate frame interval
        if len(timestamps) >= 2:
            frame_interval_s = float(np.median(np.diff(timestamps)))
        else:
            frame_interval_s = 1800.0  # Default 30 min

        log.info(f"Frame interval: {frame_interval_s:.2f}s")

        # Extract stage XY positions (T=0, one per series/position)
        stage_xy: dict[int, tuple[float, float]] = {}
        for w_idx in range(n_w):
            # Frame index at T=0 for position w_idx
            idx = w_idx * n_z * max(n_c, 1)
            try:
                md = nd.frame_metadata(idx)
                ch0 = getattr(md, "channels", [None])[0]
                if ch0 and hasattr(ch0, "position"):
                    stage = ch0.position.stagePositionUm
                    stage_xy[w_idx] = (
                        getattr(stage, "x", float("nan")),
                        getattr(stage, "y", float("nan")),
                    )
            except Exception:
                stage_xy[w_idx] = (float("nan"), float("nan"))

        # Build metadata rows: one row per (position, timepoint, channel).
        # raw_position_label is the raw ND2 P-index as a string — not a well label.
        # well_id is minted later at join_series_mapping_to_scope_metadata.
        rows = []

        for w_idx in range(n_w):
            raw_position_label = str(w_idx)
            x_um, y_um = stage_xy.get(w_idx, (float("nan"), float("nan")))

            for t_idx in range(n_t):
                time_s = timestamps[t_idx]

                for raw_channel in channel_names:
                    channel = _normalize_channel_name(raw_channel)
                    row = {
                        'experiment_id': experiment_id,
                        'raw_position_label': raw_position_label,
                        'time_int': t_idx,
                        'x_um': x_um,
                        'y_um': y_um,
                        'micrometers_per_pixel': micrometers_per_pixel,
                        'image_width_px': image_width_px,
                        'image_height_px': image_height_px,
                        'objective_magnification': objective,
                        'frame_interval_s': frame_interval_s,
                        'absolute_start_time': timestamps[0],
                        'experiment_time_s': time_s,
                        'microscope_id': 'YX1',
                        'channel': channel,
                        'z_position': 0,
                    }
                    rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(rows)

    log.info(f"Created metadata with {len(df)} rows")
    log.info(f"Positions: {df['raw_position_label'].nunique()}, Timepoints: {df['time_int'].nunique()}, Channels: {df['channel'].nunique()}")

    # Validate schema
    validate_dataframe_schema(df, REQUIRED_COLUMNS_SCOPE_METADATA, "YX1 scope metadata")

    # Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"Wrote scope metadata to {output_csv}")

    # Emit the maximal acquisition inventory from the SAME single ND2 read (record-only).
    # One row per full tensor coordinate (position, z, channel, time) — Z exploded, channel
    # mapping preserved. Nothing downstream consumes it yet.
    if acquisition_inventory_csv is not None:
        inventory_df = build_yx1_acquisition_inventory(
            experiment_id=experiment_id,
            n_t=n_t,
            n_z=n_z,
            timestamps=timestamps,
            channels=channel_mapping,
            stage_xy=stage_xy,
            micrometers_per_pixel=micrometers_per_pixel,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            objective_magnification=objective,
            source_nd2_path=nd2_path,
        )
        acquisition_inventory_csv = Path(acquisition_inventory_csv)
        acquisition_inventory_csv.parent.mkdir(parents=True, exist_ok=True)
        inventory_df.to_csv(acquisition_inventory_csv, index=False)
        log.info(
            f"Wrote acquisition inventory ({len(inventory_df)} rows) to "
            f"{acquisition_inventory_csv}"
        )

    return df



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-yx1-experiment-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument(
        "--acquisition-inventory-csv",
        type=Path,
        default=None,
        help="Optional output path for acquisition_inventory__yx1.csv (record-only).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    extract_yx1_scope_metadata(
        raw_data_dir=args.raw_yx1_experiment_dir,
        output_csv=args.output_csv,
        experiment_id=args.experiment_id,
        acquisition_inventory_csv=args.acquisition_inventory_csv,
    )


if __name__ == "__main__":
    main()
