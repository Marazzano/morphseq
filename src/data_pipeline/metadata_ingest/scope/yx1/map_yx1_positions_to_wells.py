"""
YX1 positions-to-wells mapping (CSV→CSV).

Maps YX1 ND2 positions to plate well positions using nearest-neighbor XY
matching against a reference plate grid.  Stage XY positions are read from the
scope_metadata CSV produced by extract_scope_metadata (no second ND2 open).

Vocabulary note: the ND2 P axis is a "position" (the standardized tensor axis). The mapping CSV
keeps the column ``series_number`` (the 1-based ND2 series, ``series = P + 1``) — that off-by-one
is load-bearing in the join, so it is preserved as a literal provenance column, distinct from the
0-based ``position_index`` the acquisition inventory carries.
"""

import argparse
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _load_reference_xy_coordinates(ref_csv_path: Path) -> pd.DataFrame:
    """Load reference plate XY coordinates from CSV.

    Returns DataFrame with columns [well, x_um, y_um].
    Raises FileNotFoundError if the file is absent.
    """
    if not ref_csv_path.exists():
        raise FileNotFoundError(
            f"Reference XY coordinates file not found: {ref_csv_path}. "
            "Set scope_metadata.yx1.ref_xy_csv in config.yaml."
        )
    df = pd.read_csv(ref_csv_path)
    log.info(f"Loaded reference XY coordinates: {len(df)} wells from {ref_csv_path}")
    return df


def _extract_series_xy_from_scope_csv(scope_df: pd.DataFrame) -> pd.DataFrame:
    """Extract one (raw_position_label, x_um, y_um) row per series from the scope CSV.

    The scope CSV has one row per (position, timepoint, channel).  We only need
    the T=0, first-channel row for each position to get stage XY.
    """
    required = {"raw_position_label", "x_um", "y_um", "time_int"}
    missing = required - set(scope_df.columns)
    if missing:
        raise ValueError(
            f"scope_metadata CSV is missing columns for CSV→CSV XY mapping: {sorted(missing)}. "
            "Re-run ingest_scope_metadata to regenerate it with x_um/y_um columns."
        )
    t0 = scope_df[scope_df["time_int"] == scope_df["time_int"].min()]
    # One row per position (drop duplicates from multiple channels)
    per_series = t0.drop_duplicates(subset=["raw_position_label"])[
        ["raw_position_label", "x_um", "y_um"]
    ].copy()
    per_series["P"] = pd.to_numeric(per_series["raw_position_label"], errors="coerce").astype(int)
    log.info(f"Extracted {len(per_series)} series XY positions from scope CSV")
    return per_series.reset_index(drop=True)


def _map_positions_to_wells_by_xy(
    series_positions: pd.DataFrame,
    ref_coordinates: pd.DataFrame,
    max_distance_um: float = 4500.0,
) -> tuple[dict, dict]:
    """Map P-index positions to wells via nearest-neighbour XY matching.

    Args:
        series_positions: DataFrame with columns [P, x_um, y_um]
        ref_coordinates: DataFrame with columns [well, x_um, y_um]
        max_distance_um: Reject matches beyond this distance (~half grid pitch).

    Returns:
        (mapping, diagnostics) where mapping maps P (0-based int) → well name.
    """
    from scipy.spatial import cKDTree

    log.info("Mapping series positions to wells via XY reference matching")

    ref_xy = ref_coordinates[["x_um", "y_um"]].values
    tree = cKDTree(ref_xy)

    pos_xy = series_positions[["x_um", "y_um"]].values
    distances, indices = tree.query(pos_xy, k=1)

    mapping: dict[int, str] = {}
    diagnostics: dict = {"distances": [], "rejected": [], "duplicates": []}
    wells_used: set[str] = set()

    for i, (p_idx, dist, ref_idx) in enumerate(
        zip(series_positions["P"], distances, indices)
    ):
        well = ref_coordinates.iloc[ref_idx]["well"]

        if dist > max_distance_um:
            diagnostics["rejected"].append(
                {"P": int(p_idx), "distance": float(dist), "nearest_well": well}
            )
            log.warning(
                f"  Rejected P={p_idx}: distance {dist:.1f} µm > {max_distance_um} µm"
                f" (nearest: {well})"
            )
            continue

        if well in wells_used:
            diagnostics["duplicates"].append({"P": int(p_idx), "well": well})
            log.warning(f"  Duplicate mapping: P={p_idx} → {well} (already used)")

        mapping[int(p_idx)] = well
        wells_used.add(well)
        diagnostics["distances"].append(float(dist))

    if diagnostics["distances"]:
        log.info(
            f"  Matched {len(mapping)} positions; "
            f"distance min={min(diagnostics['distances']):.1f} "
            f"max={max(diagnostics['distances']):.1f} "
            f"mean={np.mean(diagnostics['distances']):.1f} µm"
        )
    if diagnostics["rejected"]:
        log.warning(f"  Rejected {len(diagnostics['rejected'])} positions")
    if diagnostics["duplicates"]:
        log.warning(f"  Found {len(diagnostics['duplicates'])} duplicate well mappings")

    return mapping, diagnostics


def map_positions_to_wells_yx1(
    scope_metadata_csv: Path,
    output_mapping_csv: Path,
    output_provenance_json: Path,
    ref_xy_csv: Path,
    max_distance_um: float = 4500.0,
    allow_unmapped_wells: bool = False,
    row_y_tol_um: float = 1200.0,
    col_x_tol_um: float = 1200.0,
    dx_cv_tol: float = 0.15,
    dy_cv_tol: float = 0.15,
) -> pd.DataFrame:
    """Map YX1 positions to plate wells (CSV→CSV, no ND2 re-open).

    Stage XY is read from scope_metadata_csv (produced by ingest_scope_metadata).
    The reference plate grid is loaded from ref_xy_csv (config-sourced).

    Series = P + 1 (ND2 is 0-based P; series_number in the mapping is 1-based).
    """
    log.info("Mapping YX1 series to wells (CSV→CSV)")

    scope_df = pd.read_csv(scope_metadata_csv)
    log.info(f"Loaded scope metadata: {len(scope_df)} rows")

    series_positions = _extract_series_xy_from_scope_csv(scope_df)

    ref_coordinates = _load_reference_xy_coordinates(ref_xy_csv)

    from data_pipeline.metadata_ingest.scope.yx1.validate_xy_reference_grid import (
        validate_xy_reference_grid_df,
    )
    validate_xy_reference_grid_df(
        ref_coordinates,
        row_y_tol_um=float(row_y_tol_um),
        col_x_tol_um=float(col_x_tol_um),
        dx_cv_tol=float(dx_cv_tol),
        dy_cv_tol=float(dy_cv_tol),
    )

    p_to_well_map, xy_diagnostics = _map_positions_to_wells_by_xy(
        series_positions,
        ref_coordinates,
        max_distance_um=max_distance_um,
    )

    if not p_to_well_map:
        if not allow_unmapped_wells:
            raise ValueError(
                "YX1 XY reference mapping produced no results. "
                "Check ref_xy_csv path and stage XY columns in scope_metadata CSV."
            )
        log.warning("XY mapping produced no results — falling back to S-style well IDs")
        series_map: dict[int, str] = {}
        for raw in sorted(series_positions["P"].tolist()):
            series_map[int(raw) + 1] = f"S{int(raw):02d}"
        mapping_method = "unmapped_override"
    else:
        # series_number = P + 1 (1-based)
        series_map = {p + 1: well for p, well in p_to_well_map.items()}
        mapping_method = "xy_reference"
        log.info(f"Mapped {len(series_map)} wells via XY reference")

    rows = [
        {"series_number": s, "well_index": w, "mapping_method": mapping_method}
        for s, w in sorted(series_map.items())
    ]
    mapping_df = pd.DataFrame(rows)

    output_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_mapping_csv, index=False)
    log.info(f"Wrote series mapping to {output_mapping_csv}")

    # Gap / duplicate warnings
    series_nums = sorted(series_map.keys())
    expected = set(range(series_nums[0], series_nums[-1] + 1))
    gaps = expected - set(series_nums)
    warnings_list: list[str] = []
    if gaps:
        msg = f"Series number gaps: {sorted(gaps)}"
        log.warning(msg)
        warnings_list.append(msg)

    well_counts = pd.Series(list(series_map.values())).value_counts()
    dups = well_counts[well_counts > 1]
    if len(dups):
        msg = f"Duplicate well mappings: {dups.to_dict()}"
        log.warning(msg)
        warnings_list.append(msg)

    provenance = {
        "mapping_method": mapping_method,
        "total_series": len(series_map),
        "source_scope_metadata": str(scope_metadata_csv),
        "ref_xy_csv": str(ref_xy_csv),
        "max_distance_um": float(max_distance_um),
        "allow_unmapped_wells": bool(allow_unmapped_wells),
        "xy_validator": {
            "row_y_tol_um": float(row_y_tol_um),
            "col_x_tol_um": float(col_x_tol_um),
            "dx_cv_tol": float(dx_cv_tol),
            "dy_cv_tol": float(dy_cv_tol),
        },
        "mapping_summary": {
            "min_series": int(min(series_map.keys())),
            "max_series": int(max(series_map.keys())),
            "wells": sorted(series_map.values()),
        },
        "warnings": warnings_list,
    }

    if xy_diagnostics:
        provenance["xy_diagnostics"] = {
            "n_matched": len(xy_diagnostics["distances"]),
            "n_rejected": len(xy_diagnostics["rejected"]),
            "n_duplicates": len(xy_diagnostics["duplicates"]),
            "distance_stats": {
                "min": min(xy_diagnostics["distances"]) if xy_diagnostics["distances"] else None,
                "max": max(xy_diagnostics["distances"]) if xy_diagnostics["distances"] else None,
                "mean": float(np.mean(xy_diagnostics["distances"])) if xy_diagnostics["distances"] else None,
            },
            "rejected": xy_diagnostics["rejected"],
            "duplicates": xy_diagnostics["duplicates"],
        }

    with open(output_provenance_json, "w") as f:
        json.dump(provenance, f, indent=2)
    log.info(f"Wrote provenance to {output_provenance_json}")

    return mapping_df


def load_series_mapping(mapping_csv: Path) -> dict:
    """Load series-to-well mapping from CSV.  Returns {series_number: well_index}."""
    df = pd.read_csv(mapping_csv)
    return dict(zip(df["series_number"], df["well_index"]))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope-metadata-csv", type=Path, required=True)
    p.add_argument("--output-mapping-csv", type=Path, required=True)
    p.add_argument("--output-provenance-json", type=Path, required=True)
    p.add_argument("--ref-xy-csv", type=Path, required=True)
    p.add_argument("--max-distance-um", type=float, default=4500.0)
    p.add_argument("--allow-unmapped-wells", action="store_true")
    p.add_argument("--row-y-tol-um", type=float, default=1200.0)
    p.add_argument("--col-x-tol-um", type=float, default=1200.0)
    p.add_argument("--dx-cv-tol", type=float, default=0.15)
    p.add_argument("--dy-cv-tol", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    map_positions_to_wells_yx1(
        scope_metadata_csv=args.scope_metadata_csv,
        output_mapping_csv=args.output_mapping_csv,
        output_provenance_json=args.output_provenance_json,
        ref_xy_csv=args.ref_xy_csv,
        max_distance_um=args.max_distance_um,
        allow_unmapped_wells=args.allow_unmapped_wells,
        row_y_tol_um=args.row_y_tol_um,
        col_x_tol_um=args.col_x_tol_um,
        dx_cv_tol=args.dx_cv_tol,
        dy_cv_tol=args.dy_cv_tol,
    )


if __name__ == "__main__":
    main()
