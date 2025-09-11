#!/usr/bin/env python3
"""
Build03 (direct): Generate per-experiment embryo metadata CSV from SAM2 per-experiment outputs.

Goal: Validate per-experiment input discovery and write a per-experiment CSV with
stable headers, without running the entire pipeline.

Inputs (defaults under --data-root):
- sam2_csv: sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv
- masks_dir: sam2_pipeline_files/exported_masks/{exp}/masks/
- mask_manifest: sam2_pipeline_files/exported_masks/{exp}/mask_export_manifest_{exp}.json (optional)
- built01_csv: metadata/built_metadata_files/{exp}_metadata.csv (optional)

Output:
- metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv

Note: This initial implementation focuses on I/O wiring and schema. Geometry
      fields are left blank for now; we can fill them in future iterations.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import re
import math
import pandas as pd

try:
    import numpy as np
    import cv2
    _HAS_IMAGE_LIBS = True
except Exception:
    _HAS_IMAGE_LIBS = False


@dataclass
class Inputs:
    root: Path
    exp: str
    sam2_csv: Path
    masks_dir: Path
    mask_manifest: Path
    built01_csv: Path
    sam2_json: Path


def _default_inputs(root: Path, exp: str) -> Inputs:
    return Inputs(
        root=root,
        exp=exp,
        sam2_csv=root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp}.csv",
        masks_dir=root / "sam2_pipeline_files" / "exported_masks" / exp / "masks",
        mask_manifest=root / "sam2_pipeline_files" / "exported_masks" / exp / f"mask_export_manifest_{exp}.json",
        built01_csv=root / "metadata" / "built_metadata_files" / f"{exp}_metadata.csv",
        sam2_json=root / "sam2_pipeline_files" / "segmentation" / f"grounded_sam_segmentations_{exp}.json",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Build03 directly for a single experiment (per-experiment I/O)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True, help="Data root (e.g., morphseq_playground)")
    p.add_argument("--exp", required=True, help="Experiment ID (e.g., 20250529_36hpf_ctrl_atf6)")

    # Optional overrides
    p.add_argument("--sam2-csv", help="Path to per-experiment SAM2 metadata CSV")
    p.add_argument("--sam2-json", help="Path to per-experiment SAM2 JSON (fallback)")
    p.add_argument("--masks-dir", help="Path to per-experiment masks dir")
    p.add_argument("--mask-manifest", help="Path to per-experiment mask manifest JSON")
    p.add_argument("--built01-csv", help="Path to Build01 per-experiment metadata CSV")
    p.add_argument("--out-dir", help="Output dir for Build03 per-experiment CSV")

    # Optional metadata column overrides
    p.add_argument("--well-col", help="Column name for well in Build01 CSV")
    p.add_argument("--px-size-col", help="Single pixel-size column (um/pixel) in Build01 CSV")
    p.add_argument("--px-size-x-col", help="X pixel-size column (um/pixel) in Build01 CSV")
    p.add_argument("--px-size-y-col", help="Y pixel-size column (um/pixel) in Build01 CSV")

    # Behavior
    p.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if exists")
    p.add_argument("--validate-only", action="store_true", help="Validate inputs, do not write output")
    p.add_argument("--no-manifest-check", action="store_true", help="Skip manifest existence check")
    p.add_argument("--compute-geometry", action="store_true", help="Compute geometry from labeled mask images (optional)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def _derive_video_and_well(video_or_image_id: str) -> tuple[str, str]:
    # Accept both video_id (…_A01) or image_id (…_A01_ch00_t0000)
    # Extract the well token pattern _<A-H><##>
    m = re.search(r"_([A-H][0-9]{2})", video_or_image_id)
    well = m.group(1) if m else ""
    # For video id: split off suffix; for image id: strip channel/time
    if "_ch" in video_or_image_id:
        # Assume image id; video id is prefix up to the well
        vid = video_or_image_id.split("_ch")[0]
    else:
        vid = video_or_image_id
    return vid, well


def _derive_time_int(image_id: str) -> Optional[int]:
    m = re.search(r"_t(\d{4})$", image_id)
    return int(m.group(1)) if m else None


def _log(v: bool, msg: str):
    if v:
        print(msg)


def _ensure_predicted_stage_hpf(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Add `predicted_stage_hpf` using the legacy Kimmel-style formula if missing.
    
    Formula (hours post fertilization):
      start_age_hpf + (Time Rel (s)/3600) * (0.055*temperature - 0.57)
    
    Only applies if the requisite columns exist. Otherwise, the input is
    returned unchanged.
    
    This is the exact same logic from the legacy build03A_process_images.py.
    """
    needed = {"start_age_hpf", "Time Rel (s)", "temperature"}
    
    if verbose:
        print(f"      • DataFrame columns: {list(df.columns)}")
        print(f"      • Needed columns: {needed}")
        print(f"      • Columns present: {needed.intersection(set(df.columns))}")
        print(f"      • All needed present: {needed.issubset(df.columns)}")
        print(f"      • predicted_stage_hpf already exists: {'predicted_stage_hpf' in df.columns}")
    
    if needed.issubset(df.columns):
        try:
            df = df.copy()
            
            if verbose:
                print(f"      • Sample values:")
                for col in needed:
                    sample_vals = df[col].head(3).tolist()
                    print(f"        - {col}: {sample_vals}")
            
            df["predicted_stage_hpf"] = (
                df["start_age_hpf"].astype(float)
                + (df["Time Rel (s)"].astype(float) / 3600.0)
                  * (0.055 * df["temperature"].astype(float) - 0.57)
            )
            
            if verbose:
                print(f"      • Calculated predicted_stage_hpf: {df['predicted_stage_hpf'].head(3).tolist()}")
                
        except Exception as e:
            # Leave silently unchanged if types are malformed; downstream code will not rely on this.
            if verbose:
                print(f"      • ERROR in calculation: {e}")
            pass
    return df


def _collect_rows_from_sam2_csv(csv_path: Path, exp: str, verbose: bool = False) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        need_any = {"image_id", "embryo_id"}
        if not need_any.issubset(header):
            raise ValueError(f"CSV missing required columns: {need_any - header}")
        for r in reader:
            image_id = r.get("image_id", "").strip()
            embryo_id = r.get("embryo_id", "").strip()
            if not image_id or not embryo_id:
                continue
            video_id = r.get("video_id") or _derive_video_and_well(image_id)[0]
            well_id = r.get("well_id") or _derive_video_and_well(video_id)[1]
            snip_id = r.get("snip_id") or (f"{embryo_id}_t{_derive_time_int(image_id):04d}" if _derive_time_int(image_id) is not None else "")
            time_int = r.get("time_int") or (str(_derive_time_int(image_id)) if _derive_time_int(image_id) is not None else "")
            
            out = {
                "exp_id": exp,
                "video_id": video_id,
                "well_id": well_id,
                "image_id": image_id,
                "embryo_id": embryo_id,
                "snip_id": snip_id,
                "time_int": time_int,
                # Geometry placeholders (to be populated later)
                "area_px": "",
                "perimeter_px": "",
                "centroid_x_px": "",
                "centroid_y_px": "",
                "area_um2": "",
                "perimeter_um": "",
                "centroid_x_um": "",
                "centroid_y_um": "",
                # Provenance
                "exported_mask_path": r.get("exported_mask_path", ""),
                "sam2_source_json": "",
                "computed_at": datetime.now().isoformat(),
                # Flags
                "use_embryo_flag": "true",
                "predicted_stage_hpf": "",  # Will be calculated at DataFrame level
                "notes": "",
                # Include raw metadata for stage calculation
                "start_age_hpf": r.get("start_age_hpf", ""),
                "Time Rel (s)": r.get("Time Rel (s)", "") or r.get("relative_time_s", ""),
                "temperature": r.get("temperature", ""),
                # Include pixel scale data for geometry calculation
                "width_um": r.get("width_um", ""),
                "width_px": r.get("width_px", ""), 
                "height_um": r.get("height_um", ""),
                "height_px": r.get("height_px", ""),
            }
            rows.append(out)
    _log(verbose, f"   • Collected {len(rows)} rows from {csv_path.name}")
    return rows


def main() -> int:
    args = _parse_args()
    root = Path(args.data_root)
    exp = args.exp
    verbose = args.verbose

    defaults = _default_inputs(root, exp)
    sam2_csv = Path(args.sam2_csv) if args.sam2_csv else defaults.sam2_csv
    sam2_json = Path(args.sam2_json) if args.sam2_json else defaults.sam2_json
    masks_dir = Path(args.masks_dir) if args.masks_dir else defaults.masks_dir
    manifest = Path(args.mask_manifest) if args.mask_manifest else defaults.mask_manifest
    built01_csv = Path(args.built01_csv) if args.built01_csv else defaults.built01_csv

    # Default output: deposit directly under a flat per-experiment folder
    # e.g., metadata/build_03_output/expr_embryo_metadata_{exp}.csv
    out_dir = Path(args.out_dir) if args.out_dir else (root / "metadata" / "build03_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"expr_embryo_metadata_{exp}.csv"

    print(f"🏃 Build03 (direct) for {exp}")
    print(f"📁 Data root: {root}")
    print(f"📄 SAM2 CSV: {sam2_csv}")
    print(f"🗂️  Masks dir: {masks_dir}")
    print(f"🧾 Manifest: {manifest}")
    # If default Build01 CSV missing and no override, try auto-discovery by suffix
    if not built01_csv.exists() and not args.built01_csv:
        cand = _autodiscover_built01_metadata(root, exp, verbose=verbose)
        if cand:
            built01_csv = cand
    print(f"🧬 Build01 CSV: {built01_csv}")
    print(f"🖊️  Output CSV: {out_csv}")

    # Input validation (lightweight for now)
    if not sam2_csv.exists():
        print(f"❌ Missing SAM2 per-experiment CSV: {sam2_csv}")
        return 2
    if not masks_dir.exists():
        _log(verbose, f"⚠️ Masks dir not found (will proceed without geometry): {masks_dir}")
    if not manifest.exists() and not args.no_manifest_check:
        _log(verbose, f"⚠️ Manifest not found: {manifest}")
    if not built01_csv.exists():
        _log(verbose, f"⚠️ Built01 CSV not found; scale fields will remain NA: {built01_csv}")

    if out_csv.exists() and not args.overwrite and not args.validate_only:
        print(f"ℹ️  Output exists and --overwrite not set; skipping write: {out_csv}")
        return 0

    # Collect rows from SAM2 CSV
    rows = _collect_rows_from_sam2_csv(sam2_csv, exp, verbose=verbose)
    
    # Convert to DataFrame for stage calculation, then back to list of dicts
    df = pd.DataFrame(rows)
    _log(verbose, f"   • Converting to DataFrame for predicted_stage_hpf calculation...")
    df = _ensure_predicted_stage_hpf(df, verbose=verbose)
    rows = df.to_dict('records')
    _log(verbose, f"   • Applied predicted_stage_hpf calculation to {len(rows)} rows")

    # Optionally compute geometry if image libs available and masks are present
    if args.compute_geometry and _HAS_IMAGE_LIBS and masks_dir.exists():
        _log(verbose, "🧮 Computing geometry from labeled masks…")
        scale_map = _load_scale_map(
            built01_csv,
            well_col_override=args.well_col,
            x_col_override=args.px_size_x_col,
            y_col_override=args.px_size_y_col,
            single_col_override=args.px_size_col,
            verbose=verbose,
        )
        for r in rows:
            _compute_row_geometry(r, masks_dir, scale_map, verbose=verbose)
    else:
        if args.compute_geometry and not _HAS_IMAGE_LIBS:
            _log(verbose, "⚠️ Image libraries (cv2/numpy) unavailable; skipping geometry computation")
        if args.compute_geometry and not masks_dir.exists():
            _log(verbose, f"⚠️ Masks dir missing; skipping geometry: {masks_dir}")
    if args.validate_only:
        print("✅ Inputs validated and rows collectable; not writing output due to --validate-only")
        return 0

    # Write output CSV (schema-first, values partial)
    fieldnames = [
        "exp_id","video_id","well_id","image_id","embryo_id","snip_id","time_int",
        "area_px","perimeter_px","centroid_x_px","centroid_y_px",
        "area_um2","perimeter_um","centroid_x_um","centroid_y_um",
        "exported_mask_path","sam2_source_json","computed_at",
        "use_embryo_flag","predicted_stage_hpf","notes",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"✅ Build03 CSV written: {out_csv}")
    print(f"   Rows: {len(rows)}")
    return 0


    # Fill provenance for SAM2 JSON if present
    for r in rows:
        r["sam2_source_json"] = str(sam2_json) if sam2_json.exists() else r.get("sam2_source_json", "")

    return 0

# ------------------------
# Geometry helper functions
# ------------------------

def _parse_embryo_number(embryo_id: str) -> Optional[int]:
    m = re.search(r"_e(\d+)$", embryo_id)
    if not m:
        return None
    return int(m.group(1).lstrip("0") or "0")


def _derive_mask_path(masks_dir: Path, image_id: str, embryo_count_hint: Optional[int]) -> Path:
    # Default exporter naming: {image_id}_masks_emnum_{N}.png
    n = embryo_count_hint if embryo_count_hint and embryo_count_hint > 0 else 99
    return masks_dir / f"{image_id}_masks_emnum_{n}.png"


def _compute_row_geometry(row: Dict[str, str], masks_dir: Path, scale_map: Dict[str, tuple[float, float]], verbose: bool = False) -> None:
    """Populate px/um geometry for a row using its labeled mask image.

    Assumes labeled mask pixels equal embryo_number parsed from embryo_id.
    """
    try:
        mask_path = Path(row.get("exported_mask_path") or "")
        if not mask_path:
            mask_path = _derive_mask_path(masks_dir, row.get("image_id", ""), None)
        if not mask_path.exists():
            # Try a few common N values if the hint failed
            tried = []
            for n in (1, 2, 3, 4, 5, 6):
                cand = masks_dir / f"{row.get('image_id','')}_masks_emnum_{n}.png"
                tried.append(cand)
                if cand.exists():
                    mask_path = cand
                    break
            else:
                _log(verbose, f"⚠️ Mask not found for {row.get('image_id')}; tried: {', '.join(map(str,tried))}")
                return

        img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            _log(verbose, f"⚠️ Could not read mask: {mask_path}")
            return
        if img.ndim == 3:
            # Take first channel if RGB
            img = img[:, :, 0]
        embryo_num = _parse_embryo_number(row.get("embryo_id", ""))
        if embryo_num is None:
            _log(verbose, f"⚠️ Could not parse embryo number from {row.get('embryo_id')}")
            return
        binary = (img == embryo_num).astype("uint8")
        area_px = int(binary.sum())
        if area_px == 0:
            # No pixels for this embryo; skip
            return

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        largest = max(contours, key=cv2.contourArea)
        perimeter_px = float(cv2.arcLength(largest, True))
        M = cv2.moments(binary)
        cx_px = float(M["m10"]/M["m00"]) if M["m00"] else 0.0
        cy_px = float(M["m01"]/M["m00"]) if M["m00"] else 0.0

        row["area_px"] = str(area_px)
        row["perimeter_px"] = f"{perimeter_px:.2f}"
        row["centroid_x_px"] = f"{cx_px:.2f}"
        row["centroid_y_px"] = f"{cy_px:.2f}"

        # Convert to microns using SAM2 CSV pixel scale data (preferred) or Build01 scale_map (fallback)
        sx, sy = None, None
        
        # Try to get pixel scale from SAM2 row data first
        try:
            width_um = float(row.get("width_um", "") or 0)
            width_px = float(row.get("width_px", "") or 0) 
            height_um = float(row.get("height_um", "") or 0)
            height_px = float(row.get("height_px", "") or 0)
            if width_um > 0 and width_px > 0 and height_um > 0 and height_px > 0:
                sx = width_um / width_px   # um per pixel in X
                sy = height_um / height_px # um per pixel in Y
        except (ValueError, TypeError, ZeroDivisionError):
            pass
            
        # Fallback to Build01 scale map if SAM2 data not available
        if sx is None or sy is None:
            well_id = row.get("well_id", "")
            if well_id and well_id in scale_map:
                sx, sy = scale_map[well_id]
        
        # Apply scale conversion if we have valid pixel scale
        if sx and sy and sx > 0 and sy > 0:
            area_um2 = area_px * sx * sy
            per_um = perimeter_px * (sx + sy) / 2.0
            row["area_um2"] = f"{area_um2:.4f}"
            row["perimeter_um"] = f"{per_um:.4f}"
            row["centroid_x_um"] = f"{cx_px * sx:.4f}"
            row["centroid_y_um"] = f"{cy_px * sy:.4f}"
    except Exception:
        # Keep row valid even on geometry errors
        pass


def _load_scale_map(
    built01_csv: Path,
    *,
    well_col_override: Optional[str] = None,
    x_col_override: Optional[str] = None,
    y_col_override: Optional[str] = None,
    single_col_override: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, tuple[float, float]]:
    """Load well -> (um_per_px_x, um_per_px_y) if available; else empty mapping.

    Heuristics to detect column names for well and pixel size.
    """
    scale_map: Dict[str, tuple[float, float]] = {}
    if not built01_csv.exists():
        return scale_map
    try:
        with built01_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            lowers = [c.lower() for c in (r.fieldnames or [])]
            cols = {c.lower(): c for c in (r.fieldnames or [])}
            if verbose:
                print(f"      • Build01 columns ({len(cols)}): {', '.join(list(cols.values())[:20])}{' …' if len(cols)>20 else ''}")
            # Well id column: pick the first with 'well' substring if no exact match
            well_col = None
            if well_col_override and well_col_override in (cols.values()):
                well_col = well_col_override
            else:
                for k in ("well_id", "well", "wellid", "well name", "well_id_str"):
                    if k in cols:
                        well_col = cols[k]
                        break
            if not well_col:
                any_well = [c for c in lowers if "well" in c]
                well_col = cols[any_well[0]] if any_well else None
            # Pixel size candidates (broad): look for columns mentioning um and pixel
            def pixcol_pred(name: str) -> bool:
                n = name.lower()
                return ("um" in n) and ("pixel" in n or "pix" in n)
            x_cols = [c for c in lowers if ("x" in c and pixcol_pred(c))]
            y_cols = [c for c in lowers if ("y" in c and pixcol_pred(c))]
            any_cols = [c for c in lowers if pixcol_pred(c)]
            x_col = x_col_override if (x_col_override and x_col_override in cols.values()) else (cols[x_cols[0]] if x_cols else None)
            y_col = y_col_override if (y_col_override and y_col_override in cols.values()) else (cols[y_cols[0]] if y_cols else None)
            single_col = single_col_override if (single_col_override and single_col_override in cols.values()) else (cols[any_cols[0]] if any_cols else None)

            if not well_col:
                _log(verbose, "⚠️ Could not find well column in Build01 CSV")
                return scale_map

            if verbose:
                print(f"      • Using well column: {well_col if well_col else '<none>'}")
                print(f"      • Pixel size columns: x={x_col if x_col else '<none>'}, y={y_col if y_col else '<none>'}, single={single_col if single_col else '<none>'}")

            for row in r:
                well = (row.get(well_col) or "").strip()
                if not well:
                    continue
                try:
                    if x_col and y_col:
                        sx = float(row.get(x_col) or 0)
                        sy = float(row.get(y_col) or 0)
                    elif single_col:
                        s = float(row.get(single_col) or 0)
                        sx = sy = s
                    else:
                        continue
                    if sx > 0 and sy > 0:
                        scale_map[well] = (sx, sy)
                except Exception:
                    continue
    except Exception:
        pass
    _log(verbose, f"   • Loaded scale for {len(scale_map)} wells from Build01 metadata")
    return scale_map

def _autodiscover_built01_metadata(root: Path, exp: str, verbose: bool = False) -> Optional[Path]:
    """If exact Build01 CSV for exp is missing, search for a file with the same
    experiment suffix (e.g., '36hpf_ctrl_atf6') and pick the most recent by date prefix.
    """
    try:
        suffix = exp.split('_', 1)[1] if '_' in exp else exp
        meta_dir = root / "metadata" / "built_metadata_files"
        if not meta_dir.exists():
            return None
        candidates = []
        for p in meta_dir.glob("*_metadata.csv"):
            name = p.name
            if suffix in name:
                # attempt to parse leading date digits for sorting
                m = re.match(r"^(\d{8})_", name)
                date_key = m.group(1) if m else "00000000"
                candidates.append((date_key, p))
        if not candidates:
            return None
        # Pick latest by date_key
        candidates.sort(key=lambda t: t[0], reverse=True)
        choice = candidates[0][1]
        _log(verbose, f"🔎 Auto-discovered Build01 CSV: {choice}")
        return choice
    except Exception:
        return None

if __name__ == "__main__":
    raise SystemExit(main())
