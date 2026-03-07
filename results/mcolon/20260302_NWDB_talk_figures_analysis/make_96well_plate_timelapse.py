"""
Render a full 96-well plate timelapse as a single talk-ready MP4.

This composes per-well embryo "snip" JPEGs into a horizontal 8x12 plate movie.
Each well is rendered as a black circular well with the embryo snip fitted
inside (corners clipped to the well circle).

Primary inputs (read-only):
  - Embryo metadata CSV: morphseq_playground/metadata/embryo_metadata_files/{exp}_embryo_metadata.csv
  - Snip JPEGs:         morphseq_playground/training_data/bf_embryo_snips/{exp}/{embryo_id}_t{frame_index:04d}.jpg
  - YX1 plate XY coords: morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv

Defaults tuned for a first pass talk asset:
  - experiment_date=20260206 (starts ~10.75 HPF, 96 wells, good coverage)
  - output=1080p MP4, 20 fps, 300 frames
  - death_policy=freeze_at_death (freeze at last alive frame based on fraction_alive)

Usage:
  PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
  "$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/make_96well_plate_timelapse.py --experiment-date 20260206
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_WELL_RE = re.compile(r"^[A-H]\d{2}$")


@dataclass(frozen=True)
class EmbryoTrack:
    embryo_id: str
    well: str
    times_hpf: np.ndarray  # (N,) float ascending
    frame_indices: np.ndarray  # (N,) int aligned with times_hpf


@dataclass
class FrameCache:
    fi0: Optional[int] = None
    img0: Optional[np.ndarray] = None
    fi1: Optional[int] = None
    img1: Optional[np.ndarray] = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a 96-well plate timelapse mosaic MP4 from snip JPEGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment-date", default="20260206", help="Experiment date / id (snip folder name).")
    p.add_argument(
        "--embryo-metadata-dir",
        default="morphseq_playground/metadata/embryo_metadata_files",
        help="Directory containing *_embryo_metadata.csv files.",
    )
    p.add_argument(
        "--embryo-metadata-csv",
        default=None,
        help="Optional explicit embryo metadata CSV path. Overrides --embryo-metadata-dir.",
    )
    p.add_argument(
        "--snip-root",
        default="morphseq_playground/training_data/bf_embryo_snips",
        help="Root directory containing per-experiment snip JPEG folders.",
    )
    p.add_argument(
        "--yx1-coords-csv",
        default="morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv",
        help="CSV with columns well,x_um,y_um for the YX1 plate layout.",
    )
    p.add_argument("--layout", choices=["yx1_um", "grid"], default="yx1_um", help="Well layout method.")
    p.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="Rendering theme for talk slides.",
    )
    p.add_argument("--width", type=int, default=1920, help="Output video width (px).")
    p.add_argument("--height", type=int, default=1080, help="Output video height (px).")
    p.add_argument("--margin-px", type=int, default=60, help="Canvas margin around the plate (px).")
    p.add_argument("--well-radius-px", type=int, default=None, help="Override computed well radius (px).")
    p.add_argument(
        "--draw-plate-border",
        action="store_true",
        default=True,
        help="Draw a rectangular plate outline around the wells.",
    )
    p.add_argument(
        "--no-plate-border",
        dest="draw_plate_border",
        action="store_false",
        help="Disable plate border.",
    )
    p.add_argument(
        "--label-rows-cols",
        action="store_true",
        default=True,
        help="Label plate rows (A-H) and columns (1-12).",
    )
    p.add_argument(
        "--no-label-rows-cols",
        dest="label_rows_cols",
        action="store_false",
        help="Disable row/column labels.",
    )
    p.add_argument("--fps", type=int, default=20, help="Frames per second.")
    p.add_argument("--n-frames-out", type=int, default=300, help="Number of output frames.")
    p.add_argument("--t-min", type=float, default=None, help="Min HPF for playback. Default: start_age_hpf.")
    p.add_argument("--t-max", type=float, default=None, help="Max HPF for playback. Default: robust max over embryos.")
    p.add_argument(
        "--missing",
        choices=["blank", "hold_last"],
        default="hold_last",
        help="Behavior when a well runs out of frames or a JPEG is missing.",
    )
    p.add_argument(
        "--death-policy",
        choices=["ignore", "freeze_at_death", "blank_after_death"],
        default="freeze_at_death",
        help="How to handle embryo 'death' based on fraction_alive. "
        "freeze_at_death clamps to the last frame with fraction_alive > --alive-threshold.",
    )
    p.add_argument(
        "--alive-threshold",
        type=float,
        default=0.5,
        help="fraction_alive threshold considered alive (used by --death-policy).",
    )
    p.add_argument("--show-hpf", action="store_true", default=True, help="Overlay HPF label.")
    p.add_argument("--hide-hpf", dest="show_hpf", action="store_false", help="Disable HPF label overlay.")
    p.add_argument("--show-well-labels", action="store_true", default=False, help="Overlay well labels in tiles (debug).")
    p.add_argument(
        "--wells",
        default=None,
        help="Comma-separated wells to render (subset for quick iteration). Example: A01,A02,B01",
    )
    p.add_argument(
        "--out-mp4",
        default=None,
        help="Output MP4 path. Default: results/.../figures/{experiment}_plate_timelapse_1080p.mp4",
    )
    return p.parse_args()


def _ensure_even(x: int) -> int:
    x = int(x)
    return x if x % 2 == 0 else x + 1


def _all_wells_96() -> list[str]:
    return [f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)]


def _parse_wells_arg(wells: Optional[str]) -> Optional[list[str]]:
    if wells is None:
        return None
    out: list[str] = []
    for tok in str(wells).split(","):
        w = tok.strip().upper()
        if not w:
            continue
        if not _WELL_RE.match(w):
            raise ValueError(f"Invalid well in --wells: {w!r} (expected like A01)")
        out.append(w)
    if not out:
        return None
    # preserve canonical 96-well ordering
    order = {w: i for i, w in enumerate(_all_wells_96())}
    out.sort(key=lambda w: order.get(w, 10**9))
    return out


def _infer_well_from_embryo_id(embryo_id: str) -> Optional[str]:
    parts = str(embryo_id).split("_")
    for p in parts:
        p = p.strip().upper()
        if _WELL_RE.match(p):
            return p
    return None


def _snip_path(snip_root: Path, experiment_date: str, embryo_id: str, frame_index: int) -> Path:
    return snip_root / str(experiment_date) / f"{embryo_id}_t{int(frame_index):04d}.jpg"


def _load_metadata_tracks(
    embryo_metadata_csv: Path,
    *,
    wells_subset: Optional[set[str]],
    alive_threshold: float = 0.5,
) -> tuple[list[EmbryoTrack], dict[str, float], dict[str, tuple[float, int] | None], float, float]:
    """
    Returns:
      - tracks: one selected EmbryoTrack per well (best-first)
      - start_age_by_well: well -> start_age_hpf (median within selected embryo rows)
      - last_alive_by_well: well -> (last_alive_hpf, last_alive_fi) or None if unavailable
      - default_t_min: start_age_hpf (global median)
      - robust_t_max: robust max predicted_stage_hpf (0.95 quantile over per-embryo maxima)
    """
    header = pd.read_csv(embryo_metadata_csv, nrows=0)
    want = [
        "embryo_id",
        "well_id",
        "frame_index",
        "predicted_stage_hpf",
        "start_age_hpf",
        "fraction_alive",
    ]
    cols = [c for c in want if c in header.columns]
    missing_critical = [c for c in ["embryo_id", "well_id", "frame_index", "predicted_stage_hpf"] if c not in cols]
    if missing_critical:
        raise ValueError(f"Missing required columns in {embryo_metadata_csv.name}: {missing_critical}")

    df = pd.read_csv(embryo_metadata_csv, usecols=cols, low_memory=False)
    df["well_id"] = df["well_id"].astype(str).str.strip().str.upper()
    if wells_subset is not None:
        df = df[df["well_id"].isin(wells_subset)].copy()
    if df.empty:
        raise ValueError("No rows remaining after applying --wells filter.")

    df["embryo_id"] = df["embryo_id"].astype(str)
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df = df[df["frame_index"].notna() & df["predicted_stage_hpf"].notna()].copy()
    if df.empty:
        raise ValueError("No valid (frame_index, predicted_stage_hpf) rows in metadata.")

    if "start_age_hpf" in df.columns:
        df["start_age_hpf"] = pd.to_numeric(df["start_age_hpf"], errors="coerce")
    else:
        df["start_age_hpf"] = np.nan

    if "fraction_alive" in df.columns:
        df["fraction_alive"] = pd.to_numeric(df["fraction_alive"], errors="coerce")
    else:
        df["fraction_alive"] = np.nan

    # Choose a single embryo per well:
    # - prefer e01 if present
    # - then maximize survival proxy and coverage
    df["well"] = df["well_id"]
    df["is_e01"] = df["embryo_id"].astype(str).str.endswith("_e01").astype(int)

    per_emb = (
        df.groupby(["well", "embryo_id"], observed=True)
        .agg(
            n_rows=("frame_index", "size"),
            max_hpf=("predicted_stage_hpf", "max"),
            min_hpf=("predicted_stage_hpf", "min"),
            alive_mean=("fraction_alive", "mean"),
            start_age=("start_age_hpf", "median"),
            max_fi=("frame_index", lambda s: int(pd.to_numeric(s, errors="coerce").max())),
        )
        .reset_index()
    )
    per_emb["alive_mean"] = per_emb["alive_mean"].fillna(1.0)
    per_emb["start_age"] = per_emb["start_age"].fillna(per_emb["min_hpf"])
    per_emb = per_emb.merge(df.groupby(["well", "embryo_id"], observed=True)["is_e01"].max().reset_index(), on=["well", "embryo_id"], how="left")
    per_emb["is_e01"] = per_emb["is_e01"].fillna(0).astype(int)

    per_emb = per_emb.sort_values(
        ["well", "is_e01", "alive_mean", "max_hpf", "n_rows", "max_fi"],
        ascending=[True, False, False, False, False, False],
    )
    best = per_emb.drop_duplicates(subset=["well"], keep="first").copy()
    best_by_well = dict(zip(best["well"].tolist(), best["embryo_id"].tolist(), strict=False))

    # Robust t_max: 0.95 quantile of per-embryo max_hpf for selected embryos.
    robust_t_max = float(best["max_hpf"].quantile(0.95))
    if not math.isfinite(robust_t_max):
        robust_t_max = float(best["max_hpf"].max())

    # Default t_min: global median start_age among selected embryos (fallback to min_hpf).
    default_t_min = float(best["start_age"].median())
    if not math.isfinite(default_t_min):
        default_t_min = float(best["min_hpf"].min())

    # Build tracks for selected embryos.
    tracks: list[EmbryoTrack] = []
    start_age_by_well: dict[str, float] = {}
    last_alive_by_well: dict[str, tuple[float, int] | None] = {}
    for well, embryo_id in best_by_well.items():
        g = df[(df["well"] == well) & (df["embryo_id"] == embryo_id)].copy()
        g = g.sort_values(["predicted_stage_hpf", "frame_index"])
        times = g["predicted_stage_hpf"].to_numpy(dtype=float)
        fis = g["frame_index"].to_numpy(dtype=int)
        # Deduplicate identical times to avoid zero-denominator crossfades.
        # Keep the earliest frame_index per time.
        if times.size > 1:
            keep = np.concatenate([[True], times[1:] != times[:-1]])
            times = times[keep]
            fis = fis[keep]
        if times.size == 0:
            continue
        tracks.append(EmbryoTrack(embryo_id=str(embryo_id), well=str(well), times_hpf=times, frame_indices=fis))
        # per-well start_age estimate
        sa = float(best.loc[best["well"] == well, "start_age"].iloc[0])
        if math.isfinite(sa):
            start_age_by_well[str(well)] = sa

        # last alive frame (for freeze/blank at death)
        last_alive_by_well[str(well)] = None
        if "fraction_alive" in g.columns and g["fraction_alive"].notna().any():
            alive = g[pd.to_numeric(g["fraction_alive"], errors="coerce") > float(alive_threshold)].copy()
            if not alive.empty:
                # Pick the alive row with maximum predicted_stage_hpf; tie-break by max frame_index.
                alive = alive.sort_values(["predicted_stage_hpf", "frame_index"])
                last = alive.iloc[-1]
                try:
                    lh = float(last["predicted_stage_hpf"])
                    lfi = int(last["frame_index"])
                    if math.isfinite(lh):
                        last_alive_by_well[str(well)] = (lh, lfi)
                except Exception:
                    last_alive_by_well[str(well)] = None

    return tracks, start_age_by_well, last_alive_by_well, default_t_min, robust_t_max


def _load_yx1_coords(yx1_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(yx1_csv)
    if not {"well", "x_um", "y_um"}.issubset(set(df.columns)):
        raise ValueError(f"{yx1_csv} must have columns: well,x_um,y_um")
    df["well"] = df["well"].astype(str).str.strip().str.upper()
    df["x_um"] = pd.to_numeric(df["x_um"], errors="coerce")
    df["y_um"] = pd.to_numeric(df["y_um"], errors="coerce")
    df = df[df["well"].notna() & df["x_um"].notna() & df["y_um"].notna()].copy()
    return df


def _compute_centers_px(
    *,
    layout: str,
    width: int,
    height: int,
    margin_px: int,
    yx1_coords: Optional[pd.DataFrame],
) -> dict[str, tuple[int, int]]:
    wells = _all_wells_96()
    centers: dict[str, tuple[int, int]] = {}
    w = int(width)
    h = int(height)
    m = int(margin_px)
    if layout == "grid":
        pitch_x = (w - 2 * m) / (12 - 1)
        pitch_y = (h - 2 * m) / (8 - 1)
        for ri, row in enumerate("ABCDEFGH"):
            for ci in range(1, 13):
                well = f"{row}{ci:02d}"
                x = int(round(m + (ci - 1) * pitch_x))
                y = int(round(m + ri * pitch_y))
                centers[well] = (x, y)
        return centers

    if layout != "yx1_um":
        raise ValueError(f"Unknown layout: {layout}")
    if yx1_coords is None:
        raise ValueError("yx1_coords required for layout=yx1_um")

    coords = yx1_coords.set_index("well", drop=True)
    missing = [well for well in wells if well not in coords.index]
    if missing:
        raise ValueError(f"YX1 coords missing wells: {missing[:10]} (and {max(0, len(missing) - 10)} more)")

    # Flip x so columns increase left->right (A01 left, A12 right).
    x_um = (-coords.loc[wells, "x_um"].to_numpy(dtype=float)).astype(float)
    y_um = (coords.loc[wells, "y_um"].to_numpy(dtype=float)).astype(float)

    x0, x1 = float(np.min(x_um)), float(np.max(x_um))
    y0, y1 = float(np.min(y_um)), float(np.max(y_um))
    dx = max(1e-9, x1 - x0)
    dy = max(1e-9, y1 - y0)
    scale = min((w - 2 * m) / dx, (h - 2 * m) / dy)

    for well, xu, yu in zip(wells, x_um.tolist(), y_um.tolist(), strict=False):
        x = int(round(m + (float(xu) - x0) * scale))
        y = int(round(m + (float(yu) - y0) * scale))
        centers[well] = (x, y)
    return centers


def _estimate_radius_px(centers: dict[str, tuple[int, int]], *, override: Optional[int]) -> int:
    if override is not None:
        return max(6, int(override))
    # Estimate pitch from adjacent wells
    dxs = []
    dys = []
    for row in "ABCDEFGH":
        row_wells = [f"{row}{c:02d}" for c in range(1, 13)]
        for a, b in zip(row_wells[:-1], row_wells[1:], strict=False):
            xa, ya = centers[a]
            xb, yb = centers[b]
            dxs.append(abs(xb - xa))
    for col in range(1, 13):
        col_wells = [f"{r}{col:02d}" for r in "ABCDEFGH"]
        for a, b in zip(col_wells[:-1], col_wells[1:], strict=False):
            xa, ya = centers[a]
            xb, yb = centers[b]
            dys.append(abs(yb - ya))
    pitch = min(float(np.median(dxs)) if dxs else 0.0, float(np.median(dys)) if dys else 0.0)
    if not math.isfinite(pitch) or pitch <= 0:
        pitch = 120.0
    return max(10, int(round(0.42 * pitch)))


def _schedule_for_track(
    track: EmbryoTrack,
    t_out: np.ndarray,
    *,
    clamp_time: Optional[float] = None,
    clamp_fi: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each output time, return (fi0, fi1, alpha) arrays.
    """
    times = track.times_hpf
    fis = track.frame_indices
    n = int(t_out.size)
    fi0 = np.zeros(n, dtype=np.int32)
    fi1 = np.zeros(n, dtype=np.int32)
    alpha = np.zeros(n, dtype=np.float32)

    for k, t in enumerate(t_out.tolist()):
        if clamp_time is not None and clamp_fi is not None:
            if t >= float(clamp_time):
                fi0[k] = int(clamp_fi)
                fi1[k] = int(clamp_fi)
                alpha[k] = 0.0
                continue
        if t <= float(times[0]):
            fi0[k] = int(fis[0])
            fi1[k] = int(fis[0])
            alpha[k] = 0.0
            continue
        if t >= float(times[-1]):
            fi0[k] = int(fis[-1])
            fi1[k] = int(fis[-1])
            alpha[k] = 0.0
            continue
        j = int(np.searchsorted(times, t, side="left"))
        i0 = max(0, j - 1)
        i1 = min(int(times.size - 1), j)
        t0 = float(times[i0])
        t1 = float(times[i1])
        fi0[k] = int(fis[i0])
        fi1[k] = int(fis[i1])
        if i0 == i1 or t1 <= t0 + 1e-9:
            alpha[k] = 0.0
        else:
            a = float((t - t0) / (t1 - t0))
            alpha[k] = float(min(1.0, max(0.0, a)))
    return fi0, fi1, alpha


def _put_text_with_outline(img: np.ndarray, text: str, org: tuple[int, int], *, font_scale: float = 1.0) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(2.0 * font_scale)))
    outline = thickness + 3
    cv2.putText(img, text, org, font, float(font_scale), (0, 0, 0), int(outline), cv2.LINE_AA)
    cv2.putText(img, text, org, font, float(font_scale), (255, 255, 255), int(thickness), cv2.LINE_AA)


def _put_text_with_outline_colors(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    fg_bgr: tuple[int, int, int],
    outline_bgr: tuple[int, int, int],
) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(2.0 * font_scale)))
    outline = thickness + 3
    cv2.putText(img, text, org, font, float(font_scale), outline_bgr, int(outline), cv2.LINE_AA)
    cv2.putText(img, text, org, font, float(font_scale), fg_bgr, int(thickness), cv2.LINE_AA)


def _fit_into_circle_tile(
    snip_bgr: Optional[np.ndarray],
    *,
    radius: int,
    show_label: bool,
    label: str,
    well_fill_bgr: tuple[int, int, int],
    well_rim_bgr: tuple[int, int, int],
    label_fg_bgr: tuple[int, int, int],
    label_outline_bgr: tuple[int, int, int],
) -> np.ndarray:
    import cv2

    d = int(2 * radius)
    tile = np.zeros((d, d, 3), dtype=np.uint8)
    tile[:, :] = np.array(well_fill_bgr, dtype=np.uint8)
    if snip_bgr is not None:
        h, w = snip_bgr.shape[:2]
        if h > 0 and w > 0:
            scale = min(d / float(w), d / float(h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(snip_bgr, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
            x0 = int((d - new_w) // 2)
            y0 = int((d - new_h) // 2)
            tile[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    # Clip corners to circle.
    mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), int(radius - 1), 255, thickness=-1, lineType=cv2.LINE_AA)
    tile[mask == 0] = 0

    # Well rim
    cv2.circle(tile, (radius, radius), int(radius - 1), tuple(map(int, well_rim_bgr)), thickness=2, lineType=cv2.LINE_AA)

    if show_label and label:
        _put_text_with_outline_colors(
            tile,
            label,
            (8, d - 10),
            font_scale=max(0.35, radius / 80.0),
            fg_bgr=label_fg_bgr,
            outline_bgr=label_outline_bgr,
        )

    return tile


def _plate_bbox(centers: dict[str, tuple[int, int]], wells: list[str], radius: int, pad: int) -> tuple[int, int, int, int]:
    xs = [centers[w][0] for w in wells]
    ys = [centers[w][1] for w in wells]
    x0 = int(min(xs) - radius - pad)
    x1 = int(max(xs) + radius + pad)
    y0 = int(min(ys) - radius - pad)
    y1 = int(max(ys) + radius + pad)
    return x0, y0, x1, y1


def _read_snp(snip_path: Path) -> Optional[np.ndarray]:
    import cv2

    img = cv2.imread(str(snip_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def main() -> None:
    args = _parse_args()

    import cv2

    width = _ensure_even(int(args.width))
    height = _ensure_even(int(args.height))
    fps = int(args.fps)
    n_frames_out = int(args.n_frames_out)
    margin_px = int(args.margin_px)

    wells_list = _parse_wells_arg(args.wells)
    wells_subset = set(wells_list) if wells_list is not None else None

    snip_root = Path(args.snip_root)
    exp = str(args.experiment_date)

    theme = str(args.theme)
    if theme == "dark":
        canvas_bg_bgr = (0, 0, 0)
        plate_border_bgr = (255, 255, 255)
        label_fg_bgr = (255, 255, 255)
        label_outline_bgr = (0, 0, 0)
        well_fill_bgr = (0, 0, 0)
        well_rim_bgr = (125, 125, 125)
    else:
        canvas_bg_bgr = (255, 255, 255)
        plate_border_bgr = (0, 0, 0)
        label_fg_bgr = (0, 0, 0)
        label_outline_bgr = (255, 255, 255)
        well_fill_bgr = (255, 255, 255)
        well_rim_bgr = (170, 170, 170)

    if args.embryo_metadata_csv:
        embryo_csv = Path(args.embryo_metadata_csv)
    else:
        embryo_csv = Path(args.embryo_metadata_dir) / f"{exp}_embryo_metadata.csv"
    if not embryo_csv.exists():
        raise FileNotFoundError(f"Missing embryo metadata CSV: {embryo_csv}")

    tracks, start_age_by_well, last_alive_by_well, default_t_min, robust_t_max = _load_metadata_tracks(
        embryo_csv,
        wells_subset=wells_subset,
        alive_threshold=float(args.alive_threshold),
    )
    if not tracks:
        raise RuntimeError("No embryo tracks selected.")

    t_min = float(args.t_min) if args.t_min is not None else float(default_t_min)
    t_max = float(args.t_max) if args.t_max is not None else float(robust_t_max)
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError(f"Invalid (t_min,t_max)=({t_min},{t_max})")

    t_out = np.linspace(t_min, t_max, num=int(n_frames_out), dtype=float)

    # Layout
    yx1_coords = None
    if args.layout == "yx1_um":
        yx1_coords = _load_yx1_coords(Path(args.yx1_coords_csv))
    centers = _compute_centers_px(
        layout=str(args.layout),
        width=int(width),
        height=int(height),
        margin_px=int(margin_px),
        yx1_coords=yx1_coords,
    )
    radius = _estimate_radius_px(centers, override=args.well_radius_px)

    # Track lookup per well
    track_by_well: dict[str, EmbryoTrack] = {t.well: t for t in tracks}
    cache_by_well: dict[str, FrameCache] = {w: FrameCache() for w in track_by_well.keys()}

    # Precompute schedules for selected wells
    fi0_by_well: dict[str, np.ndarray] = {}
    fi1_by_well: dict[str, np.ndarray] = {}
    alpha_by_well: dict[str, np.ndarray] = {}
    for w, tr in track_by_well.items():
        clamp_time = None
        clamp_fi = None
        if str(args.death_policy) in {"freeze_at_death", "blank_after_death"}:
            last = last_alive_by_well.get(w)
            if last is not None:
                clamp_time, clamp_fi = float(last[0]), int(last[1])
        fi0, fi1, alpha = _schedule_for_track(tr, t_out, clamp_time=clamp_time, clamp_fi=clamp_fi)
        fi0_by_well[w] = fi0
        fi1_by_well[w] = fi1
        alpha_by_well[w] = alpha

    # Output path
    if args.out_mp4:
        out_mp4 = Path(args.out_mp4)
    else:
        out_mp4 = Path(__file__).resolve().parent / "figures" / f"{exp}_plate_timelapse_1080p.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

    wells_to_draw = wells_list if wells_list is not None else _all_wells_96()
    bbox_pad = int(round(0.75 * radius))
    plate_x0, plate_y0, plate_x1, plate_y1 = _plate_bbox(centers, wells_to_draw, radius=radius, pad=bbox_pad)
    # Row/col label anchor positions
    row_y = {row: centers[f"{row}01"][1] for row in "ABCDEFGH"}
    col_x = {col: centers[f"A{col:02d}"][0] for col in range(1, 13)}
    row_label_x = int(plate_x0 + int(round(0.35 * radius)))
    col_label_y = int(plate_y0 + int(round(0.65 * radius)))
    row_label_margin = int(round(0.55 * radius))
    col_label_margin = int(round(0.35 * radius))
    axis_font_scale = max(0.6, radius / 65.0)
    border_thickness = max(2, int(round(radius / 22.0)))

    try:
        for k, t in enumerate(t_out.tolist()):
            canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
            canvas[:, :] = np.array(canvas_bg_bgr, dtype=np.uint8)

            if args.draw_plate_border:
                x0 = max(0, plate_x0)
                y0 = max(0, plate_y0)
                x1 = min(int(width) - 1, plate_x1)
                y1 = min(int(height) - 1, plate_y1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), plate_border_bgr, thickness=int(border_thickness), lineType=cv2.LINE_AA)

            if args.label_rows_cols:
                # Row labels (A-H) along left
                for row in "ABCDEFGH":
                    y = int(row_y[row])
                    _put_text_with_outline_colors(
                        canvas,
                        row,
                        (max(8, plate_x0 - row_label_margin), y + int(round(0.35 * radius))),
                        font_scale=axis_font_scale,
                        fg_bgr=label_fg_bgr,
                        outline_bgr=label_outline_bgr,
                    )
                # Column labels (1-12) along top
                for col in range(1, 13):
                    x = int(col_x[col])
                    _put_text_with_outline_colors(
                        canvas,
                        str(col),
                        (x - int(round(0.22 * radius)), max(20, plate_y0 - col_label_margin)),
                        font_scale=axis_font_scale,
                        fg_bgr=label_fg_bgr,
                        outline_bgr=label_outline_bgr,
                    )

            for well in wells_to_draw:
                cx, cy = centers[well]
                tr = track_by_well.get(well)
                if tr is None:
                    tile = _fit_into_circle_tile(
                        None,
                        radius=radius,
                        show_label=bool(args.show_well_labels),
                        label=str(well),
                        well_fill_bgr=well_fill_bgr,
                        well_rim_bgr=well_rim_bgr,
                        label_fg_bgr=label_fg_bgr,
                        label_outline_bgr=label_outline_bgr,
                    )
                else:
                    fi0 = int(fi0_by_well[well][k])
                    fi1 = int(fi1_by_well[well][k])
                    a = float(alpha_by_well[well][k])

                    fc = cache_by_well[well]

                    def load_cached(fi: int) -> Optional[np.ndarray]:
                        if fc.fi0 == fi and fc.img0 is not None:
                            return fc.img0
                        if fc.fi1 == fi and fc.img1 is not None:
                            return fc.img1
                        p = _snip_path(snip_root, exp, tr.embryo_id, int(fi))
                        img = _read_snp(p)
                        # Keep a tiny 2-slot cache, updating the older slot.
                        if fc.fi0 is None or fc.fi0 == fi or (fc.fi1 is not None and fc.fi0 == fc.fi1):
                            fc.fi0, fc.img0 = int(fi), img
                        elif fc.fi1 is None or fc.fi1 == fi:
                            fc.fi1, fc.img1 = int(fi), img
                        else:
                            # Replace the "farther" index heuristically
                            if abs(int(fi) - int(fc.fi0)) >= abs(int(fi) - int(fc.fi1)):
                                fc.fi0, fc.img0 = int(fi), img
                            else:
                                fc.fi1, fc.img1 = int(fi), img
                        return img

                    img0 = load_cached(fi0)
                    img1 = load_cached(fi1) if fi1 != fi0 else img0

                    snip = None
                    if img0 is None and img1 is None:
                        snip = None
                    elif img0 is None:
                        snip = img1
                    elif img1 is None:
                        snip = img0
                    else:
                        if fi0 == fi1 or a <= 0.0:
                            snip = img0
                        elif a >= 1.0:
                            snip = img1
                        else:
                            snip = cv2.addWeighted(img0, float(1.0 - a), img1, float(a), 0.0)

                    # Missing behavior: if blank, drop snip when we're past the embryo's coverage.
                    if args.missing == "blank" and tr.times_hpf.size > 0:
                        if t > float(tr.times_hpf[-1]) + 1e-6:
                            snip = None

                    # Death behavior: if blank_after_death, blank when we're past the last alive time.
                    if str(args.death_policy) == "blank_after_death":
                        last = last_alive_by_well.get(well)
                        if last is None:
                            snip = None
                        else:
                            if t > float(last[0]) + 1e-6:
                                snip = None

                    tile = _fit_into_circle_tile(
                        snip,
                        radius=radius,
                        show_label=bool(args.show_well_labels),
                        label=str(well),
                        well_fill_bgr=well_fill_bgr,
                        well_rim_bgr=well_rim_bgr,
                        label_fg_bgr=label_fg_bgr,
                        label_outline_bgr=label_outline_bgr,
                    )

                d = tile.shape[0]
                x0 = int(cx - d // 2)
                y0 = int(cy - d // 2)
                x1 = x0 + d
                y1 = y0 + d

                # Clip to canvas
                cx0 = max(0, x0)
                cy0 = max(0, y0)
                cx1 = min(int(width), x1)
                cy1 = min(int(height), y1)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                tx0 = cx0 - x0
                ty0 = cy0 - y0
                tx1 = tx0 + (cx1 - cx0)
                ty1 = ty0 + (cy1 - cy0)
                canvas[cy0:cy1, cx0:cx1] = tile[ty0:ty1, tx0:tx1]

            if args.show_hpf:
                label = f"{exp}   {t:.1f} HPF"
                _put_text_with_outline_colors(
                    canvas,
                    label,
                    (40, 60),
                    font_scale=1.2,
                    fg_bgr=label_fg_bgr,
                    outline_bgr=label_outline_bgr,
                )

            writer.write(canvas)

            if (k + 1) % 25 == 0 or (k + 1) == n_frames_out:
                print(f"[{k+1:>4d}/{n_frames_out}] t={t:.2f} HPF  radius={radius}px  out={out_mp4.name}")
    finally:
        writer.release()

    print(f"Wrote: {out_mp4}")


if __name__ == "__main__":
    main()
