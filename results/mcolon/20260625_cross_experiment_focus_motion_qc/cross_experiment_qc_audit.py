#!/usr/bin/env python
"""
Cross-experiment focus/motion QC audit.

This is audit-first: preflight and existing metrics are preferred, and expensive
recomputation only happens with --recompute-missing or --recompute-all.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

HERE = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS = ("20250305", "20251125", "20260206")

ENTROPY_CUT = -0.55
SHARPNESS_RATIO_CUT = 1.5
N_DECILES = 10

FOCUS_METRIC = "ff_rel_entropy"
MOTION_METRIC = "ncc_p05"

FOCUS_COLS = [
    "experiment_id",
    "well_id",
    "well",
    "time_index",
    "frame_index",
    "snip_id",
    "image_id",
    "image_path",
    "mask_path",
    "predicted_stage_hpf",
    "dead_flag",
    "dead_flag2",
    "use_embryo_flag",
    "not_dead_snip",
    "ff_entropy_emb",
    "ff_entropy_bg",
    "ff_rel_entropy",
    "ff_lap_abs_mean_emb",
    "ff_lap_abs_mean_bg",
    "ff_rel_lap_abs_mean",
    "ff_lap_abs_ratio",
    "ff_mean_emb",
    "ff_mean_bg",
    "ff_rel_mean",
    "ff_iqr_emb",
    "ff_iqr_bg",
    "ff_rel_iqr",
    "focus_tile_status",
    "focus_pass_fail_status",
]

MOTION_COLS = [
    "experiment_id",
    "well_id",
    "well",
    "time_index",
    "frame_index",
    "snip_id",
    "image_id",
    "nd2_path",
    "p",
    "predicted_stage_hpf",
    "ncc_p05",
    "ncc_min",
    "bad_pair_frac",
    "ncc_bad_tile_frac",
    "longest_bad_run",
    "ncc_mean",
    "local_ncc_std_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--metric-type", choices=("focus", "motion", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recompute-missing", action="store_true")
    parser.add_argument("--recompute-all", action="store_true")
    parser.add_argument("--smoke-deciles", type=int, default=None)
    parser.add_argument("--examples-per-experiment-decile", type=int, default=6)
    parser.add_argument("--focus-limit-per-experiment", type=int, default=None)
    parser.add_argument("--motion-workers", type=int, default=1)
    parser.add_argument("--motion-limit-per-experiment", type=int, default=None)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def experiments_from_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def tables_dir(out_dir: Path) -> Path:
    path = out_dir / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(out_dir: Path) -> Path:
    path = out_dir / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_path(experiment: str) -> Path:
    return MORPHSEQ_ROOT / "morphseq_playground/metadata/build04_output" / f"qc_staged_{experiment}.csv"


def nd2_path_for_experiment(experiment: str) -> Path | None:
    nd2_dir = MORPHSEQ_ROOT / "morphseq_playground/raw_image_data/YX1" / experiment
    paths = sorted(nd2_dir.glob("*.nd2"))
    return paths[0] if len(paths) == 1 else None


def mask_path_for_row(row: pd.Series, experiment: str) -> Path:
    exported = str(row.get("exported_mask_path", "")).strip()
    if exported and exported != "nan":
        p = Path(exported)
        if p.is_absolute() and p.exists():
            return p
        candidate = (
            MORPHSEQ_ROOT
            / "morphseq_playground/sam2_pipeline_files/exported_masks"
            / experiment
            / "masks"
            / p.name
        )
        return candidate

    image_id = str(row.get("image_id", "")).strip()
    if image_id:
        return (
            MORPHSEQ_ROOT
            / "morphseq_playground/sam2_pipeline_files/exported_masks"
            / experiment
            / "masks"
            / f"{image_id}_masks_emnum_1.png"
        )
    return Path("")


def image_path_for_row(row: pd.Series, experiment: str) -> tuple[Path, str]:
    image_id = str(row.get("image_id", "")).strip()
    video_id = str(row.get("video_id", "")).strip()
    raw = str(row.get("image_path", "")).strip()
    notes = []

    reconstructed = None
    if image_id and video_id:
        reconstructed = (
            MORPHSEQ_ROOT
            / "morphseq_playground/sam2_pipeline_files/raw_data_organized"
            / experiment
            / "images"
            / video_id
            / f"{image_id}.jpg"
        )

    if "morphseq_CORRUPT_OLD" in raw:
        notes.append("metadata_image_path_points_to_morphseq_CORRUPT_OLD")
        if reconstructed is not None and reconstructed.exists():
            return reconstructed, ";".join(notes + ["used_reconstructed_path"])

    if raw and raw != "nan":
        path = Path(raw)
        if path.exists():
            return path, ";".join(notes)

    if reconstructed is not None:
        return reconstructed, ";".join(notes + ["used_reconstructed_path"])

    return Path(raw), ";".join(notes + ["image_path_unresolved"])


def load_metadata(experiment: str) -> pd.DataFrame:
    path = metadata_path(experiment)
    df = pd.read_csv(path)
    if "well_id" not in df and "well" in df:
        df["well_id"] = df["well"]
    if "time_index" not in df:
        if "time_int" in df:
            df["time_index"] = df["time_int"]
        elif "frame_index" in df:
            df["time_index"] = df["frame_index"]
    if "dead_flag" not in df:
        df["dead_flag"] = False
    if "dead_flag2" not in df:
        df["dead_flag2"] = False
    df["not_dead_snip"] = ~(as_bool(df["dead_flag"]) | as_bool(df["dead_flag2"]))
    df["experiment_id"] = experiment
    return df


def existing_focus_metrics(out_dir: Path) -> pd.DataFrame:
    path = tables_dir(out_dir) / "focus_metrics_by_experiment.csv"
    if not path.exists():
        return pd.DataFrame(columns=FOCUS_COLS)
    return pd.read_csv(path)


def existing_motion_metrics(out_dir: Path) -> pd.DataFrame:
    path = tables_dir(out_dir) / "motion_metrics_by_experiment.csv"
    if not path.exists():
        return pd.DataFrame(columns=MOTION_COLS)
    return pd.read_csv(path)


def entropy_u8(values: np.ndarray) -> float:
    hist, _ = np.histogram(values, bins=256, range=(0, 255))
    p = hist[hist > 0].astype(np.float64)
    if p.size == 0:
        return float("nan")
    p /= p.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def focus_status(row: pd.Series) -> tuple[str, str]:
    entropy_reject = bool(row.get("ff_rel_entropy", np.nan) < ENTROPY_CUT)
    sharp_reject = bool(row.get("ff_lap_abs_ratio", np.nan) < SHARPNESS_RATIO_CUT)
    if entropy_reject and sharp_reject:
        return "rejected_by_both", "reject"
    if entropy_reject:
        return "entropy_only_reject", "reject"
    if sharp_reject:
        return "sharpness_only_reject", "reject"
    return "accepted", "pass"


def compute_focus_metric(image_path: Path, mask_path: Path) -> dict[str, float]:
    import scipy.ndimage as ndi

    image = np.array(Image.open(image_path).convert("L"))
    mask = np.array(Image.open(mask_path)).astype(bool)
    if image.shape != mask.shape:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.array(mask_img.resize((image.shape[1], image.shape[0]), Image.NEAREST)).astype(bool)
    if not mask.any():
        raise ValueError("empty mask")

    bg = ndi.binary_erosion(~mask, iterations=10)
    if not bg.any():
        bg = ~mask

    emb_px = image[mask]
    bg_px = image[bg]
    lap_abs = np.abs(ndi.laplace(image.astype(np.float32)))
    emb_lap = lap_abs[mask]
    bg_lap = lap_abs[bg]

    emb_lap_mean = float(np.mean(emb_lap))
    bg_lap_mean = float(np.mean(bg_lap))
    emb_iqr = float(np.percentile(emb_px, 95) - np.percentile(emb_px, 5))
    bg_iqr = float(np.percentile(bg_px, 95) - np.percentile(bg_px, 5))

    return {
        "ff_entropy_emb": entropy_u8(emb_px),
        "ff_entropy_bg": entropy_u8(bg_px),
        "ff_rel_entropy": entropy_u8(emb_px) - entropy_u8(bg_px),
        "ff_lap_abs_mean_emb": emb_lap_mean,
        "ff_lap_abs_mean_bg": bg_lap_mean,
        "ff_rel_lap_abs_mean": emb_lap_mean - bg_lap_mean,
        "ff_lap_abs_ratio": emb_lap_mean / (bg_lap_mean + 1e-9),
        "ff_mean_emb": float(np.mean(emb_px)),
        "ff_mean_bg": float(np.mean(bg_px)),
        "ff_rel_mean": float(np.mean(emb_px) - np.mean(bg_px)),
        "ff_iqr_emb": emb_iqr,
        "ff_iqr_bg": bg_iqr,
        "ff_rel_iqr": emb_iqr - bg_iqr,
    }


def compute_focus_metrics_for_experiment(
    experiment: str,
    metadata: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    rows = []
    all_snips = metadata.copy()
    if limit is not None:
        all_snips = all_snips.head(limit)
    total = len(all_snips)
    for i, (_, row) in enumerate(all_snips.iterrows(), start=1):
        image_path, _ = image_path_for_row(row, experiment)
        mask_path = mask_path_for_row(row, experiment)
        base = {
            "experiment_id": experiment,
            "well_id": row.get("well_id", row.get("well", "")),
            "well": row.get("well", row.get("well_id", "")),
            "time_index": row.get("time_index", row.get("time_int", row.get("frame_index", np.nan))),
            "frame_index": row.get("frame_index", np.nan),
            "snip_id": row.get("snip_id", ""),
            "image_id": row.get("image_id", ""),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "predicted_stage_hpf": row.get("predicted_stage_hpf", np.nan),
            "dead_flag": row.get("dead_flag", False),
            "dead_flag2": row.get("dead_flag2", False),
            "use_embryo_flag": row.get("use_embryo_flag", np.nan),
            "not_dead_snip": bool(row.get("not_dead_snip", False)),
        }
        try:
            metrics = compute_focus_metric(image_path, mask_path)
            status, pass_fail = focus_status(metrics)
            rows.append({**base, **metrics, "focus_tile_status": status, "focus_pass_fail_status": pass_fail})
        except Exception as exc:
            rows.append({**base, "focus_tile_status": "metric_error", "focus_pass_fail_status": "metric_error", "metric_error": str(exc)})
        if i % 500 == 0 or i == total:
            print(f"[focus {experiment}] {i}/{total} all snips processed", flush=True)
    return pd.DataFrame(rows)


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for val in values.astype(bool):
        if val:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def compute_motion_metric_for_stack(nd2_path: Path, t: int, p: int) -> dict[str, float]:
    import nd2
    from src.data_pipeline.quality_control.zstack_motion_qc import compute_local_ncc_grid, ncc_stack_summary

    with nd2.ND2File(str(nd2_path)) as nd:
        stack = nd.to_dask()[t, p].compute().astype(np.float32)

    ncc_grid = compute_local_ncc_grid(stack, tile_size=128)
    summary = ncc_stack_summary(ncc_grid, bad_thresh=0.90)
    flat = ncc_grid.ravel()
    flat = flat[~np.isnan(flat)]
    pair_means = np.nanmean(ncc_grid.reshape(ncc_grid.shape[0], -1), axis=1)
    summary["ncc_p05"] = float(np.percentile(flat, 5)) if flat.size else float("nan")
    summary["ncc_bad_tile_frac"] = float(np.mean(flat < 0.90)) if flat.size else float("nan")
    summary["longest_bad_run"] = longest_true_run(pair_means < 0.90) if pair_means.size else 0
    return summary


def compute_motion_metrics_for_experiment(
    experiment: str,
    metadata: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    nd2_path = nd2_path_for_experiment(experiment)
    if nd2_path is None:
        print(f"[motion {experiment}] no unique ND2 path; skipping", flush=True)
        return pd.DataFrame(columns=MOTION_COLS)

    all_snips = metadata.copy()
    if "nd2_series_num" not in all_snips.columns:
        print(f"[motion {experiment}] metadata lacks nd2_series_num; skipping", flush=True)
        return pd.DataFrame(columns=MOTION_COLS)

    all_snips["p"] = pd.to_numeric(all_snips["nd2_series_num"], errors="coerce").astype("Int64") - 1
    all_snips["time_index"] = pd.to_numeric(all_snips["time_index"], errors="coerce").astype("Int64")
    unique = all_snips.dropna(subset=["p", "time_index"]).drop_duplicates(["time_index", "p"])
    unique = unique.sort_values(["time_index", "p"]).reset_index(drop=True)
    if limit is not None:
        unique = unique.head(limit)

    rows = []
    total = len(unique)
    for i, row in unique.iterrows():
        t = int(row["time_index"])
        p = int(row["p"])
        base = {
            "experiment_id": experiment,
            "well_id": row.get("well_id", row.get("well", "")),
            "well": row.get("well", row.get("well_id", "")),
            "time_index": t,
            "frame_index": row.get("frame_index", np.nan),
            "snip_id": row.get("snip_id", ""),
            "image_id": row.get("image_id", ""),
            "image_path": str(image_path_for_row(row, experiment)[0]),
            "mask_path": str(mask_path_for_row(row, experiment)),
            "nd2_path": str(nd2_path),
            "p": p,
            "predicted_stage_hpf": row.get("predicted_stage_hpf", np.nan),
            "not_dead_snip": bool(row.get("not_dead_snip", False)),
        }
        try:
            rows.append({**base, **compute_motion_metric_for_stack(nd2_path, t, p)})
        except Exception as exc:
            rows.append({**base, "motion_metric_error": str(exc)})
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"[motion {experiment}] {i + 1}/{total} stacks processed", flush=True)
    return pd.DataFrame(rows)


def preflight_experiment(experiment: str, metadata: pd.DataFrame, focus_df: pd.DataFrame, motion_df: pd.DataFrame) -> dict:
    sample = metadata.head(250).copy()
    image_missing = 0
    mask_missing = 0
    override_notes = []
    for _, row in sample.iterrows():
        image_path, note = image_path_for_row(row, experiment)
        mask_path = mask_path_for_row(row, experiment)
        image_missing += int(not image_path.exists())
        mask_missing += int(not mask_path.exists())
        if note:
            override_notes.append(note)

    nd2_path = nd2_path_for_experiment(experiment)
    focus_existing = focus_df[focus_df.get("experiment_id", pd.Series(dtype=str)).astype(str) == experiment] if not focus_df.empty else pd.DataFrame()
    motion_existing = motion_df[motion_df.get("experiment_id", pd.Series(dtype=str)).astype(str) == experiment] if not motion_df.empty else pd.DataFrame()
    not_dead_count = int(metadata["not_dead_snip"].sum())

    fatal = ""
    if image_missing == len(sample):
        fatal = "all_sampled_image_paths_missing"
    elif mask_missing == len(sample):
        fatal = "all_sampled_mask_paths_missing"

    notes = sorted(set(";".join(override_notes).split(";")) - {""})
    return {
        "experiment_id": experiment,
        "metadata_csv_exists": metadata_path(experiment).exists(),
        "metadata_csv_path": str(metadata_path(experiment)),
        "n_frames_metadata": int(len(metadata)),
        "n_not_dead_snips": not_dead_count,
        "image_paths_resolved": image_missing == 0,
        "n_image_paths_missing": int(image_missing),
        "mask_paths_resolved": mask_missing == 0,
        "n_mask_paths_missing": int(mask_missing),
        "source_stack_resolved": nd2_path is not None and nd2_path.exists(),
        "source_stack_path": str(nd2_path) if nd2_path is not None else "",
        "metrics_focus_existing": len(focus_existing) > 0,
        "n_focus_metric_rows": int(len(focus_existing)),
        "metrics_motion_existing": len(motion_existing) > 0,
        "n_motion_metric_rows": int(len(motion_existing)),
        "needs_focus_recompute": len(focus_existing) == 0,
        "needs_motion_recompute": len(motion_existing) == 0,
        "fatal_error": fatal,
        "notes": ";".join(notes),
    }


def write_preflight(experiments: list[str], out_dir: Path, focus_df: pd.DataFrame, motion_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metadata_by_exp = {}
    rows = []
    for experiment in experiments:
        try:
            metadata = load_metadata(experiment)
            metadata_by_exp[experiment] = metadata
            rows.append(preflight_experiment(experiment, metadata, focus_df, motion_df))
        except Exception as exc:
            rows.append(
                {
                    "experiment_id": experiment,
                    "metadata_csv_exists": metadata_path(experiment).exists(),
                    "metadata_csv_path": str(metadata_path(experiment)),
                    "n_frames_metadata": 0,
                    "n_not_dead_snips": 0,
                    "fatal_error": str(exc),
                    "notes": "metadata_load_failed",
                }
            )
    preflight = pd.DataFrame(rows)
    preflight.to_csv(tables_dir(out_dir) / "qc_metric_availability_preflight.csv", index=False)
    print(f"Saved preflight -> {tables_dir(out_dir) / 'qc_metric_availability_preflight.csv'}", flush=True)
    return preflight, metadata_by_exp


def metric_requested(metric_type: str, metric: str) -> bool:
    return metric_type == "both" or metric_type == metric


def merge_replace(existing: pd.DataFrame, new: pd.DataFrame, experiments: list[str]) -> pd.DataFrame:
    if existing.empty:
        return new
    keep = existing[~existing["experiment_id"].astype(str).isin(experiments)].copy()
    return pd.concat([keep, new], ignore_index=True)


def assign_deciles(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    rows = []
    for experiment, grp in df.dropna(subset=[metric]).groupby("experiment_id", observed=True):
        ordered = grp.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
        if len(ordered) == 0:
            continue
        if len(ordered) < 2:
            ordered["decile"] = 1
        else:
            labels = pd.qcut(ordered[metric].rank(method="first"), N_DECILES, labels=False, duplicates="drop")
            ordered["decile"] = labels.fillna(0).astype(int) + 1
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(df.columns) + ["decile"])


def sample_decile_manifest(
    df: pd.DataFrame,
    metric_type: str,
    metric: str,
    figure_path: Path,
    n_per_decile: int,
    smoke_deciles: int | None,
    higher_is_better: bool,
) -> pd.DataFrame:
    with_deciles = assign_deciles(df, metric, higher_is_better=higher_is_better)
    deciles = sorted(with_deciles["decile"].dropna().unique())
    if smoke_deciles is not None:
        deciles = deciles[: max(1, smoke_deciles)]
    rows = []
    for experiment in sorted(with_deciles["experiment_id"].astype(str).unique()):
        for decile in deciles:
            grp = with_deciles[
                (with_deciles["experiment_id"].astype(str) == experiment)
                & (with_deciles["decile"].astype(int) == int(decile))
            ].sort_values(metric, ascending=not higher_is_better)
            if grp.empty:
                continue
            if len(grp) <= n_per_decile:
                picks = grp
            else:
                idx = np.linspace(0, len(grp) - 1, n_per_decile).astype(int)
                picks = grp.iloc[idx]
            for rank, (_, row) in enumerate(picks.iterrows(), start=1):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "metric_type": metric_type,
                        "decile": int(decile),
                        "sample_rank_within_decile": rank,
                        "well_id": row.get("well_id", row.get("well", "")),
                        "time_index": row.get("time_index", row.get("frame_index", np.nan)),
                        "image_path": row.get("image_path", ""),
                        "mask_path": row.get("mask_path", ""),
                        "primary_metric": row.get(metric, np.nan),
                        "pass_fail_status": row.get("focus_pass_fail_status", ""),
                        "tile_status": row.get("focus_tile_status", ""),
                        "figure_path": str(figure_path),
                        "snip_id": row.get("snip_id", ""),
                        "image_id": row.get("image_id", ""),
                    }
                )
    return pd.DataFrame(rows)


def border_style(status: str) -> tuple[str, float]:
    if status == "rejected_by_both":
        return "#7b1fa2", 3.0
    if status == "entropy_only_reject":
        return "#b5651d", 2.5
    if status == "sharpness_only_reject":
        return "#1f77b4", 2.5
    return "#1a8820", 1.8


def plot_gallery(manifest: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if manifest.empty:
        return
    experiments = sorted(manifest["experiment_id"].astype(str).unique())
    deciles = sorted(manifest["decile"].dropna().astype(int).unique())
    block_rows = 2
    block_cols = 3
    ncols = len(experiments) * block_cols
    nrows = len(deciles) * block_rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 1.8), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for d_i, decile in enumerate(deciles):
        for e_i, experiment in enumerate(experiments):
            grp = manifest[
                (manifest["decile"].astype(int) == decile)
                & (manifest["experiment_id"].astype(str) == experiment)
            ].sort_values("sample_rank_within_decile")
            for j, (_, row) in enumerate(grp.head(block_rows * block_cols).iterrows()):
                rr = d_i * block_rows + j // block_cols
                cc = e_i * block_cols + j % block_cols
                ax = axes[rr, cc]
                ax.axis("on")
                image_path = Path(str(row.get("image_path", "")))
                if image_path.exists():
                    ax.imshow(np.array(Image.open(image_path).convert("L")), cmap="gray")
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
                color, lw = border_style(str(row.get("tile_status", "")))
                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(lw)
                ax.set_title(
                    f"{experiment} {row.get('well_id', '')} t{row.get('time_index', '')}\n"
                    f"{metric}={row.get('primary_metric', np.nan):.3g}",
                    fontsize=5.5,
                )
            axes[d_i * block_rows, e_i * block_cols].set_ylabel(f"D{decile}", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved gallery -> {out_path}", flush=True)


def write_focus_threshold_summary(focus_df: pd.DataFrame, out_dir: Path) -> None:
    if focus_df.empty:
        return
    rows = []
    for exp, grp in focus_df.groupby("experiment_id", observed=True):
        valid = grp[grp[FOCUS_METRIC].notna()].copy()
        entropy = valid["ff_rel_entropy"] < ENTROPY_CUT
        sharp = valid["ff_lap_abs_ratio"] < SHARPNESS_RATIO_CUT
        rows.append(
            {
                "experiment_id": exp,
                "n_not_dead_snips": int(len(valid)),
                "entropy_only_reject_frac": float((entropy & ~sharp).mean()) if len(valid) else np.nan,
                "sharpness_only_reject_frac": float((~entropy & sharp).mean()) if len(valid) else np.nan,
                "reject_by_both_frac": float((entropy & sharp).mean()) if len(valid) else np.nan,
                "any_reject_frac": float((entropy | sharp).mean()) if len(valid) else np.nan,
                "ff_rel_entropy_threshold": ENTROPY_CUT,
                "ff_lap_abs_ratio_threshold": SHARPNESS_RATIO_CUT,
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir(out_dir) / "focus_threshold_summary_not_dead_snips.csv", index=False)


def write_motion_threshold_summary(motion_df: pd.DataFrame, out_dir: Path) -> None:
    if motion_df.empty:
        pd.DataFrame(columns=["experiment_id", "n_motion_rows"]).to_csv(
            tables_dir(out_dir) / "motion_threshold_summary.csv", index=False
        )
        return
    rows = []
    for exp, grp in motion_df.groupby("experiment_id", observed=True):
        rows.append(
            {
                "experiment_id": exp,
                "n_motion_rows": int(len(grp)),
                "ncc_p05_median": float(grp["ncc_p05"].median()) if "ncc_p05" in grp else np.nan,
                "ncc_p05_q10": float(grp["ncc_p05"].quantile(0.10)) if "ncc_p05" in grp else np.nan,
                "bad_pair_frac_mean": float(grp["bad_pair_frac"].mean()) if "bad_pair_frac" in grp else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir(out_dir) / "motion_threshold_summary.csv", index=False)


def plot_focus_reject_rate(focus_df: pd.DataFrame, out_dir: Path) -> None:
    if focus_df.empty or "predicted_stage_hpf" not in focus_df:
        return
    df = focus_df.dropna(subset=["predicted_stage_hpf", "ff_rel_entropy", "ff_lap_abs_ratio"]).copy()
    if df.empty:
        return
    experiments = sorted(df["experiment_id"].astype(str).unique())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10, 3.2 * len(experiments)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.arange(0, max(120, math.ceil(df["predicted_stage_hpf"].max() / 5) * 5 + 5), 5)
    for ax, exp in zip(axes, experiments):
        sub = df[df["experiment_id"].astype(str) == exp].copy()
        sub["stage_bin"] = pd.cut(sub["predicted_stage_hpf"], bins=bins, include_lowest=True)
        rows = []
        for interval, grp in sub.groupby("stage_bin", observed=True):
            if grp.empty:
                continue
            ent = grp["ff_rel_entropy"] < ENTROPY_CUT
            sharp = grp["ff_lap_abs_ratio"] < SHARPNESS_RATIO_CUT
            center = (interval.left + interval.right) / 2
            rows.append(
                {
                    "stage": center,
                    "reject_by_both": 100 * float((ent & sharp).mean()),
                    "entropy_only": 100 * float((ent & ~sharp).mean()),
                    "sharpness_only": 100 * float((~ent & sharp).mean()),
                    "any_reject": 100 * float((ent | sharp).mean()),
                }
            )
        plot_df = pd.DataFrame(rows)
        if plot_df.empty:
            continue
        ax.plot(plot_df["stage"], plot_df["reject_by_both"], marker="o", color="#7b1fa2", label="reject by BOTH")
        ax.plot(plot_df["stage"], plot_df["entropy_only"], marker="o", color="#b5651d", label="entropy only")
        ax.plot(plot_df["stage"], plot_df["sharpness_only"], marker="o", color="#1f77b4", label="sharpness only")
        ax.plot(plot_df["stage"], plot_df["any_reject"], linestyle="--", color="#888888", label="any reject")
        ax.set_title(f"{exp} not-dead snips (n={len(sub)})")
        ax.set_ylabel("% rejected")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("predicted_stage_hpf")
    fig.suptitle("Focus-QC reject rate vs stage by experiment (not-dead snips)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = figures_dir(out_dir) / "focus_reject_rate_by_experiment_not_dead_snips.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved focus reject-rate plot -> {out}", flush=True)


def plot_focus_histograms(focus_df: pd.DataFrame, out_dir: Path) -> None:
    if focus_df.empty:
        return
    df = focus_df.dropna(subset=["ff_rel_entropy"]).copy()
    if df.empty:
        return
    experiments = sorted(df["experiment_id"].astype(str).unique())
    fig, axes = plt.subplots(len(experiments), 2, figsize=(12, 3.1 * len(experiments)), squeeze=False)
    for r, exp in enumerate(experiments):
        sub = df[df["experiment_id"].astype(str) == exp]
        ax = axes[r, 0]
        ax.hist(sub["ff_rel_entropy"].dropna(), bins=60, color="#4c72b0", alpha=0.82)
        ax.axvline(ENTROPY_CUT, color="#b5651d", linestyle="--", linewidth=1.8, label=f"threshold {ENTROPY_CUT:g}")
        ax.set_title(f"{exp}: ff_rel_entropy (not-dead snips, n={len(sub)})")
        ax.set_ylabel("snips")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = axes[r, 1]
        if "ff_lap_abs_ratio" in sub:
            ax.hist(sub["ff_lap_abs_ratio"].dropna(), bins=60, color="#55a868", alpha=0.82)
            ax.axvline(SHARPNESS_RATIO_CUT, color="#1f77b4", linestyle="--", linewidth=1.8, label=f"threshold {SHARPNESS_RATIO_CUT:g}")
            ax.set_title(f"{exp}: ff_lap_abs_ratio")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
        else:
            ax.axis("off")
    axes[-1, 0].set_xlabel("ff_rel_entropy")
    axes[-1, 1].set_xlabel("ff_lap_abs_ratio")
    fig.suptitle("Focus metric histograms by experiment (not-dead snips)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = figures_dir(out_dir) / "focus_metric_histograms_by_experiment_not_dead_snips.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved focus histograms -> {out}", flush=True)


def plot_motion_distribution(motion_df: pd.DataFrame, out_dir: Path) -> None:
    if motion_df.empty or "ncc_p05" not in motion_df:
        return
    df = motion_df.dropna(subset=["ncc_p05"]).copy()
    if df.empty:
        return
    experiments = sorted(df["experiment_id"].astype(str).unique())
    fig, axes = plt.subplots(len(experiments), 1, figsize=(9, 3 * len(experiments)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, exp in zip(axes, experiments):
        sub = df[df["experiment_id"].astype(str) == exp]
        ax.hist(sub["ncc_p05"].dropna(), bins=60, color="#8172b2", alpha=0.85)
        ax.set_title(f"{exp}: ncc_p05 (n={len(sub)})")
        ax.set_ylabel("stacks")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("ncc_p05")
    fig.tight_layout()
    out = figures_dir(out_dir) / "motion_metric_distribution_by_experiment.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved motion distribution -> {out}", flush=True)


def write_outputs(
    focus_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    out_dir: Path,
    examples_per_decile: int,
    smoke_deciles: int | None,
) -> None:
    if not focus_df.empty:
        focus_df.to_csv(tables_dir(out_dir) / "focus_metrics_by_experiment.csv", index=False)
        focus_not_dead = focus_df[focus_df["not_dead_snip"].astype(bool)].copy() if "not_dead_snip" in focus_df else focus_df.copy()
        write_focus_threshold_summary(focus_not_dead, out_dir)
        plot_focus_reject_rate(focus_not_dead, out_dir)
        plot_focus_histograms(focus_not_dead, out_dir)
        focus_fig = figures_dir(out_dir) / "focus_deciles_by_experiment_ff_rel_entropy.png"
        focus_manifest = sample_decile_manifest(
            focus_not_dead.dropna(subset=[FOCUS_METRIC]).copy(),
            "focus",
            FOCUS_METRIC,
            focus_fig,
            examples_per_decile,
            smoke_deciles,
            higher_is_better=False,
        )
        focus_manifest.to_csv(tables_dir(out_dir) / "focus_decile_gallery_samples.csv", index=False)
        plot_gallery(focus_manifest, FOCUS_METRIC, focus_fig, "Focus deciles by experiment: ff_rel_entropy (low to high)")

    if not motion_df.empty:
        motion_df.to_csv(tables_dir(out_dir) / "motion_metrics_by_experiment.csv", index=False)
        motion_not_dead = motion_df[motion_df["not_dead_snip"].astype(bool)].copy() if "not_dead_snip" in motion_df else motion_df.copy()
        write_motion_threshold_summary(motion_not_dead, out_dir)
        plot_motion_distribution(motion_not_dead, out_dir)
        if "ncc_p05" in motion_not_dead:
            motion_fig = figures_dir(out_dir) / "motion_deciles_by_experiment_ncc_p05.png"
            motion_for_gallery = motion_not_dead.copy()
            if "image_path" not in motion_for_gallery:
                motion_for_gallery["image_path"] = ""
            if "mask_path" not in motion_for_gallery:
                motion_for_gallery["mask_path"] = ""
            motion_manifest = sample_decile_manifest(
                motion_for_gallery.dropna(subset=[MOTION_METRIC]),
                "motion",
                MOTION_METRIC,
                motion_fig,
                examples_per_decile,
                smoke_deciles,
                higher_is_better=False,
            )
            motion_manifest.to_csv(tables_dir(out_dir) / "motion_decile_gallery_samples.csv", index=False)
            plot_gallery(motion_manifest, MOTION_METRIC, motion_fig, "Motion deciles by experiment: ncc_p05 (low to high)")
    else:
        write_motion_threshold_summary(motion_df, out_dir)
        pd.DataFrame(columns=[
            "experiment_id",
            "metric_type",
            "decile",
            "sample_rank_within_decile",
            "well_id",
            "time_index",
            "image_path",
            "mask_path",
            "primary_metric",
            "pass_fail_status",
            "tile_status",
            "figure_path",
        ]).to_csv(tables_dir(out_dir) / "motion_decile_gallery_samples.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir(out_dir)
    figures_dir(out_dir)

    experiments = experiments_from_arg(args.experiments)
    focus_df = existing_focus_metrics(out_dir)
    motion_df = existing_motion_metrics(out_dir)

    preflight, metadata_by_exp = write_preflight(experiments, out_dir, focus_df, motion_df)
    print(preflight.to_string(index=False), flush=True)
    if args.preflight_only:
        return

    fatal = preflight[preflight["fatal_error"].fillna("").astype(str) != ""]
    if not fatal.empty:
        raise RuntimeError(f"Preflight fatal errors present:\n{fatal[['experiment_id', 'fatal_error']].to_string(index=False)}")

    recompute = args.recompute_missing or args.recompute_all
    new_focus = []
    new_motion = []

    if metric_requested(args.metric_type, "focus"):
        for experiment in experiments:
            existing = focus_df[focus_df.get("experiment_id", pd.Series(dtype=str)).astype(str) == experiment] if not focus_df.empty else pd.DataFrame()
            if args.recompute_all or (args.recompute_missing and existing.empty):
                new_focus.append(
                    compute_focus_metrics_for_experiment(
                        experiment,
                        metadata_by_exp[experiment],
                        limit=args.focus_limit_per_experiment,
                    )
                )
            elif not existing.empty:
                new_focus.append(existing)
            elif not recompute:
                print(f"[focus {experiment}] metrics missing; rerun with --recompute-missing or --recompute-all", flush=True)

    if metric_requested(args.metric_type, "motion"):
        for experiment in experiments:
            existing = motion_df[motion_df.get("experiment_id", pd.Series(dtype=str)).astype(str) == experiment] if not motion_df.empty else pd.DataFrame()
            if args.recompute_all or (args.recompute_missing and existing.empty):
                new_motion.append(
                    compute_motion_metrics_for_experiment(
                        experiment,
                        metadata_by_exp[experiment],
                        limit=args.motion_limit_per_experiment,
                    )
                )
            elif not existing.empty:
                new_motion.append(existing)
            elif not recompute:
                print(f"[motion {experiment}] metrics missing; rerun with --recompute-missing or --recompute-all", flush=True)

    final_focus = pd.concat(new_focus, ignore_index=True) if new_focus else focus_df
    final_motion = pd.concat(new_motion, ignore_index=True) if new_motion else motion_df
    if new_focus:
        final_focus = merge_replace(focus_df, final_focus, experiments)
    if new_motion:
        final_motion = merge_replace(motion_df, final_motion, experiments)

    write_outputs(
        final_focus,
        final_motion,
        out_dir,
        examples_per_decile=args.examples_per_experiment_decile,
        smoke_deciles=args.smoke_deciles,
    )

    # Refresh preflight after any newly generated metrics.
    write_preflight(experiments, out_dir, final_focus, final_motion)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
