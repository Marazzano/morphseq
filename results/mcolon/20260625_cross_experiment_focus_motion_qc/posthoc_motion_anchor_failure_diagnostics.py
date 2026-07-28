#!/usr/bin/env python
"""Diagnose motion-QC anchor failures.

The current cross-experiment motion table stores stack-level NCC summaries.
This script reopens a small set of anchor stacks and writes pair-level metrics
so we can see why visibly moving stacks can have bad_pair_frac == 0.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

from src.data_pipeline.quality_control.zstack_motion_qc.grids import (  # noqa: E402
    compute_local_ncc_grid,
    tile_origin_coords,
)
from src.data_pipeline.quality_control.zstack_motion_qc.kernels import ncc  # noqa: E402


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures" / "motion_anchor_failure_modes"
MOTION_CSV = TABLES / "motion_metrics_by_experiment.csv"

BAD_NCC_THRESH = 0.90
NCC_P05_CUT = 0.85
BAD_PAIR_FRAC_CUT = 0.10
TILE_SIZE = 128
STRIDE = 128
MIN_TILE_COVERAGE = 0.10

ANCHORS = [
    ("20251125", "H05", 19, "visible_motion_missed_bad_pair"),
    ("20251125", "H05", 25, "visible_motion_missed_bad_pair"),
    ("20251125", "D06", 51, "visible_motion_missed_bad_pair"),
    ("20251125", "H05", 162, "motion_caught_dead_snip"),
]


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def load_anchor_rows() -> pd.DataFrame:
    df = pd.read_csv(MOTION_CSV)
    rows = []
    for experiment, well, time_index, label in ANCHORS:
        sub = df[
            (df["experiment_id"].astype(str) == experiment)
            & (df["well_id"].astype(str) == well)
            & (pd.to_numeric(df["time_index"], errors="coerce") == time_index)
        ].copy()
        if sub.empty:
            rows.append(
                {
                    "experiment_id": experiment,
                    "well_id": well,
                    "time_index": time_index,
                    "anchor_label": label,
                    "anchor_error": "missing_motion_row",
                }
            )
            continue
        row = sub.iloc[0].to_dict()
        row["anchor_label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def load_stack(nd2_path: Path, time_index: int, p: int) -> np.ndarray:
    import nd2

    with nd2.ND2File(str(nd2_path)) as nd:
        stack = nd.to_dask()[time_index, p].compute()
    return stack.astype(np.float32)


def load_mask(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    return np.array(Image.open(mask_path)) > 0


def valid_tiles_from_mask(mask: np.ndarray, min_coverage: float) -> np.ndarray:
    y_origins, x_origins = tile_origin_coords(mask.shape, TILE_SIZE, STRIDE)
    valid = np.zeros((len(y_origins), len(x_origins)), dtype=bool)
    for iy, y0 in enumerate(y_origins):
        y1 = min(y0 + TILE_SIZE, mask.shape[0])
        for ix, x0 in enumerate(x_origins):
            x1 = min(x0 + TILE_SIZE, mask.shape[1])
            valid[iy, ix] = float(mask[y0:y1, x0:x1].mean()) >= min_coverage
    return valid


def summarize_scope(ncc_values: np.ndarray, scope: str) -> tuple[dict[str, float], pd.DataFrame]:
    pair_rows = []
    for pair_idx in range(ncc_values.shape[0]):
        vals = ncc_values[pair_idx].ravel()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            pair_rows.append(
                {
                    "scope": scope,
                    "pair_idx": pair_idx,
                    "pair_mean_ncc": np.nan,
                    "pair_p05_ncc": np.nan,
                    "pair_min_ncc": np.nan,
                    "pair_bad_tile_frac": np.nan,
                    "pair_is_bad_by_mean": True,
                    "pair_is_bad_by_tile_frac_010": True,
                    "pair_n_tiles": 0,
                }
            )
            continue

        pair_bad_tile_frac = float(np.mean(vals < BAD_NCC_THRESH))
        pair_rows.append(
            {
                "scope": scope,
                "pair_idx": pair_idx,
                "pair_mean_ncc": float(np.mean(vals)),
                "pair_p05_ncc": float(np.percentile(vals, 5)),
                "pair_min_ncc": float(np.min(vals)),
                "pair_bad_tile_frac": pair_bad_tile_frac,
                "pair_is_bad_by_mean": bool(np.mean(vals) < BAD_NCC_THRESH),
                "pair_is_bad_by_tile_frac_010": bool(pair_bad_tile_frac > 0.10),
                "pair_n_tiles": int(vals.size),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    flat = ncc_values.ravel()
    flat = flat[~np.isnan(flat)]
    pair_means = pair_df["pair_mean_ncc"].to_numpy(dtype=float)
    summary = {
        f"{scope}_ncc_mean": float(np.mean(flat)) if flat.size else np.nan,
        f"{scope}_ncc_min": float(np.min(flat)) if flat.size else np.nan,
        f"{scope}_ncc_p05": float(np.percentile(flat, 5)) if flat.size else np.nan,
        f"{scope}_bad_pair_frac_by_mean": float(np.mean(pair_means < BAD_NCC_THRESH))
        if pair_means.size
        else np.nan,
        f"{scope}_bad_tile_frac": float(np.mean(flat < BAD_NCC_THRESH)) if flat.size else np.nan,
        f"{scope}_bad_pair_frac_by_tile_frac_010": float(pair_df["pair_is_bad_by_tile_frac_010"].mean())
        if len(pair_df)
        else np.nan,
        f"{scope}_longest_bad_run_by_mean": longest_true_run(pair_means < BAD_NCC_THRESH)
        if pair_means.size
        else 0,
        f"{scope}_longest_bad_run_by_tile_frac_010": longest_true_run(
            pair_df["pair_is_bad_by_tile_frac_010"].to_numpy(dtype=bool)
        )
        if len(pair_df)
        else 0,
    }
    return summary, pair_df


def summarize_mask_pixels(stack: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    """One NCC per adjacent z-pair using all pixels inside the embryo mask."""
    vals = []
    rows = []
    for pair_idx in range(stack.shape[0] - 1):
        pair_ncc = ncc(stack[pair_idx][mask], stack[pair_idx + 1][mask])
        vals.append(pair_ncc)
        rows.append(
            {
                "scope": "mask_pixels",
                "pair_idx": pair_idx,
                "pair_mean_ncc": pair_ncc,
                "pair_p05_ncc": pair_ncc,
                "pair_min_ncc": pair_ncc,
                "pair_bad_tile_frac": np.nan,
                "pair_is_bad_by_mean": bool(np.isfinite(pair_ncc) and pair_ncc < BAD_NCC_THRESH),
                "pair_is_bad_by_tile_frac_010": np.nan,
                "pair_n_tiles": np.nan,
                "pair_n_pixels": int(mask.sum()),
            }
        )
    pair_df = pd.DataFrame(rows)
    arr = np.array(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    summary = {
        "mask_pixels_ncc_mean": float(np.mean(finite)) if finite.size else np.nan,
        "mask_pixels_ncc_min": float(np.min(finite)) if finite.size else np.nan,
        "mask_pixels_ncc_p05": float(np.percentile(finite, 5)) if finite.size else np.nan,
        "mask_pixels_bad_pair_frac_by_ncc": float(np.mean(arr < BAD_NCC_THRESH)) if arr.size else np.nan,
        "mask_pixels_longest_bad_run_by_ncc": longest_true_run(arr < BAD_NCC_THRESH) if arr.size else 0,
        "mask_pixels_n_pixels": int(mask.sum()),
    }
    return summary, pair_df


def crop_bounds(mask: np.ndarray, pad: int = 60) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def normalize_image(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def plot_pair_profile(pair_df: pd.DataFrame, row: pd.Series, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    scopes = [scope for scope in ["all_tiles", "mask_tiles"] if scope in set(pair_df["scope"])]
    colors = {"all_tiles": "#6f6f6f", "mask_tiles": "#1f77b4"}

    for scope in scopes:
        sub = pair_df[pair_df["scope"] == scope].sort_values("pair_idx")
        x = sub["pair_idx"].to_numpy()
        axes[0].plot(x, sub["pair_mean_ncc"], marker="o", color=colors[scope], label=f"{scope} mean")
        axes[0].plot(
            x,
            sub["pair_p05_ncc"],
            marker="s",
            color=colors[scope],
            alpha=0.55,
            linestyle="--",
            label=f"{scope} p05",
        )
        axes[1].plot(x, sub["pair_bad_tile_frac"], marker="o", color=colors[scope], label=scope)
        axes[2].plot(x, sub["pair_min_ncc"], marker="o", color=colors[scope], label=scope)

    axes[0].axhline(BAD_NCC_THRESH, color="red", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("pair NCC")
    axes[0].set_ylim(-0.05, 1.02)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].axhline(0.10, color="red", linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("pair bad-tile frac")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    axes[2].axhline(NCC_P05_CUT, color="red", linestyle=":", linewidth=1.2)
    axes[2].set_ylabel("pair min NCC")
    axes[2].set_xlabel("adjacent z-pair index")
    axes[2].set_ylim(-1.02, 1.02)
    axes[2].legend(fontsize=8)

    title = (
        f"{row['experiment_id']} {row['well_id']} t{int(row['time_index'])} "
        f"| table p05={float(row.get('ncc_p05', np.nan)):.3f} "
        f"bad_pair={float(row.get('bad_pair_frac', np.nan)):.3f} "
        f"bad_tile={float(row.get('ncc_bad_tile_frac', np.nan)):.3f} "
        f"not_dead={row.get('not_dead_snip', '')}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_worst_pair(stack: np.ndarray, ncc_grid: np.ndarray, mask: np.ndarray | None, row: pd.Series, out_path: Path) -> None:
    flat = ncc_grid.reshape(ncc_grid.shape[0], -1)
    pair_p05 = np.nanpercentile(flat, 5, axis=1)
    worst_pair = int(np.nanargmin(pair_p05))

    z0 = stack[worst_pair]
    z1 = stack[worst_pair + 1]
    diff = np.abs(z1 - z0)

    if mask is None:
        y0, y1, x0, x1 = 0, stack.shape[1], 0, stack.shape[2]
    else:
        y0, y1, x0, x1 = crop_bounds(mask)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = [
        (z0, f"z{worst_pair}"),
        (z1, f"z{worst_pair + 1}"),
        (diff, "abs diff"),
    ]
    for ax, (image, title) in zip(axes[:3], panels):
        ax.imshow(normalize_image(image[y0:y1, x0:x1]), cmap="gray", interpolation="nearest")
        if mask is not None:
            crop_mask = mask[y0:y1, x0:x1]
            ax.contour(crop_mask.astype(float), levels=[0.5], colors=["#00aa33"], linewidths=0.8)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    heat = ncc_grid[worst_pair]
    im = axes[3].imshow(heat, vmin=0, vmax=1, cmap="magma_r", interpolation="nearest")
    axes[3].set_title(f"NCC grid pair {worst_pair}", fontsize=9)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    for iy in range(heat.shape[0]):
        for ix in range(heat.shape[1]):
            if np.isfinite(heat[iy, ix]) and heat[iy, ix] < BAD_NCC_THRESH:
                axes[3].add_patch(Rectangle((ix - 0.5, iy - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.0))
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{row['experiment_id']} {row['well_id']} t{int(row['time_index'])}: worst pair by p05",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    anchor_rows = load_anchor_rows()
    summary_rows = []
    pair_tables = []

    for _, row in anchor_rows.iterrows():
        if row.get("anchor_error"):
            summary_rows.append(row.to_dict())
            continue

        experiment = str(row["experiment_id"])
        well = str(row["well_id"])
        time_index = int(row["time_index"])
        print(f"[anchor] {experiment} {well} t{time_index}", flush=True)

        stack = load_stack(Path(row["nd2_path"]), time_index, int(row["p"]))
        ncc_grid = compute_local_ncc_grid(stack, tile_size=TILE_SIZE, stride=STRIDE)
        mask = load_mask(Path(row["mask_path"]))

        all_summary, all_pairs = summarize_scope(ncc_grid, "all_tiles")
        all_pairs.insert(0, "experiment_id", experiment)
        all_pairs.insert(1, "well_id", well)
        all_pairs.insert(2, "time_index", time_index)
        pair_tables.append(all_pairs)

        combined_summary = {
            "experiment_id": experiment,
            "well_id": well,
            "time_index": time_index,
            "anchor_label": row["anchor_label"],
            "not_dead_snip": row.get("not_dead_snip", np.nan),
            "table_ncc_p05": row.get("ncc_p05", np.nan),
            "table_bad_pair_frac": row.get("bad_pair_frac", np.nan),
            "table_ncc_bad_tile_frac": row.get("ncc_bad_tile_frac", np.nan),
            "table_current_refined_fail": bool(
                float(row.get("ncc_p05", np.nan)) < NCC_P05_CUT
                and float(row.get("bad_pair_frac", np.nan)) > BAD_PAIR_FRAC_CUT
            ),
            **all_summary,
        }

        if mask is not None:
            valid_tiles = valid_tiles_from_mask(mask, MIN_TILE_COVERAGE)
            if valid_tiles.any():
                mask_grid = ncc_grid[:, valid_tiles]
                mask_summary, mask_pairs = summarize_scope(mask_grid[:, :, None], "mask_tiles")
                mask_pairs.insert(0, "experiment_id", experiment)
                mask_pairs.insert(1, "well_id", well)
                mask_pairs.insert(2, "time_index", time_index)
                pair_tables.append(mask_pairs)
                combined_summary.update(mask_summary)
                combined_summary["mask_n_valid_tiles"] = int(valid_tiles.sum())
            else:
                combined_summary["anchor_error"] = "mask_has_no_valid_tiles"
            pixel_summary, pixel_pairs = summarize_mask_pixels(stack, mask)
            pixel_pairs.insert(0, "experiment_id", experiment)
            pixel_pairs.insert(1, "well_id", well)
            pixel_pairs.insert(2, "time_index", time_index)
            pair_tables.append(pixel_pairs)
            combined_summary.update(pixel_summary)
        else:
            combined_summary["anchor_error"] = "mask_missing"

        pair_df_for_anchor = pd.concat(
            [tbl for tbl in pair_tables if str(tbl["well_id"].iloc[0]) == well and int(tbl["time_index"].iloc[0]) == time_index],
            ignore_index=True,
        )
        prefix = f"{experiment}_{well}_t{time_index:04d}"
        plot_pair_profile(pair_df_for_anchor, row, FIGURES / f"{prefix}_pair_profile.png")
        plot_worst_pair(stack, ncc_grid, mask, row, FIGURES / f"{prefix}_worst_pair.png")

        summary_rows.append(combined_summary)

    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    summary_path = TABLES / "motion_anchor_failure_summary.csv"
    pair_path = TABLES / "motion_anchor_pair_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    print(f"saved {summary_path}")
    print(f"saved {pair_path}")
    print(f"saved figures in {FIGURES}")


if __name__ == "__main__":
    main()
