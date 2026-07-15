"""Generate cleaned and improved A02 focus-stack comparison images.

Both variants use deterministic shared bounds across every tile/Z plane and
gather the selected pixels from the raw uint16 stacks. The only difference is
whether the LoG scoring tensor is clipped at the shared display bounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio
import torch

from src.data_pipeline.acquisition.image_building.shared.log_focus import LoG_focus_stacker
from src.data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    PreComputeStitchParams,
    TileSpec,
    stitch_frame_tiles,
)


EXPERIMENT = "20260702_hotchem_30hpf_plate01"
WELL = "A02"
TIME = 0
ROOT = Path("pipeline/output/acquisition") / EXPERIMENT
STEM = f"{EXPERIMENT}_{WELL}_BF_t{TIME:04d}"
FOCUS_DIR = ROOT / "materialized_images" / f"{EXPERIMENT}_{WELL}" / "BF/projection/focus_stack"
MANIFEST = FOCUS_DIR / "raw_tile_manifest" / f"{STEM}.json"
MASTER = ROOT / "ingest_metadata/keyence_stitch_map__keyence.json"
CURRENT = FOCUS_DIR / f"{STEM}.png"
OUT = Path("results/mcolon/20260712_legacy_focus_stack_comparison/outputs")


def exact_uint16_percentile_bounds(raw: np.ndarray, low: float, high: float) -> tuple[int, int]:
    """Return deterministic nearest-rank percentiles from a uint16 histogram."""
    if raw.dtype != np.uint16:
        raise TypeError(f"Expected uint16 raw data; got {raw.dtype}.")
    counts = np.bincount(raw.reshape(-1), minlength=65536)
    cumulative = np.cumsum(counts, dtype=np.int64)
    n = int(cumulative[-1])
    low_rank = int(np.floor((low / 100.0) * (n - 1))) + 1
    high_rank = int(np.floor((high / 100.0) * (n - 1))) + 1
    lo = int(np.searchsorted(cumulative, low_rank, side="left"))
    hi = int(np.searchsorted(cumulative, high_rank, side="left"))
    if hi <= lo:
        raise ValueError(f"Degenerate shared bounds: lo={lo}, hi={hi}.")
    return lo, hi


def shared_affine(raw: np.ndarray, lo: int, hi: int, *, clip: bool) -> np.ndarray:
    out = (raw.astype(np.float32) - float(lo)) / float(hi - lo)
    return np.clip(out, 0.0, 1.0) if clip else out


def gather_raw(raw: np.ndarray, focus_index: np.ndarray) -> np.ndarray:
    return np.take_along_axis(raw, focus_index[:, None, :, :], axis=1).squeeze(1)


def display_u8(focused_raw: np.ndarray, lo: int, hi: int) -> np.ndarray:
    mapped = np.clip(
        (focused_raw.astype(np.float32) - float(lo)) / float(hi - lo),
        0.0,
        1.0,
    )
    return np.rint(mapped * 255.0).astype(np.uint8)


def stitch(tiles: np.ndarray, tile_ids: list[str]) -> np.ndarray:
    specs = [TileSpec(tile_id=tile_id, image=tiles[i]) for i, tile_id in enumerate(tile_ids)]
    return stitch_frame_tiles(
        specs,
        FrameTilingConfig(orientation="horizontal"),
        PreComputeStitchParams(master_params_path=MASTER),
    ).stitched


def metrics(image: np.ndarray) -> dict[str, float | int]:
    return {
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "p01": float(np.percentile(image, 1)),
        "p99": float(np.percentile(image, 99)),
        "fraction_0": float(np.mean(image == 0)),
        "fraction_255": float(np.mean(image == 255)),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    tile_ids = [str(tile["tile_id"]) for tile in manifest["tiles"]]
    raw = np.stack(
        [
            np.stack([skio.imread(path) for path in tile["source_tiff_paths"]], axis=0)
            for tile in manifest["tiles"]
        ],
        axis=0,
    )
    lo, hi = exact_uint16_percentile_bounds(raw, 0.1, 99.9)

    variants: dict[str, np.ndarray] = {}
    index_maps: dict[str, np.ndarray] = {}
    for name, clip_scores in (("shared_clean", True), ("improved_unclipped", False)):
        score_input = shared_affine(raw, lo, hi, clip=clip_scores)
        _, abs_log = LoG_focus_stacker(score_input, filter_size=3, device="cpu")
        focus_index = torch.argmax(abs_log, dim=1).cpu().numpy().astype(np.int16)
        focused_raw = gather_raw(raw, focus_index)
        variants[name] = stitch(display_u8(focused_raw, lo, hi), tile_ids)
        index_maps[name] = focus_index

    OUT.mkdir(parents=True, exist_ok=True)
    current = np.asarray(skio.imread(CURRENT))
    rows: list[dict[str, object]] = []
    for name, image in variants.items():
        if image.shape != current.shape:
            raise RuntimeError(f"{name} shape {image.shape} != current shape {current.shape}.")
        path = OUT / f"{STEM}__{name}.png"
        skio.imsave(path, image, check_contrast=False)
        row: dict[str, object] = {"variant": name, "path": str(path), "lo": lo, "hi": hi}
        row.update(metrics(image))
        row["mean_abs_difference_from_current"] = float(
            np.abs(image.astype(np.int16) - current.astype(np.int16)).mean()
        )
        rows.append(row)

    idx_diff = index_maps["shared_clean"] != index_maps["improved_unclipped"]
    np.savez_compressed(
        OUT / f"{STEM}__improved_focus_indices.npz",
        shared_clean=index_maps["shared_clean"],
        improved_unclipped=index_maps["improved_unclipped"],
        tile_ids=np.asarray(tile_ids),
        lo=np.asarray(lo),
        hi=np.asarray(hi),
    )
    summary = pd.DataFrame(rows)
    summary["shared_vs_unclipped_focus_index_difference_fraction"] = float(idx_diff.mean())
    summary_path = OUT / f"{STEM}__improved_comparison_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Shared exact bounds: lo={lo}, hi={hi}")
    print(
        "Shared-clipped vs improved-unclipped focus-index difference: "
        f"{idx_diff.sum():,}/{idx_diff.size:,} ({idx_diff.mean():.4%})"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
