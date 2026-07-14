"""Compare pipeline Keyence focus stacks with the legacy Build01 focus stage.

This intentionally imports the old Build01 dataset and LoG implementation rather
than reproducing their math.  Stitching uses the pipeline stitch map so that the
comparison changes only the focus-stack stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio
from skimage import util

from src.build.data_classes import MultiTileZStackDataset
from src.build.export_utils import LoG_focus_stacker
from src.data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    PreComputeStitchParams,
    TileSpec,
    stitch_frame_tiles,
)


def _legacy_focus_tiles(manifest_path: Path, device: str) -> list[TileSpec]:
    manifest = json.loads(manifest_path.read_text())
    entry = {
        "tile_zpaths": [tile["source_tiff_paths"] for tile in manifest["tiles"]],
    }
    # This is the exact legacy preprocessing path: a joint percentile range
    # across tiles, rescaling to [0, 1], then conversion to float16.
    legacy_sample = MultiTileZStackDataset([entry])[0]["data"]
    n_tiles, z, height, width = legacy_sample.shape
    flat = legacy_sample.reshape(n_tiles, z, height, width)
    ff, _ = LoG_focus_stacker(flat, filter_size=3, device=device)
    ff_np = ff.detach().cpu().numpy()
    return [
        TileSpec(tile_id=str(tile["tile_id"]), image=util.img_as_ubyte(ff_np[idx]))
        for idx, tile in enumerate(manifest["tiles"])
    ]


def _white_metrics(image: np.ndarray) -> dict[str, float | int]:
    return {
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "p99_9": float(np.percentile(image, 99.9)),
        "n_255": int(np.count_nonzero(image == 255)),
        "frac_255": float(np.mean(image == 255)),
        "frac_ge_250": float(np.mean(image >= 250)),
    }


def compare_well(
    experiment_root: Path,
    experiment: str,
    well: str,
    output_dir: Path,
    device: str,
) -> dict[str, object]:
    well_id = f"{experiment}_{well}"
    focus_dir = (
        experiment_root / "materialized_images" / well_id / "BF" / "projection" / "focus_stack"
    )
    stem = f"{well_id}_BF_t0000"
    current_path = focus_dir / f"{stem}.png"
    manifest_path = focus_dir / "raw_tile_manifest" / f"{stem}.json"
    master_path = experiment_root / "ingest_metadata" / "keyence_stitch_map__keyence.json"

    if not current_path.exists() or not manifest_path.exists() or not master_path.exists():
        raise FileNotFoundError(
            f"Missing comparison input for {well_id}: current={current_path.exists()}, "
            f"manifest={manifest_path.exists()}, stitch_map={master_path.exists()}"
        )

    legacy_tiles = _legacy_focus_tiles(manifest_path, device)
    stitched = stitch_frame_tiles(
        legacy_tiles,
        FrameTilingConfig(orientation="horizontal"),
        PreComputeStitchParams(master_params_path=master_path),
    ).stitched

    current = np.asarray(skio.imread(current_path))
    # The pipeline write policy rotates the horizontal Keyence mosaic into the
    # portrait orientation used by the materialized image contract.
    candidates = [stitched, np.rot90(stitched, 1), np.rot90(stitched, -1)]
    same_shape = [candidate for candidate in candidates if candidate.shape == current.shape]
    if len(same_shape) != 1:
        raise RuntimeError(
            f"Could not uniquely orient legacy mosaic {stitched.shape} to current {current.shape}."
        )
    legacy = same_shape[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = output_dir / f"{stem}__legacy_build01.png"
    diff_path = output_dir / f"{stem}__absdiff.png"
    skio.imsave(legacy_path, legacy, check_contrast=False)
    absdiff = np.abs(current.astype(np.int16) - legacy.astype(np.int16)).astype(np.uint8)
    skio.imsave(diff_path, absdiff, check_contrast=False)

    row: dict[str, object] = {"well": well, "current_path": str(current_path), "legacy_path": str(legacy_path)}
    row.update({f"current_{k}": v for k, v in _white_metrics(current).items()})
    row.update({f"legacy_{k}": v for k, v in _white_metrics(legacy).items()})
    row["mean_abs_diff"] = float(absdiff.mean())
    row["p99_abs_diff"] = float(np.percentile(absdiff, 99))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--wells", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Legacy sampling was random.  Fixing the seed makes this diagnostic repeatable
    # without changing the legacy algorithm or sampling distribution.
    np.random.seed(0)
    rows = [
        compare_well(args.experiment_root, args.experiment, well, args.output_dir, args.device)
        for well in args.wells
    ]
    summary_path = args.output_dir / "comparison_metrics.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
