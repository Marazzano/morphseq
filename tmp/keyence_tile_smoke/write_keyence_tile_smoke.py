from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import skimage.io as skio

from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FallbackParams,
    FrameTilingConfig,
    TileSpec,
    stitch_frame_tiles,
)
from data_pipeline.acquisition.metadata_ingest.scope.keyence.raw_plane_parsing import (
    _infer_keyence_stack_lookup,
)


REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data_pipeline_output/inputs/raw_image_data/Keyence/20260702_hotchem_36hpf_plate02"
OUT_DIR = REPO / "tmp/keyence_tile_smoke"
STITCH_DOWNSAMPLE = 4


def _project_stack(paths: list[Path]) -> np.ndarray:
    stack = np.stack([skio.imread(str(path)) for path in paths], axis=0)
    projected = np.max(stack, axis=0)
    if projected.dtype != np.uint8:
        im = projected.astype(np.float32)
        lo = float(np.percentile(im, 0.1))
        hi = float(np.percentile(im, 99.9))
        if hi <= lo:
            hi = lo + 1.0
        projected = np.clip((im - lo) / (hi - lo), 0.0, 1.0)
        projected = (projected * 255.0).astype(np.uint8)
    return projected


def _tile_specs_for_key(
    lookup: dict[tuple[str, int], dict[int, list[Path]]],
    key: tuple[str, int],
    cache: dict[tuple[str, int], list[TileSpec]],
) -> list[TileSpec]:
    if key in cache:
        return cache[key]
    tile_stacks = lookup[key]
    tile_specs = [
        TileSpec(tile_id=str(tile_id), image=_project_stack(tile_stacks[tile_id]))
        for tile_id in sorted(tile_stacks)
    ]
    cache[key] = tile_specs
    return tile_specs


def _build_master_params(
    tile_specs: list[TileSpec],
    orientation: str,
    out_path: Path,
) -> dict:
    tile_h, tile_w = tile_specs[0].image.shape[:2]
    n_tiles = len(tile_specs)
    med = np.zeros((n_tiles, 2), dtype=float)
    for idx in range(n_tiles):
        if orientation == "vertical":
            med[idx, :] = [idx * tile_h, 0.0]
        else:
            med[idx, :] = [0.0, idx * tile_w]
    coords_out = {
        str(idx): [float(med[idx, 0]), float(med[idx, 1])]
        for idx in range(n_tiles)
    }
    shape = [n_tiles, 1] if orientation == "vertical" else [1, n_tiles]
    metadata = {
        "shape": shape,
        "size": n_tiles,
        "tile_shape": [int(tile_h), int(tile_w)],
    }
    out_path.write_text(json.dumps({"metadata": metadata, "coords": coords_out}, indent=2) + "\n")
    return {
        "path": str(out_path.relative_to(REPO)),
        "source": "synthetic_raster_geometry",
        "tile_shape_yx": [int(tile_h), int(tile_w)],
        "metadata": metadata,
        "coords": coords_out,
    }


def main() -> None:
    lookup = _infer_keyence_stack_lookup(RAW_DIR)
    key = ("A01", 0)
    if key not in lookup:
        key = sorted(k for k, tiles in lookup.items() if len(tiles) >= 3)[0]

    cache: dict[tuple[str, int], list[TileSpec]] = {}
    tile_specs = _tile_specs_for_key(lookup, key, cache)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tile_paths = {}
    for spec in tile_specs:
        tile_path = OUT_DIR / f"{key[0]}_t{key[1]:04d}_tile{spec.tile_id}_maxproj.png"
        skio.imsave(str(tile_path), spec.image, check_contrast=False)
        tile_paths[spec.tile_id] = str(tile_path.relative_to(REPO))

    stitch_tile_specs = [
        TileSpec(tile_id=spec.tile_id, image=spec.image[::STITCH_DOWNSAMPLE, ::STITCH_DOWNSAMPLE])
        for spec in tile_specs
    ]

    qc = {
        "raw_dir": str(RAW_DIR.relative_to(REPO)),
        "well_index": key[0],
        "time_index": key[1],
        "tile_count": len(tile_specs),
        "tile_paths": tile_paths,
        "orientations": {},
    }

    for orientation in ("vertical", "horizontal"):
        master_path = OUT_DIR / f"{key[0]}_t{key[1]:04d}_master_params_{orientation}.json"
        master_info = _build_master_params(
            tile_specs=stitch_tile_specs,
            orientation=orientation,
            out_path=master_path,
        )
        result = stitch_frame_tiles(
            stitch_tile_specs,
            FrameTilingConfig(orientation=orientation, mode="prior_only"),
            FallbackParams(master_params_path=master_path),
        )
        out_path = OUT_DIR / f"{key[0]}_t{key[1]:04d}_stitched_{orientation}_ds{STITCH_DOWNSAMPLE}.png"
        skio.imsave(str(out_path), result.stitched, check_contrast=False)
        qc["orientations"][orientation] = {
            "master_params": master_info,
            "stitched_path": str(out_path.relative_to(REPO)),
            "stitch_downsample": STITCH_DOWNSAMPLE,
            "shape_yx": list(result.stitched.shape[:2]),
            "fallback_used": result.fallback_used,
            "qc_passed": result.qc.passed,
            "qc_reasons": list(result.qc.reasons),
            "qc_metrics": result.qc.metrics,
            "tile_transforms": {
                tile_id: {
                    "dx_px": tr.dx_px,
                    "dy_px": tr.dy_px,
                    "source": tr.source,
                }
                for tile_id, tr in result.tile_transforms.items()
            },
        }

    qc_path = OUT_DIR / f"{key[0]}_t{key[1]:04d}_stitch_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")
    print(json.dumps({"qc_path": str(qc_path.relative_to(REPO)), **qc}, indent=2))


if __name__ == "__main__":
    main()
