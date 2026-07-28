"""Prove focus_stack_group works end-to-end on a YX1 input and a Keyence (CANS) input.

YX1 path:     one raw (Z,Y,X) uint16 stack -> focus_stack_group -> single uint8 (Y,X) image
                (identity composition; no stitching).
Keyence path: all raw tile stacks for one (well,channel,time) frame -> focus_stack_group ->
                per-tile uint8 -> stitch_frame_tiles -> one stitched mosaic. Verified
                byte-identical to the visually validated shared_clean reference stitch.

Outputs land in this directory's ./outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nd2
import skimage.io as skio

from data_pipeline.acquisition.image_building.shared.focus_stack_group import (
    FocusStackConfig,
    focus_stack_group,
)
from data_pipeline.acquisition.image_building.scope.yx1.stitched_ff_builder import (
    _determine_bf_channel,
    _get_stack,
)
from data_pipeline.acquisition.image_building.utils.frame_tiler import (
    FrameTilingConfig,
    PreComputeStitchParams,
    TileSpec,
    stitch_frame_tiles,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def prove_yx1() -> None:
    print("\n=== YX1 proof (identity composition) ===")
    nd2_path = Path(
        "pipeline/input/raw_image_data/YX1/20240314/sox10GFP_multi_timelapse.nd2"
    )
    nd = nd2.ND2File(nd2_path)
    try:
        dask_arr = nd.to_dask()  # (T,W,Z,C,Y,X) or (T,W,Z,Y,X)
        channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]
        bf_idx = _determine_bf_channel(channel_names)
        if dask_arr.ndim == 6:
            dask_arr = dask_arr[:, :, :, bf_idx, :, :]
        # One well, one time: a single (Z,Y,X) uint16 stack.
        stack = _get_stack(dask_arr, t=0, w=0)
        print(f"  raw stack shape={stack.shape} dtype={stack.dtype}")
        assert stack.dtype == np.uint16, stack.dtype

        result = focus_stack_group([stack], config=FocusStackConfig(), device="cpu")
        assert len(result.tiles) == 1
        img = result.tiles[0].projection_u8
        fim = result.tiles[0].focus_index_map
        print(f"  bounds lo={result.intensity_lo} hi={result.intensity_hi}")
        print(f"  projection shape={img.shape} dtype={img.dtype} "
              f"(min={img.min()} max={img.max()} mean={img.mean():.1f})")
        print(f"  focus_index_map shape={fim.shape} dtype={fim.dtype} "
              f"range=[{fim.min()},{fim.max()}] of Z={stack.shape[0]}")
        assert img.shape == stack.shape[1:] and img.dtype == np.uint8
        assert fim.shape == stack.shape[1:] and fim.dtype == np.int32
        assert 0 <= fim.min() and fim.max() < stack.shape[0]
        skio.imsave(OUT / "yx1_A01_t0000_focus.png", img, check_contrast=False)
        print("  PASS: single focus-stacked uint8 image written "
              "(yx1_A01_t0000_focus.png)")
    finally:
        nd.close()


def prove_keyence() -> None:
    print("\n=== Keyence (CANS) proof (mosaic composition) ===")
    exp = "20260702_hotchem_30hpf_plate01"
    well = "A02"
    root = Path("pipeline/output/acquisition") / exp
    focus_dir = (
        root / "materialized_images" / f"{exp}_{well}" / "BF/projection/focus_stack"
    )
    manifest_path = focus_dir / "raw_tile_manifest" / f"{exp}_{well}_BF_t0000.json"
    master = root / "ingest_metadata/keyence_stitch_map__keyence.json"
    ref_stitch = Path(
        "results/mcolon/20260712_legacy_focus_stack_comparison/outputs/"
        f"{exp}_{well}_BF_t0000__shared_clean.png"
    )

    manifest = json.loads(manifest_path.read_text())
    tile_ids = [str(t["tile_id"]) for t in manifest["tiles"]]
    tiles = [
        np.stack([skio.imread(p) for p in t["source_tiff_paths"]], axis=0)
        for t in manifest["tiles"]
    ]
    print(f"  {len(tiles)} tiles, each shape={tiles[0].shape} dtype={tiles[0].dtype}")

    # ONE group operation across ALL tiles -> shared bounds + per-tile uint8 projections.
    result = focus_stack_group(tiles, config=FocusStackConfig(), device="cpu")
    print(f"  shared bounds lo={result.intensity_lo} hi={result.intensity_hi}")
    assert len(result.tiles) == len(tiles)

    # Stitch the shared-bound per-tile projections into one mosaic.
    specs = [
        TileSpec(tile_id=tid, image=result.tiles[i].projection_u8)
        for i, tid in enumerate(tile_ids)
    ]
    stitched = stitch_frame_tiles(
        specs,
        FrameTilingConfig(orientation="horizontal"),
        PreComputeStitchParams(master_params_path=master),
    ).stitched
    print(f"  stitched mosaic shape={stitched.shape} dtype={stitched.dtype} "
          f"(min={stitched.min()} max={stitched.max()} mean={stitched.mean():.1f})")
    skio.imsave(OUT / f"keyence_{well}_t0000_stitched.png", stitched, check_contrast=False)

    # Prove equivalence to the visually validated reference stitch.
    if ref_stitch.exists():
        ref = np.asarray(skio.imread(ref_stitch))
        identical = ref.shape == stitched.shape and np.array_equal(ref, stitched)
        mae = float(np.abs(ref.astype(np.int16) - stitched.astype(np.int16)).mean()) \
            if ref.shape == stitched.shape else float("nan")
        print(f"  vs validated shared_clean reference: "
              f"identical={identical} MAE={mae:.4f} (ref shape={ref.shape})")
        assert identical, "stitched mosaic must match the validated shared_clean reference"
        print("  PASS: stitched mosaic byte-identical to validated reference "
              f"(keyence_{well}_t0000_stitched.png)")
    else:
        print(f"  (reference {ref_stitch} not found — wrote mosaic without byte check)")


if __name__ == "__main__":
    prove_keyence()
    prove_yx1()
    print("\nAll proofs passed.")
