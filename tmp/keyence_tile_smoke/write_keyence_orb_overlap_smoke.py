from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import skimage.io as skio


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "tmp/keyence_tile_smoke"


def _load_tiles() -> list[np.ndarray]:
    return [
        skio.imread(OUT_DIR / f"A01_t0000_tile{idx}_maxproj.png")[::4, ::4]
        for idx in (1, 2, 3)
    ]


def _estimate_offsets(tiles: list[np.ndarray]) -> list[tuple[float, float]]:
    orb = cv2.ORB_create(nfeatures=3000, edgeThreshold=5, fastThreshold=5)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    coords = [(0.0, 0.0)]  # y, x
    for prev, curr in zip(tiles[:-1], tiles[1:]):
        kp_prev, desc_prev = orb.detectAndCompute(prev, None)
        kp_curr, desc_curr = orb.detectAndCompute(curr, None)
        matches = matcher.match(desc_prev, desc_curr)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])
        matrix, _inliers = cv2.estimateAffinePartial2D(
            pts_curr,
            pts_prev,
            method=cv2.RANSAC,
            ransacReprojThreshold=5,
            maxIters=5000,
        )
        if matrix is None:
            deltas = pts_prev - pts_curr
            dx, dy = np.median(deltas, axis=0)
        else:
            dx = float(matrix[0, 2])
            dy = float(matrix[1, 2])
        prev_y, prev_x = coords[-1]
        coords.append((prev_y + float(dy), prev_x + float(dx)))
    return coords


def _paint(tiles: list[np.ndarray], coords: list[tuple[float, float]]) -> np.ndarray:
    min_y = min(y for y, _x in coords)
    min_x = min(x for _y, x in coords)
    norm = [(y - min_y, x - min_x) for y, x in coords]
    tile_h, tile_w = tiles[0].shape[:2]
    out_h = int(np.ceil(max(y + tile_h for y, _x in norm)))
    out_w = int(np.ceil(max(x + tile_w for _y, x in norm)))
    canvas = np.zeros((out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)
    for tile, (y, x) in zip(tiles, norm):
        yy = int(round(y))
        xx = int(round(x))
        h, w = tile.shape[:2]
        canvas[yy : yy + h, xx : xx + w] += tile.astype(np.float32)
        weight[yy : yy + h, xx : xx + w] += 1.0
    canvas = canvas / np.maximum(weight, 1.0)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main() -> None:
    tiles = _load_tiles()
    coords = _estimate_offsets(tiles)
    mosaic = _paint(tiles, coords)
    out_path = OUT_DIR / "A01_t0000_stitched_horizontal_orb_ds4.png"
    skio.imsave(str(out_path), mosaic, check_contrast=False)
    qc_path = OUT_DIR / "A01_t0000_stitched_horizontal_orb_ds4.json"
    qc_path.write_text(
        json.dumps(
            {
                "stitched_path": str(out_path.relative_to(REPO)),
                "coords_yx": coords,
                "note": "Diagnostic ORB overlap mosaic, not production stitch2d output.",
            },
            indent=2,
        )
        + "\n"
    )
    print(out_path.relative_to(REPO))


if __name__ == "__main__":
    main()
