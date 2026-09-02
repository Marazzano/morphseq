"""Full-resolution per-pair side-by-sides: legacy | regenerated | |difference|.

The contact sheets in ``snip_pair_panels.py`` pack many pairs into one figure, which downsamples
each snip well below its native 576x256 and makes fine tissue detail unassessable. This writes one
file per pair at exact 1:1 pixels (optionally integer-upscaled), so what you see is what was
rendered -- no resampling anywhere in the path.

Files are named by difficulty rank, so directory sort order is difference order:
``rank007_D02_mean13.36.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio
from matplotlib import colormaps
from PIL import Image, ImageDraw

GUTTER_PX = 8
HEADER_PX = 22


def load_gray(path: Path) -> np.ndarray:
    image = skio.imread(str(path))
    return image[..., 0] if image.ndim == 3 else image


def composite(legacy: np.ndarray, regenerated: np.ndarray, diff: np.ndarray,
              label: str, scale: int) -> Image.Image:
    height, width = legacy.shape
    # Diff gets a perceptual colormap, scaled to ITS OWN max so faint structure stays visible;
    # the max is printed in the header so panels are never compared without their scale.
    ceiling = max(1, int(diff.max()))
    coloured = (colormaps["magma"](diff / ceiling)[..., :3] * 255).astype(np.uint8)

    panels = [
        np.repeat(legacy[:, :, None], 3, axis=2),
        np.repeat(regenerated[:, :, None], 3, axis=2),
        coloured,
    ]
    total_width = width * 3 + GUTTER_PX * 2
    canvas = np.zeros((height, total_width, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        x = i * (width + GUTTER_PX)
        canvas[:, x:x + width] = panel

    image = Image.fromarray(canvas)
    if scale > 1:
        # NEAREST only: any smoothing would invent detail the renderer did not produce.
        image = image.resize((total_width * scale, height * scale), Image.NEAREST)

    out = Image.new("RGB", (image.width, image.height + HEADER_PX), (0, 0, 0))
    out.paste(image, (0, HEADER_PX))
    draw = ImageDraw.Draw(out)
    draw.text((4, 5), f"{label}   |   diff colour scale 0-{ceiling}", fill=(235, 235, 235))
    for i, name in enumerate(("legacy", "regenerated", "|difference|")):
        draw.text((4 + i * (image.width // 3), 5 + 0), "", fill=(235, 235, 235))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--snips-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2,
                        help="Integer NEAREST upscale for viewing. 1 = exact native pixels.")
    parser.add_argument("--include-flipped", action="store_true",
                        help="Also emit flipped pairs (shown de-rotated).")
    parser.add_argument("--median-band", type=int, default=None,
                        help="Emit only the N pairs straddling the median difference. Uses the same "
                             "selection as snip_pair_panels.py, so the files match that sheet.")
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    comparable = results[results["comparable"]] if "comparable" in results else results
    selected = comparable if args.include_flipped else comparable[~comparable["orientation_flipped"]]
    selected = selected.sort_values("mean_abs_diff").reset_index(drop=True)

    rank_offset = 0
    if args.median_band:
        midpoint = len(selected) // 2
        lo = max(0, midpoint - args.median_band // 2)
        hi = midpoint + (args.median_band + 1) // 2
        rank_offset = lo
        selected = selected.iloc[lo:hi].reset_index(drop=True)
        print(f"median band: ranks {lo + 1}-{hi} of {midpoint * 2 if midpoint else 0}+ pairs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for rank, row in enumerate(selected.itertuples(), rank_offset + 1):
        legacy_path = next(
            (args.legacy_dir / f"{row.snip_id}{ext}"
             for ext in (".jpg", ".jpeg", ".png")
             if (args.legacy_dir / f"{row.snip_id}{ext}").exists()), None)
        regenerated_path = next(args.snips_root.rglob(f"{row.regenerated_snip_id}.png"), None)
        if legacy_path is None or regenerated_path is None:
            print(f"  rank{rank:03d} {row.snip_id}: unresolved, skipped")
            skipped += 1
            continue

        legacy = load_gray(legacy_path)
        regenerated = load_gray(regenerated_path)
        if bool(row.orientation_flipped):
            regenerated = np.rot90(regenerated, 2)
        diff = np.abs(legacy.astype(np.int16) - regenerated.astype(np.int16)).astype(np.uint16)

        well = row.snip_id.split("_")[-3]
        label = f"{row.snip_id}  mean {row.mean_abs_diff:.2f}  max {int(row.max_abs_diff)}"
        out_path = args.out_dir / f"rank{rank:03d}_{well}_mean{row.mean_abs_diff:05.2f}.png"
        composite(legacy, regenerated, diff, label, args.scale).save(out_path)
        written += 1

    print(f"wrote {written} full-res pairs to {args.out_dir}  (skipped {skipped})")
    if written:
        mid = written // 2
        print(f"median-difficulty pairs are around rank{mid:03d}; sort order == difference order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
