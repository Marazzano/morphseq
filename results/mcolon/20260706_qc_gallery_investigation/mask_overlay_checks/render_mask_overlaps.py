from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

from data_pipeline.object_extraction.segmentation.masks.mask_rle import decode_binary_mask_rle


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent

CASES = [
    {
        "label": "20250912_H01_e01_BF_t0003",
        "well": "20250912_H01",
        "image_id": "20250912_H01_BF_t0003",
        "target_snip_id": "20250912_H01_e01_BF_t0003",
    },
    {
        "label": "20250912_D04_e01_BF_t0001",
        "well": "20250912_D04",
        "image_id": "20250912_D04_BF_t0001",
        "target_snip_id": "20250912_D04_e01_BF_t0001",
    },
]

COLORS = np.array(
    [
        [0, 180, 255],
        [255, 170, 0],
        [0, 220, 120],
        [230, 80, 255],
        [255, 80, 80],
        [130, 210, 255],
        [255, 230, 80],
        [130, 120, 255],
    ],
    dtype=np.uint8,
)


def main() -> None:
    all_summary: list[dict[str, object]] = []
    all_pairs: list[dict[str, object]] = []
    for case in CASES:
        summary_rows, pair_rows = render_case(case)
        all_summary.extend(summary_rows)
        all_pairs.extend(pair_rows)

    pd.DataFrame(all_summary).to_csv(OUT_DIR / "mask_overlay_summary.csv", index=False)
    pd.DataFrame(all_pairs).to_csv(OUT_DIR / "mask_overlap_pairs.csv", index=False)


def render_case(case: dict[str, str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    well = case["well"]
    image_id = case["image_id"]
    target_snip_id = case["target_snip_id"]
    label = case["label"]

    snips = pd.read_csv(
        ROOT / f"data_pipeline_output/object_extraction/20250912/snips/per_well/{well}/{well}_snip_inventory.csv"
    )
    masks_df = pd.read_csv(
        ROOT / f"data_pipeline_output/object_extraction/20250912/frame_masks/per_well/{well}/{well}_frame_masks.csv"
    )

    frame_snips = snips.loc[snips["image_id"].eq(image_id)].copy()
    frame_masks = masks_df.loc[masks_df["image_id"].eq(image_id)].copy()
    if frame_snips.empty:
        raise ValueError(f"No snip rows for {image_id}")
    if frame_masks.empty:
        raise ValueError(f"No mask rows for {image_id}")

    snip_by_mask = frame_snips.set_index("mask_id", drop=False)
    image_path = Path(frame_snips["source_image_path"].iloc[0])
    image = np.asarray(Image.open(image_path).convert("L"))
    base_rgb = _normalize_to_rgb(image)

    decoded: list[dict[str, object]] = []
    for _, row in frame_masks.sort_values("mask_id").iterrows():
        mask = decode_binary_mask_rle(json.loads(row["mask_rle"]))
        if mask.shape != image.shape:
            raise ValueError(f"{image_id}: mask shape {mask.shape} != image shape {image.shape}")
        snip_row = snip_by_mask.loc[row["mask_id"]] if row["mask_id"] in snip_by_mask.index else None
        decoded.append(
            {
                "mask_id": row["mask_id"],
                "track_id": row["track_id"],
                "snip_id": "" if snip_row is None else snip_row["snip_id"],
                "embryo_id": "" if snip_row is None else snip_row["embryo_id"],
                "physical_embryo_id": "" if snip_row is None else snip_row["physical_embryo_id"],
                "mask": mask,
                "area_px": int(mask.sum()),
                "is_target": bool(snip_row is not None and snip_row["snip_id"] == target_snip_id),
            }
        )

    stack = np.stack([d["mask"] for d in decoded], axis=0)
    overlap_count = stack.sum(axis=0)
    union = overlap_count > 0
    overlap = overlap_count >= 2

    overlay = _draw_overlay(base_rgb, decoded, title=f"{label}: all masks on BF projection")
    overlap_img = _draw_overlap(base_rgb, union, overlap, title=f"{label}: overlap map")

    overlay.save(OUT_DIR / f"{label}_all_masks_overlay.png")
    overlap_img.save(OUT_DIR / f"{label}_overlap_map.png")

    crop_box = _crop_box(union, margin=90, shape=image.shape)
    overlay.crop(crop_box).save(OUT_DIR / f"{label}_crop_all_masks_overlay.png")
    overlap_img.crop(crop_box).save(OUT_DIR / f"{label}_crop_overlap_map.png")

    summary_rows = []
    for d in decoded:
        m = d["mask"]
        overlap_px = int((m & overlap).sum())
        summary_rows.append(
            {
                "case": label,
                "image_id": image_id,
                "target_snip_id": target_snip_id,
                "snip_id": d["snip_id"],
                "embryo_id": d["embryo_id"],
                "physical_embryo_id": d["physical_embryo_id"],
                "mask_id": d["mask_id"],
                "track_id": d["track_id"],
                "is_target": d["is_target"],
                "area_px": d["area_px"],
                "overlap_px": overlap_px,
                "overlap_frac_of_mask": overlap_px / max(int(d["area_px"]), 1),
            }
        )

    pair_rows = []
    for i, a in enumerate(decoded):
        for b in decoded[i + 1 :]:
            inter = int((a["mask"] & b["mask"]).sum())
            if inter == 0:
                continue
            denom = min(int(a["area_px"]), int(b["area_px"]))
            pair_rows.append(
                {
                    "case": label,
                    "image_id": image_id,
                    "snip_id_a": a["snip_id"],
                    "snip_id_b": b["snip_id"],
                    "mask_id_a": a["mask_id"],
                    "mask_id_b": b["mask_id"],
                    "intersection_px": inter,
                    "intersection_frac_of_smaller": inter / max(denom, 1),
                }
            )

    return summary_rows, pair_rows


def _normalize_to_rgb(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99.7])
    if hi <= lo:
        scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    return np.repeat(scaled[:, :, None], 3, axis=2)


def _draw_overlay(base_rgb: np.ndarray, decoded: list[dict[str, object]], *, title: str) -> Image.Image:
    out = base_rgb.astype(np.float32)
    for idx, d in enumerate(decoded):
        color = COLORS[idx % len(COLORS)].astype(np.float32)
        mask = d["mask"]
        alpha = 0.28 if not d["is_target"] else 0.42
        out[mask] = (1 - alpha) * out[mask] + alpha * color
        edge = binary_dilation(mask) ^ binary_erosion(mask)
        out[edge] = color
        if d["is_target"]:
            thick = binary_dilation(edge, iterations=2)
            out[thick] = np.array([255, 255, 255], dtype=np.float32)
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    _annotate(image, title, decoded)
    return image


def _draw_overlap(base_rgb: np.ndarray, union: np.ndarray, overlap: np.ndarray, *, title: str) -> Image.Image:
    out = base_rgb.astype(np.float32)
    out[union] = 0.75 * out[union] + 0.25 * np.array([0, 160, 255], dtype=np.float32)
    out[overlap] = 0.25 * out[overlap] + 0.75 * np.array([255, 0, 255], dtype=np.float32)
    edge = binary_dilation(overlap) ^ binary_erosion(overlap)
    out[edge] = np.array([255, 255, 255], dtype=np.float32)
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    _annotate(image, title, [])
    return image


def _annotate(image: Image.Image, title: str, decoded: list[dict[str, object]]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    pad = 8
    lines = [title]
    for idx, d in enumerate(decoded):
        prefix = "*" if d["is_target"] else " "
        lines.append(f"{prefix} {idx}: {d['snip_id']} {d['mask_id']}")
    text_h = 13 * len(lines) + 2 * pad
    draw.rectangle((0, 0, image.width, text_h), fill=(0, 0, 0))
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255), font=font)
        y += 13


def _crop_box(mask: np.ndarray, *, margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, shape[1], shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, shape[1])
    y1 = min(int(ys.max()) + margin + 1, shape[0])
    return (x0, y0, x1, y1)


if __name__ == "__main__":
    main()
