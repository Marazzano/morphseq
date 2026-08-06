"""Compute embryo-specific QC metrics for a SAM2 mask manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


HERE = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument(
    "--output",
    type=Path,
    help="Output CSV (default: sam2_mask_qc.csv beside the input manifest).",
)
args = parser.parse_args()

manifest_path = args.manifest.expanduser().resolve()
output_path = (
    args.output.expanduser().resolve()
    if args.output
    else manifest_path.with_name("sam2_mask_qc.csv")
)
try:
    output_path.relative_to(HERE)
except ValueError as exc:
    raise ValueError(f"Refusing to write outside {HERE}: {output_path}") from exc

manifest = pd.read_csv(manifest_path)
qc_rows: list[dict] = []
structure = np.ones((3, 3), dtype=np.uint8)

for image_id, group in manifest.groupby("image_id", sort=False):
    group = group.sort_values("embryo_position")
    masks = [
        np.asarray(Image.open(path).convert("L")) > 0
        for path in group["mask_path"]
    ]
    areas = np.asarray([int(mask.sum()) for mask in masks])

    for local_index, (_, row) in enumerate(group.iterrows()):
        mask = masks[local_index]
        area = int(areas[local_index])
        labels, component_count_all = ndimage.label(mask, structure=structure)
        component_areas = np.bincount(labels.ravel())[1:]
        component_areas = np.sort(component_areas)[::-1]
        largest_component_area = (
            int(component_areas[0]) if component_areas.size else 0
        )
        component_count_ge_100px = int((component_areas >= 100).sum())

        x1 = max(0, int(row["prompt_x1_px"]))
        y1 = max(0, int(row["prompt_y1_px"]))
        x2 = min(mask.shape[1], int(row["prompt_x2_px"]))
        y2 = min(mask.shape[0], int(row["prompt_y2_px"]))
        inside_prompt_area = int(mask[y1:y2, x1:x2].sum())

        overlap_metrics = []
        for other_index, other_mask in enumerate(masks):
            if other_index == local_index:
                continue
            intersection = int(np.logical_and(mask, other_mask).sum())
            union = int(np.logical_or(mask, other_mask).sum())
            overlap_metrics.append(
                (
                    intersection,
                    intersection / union if union else 0.0,
                    intersection / area if area else 0.0,
                    int(group.iloc[other_index]["embryo_position"]),
                )
            )
        max_overlap = max(overlap_metrics, default=(0, 0.0, 0.0, -1))

        qc_rows.append(
            {
                **row.to_dict(),
                "component_count_all": int(component_count_all),
                "component_count_ge_100px": component_count_ge_100px,
                "largest_component_area_px": largest_component_area,
                "largest_component_fraction": (
                    largest_component_area / area if area else 0.0
                ),
                "mask_inside_prompt_fraction": (
                    inside_prompt_area / area if area else 0.0
                ),
                "touches_fov_border": bool(
                    mask[0, :].any()
                    or mask[-1, :].any()
                    or mask[:, 0].any()
                    or mask[:, -1].any()
                ),
                "max_overlap_other_position": max_overlap[3],
                "max_pair_overlap_px": max_overlap[0],
                "max_pair_iou": max_overlap[1],
                "max_overlap_fraction_of_this_mask": max_overlap[2],
            }
        )

qc = pd.DataFrame(qc_rows)
qc.to_csv(output_path, index=False)
print(
    f"Wrote {len(qc)} embryo-level QC rows across "
    f"{qc['image_id'].nunique()} FOVs -> {output_path}"
)
