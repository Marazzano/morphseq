"""Render confidence-labeled contact sheets from existing SAM2 mask snips."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


HERE = Path(__file__).parent.resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument(
    "--output",
    type=Path,
    help="Output directory (default: mask_contact_sheets beside manifest).",
)
args = parser.parse_args()

manifest_path = args.manifest.expanduser().resolve()
output_dir = (
    args.output.expanduser().resolve()
    if args.output
    else manifest_path.parent / "mask_contact_sheets"
)
try:
    output_dir.relative_to(HERE)
except ValueError as exc:
    raise ValueError(f"Refusing to write outside {HERE}: {output_dir}") from exc
output_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(manifest_path)
for (_, stem), group in manifest.groupby(["experiment_id", "stem"], sort=False):
    group = group.sort_values("embryo_position")
    crops = []
    for _, row in group.iterrows():
        crop = Image.open(row["mask_snip_path"]).convert("RGB")
        ImageDraw.Draw(crop).text(
            (6, 4),
            f"{int(row['embryo_position'])}: SAM {row['mask_score']:.3f}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        crops.append(crop)

    cell_width = max(crop.width for crop in crops)
    cell_height = max(crop.height for crop in crops)
    sheet = Image.new(
        "RGB", (cell_width * 4 + 10, cell_height * 2 + 6), (255, 255, 255)
    )
    for index, crop in enumerate(crops):
        row_index, column_index = divmod(index, 4)
        sheet.paste(
            crop,
            (column_index * (cell_width + 2), row_index * (cell_height + 2)),
        )
    experiment_id = group["experiment_id"].iloc[0]
    sheet.save(output_dir / f"{experiment_id}_{stem}.jpg", quality=88)

print(
    f"Rendered {manifest['image_id'].nunique()} confidence-labeled contact sheets "
    f"-> {output_dir}"
)
