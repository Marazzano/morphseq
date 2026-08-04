"""Drive channel_intensity over the pbx smoke wells, outside Snakemake.

The DAG wiring is a separate piece of work; this proves the MEASUREMENT path on real RFP pixels so
the dosage question can be asked today. Same kernels the rule will call.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

OUT = Path(".pbx_smoke/out")
EXP = "20260624_2x_td_bf_pbx_coll_plate01"
PRODUCT = "RFP__projection__max"

sys.path.insert(0, "src")

from data_pipeline.object_extraction.channel_intensity.entrypoint import run_channel_intensity  # noqa: E402

masks_root = OUT / "object_extraction" / EXP / "frame_masks" / "per_well"
inv_root = OUT / "acquisition" / EXP / "frame_inventory" / "per_well"
dest_root = OUT / "object_extraction" / EXP / "channel_intensity" / "per_well"

wells = sorted(p.name for p in masks_root.iterdir() if p.is_dir())
print(f"{len(wells)} wells: {wells}")

ok, failed = [], []
for well in wells:
    out_csv = dest_root / well / PRODUCT / "channel_intensity.csv"
    try:
        frame = run_channel_intensity(
            frame_masks_csv=masks_root / well / f"{well}_frame_masks.csv",
            frame_inventory_csv=inv_root / well / f"{well}_frame_inventory.csv",
            output_csv=out_csv,
            source_image_product_key=PRODUCT,
        )
        print(f"  {well}: {len(frame)} rows -> {out_csv}")
        ok.append(well)
    except Exception as exc:  # noqa: BLE001 - a smoke driver reports every well, then summarises
        print(f"  {well}: FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failed.append(well)

print(f"\nok={len(ok)} failed={len(failed)}")
sys.exit(1 if failed else 0)
