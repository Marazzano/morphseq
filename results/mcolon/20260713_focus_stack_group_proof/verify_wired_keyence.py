"""End-to-end check: the WIRED Keyence materializer reproduces the validated output.

Drives the real ``materialize_keyence_product_for_well`` on A02 / t0000 (candidate mode,
so nothing production is touched) using the on-disk acquisition inventory + stitch map, then
compares the written candidate projection PNG to the visually validated ``shared_clean``
reference stitch.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import skimage.io as skio

from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ResolvedImageProduct,
)
from data_pipeline.acquisition.image_materialization.scope.keyence.materialize_well_keyence import (
    materialize_keyence_product_for_well,
)

EXP = "20260702_hotchem_30hpf_plate01"
WELL = "A02"
ROOT = Path("pipeline/output/acquisition") / EXP
INV = ROOT / "ingest_metadata/acquisition_inventory__keyence.csv"
MASTER = ROOT / "ingest_metadata/keyence_stitch_map__keyence.json"
REF = Path(
    "results/mcolon/20260712_legacy_focus_stack_comparison/outputs/"
    f"{EXP}_{WELL}_BF_t0000__shared_clean.png"
)


def main() -> None:
    df = pd.read_csv(INV)
    well_id = f"{EXP}_{WELL}"
    well_df = df[(df["well_id"] == well_id) & (df["time_index"] == 0)].copy()
    print(f"A02 t0000 inventory rows: {len(well_df)} "
          f"tiles={sorted(well_df['tile_id'].unique())} "
          f"z={sorted(well_df['z_index'].unique())}")

    resolved = ResolvedImageProduct(
        channel_id="BF",
        image_product_type="projection",
        projection_method="focus_stack",
        xy_composition="mosaic",
    )

    with tempfile.TemporaryDirectory() as tmp:
        built_dir = Path(tmp) / "built_image_data"
        inv_out = materialize_keyence_product_for_well(
            experiment_id=EXP,
            well_id=well_id,
            well_index=WELL,
            well_acquisition_inventory_df=well_df,
            built_image_data_dir=built_dir,
            resolved_product=resolved,
            device="cpu",
            candidate=True,
            master_params_path=MASTER,
            config=None,
            input_root=None,
        )
        assert len(inv_out) == 1, inv_out
        row = inv_out.iloc[0]
        out_path = Path(row["image_path"])
        print(f"wired materializer wrote: {out_path.name} "
              f"({row['image_width_px']}x{row['image_height_px']}, "
              f"dtype={row['pixel_dtype']}, proj={row['projection_method']})")
        written = np.asarray(skio.imread(out_path))

    ref = np.asarray(skio.imread(REF))
    if written.shape != ref.shape:
        raise SystemExit(
            f"FAIL: shape mismatch wired={written.shape} ref={ref.shape}"
        )
    mae = float(np.abs(written.astype(np.int16) - ref.astype(np.int16)).mean())
    identical = bool(np.array_equal(written, ref))
    print(f"wired vs validated shared_clean reference: "
          f"shape={written.shape} identical={identical} MAE={mae:.4f}")
    if not identical:
        raise SystemExit("FAIL: wired Keyence output is not byte-identical to reference")
    print("PASS: wired Keyence materializer is byte-identical to validated reference.")


if __name__ == "__main__":
    main()
