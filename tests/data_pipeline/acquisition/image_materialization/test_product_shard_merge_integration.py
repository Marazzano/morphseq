"""Multi-well, mixed-product merge seam for the product-shard frame_inventory chain.

The single-well product-shard chain (assemble one well, validate it) is covered in
``test_product_shard_assembly.py``. This file proves the seam single-well runs never exercised:
DIFFERENT wells with DIFFERENT active product sets flowing through

    product inventory shards (per well, per product)
        -> assemble_well_frame_inventory          (per well; image_materialization)
        -> merge_frame_inventory_shards           (experiment; metadata_ingest)
        -> validate_frame_inventory(scope="merged")  (grouped-by-well L0–L3)

Real experiments have uneven product availability — one well may carry BF projection + BF z_stack
while its neighbour carries projection only. The merged canonical inventory must hold both, keep
z_stack rows' ``z_index``, keep projection rows' ``z_index`` NA, and the grouped temporal/identity
checks must not be broken by the extra z_stack rows (they share a time_index with the projection,
so each well stays rectangular). No GPU, no real pixels: the merged validator runs check_sources=off.
"""

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.acquisition.image_materialization.product_shard_assembly import (
    assemble_well_frame_inventory,
    discover_product_shards_for_well,
)
from data_pipeline.acquisition.metadata_ingest.frame_inventory import (
    merge_frame_inventory_shards,
    validate_frame_inventory,
)

EXP = "20250912"
B01 = f"{EXP}_B01"
C01 = f"{EXP}_C01"
N_Z = 15  # z planes on the well that has a z_stack product


# ── fixtures: product-inventory shard rows (one row per materialized pixel file) ───────────────


def _projection_row(well_id: str, time_index: int) -> dict:
    well_index = well_id.split("_")[-1]
    return {
        "experiment_id": EXP,
        "well_index": well_index,
        "channel_id": "BF",
        "time_index": time_index,
        "elapsed_time_s": float(time_index * 120),
        "acquisition_time_s": float(time_index * 120),
        "z_index": pd.NA,
        "image_product_type": "projection",
        "projection_method": "focus_stack",
        "image_path": (
            f"built_image_data/{EXP}/materialized_images/{well_id}/projection/BF/"
            f"{well_id}_BF_t{time_index:04d}.png"
        ),
        "image_micrometers_per_pixel": 0.75,
        "image_width_px": 1024,
        "image_height_px": 768,
        "orientation": "none",
        "image_file_format": "png",
        "pixel_dtype": "uint8",
        "downsample_factor": 1,
        "downsample_method": "none",
        "jpeg_quality": pd.NA,
    }


def _z_stack_row(well_id: str, time_index: int, z_index: int) -> dict:
    row = _projection_row(well_id, time_index)
    row["z_index"] = z_index
    row["image_product_type"] = "z_stack"
    row["projection_method"] = pd.NA
    row["image_path"] = (
        f"built_image_data/{EXP}/materialized_images/{well_id}/z_stack/BF/"
        f"{well_id}_BF_z{z_index:04d}_t{time_index:04d}.png"
    )
    return row


def _write_validated_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    path.with_name(path.name + ".validated").write_text("validated\n", encoding="utf-8")


def _assemble_well(tmp_path: Path, well_id: str, shards: dict[str, list[dict]]) -> Path:
    """Write the well's product shards, discover them, assemble the canonical per-well inventory."""
    products_dir = tmp_path / "frame_inventory" / "product_inventories" / "per_well" / well_id
    for product_key, rows in shards.items():
        _write_validated_shard(
            products_dir / f"{well_id}_{product_key}_frame_inventory.csv", rows
        )
    manifest = tmp_path / f"{well_id}_available_products.csv"
    discover_product_shards_for_well(
        experiment_id=EXP,
        well_id=well_id,
        frame_inventory_products_dir=products_dir,
        output_csv=manifest,
    )
    canonical = tmp_path / f"{well_id}_frame_inventory.csv"
    assemble_well_frame_inventory(
        discovered_product_shards_csv=manifest,
        output_csv=canonical,
    )
    return canonical


# ── the seam: two wells, uneven product sets, merge + strict grouped validation ────────────────


def test_two_well_mixed_product_merge_validates(tmp_path):
    # B01: projection + z_stack (1 + N_Z planes, all at time_index 0). C01: projection only.
    b01 = _assemble_well(
        tmp_path,
        B01,
        {
            "BF__projection__focus_stack": [_projection_row(B01, 0)],
            "BF__z_stack": [_z_stack_row(B01, 0, z) for z in range(N_Z)],
        },
    )
    c01 = _assemble_well(
        tmp_path,
        C01,
        {"BF__projection__focus_stack": [_projection_row(C01, 0)]},
    )

    merged_csv = tmp_path / f"{EXP}_frame_inventory.csv"
    merged = merge_frame_inventory_shards(input_csvs=[b01, c01], output_csv=merged_csv)

    # Row accounting: B01 = 1 projection + N_Z z planes; C01 = 1 projection.
    b01_rows = merged[merged["well_index"] == "B01"]
    c01_rows = merged[merged["well_index"] == "C01"]
    assert len(b01_rows) == 1 + N_Z
    assert len(c01_rows) == 1
    assert len(merged) == 2 + N_Z

    # z_stack rows retain their plane index; projection rows are NA.
    z_rows = merged[merged["image_product_type"] == "z_stack"]
    proj_rows = merged[merged["image_product_type"] == "projection"]
    assert len(z_rows) == N_Z
    assert sorted(int(z) for z in z_rows["z_index"]) == list(range(N_Z))
    assert proj_rows["z_index"].isna().all()

    # The merged node validates STRICTLY (grouped by well), with z_stack rows present, no GPU/pixels.
    flag = tmp_path / f"{EXP}_frame_inventory.csv.validated"
    validate_frame_inventory(
        merged_csv, flag, check_sources=False, validation_scope="merged"
    )
    assert flag.read_text(encoding="utf-8") == "validated\n"


def test_merge_is_order_independent(tmp_path):
    # Swapping shard order must not change the merged content (merge sorts on the identity atoms).
    b01 = _assemble_well(
        tmp_path,
        B01,
        {
            "BF__projection__focus_stack": [_projection_row(B01, 0)],
            "BF__z_stack": [_z_stack_row(B01, 0, z) for z in range(N_Z)],
        },
    )
    c01 = _assemble_well(
        tmp_path,
        C01,
        {"BF__projection__focus_stack": [_projection_row(C01, 0)]},
    )

    forward = merge_frame_inventory_shards([b01, c01], tmp_path / "fwd.csv")
    reverse = merge_frame_inventory_shards([c01, b01], tmp_path / "rev.csv")
    pd.testing.assert_frame_equal(forward, reverse)


def test_merge_rejects_duplicate_z_plane_across_shards(tmp_path):
    # The same (well, channel, time, z) plane appearing in two shards is a duplicate frame — the
    # z-aware derived image_id must collide at merge scope (the NA-landmine fix holds across the
    # merge, not just per well). This is the guardrail that keeps a double-counted plane out of the
    # canonical experiment inventory.
    s1 = tmp_path / "s1.csv"
    s2 = tmp_path / "s2.csv"
    pd.DataFrame([_z_stack_row(B01, 0, 0)]).to_csv(s1, index=False)
    pd.DataFrame([_z_stack_row(B01, 0, 0)]).to_csv(s2, index=False)

    with pytest.raises(ValueError, match="(?i)duplicate"):
        merge_frame_inventory_shards([s1, s2], tmp_path / "m.csv")
