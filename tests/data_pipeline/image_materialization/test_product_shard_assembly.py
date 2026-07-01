from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.image_materialization.product_shard_assembly import (
    assemble_well_frame_inventory,
    discover_product_shards_for_well,
    product_key_from_frame_inventory_product_filename,
)

EXP = "20250912"
WELL = "20250912_B01"


def _projection_row(time_index: int) -> dict:
    return {
        "experiment_id": EXP,
        "well_index": "B01",
        "channel_id": "BF",
        "time_index": time_index,
        "elapsed_time_s": float(time_index * 120),
        "acquisition_time_s": float(time_index * 120),
        "z_index": pd.NA,
        "image_product_type": "projection",
        "projection_method": "focus_stack",
        "source_image_path": (
            f"built_image_data/{EXP}/materialized_images/{WELL}/projection/BF/"
            f"{WELL}_BF_t{time_index:04d}.png"
        ),
        "source_micrometers_per_pixel": 0.75,
        "source_image_width_px": 1024,
        "source_image_height_px": 768,
        "image_width_px": 1024,
        "image_height_px": 768,
        "image_file_format": "png",
        "pixel_dtype": "uint8",
        "downsample_factor": 1,
        "downsample_method": "none",
        "jpeg_quality": pd.NA,
    }


def _z_stack_row(time_index: int, z_index: int) -> dict:
    row = _projection_row(time_index)
    row["z_index"] = z_index
    row["image_product_type"] = "z_stack"
    row["projection_method"] = pd.NA
    row["source_image_path"] = (
        f"built_image_data/{EXP}/materialized_images/{WELL}/z_stack/BF/"
        f"{WELL}_BF_z{z_index:04d}_t{time_index:04d}.png"
    )
    return row


def _write_inventory(path: Path, rows: list[dict], *, validated: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    if validated:
        path.with_name(path.name + ".validated").write_text("validated\n", encoding="utf-8")


def test_product_key_from_frame_inventory_product_filename():
    product_key = product_key_from_frame_inventory_product_filename(
        Path(f"{WELL}_BF__projection__focus_stack_frame_inventory.csv"),
        well_id=WELL,
    )

    assert product_key == "BF__projection__focus_stack"


def test_product_key_from_frame_inventory_product_filename_rejects_malformed():
    with pytest.raises(ValueError, match="Malformed"):
        product_key_from_frame_inventory_product_filename(
            Path(f"{WELL}_frame_inventory.csv"),
            well_id=WELL,
        )


def test_discover_product_shards_includes_only_validated_shards(tmp_path):
    products_dir = tmp_path / "frame_inventory_products" / "per_well" / WELL
    projection = products_dir / f"{WELL}_BF__projection__focus_stack_frame_inventory.csv"
    z_stack = products_dir / f"{WELL}_BF__z_stack_frame_inventory.csv"
    out = tmp_path / "discovered_product_shards.csv"
    _write_inventory(projection, [_projection_row(0)], validated=True)
    _write_inventory(z_stack, [_z_stack_row(0, 0)], validated=False)

    manifest = discover_product_shards_for_well(
        experiment_id=EXP,
        well_id=WELL,
        frame_inventory_products_dir=products_dir,
        output_csv=out,
    )

    assert out.exists()
    assert list(manifest["product_key"]) == ["BF__projection__focus_stack"]
    assert list(pd.read_csv(out)["product_key"]) == ["BF__projection__focus_stack"]


def test_discover_product_shards_rejects_validated_malformed_filename(tmp_path):
    products_dir = tmp_path / "frame_inventory_products" / "per_well" / WELL
    malformed = products_dir / f"{WELL}_frame_inventory.csv"
    _write_inventory(malformed, [_projection_row(0)], validated=True)

    with pytest.raises(ValueError, match="Malformed"):
        discover_product_shards_for_well(
            experiment_id=EXP,
            well_id=WELL,
            frame_inventory_products_dir=products_dir,
            output_csv=tmp_path / "discovered.csv",
        )


def test_assemble_well_frame_inventory_concatenates_and_sorts(tmp_path):
    products_dir = tmp_path / "frame_inventory_products" / "per_well" / WELL
    projection = products_dir / f"{WELL}_BF__projection__focus_stack_frame_inventory.csv"
    z_stack = products_dir / f"{WELL}_BF__z_stack_frame_inventory.csv"
    manifest_csv = tmp_path / "discovered_product_shards.csv"
    out = tmp_path / f"{WELL}_frame_inventory.csv"
    _write_inventory(projection, [_projection_row(1), _projection_row(0)], validated=True)
    _write_inventory(z_stack, [_z_stack_row(0, 1), _z_stack_row(0, 0)], validated=True)
    discover_product_shards_for_well(
        experiment_id=EXP,
        well_id=WELL,
        frame_inventory_products_dir=products_dir,
        output_csv=manifest_csv,
    )

    assembled = assemble_well_frame_inventory(
        discovered_product_shards_csv=manifest_csv,
        output_csv=out,
    )

    assert out.exists()
    assert list(assembled["image_product_type"]) == [
        "z_stack",
        "z_stack",
        "projection",
        "projection",
    ]
    assert list(assembled["time_index"]) == [0, 0, 0, 1]
    assert assembled["z_index"].iloc[0] == 0
    assert assembled["z_index"].iloc[1] == 1
    assert pd.isna(assembled["z_index"].iloc[2])


def test_assemble_well_frame_inventory_rejects_unvalidated_manifest_row(tmp_path):
    products_dir = tmp_path / "frame_inventory_products" / "per_well" / WELL
    projection = products_dir / f"{WELL}_BF__projection__focus_stack_frame_inventory.csv"
    _write_inventory(projection, [_projection_row(0)], validated=True)
    manifest = pd.DataFrame(
        [
            {
                "experiment_id": EXP,
                "well_id": WELL,
                "product_key": "BF__projection__focus_stack",
                "product_inventory_csv": str(projection),
                "product_inventory_validated": str(projection) + ".missing",
            }
        ]
    )
    manifest_csv = tmp_path / "discovered_product_shards.csv"
    manifest.to_csv(manifest_csv, index=False)

    with pytest.raises(ValueError, match="not validated"):
        assemble_well_frame_inventory(
            discovered_product_shards_csv=manifest_csv,
            output_csv=tmp_path / f"{WELL}_frame_inventory.csv",
        )
