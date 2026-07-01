"""Tests for materialized_image_readers.py.

Covers: resolve_* (rows), load_* (pixels), the grain-resolving validation inside the resolvers,
the wrapper functions, and the no-hand-parsing guard.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from data_pipeline.acquisition.image_materialization.materialized_image_readers import (
    load_materialized_image_from_row,
    load_materialized_images_from_rows,
    load_projection_image,
    load_z_stack_images_from_image_id,
    resolve_projection_row_from_image_id,
    resolve_z_stack_rows_from_image_id,
)

WELL_ID = "20250912_B01"
CHANNEL = "BF"
PROJECTION_PRODUCT_KEY = "BF__projection__focus_stack"
Z_STACK_PRODUCT_KEY = "BF__z_stack"


def _write_image(path: Path, fill: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((4, 4), fill, dtype="uint8")
    cv2.imwrite(str(path), img)


def _projection_row(tmp_path: Path, *, time_index: int = 7) -> dict:
    image_id = f"{WELL_ID}_{CHANNEL}_t{time_index:04d}"
    path = tmp_path / f"{image_id}.png"
    _write_image(path)
    return {
        "well_id": WELL_ID,
        "channel_id": CHANNEL,
        "time_index": time_index,
        "z_index": None,
        "image_id": image_id,
        "image_product_type": "projection",
        "projection_method": "focus_stack",
        "source_image_path": str(path),
    }


def _z_stack_row(tmp_path: Path, *, time_index: int = 7, z_index: int) -> dict:
    image_id = f"{WELL_ID}_{CHANNEL}_z{z_index:04d}_t{time_index:04d}"
    path = tmp_path / f"{image_id}.jpg"
    _write_image(path, fill=10 * z_index)
    return {
        "well_id": WELL_ID,
        "channel_id": CHANNEL,
        "time_index": time_index,
        "z_index": z_index,
        "image_id": image_id,
        "image_product_type": "z_stack",
        "projection_method": None,
        "source_image_path": str(path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# resolve_projection_row_from_image_id
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_projection_row_one_row(tmp_path):
    row = _projection_row(tmp_path)
    df = pd.DataFrame([row])
    resolved = resolve_projection_row_from_image_id(
        df, image_id=row["image_id"], product_key=PROJECTION_PRODUCT_KEY
    )
    assert resolved["source_image_path"] == row["source_image_path"]


def test_resolve_projection_row_zero_rows_fails_loud(tmp_path):
    df = pd.DataFrame([_projection_row(tmp_path, time_index=7)])
    with pytest.raises(ValueError, match="no row found"):
        resolve_projection_row_from_image_id(
            df,
            image_id=f"{WELL_ID}_{CHANNEL}_t0099",
            product_key=PROJECTION_PRODUCT_KEY,
        )


def test_resolve_projection_row_duplicate_rows_fails_loud(tmp_path):
    row = _projection_row(tmp_path)
    df = pd.DataFrame([row, row])
    with pytest.raises(ValueError, match="expected exactly one"):
        resolve_projection_row_from_image_id(
            df, image_id=row["image_id"], product_key=PROJECTION_PRODUCT_KEY
        )


def test_resolve_projection_row_z_plane_id_fails_loud(tmp_path):
    df = pd.DataFrame([_projection_row(tmp_path)])
    with pytest.raises(ValueError, match="z_index"):
        resolve_projection_row_from_image_id(
            df,
            image_id=f"{WELL_ID}_{CHANNEL}_z0003_t0007",
            product_key=PROJECTION_PRODUCT_KEY,
        )


def test_resolve_projection_row_z_stack_product_key_fails_loud(tmp_path):
    df = pd.DataFrame([_projection_row(tmp_path)])
    with pytest.raises(ValueError, match="not a projection product"):
        resolve_projection_row_from_image_id(
            df,
            image_id=f"{WELL_ID}_{CHANNEL}_t0007",
            product_key=Z_STACK_PRODUCT_KEY,
        )


# ─────────────────────────────────────────────────────────────────────────────
# resolve_z_stack_rows_from_image_id — auto-resolve
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_z_stack_rows_from_projection_id_auto_resolves_full_stack(tmp_path):
    """The normal case: caller holds a PROJECTION-grain image_id and wants the z-stack."""
    rows = [_z_stack_row(tmp_path, z_index=z) for z in range(3)]
    df = pd.DataFrame(rows)
    projection_id = f"{WELL_ID}_{CHANNEL}_t0007"  # z-less, same timepoint

    resolved = resolve_z_stack_rows_from_image_id(
        df, image_id=projection_id, product_key=Z_STACK_PRODUCT_KEY
    )
    assert resolved["z_index"].tolist() == [0, 1, 2]


def test_resolve_z_stack_rows_from_z_plane_id_resolves_same_stack(tmp_path):
    """A z-plane id's specific plane is discarded; the whole stack still comes back."""
    rows = [_z_stack_row(tmp_path, z_index=z) for z in range(3)]
    df = pd.DataFrame(rows)
    z_plane_id = f"{WELL_ID}_{CHANNEL}_z0001_t0007"

    resolved = resolve_z_stack_rows_from_image_id(
        df, image_id=z_plane_id, product_key=Z_STACK_PRODUCT_KEY
    )
    assert resolved["z_index"].tolist() == [0, 1, 2]


def test_resolve_z_stack_rows_duplicate_plane_fails_loud(tmp_path):
    rows = [_z_stack_row(tmp_path, z_index=0), _z_stack_row(tmp_path, z_index=0)]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="duplicate z_index"):
        resolve_z_stack_rows_from_image_id(
            df, image_id=f"{WELL_ID}_{CHANNEL}_t0007", product_key=Z_STACK_PRODUCT_KEY
        )


def test_resolve_z_stack_rows_projection_only_timepoint_fails_loud(tmp_path):
    df = pd.DataFrame([_projection_row(tmp_path)])
    with pytest.raises(ValueError, match="no z_stack rows found"):
        resolve_z_stack_rows_from_image_id(
            df, image_id=f"{WELL_ID}_{CHANNEL}_t0007", product_key=Z_STACK_PRODUCT_KEY
        )


def test_resolve_z_stack_rows_projection_product_key_fails_loud(tmp_path):
    rows = [_z_stack_row(tmp_path, z_index=0)]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="not a z_stack product"):
        resolve_z_stack_rows_from_image_id(
            df, image_id=f"{WELL_ID}_{CHANNEL}_t0007", product_key=PROJECTION_PRODUCT_KEY
        )


# ─────────────────────────────────────────────────────────────────────────────
# load_materialized_image_from_row / _from_rows
# ─────────────────────────────────────────────────────────────────────────────


def test_load_materialized_image_from_row_reads_recorded_path(tmp_path):
    row_dict = _projection_row(tmp_path)
    row = pd.Series(row_dict)
    image = load_materialized_image_from_row(row)
    assert image is not None
    assert image.shape[:2] == (4, 4)


def test_load_materialized_image_from_row_reads_recorded_path_not_reconstructed(tmp_path):
    """The row's source_image_path differs from any grammar-reconstructed path — the loader must
    still load the RECORDED path, never recompute one."""
    odd_path = tmp_path / "totally_nonstandard_name.png"
    _write_image(odd_path, fill=200)
    row = pd.Series({**_projection_row(tmp_path), "source_image_path": str(odd_path)})
    image = load_materialized_image_from_row(row)
    assert (image == 200).all()


def test_load_materialized_image_from_row_image_root_resolves_relative_path(tmp_path):
    image_id = f"{WELL_ID}_{CHANNEL}_t0007"
    rel_path = Path("nested") / f"{image_id}.png"
    _write_image(tmp_path / rel_path)
    row = pd.Series({**_projection_row(tmp_path), "source_image_path": str(rel_path)})
    image = load_materialized_image_from_row(row, image_root=tmp_path)
    assert image is not None


def test_load_materialized_image_from_row_reads_jpg(tmp_path):
    row = pd.Series(_z_stack_row(tmp_path, z_index=0))
    image = load_materialized_image_from_row(row)
    assert image is not None


def test_load_materialized_images_from_rows_preserves_order(tmp_path):
    rows = pd.DataFrame([_z_stack_row(tmp_path, z_index=z) for z in [2, 0, 1]])
    images = load_materialized_images_from_rows(rows)
    assert len(images) == 3
    # fill values were 10*z_index — confirm order matches input row order, not sorted
    assert [int(img.flat[0]) for img in images] == [20, 0, 10]


# ─────────────────────────────────────────────────────────────────────────────
# wrappers: resolve + load in one call
# ─────────────────────────────────────────────────────────────────────────────


def test_load_projection_image_wrapper(tmp_path):
    row = _projection_row(tmp_path)
    df = pd.DataFrame([row])
    image = load_projection_image(
        df, image_id=row["image_id"], product_key=PROJECTION_PRODUCT_KEY
    )
    assert image is not None


def test_load_z_stack_images_from_image_id_wrapper(tmp_path):
    rows = [_z_stack_row(tmp_path, z_index=z) for z in range(3)]
    df = pd.DataFrame(rows)
    images, resolved_rows = load_z_stack_images_from_image_id(
        df, image_id=f"{WELL_ID}_{CHANNEL}_t0007", product_key=Z_STACK_PRODUCT_KEY
    )
    assert len(images) == 3
    assert resolved_rows["z_index"].tolist() == [0, 1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# guard: module never hand-parses image_id
# ─────────────────────────────────────────────────────────────────────────────


def test_module_imports_parser_never_hand_parses():
    src = Path(__file__).parents[4] / "src" / "data_pipeline" / "acquisition" / "image_materialization" / "materialized_image_readers.py"
    text = src.read_text()
    assert "parse_image_id_with_z_index" in text
    assert "import re" not in text
    assert ".split(" not in text
