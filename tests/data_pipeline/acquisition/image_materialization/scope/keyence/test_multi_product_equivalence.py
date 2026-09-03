"""Equivalence + sharing tests for the Keyence multi-product fanout.

The refactor's promise is that materializing N products in ONE process is byte-for-byte identical
to N separate single-product runs, while doing the shared work once. These tests assert both halves
against the REAL focus/stitch primitives — no mocking of the image math — because a mocked stitch
cannot prove pixels are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_pipeline.acquisition.image_materialization.materialization_plan import (
    ResolvedImageProduct,
)
from data_pipeline.acquisition.image_materialization.resolved_product_plans import (
    image_product_key_for_resolved_product,
)
from data_pipeline.acquisition.image_materialization.scope.keyence import (
    materialize_well_keyence as mwk,
)
from data_pipeline.acquisition.image_materialization.scope.keyence import frame_materials as fm
from data_pipeline.shared.identifiers.constructors import build_well_id

EXPERIMENT_ID = "20250912"
WELL_INDEX = "B01"
WELL_ID = build_well_id(EXPERIMENT_ID, WELL_INDEX)

PROJECTION = ResolvedImageProduct(
    channel_id="BF",
    image_product_type="projection",
    projection_method="focus_stack",
    xy_composition="mosaic",
)
Z_STACK = ResolvedImageProduct(
    channel_id="BF",
    image_product_type="z_stack",
    projection_method=None,
    xy_composition="mosaic",
)
# A single tile keeps the real stitcher deterministic (it short-circuits to identity) so this
# test measures the FANOUT, not stitch2d's feature matching.
N_TILES = 1
N_Z = 3
TILE_SHAPE = (8, 8)


def _make_inventory(tmp_path: Path, *, n_time: int = 2, channel_id: str = "BF") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for time_index in range(n_time):
        for tile_id in range(N_TILES):
            for z_index in range(N_Z):
                raw = tmp_path / "raw" / f"{channel_id}_t{time_index}_x{tile_id}_z{z_index}.tif"
                raw.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    rng.integers(0, 4096, size=TILE_SHAPE, dtype=np.uint16)
                ).save(raw)
                rows.append({
                    "experiment_id": EXPERIMENT_ID,
                    "position_index": 13,
                    "channel_id": channel_id,
                    "raw_channel_name": "CH1" if channel_id == "BF" else "CH2",
                    "time_index": time_index,
                    "elapsed_time_s": float(time_index * 60),
                    "micrometers_per_pixel": 6.5,
                    "image_width_px": TILE_SHAPE[1],
                    "image_height_px": TILE_SHAPE[0],
                    "microscope_id": "Keyence",
                    "well_index": WELL_INDEX,
                    "well_id": WELL_ID,
                    "tile_id": tile_id,
                    "position_index_within_well": tile_id,
                    "n_tiles_in_well": N_TILES,
                    "z_index": z_index,
                    "channel_index": 1 if channel_id == "BF" else 2,
                    "time_index_claimed": time_index,
                    "acquisition_time_s": float(time_index * 60),
                    "objective_magnification": "4x",
                    "orientation": "horizontal",
                    "source_tiff_path": str(raw),
                    "stage_x_nm": 0,
                    "stage_y_nm": 0,
                    "stage_z_nm": 0,
                })
    return pd.DataFrame(rows)


def _digest_tree(root: Path) -> dict[str, bytes]:
    """Map every materialized file to its content, keyed by path relative to its output root."""
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


def _pixels(root: Path) -> dict[str, tuple[tuple[int, ...], str, int]]:
    """Shape/dtype/checksum of every raster under root, keyed by relative path."""
    out = {}
    for rel, _ in _digest_tree(root).items():
        path = root / rel
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        arr = np.asarray(Image.open(path))
        out[rel] = (arr.shape, str(arr.dtype), int(arr.sum()))
    return out


def _run_single(tmp_path: Path, inventory: pd.DataFrame, product, out_dir: Path) -> pd.DataFrame:
    return mwk.materialize_keyence_product_for_well(
        experiment_id=EXPERIMENT_ID,
        well_id=WELL_ID,
        well_index=WELL_INDEX,
        well_acquisition_inventory_df=inventory,
        built_image_data_dir=out_dir,
        resolved_product=product,
        device="cpu",
        candidate=False,
    )


class TestOldVsNewEquivalence:
    """N products in one process == N separate single-product runs, byte for byte."""

    @pytest.fixture
    def scenario(self, tmp_path):
        inventory = _make_inventory(tmp_path)
        separate_dir = tmp_path / "separate"
        combined_dir = tmp_path / "combined"

        separate = {
            image_product_key_for_resolved_product(p): _run_single(
                tmp_path, inventory, p, separate_dir
            )
            for p in (PROJECTION, Z_STACK)
        }
        combined = mwk.materialize_keyence_products_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory,
            built_image_data_dir=combined_dir,
            resolved_products=(PROJECTION, Z_STACK),
            device="cpu",
            candidate=False,
        )
        return separate, combined, separate_dir, combined_dir

    def test_same_product_keys(self, scenario):
        separate, combined, _, _ = scenario
        assert set(combined) == set(separate)
        assert set(combined) == {"BF__projection__focus_stack", "BF__z_stack"}

    def test_inventory_rows_identical(self, scenario):
        separate, combined, separate_dir, combined_dir = scenario
        for product_key, expected in separate.items():
            actual = combined[product_key]
            # Column ORDER is load-bearing: assemble_well_frame_inventory compares column lists
            # order-sensitively across shards.
            assert list(actual.columns) == list(expected.columns), product_key
            # Paths differ only by output root; normalize before comparing bytes.
            norm_expected = expected.assign(
                image_path=expected["image_path"].str.replace(
                    str(separate_dir), "<ROOT>", regex=False
                ),
                index_map_path=expected["index_map_path"].astype(str).str.replace(
                    str(separate_dir), "<ROOT>", regex=False
                ),
            ).drop(columns=["raw_tile_path", "raw_tile_manifest_path"])
            norm_actual = actual.assign(
                image_path=actual["image_path"].str.replace(
                    str(combined_dir), "<ROOT>", regex=False
                ),
                index_map_path=actual["index_map_path"].astype(str).str.replace(
                    str(combined_dir), "<ROOT>", regex=False
                ),
            ).drop(columns=["raw_tile_path", "raw_tile_manifest_path"])
            pd.testing.assert_frame_equal(norm_actual, norm_expected, check_like=False)

    def test_same_output_filenames(self, scenario):
        _, _, separate_dir, combined_dir = scenario
        assert set(_digest_tree(separate_dir)) == set(_digest_tree(combined_dir))

    def test_pixels_dtype_and_shape_identical(self, scenario):
        _, _, separate_dir, combined_dir = scenario
        expected = _pixels(separate_dir)
        actual = _pixels(combined_dir)
        assert expected, "no rasters were written - the fixture proves nothing"
        assert actual == expected

    def test_written_bytes_identical(self, scenario):
        """The strongest form: encoded files match, so no write-policy drift slipped in."""
        _, _, separate_dir, combined_dir = scenario
        expected = _digest_tree(separate_dir)
        actual = _digest_tree(combined_dir)
        for rel in expected:
            if rel.endswith(".json"):
                continue  # provenance manifests embed absolute source paths
            assert actual[rel] == expected[rel], rel


class TestSharedWorkHappensOnce:
    """The point of the refactor: two products, one read / one focus / one geometry solve."""

    def test_two_products_read_each_plane_once(self, tmp_path, monkeypatch):
        inventory = _make_inventory(tmp_path, n_time=1)
        real_read = fm.read_keyence_plane
        calls: list[str] = []

        def counting_read(path, **kwargs):
            calls.append(str(path))
            return real_read(path, **kwargs)

        monkeypatch.setattr(fm, "read_keyence_plane", counting_read)
        mwk.materialize_keyence_products_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory,
            built_image_data_dir=tmp_path / "out",
            resolved_products=(PROJECTION, Z_STACK),
            device="cpu",
            candidate=False,
        )
        # N_TILES * N_Z planes exist for the single timepoint; each is read exactly once even
        # though two products consume them.
        assert len(calls) == N_TILES * N_Z
        assert len(set(calls)) == N_TILES * N_Z

    def test_two_products_focus_stack_once(self, tmp_path, monkeypatch):
        inventory = _make_inventory(tmp_path, n_time=1)
        real_focus = fm.focus_stack_group
        count = {"n": 0}

        def counting_focus(*args, **kwargs):
            count["n"] += 1
            return real_focus(*args, **kwargs)

        monkeypatch.setattr(fm, "focus_stack_group", counting_focus)
        mwk.materialize_keyence_products_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory,
            built_image_data_dir=tmp_path / "out",
            resolved_products=(PROJECTION, Z_STACK),
            device="cpu",
            candidate=False,
        )
        assert count["n"] == 1

    def test_single_product_run_reads_the_same_planes_once(self, tmp_path, monkeypatch):
        """Guards the wrapper: one product must not read more than the combined run did."""
        inventory = _make_inventory(tmp_path, n_time=1)
        real_read = fm.read_keyence_plane
        calls: list[str] = []
        monkeypatch.setattr(
            fm, "read_keyence_plane",
            lambda path, **kw: (calls.append(str(path)), real_read(path, **kw))[1],
        )
        _run_single(tmp_path, inventory, Z_STACK, tmp_path / "out")
        assert len(calls) == N_TILES * N_Z


class TestChannelIsolation:
    """Materials are never shared across channels: reductions and policies are per-channel."""

    def test_each_channel_is_prepared_separately(self, tmp_path, monkeypatch):
        bf = _make_inventory(tmp_path / "bf", n_time=1, channel_id="BF")
        rfp = _make_inventory(tmp_path / "rfp", n_time=1, channel_id="RFP")
        inventory = pd.concat([bf, rfp], ignore_index=True)

        seen: list[str] = []
        real_prepare = mwk.prepare_keyence_frame

        def spy(**kwargs):
            seen.append(kwargs["channel_id"])
            materials = real_prepare(**kwargs)
            assert (materials.channel_id == kwargs["channel_id"])
            # Every row handed to prepare must belong to that one channel.
            assert set(kwargs["frame_rows"]["channel_id"]) == {kwargs["channel_id"]}
            return materials

        monkeypatch.setattr(mwk, "prepare_keyence_frame", spy)
        rfp_product = ResolvedImageProduct(
            channel_id="RFP",
            image_product_type="projection",
            projection_method="focus_stack",
            xy_composition="mosaic",
        )
        result = mwk.materialize_keyence_products_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory,
            built_image_data_dir=tmp_path / "out",
            resolved_products=(PROJECTION, rfp_product),
            device="cpu",
            candidate=False,
        )
        assert sorted(seen) == ["BF", "RFP"]
        assert set(result) == {"BF__projection__focus_stack", "RFP__projection__focus_stack"}
        for product_key, df in result.items():
            expected_channel = product_key.split("__")[0]
            assert set(df["channel_id"]) == {expected_channel}


def test_duplicate_product_keys_fail_loud(tmp_path):
    inventory = _make_inventory(tmp_path, n_time=1)
    with pytest.raises(ValueError, match="Duplicate image product keys"):
        mwk.materialize_keyence_products_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory,
            built_image_data_dir=tmp_path / "out",
            resolved_products=(PROJECTION, PROJECTION),
            device="cpu",
            candidate=False,
        )


def test_no_products_fails_loud(tmp_path):
    inventory = _make_inventory(tmp_path, n_time=1)
    with pytest.raises(ValueError, match="No resolved products"):
        mwk.materialize_keyence_products_for_well(
            experiment_id=EXPERIMENT_ID,
            well_id=WELL_ID,
            well_index=WELL_INDEX,
            well_acquisition_inventory_df=inventory,
            built_image_data_dir=tmp_path / "out",
            resolved_products=(),
            device="cpu",
            candidate=False,
        )
