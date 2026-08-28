from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional

from src.core.data.asset_selection import group_resolved_asset_rows, validate_vanilla_resolved_view
from src.core.data.data_transforms import (
    RESIZE_ANTIALIAS,
    RESIZE_INTERPOLATION,
    basic_transform,
    contrastive_transform,
)
from src.core.data.dataset_classes import (
    BasicDataset,
    NTXentDataset,
    collate_manifest_dataset_output,
)
from tests.core.fixtures.manifest_v2 import BF_PRODUCT, synthetic_manifest_v2


def _write_fixture_pixels(manifest, *, shape: tuple[int, int] = (32, 64)) -> None:
    for row_number, row in manifest.asset_table.iterrows():
        path = Path(row["processed_snip_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        yy, xx = np.indices(shape)
        pixels = ((yy * 11 + xx * 7 + row_number * 19) % 256).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(path)
        mask_path = Path(row["embryo_mask_snip_path"])
        Image.fromarray(np.full(shape, 255, dtype=np.uint8), mode="L").save(mask_path)


@pytest.fixture
def manifest_with_pixels(tmp_path):
    manifest = synthetic_manifest_v2(tmp_path)
    _write_fixture_pixels(manifest)
    return manifest


def test_dataset_row_pixel_ids_and_metadata_agree(manifest_with_pixels) -> None:
    manifest = manifest_with_pixels
    dataset = BasicDataset(
        resolved_sample_table=manifest.resolved_sample_table,
        product_key=manifest.policy.selected_product_key,
    )

    item = dataset[1]
    row = manifest.resolved_sample_table.iloc[1]
    assert item.data.shape == (1, 288, 128)
    assert item.data.dtype == torch.float32
    assert torch.isfinite(item.data).all()
    assert 0.0 <= float(item.data.min()) <= float(item.data.max()) <= 1.0
    assert item.snip_id == row.snip_id
    assert item.physical_embryo_id == row.physical_embryo_id
    assert item.snip_product_key == row.snip_product_key
    assert item.z_index is None
    assert item.asset_key == (row.snip_id, row.snip_product_key, None)
    assert item.processed_snip_path == row.processed_snip_path
    assert item.embryo_mask_snip_path == row.embryo_mask_snip_path
    assert item.incubation_temperature_c == row.incubation_temperature_c
    assert item.elapsed_time_s == row.elapsed_time_s
    assert item.time_index == row.time_index
    assert item.predicted_stage_hpf == row.predicted_stage_hpf
    assert item.stage_status == row.stage_status
    assert item.stage_model_version == row.stage_model_version
    assert item.metadata["sa_outlier_flag"] is False
    assert item.metadata["sa_qc_applicability"] == "exclusion"


def test_resolved_view_is_only_order_authority_with_sibling_assets(manifest_with_pixels) -> None:
    manifest = manifest_with_pixels
    assert len(manifest.asset_table) == 7
    assert manifest.asset_table.loc[
        manifest.asset_table.snip_id.eq("snip::alpha"), "snip_product_key"
    ].nunique() == 3
    assert manifest.asset_table.loc[
        manifest.asset_table.snip_product_key.eq("BF__z_stack__no_change"), "z_index"
    ].tolist() == [0, 1, 2]

    dataset = BasicDataset(
        manifest.resolved_sample_table,
        manifest.policy.selected_product_key,
    )
    assert [dataset[index].snip_id for index in range(len(dataset))] == manifest.resolved_sample_table[
        "snip_id"
    ].tolist()
    assert all(dataset[index].snip_product_key == BF_PRODUCT for index in range(len(dataset)))
    assert not hasattr(dataset, "asset_table")


def test_resolved_grouping_seam_preserves_ordered_row_lists(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table
    future_multi_asset_view = pd.concat(
        [table.iloc[[0]], table.iloc[[0]], table.iloc[[1]]],
        ignore_index=True,
    )
    groups = group_resolved_asset_rows(future_multi_asset_view)
    assert groups[0].snip_id == "snip::alpha"
    assert groups[0].resolved_row_references == (0, 1)
    assert groups[1].snip_id == "snip::beta"
    assert groups[1].resolved_row_references == (2,)


def test_vanilla_resolved_guard_names_zero_and_multiple_references(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table
    with pytest.raises(ValueError, match=r"missing snip_id=.*snip::missing"):
        validate_vanilla_resolved_view(
            table,
            product_key=BF_PRODUCT,
            expected_observation_ids=[*table.snip_id.tolist(), "snip::missing"],
        )

    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"exactly one.*snip::alpha"):
        validate_vanilla_resolved_view(duplicate, product_key=BF_PRODUCT)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("snip_product_key", "RFP__projection__max__no_change"),
        ("z_index", 0),
    ],
)
def test_vanilla_resolved_guard_rejects_wrong_product_or_plane(
    manifest_with_pixels,
    column,
    value,
) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    table.loc[0, column] = value
    with pytest.raises(ValueError, match="snip::alpha"):
        BasicDataset(table, BF_PRODUCT)


def test_deterministic_transform_matches_declared_pil_reference_exactly(tmp_path) -> None:
    yy, xx = np.indices((37, 91))
    rgb = np.stack(
        [
            (xx * 5 + yy * 3) % 256,
            (xx * 13 + 17) % 256,
            (yy * 19 + 29) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    path = tmp_path / "reference.png"
    Image.fromarray(rgb, mode="RGB").save(path)

    with Image.open(path) as image:
        actual = basic_transform((23, 41))(image)
    with Image.open(path) as image:
        expected_pil = tv_functional.resize(
            image.convert("L"),
            [23, 41],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        expected = tv_functional.to_tensor(expected_pil).to(torch.float32)

    assert RESIZE_INTERPOLATION is InterpolationMode.BILINEAR
    assert RESIZE_ANTIALIAS is True
    assert float((actual - expected).abs().max()) == 0.0


def test_non_default_size_works_for_basic_and_contrastive(manifest_with_pixels) -> None:
    manifest = manifest_with_pixels
    basic = BasicDataset(
        manifest.resolved_sample_table,
        BF_PRODUCT,
        input_dim=(1, 24, 17),
    )
    metric_table = manifest.resolved_sample_table.copy()
    metric_table["metric_group_code"] = 0
    paired = NTXentDataset(
        metric_table,
        BF_PRODUCT,
        pair_row_references=[0, 1, 2],
        policy_name="test_only_single_group",
        input_dim=(1, 24, 17),
    )
    assert basic[0].data.shape == (1, 24, 17)
    assert paired[0].data.shape == (2, 1, 24, 17)
    with Image.open(manifest.resolved_sample_table.iloc[0].processed_snip_path) as image:
        assert contrastive_transform((24, 17))(image).shape == (1, 24, 17)


class _IndependentViewTransform:
    def __init__(self, size: tuple[int, int]) -> None:
        self.base = basic_transform(size)
        self.call_count = 0

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = self.base(image)
        tensor = tensor.clone()
        tensor[:, 0, 0] = float(self.call_count % 2)
        self.call_count += 1
        return tensor


def test_metric_views_are_independently_transformed_after_resize(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    table["metric_group_code"] = 0
    transform = _IndependentViewTransform((19, 13))
    dataset = NTXentDataset(
        table,
        BF_PRODUCT,
        pair_row_references=[0, 1, 2],
        policy_name="dummy_a4_pairing",
        input_dim=(1, 19, 13),
        transform=transform,
    )

    item = dataset[0]
    assert transform.call_count == 2
    assert item.data.shape == (2, 1, 19, 13)
    assert not torch.equal(item.data[0], item.data[1])
    assert item.self_stats == ["animal::alpha", 24.0, 0]
    assert item.other_stats == ["animal::alpha", 24.0, 0]


def test_nullable_metadata_collates_without_loss(manifest_with_pixels) -> None:
    dataset = BasicDataset(
        manifest_with_pixels.resolved_sample_table,
        BF_PRODUCT,
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=3,
                collate_fn=collate_manifest_dataset_output,
            )
        )
    )
    assert batch.data.shape == (3, 1, 288, 128)
    assert batch.snip_id == ["snip::alpha", "snip::beta", "snip::gamma"]
    assert batch.z_index == [None, None, None]
    assert batch.asset_key == [
        ("snip::alpha", BF_PRODUCT, None),
        ("snip::beta", BF_PRODUCT, None),
        ("snip::gamma", BF_PRODUCT, None),
    ]
    assert torch.equal(batch.incubation_temperature_c, torch.tensor([28.5, 28.5, 28.5]))
    assert torch.equal(batch.elapsed_time_s, torch.tensor([0.0, 900.0, 0.0]))
    assert batch.metadata["chem_perturbation"] == [None, None, None]


def test_worker_storage_has_no_dataframe_or_imagefolder_boundary(manifest_with_pixels) -> None:
    dataset = BasicDataset(manifest_with_pixels.resolved_sample_table, BF_PRODUCT)
    assert all(not isinstance(value, pd.DataFrame) for value in vars(dataset).values())
    assert dataset.worker_storage_kind("split") == "categorical_codes"
    assert dataset.worker_storage_kind("time_index").startswith("array:")
    source = inspect.getsource(type(dataset))
    assert "ImageFolder" not in source
    assert "make_seq_key" not in source
    assert "glob(" not in source
