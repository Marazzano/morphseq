from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.core.data.asset_selection import (
    AssetSelectionError,
    VanillaBFProjectionSelector,
    normalize_nullable_z_index,
    resolve_asset_row_groups,
)
from src.core.data.data_transforms import (
    ContrastiveTransform,
    RESIZE_ANTIALIAS,
    RESIZE_INTERPOLATION,
    basic_transform,
    contrastive_transform,
)
from src.core.data.dataset_classes import (
    BasicDataset,
    DecodeFailureLimitError,
    ManifestDataset,
    ManifestPairedDataset,
    collate_manifest_dataset_output,
)


BF_PRODUCT = "BF__projection__focus_stack__clahe_blend"
RFP_PRODUCT = "RFP__projection__max"
Z_PRODUCT = "BF__z_plane__raw"


def _write_image(path: Path, offset: int) -> None:
    rows = np.arange(24, dtype=np.uint8).reshape(4, 6)
    red = (rows + offset).astype(np.uint8)
    rgb = np.stack([red, np.flipud(red), np.fliplr(red)], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)


def _contract_v2_tables(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observation_rows = []
    specs = [
        ("observation::opaque/a", "physical::A", "train", 11, 12.5, 1),
        ("observation::opaque/b", "physical::B", "train", 12, 13.0, 1),
        ("observation::opaque/e", "physical::E", "eval", 13, 14.0, 2),
        ("observation::opaque/t", "physical::T", "test", 14, 15.0, 2),
    ]
    image_paths: dict[str, Path] = {}
    for index, (
        snip_id,
        physical_id,
        split,
        embryo_code,
        stage,
        group_code,
    ) in enumerate(specs):
        path = tmp_path / f"fixture-{index}.png"
        _write_image(path, offset=index * 7)
        image_paths[snip_id] = path
        observation_rows.append(
            {
                "snip_id": snip_id,
                "physical_embryo_id": physical_id,
                "embryo_id": f"declared-embryo-{index}",
                "experiment_id": f"experiment-{index // 2}",
                "well_id": f"well-{index}",
                "image_id": f"image-{index}",
                "time_index": index,
                "split": split,
                "incubation_temperature_c": 28.5 + index / 10,
                "temperature_status": "available",
                "elapsed_time_s": 30.25 + index,
                "elapsed_time_status": "available",
                "predicted_stage_hpf": stage,
                "stage_status": "available",
                "stage_model_version": "clock-v2",
                "start_age_hpf": 6.0,
                "start_age_source": "plate_metadata",
                "use_snip": True,
                "qc_status": "evaluated",
                "sa_outlier_flag": False,
                "sa_outlier_applicable": True,
                "metric_embryo_code": embryo_code,
                "metric_group_code": group_code,
            }
        )

    asset_rows = []
    first_snip_id = specs[0][0]
    asset_rows.extend(
        [
            {
                "snip_id": first_snip_id,
                "snip_product_key": RFP_PRODUCT,
                "z_index": pd.NA,
                "processed_snip_path": str(image_paths[first_snip_id]),
                "embryo_mask_snip_path": "mask-rfp.png",
                "is_valid_snip": True,
                "source_micrometers_per_pixel": 2.0,
                "snip_micrometers_per_pixel": 6.5,
            },
            {
                "snip_id": first_snip_id,
                "snip_product_key": Z_PRODUCT,
                "z_index": 0,
                "processed_snip_path": str(image_paths[first_snip_id]),
                "embryo_mask_snip_path": "mask-z0.png",
                "is_valid_snip": True,
                "source_micrometers_per_pixel": 2.0,
                "snip_micrometers_per_pixel": 6.5,
            },
            {
                "snip_id": first_snip_id,
                "snip_product_key": Z_PRODUCT,
                "z_index": 1,
                "processed_snip_path": str(image_paths[first_snip_id]),
                "embryo_mask_snip_path": "mask-z1.png",
                "is_valid_snip": True,
                "source_micrometers_per_pixel": 2.0,
                "snip_micrometers_per_pixel": 6.5,
            },
        ]
    )
    for snip_id, *_ in specs:
        asset_rows.append(
            {
                "snip_id": snip_id,
                "snip_product_key": BF_PRODUCT,
                "z_index": pd.NA,
                "processed_snip_path": str(image_paths[snip_id]),
                "embryo_mask_snip_path": f"mask-{snip_id}.png",
                "is_valid_snip": "True" if snip_id == first_snip_id else True,
                "source_micrometers_per_pixel": 2.0,
                "snip_micrometers_per_pixel": 6.5,
            }
        )
    return pd.DataFrame(observation_rows), pd.DataFrame(asset_rows)


def test_dataset_row_pixels_ids_and_complete_metadata_agree(tmp_path: Path) -> None:
    observations, assets = _contract_v2_tables(tmp_path)
    dataset = ManifestDataset(
        observations,
        assets,
        split="train",
        snip_product_key=BF_PRODUCT,
    )

    item = dataset[0]
    assert item.data.shape == (1, 288, 128)
    assert item.data.dtype == torch.float32
    assert torch.isfinite(item.data).all()
    assert 0.0 <= float(item.data.min()) <= float(item.data.max()) <= 1.0
    assert item.snip_id == observations.iloc[0].snip_id
    assert item.physical_embryo_id == observations.iloc[0].physical_embryo_id
    assert item.snip_product_key == BF_PRODUCT
    assert item.z_index is None
    assert item.split == "train"
    assert item.processed_snip_path == item.label
    assert (
        item.incubation_temperature_c == observations.iloc[0].incubation_temperature_c
    )
    assert item.elapsed_time_s == observations.iloc[0].elapsed_time_s
    assert item.time_index == observations.iloc[0].time_index
    assert item.predicted_stage_hpf == observations.iloc[0].predicted_stage_hpf
    assert item.stage_status == "available"
    assert item.stage_model_version == "clock-v2"
    assert item.start_age_source == "plate_metadata"
    assert item.sa_outlier_flag is False
    assert (
        item.asset_metadata["embryo_mask_snip_path"] == "mask-observation::opaque/a.png"
    )
    assert item.asset_metadata["source_micrometers_per_pixel"] == 2.0
    assert item.asset_metadata["snip_micrometers_per_pixel"] == 6.5

    # Construction has severed source-DataFrame ownership and compacted worker storage.
    assert all(not isinstance(value, pd.DataFrame) for value in vars(dataset).values())
    observations.loc[0, "incubation_temperature_c"] = 99.0
    assert dataset[0].incubation_temperature_c == 28.5


def test_vanilla_selector_is_exact_with_bf_rfp_and_z_siblings(tmp_path: Path) -> None:
    observations, assets = _contract_v2_tables(tmp_path)
    original_assets = assets.copy(deep=True)
    groups = resolve_asset_row_groups(
        observations.iloc[:1], assets, VanillaBFProjectionSelector(BF_PRODUCT)
    )
    assert len(groups) == 1
    selected = assets.iloc[groups[0].asset_row_positions[0]]
    assert selected.snip_product_key == BF_PRODUCT
    assert pd.isna(selected.z_index)
    pd.testing.assert_frame_equal(assets, original_assets)


def test_selector_row_list_seam_preserves_ordered_z_assets(tmp_path: Path) -> None:
    observations, assets = _contract_v2_tables(tmp_path)

    class OrderedZSelector:
        name = "synthetic_ordered_z"

        def select_row_positions(
            self, *, snip_id, candidate_row_positions, asset_table
        ):
            return [
                position
                for position in candidate_row_positions
                if asset_table.iloc[position].snip_product_key == Z_PRODUCT
            ]

    groups = resolve_asset_row_groups(observations.iloc[:1], assets, OrderedZSelector())
    assert [
        normalize_nullable_z_index(
            assets.iloc[position].z_index, snip_id=groups[0].snip_id
        )
        for position in groups[0].asset_row_positions
    ] == [0, 1]


@pytest.mark.parametrize("mode", ["zero", "multiple"])
def test_vanilla_selector_fails_by_snip_id_for_zero_or_multiple_assets(
    tmp_path: Path, mode: str
) -> None:
    observations, assets = _contract_v2_tables(tmp_path)
    snip_id = observations.iloc[0].snip_id
    bf_mask = assets.snip_id.eq(snip_id) & assets.snip_product_key.eq(BF_PRODUCT)
    if mode == "zero":
        assets.loc[bf_mask, "snip_product_key"] = "different-product"
    else:
        duplicate = assets.loc[bf_mask].copy()
        duplicate.loc[:, "processed_snip_path"] = str(tmp_path / "duplicate.png")
        assets = pd.concat([assets, duplicate], ignore_index=True)

    with pytest.raises(AssetSelectionError, match=snip_id):
        ManifestDataset(
            observations,
            assets,
            split="train",
            snip_product_key=BF_PRODUCT,
        )


def test_deterministic_pixels_match_pinned_legacy_reference(tmp_path: Path) -> None:
    path = tmp_path / "reference.png"
    _write_image(path, offset=9)
    target_size = (9, 5)
    transform = basic_transform(target_size)
    with Image.open(path) as image:
        actual = transform(image)
    with Image.open(path) as image:
        expected_image = TF.resize(
            image.convert("L"),
            list(target_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        expected = TF.pil_to_tensor(expected_image).to(torch.float32).div(255.0)

    assert RESIZE_INTERPOLATION == InterpolationMode.BILINEAR
    assert RESIZE_ANTIALIAS is True
    assert actual.shape == (1, 9, 5)
    assert float(torch.max(torch.abs(actual - expected))) == 0.0


def test_non_default_target_size_works_for_basic_and_contrastive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sized.png"
    _write_image(path, offset=3)
    with Image.open(path) as image:
        basic = basic_transform((10, 7))(image)
    with Image.open(path) as image:
        contrastive = contrastive_transform((10, 7))(image)
    assert basic.shape == (1, 10, 7)
    assert contrastive.shape == (1, 10, 7)


def test_metric_views_are_independent_after_deterministic_resize(
    tmp_path: Path,
) -> None:
    observations, assets = _contract_v2_tables(tmp_path)

    class CountingAugmentation:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return (
                torch.zeros_like(tensor) if self.calls == 1 else torch.ones_like(tensor)
            )

    augmentation = CountingAugmentation()
    transform = ContrastiveTransform((8, 5), augmentation=augmentation)
    pair_map = {
        "observation::opaque/a": "observation::opaque/a",
        "observation::opaque/b": "observation::opaque/b",
    }
    dataset = ManifestPairedDataset(
        observations,
        assets,
        split="train",
        snip_product_key=BF_PRODUCT,
        transform=transform,
        pair_snip_ids=pair_map,
        metric_stats_columns=(
            "metric_embryo_code",
            "predicted_stage_hpf",
            "metric_group_code",
        ),
    )

    item = dataset[0]
    assert augmentation.calls == 2
    assert item.data.shape == (2, 1, 8, 5)
    assert torch.count_nonzero(item.data[0]) == 0
    assert torch.all(item.data[1] == 1)
    assert item.self_stats == [11, 12.5, 1]
    assert item.other_stats == [11, 12.5, 1]

    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_manifest_dataset_output,
            )
        )
    )
    assert len(batch.self_stats) == 3
    assert len(batch.other_stats) == 3
    assert all(value.shape == (2,) for value in batch.self_stats)
    assert all(value.shape == (2,) for value in batch.other_stats)
    assert batch.data.shape == (2, 2, 1, 8, 5)


def test_collate_preserves_nullable_z_temperature_and_elapsed_time(
    tmp_path: Path,
) -> None:
    observations, assets = _contract_v2_tables(tmp_path)
    dataset = ManifestDataset(
        observations,
        assets,
        split="train",
        snip_product_key=BF_PRODUCT,
        target_size=(8, 5),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_manifest_dataset_output,
    )
    batch = next(iter(loader))
    assert batch.data.shape == (2, 1, 8, 5)
    assert batch.z_index == [None, None]
    assert batch.snip_id == observations.iloc[:2].snip_id.tolist()
    assert (
        batch.incubation_temperature_c.tolist()
        == observations.iloc[:2].incubation_temperature_c.tolist()
    )
    assert (
        batch.elapsed_time_s.tolist() == observations.iloc[:2].elapsed_time_s.tolist()
    )
    assert batch.temperature_status == ["available", "available"]
    assert batch.elapsed_time_status == ["available", "available"]


def test_decode_recovery_stays_in_split_and_records_failed_identity(
    tmp_path: Path,
) -> None:
    observations, assets = _contract_v2_tables(tmp_path)
    first_snip_id = observations.iloc[0].snip_id
    bf_mask = assets.snip_id.eq(first_snip_id) & assets.snip_product_key.eq(BF_PRODUCT)
    assets.loc[bf_mask, "processed_snip_path"] = str(tmp_path / "missing.png")
    dataset = ManifestDataset(
        observations,
        assets,
        split="train",
        snip_product_key=BF_PRODUCT,
        max_decode_failures=1,
    )

    recovered = dataset[0]
    assert recovered.snip_id == observations.iloc[1].snip_id
    assert recovered.physical_embryo_id in dataset.physical_embryo_ids
    assert recovered.split == "train"
    assert dataset.decode_failure_count == 1
    assert dataset.decode_failure_records[0].snip_id == first_snip_id
    assert dataset.decode_failure_records[0].snip_product_key == BF_PRODUCT
    assert dataset.decode_failure_records[0].z_index is None

    strict_dataset = ManifestDataset(
        observations,
        assets,
        split="train",
        snip_product_key=BF_PRODUCT,
        max_decode_failures=0,
    )
    with pytest.raises(DecodeFailureLimitError, match=first_snip_id):
        strict_dataset[0]


def test_manifest_classes_have_no_imagefolder_or_root_constructor() -> None:
    assert issubclass(ManifestDataset, Dataset)
    assert BasicDataset is ManifestDataset
    assert "root" not in inspect.signature(ManifestDataset).parameters


def test_dataset_does_not_coerce_or_reconstruct_opaque_ids(tmp_path: Path) -> None:
    observations, assets = _contract_v2_tables(tmp_path)
    observations.loc[0, "snip_id"] = 123
    with pytest.raises(ValueError, match="opaque strings"):
        ManifestDataset(
            observations,
            assets,
            split="train",
            snip_product_key=BF_PRODUCT,
        )


def test_manifest_result_constructor_follows_resolved_sample_view_order(
    tmp_path: Path,
) -> None:
    observations, assets = _contract_v2_tables(tmp_path)
    assets = assets.copy()
    assets["asset_row_index"] = np.arange(len(assets), dtype=np.int64)
    train_ids = observations.loc[observations.split.eq("train"), "snip_id"].tolist()
    resolved_rows = []
    for snip_id in reversed(train_ids):
        selected_asset = assets.loc[
            assets.snip_id.eq(snip_id)
            & assets.snip_product_key.eq(BF_PRODUCT)
            & assets.z_index.isna()
        ].iloc[0]
        resolved_rows.append(
            {
                "snip_id": snip_id,
                "snip_product_key": BF_PRODUCT,
                "z_index": pd.NA,
                "split": "train",
                "asset_row_index": int(selected_asset.asset_row_index),
            }
        )
    resolved_view = pd.DataFrame(resolved_rows)
    manifest = SimpleNamespace(
        observation_table=observations,
        asset_table=assets,
        selected_sample_view=resolved_view,
        policy=SimpleNamespace(
            assets=SimpleNamespace(
                vanilla_product_key=BF_PRODUCT,
                z_mode="projection_null",
            )
        ),
    )

    dataset = ManifestDataset.from_manifest_result(
        manifest,
        split="train",
        target_size=(8, 5),
    )
    assert dataset.snip_ids == tuple(reversed(train_ids))
    assert [dataset[index].snip_id for index in range(len(dataset))] == list(
        reversed(train_ids)
    )
    assert [dataset[index].asset_row_reference for index in range(len(dataset))] == [
        row["asset_row_index"] for row in resolved_rows
    ]
