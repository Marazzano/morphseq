from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler

from src.core.data.dataset_classes import (
    BasicDataset,
    ManifestDecodeError,
    NTXentDataset,
    collate_manifest_dataset_output,
)
from src.core.lightning.pl_wrappers import LitModel
from tests.core.fixtures.manifest_v2 import BF_PRODUCT, synthetic_manifest_v2


def _write_pixels(manifest) -> None:
    for row_number, row in manifest.asset_table.iterrows():
        path = Path(row.processed_snip_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.full((31, 47), row_number * 23, dtype=np.uint8)
        Image.fromarray(pixels, mode="L").save(path)


class _LoaderDataConfig:
    def __init__(self, table) -> None:
        self.resolved_sample_table = table
        self.batch_size = 1
        self.num_workers = 0
        self.loader_seed = 817
        self.pin_memory = False
        self.persistent_workers = False
        self.drop_last_train = False
        self.created_splits: list[str] = []

    def create_dataset(self, *, split: str) -> BasicDataset:
        self.created_splits.append(split)
        return BasicDataset(
            resolved_sample_table=self.resolved_sample_table,
            product_key=BF_PRODUCT,
            split=split,
        )


class _LoaderOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, value):
        return value


def _lit_model(data_config: _LoaderDataConfig) -> LitModel:
    return LitModel(
        model=_LoaderOnlyModel(),
        loss_fn=nn.Identity(),
        data_cfg=data_config,
        train_cfg=SimpleNamespace(),
    )


@pytest.fixture
def manifest_with_pixels(tmp_path):
    manifest = synthetic_manifest_v2(tmp_path)
    _write_pixels(manifest)
    return manifest


def test_lightning_loaders_are_split_local_and_test_is_explicit(manifest_with_pixels) -> None:
    config = _LoaderDataConfig(manifest_with_pixels.resolved_sample_table)
    model = _lit_model(config)

    train_loader = model.train_dataloader()
    eval_loader = model.val_dataloader()
    test_loader = model.test_dataloader()

    assert config.created_splits == ["train", "eval", "test"]
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(eval_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    assert train_loader.batch_sampler.drop_last is False
    assert eval_loader.batch_sampler.drop_last is False
    assert test_loader.batch_sampler.drop_last is False

    train_batch = next(iter(train_loader))
    eval_batch = next(iter(eval_loader))
    test_batch = next(iter(test_loader))
    assert train_batch.split == ["train"]
    assert eval_batch.split == ["eval"]
    assert test_batch.split == ["test"]
    assert set(train_batch.physical_embryo_id).isdisjoint(eval_batch.physical_embryo_id)
    assert set(train_batch.physical_embryo_id).isdisjoint(test_batch.physical_embryo_id)
    assert set(eval_batch.physical_embryo_id).isdisjoint(test_batch.physical_embryo_id)


def test_dataset_rejects_physical_embryo_crossing_splits(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    table.loc[table.snip_id.eq("snip::beta"), "physical_embryo_id"] = "animal::alpha"
    with pytest.raises(ValueError, match=r"physical_embryo_id=.*animal::alpha"):
        BasicDataset(table, BF_PRODUCT)


def test_loader_shuffle_is_reproducible_without_positional_sampler(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    table.loc[table.snip_id.eq("snip::beta"), "split"] = "train"
    first = _lit_model(_LoaderDataConfig(table)).train_dataloader()
    second = _lit_model(_LoaderDataConfig(table)).train_dataloader()

    first_order = [batch.snip_id[0] for batch in first]
    second_order = [batch.snip_id[0] for batch in second]
    assert first_order == second_order
    assert set(first_order) == {"snip::alpha", "snip::beta"}


def test_decode_resampling_records_failure_and_stays_in_split(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    table.loc[table.snip_id.eq("snip::beta"), "split"] = "train"
    Path(table.loc[table.snip_id.eq("snip::alpha"), "processed_snip_path"].item()).unlink()
    dataset = BasicDataset(
        table,
        BF_PRODUCT,
        split="train",
        max_decode_failures=1,
    )

    item = dataset[0]
    assert item.snip_id == "snip::beta"
    assert item.requested_index == 0
    assert item.index == 1
    assert item.split == "train"
    report = dataset.decode_failure_report()
    assert report["count"] == 1
    assert report["failed_assets"] == [
        {
            "resolved_row_reference": 0,
            "snip_id": "snip::alpha",
            "snip_product_key": BF_PRODUCT,
            "z_index": None,
            "processed_snip_path": table.iloc[0].processed_snip_path,
            "split": "train",
        }
    ]


def test_decode_resampling_never_crosses_to_another_split(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    Path(table.iloc[0].processed_snip_path).unlink()
    dataset = BasicDataset(table, BF_PRODUCT, max_decode_failures=10)

    with pytest.raises(ManifestDecodeError, match=r"No decodable same-split replacement.*snip::alpha"):
        dataset[0]
    assert dataset.decode_failure_count == 1


def test_decode_failures_abort_above_configured_threshold(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    table.loc[table.snip_id.eq("snip::beta"), "split"] = "train"
    Path(table.iloc[0].processed_snip_path).unlink()
    Path(table.iloc[1].processed_snip_path).unlink()
    dataset = BasicDataset(table, BF_PRODUCT, split="train", max_decode_failures=1)

    with pytest.raises(ManifestDecodeError, match=r"threshold exceeded.*count=2, threshold=1"):
        dataset[0]
    assert dataset.decode_failure_count == 2
    assert {failure["snip_id"] for failure in dataset.decode_failure_report()["failed_assets"]} == {
        "snip::alpha",
        "snip::beta",
    }


def test_test_only_metric_batch_preserves_stats_contract(manifest_with_pixels) -> None:
    table = manifest_with_pixels.resolved_sample_table.copy()
    table["metric_group_code"] = [0, 0, 0]
    table["split"] = "train"
    dataset = NTXentDataset(
        table,
        BF_PRODUCT,
        pair_row_references=[1, 2, 0],
        policy_name="test_only_single_group",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_manifest_dataset_output,
        shuffle=False,
    )

    batch = next(iter(loader))
    assert batch.data.shape == (2, 2, 1, 288, 128)
    assert len(batch.self_stats) == 3
    assert len(batch.other_stats) == 3
    assert list(batch.self_stats[0]) == ["animal::alpha", "animal::beta"]
    assert torch.equal(batch.self_stats[1], torch.tensor([24.0, 24.25]))
    assert torch.equal(batch.self_stats[2], torch.tensor([0, 0]))
    assert list(batch.other_stats[0]) == ["animal::beta", "animal::gamma"]


def test_worker_loader_uses_compact_dataset_without_source_dataframe(manifest_with_pixels) -> None:
    dataset = BasicDataset(
        manifest_with_pixels.resolved_sample_table,
        BF_PRODUCT,
        split="train",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=collate_manifest_dataset_output,
    )
    batch = next(iter(loader))
    assert batch.snip_id == ["snip::alpha"]
