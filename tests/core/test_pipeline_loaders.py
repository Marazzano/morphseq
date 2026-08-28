from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from src.core.data.dataset_classes import ManifestDataset
from src.core.lightning.pl_wrappers import LitModel


BF_PRODUCT = "BF__projection__focus_stack__clahe_blend"


def _tables(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations: list[dict[str, object]] = []
    assets: list[dict[str, object]] = []
    split_specs = [
        ("train", "physical/train-a"),
        ("train", "physical/train-b"),
        ("eval", "physical/eval-a"),
        ("eval", "physical/eval-b"),
        ("test", "physical/test-a"),
        ("test", "physical/test-b"),
    ]
    for index, (split, physical_embryo_id) in enumerate(split_specs):
        snip_id = f"opaque observation {index}"
        path = tmp_path / f"loader-{index}.png"
        Image.fromarray(
            np.full((6, 4), fill_value=20 + index, dtype=np.uint8), mode="L"
        ).save(path)
        observations.append(
            {
                "snip_id": snip_id,
                "physical_embryo_id": physical_embryo_id,
                "split": split,
                "incubation_temperature_c": 27.0 + index / 4,
                "temperature_status": "available",
                "elapsed_time_s": 100.5 + index,
                "elapsed_time_status": "available",
                "time_index": index,
                "predicted_stage_hpf": 10.0 + index,
                "stage_status": "available" if index != 5 else "unavailable",
                "stage_model_version": "clock-v2" if index != 5 else None,
            }
        )
        assets.append(
            {
                "snip_id": snip_id,
                "snip_product_key": BF_PRODUCT,
                "z_index": pd.NA,
                "processed_snip_path": str(path),
                "is_valid_snip": True,
            }
        )
    return pd.DataFrame(observations), pd.DataFrame(assets)


class _ManifestDataConfig:
    def __init__(self, observations: pd.DataFrame, assets: pd.DataFrame) -> None:
        self.observations = observations
        self.assets = assets
        self.batch_size = 2
        self.num_workers = 0
        self.drop_last = False
        self.loader_seed = 123
        self.calls: list[str] = []

    def create_dataset(self, *, split: str) -> ManifestDataset:
        self.calls.append(split)
        return ManifestDataset(
            self.observations,
            self.assets,
            split=split,
            snip_product_key=BF_PRODUCT,
            target_size=(8, 5),
        )


def _lit_model(data_config: object) -> LitModel:
    return LitModel(
        model=torch.nn.Identity(),
        loss_fn=torch.nn.Identity(),
        data_cfg=data_config,
        train_cfg=SimpleNamespace(),
    )


def test_lightning_loaders_are_split_local_and_metadata_complete(
    tmp_path: Path,
) -> None:
    observations, assets = _tables(tmp_path)
    config = _ManifestDataConfig(observations, assets)
    module = _lit_model(config)

    train_loader = module.train_dataloader()
    eval_loader = module.val_dataloader()
    test_loader = module.test_dataloader()

    assert train_loader.dataset.split == "train"
    assert eval_loader.dataset.split == "eval"
    assert test_loader.dataset.split == "test"
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(eval_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)

    identity_sets = [
        loader.dataset.physical_embryo_ids
        for loader in (train_loader, eval_loader, test_loader)
    ]
    assert identity_sets[0].isdisjoint(identity_sets[1])
    assert identity_sets[0].isdisjoint(identity_sets[2])
    assert identity_sets[1].isdisjoint(identity_sets[2])

    expected_splits = ["train", "eval", "test"]
    for loader, split in zip((train_loader, eval_loader, test_loader), expected_splits):
        batch = next(iter(loader))
        assert batch.data.shape == (2, 1, 8, 5)
        assert batch.data.dtype == torch.float32
        assert torch.isfinite(batch.data).all()
        assert batch.split == [split, split]
        assert len(batch.snip_id) == 2
        assert len(batch.physical_embryo_id) == 2
        assert batch.snip_product_key == [BF_PRODUCT, BF_PRODUCT]
        assert batch.z_index == [None, None]
        assert batch.temperature_status == ["available", "available"]
        assert batch.elapsed_time_status == ["available", "available"]
        assert len(batch.incubation_temperature_c) == 2
        assert len(batch.elapsed_time_s) == 2
        assert len(batch.time_index) == 2
        assert len(batch.predicted_stage_hpf) == 2
        assert len(batch.stage_status) == 2
        assert len(batch.stage_model_version) == 2
        assert len(batch.processed_snip_path) == 2

    # Datasets are cached by split rather than recreated as full positional views.
    assert module.train_dataloader().dataset is train_loader.dataset
    assert config.calls == ["train", "eval", "test"]


def test_temperature_and_elapsed_time_survive_collate_unchanged(tmp_path: Path) -> None:
    observations, assets = _tables(tmp_path)
    module = _lit_model(_ManifestDataConfig(observations, assets))
    batch = next(iter(module.val_dataloader()))
    expected = observations.loc[observations.split.eq("eval")].reset_index(drop=True)
    assert sorted(batch.incubation_temperature_c.tolist()) == sorted(
        expected.incubation_temperature_c.tolist()
    )
    assert sorted(batch.elapsed_time_s.tolist()) == sorted(
        expected.elapsed_time_s.tolist()
    )
    assert sorted(batch.time_index.tolist()) == sorted(expected.time_index.tolist())


def test_loader_rejects_legacy_full_dataset_constructor() -> None:
    class LegacyConfig:
        batch_size = 1
        num_workers = 0

        def create_dataset(self):
            raise AssertionError("must not be reached")

    with pytest.raises(TypeError, match="named split"):
        _lit_model(LegacyConfig()).train_dataloader()


def test_loader_hard_fails_if_physical_embryo_crosses_splits() -> None:
    class MinimalDataset(Dataset):
        def __init__(self, split: str, identity: str) -> None:
            self.split = split
            self.physical_embryo_ids = frozenset({identity})

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return {"data": torch.zeros(1, 2, 2)}

    class BadConfig:
        batch_size = 1
        num_workers = 0

        def create_dataset(self, *, split: str) -> Dataset:
            return MinimalDataset(split, "same explicit physical embryo")

    module = _lit_model(BadConfig())
    module.train_dataloader()
    with pytest.raises(ValueError, match="physical_embryo_id"):
        module.val_dataloader()


def test_test_loader_failure_is_explicit_when_split_is_unsupported() -> None:
    class MinimalDataset(Dataset):
        split = "train"
        physical_embryo_ids = frozenset({"physical/train"})

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return {"data": torch.zeros(1, 2, 2)}

    class NoTestConfig:
        batch_size = 1
        num_workers = 0

        def create_dataset(self, *, split: str) -> Dataset:
            if split == "test":
                raise NotImplementedError("test split unavailable by declared policy")
            return MinimalDataset()

    with pytest.raises(NotImplementedError, match="test split unavailable"):
        _lit_model(NoTestConfig()).test_dataloader()


def test_worker_storage_is_dataframe_free_and_decode_failures_are_shared(
    tmp_path: Path,
) -> None:
    observations, assets = _tables(tmp_path)
    first_train_id = observations.loc[observations.split.eq("train"), "snip_id"].iloc[0]
    selected = assets.snip_id.eq(first_train_id) & assets.snip_product_key.eq(
        BF_PRODUCT
    )
    assets.loc[selected, "processed_snip_path"] = str(tmp_path / "missing-worker.png")
    dataset = ManifestDataset(
        observations,
        assets,
        split="train",
        snip_product_key=BF_PRODUCT,
        target_size=(8, 5),
        max_decode_failures=2,
    )
    assert all(not isinstance(value, pd.DataFrame) for value in vars(dataset).values())

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )
    batches = list(loader)
    assert len(batches) == 2
    assert dataset.decode_failure_count == 1
    assert dataset.decode_failure_records[0].snip_id == first_train_id
    assert all(batch.split == ["train"] for batch in batches)
