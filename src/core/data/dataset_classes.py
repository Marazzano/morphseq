"""Manifest-backed datasets for core-model training and evaluation.

The dataset order is the filtered observation-table order. Construction resolves
each observation through an explicit asset selector, then drops both source
DataFrames in favor of compact columnar storage. ``__getitem__`` opens only the
resolved asset path; it performs no directory search, filename parsing, or ID
reconstruction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import multiprocessing as mp
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from PIL import Image
from pythae.data.datasets import DatasetOutput
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from src.core.data.asset_selection import (
    VanillaBFProjectionSelector,
    normalize_nullable_z_index,
    resolve_asset_row_groups,
)
from src.core.data.data_transforms import DEFAULT_TARGET_SIZE, basic_transform

if TYPE_CHECKING:
    from src.core.data.manifest_types import ManifestResult


REQUIRED_OBSERVATION_COLUMNS = frozenset(
    {
        "snip_id",
        "physical_embryo_id",
        "split",
        "incubation_temperature_c",
        "temperature_status",
        "elapsed_time_s",
        "elapsed_time_status",
        "time_index",
        "predicted_stage_hpf",
        "stage_status",
        "stage_model_version",
    }
)
REQUIRED_ASSET_COLUMNS = frozenset(
    {
        "snip_id",
        "snip_product_key",
        "z_index",
        "processed_snip_path",
        "is_valid_snip",
    }
)


class ManifestDatasetError(ValueError):
    """Raised when resolved manifest rows violate the dataset boundary."""


class AssetDecodeError(RuntimeError):
    """Raised when a selected raster cannot become a valid model-input tensor."""


class DecodeFailureLimitError(RuntimeError):
    """Raised after the configured split-local decode failure threshold is exceeded."""


@dataclass(frozen=True)
class DecodeFailureRecord:
    split: str
    snip_id: str
    snip_product_key: str
    z_index: int | None
    processed_snip_path: str


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return False
    result = pd.isna(value)
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class _CompactColumn:
    def at(self, index: int) -> object:
        raise NotImplementedError


@dataclass(frozen=True)
class _CategoricalColumn(_CompactColumn):
    categories: tuple[str, ...]
    codes: np.ndarray

    def at(self, index: int) -> str | None:
        code = int(self.codes[index])
        return None if code < 0 else self.categories[code]


@dataclass(frozen=True)
class _BooleanColumn(_CompactColumn):
    values: np.ndarray

    def at(self, index: int) -> bool | None:
        value = int(self.values[index])
        return None if value < 0 else bool(value)


@dataclass(frozen=True)
class _IntegerColumn(_CompactColumn):
    values: np.ndarray
    missing: np.ndarray

    def at(self, index: int) -> int | None:
        return None if bool(self.missing[index]) else int(self.values[index])


@dataclass(frozen=True)
class _FloatColumn(_CompactColumn):
    values: np.ndarray
    missing: np.ndarray

    def at(self, index: int) -> float | None:
        return None if bool(self.missing[index]) else float(self.values[index])


@dataclass(frozen=True)
class _ObjectColumn(_CompactColumn):
    values: tuple[object, ...]

    def at(self, index: int) -> object:
        value = self.values[index]
        return None if _is_missing_scalar(value) else _python_scalar(value)


def _compact_column(values: Sequence[object]) -> _CompactColumn:
    non_missing = [value for value in values if not _is_missing_scalar(value)]

    if all(isinstance(value, (str, Path)) for value in non_missing):
        categories: list[str] = []
        code_by_value: dict[str, int] = {}
        codes = np.full(len(values), -1, dtype=np.int32)
        for index, value in enumerate(values):
            if _is_missing_scalar(value):
                continue
            text = str(value)
            if text not in code_by_value:
                code_by_value[text] = len(categories)
                categories.append(text)
            codes[index] = code_by_value[text]
        return _CategoricalColumn(tuple(categories), codes)

    if all(isinstance(value, (bool, np.bool_)) for value in non_missing):
        encoded = np.full(len(values), -1, dtype=np.int8)
        for index, value in enumerate(values):
            if not _is_missing_scalar(value):
                encoded[index] = int(bool(value))
        return _BooleanColumn(encoded)

    if all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        for value in non_missing
    ):
        encoded = np.zeros(len(values), dtype=np.int64)
        missing = np.zeros(len(values), dtype=np.bool_)
        for index, value in enumerate(values):
            if _is_missing_scalar(value):
                missing[index] = True
            else:
                encoded[index] = int(value)
        return _IntegerColumn(encoded, missing)

    if all(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        for value in non_missing
    ):
        encoded = np.zeros(len(values), dtype=np.float64)
        missing = np.zeros(len(values), dtype=np.bool_)
        for index, value in enumerate(values):
            if _is_missing_scalar(value):
                missing[index] = True
            else:
                encoded[index] = float(value)
        return _FloatColumn(encoded, missing)

    return _ObjectColumn(tuple(_python_scalar(value) for value in values))


class _CompactTable:
    """DataFrame-free worker storage with categorical string coding."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.columns = tuple(str(column) for column in frame.columns)
        self._columns = {
            str(column): _compact_column(frame[column].tolist())
            for column in frame.columns
        }
        self._length = len(frame)

    def __len__(self) -> int:
        return self._length

    def value(self, column: str, index: int) -> object:
        return self._columns[column].at(index)

    def row(self, index: int) -> dict[str, object]:
        return {column: values.at(index) for column, values in self._columns.items()}


def _require_columns(
    frame: pd.DataFrame, required: frozenset[str], *, table_name: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ManifestDatasetError(
            f"Contract-v2 {table_name} is missing required dataset columns: {missing}."
        )


def _require_opaque_string_ids(
    frame: pd.DataFrame, columns: Sequence[str], *, table_name: str
) -> None:
    for column in columns:
        invalid = [
            value
            for value in frame[column].tolist()
            if not isinstance(value, str) or not value
        ]
        if invalid:
            raise ManifestDatasetError(
                f"Contract-v2 {table_name} column={column!r} contains non-string or empty IDs: "
                f"{invalid[:10]!r}. IDs are opaque strings and are never coerced or rebuilt."
            )


def _validate_observation_splits(observation_table: pd.DataFrame) -> None:
    assigned = observation_table.loc[observation_table["split"].notna()]
    split_counts = (
        assigned.loc[:, ["physical_embryo_id", "split"]]
        .drop_duplicates()
        .groupby("physical_embryo_id", dropna=False)["split"]
        .nunique()
    )
    crossed = split_counts[split_counts > 1].index.tolist()
    if crossed:
        raise ManifestDatasetError(
            "physical_embryo_id values cross observation splits: "
            f"{[str(value) for value in crossed]!r}."
        )


def _split_observations_in_resolved_order(
    observation_table: pd.DataFrame,
    *,
    split: str,
    resolved_sample_view: pd.DataFrame | None,
    snip_product_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if resolved_sample_view is None:
        split_observations = observation_table.loc[
            observation_table["split"].eq(split)
        ].reset_index(drop=True)
        return split_observations, None

    required_view_columns = {
        "snip_id",
        "snip_product_key",
        "z_index",
        "split",
        "asset_row_index",
    }
    missing = sorted(required_view_columns - set(resolved_sample_view.columns))
    if missing:
        raise ManifestDatasetError(
            f"Resolved sample view is missing required columns: {missing!r}."
        )
    split_view = resolved_sample_view.loc[
        resolved_sample_view["split"].eq(split)
    ].reset_index(drop=True)
    if split_view["snip_id"].duplicated().any():
        duplicates = split_view.loc[
            split_view["snip_id"].duplicated(keep=False), "snip_id"
        ].tolist()
        raise ManifestDatasetError(
            f"Resolved sample view contains duplicate split={split!r} snip_id values: "
            f"{duplicates!r}."
        )

    wrong_products = split_view.loc[
        ~split_view["snip_product_key"].eq(snip_product_key),
        ["snip_id", "snip_product_key"],
    ]
    non_null_z = split_view.loc[split_view["z_index"].notna(), ["snip_id", "z_index"]]
    if not wrong_products.empty or not non_null_z.empty:
        raise ManifestDatasetError(
            f"Resolved sample view disagrees with vanilla product={snip_product_key!r}, "
            f"z_index=null for split={split!r}; wrong_products="
            f"{wrong_products.to_dict('records')!r}, non_null_z="
            f"{non_null_z.to_dict('records')!r}."
        )

    observation_by_id = observation_table.set_index("snip_id", drop=False)
    missing_observations = [
        snip_id
        for snip_id in split_view["snip_id"].tolist()
        if snip_id not in observation_by_id.index
    ]
    if missing_observations:
        raise ManifestDatasetError(
            f"Resolved sample view names observations absent from the observation table: "
            f"{missing_observations!r}."
        )
    ordered = observation_by_id.loc[split_view["snip_id"].tolist()].reset_index(
        drop=True
    )
    split_disagreement = ordered.loc[~ordered["split"].eq(split), ["snip_id", "split"]]
    if not split_disagreement.empty:
        raise ManifestDatasetError(
            f"Resolved sample view split={split!r} disagrees with observation rows: "
            f"{split_disagreement.to_dict('records')!r}."
        )
    return ordered, split_view


def _validate_observation_uniqueness(observation_table: pd.DataFrame) -> None:
    duplicate_mask = observation_table["snip_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = observation_table.loc[duplicate_mask, "snip_id"].tolist()
        raise ManifestDatasetError(
            f"Contract-v2 observation table contains duplicate snip_id values: {duplicates!r}."
        )


def _validate_image_tensor(tensor: object, *, snip_id: str, path: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise AssetDecodeError(
            f"snip_id={snip_id!r} asset={path!r} transform returned {type(tensor)!r}, "
            "expected torch.Tensor."
        )
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise AssetDecodeError(
            f"snip_id={snip_id!r} asset={path!r} produced shape {tuple(tensor.shape)!r}; "
            "expected grayscale [1, H, W]."
        )
    if tensor.dtype != torch.float32:
        raise AssetDecodeError(
            f"snip_id={snip_id!r} asset={path!r} produced dtype={tensor.dtype}; "
            "expected torch.float32."
        )
    if not torch.isfinite(tensor).all():
        raise AssetDecodeError(
            f"snip_id={snip_id!r} asset={path!r} produced non-finite pixels."
        )
    minimum = float(tensor.min())
    maximum = float(tensor.max())
    if minimum < 0.0 or maximum > 1.0:
        raise AssetDecodeError(
            f"snip_id={snip_id!r} asset={path!r} produced pixel range "
            f"[{minimum}, {maximum}], expected [0, 1]."
        )
    return tensor


def collate_manifest_dataset_output(batch: Sequence[DatasetOutput]) -> DatasetOutput:
    """Collate tensors while preserving nullable and diagnostic metadata exactly."""

    if not batch:
        raise ValueError("Cannot collate an empty manifest batch.")
    keys = tuple(batch[0].keys())
    if any(tuple(item.keys()) != keys for item in batch[1:]):
        raise ManifestDatasetError(
            "Manifest items in one batch do not expose identical keys."
        )

    collated: dict[str, object] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if key in {"observation_metadata", "asset_metadata"}:
            collated[key] = values
            continue
        if any(value is None for value in values) or any(
            isinstance(value, (str, Path, dict)) for value in values
        ):
            collated[key] = [
                str(value) if isinstance(value, Path) else value for value in values
            ]
            continue
        try:
            collated[key] = default_collate(values)
        except (TypeError, RuntimeError):
            collated[key] = values
    return DatasetOutput(**collated)


class ManifestDataset(Dataset):
    """One exact selected asset per contract-v2 observation in one named split."""

    collate_fn = staticmethod(collate_manifest_dataset_output)

    def __init__(
        self,
        observation_table: pd.DataFrame,
        asset_table: pd.DataFrame,
        *,
        split: str,
        snip_product_key: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        target_size: Sequence[int] = DEFAULT_TARGET_SIZE,
        resolved_sample_view: pd.DataFrame | None = None,
        max_decode_failures: int = 10,
    ) -> None:
        if not isinstance(observation_table, pd.DataFrame):
            raise TypeError(
                "observation_table must be a pandas.DataFrame matching contract v2."
            )
        if not isinstance(asset_table, pd.DataFrame):
            raise TypeError(
                "asset_table must be a pandas.DataFrame matching contract v2."
            )
        if resolved_sample_view is not None and not isinstance(
            resolved_sample_view, pd.DataFrame
        ):
            raise TypeError(
                "resolved_sample_view must be a pandas.DataFrame when provided."
            )
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty configured name.")
        if max_decode_failures < 0:
            raise ValueError("max_decode_failures must be non-negative.")

        _require_columns(
            observation_table,
            REQUIRED_OBSERVATION_COLUMNS,
            table_name="observation table",
        )
        _require_columns(asset_table, REQUIRED_ASSET_COLUMNS, table_name="asset table")
        _require_opaque_string_ids(
            observation_table,
            ("snip_id", "physical_embryo_id"),
            table_name="observation table",
        )
        _require_opaque_string_ids(asset_table, ("snip_id",), table_name="asset table")
        _validate_observation_uniqueness(observation_table)
        _validate_observation_splits(observation_table)

        split_observations, split_sample_view = _split_observations_in_resolved_order(
            observation_table,
            split=split,
            resolved_sample_view=resolved_sample_view,
            snip_product_key=snip_product_key,
        )
        if split_observations.empty:
            raise ManifestDatasetError(
                f"Configured split={split!r} contains no observations."
            )

        resolved_selector = VanillaBFProjectionSelector(snip_product_key)
        groups = resolve_asset_row_groups(
            split_observations, asset_table, resolved_selector
        )
        non_vanilla = [
            (group.snip_id, len(group.asset_row_positions))
            for group in groups
            if len(group.asset_row_positions) != 1
        ]
        if non_vanilla:
            raise ManifestDatasetError(
                "ManifestDataset requires exactly one selected asset per observation; "
                f"selector={resolved_selector.name!r} returned {non_vanilla!r}. Future ordered "
                "asset groups must use a z-aware dataset chosen in Track O3."
            )

        selected_positions = [group.asset_row_positions[0] for group in groups]
        selected_assets = asset_table.iloc[selected_positions].reset_index(drop=True)
        selected_references = (
            selected_assets["asset_row_index"].to_numpy(dtype=np.int64, copy=True)
            if "asset_row_index" in selected_assets
            else np.asarray(selected_positions, dtype=np.int64)
        )
        if split_sample_view is not None:
            expected_references = split_sample_view["asset_row_index"].to_numpy(
                dtype=np.int64, copy=True
            )
            if not np.array_equal(selected_references, expected_references):
                raise ManifestDatasetError(
                    f"Vanilla selector asset references disagree with the resolved sample-view "
                    f"order for split={split!r}: selector={selected_references.tolist()!r}, "
                    f"view={expected_references.tolist()!r}."
                )
        selected_snip_ids = selected_assets["snip_id"].tolist()
        observation_snip_ids = split_observations["snip_id"].tolist()
        if selected_snip_ids != observation_snip_ids:
            raise ManifestDatasetError(
                "Resolved asset order disagrees with observation order: "
                f"observations={observation_snip_ids!r}, assets={selected_snip_ids!r}."
            )

        self.split = split
        self.selector_name = resolved_selector.name
        self.snip_product_key = snip_product_key
        self.transform = transform or basic_transform(target_size=target_size)
        self.max_decode_failures = int(max_decode_failures)
        self._observations = _CompactTable(split_observations)
        self._assets = _CompactTable(selected_assets)
        self._selected_asset_row_positions = selected_references
        self.physical_embryo_ids = frozenset(
            split_observations["physical_embryo_id"].tolist()
        )

        # Shared fixed-capacity storage keeps failure counts/IDs visible when Linux
        # DataLoader workers fork this compact dataset. The threshold breach itself
        # occupies the final slot so the offending asset remains reportable.
        failure_capacity = max(1, self.max_decode_failures + 1)
        self._decode_failure_count = mp.Value("q", 0)
        self._decode_failure_positions = mp.RawArray("q", failure_capacity)
        self._decode_failure_capacity = failure_capacity

    def __len__(self) -> int:
        return len(self._observations)

    @classmethod
    def from_manifest_result(
        cls,
        manifest: ManifestResult,
        *,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        target_size: Sequence[int] = DEFAULT_TARGET_SIZE,
        max_decode_failures: int = 10,
        **kwargs: object,
    ) -> "ManifestDataset":
        """Construct from A1's named ``ManifestResult`` public boundary."""

        z_mode = manifest.policy.assets.z_mode
        if z_mode != "projection_null":
            raise ManifestDatasetError(
                f"Vanilla ManifestDataset requires policy.assets.z_mode='projection_null', "
                f"got {z_mode!r}."
            )
        return cls(
            manifest.observation_table,
            manifest.asset_table,
            split=split,
            snip_product_key=manifest.policy.assets.vanilla_product_key,
            transform=transform,
            target_size=target_size,
            resolved_sample_view=manifest.selected_sample_view,
            max_decode_failures=max_decode_failures,
            **kwargs,
        )

    @property
    def snip_ids(self) -> tuple[str, ...]:
        return tuple(
            self._observations.value("snip_id", index) for index in range(len(self))
        )

    @property
    def decode_failure_count(self) -> int:
        return int(self._decode_failure_count.value)

    @property
    def decode_failure_records(self) -> tuple[DecodeFailureRecord, ...]:
        recorded = min(self.decode_failure_count, self._decode_failure_capacity)
        return tuple(
            self._failure_record(int(self._decode_failure_positions[index]))
            for index in range(recorded)
        )

    def reset_decode_failures(self) -> None:
        with self._decode_failure_count.get_lock():
            self._decode_failure_count.value = 0

    def _failure_record(self, index: int) -> DecodeFailureRecord:
        snip_id = self._observations.value("snip_id", index)
        assert isinstance(snip_id, str)
        z_index = normalize_nullable_z_index(
            self._assets.value("z_index", index), snip_id=snip_id
        )
        return DecodeFailureRecord(
            split=self.split,
            snip_id=snip_id,
            snip_product_key=str(self._assets.value("snip_product_key", index)),
            z_index=z_index,
            processed_snip_path=str(self._assets.value("processed_snip_path", index)),
        )

    def _record_decode_failure(self, index: int, cause: BaseException) -> None:
        with self._decode_failure_count.get_lock():
            position = int(self._decode_failure_count.value)
            self._decode_failure_count.value += 1
            count = int(self._decode_failure_count.value)
            if position < self._decode_failure_capacity:
                self._decode_failure_positions[position] = index

        record = self._failure_record(index)
        if count > self.max_decode_failures:
            raise DecodeFailureLimitError(
                f"Decode failure threshold exceeded for split={self.split!r}: count={count}, "
                f"max_decode_failures={self.max_decode_failures}; snip_id={record.snip_id!r}, "
                f"asset={record.processed_snip_path!r}, product={record.snip_product_key!r}, "
                f"z_index={record.z_index!r}. Last error: {cause}"
            ) from cause

    def _decode_at(self, index: int) -> torch.Tensor:
        snip_id = self._observations.value("snip_id", index)
        assert isinstance(snip_id, str)
        path = str(self._assets.value("processed_snip_path", index))
        try:
            with Image.open(path) as image:
                image.load()
                decoded = image.copy()
            tensor = self.transform(decoded)
            return _validate_image_tensor(tensor, snip_id=snip_id, path=path)
        except AssetDecodeError:
            raise
        except (OSError, SyntaxError, ValueError) as exc:
            raise AssetDecodeError(
                f"Failed to decode selected asset for snip_id={snip_id!r}, split={self.split!r}, "
                f"path={path!r}: {exc}"
            ) from exc

    def _load_with_recovery(self, requested_index: int) -> tuple[torch.Tensor, int]:
        if requested_index < 0:
            requested_index += len(self)
        if requested_index < 0 or requested_index >= len(self):
            raise IndexError(requested_index)

        for offset in range(len(self)):
            candidate_index = (requested_index + offset) % len(self)
            try:
                return self._decode_at(candidate_index), candidate_index
            except AssetDecodeError as exc:
                self._record_decode_failure(candidate_index, exc)

        records = self.decode_failure_records
        raise DecodeFailureLimitError(
            f"No decodable assets remain in split={self.split!r}; attempted snip IDs "
            f"{[record.snip_id for record in records]!r}."
        )

    def _payload(
        self, index: int, requested_index: int, data: torch.Tensor
    ) -> DatasetOutput:
        observation = self._observations.row(index)
        asset = self._assets.row(index)
        snip_id = observation["snip_id"]
        assert isinstance(snip_id, str)
        z_index = normalize_nullable_z_index(asset["z_index"], snip_id=snip_id)

        payload: dict[str, object] = {}
        for key, value in observation.items():
            payload.setdefault(key, value)
        for key, value in asset.items():
            payload.setdefault(key, value)
        payload.update(
            {
                "data": data,
                "label": str(asset["processed_snip_path"]),
                "index": index,
                "requested_index": requested_index,
                "snip_id": snip_id,
                "physical_embryo_id": observation["physical_embryo_id"],
                "snip_product_key": str(asset["snip_product_key"]),
                "z_index": z_index,
                "split": self.split,
                "incubation_temperature_c": observation["incubation_temperature_c"],
                "temperature_status": observation["temperature_status"],
                "elapsed_time_s": observation["elapsed_time_s"],
                "elapsed_time_status": observation["elapsed_time_status"],
                "time_index": observation["time_index"],
                "predicted_stage_hpf": observation["predicted_stage_hpf"],
                "stage_status": observation["stage_status"],
                "stage_model_version": observation["stage_model_version"],
                "processed_snip_path": str(asset["processed_snip_path"]),
                "asset_row_reference": int(self._selected_asset_row_positions[index]),
                "observation_metadata": observation,
                "asset_metadata": asset,
            }
        )
        return DatasetOutput(**payload)

    def __getitem__(self, index: int) -> DatasetOutput:
        tensor, resolved_index = self._load_with_recovery(index)
        return self._payload(resolved_index, index, tensor)


class ManifestPairedDataset(ManifestDataset):
    """Minimal explicit, test-only paired path retained for Track A4 plumbing.

    Pair membership is supplied as an opaque-ID mapping. This class does not infer
    relations, scan candidates, or define scientific pairing semantics.
    """

    def __init__(
        self,
        *args: object,
        pair_snip_ids: Mapping[str, str],
        metric_stats_columns: tuple[str, str, str],
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if len(metric_stats_columns) != 3:
            raise ValueError(
                "metric_stats_columns must name the three compatibility fields for "
                "[embryo, stage, metric_group]."
            )
        missing_metric_columns = [
            column
            for column in metric_stats_columns
            if column not in self._observations.columns
        ]
        if missing_metric_columns:
            raise ManifestDatasetError(
                f"Test-only paired dataset is missing metric compatibility columns: "
                f"{missing_metric_columns!r}."
            )

        index_by_snip_id = {
            snip_id: index for index, snip_id in enumerate(self.snip_ids)
        }
        pair_indices = np.empty(len(self), dtype=np.int64)
        for index, snip_id in enumerate(self.snip_ids):
            if snip_id not in pair_snip_ids:
                raise ManifestDatasetError(
                    f"Explicit test-only pair mapping has no entry for snip_id={snip_id!r}, "
                    f"split={self.split!r}."
                )
            paired_snip_id = pair_snip_ids[snip_id]
            if not isinstance(paired_snip_id, str) or not paired_snip_id:
                raise ManifestDatasetError(
                    f"Explicit pair for snip_id={snip_id!r} has non-string or empty "
                    f"paired_snip_id={paired_snip_id!r}; IDs are opaque strings."
                )
            if paired_snip_id not in index_by_snip_id:
                raise ManifestDatasetError(
                    f"Explicit pair for snip_id={snip_id!r} names paired_snip_id="
                    f"{paired_snip_id!r}, which is absent from split={self.split!r}."
                )
            pair_indices[index] = index_by_snip_id[paired_snip_id]
        self._pair_indices = pair_indices
        self.metric_stats_columns = tuple(metric_stats_columns)

    def _metric_stats(self, index: int) -> list[int | float]:
        values = [
            self._observations.value(column, index)
            for column in self.metric_stats_columns
        ]
        if any(
            value is None
            or not isinstance(value, Real)
            or isinstance(value, (bool, np.bool_))
            for value in values
        ):
            snip_id = self._observations.value("snip_id", index)
            raise ManifestDatasetError(
                f"Metric compatibility fields for snip_id={snip_id!r} must be three finite "
                f"numeric values; columns={self.metric_stats_columns!r}, values={values!r}."
            )
        if not all(np.isfinite(float(value)) for value in values):
            raise ManifestDatasetError(
                f"Metric compatibility fields contain non-finite values: {values!r}."
            )
        return [value for value in values if isinstance(value, (int, float))]

    def __getitem__(self, index: int) -> DatasetOutput:
        anchor_tensor, anchor_index = self._load_with_recovery(index)
        pair_requested_index = int(self._pair_indices[anchor_index])
        pair_tensor, pair_index = self._load_with_recovery(pair_requested_index)

        output = self._payload(
            anchor_index,
            index,
            torch.stack([anchor_tensor, pair_tensor], dim=0),
        )
        output.label = [
            str(self._assets.value("processed_snip_path", anchor_index)),
            str(self._assets.value("processed_snip_path", pair_index)),
        ]
        output.index = [anchor_index, pair_index]
        output.self_stats = self._metric_stats(anchor_index)
        output.other_stats = self._metric_stats(pair_index)
        output.other_snip_id = self._observations.value("snip_id", pair_index)
        output.other_physical_embryo_id = self._observations.value(
            "physical_embryo_id", pair_index
        )
        output.other_processed_snip_path = str(
            self._assets.value("processed_snip_path", pair_index)
        )
        return output


# Compatibility import names. These are plain manifest-backed Dataset classes;
# no legacy root/ImageFolder constructor remains behind the names.
BasicDataset = ManifestDataset
BasicEvalDataset = ManifestDataset
NTXentDataset = ManifestPairedDataset
