"""Plain datasets over the manifest adapter's resolved sample view.

There is intentionally no directory discovery or legacy positional metadata path
in this module.  Construction compacts a deterministic resolved DataFrame once;
workers retain arrays/category codes and open only each selected row's declared
asset path.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from pythae.data.datasets import DatasetOutput
import torch
from torch.utils.data import Dataset, get_worker_info
from torch.utils.data._utils.collate import default_collate

from src.core.data.asset_selection import validate_vanilla_resolved_view
from src.core.data.data_transforms import basic_transform, contrastive_transform
from src.core.metric.pairing import MetricPairSampler, PairSelection


REQUIRED_RESOLVED_COLUMNS = frozenset(
    {
        "snip_id",
        "physical_embryo_id",
        "snip_product_key",
        "z_index",
        "split",
        "processed_snip_path",
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
SPLIT_NAMES = frozenset({"train", "eval", "test"})
LOGGER = logging.getLogger(__name__)


class ManifestDecodeError(RuntimeError):
    """Decode failure threshold or same-split replacement exhaustion."""


def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _python_scalar(value: Any) -> Any:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class _CompactColumn:
    values: Any
    valid: np.ndarray | None = None
    categories: tuple[Any, ...] | None = None

    def get(self, index: int) -> Any:
        if self.valid is not None and not bool(self.valid[index]):
            return None
        if self.categories is not None:
            code = int(self.values[index])
            return None if code < 0 else self.categories[code]
        value = self.values[index]
        return _python_scalar(value)

    @property
    def storage_kind(self) -> str:
        if self.categories is not None:
            return "categorical_codes"
        if isinstance(self.values, np.ndarray):
            return f"array:{self.values.dtype}"
        return "immutable_values"


def _compact_column(series: pd.Series) -> _CompactColumn:
    raw = [_python_scalar(value) for value in series.tolist()]
    valid = np.asarray([value is not None for value in raw], dtype=np.bool_)
    non_null = [value for value in raw if value is not None]

    if non_null and all(isinstance(value, (bool, np.bool_)) for value in non_null):
        values = np.asarray([False if value is None else bool(value) for value in raw], dtype=np.bool_)
        return _CompactColumn(values=values, valid=None if valid.all() else valid)

    if non_null and all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        for value in non_null
    ):
        values = np.asarray([0 if value is None else int(value) for value in raw], dtype=np.int64)
        return _CompactColumn(values=values, valid=None if valid.all() else valid)

    if non_null and all(
        isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
        for value in non_null
    ):
        values = np.asarray([0.0 if value is None else float(value) for value in raw], dtype=np.float64)
        return _CompactColumn(values=values, valid=None if valid.all() else valid)

    if all(value is None or isinstance(value, str) for value in raw):
        categories: list[str] = []
        category_codes: dict[str, int] = {}
        codes = np.full(len(raw), -1, dtype=np.int32)
        for index, value in enumerate(raw):
            if value is None:
                continue
            if value not in category_codes:
                category_codes[value] = len(categories)
                categories.append(value)
            codes[index] = category_codes[value]
        return _CompactColumn(values=codes, categories=tuple(categories))

    return _CompactColumn(values=tuple(raw))


class _CompactResolvedRows:
    """DataFrame-free, worker-facing representation of resolved rows."""

    def __init__(self, table: pd.DataFrame) -> None:
        self._length = len(table)
        self._columns = {
            str(column): _compact_column(table[column])
            for column in table.columns
        }

    def __len__(self) -> int:
        return self._length

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def get(self, column: str, index: int) -> Any:
        return self._columns[column].get(index)

    def row(self, index: int) -> dict[str, Any]:
        return {column: values.get(index) for column, values in self._columns.items()}

    def storage_kind(self, column: str) -> str:
        return self._columns[column].storage_kind


class DecodeFailureTracker:
    """Small shared counter/index buffer suitable for DataLoader worker copies."""

    def __init__(self, max_decode_failures: int) -> None:
        self._capacity = max(1, max_decode_failures + 1)
        self._count = mp.Value("q", 0)
        self._row_references = mp.Array("q", [-1] * self._capacity)

    def record(self, resolved_row_reference: int) -> int:
        with self._count.get_lock():
            failure_number = int(self._count.value)
            self._count.value += 1
            count = int(self._count.value)
        if failure_number < self._capacity:
            with self._row_references.get_lock():
                self._row_references[failure_number] = int(resolved_row_reference)
        return count

    @property
    def count(self) -> int:
        return int(self._count.value)

    def row_references(self) -> tuple[int, ...]:
        length = min(self.count, self._capacity)
        with self._row_references.get_lock():
            return tuple(int(value) for value in self._row_references[:length] if value >= 0)


def _validate_input_dim(input_dim: Sequence[int]) -> tuple[int, int, int]:
    dimensions = tuple(input_dim)
    if len(dimensions) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions
    ):
        raise ValueError(f"input_dim must be positive integer (channels, height, width), got {input_dim!r}.")
    if dimensions[0] != 1:
        raise ValueError(
            f"Track A manifest datasets support one grayscale channel, got input_dim={dimensions!r}."
        )
    return dimensions


class BasicDataset(Dataset):
    """One deterministic grayscale tensor per adapter-resolved observation."""

    def __init__(
        self,
        resolved_sample_table: pd.DataFrame,
        product_key: str,
        input_dim: Sequence[int] = (1, 288, 128),
        split: str | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        max_decode_failures: int = 10,
        expected_observation_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(resolved_sample_table, pd.DataFrame):
            raise TypeError(
                "BasicDataset requires the adapter's resolved_sample_table pandas DataFrame; "
                f"got {type(resolved_sample_table)!r}."
            )
        missing_columns = sorted(REQUIRED_RESOLVED_COLUMNS - set(resolved_sample_table.columns))
        if missing_columns:
            raise ValueError(f"Resolved sample table is missing dataset columns: {missing_columns}.")
        if split is not None and split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {sorted(SPLIT_NAMES)}, got {split!r}.")
        if isinstance(max_decode_failures, bool) or not isinstance(max_decode_failures, int):
            raise TypeError("max_decode_failures must be a non-negative integer.")
        if max_decode_failures < 0:
            raise ValueError("max_decode_failures must be a non-negative integer.")

        self.input_dim = _validate_input_dim(input_dim)
        self.product_key = product_key
        self.split = split
        self.max_decode_failures = max_decode_failures
        self.transform = transform or basic_transform(self.input_dim[1:])

        split_values = resolved_sample_table["split"].map(str)
        invalid_splits = sorted(set(split_values) - SPLIT_NAMES)
        if invalid_splits:
            raise ValueError(f"Resolved sample table has invalid split values: {invalid_splits}.")
        for identity_column in ("snip_id", "physical_embryo_id"):
            crossing = (
                resolved_sample_table.assign(_split=split_values)
                .groupby(identity_column, sort=False, dropna=False)["_split"]
                .nunique()
            )
            crossing_ids = crossing[crossing > 1].index.tolist()
            if crossing_ids:
                raise ValueError(
                    f"Resolved sample table crosses splits at {identity_column}={crossing_ids}."
                )

        if split is None:
            selected_positions = np.arange(len(resolved_sample_table), dtype=np.int64)
        else:
            selected_positions = np.flatnonzero(split_values.eq(split).to_numpy()).astype(np.int64)
        if len(selected_positions) == 0:
            raise ValueError(f"Resolved sample view has no rows for required split={split!r}.")

        selected_table = resolved_sample_table.iloc[selected_positions].reset_index(drop=True)
        selected_expected_ids = expected_observation_ids
        if expected_observation_ids is not None and split is not None:
            selected_expected_ids = [
                snip_id
                for snip_id in expected_observation_ids
                if snip_id in set(selected_table["snip_id"].map(str))
            ]
        validate_vanilla_resolved_view(
            selected_table,
            product_key=product_key,
            expected_observation_ids=selected_expected_ids,
        )

        self._rows = _CompactResolvedRows(selected_table)
        self._resolved_row_references = selected_positions
        self._decode_failures = DecodeFailureTracker(max_decode_failures)
        self._split_indices: dict[str, tuple[int, ...]] = {}
        self._index_within_split: dict[int, int] = {}
        for dataset_index in range(len(self._rows)):
            row_split = str(self._rows.get("split", dataset_index))
            self._split_indices.setdefault(row_split, tuple())
        for row_split in tuple(self._split_indices):
            indices = tuple(
                index for index in range(len(self._rows))
                if self._rows.get("split", index) == row_split
            )
            self._split_indices[row_split] = indices
            self._index_within_split.update({index: position for position, index in enumerate(indices)})

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def metadata_columns(self) -> tuple[str, ...]:
        return self._rows.column_names

    def worker_storage_kind(self, column: str) -> str:
        """Expose compact storage strategy for diagnostics/tests."""

        return self._rows.storage_kind(column)

    @property
    def decode_failure_count(self) -> int:
        return self._decode_failures.count

    def decode_failure_report(self) -> dict[str, Any]:
        failures = []
        for resolved_reference in self._decode_failures.row_references():
            matching = np.flatnonzero(self._resolved_row_references == resolved_reference)
            if not len(matching):
                continue
            index = int(matching[0])
            failures.append(
                {
                    "resolved_row_reference": resolved_reference,
                    "snip_id": self._rows.get("snip_id", index),
                    "snip_product_key": self._rows.get("snip_product_key", index),
                    "z_index": self._rows.get("z_index", index),
                    "processed_snip_path": self._rows.get("processed_snip_path", index),
                    "split": self._rows.get("split", index),
                }
            )
        return {"count": self.decode_failure_count, "failed_assets": failures}

    def _decode(self, index: int) -> torch.Tensor:
        path = self._rows.get("processed_snip_path", index)
        with Image.open(path) as image:
            tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Image transform returned {type(tensor)!r} for snip_id={self._rows.get('snip_id', index)!r}."
            )
        tensor = tensor.to(dtype=torch.float32)
        if tuple(tensor.shape) != self.input_dim:
            raise ValueError(
                f"Image tensor shape {tuple(tensor.shape)!r} does not match input_dim={self.input_dim!r} "
                f"for snip_id={self._rows.get('snip_id', index)!r}, path={path!r}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"Image tensor contains non-finite values for snip_id={self._rows.get('snip_id', index)!r}, "
                f"path={path!r}."
            )
        if tensor.numel() and (float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0):
            raise ValueError(
                f"Image tensor is outside [0,1] for snip_id={self._rows.get('snip_id', index)!r}, path={path!r}."
            )
        return tensor

    def _replacement_index(self, index: int, attempt: int) -> int | None:
        row_split = self._rows.get("split", index)
        candidates = self._split_indices[row_split]
        if attempt >= len(candidates):
            return None
        position = self._index_within_split[index]
        return candidates[(position + attempt) % len(candidates)]

    def _load_with_same_split_resampling(self, requested_index: int) -> tuple[torch.Tensor, int]:
        current_index = requested_index
        attempted: list[int] = []
        while current_index not in attempted:
            attempted.append(current_index)
            try:
                return self._decode(current_index), current_index
            except OSError as error:
                resolved_reference = int(self._resolved_row_references[current_index])
                failure_count = self._decode_failures.record(resolved_reference)
                snip_id = self._rows.get("snip_id", current_index)
                path = self._rows.get("processed_snip_path", current_index)
                row_split = self._rows.get("split", current_index)
                LOGGER.warning(
                    "Failed to decode resolved asset: snip_id=%r product=%r split=%r path=%r "
                    "failure_count=%d threshold=%d error=%s",
                    snip_id,
                    self.product_key,
                    row_split,
                    path,
                    failure_count,
                    self.max_decode_failures,
                    error,
                )
                if failure_count > self.max_decode_failures:
                    raise ManifestDecodeError(
                        "Decode failure threshold exceeded: "
                        f"count={failure_count}, threshold={self.max_decode_failures}, "
                        f"snip_id={snip_id!r}, product={self.product_key!r}, split={row_split!r}, "
                        f"path={path!r}."
                    ) from error
                replacement = self._replacement_index(requested_index, len(attempted))
                if replacement is None or replacement in attempted:
                    raise ManifestDecodeError(
                        "No decodable same-split replacement remains for "
                        f"snip_id={snip_id!r}, product={self.product_key!r}, split={row_split!r}, "
                        f"failed_path={path!r}, attempted_dataset_indices={attempted}."
                    ) from error
                current_index = replacement

        raise AssertionError("same-split decode retry loop terminated unexpectedly")

    def _item_from_loaded(
        self,
        tensor: torch.Tensor,
        actual_index: int,
        requested_index: int,
    ) -> DatasetOutput:
        metadata = self._rows.row(actual_index)
        snip_id = metadata["snip_id"]
        product_key = metadata["snip_product_key"]
        z_index = metadata["z_index"]
        output = DatasetOutput(
            data=tensor,
            label=snip_id,
            index=actual_index,
            requested_index=requested_index,
            snip_id=snip_id,
            physical_embryo_id=metadata["physical_embryo_id"],
            snip_product_key=product_key,
            z_index=z_index,
            asset_key=(snip_id, product_key, z_index),
            split=metadata["split"],
            incubation_temperature_c=metadata["incubation_temperature_c"],
            temperature_status=metadata["temperature_status"],
            elapsed_time_s=metadata["elapsed_time_s"],
            elapsed_time_status=metadata["elapsed_time_status"],
            time_index=metadata["time_index"],
            predicted_stage_hpf=metadata["predicted_stage_hpf"],
            stage_status=metadata["stage_status"],
            stage_model_version=metadata["stage_model_version"],
            processed_snip_path=metadata["processed_snip_path"],
            resolved_row_reference=int(self._resolved_row_references[actual_index]),
            metadata=metadata,
        )
        if "asset_row_reference" in metadata:
            output["asset_row_reference"] = metadata["asset_row_reference"]
        if "embryo_mask_snip_path" in metadata:
            output["embryo_mask_snip_path"] = metadata["embryo_mask_snip_path"]
        if "start_age_hpf" in metadata:
            output["start_age_hpf"] = metadata["start_age_hpf"]
        if "start_age_source" in metadata:
            output["start_age_source"] = metadata["start_age_source"]
        return output

    def __getitem__(self, index: int) -> DatasetOutput:
        if not 0 <= index < len(self):
            raise IndexError(index)
        tensor, actual_index = self._load_with_same_split_resampling(index)
        return self._item_from_loaded(tensor, actual_index, index)


class BasicEvalDataset(BasicDataset):
    """Named compatibility surface for evaluation over a resolved view."""


class NTXentDataset(BasicDataset):
    """Paired views from either the A4 static stub or C2's indexed sampler."""

    def __init__(
        self,
        resolved_sample_table: pd.DataFrame,
        product_key: str,
        pair_row_references: Sequence[int] | None = None,
        *,
        policy_name: str | None = None,
        scientific_policy: bool | None = None,
        pair_sampler: MetricPairSampler | None = None,
        distributed_rank: int = 0,
        stats_columns: tuple[str, str, str] | None = None,
        input_dim: Sequence[int] = (1, 288, 128),
        split: str | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        max_decode_failures: int = 10,
        expected_observation_ids: Sequence[str] | None = None,
    ) -> None:
        if (pair_row_references is None) == (pair_sampler is None):
            raise ValueError(
                "NTXentDataset requires exactly one pair source: pair_row_references for the "
                "A4 static stub or pair_sampler for C2 indexed selection."
            )
        if pair_sampler is None:
            is_scientific = bool(scientific_policy)
            if policy_name is None or is_scientific or not any(
                token in policy_name.lower() for token in ("test_only", "dummy")
            ):
                raise ValueError(
                    "Static Track A pairs require a policy name containing 'test_only' or "
                    "'dummy' and scientific_policy=False."
                )
        else:
            sampler_policy = pair_sampler.policy
            if policy_name is not None and policy_name != sampler_policy.name:
                raise ValueError(
                    f"dataset policy_name={policy_name!r} disagrees with indexed pair policy "
                    f"{sampler_policy.name!r}."
                )
            if (
                scientific_policy is not None
                and bool(scientific_policy) != sampler_policy.scientific_policy
            ):
                raise ValueError(
                    "dataset scientific_policy disagrees with the indexed pairing policy."
                )
            policy_name = sampler_policy.name
        if (
            isinstance(distributed_rank, bool)
            or not isinstance(distributed_rank, int)
            or distributed_rank < 0
        ):
            raise ValueError("distributed_rank must be a non-negative integer.")
        paired_transform = transform or contrastive_transform(tuple(input_dim)[1:])
        super().__init__(
            resolved_sample_table=resolved_sample_table,
            product_key=product_key,
            input_dim=input_dim,
            split=split,
            transform=paired_transform,
            max_decode_failures=max_decode_failures,
            expected_observation_ids=expected_observation_ids,
        )
        self._pair_sampler = pair_sampler
        self._distributed_rank = distributed_rank
        if pair_sampler is None:
            assert pair_row_references is not None
            if len(pair_row_references) != len(self):
                raise ValueError(
                    f"pair_row_references length {len(pair_row_references)} does not match "
                    f"dataset length {len(self)}."
                )
            self._pair_row_references = np.asarray(pair_row_references, dtype=np.int64)
            default_stats = (
                "physical_embryo_id",
                "predicted_stage_hpf",
                "metric_group_code",
            )
        else:
            self._pair_row_references = None
            dataset_snip_ids = tuple(
                self._rows.get("snip_id", index) for index in range(len(self))
            )
            if dataset_snip_ids != pair_sampler.snip_ids:
                raise ValueError(
                    "C2 pair index row order does not match the dataset's split-local resolved "
                    "observation order."
                )
            default_stats = (
                "physical_embryo_id",
                pair_sampler.policy.stage_column,
                "metric_group",
            )
        self._stats_columns = default_stats if stats_columns is None else stats_columns
        missing_stats = [
            column for column in self._stats_columns if column not in self.metadata_columns
        ]
        if missing_stats:
            raise ValueError(f"Metric stats columns are missing: {missing_stats}.")
        if self._pair_row_references is not None:
            for anchor_index, pair_index in enumerate(self._pair_row_references.tolist()):
                if not 0 <= pair_index < len(self):
                    raise ValueError(
                        f"Pair row reference {pair_index} for anchor index {anchor_index} "
                        f"is outside dataset length {len(self)}."
                    )
                anchor_split = self._rows.get("split", anchor_index)
                pair_split = self._rows.get("split", pair_index)
                if anchor_split != pair_split:
                    raise ValueError(
                        "Test-only pair crosses splits: "
                        f"anchor snip_id={self._rows.get('snip_id', anchor_index)!r} split={anchor_split!r}, "
                        f"other snip_id={self._rows.get('snip_id', pair_index)!r} split={pair_split!r}."
                    )
        self._policy_name = policy_name

    def _stats(self, index: int) -> list[Any]:
        return [self._rows.get(column, index) for column in self._stats_columns]

    def set_epoch(self, epoch: int) -> None:
        """Advance C2's deterministic pair seed coordinate."""

        if self._pair_sampler is not None:
            self._pair_sampler.set_epoch(epoch)

    def _indexed_selection(self, anchor_index: int, *, draw_index: int = 0) -> PairSelection:
        if self._pair_sampler is None:
            raise AssertionError("indexed selection requested for a static-pair dataset")
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        return self._pair_sampler.sample(
            anchor_index,
            worker_id=worker_id,
            rank=self._distributed_rank,
            draw_index=draw_index,
        )

    def _load_indexed_other(
        self, anchor_index: int, selection: PairSelection
    ) -> tuple[torch.Tensor, int, str]:
        """Retry only legal indexed positives after a pair-image decode failure."""

        if self._pair_sampler is None:
            raise AssertionError("indexed pair loading requested for a static-pair dataset")
        tensor = self._decode_indexed_candidate(anchor_index, selection.other_index)
        if tensor is not None:
            return tensor, selection.other_index, selection.candidate_kind

        # Candidate materialization is confined to the exceptional decode-recovery
        # path; normal __getitem__ selection remains an indexed range query.
        fallback_indices = (
            index
            for index in self._pair_sampler.pair_index.candidate_indices(anchor_index)
            if index != selection.other_index
        )
        for other_index in fallback_indices:
            tensor = self._decode_indexed_candidate(anchor_index, other_index)
            if tensor is None:
                continue
            candidate_kind = (
                "same_embryo"
                if self._rows.get("physical_embryo_id", anchor_index)
                == self._rows.get("physical_embryo_id", other_index)
                else "different_embryo"
            )
            return tensor, other_index, candidate_kind
        raise ManifestDecodeError(
            "No decodable legal metric positive remains for "
            f"anchor_snip_id={self._rows.get('snip_id', anchor_index)!r}, "
            f"policy={self._policy_name!r}, split={self._rows.get('split', anchor_index)!r}."
        )

    def _decode_indexed_candidate(
        self, anchor_index: int, other_index: int
    ) -> torch.Tensor | None:
        try:
            return self._decode(other_index)
        except OSError as error:
            resolved_reference = int(self._resolved_row_references[other_index])
            failure_count = self._decode_failures.record(resolved_reference)
            snip_id = self._rows.get("snip_id", other_index)
            path = self._rows.get("processed_snip_path", other_index)
            LOGGER.warning(
                "Failed to decode indexed metric positive: anchor_snip_id=%r "
                "other_snip_id=%r product=%r split=%r path=%r failure_count=%d "
                "threshold=%d error=%s",
                self._rows.get("snip_id", anchor_index),
                snip_id,
                self.product_key,
                self._rows.get("split", other_index),
                path,
                failure_count,
                self.max_decode_failures,
                error,
            )
            if failure_count > self.max_decode_failures:
                raise ManifestDecodeError(
                    "Metric-pair decode failure threshold exceeded: "
                    f"count={failure_count}, threshold={self.max_decode_failures}, "
                    f"anchor_snip_id={self._rows.get('snip_id', anchor_index)!r}, "
                    f"other_snip_id={snip_id!r}, policy={self._policy_name!r}."
                ) from error
            return None

    def __getitem__(self, index: int) -> DatasetOutput:
        if not 0 <= index < len(self):
            raise IndexError(index)
        anchor_tensor, anchor_index = self._load_with_same_split_resampling(index)
        if self._pair_sampler is None:
            assert self._pair_row_references is not None
            pair_requested_index = int(self._pair_row_references[anchor_index])
            other_tensor, other_index = self._load_with_same_split_resampling(
                pair_requested_index
            )
            candidate_kind = (
                "same_embryo"
                if self._rows.get("physical_embryo_id", anchor_index)
                == self._rows.get("physical_embryo_id", other_index)
                else "different_embryo"
            )
            pair_seed = None
        else:
            selection = self._indexed_selection(anchor_index)
            pair_requested_index = selection.other_index
            other_tensor, other_index, candidate_kind = self._load_indexed_other(
                anchor_index, selection
            )
            pair_seed = selection.seed
        output = self._item_from_loaded(anchor_tensor, anchor_index, index)
        output["data"] = torch.stack([anchor_tensor, other_tensor], dim=0)
        output["label"] = [
            self._rows.get("snip_id", anchor_index),
            self._rows.get("snip_id", other_index),
        ]
        output["index"] = [anchor_index, other_index]
        output["self_stats"] = self._stats(anchor_index)
        output["other_stats"] = self._stats(other_index)
        output["other_snip_id"] = self._rows.get("snip_id", other_index)
        output["other_asset_key"] = (
            self._rows.get("snip_id", other_index),
            self._rows.get("snip_product_key", other_index),
            self._rows.get("z_index", other_index),
        )
        output["other_metadata"] = self._rows.row(other_index)
        output["pair_candidate_kind"] = candidate_kind
        output["pair_policy_name"] = self._policy_name
        output["pair_seed"] = pair_seed
        output["pair_requested_index"] = pair_requested_index
        if self._pair_sampler is not None:
            output["pair_sampler_age_window"] = self._pair_sampler.policy.sampler_age_window
            output["pair_stage_source"] = self._pair_sampler.policy.stage_source
        return output


def _collate_nullable(values: Sequence[Any], *, preserve_sequence: bool = False) -> Any:
    if preserve_sequence or any(value is None for value in values):
        return list(values)
    if isinstance(values[0], Mapping):
        return {
            key: _collate_nullable([value[key] for value in values])
            for key in values[0]
        }
    try:
        return default_collate(list(values))
    except (TypeError, RuntimeError):
        return list(values)


def collate_manifest_dataset_output(batch: Sequence[DatasetOutput]) -> DatasetOutput:
    """Collate DatasetOutput while preserving nullable metadata and asset keys."""

    if not batch:
        raise ValueError("Cannot collate an empty manifest batch.")
    preserve_keys = {"asset_key", "other_asset_key"}
    return DatasetOutput(
        **{
            key: _collate_nullable(
                [item[key] for item in batch],
                preserve_sequence=key in preserve_keys,
            )
            for key in batch[0]
        }
    )
