"""Exact, versioned mapping from observation metadata to metric groups.

Metric-group names are the persisted identity.  Integer codes are a deterministic
runtime convenience and are deliberately derived from sorted group names rather
than observation order.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


MAPPING_FORMAT_VERSION = "1.0"


class MetricMappingError(ValueError):
    """Base error for invalid or incomplete metric mappings."""


class MetricMappingValidationError(MetricMappingError):
    """The mapping artifact itself is invalid."""


class MetricMappingCoverageError(MetricMappingError):
    """Observation metadata contains values absent from the explicit artifact."""


@dataclass(frozen=True)
class MetricMappingEntry:
    """One exact metadata tuple mapped to a canonical metric-group name."""

    source_values: tuple[Any, ...]
    metric_group: str

    def __post_init__(self) -> None:
        normalized = tuple(_normalize_artifact_value(value) for value in self.source_values)
        object.__setattr__(self, "source_values", normalized)
        if not isinstance(self.metric_group, str) or not self.metric_group.strip():
            raise MetricMappingValidationError(
                "mapping entries require a non-empty canonical metric_group"
            )


@dataclass(frozen=True)
class MetricMappingArtifact:
    """Reviewable exact mapping policy supplied to the observation mapper."""

    name: str
    version: str
    metadata_columns: tuple[str, ...]
    entries: tuple[MetricMappingEntry, ...]
    scientific_policy: bool = False
    format_version: str = MAPPING_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != MAPPING_FORMAT_VERSION:
            raise MetricMappingValidationError(
                f"unsupported metric mapping format_version={self.format_version!r}; "
                f"expected {MAPPING_FORMAT_VERSION!r}"
            )
        for field_name, value in (("name", self.name), ("version", self.version)):
            if not isinstance(value, str) or not value.strip():
                raise MetricMappingValidationError(
                    f"metric mapping {field_name} must be a non-empty string"
                )
        columns = tuple(self.metadata_columns)
        object.__setattr__(self, "metadata_columns", columns)
        if not columns or any(not isinstance(column, str) or not column for column in columns):
            raise MetricMappingValidationError(
                "metric mapping metadata_columns must contain non-empty column names"
            )
        if len(set(columns)) != len(columns):
            raise MetricMappingValidationError(
                f"metric mapping metadata_columns contain duplicates: {columns!r}"
            )

        entries = tuple(self.entries)
        if not entries:
            raise MetricMappingValidationError("metric mapping artifact has no entries")
        seen: dict[tuple[tuple[str, Any], ...], MetricMappingEntry] = {}
        for entry in entries:
            if not isinstance(entry, MetricMappingEntry):
                raise MetricMappingValidationError(
                    "metric mapping entries must be MetricMappingEntry instances"
                )
            if len(entry.source_values) != len(columns):
                raise MetricMappingValidationError(
                    f"mapping entry {entry.source_values!r} has {len(entry.source_values)} "
                    f"values for metadata_columns={columns!r}"
                )
            key = _source_key(entry.source_values)
            if key in seen:
                raise MetricMappingValidationError(
                    "duplicate exact metadata tuple in metric mapping: "
                    f"columns={columns!r}, values={entry.source_values!r}"
                )
            seen[key] = entry
        entries = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    _source_sort_key(entry.source_values),
                    entry.metric_group,
                ),
            )
        )
        object.__setattr__(self, "entries", entries)
        if not isinstance(self.scientific_policy, bool):
            raise MetricMappingValidationError("scientific_policy must be a boolean")

    @property
    def group_names(self) -> tuple[str, ...]:
        """Canonical persisted group identities in deterministic order."""

        return tuple(sorted({entry.metric_group for entry in self.entries}))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-ready representation."""

        entries = sorted(
            self.entries,
            key=lambda entry: (_source_sort_key(entry.source_values), entry.metric_group),
        )
        return {
            "format_version": self.format_version,
            "name": self.name,
            "version": self.version,
            "scientific_policy": self.scientific_policy,
            "metadata_columns": list(self.metadata_columns),
            "entries": [
                {
                    "source_values": list(entry.source_values),
                    "metric_group": entry.metric_group,
                }
                for entry in entries
            ],
        }

    def to_json(self) -> str:
        """Serialize deterministically for review and provenance."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricMappingArtifact":
        try:
            raw_entries = payload["entries"]
            entries = tuple(
                MetricMappingEntry(
                    source_values=tuple(entry["source_values"]),
                    metric_group=entry["metric_group"],
                )
                for entry in raw_entries
            )
            return cls(
                format_version=payload.get("format_version", MAPPING_FORMAT_VERSION),
                name=payload["name"],
                version=payload["version"],
                scientific_policy=payload.get("scientific_policy", False),
                metadata_columns=tuple(payload["metadata_columns"]),
                entries=entries,
            )
        except (KeyError, TypeError) as exc:
            raise MetricMappingValidationError(
                f"invalid metric mapping payload: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, text: str) -> "MetricMappingArtifact":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MetricMappingValidationError(
                f"invalid metric mapping JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise MetricMappingValidationError("metric mapping JSON must contain an object")
        return cls.from_dict(payload)

    @classmethod
    def read_json(cls, path: str | Path) -> "MetricMappingArtifact":
        """Load a required, explicitly selected mapping artifact."""

        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"required metric mapping artifact does not exist: {artifact_path}"
            )
        return cls.from_json(artifact_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class MappedMetricGroups:
    """Observation-aligned canonical identities and optional runtime codes."""

    group_names: tuple[str | None, ...]
    group_codes: tuple[int | None, ...]
    code_to_group: tuple[str, ...]


class CompiledMetricMapping:
    """Validated exact lookup compiled from one required mapping artifact."""

    def __init__(self, artifact: MetricMappingArtifact):
        if not isinstance(artifact, MetricMappingArtifact):
            raise TypeError("artifact must be a required MetricMappingArtifact")
        self.artifact = artifact
        self.code_to_group = artifact.group_names
        self.group_to_code = {
            group_name: code for code, group_name in enumerate(self.code_to_group)
        }
        self._source_to_group = {
            _source_key(entry.source_values): entry.metric_group
            for entry in artifact.entries
        }

    def map_records(
        self, records: Iterable[Mapping[str, Any]]
    ) -> MappedMetricGroups:
        """Map records exactly, collecting every uncovered non-null tuple before failing.

        A record whose configured metadata values are all null remains unassigned.  A
        partially null tuple still carries non-null source data and must be explicitly
        covered, so it is reported as uncovered.
        """

        mapped_names: list[str | None] = []
        uncovered: set[tuple[tuple[str, Any], ...]] = set()
        missing_columns: set[str] = set()

        for record in records:
            absent = set(self.artifact.metadata_columns) - set(record)
            if absent:
                missing_columns.update(absent)
                continue
            raw_values = tuple(record[column] for column in self.artifact.metadata_columns)
            if all(_is_null(value) for value in raw_values):
                mapped_names.append(None)
                continue
            if any(_is_non_scalar(value) for value in raw_values):
                raise MetricMappingCoverageError(
                    "configured metric metadata values must be scalar: "
                    f"columns={self.artifact.metadata_columns!r}, values={raw_values!r}"
                )
            key = _record_source_key(raw_values)
            group_name = self._source_to_group.get(key)
            if group_name is None:
                uncovered.add(key)
                mapped_names.append(None)
            else:
                mapped_names.append(group_name)

        if missing_columns:
            raise MetricMappingCoverageError(
                f"metric mapping {self.artifact.name!r} requires missing observation "
                f"metadata columns: {sorted(missing_columns)!r}"
            )
        if uncovered:
            rendered = [
                tuple("<null>" if value_type == "null" else value for value_type, value in key)
                for key in sorted(uncovered, key=_typed_key_sort_key)
            ]
            raise MetricMappingCoverageError(
                f"metric mapping {self.artifact.name!r}@{self.artifact.version} has "
                f"uncovered non-null metadata tuples for "
                f"columns={self.artifact.metadata_columns!r}: {rendered!r}"
            )

        return MappedMetricGroups(
            group_names=tuple(mapped_names),
            group_codes=tuple(
                None if name is None else self.group_to_code[name] for name in mapped_names
            ),
            code_to_group=self.code_to_group,
        )


def _normalize_artifact_value(value: Any) -> Any:
    if value is None:
        raise MetricMappingValidationError(
            "mapping artifact source_values cannot be null; null observations remain unassigned"
        )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise MetricMappingValidationError(
                f"mapping artifact source value must be finite, got {value!r}"
            )
        return int(normalized) if normalized.is_integer() else normalized
    raise MetricMappingValidationError(
        "mapping artifact source_values must be JSON scalar strings, booleans, or numbers; "
        f"got {type(value).__name__}"
    )


def _normalize_record_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        normalized = float(value)
        if math.isnan(normalized):
            return None
        if not math.isfinite(normalized):
            raise MetricMappingCoverageError(
                f"metric metadata source value must be finite, got {value!r}"
            )
        return int(normalized) if normalized.is_integer() else normalized
    if _is_null(value):
        return None
    return value


def _is_null(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result)
    except (TypeError, ValueError):
        return False


def _is_non_scalar(value: Any) -> bool:
    if isinstance(value, (str, bytes, bool, numbers.Number)) or _is_null(value):
        return False
    return isinstance(value, (Mapping, Sequence, set))


def _typed_value_key(value: Any) -> tuple[str, Any]:
    value = _normalize_record_value(value)
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, int):
        return ("number", value)
    if isinstance(value, float):
        return ("number", value)
    raise MetricMappingCoverageError(
        "metric metadata source values must be scalar strings, booleans, or numbers; "
        f"got {type(value).__name__}"
    )


def _source_key(values: Sequence[Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(_typed_value_key(value) for value in values)


def _record_source_key(values: Sequence[Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(_typed_value_key(value) for value in values)


def _source_sort_key(values: Sequence[Any]) -> str:
    return json.dumps(list(values), sort_keys=True, separators=(",", ":"))


def _typed_key_sort_key(key: tuple[tuple[str, Any], ...]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))
