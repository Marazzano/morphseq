"""Local, reproducible provenance for manifest-backed core-model runs.

The bundle written here is deliberately independent of Hydra's working directory and
of any online logger.  W&B, when enabled, receives this exact local directory as one
artifact; local persistence is always the authority.
"""

from __future__ import annotations

import csv
import hashlib
import json
import numbers
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import yaml


PROVENANCE_FORMAT_VERSION = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_POLICY_FIELDS = {
    "experiment_ids",
    "selected_product_key",
    "z_selection_mode",
    "qc",
    "stage",
    "covariates",
}
_ALLOWED_SPLITS = {"train", "eval", "test"}


@dataclass(frozen=True)
class RunProvenanceBundle:
    """All information required to regenerate one selected run cohort."""

    resolved_config: Mapping[str, Any]
    observation_ids: Sequence[str]
    selected_asset_keys: Sequence[tuple[str, str, int | None]]
    split_assignments: Any
    policies: Any
    sources: Sequence[Any]
    cohort_report: Any
    mapping_policy: Any
    adapter_git_revision: str

    @classmethod
    def from_manifest_result(
        cls,
        *,
        manifest_result: Any,
        resolved_config: Mapping[str, Any],
        adapter_git_revision: str,
        mapping_policy: Any | None = None,
    ) -> "RunProvenanceBundle":
        """Build a bundle from A1's typed result without importing pipeline code."""

        resolved = manifest_result.resolved_sample_table
        required = {"snip_id", "snip_product_key", "z_index"}
        missing = required - set(resolved.columns)
        if missing:
            raise ValueError(
                "resolved_sample_table is missing provenance key columns: "
                f"{sorted(missing)}"
            )
        observation_ids = tuple(resolved["snip_id"].tolist())
        asset_keys = tuple(
            (
                row.snip_id,
                row.snip_product_key,
                _normalize_z_index(row.z_index),
            )
            for row in resolved[["snip_id", "snip_product_key", "z_index"]].itertuples(
                index=False
            )
        )
        if mapping_policy is None:
            mapping_policy = manifest_result.policy.metric_mapping
        return cls(
            resolved_config=resolved_config,
            observation_ids=observation_ids,
            selected_asset_keys=asset_keys,
            split_assignments=manifest_result.split_assignments,
            policies=manifest_result.policy,
            sources=manifest_result.source_inventory,
            cohort_report=manifest_result.cohort_report,
            mapping_policy=mapping_policy,
            adapter_git_revision=adapter_git_revision,
        )


@dataclass(frozen=True)
class WrittenRunProvenance:
    """Paths and hashes for a completed local provenance bundle."""

    root: Path
    files: tuple[Path, ...]
    file_sha256: Mapping[str, str]


def write_run_provenance(
    *,
    run_artifacts_dir: str | Path,
    bundle: RunProvenanceBundle,
    wandb_run: Any | None = None,
    wandb_artifact_factory: Callable[..., Any] | None = None,
    wandb_artifact_name: str = "run-provenance",
) -> WrittenRunProvenance:
    """Validate and write one complete provenance bundle.

    ``run_artifacts_dir`` must be absolute and must not already exist.  This makes
    the caller choose one stable, per-run location instead of accidentally writing
    relative to Hydra's changing working directory or overwriting another run.
    """

    root = Path(run_artifacts_dir)
    if not root.is_absolute():
        raise ValueError(
            "run_artifacts_dir must be an absolute configured path; "
            f"got {run_artifacts_dir!r}"
        )
    if root.exists():
        raise FileExistsError(f"run_artifacts_dir already exists: {root}")

    normalized = _validate_and_normalize(bundle)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()

    _write_yaml(root / "resolved_config.yaml", normalized["resolved_config"])
    _write_csv(
        root / "observation_ids.csv",
        ("order_index", "snip_id"),
        (
            {"order_index": index, "snip_id": snip_id}
            for index, snip_id in enumerate(normalized["observation_ids"])
        ),
    )
    _write_csv(
        root / "selected_asset_keys.csv",
        ("order_index", "snip_id", "snip_product_key", "z_index"),
        (
            {
                "order_index": index,
                "snip_id": key[0],
                "snip_product_key": key[1],
                "z_index": "" if key[2] is None else key[2],
            }
            for index, key in enumerate(normalized["selected_asset_keys"])
        ),
    )
    _write_csv(
        root / "split_assignments.csv",
        ("physical_embryo_id", "split"),
        normalized["split_assignments"],
    )
    _write_json(root / "policies.json", normalized["policies"])
    _write_json(root / "sources.json", normalized["sources"])
    _write_json(root / "cohort_report.json", normalized["cohort_report"])
    _write_json(root / "mapping_policy.json", normalized["mapping_policy"])
    (root / "adapter_git_revision.txt").write_text(
        normalized["adapter_git_revision"] + "\n", encoding="utf-8"
    )

    payload_files = tuple(sorted(path for path in root.iterdir() if path.is_file()))
    file_sha256 = {path.name: _sha256_file(path) for path in payload_files}
    identity_hashes = {
        "observation_ids_sha256": file_sha256["observation_ids.csv"],
        "selected_asset_keys_sha256": file_sha256["selected_asset_keys.csv"],
        "split_assignments_sha256": file_sha256["split_assignments.csv"],
    }
    _write_json(
        root / "provenance_manifest.json",
        {
            "format_version": PROVENANCE_FORMAT_VERSION,
            "files": file_sha256,
            "identity_hashes": identity_hashes,
        },
    )
    manifest_path = root / "provenance_manifest.json"
    all_files = tuple(sorted((*payload_files, manifest_path)))

    if wandb_run is not None:
        _publish_to_wandb(
            root=root,
            wandb_run=wandb_run,
            artifact_factory=wandb_artifact_factory,
            artifact_name=wandb_artifact_name,
        )

    return WrittenRunProvenance(
        root=root,
        files=all_files,
        file_sha256=file_sha256,
    )


def read_selected_identity(
    run_artifacts_dir: str | Path,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, int | None], ...]]:
    """Read ordered observation IDs and selected asset keys from a local bundle."""

    root = Path(run_artifacts_dir)
    with (root / "observation_ids.csv").open(newline="", encoding="utf-8") as handle:
        observations = tuple(row["snip_id"] for row in csv.DictReader(handle))
    with (root / "selected_asset_keys.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        asset_keys = tuple(
            (
                row["snip_id"],
                row["snip_product_key"],
                None if row["z_index"] == "" else int(row["z_index"]),
            )
            for row in csv.DictReader(handle)
        )
    return observations, asset_keys


def _validate_and_normalize(bundle: RunProvenanceBundle) -> dict[str, Any]:
    observation_ids = tuple(bundle.observation_ids)
    if not observation_ids:
        raise ValueError("provenance requires at least one ordered observation ID")
    if any(not isinstance(value, str) or not value for value in observation_ids):
        raise ValueError("observation IDs must be non-empty opaque strings")
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("ordered observation IDs contain duplicates")

    asset_keys = tuple(_normalize_asset_key(key) for key in bundle.selected_asset_keys)
    if tuple(key[0] for key in asset_keys) != observation_ids:
        raise ValueError(
            "selected asset key order does not exactly match ordered observation IDs"
        )
    if len(set(asset_keys)) != len(asset_keys):
        raise ValueError("selected asset keys contain duplicates")

    splits = _records(bundle.split_assignments)
    split_ids: set[str] = set()
    for row in splits:
        embryo_id = row.get("physical_embryo_id")
        split = row.get("split")
        if not isinstance(embryo_id, str) or not embryo_id:
            raise ValueError("split assignment has an invalid physical_embryo_id")
        if embryo_id in split_ids:
            raise ValueError(
                f"duplicate split assignment for physical_embryo_id={embryo_id!r}"
            )
        if split not in _ALLOWED_SPLITS:
            raise ValueError(
                f"physical_embryo_id={embryo_id!r} has invalid split={split!r}"
            )
        split_ids.add(embryo_id)

    policies = _json_ready(bundle.policies)
    if not isinstance(policies, dict):
        raise TypeError("policies must serialize to a mapping")
    missing_policy_fields = _REQUIRED_POLICY_FIELDS - set(policies)
    if missing_policy_fields:
        raise ValueError(
            "manifest policies missing required provenance fields: "
            f"{sorted(missing_policy_fields)}"
        )

    sources = [_json_ready(source) for source in bundle.sources]
    for source in sources:
        required = {
            "experiment_id",
            "source_name",
            "path",
            "size_bytes",
            "mtime_ns",
            "row_count",
            "sha256",
        }
        if not isinstance(source, dict) or required - set(source):
            missing = sorted(required - set(source)) if isinstance(source, dict) else sorted(required)
            raise ValueError(f"source inventory row is missing fields: {missing}")
        if not _SHA256_RE.fullmatch(str(source["sha256"])):
            raise ValueError(
                f"source {source['source_name']!r} has invalid full SHA-256: "
                f"{source['sha256']!r}"
            )

    adapter_revision = str(bundle.adapter_git_revision)
    if not _GIT_REVISION_RE.fullmatch(adapter_revision):
        raise ValueError(
            "adapter_git_revision must be a full lowercase 40-character git SHA"
        )

    resolved_config = _json_ready(bundle.resolved_config)
    if not isinstance(resolved_config, dict):
        raise TypeError("resolved_config must serialize to a mapping")
    mapping_policy = _json_ready(bundle.mapping_policy)
    if not isinstance(mapping_policy, dict):
        raise TypeError("mapping_policy must serialize to a mapping")

    return {
        "resolved_config": resolved_config,
        "observation_ids": observation_ids,
        "selected_asset_keys": asset_keys,
        "split_assignments": splits,
        "policies": policies,
        "sources": sources,
        "cohort_report": _json_ready(bundle.cohort_report),
        "mapping_policy": mapping_policy,
        "adapter_git_revision": adapter_revision,
    }


def _normalize_asset_key(value: Any) -> tuple[str, str, int | None]:
    if isinstance(value, Mapping):
        value = (value.get("snip_id"), value.get("snip_product_key"), value.get("z_index"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"asset key must have exactly three fields, got {value!r}")
    snip_id, product_key, z_index = value
    if not isinstance(snip_id, str) or not snip_id:
        raise ValueError(f"asset key has invalid snip_id: {snip_id!r}")
    if not isinstance(product_key, str) or not product_key:
        raise ValueError(f"asset key has invalid snip_product_key: {product_key!r}")
    return snip_id, product_key, _normalize_z_index(z_index)


def _normalize_z_index(value: Any) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"z_index must be null or an integer, got {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(f"z_index must be non-negative, got {value}")
    return value


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return [_json_ready(row) for row in value.to_dict(orient="records")]
    return [_json_ready(row) for row in value]


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            pass
    if value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not provenance-serializable: {type(value).__name__}")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_to_wandb(
    *,
    root: Path,
    wandb_run: Any,
    artifact_factory: Callable[..., Any] | None,
    artifact_name: str,
) -> None:
    if artifact_factory is None:
        import wandb

        artifact_factory = wandb.Artifact
    artifact = artifact_factory(
        artifact_name,
        type="run-provenance",
        metadata={"format_version": PROVENANCE_FORMAT_VERSION},
    )
    artifact.add_dir(str(root))
    wandb_run.log_artifact(artifact)
