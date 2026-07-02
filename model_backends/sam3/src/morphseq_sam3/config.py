"""Torch-free configuration helpers for the SAM3 backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import configparser
import os
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[2]
BACKEND_TOML = BACKEND_DIR / "backend.toml"


@dataclass(frozen=True)
class Sam3BackendConfig:
    backend_dir: Path
    backend_name: str
    env_name: str
    runtime_kind: str
    upstream_repo: str
    upstream_commit: str
    vendor_path: Path
    model_cache_root: Path
    artifact_root: Path
    hf_cache_dir: Path
    model_id: str
    revision: str
    checkpoint_filename: str
    checkpoint_path: Path
    setup_report_path: Path
    vendor_manifest_path: Path
    checkpoint_manifest_path: Path
    sam3_version: str
    device: str
    use_fa3: bool
    use_rope_real: bool


def load_backend_metadata(path: Path = BACKEND_TOML) -> dict[str, Any]:
    try:
        import tomllib

        with path.open("rb") as handle:
            return tomllib.load(handle)
    except ModuleNotFoundError:
        try:
            import tomli

            with path.open("rb") as handle:
                return tomli.load(handle)
        except ModuleNotFoundError:
            parser = configparser.ConfigParser()
            parser.read(path)
            return {
                section: {
                    key: _coerce_toml_scalar(value)
                    for key, value in parser.items(section)
                }
                for section in parser.sections()
            }


def _coerce_toml_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1]
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return stripped


def _expand_absolute(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute or start with '~': {value}")
    return path


def _project_root() -> Path:
    return BACKEND_DIR.parents[1]


def _default_model_cache(metadata: dict[str, Any]) -> Path:
    env_value = os.environ.get("MORPHSEQ_MODEL_CACHE")
    if env_value:
        return _expand_absolute(env_value, label="MORPHSEQ_MODEL_CACHE")
    return _expand_absolute(
        metadata["artifacts"]["default_model_cache"],
        label="default_model_cache",
    )


def _default_vendor_path(metadata: dict[str, Any]) -> Path:
    env_value = os.environ.get("MORPHSEQ_SAM3_VENDOR_PATH")
    if env_value:
        return _expand_absolute(env_value, label="MORPHSEQ_SAM3_VENDOR_PATH")
    vendor_root = _expand_absolute(
        metadata["runtime"]["default_vendor_root"],
        label="default_vendor_root",
    )
    return vendor_root / metadata["runtime"]["vendor_dir_name"]


def build_config(
    *,
    model_cache: str | Path | None = None,
    vendor_path: str | Path | None = None,
    sam3_version: str | None = None,
    device: str = "auto",
    use_fa3: bool | None = None,
    use_rope_real: bool | None = None,
) -> Sam3BackendConfig:
    metadata = load_backend_metadata()
    backend = metadata["backend"]
    artifacts = metadata["artifacts"]
    reports = metadata["reports"]

    model_cache_root = (
        _expand_absolute(model_cache, label="model_cache")
        if model_cache is not None
        else _default_model_cache(metadata)
    )
    resolved_vendor_path = (
        _expand_absolute(vendor_path, label="vendor_path")
        if vendor_path is not None
        else _default_vendor_path(metadata)
    )

    artifact_root = model_cache_root / backend["name"]
    hf_cache_dir = artifact_root / artifacts["hf_cache_subdir"]
    metadata_root = artifact_root / "metadata"
    project_root = _project_root()

    version = sam3_version or backend["default_version"]
    return Sam3BackendConfig(
        backend_dir=BACKEND_DIR,
        backend_name=backend["name"],
        env_name=backend["env"],
        runtime_kind=backend["runtime_kind"],
        upstream_repo=metadata["runtime"]["upstream_repo"],
        upstream_commit=metadata["runtime"]["upstream_commit"],
        vendor_path=resolved_vendor_path,
        model_cache_root=model_cache_root,
        artifact_root=artifact_root,
        hf_cache_dir=hf_cache_dir,
        model_id=artifacts["model_id"],
        revision=artifacts["revision"],
        checkpoint_filename=artifacts["checkpoint_filename"],
        checkpoint_path=hf_cache_dir / artifacts["checkpoint_filename"],
        setup_report_path=project_root / reports["setup_report"],
        vendor_manifest_path=metadata_root / Path(reports["vendor_manifest"]).name,
        checkpoint_manifest_path=metadata_root / Path(reports["checkpoint_manifest"]).name,
        sam3_version=version,
        device=device,
        use_fa3=backend["default_use_fa3"] if use_fa3 is None else use_fa3,
        use_rope_real=backend["default_use_rope_real"]
        if use_rope_real is None
        else use_rope_real,
    )
