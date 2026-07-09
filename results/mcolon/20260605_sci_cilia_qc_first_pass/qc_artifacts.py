"""Artifact helpers for the sequenced-only SCI cilia QC workflow."""

from __future__ import annotations

import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sci_cilia_qc_config import MODELS_DIR, PREDICTIONS_DIR, RUN_DIR


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RUN_DIR,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_model_bundle(model_id: str, model: dict, metadata: dict) -> None:
    ensure_dirs()
    bundle_path = MODELS_DIR / f"{model_id}.pkl"
    meta_path = MODELS_DIR / f"{model_id}.metadata.json"
    with bundle_path.open("wb") as fh:
        pickle.dump({"model_id": model_id, "model": model, "metadata": metadata}, fh)
    meta = dict(metadata)
    meta["model_id"] = model_id
    meta["bundle_path"] = str(bundle_path.relative_to(RUN_DIR))
    meta["created_utc"] = datetime.now(timezone.utc).isoformat()
    meta["git_commit"] = git_commit()
    meta_path.write_text(json.dumps(json_safe(meta), indent=2, sort_keys=True) + "\n")


def load_model_bundle(model_id: str) -> dict:
    with (MODELS_DIR / f"{model_id}.pkl").open("rb") as fh:
        return pickle.load(fh)


def prediction_path(name: str) -> Path:
    ensure_dirs()
    return PREDICTIONS_DIR / name


def compatibility_copy(src: Path, dst: Path) -> None:
    """Copy a generated artifact to an older path that legacy plot scripts still read."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
