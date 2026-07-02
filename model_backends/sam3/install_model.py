#!/usr/bin/env python
"""Install or verify SAM3 model artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphseq_sam3.config import build_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_checkpoint(config) -> Path:
    config.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(config.hf_cache_dir))
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=config.model_id,
            filename=config.checkpoint_filename,
            revision=config.revision,
        )
    )


def build_manifest(config, *, checkpoint_path: Path | None, ok: bool, error: Exception | None = None) -> dict[str, object]:
    manifest: dict[str, object] = {
        "backend_name": config.backend_name,
        "artifact_kind": "hf_raw_checkpoint",
        "model_id": config.model_id,
        "revision": config.revision,
        "hf_cache_dir": str(config.hf_cache_dir),
        "checkpoint_filename": config.checkpoint_filename,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else None,
        "hf_access_ok": ok,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
    return manifest


def write_manifest(config, manifest: dict[str, object]) -> None:
    config.checkpoint_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.checkpoint_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", help="Absolute model cache root. Defaults to MORPHSEQ_MODEL_CACHE or ~/.cache/morphseq/models.")
    parser.add_argument("--vendor-path", help="Absolute SAM3 vendor checkout path. Accepted for symmetry with other backend commands.")
    parser.add_argument("--check-only", action="store_true", help="Do not download; only check the expected manifest/checkpoint state.")
    parser.add_argument("--print-json", action="store_true", help="Print the manifest JSON to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(model_cache=args.model_cache, vendor_path=args.vendor_path)
    checkpoint_path: Path | None = None
    error: Exception | None = None
    ok = False

    try:
        if args.check_only:
            if config.checkpoint_manifest_path.exists():
                prior = json.loads(config.checkpoint_manifest_path.read_text())
                checkpoint_value = prior.get("checkpoint_path")
                checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
            if not checkpoint_path or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAM3 checkpoint is missing; run install-model sam3: {config.checkpoint_manifest_path}"
                )
        else:
            checkpoint_path = fetch_checkpoint(config)
        ok = bool(checkpoint_path and checkpoint_path.exists())
    except Exception as exc:
        error = exc

    manifest = build_manifest(config, checkpoint_path=checkpoint_path, ok=ok, error=error)
    write_manifest(config, manifest)
    if args.print_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
