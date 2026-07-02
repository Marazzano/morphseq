#!/usr/bin/env python
"""Install or verify the SAM3 backend runtime/code step."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphseq_sam3.config import build_config


def run_git(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_output(args: list[str], *, cwd: Path) -> str:
    return run_git(args, cwd=cwd).stdout.strip()


def ensure_checkout(config, *, check_only: bool) -> None:
    if config.vendor_path.exists():
        return
    if check_only:
        raise FileNotFoundError(f"SAM3 vendor checkout is missing: {config.vendor_path}")
    config.vendor_path.parent.mkdir(parents=True, exist_ok=True)
    run_git(["clone", config.upstream_repo, str(config.vendor_path)])


def checkout_commit(config, *, check_only: bool) -> None:
    current = git_output(["rev-parse", "HEAD"], cwd=config.vendor_path)
    if current.startswith(config.upstream_commit):
        return
    if check_only:
        raise RuntimeError(
            f"SAM3 vendor checkout is at {current}, expected {config.upstream_commit}"
        )
    run_git(["fetch", "--all", "--tags"], cwd=config.vendor_path)
    run_git(["checkout", config.upstream_commit], cwd=config.vendor_path)


def inspect_checkout(config) -> dict[str, object]:
    dirty = bool(git_output(["status", "--porcelain"], cwd=config.vendor_path))
    commit = git_output(["rev-parse", "HEAD"], cwd=config.vendor_path)
    remote = git_output(["config", "--get", "remote.origin.url"], cwd=config.vendor_path)
    import_ok = (config.vendor_path / "sam3" / "model_builder.py").exists()
    return {
        "backend_name": config.backend_name,
        "runtime_kind": config.runtime_kind,
        "upstream_repo": config.upstream_repo,
        "expected_commit": config.upstream_commit,
        "vendor_path": str(config.vendor_path),
        "vendor_remote": remote,
        "vendor_commit": commit,
        "vendor_dirty": dirty,
        "native_import_file_present": import_ok,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(config, manifest: dict[str, object]) -> None:
    config.vendor_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.vendor_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", help="Absolute model cache root. Defaults to MORPHSEQ_MODEL_CACHE or ~/.cache/morphseq/models.")
    parser.add_argument("--vendor-path", help="Absolute SAM3 vendor checkout path. Defaults to MORPHSEQ_SAM3_VENDOR_PATH or ~/.cache/morphseq/vendor/sam3.")
    parser.add_argument("--check-only", action="store_true", help="Do not clone/fetch/checkout; only verify the current checkout.")
    parser.add_argument("--print-json", action="store_true", help="Print the manifest JSON to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(model_cache=args.model_cache, vendor_path=args.vendor_path)
    try:
        ensure_checkout(config, check_only=args.check_only)
        checkout_commit(config, check_only=args.check_only)
        manifest = inspect_checkout(config)
        manifest["install_backend_ok"] = bool(manifest["native_import_file_present"])
        write_manifest(config, manifest)
    except Exception as exc:
        manifest = {
            "backend_name": config.backend_name,
            "runtime_kind": config.runtime_kind,
            "vendor_path": str(config.vendor_path),
            "expected_commit": config.upstream_commit,
            "install_backend_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_manifest(config, manifest)
        if args.print_json:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        return 1

    if args.print_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["install_backend_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
