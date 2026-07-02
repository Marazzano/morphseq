#!/usr/bin/env python
"""Verify SAM3 backend readiness and write setup_report.json."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.util
import json
import platform
import sys
import traceback
from tempfile import TemporaryDirectory


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
PROJECT_ROOT = THIS_DIR.parents[1]
PROJECT_SRC_DIR = PROJECT_ROOT / "src"
for import_path in (SRC_DIR, PROJECT_ROOT, PROJECT_SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from morphseq_sam3.config import build_config


def read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def torch_status() -> dict[str, object]:
    status: dict[str, object] = {
        "torch_import_ok": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "compute_capability": None,
        "bf16_supported": None,
        "recommended_precision": "fp32",
    }
    try:
        import torch

        status["torch_import_ok"] = True
        status["torch_version"] = getattr(torch, "__version__", None)
        status["cuda_version"] = getattr(torch.version, "cuda", None)
        status["cuda_available"] = bool(torch.cuda.is_available())
        if status["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            status["gpu_name"] = props.name
            status["compute_capability"] = f"{props.major}.{props.minor}"
            status["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            status["recommended_precision"] = "bf16" if status["bf16_supported"] else "fp16"
    except Exception as exc:
        status["torch_error_type"] = type(exc).__name__
        status["torch_error"] = str(exc)
    return status


def package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def flash_attn_import_ok() -> bool:
    return bool(
        importlib.util.find_spec("flash_attn")
        or importlib.util.find_spec("flash_attn_interface")
    )


def try_adapter_import() -> tuple[bool, str | None]:
    module_name = "data_pipeline.object_extraction.detection.backends.sam3.adapt_sam3_detections"
    try:
        __import__(module_name)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_contract_write_smoke() -> tuple[bool, str | None]:
    try:
        import pandas as pd

        from data_pipeline.object_extraction.detection.backends.sam3.adapt_sam3_detections import (
            adapt_sam3_detections,
        )
        from data_pipeline.object_extraction.detection.validate_frame_detections import (
            validate_frame_detection_block,
        )

        image_id = "20250912_B01_BF_t0000"
        identity_row = {
            "experiment_id": "20250912",
            "well_id": "20250912_B01",
            "image_id": image_id,
            "time_index": 0,
            "z_index": pd.NA,
            "channel_id": "BF",
            "source_image_path": f"images/{image_id}.png",
            "image_width_px": 1000,
            "image_height_px": 800,
        }
        detections = [
            {
                "box_xyxy": [100.0, 80.0, 300.0, 240.0],
                "score": 0.95,
                "label": "zebrafish embryo",
            }
        ]
        with TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "frame_detections.csv"
            df = adapt_sam3_detections(
                detections,
                identity_row=identity_row,
                output_csv=output_csv,
            )
            validate_frame_detection_block(df)
            roundtrip = pd.read_csv(output_csv)
            if len(roundtrip) != 1:
                raise ValueError(f"Expected one smoke row, got {len(roundtrip)}")
        return True, None
    except Exception:
        return False, traceback.format_exc(limit=8)


def check_forbidden_adapter_imports() -> tuple[bool, str | None]:
    forbidden_roots = {
        "snakemake",
        "pipeline_orchestrator",
        "torch",
        "sam3",
        "detectron2",
        "groundingdino",
    }
    try:
        before = set(sys.modules)
        from data_pipeline.object_extraction.detection.backends.sam3 import adapt_sam3_detections  # noqa: F401

        after = set(sys.modules)
        imported = after - before
        offenders = sorted(
            name for name in imported if name.split(".", 1)[0] in forbidden_roots
        )
        if offenders:
            return False, f"Forbidden adapter imports: {offenders[:20]}"
        return True, None
    except Exception:
        return False, traceback.format_exc(limit=8)


def load_predictor_if_requested(config, *, load: bool) -> tuple[bool, str | None]:
    if not load:
        return False, "skipped"
    try:
        from morphseq_sam3.loader import load_sam3_predictor

        load_sam3_predictor(config)
        return True, None
    except Exception:
        return False, traceback.format_exc(limit=8)


def build_report(config, *, load: bool, run_fa3_smoke: bool) -> dict[str, object]:
    vendor_manifest = read_json(config.vendor_manifest_path)
    checkpoint_manifest = read_json(config.checkpoint_manifest_path)
    torch_info = torch_status()
    adapter_import_ok, adapter_error = try_adapter_import()
    contract_write_smoke_ok, contract_write_error = (
        run_contract_write_smoke() if adapter_import_ok else (False, "adapter import failed")
    )
    forbidden_imports_ok, forbidden_imports_error = (
        check_forbidden_adapter_imports() if adapter_import_ok else (False, "adapter import failed")
    )
    real_model_load_ok, load_error = load_predictor_if_requested(config, load=load)

    checkpoint_exists = bool(
        checkpoint_manifest
        and checkpoint_manifest.get("checkpoint_path")
        and Path(str(checkpoint_manifest["checkpoint_path"])).exists()
    )
    vendor_ok = bool(
        vendor_manifest
        and vendor_manifest.get("install_backend_ok")
        and config.vendor_path.exists()
    )
    flash_ok = flash_attn_import_ok()
    fa3_available = False
    fa3_smoke_test = "skipped"
    if run_fa3_smoke:
        fa3_smoke_test = "not_implemented"

    exemplar_prompt_smoke_ok = False
    sam3_detection_ok = all(
        [
            real_model_load_ok,
            exemplar_prompt_smoke_ok,
            adapter_import_ok,
            contract_write_smoke_ok,
        ]
    )
    if sam3_detection_ok:
        backend_class = "integrated"
    elif vendor_ok and checkpoint_exists and real_model_load_ok:
        backend_class = "detached"
    else:
        backend_class = "unavailable"

    return {
        "backend_name": config.backend_name,
        "backend_class": backend_class,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torchvision_version": package_version("torchvision"),
        "model_cache_root": str(config.model_cache_root),
        **torch_info,
        "sam3_runtime_kind": config.runtime_kind,
        "sam3_vendor_path": str(config.vendor_path),
        "sam3_vendor_commit": vendor_manifest.get("vendor_commit") if vendor_manifest else None,
        "sam3_vendor_expected_commit": config.upstream_commit,
        "sam3_version": config.sam3_version,
        "sam3_device": config.device,
        "use_fa3": config.use_fa3,
        "use_rope_real": config.use_rope_real,
        "artifact_kind": "hf_raw_checkpoint",
        "artifact_root": str(config.artifact_root),
        "artifact_manifest_path": str(config.checkpoint_manifest_path),
        "hf_model_id": config.model_id,
        "hf_revision": config.revision,
        "hf_cache_dir": str(config.hf_cache_dir),
        "checkpoint_path": checkpoint_manifest.get("checkpoint_path") if checkpoint_manifest else None,
        "checkpoint_sha256": checkpoint_manifest.get("checkpoint_sha256") if checkpoint_manifest else None,
        "hf_access_ok": bool(checkpoint_manifest and checkpoint_manifest.get("hf_access_ok")),
        "vendor_checkout_ok": vendor_ok,
        "checkpoint_exists": checkpoint_exists,
        "adapter_import_ok": adapter_import_ok,
        "adapter_import_error": adapter_error,
        "forbidden_imports_ok": forbidden_imports_ok,
        "forbidden_imports_error": forbidden_imports_error,
        "real_model_load_ok": real_model_load_ok,
        "real_model_load_error": load_error,
        "exemplar_prompt_smoke_ok": exemplar_prompt_smoke_ok,
        "contract_write_smoke_ok": contract_write_smoke_ok,
        "contract_write_smoke_error": contract_write_error,
        "sam3_detection_ok": sam3_detection_ok,
        "flash_attn_import_ok": flash_ok,
        "fa3_available": fa3_available,
        "fa3_smoke_test": fa3_smoke_test,
        "production_use_fa3": fa3_available,
        "equivalence_status": "not_started",
        "equivalence_report_path": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", help="Absolute model cache root. Defaults to MORPHSEQ_MODEL_CACHE or ~/.cache/morphseq/models.")
    parser.add_argument("--vendor-path", help="Absolute SAM3 vendor checkout path.")
    parser.add_argument("--sam3-version", default=None, help="SAM3 version to verify: sam3.1 or sam3.")
    parser.add_argument("--device", default="auto", help="Device argument passed to the loader when --load is used.")
    parser.add_argument("--use-fa3", action="store_true", help="Request FA3 in the loader config. Use only with the FA3 smoke.")
    parser.add_argument("--load", action="store_true", help="Actually build the native SAM3 predictor.")
    parser.add_argument("--fa3-smoke", action="store_true", help="Reserve flag for the production FA3 inference smoke.")
    parser.add_argument("--print-json", action="store_true", help="Print setup report JSON to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(
        model_cache=args.model_cache,
        vendor_path=args.vendor_path,
        sam3_version=args.sam3_version,
        device=args.device,
        use_fa3=args.use_fa3,
    )
    report = build_report(config, load=args.load, run_fa3_smoke=args.fa3_smoke)
    config.setup_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.setup_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["sam3_detection_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
