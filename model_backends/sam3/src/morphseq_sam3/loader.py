"""Heavy native SAM3 loader boundary."""

from __future__ import annotations

from pathlib import Path
import os
import sys

from morphseq_sam3.config import Sam3BackendConfig


def ensure_vendor_on_path(vendor_path: Path) -> None:
    vendor = str(vendor_path)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def configure_hf_cache(config: Sam3BackendConfig) -> None:
    config.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(config.hf_cache_dir))


def import_model_builder(config: Sam3BackendConfig):
    ensure_vendor_on_path(config.vendor_path)
    from sam3.model_builder import build_sam3_predictor

    return build_sam3_predictor


def load_sam3_predictor(config: Sam3BackendConfig):
    """Build the native SAM3 predictor.

    Keep this function as the only routine that imports native SAM3. Callers that only need config,
    manifests, or setup reports should not pay the heavy import cost.
    """

    configure_hf_cache(config)
    build_sam3_predictor = import_model_builder(config)
    kwargs = {
        "version": config.sam3_version,
    }
    if config.sam3_version == "sam3.1":
        kwargs["use_rope_real"] = config.use_rope_real
        kwargs["use_fa3"] = config.use_fa3
    return build_sam3_predictor(**kwargs)
