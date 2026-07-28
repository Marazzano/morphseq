"""Exact legacy-vs-current LoG audit on the A02 source tiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import skimage.io as skio
import torch

from src.build.export_utils import LoG_focus_stacker as legacy_log
from src.data_pipeline.acquisition.image_building.shared.log_focus import (
    LoG_focus_stacker as current_log,
)


ROOT = Path("pipeline/output/acquisition/20260702_hotchem_30hpf_plate01")
MANIFEST = ROOT / (
    "materialized_images/20260702_hotchem_30hpf_plate01_A02/BF/projection/focus_stack/"
    "raw_tile_manifest/20260702_hotchem_30hpf_plate01_A02_BF_t0000.json"
)
OUT = Path("results/mcolon/20260712_legacy_focus_stack_comparison/outputs/log_equivalence.json")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def comparison(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape_equal": a.shape == b.shape,
        "dtype_equal": a.dtype == b.dtype,
        "array_equal": bool(np.array_equal(a, b)),
        "sha256_equal": digest(a) == digest(b),
        "legacy_sha256": digest(a),
        "current_sha256": digest(b),
        "n_different": int(np.count_nonzero(a != b)),
        "max_abs_difference": float(delta.max(initial=0.0)),
        "mean_abs_difference": float(delta.mean()),
    }


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    report: dict[str, object] = {
        "manifest": str(MANIFEST),
        "filter_size": 3,
        "device": "cpu",
        "tiles": [],
    }
    for tile in payload["tiles"]:
        stack = np.stack([skio.imread(path) for path in tile["source_tiff_paths"]]).astype(np.float32)
        old_ff, old_log = legacy_log(stack, filter_size=3, device="cpu")
        new_ff, new_log = current_log(stack, filter_size=3, device="cpu")
        old_ff_np = old_ff.detach().cpu().numpy()
        new_ff_np = new_ff.detach().cpu().numpy()
        old_log_np = old_log.detach().cpu().numpy()
        new_log_np = new_log.detach().cpu().numpy()
        report["tiles"].append(
            {
                "tile_id": str(tile["tile_id"]),
                "input_shape": list(stack.shape),
                "input_dtype": str(stack.dtype),
                "input_sha256": digest(stack),
                "focus_image": comparison(old_ff_np, new_ff_np),
                "log_response": comparison(old_log_np, new_log_np),
                "focus_index_map": comparison(
                    torch.argmax(old_log, dim=0).cpu().numpy(),
                    torch.argmax(new_log, dim=0).cpu().numpy(),
                ),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
