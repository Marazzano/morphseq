"""Locate divergence between legacy and current A02 focus-stack inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import skimage.io as skio
import torch

from src.build.data_classes import MultiTileZStackDataset
from src.data_pipeline.acquisition.image_building.shared.log_focus import (
    LoG_focus_stacker,
    im_rescale,
)


ROOT = Path("pipeline/output/acquisition/20260702_hotchem_30hpf_plate01")
MANIFEST = ROOT / (
    "materialized_images/20260702_hotchem_30hpf_plate01_A02/BF/projection/focus_stack/"
    "raw_tile_manifest/20260702_hotchem_30hpf_plate01_A02_BF_t0000.json"
)
OUT = Path(
    "results/mcolon/20260712_legacy_focus_stack_comparison/outputs/"
    "preprocessing_divergence.json"
)


def stats(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "p0_1": float(np.percentile(array, 0.1)),
        "p99_9": float(np.percentile(array, 99.9)),
        "n_zero": int(np.count_nonzero(array == 0)),
        "n_max": int(np.count_nonzero(array == array.max())),
    }


def main() -> None:
    np.random.seed(0)
    manifest = json.loads(MANIFEST.read_text())
    path_groups = [tile["source_tiff_paths"] for tile in manifest["tiles"]]
    raw = np.stack(
        [np.stack([skio.imread(path) for path in paths]) for paths in path_groups], axis=0
    )

    # Exact legacy Build01 input construction.
    legacy = MultiTileZStackDataset([{"tile_zpaths": path_groups}])[0]["data"].numpy()

    # Exact current Keyence path: materialize_well_keyence rescales each tile,
    # then materialize_ff_projection rescales the already-rescaled tile again.
    current_once = np.stack([im_rescale(tile)[0].astype(np.float32) for tile in raw])
    current_twice = np.stack([im_rescale(tile)[0].astype(np.float32) for tile in current_once])

    _, legacy_log = LoG_focus_stacker(legacy, filter_size=3, device="cpu")
    _, current_once_log = LoG_focus_stacker(current_once, filter_size=3, device="cpu")
    _, current_twice_log = LoG_focus_stacker(current_twice, filter_size=3, device="cpu")
    legacy_idx = torch.argmax(legacy_log, dim=1).cpu().numpy()
    once_idx = torch.argmax(current_once_log, dim=1).cpu().numpy()
    twice_idx = torch.argmax(current_twice_log, dim=1).cpu().numpy()

    report: dict[str, object] = {
        "manifest": str(MANIFEST),
        "raw": stats(raw),
        "legacy_actual_input": stats(legacy),
        "current_after_first_rescale": stats(current_once),
        "current_actual_input_after_second_rescale": stats(current_twice),
        "per_tile": [],
    }
    for tile_i, tile in enumerate(manifest["tiles"]):
        old_vs_once = legacy_idx[tile_i] != once_idx[tile_i]
        old_vs_twice = legacy_idx[tile_i] != twice_idx[tile_i]
        once_vs_twice = once_idx[tile_i] != twice_idx[tile_i]
        report["per_tile"].append(
            {
                "tile_id": str(tile["tile_id"]),
                "legacy_input": stats(legacy[tile_i]),
                "current_once_input": stats(current_once[tile_i]),
                "current_twice_input": stats(current_twice[tile_i]),
                "legacy_vs_current_once_n_different_focus_indices": int(old_vs_once.sum()),
                "legacy_vs_current_once_fraction_different_focus_indices": float(old_vs_once.mean()),
                "legacy_vs_current_twice_n_different_focus_indices": int(old_vs_twice.sum()),
                "legacy_vs_current_twice_fraction_different_focus_indices": float(old_vs_twice.mean()),
                "current_once_vs_twice_n_different_focus_indices": int(once_vs_twice.sum()),
                "current_once_vs_twice_fraction_different_focus_indices": float(once_vs_twice.mean()),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
