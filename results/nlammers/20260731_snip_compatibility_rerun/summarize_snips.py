"""Summarize a canonical snip tree before/after the 6.5/75 compatibility rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as skio


EXPERIMENT = "20250612_24hpf_ctrl_atf6"
LEGACY_DIR = Path(
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/"
    "bf_embryo_snips/20250612_24hpf_ctrl_atf6"
)


def _resolve_in_snips_root(snips_root: Path, inventory_path: str) -> Path:
    path = Path(str(inventory_path))
    parts = path.parts
    try:
        snips_index = parts.index("snips")
    except ValueError as exc:
        raise ValueError(f"inventory path has no 'snips' component: {path}") from exc
    return snips_root.joinpath(*parts[snips_index + 1 :])


def _load_gray(path: Path) -> np.ndarray:
    image = skio.imread(str(path))
    if image.ndim == 3:
        image = image[:, :, 0]
    return np.asarray(image, dtype=np.uint8)


def _pooled_summary(images: list[np.ndarray]) -> dict[str, float | int]:
    foreground = np.concatenate([image[image > 30] for image in images])
    return {
        "n_images": len(images),
        "n_foreground_pixels": int(foreground.size),
        "foreground_mean": float(foreground.mean()),
        "foreground_p95": float(np.percentile(foreground, 95)),
        "foreground_fraction_ge_250": float(np.mean(foreground >= 250)),
    }


def summarize(snips_root: Path) -> dict[str, object]:
    inventory_path = snips_root / f"{EXPERIMENT}_snip_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    valid = inventory[inventory["is_valid_snip"].astype(bool)].copy()
    valid = valid.sort_values(["well_id", "physical_embryo_id", "snip_id"])

    images: list[np.ndarray] = []
    mask_areas: list[int] = []
    primary_current: list[np.ndarray] = []
    primary_legacy: list[np.ndarray] = []
    missing: list[str] = []

    for _, row in valid.iterrows():
        image_path = _resolve_in_snips_root(snips_root, str(row["processed_snip_path"]))
        mask_path = _resolve_in_snips_root(snips_root, str(row["embryo_mask_snip_path"]))
        image = _load_gray(image_path)
        mask = _load_gray(mask_path) > 0
        images.append(image)
        mask_areas.append(int(mask.sum()))

    for _, row in valid.drop_duplicates("well_id", keep="first").iterrows():
        well = str(row["well_id"]).removeprefix(f"{EXPERIMENT}_")
        legacy_path = LEGACY_DIR / f"{EXPERIMENT}_{well}_e00_t0000.jpg"
        if not legacy_path.exists():
            missing.append(str(legacy_path))
            continue
        current_path = _resolve_in_snips_root(snips_root, str(row["processed_snip_path"]))
        primary_current.append(_load_gray(current_path))
        primary_legacy.append(_load_gray(legacy_path))

    current_primary = _pooled_summary(primary_current)
    legacy_primary = _pooled_summary(primary_legacy)
    return {
        "experiment_id": EXPERIMENT,
        "inventory_path": str(inventory_path),
        "inventory_mtime_ns": inventory_path.stat().st_mtime_ns,
        "n_inventory_rows": int(len(inventory)),
        "n_valid_snips": int(len(valid)),
        "n_wells": int(valid["well_id"].nunique()),
        "n_physical_embryos": int(valid["physical_embryo_id"].nunique()),
        "image_shapes": [list(shape) for shape in sorted({tuple(image.shape) for image in images})],
        "all_current_snips": _pooled_summary(images),
        "saved_embryo_mask_area": {
            "mean": float(np.mean(mask_areas)),
            "median": float(np.median(mask_areas)),
            "min": int(np.min(mask_areas)),
            "max": int(np.max(mask_areas)),
        },
        "paired_primary_count": len(primary_current),
        "paired_current_primary": current_primary,
        "paired_legacy_primary": legacy_primary,
        "paired_current_to_legacy": {
            "foreground_pixel_count_ratio": float(
                current_primary["n_foreground_pixels"] / legacy_primary["n_foreground_pixels"]
            ),
            "saturation_ratio": float(
                current_primary["foreground_fraction_ge_250"]
                / legacy_primary["foreground_fraction_ge_250"]
            ),
        },
        "missing_legacy_paths": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snips-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.snips_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
