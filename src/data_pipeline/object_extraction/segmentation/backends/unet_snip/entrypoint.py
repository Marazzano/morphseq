"""snip_auxiliary_masks entrypoint — the thin filesystem adapter for per-snip UNet masks.

Loads the per-well snip_inventory shard, resolves each ``processed_snip_path`` against the data
root, loads the UNet predictors from config, runs all auxiliary-mask families on each snip crop,
validates against the snip_inventory it was built from, then writes the per-well shard + sentinel.
No path-minting and no Torch behaviour live here (the predictor owns that).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.object_extraction.segmentation.backends.unet_snip.model_loader import (
    load_unet_snip_predictors,
    parse_unet_snip_model_config,
)
from data_pipeline.object_extraction.segmentation.backends.unet_snip.run_unet_snip import (
    UNET_SNIP_BACKEND_LABEL,
    run_unet_for_snip_inventory,
)
from data_pipeline.object_extraction.segmentation.backends.unet_snip.snip_auxiliary_masks_contract import (
    validate_snip_auxiliary_masks,
    validate_snip_auxiliary_masks_against_snip_inventory,
)
from data_pipeline.object_extraction.snip_processing.io import resolve_snip_inventory_image_paths


def run_snip_auxiliary_masks(
    *,
    snip_inventory_csv: Path,
    output_root: Path,
    output_csv: Path,
    unet_snip_config: dict,
    snip_frame_shape: tuple[int, int],
) -> None:
    snip_inventory = pd.read_csv(snip_inventory_csv)
    output_root = Path(output_root)

    # artifact_shape is the auxiliary-mask product's name for the snip-world law: the single
    # snip_frame_shape source of truth, never the unet_snip block. Every mask lands on this grid,
    # the same one as the cropped embryo mask. The rename happens at exactly this boundary.
    artifact_shape = (int(snip_frame_shape[0]), int(snip_frame_shape[1]))

    specs = parse_unet_snip_model_config(unet_snip_config, artifact_shape=artifact_shape)
    device = str(unet_snip_config.get("device", "cuda"))
    predictors = load_unet_snip_predictors(specs, device=device)
    checkpoint_paths = {spec.mask_type: str(spec.checkpoint_path) for spec in specs}

    resolved_inventory = resolve_snip_inventory_image_paths(snip_inventory, output_root=output_root)
    resolved_inventory = snip_inventory.merge(resolved_inventory, on="snip_id", how="left")
    resolved_inventory["processed_snip_path"] = resolved_inventory["resolved_image_path"]
    resolved_inventory = resolved_inventory.drop(columns=["resolved_image_path"])

    masks_dir = output_root / "object_extraction"
    df = run_unet_for_snip_inventory(
        resolved_inventory,
        predictors,
        output_dir=masks_dir,
        model_id=str(unet_snip_config.get("model_id", UNET_SNIP_BACKEND_LABEL)),
        model_backend=UNET_SNIP_BACKEND_LABEL,
        checkpoint_paths=checkpoint_paths,
        artifact_shape=artifact_shape,
    )

    # Contract (self) + cross-check identity against the snip_inventory it was built from.
    validate_snip_auxiliary_masks(df)
    validate_snip_auxiliary_masks_against_snip_inventory(df, snip_inventory)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
