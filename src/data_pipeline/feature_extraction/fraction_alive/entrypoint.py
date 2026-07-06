"""fraction_alive entrypoint — thin filesystem adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compute import MISSING_VIA_FAIL, compute_fraction_alive_features
from .contract import validate_fraction_alive_features


def run_fraction_alive(
    *,
    snip_inventory_csv: Path,
    snip_auxiliary_masks_csv: Path,
    physical_embryo_registry_csv: Path,
    output_csv: Path,
    snip_frame_shape,
    output_root: Path | None = None,
    missing_via_policy: str = MISSING_VIA_FAIL,
) -> None:
    snip_inventory = pd.read_csv(snip_inventory_csv)
    snip_auxiliary_masks = pd.read_csv(snip_auxiliary_masks_csv)
    snip_auxiliary_masks["is_valid_auxiliary_mask"] = snip_auxiliary_masks[
        "is_valid_auxiliary_mask"
    ].astype(bool)
    registry = pd.read_csv(physical_embryo_registry_csv)

    df = compute_fraction_alive_features(
        snip_inventory,
        snip_auxiliary_masks,
        snip_frame_shape=snip_frame_shape,
        output_root=output_root,
        missing_via_policy=missing_via_policy,
    )
    validate_fraction_alive_features(df, physical_embryo_registry_df=registry, check_sources=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
