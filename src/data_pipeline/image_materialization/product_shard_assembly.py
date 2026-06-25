"""Discover validated image-product shards and assemble canonical frame inventory.

Product materialization writes one frame-inventory shard per ``(well_id, product_key)``. This module
owns the next boundary: discover which validated product shards are active on disk, then concatenate
them into the canonical per-well ``frame_inventory`` shard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.image_materialization.frame_inventory_contract import (
    UNIQUE_FRAME_INVENTORY_KEY_COLUMNS,
)
from data_pipeline.image_materialization.image_product_keys import parse_image_product_key
from data_pipeline.metadata_ingest.frame_inventory.frame_inventory_validation import (
    _read_frame_inventory_table,
    _validate_unique_keys,
)

DISCOVERED_PRODUCT_SHARDS_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "well_id",
    "product_key",
    "frame_inventory_product_csv",
    "frame_inventory_product_validated",
)

_PRODUCT_SHARD_SUFFIX = "_frame_inventory.csv"


def product_key_from_frame_inventory_product_filename(path: Path, *, well_id: str) -> str:
    """Extract and validate product_key from ``{well_id}_{product_key}_frame_inventory.csv``."""
    name = Path(path).name
    prefix = f"{well_id}_"
    if not name.startswith(prefix) or not name.endswith(_PRODUCT_SHARD_SUFFIX):
        raise ValueError(
            f"Malformed frame_inventory product shard filename {name!r}. Expected "
            f"{well_id}_{{product_key}}{_PRODUCT_SHARD_SUFFIX}."
        )
    product_key = name[len(prefix) : -len(_PRODUCT_SHARD_SUFFIX)]
    if not product_key:
        raise ValueError(
            f"Malformed frame_inventory product shard filename {name!r}. Expected "
            f"{well_id}_{{product_key}}{_PRODUCT_SHARD_SUFFIX}."
        )
    try:
        parse_image_product_key(product_key)
    except ValueError as exc:
        raise ValueError(
            f"Malformed frame_inventory product shard filename {name!r}: {exc}"
        ) from exc
    return product_key


def discover_product_shards_for_well(
    *,
    experiment_id: str,
    well_id: str,
    frame_inventory_products_dir: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Write a manifest of validated product shards currently active for one well.

    The filesystem is the source of truth: a product shard is included only when both the CSV and
    its ``.csv.validated`` sidecar exist. Current config products are deliberately not consulted.
    """
    frame_inventory_products_dir = Path(frame_inventory_products_dir)
    rows: list[dict] = []
    seen: set[str] = set()
    for csv_path in sorted(frame_inventory_products_dir.glob("*.csv")):
        sentinel = csv_path.with_name(csv_path.name + ".validated")
        if not sentinel.exists():
            continue
        product_key = product_key_from_frame_inventory_product_filename(csv_path, well_id=well_id)
        if product_key in seen:
            raise ValueError(
                f"Duplicate product_key {product_key!r} discovered in {frame_inventory_products_dir}."
            )
        seen.add(product_key)
        rows.append({
            "experiment_id": str(experiment_id),
            "well_id": str(well_id),
            "product_key": product_key,
            "frame_inventory_product_csv": str(csv_path),
            "frame_inventory_product_validated": str(sentinel),
        })

    if not rows:
        raise ValueError(
            f"No validated frame_inventory product shards found for well {well_id!r} in "
            f"{frame_inventory_products_dir}. Build and validate product shards before assembly."
        )

    manifest = pd.DataFrame(rows, columns=list(DISCOVERED_PRODUCT_SHARDS_COLUMNS))
    manifest = manifest.sort_values("product_key").reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    return manifest


def assemble_well_frame_inventory(
    *,
    discovered_product_shards_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Assemble validated product shards into the canonical per-well frame_inventory CSV."""
    manifest = pd.read_csv(discovered_product_shards_csv)
    missing = set(DISCOVERED_PRODUCT_SHARDS_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(
            f"discovered_product_shards manifest is missing required columns: {sorted(missing)}"
        )
    if manifest.empty:
        raise ValueError("discovered_product_shards manifest is empty; cannot assemble frame_inventory.")
    if manifest["product_key"].duplicated().any():
        dupes = manifest.loc[
            manifest["product_key"].duplicated(keep=False), "product_key"
        ].tolist()
        raise ValueError(f"Duplicate product_key values in discovered_product_shards: {dupes}")

    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for row in manifest.to_dict(orient="records"):
        csv_path = Path(row["frame_inventory_product_csv"])
        sentinel = Path(row["frame_inventory_product_validated"])
        if not csv_path.exists():
            raise ValueError(f"Product frame_inventory shard does not exist: {csv_path}")
        if not sentinel.exists():
            raise ValueError(f"Product frame_inventory shard is not validated: missing {sentinel}")
        product_key_from_frame_inventory_product_filename(csv_path, well_id=str(row["well_id"]))
        frame = _read_frame_inventory_table(csv_path)
        columns = list(frame.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                f"Product frame_inventory shard {csv_path} has columns {columns}, "
                f"expected {expected_columns}."
            )
        frames.append(frame)

    assembled = pd.concat(frames, axis=0, ignore_index=True)
    _validate_unique_keys(assembled, context="frame_inventory")
    sort_cols = [c for c in list(UNIQUE_FRAME_INVENTORY_KEY_COLUMNS) if c in assembled.columns]
    if sort_cols:
        assembled = assembled.sort_values(sort_cols).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    assembled.to_csv(output_csv, index=False)
    return assembled
