"""Rendering-provenance extension for the canonical ``snip_inventory``.

The identity and product-aware row schema remains owned by
``physical_embryo_registry.snip_identity_contract``. This module adds the
strict physical-scale write gate and the versioned rendering-sidecar read gate.
Legacy inventories remain readable only when callers explicitly choose the
compatibility mode; new writers must provide finite positive scales on every
row.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.snip_identity_contract import (
    SNIP_INVENTORY_COLUMNS as IDENTITY_SNIP_INVENTORY_COLUMNS,
    validate_snip_inventory_contract as validate_identity_snip_inventory_contract,
)


SNIP_RENDERING_CONTRACT_VERSION = "snip-rendering-v2"

SNIP_INVENTORY_RENDERING_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_micrometers_per_pixel",
    "snip_micrometers_per_pixel",
)

# The product-aware identity contract already carries these columns. This alias
# gives the writer one schema import while this module adds value validation.
SNIP_INVENTORY_WRITE_COLUMNS: tuple[str, ...] = IDENTITY_SNIP_INVENTORY_COLUMNS


@dataclass(frozen=True)
class SnipInventoryContractStatus:
    """Rendering-provenance state observed while reading an inventory."""

    contract_version: str | None
    sidecar_present: bool
    present_rendering_provenance_columns: tuple[str, ...]
    missing_rendering_provenance_columns: tuple[str, ...]
    snip_product_keys: tuple[str, ...]

    @property
    def has_complete_rendering_provenance(self) -> bool:
        return (
            not self.missing_rendering_provenance_columns
            and self.sidecar_present
            and self.contract_version == SNIP_RENDERING_CONTRACT_VERSION
        )


@dataclass(frozen=True)
class SnipInventoryReadResult:
    inventory: pd.DataFrame
    contract_status: SnipInventoryContractStatus


def _product_keys(df: pd.DataFrame) -> tuple[str, ...]:
    if "snip_product_key" not in df.columns:
        return ()
    return tuple(sorted({str(value) for value in df["snip_product_key"].dropna()}))


def _rendering_contract_status(
    df: pd.DataFrame,
    *,
    contract_version: str | None = None,
    sidecar_present: bool = False,
) -> SnipInventoryContractStatus:
    present = tuple(
        column
        for column in SNIP_INVENTORY_RENDERING_PROVENANCE_COLUMNS
        if column in df.columns
    )
    missing = tuple(
        column
        for column in SNIP_INVENTORY_RENDERING_PROVENANCE_COLUMNS
        if column not in df.columns
    )
    return SnipInventoryContractStatus(
        contract_version=contract_version,
        sidecar_present=sidecar_present,
        present_rendering_provenance_columns=present,
        missing_rendering_provenance_columns=missing,
        snip_product_keys=_product_keys(df),
    )


def _require_finite_positive_column(
    df: pd.DataFrame, column: str, *, scope_label: str
) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float)) | (values <= 0)
    if invalid.any():
        columns = [
            value
            for value in ("snip_id", "snip_product_key", column)
            if value in df.columns
        ]
        examples = df.loc[invalid, columns].head(5).to_dict(orient="records")
        raise ValueError(
            f"{scope_label}: {column} is required as a finite positive number on every "
            f"written row; found {int(invalid.sum())} invalid value(s). Examples: {examples}"
        )


def validate_snip_inventory(
    df: pd.DataFrame,
    *,
    require_rendering_provenance: bool = False,
    scope_label: str = "snip_inventory",
) -> SnipInventoryContractStatus:
    """Validate identity plus the optional strict rendering-provenance layer."""

    status = _rendering_contract_status(df)
    if require_rendering_provenance and status.missing_rendering_provenance_columns:
        raise ValueError(
            f"{scope_label}: missing rendering provenance columns required on write: "
            f"{list(status.missing_rendering_provenance_columns)}"
        )

    # Compatibility placeholders support genuinely older tables at read time and
    # never mutate the caller's dataframe.
    identity_view = df.copy()
    for column in SNIP_INVENTORY_RENDERING_PROVENANCE_COLUMNS:
        if column not in identity_view.columns:
            identity_view[column] = pd.NA
    validate_identity_snip_inventory_contract(identity_view, scope_label=scope_label)

    if require_rendering_provenance and len(df):
        for column in SNIP_INVENTORY_RENDERING_PROVENANCE_COLUMNS:
            _require_finite_positive_column(df, column, scope_label=scope_label)

    return status


def read_snip_inventory(
    path: Path,
    *,
    sidecar_path: Path | None = None,
    require_sidecar: bool = False,
) -> SnipInventoryReadResult:
    """Read an inventory and validate any co-located product-aware sidecar."""

    path = Path(path)
    inventory = (
        pd.read_parquet(path)
        if path.suffix.casefold() in {".parquet", ".pq"}
        else pd.read_csv(path)
    )
    status = validate_snip_inventory(
        inventory,
        require_rendering_provenance=require_sidecar,
        scope_label=str(path),
    )
    resolved_sidecar = (
        Path(sidecar_path)
        if sidecar_path is not None
        else path.with_name(path.name + ".provenance.json")
    )
    if not resolved_sidecar.exists():
        if require_sidecar:
            raise FileNotFoundError(
                f"{path}: missing snip rendering sidecar required for a new inventory: "
                f"{resolved_sidecar}"
            )
        return SnipInventoryReadResult(inventory=inventory, contract_status=status)

    try:
        sidecar = json.loads(resolved_sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable snip rendering sidecar: {resolved_sidecar}") from exc

    version = sidecar.get("contract_version")
    if version != SNIP_RENDERING_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported snip rendering contract version {version!r} in {resolved_sidecar}; "
            f"expected {SNIP_RENDERING_CONTRACT_VERSION!r}."
        )
    products = sidecar.get("products")
    if not isinstance(products, dict):
        raise ValueError(
            f"{resolved_sidecar}: rendering contract must contain a product-keyed 'products' object."
        )
    missing_products = sorted(set(_product_keys(inventory)) - set(products))
    if missing_products:
        raise ValueError(
            f"{resolved_sidecar}: sidecar does not describe inventory snip product(s) "
            f"{missing_products}."
        )

    status = _rendering_contract_status(
        inventory,
        contract_version=str(version),
        sidecar_present=True,
    )
    return SnipInventoryReadResult(inventory=inventory, contract_status=status)
