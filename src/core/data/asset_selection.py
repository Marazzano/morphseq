"""Dataset-side validation of an adapter-resolved asset view.

The manifest adapter owns cohort and asset policy.  This module deliberately does
not inspect the full asset table or choose among sibling products.  It validates
and groups rows that the adapter has already resolved, leaving a row-list seam for
future selectors that may resolve several assets for one observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


VANILLA_REQUIRED_COLUMNS = frozenset(
    {
        "snip_id",
        "snip_product_key",
        "z_index",
        "processed_snip_path",
    }
)


@dataclass(frozen=True)
class ResolvedAssetGroup:
    """Ordered resolved-row references for one opaque observation identity."""

    snip_id: str
    resolved_row_references: tuple[int, ...]


def group_resolved_asset_rows(
    resolved_sample_table: pd.DataFrame,
) -> tuple[ResolvedAssetGroup, ...]:
    """Group an already-resolved view without changing its observation order.

    The references are zero-based positions in ``resolved_sample_table`` rather
    than source DataFrame index labels.  A future z-aware adapter may resolve a
    row list for one observation; the vanilla validator below requires length one.
    """

    if "snip_id" not in resolved_sample_table.columns:
        raise ValueError("Resolved sample table is missing required column 'snip_id'.")

    ordered_ids: list[str] = []
    row_references: dict[str, list[int]] = {}
    for row_reference, raw_snip_id in enumerate(resolved_sample_table["snip_id"].tolist()):
        if pd.isna(raw_snip_id):
            raise ValueError(
                f"Resolved sample row {row_reference} has null snip_id; observation IDs are required."
            )
        snip_id = str(raw_snip_id)
        if snip_id not in row_references:
            ordered_ids.append(snip_id)
            row_references[snip_id] = []
        row_references[snip_id].append(row_reference)

    return tuple(
        ResolvedAssetGroup(snip_id, tuple(row_references[snip_id]))
        for snip_id in ordered_ids
    )


def validate_vanilla_resolved_view(
    resolved_sample_table: pd.DataFrame,
    *,
    product_key: str,
    expected_observation_ids: Sequence[str] | None = None,
) -> tuple[ResolvedAssetGroup, ...]:
    """Validate exact BF/null-z rows selected by the manifest adapter.

    This is a guard at the dataset boundary, not a second selection policy.  It
    never consults sibling assets and never repairs an invalid resolved view by
    taking the first row.
    """

    missing_columns = sorted(VANILLA_REQUIRED_COLUMNS - set(resolved_sample_table.columns))
    if missing_columns:
        raise ValueError(
            "Resolved sample table is missing vanilla asset columns: "
            f"{missing_columns}. Configured product={product_key!r}."
        )
    if not isinstance(product_key, str) or not product_key:
        raise ValueError(f"Configured vanilla product_key must be a non-empty string, got {product_key!r}.")

    groups = group_resolved_asset_rows(resolved_sample_table)
    group_ids = [group.snip_id for group in groups]

    if expected_observation_ids is not None:
        expected = [str(value) for value in expected_observation_ids]
        expected_set = set(expected)
        actual_set = set(group_ids)
        missing_ids = [snip_id for snip_id in expected if snip_id not in actual_set]
        extra_ids = [snip_id for snip_id in group_ids if snip_id not in expected_set]
        if missing_ids or extra_ids:
            raise ValueError(
                "Resolved vanilla observation coverage mismatch for "
                f"product={product_key!r}: missing snip_id={missing_ids}, extra snip_id={extra_ids}."
            )

    for group in groups:
        if len(group.resolved_row_references) != 1:
            details = resolved_sample_table.iloc[list(group.resolved_row_references)][
                ["snip_product_key", "z_index", "processed_snip_path"]
            ].to_dict("records")
            raise ValueError(
                "Vanilla resolved view requires exactly one asset-row reference for "
                f"snip_id={group.snip_id!r}, product={product_key!r}; "
                f"found {len(group.resolved_row_references)} rows: {details}."
            )

        row_reference = group.resolved_row_references[0]
        row = resolved_sample_table.iloc[row_reference]
        if row["snip_product_key"] != product_key or not pd.isna(row["z_index"]):
            raise ValueError(
                "Resolved vanilla row does not match the configured projection asset for "
                f"snip_id={group.snip_id!r}, product={product_key!r}: "
                f"resolved product={row['snip_product_key']!r}, z_index={row['z_index']!r}, "
                f"path={row['processed_snip_path']!r}."
            )

        path = row["processed_snip_path"]
        if pd.isna(path) or not str(path):
            raise ValueError(
                "Resolved vanilla row has no processed asset path for "
                f"snip_id={group.snip_id!r}, product={product_key!r}."
            )

        if "is_valid_snip" in resolved_sample_table.columns:
            validity = row["is_valid_snip"]
            if isinstance(validity, str):
                normalized = validity.strip().lower()
                if normalized not in {"true", "false"}:
                    raise ValueError(
                        f"Invalid is_valid_snip={validity!r} for snip_id={group.snip_id!r}."
                    )
                validity = normalized == "true"
            if pd.isna(validity) or not bool(validity):
                raise ValueError(
                    "Resolved vanilla row is not a valid materialized asset for "
                    f"snip_id={group.snip_id!r}, product={product_key!r}."
                )

    return groups
