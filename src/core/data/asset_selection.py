"""Asset-row selection for manifest-backed core datasets.

Selectors consume the contract-v2 observation and asset tables.  They never inspect
directories or infer product/plane semantics from paths or identifiers.  The public
``resolve_asset_row_groups`` result deliberately contains an ordered tuple of asset
row positions for each observation; vanilla currently requires one row, while this
shape leaves future z-plane grouping to a later policy decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


class AssetSelectionError(ValueError):
    """Raised when a configured selector cannot resolve an observation exactly."""


@dataclass(frozen=True)
class ResolvedAssetRowGroup:
    """Ordered asset-table row positions selected for one observation."""

    observation_row_position: int
    snip_id: str
    asset_row_positions: tuple[int, ...]


class AssetRowSelector(Protocol):
    """Policy seam for resolving one observation to ordered asset rows."""

    name: str

    def select_row_positions(
        self,
        *,
        snip_id: str,
        candidate_row_positions: Sequence[int],
        asset_table: pd.DataFrame,
    ) -> Sequence[int]:
        """Return selected positions in the intended model-input order."""


def normalize_nullable_z_index(value: object, *, snip_id: str) -> int | None:
    """Normalize one contract-v2 nullable plane index without guessing semantics."""

    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise AssetSelectionError(
            f"snip_id={snip_id!r} has boolean z_index={value!r}; expected null or a "
            "non-negative integer plane index."
        )
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        result = int(value)
    else:
        raise AssetSelectionError(
            f"snip_id={snip_id!r} has invalid z_index={value!r}; expected null or a "
            "non-negative integer plane index."
        )
    if result < 0:
        raise AssetSelectionError(
            f"snip_id={snip_id!r} has negative z_index={result}; plane indices must be "
            "non-negative."
        )
    return result


def _normalized_bool(value: object, *, snip_id: str, field: str) -> bool:
    """Parse the two known string spellings without ever calling ``bool(str)``."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value == "True":
        return True
    if value == "False":
        return False
    raise AssetSelectionError(
        f"snip_id={snip_id!r} has non-boolean {field}={value!r}; expected True or False."
    )


def _available_asset_summary(
    asset_table: pd.DataFrame, candidate_row_positions: Sequence[int]
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for position in candidate_row_positions:
        row = asset_table.iloc[position]
        summaries.append(
            {
                "snip_product_key": row.get("snip_product_key"),
                "z_index": row.get("z_index"),
                "is_valid_snip": row.get("is_valid_snip"),
                "processed_snip_path": row.get("processed_snip_path"),
            }
        )
    return summaries


@dataclass(frozen=True)
class VanillaBFProjectionSelector:
    """Select one exact, valid, null-z BF projection asset per observation."""

    snip_product_key: str
    require_valid_snip: bool = True
    name: str = "vanilla_bf_projection_null_z"

    def __post_init__(self) -> None:
        if not isinstance(self.snip_product_key, str) or not self.snip_product_key:
            raise ValueError("snip_product_key must be a non-empty explicit string.")

    def select_row_positions(
        self,
        *,
        snip_id: str,
        candidate_row_positions: Sequence[int],
        asset_table: pd.DataFrame,
    ) -> Sequence[int]:
        matches: list[int] = []
        for position in candidate_row_positions:
            row = asset_table.iloc[position]
            if row["snip_product_key"] != self.snip_product_key:
                continue
            if normalize_nullable_z_index(row["z_index"], snip_id=snip_id) is not None:
                continue
            if self.require_valid_snip and not _normalized_bool(
                row["is_valid_snip"], snip_id=snip_id, field="is_valid_snip"
            ):
                continue
            matches.append(int(position))

        if len(matches) != 1:
            available = _available_asset_summary(asset_table, candidate_row_positions)
            raise AssetSelectionError(
                f"Asset policy {self.name!r} required exactly one valid asset for "
                f"snip_id={snip_id!r}, snip_product_key={self.snip_product_key!r}, "
                f"z_index=null; found {len(matches)}. Available assets: {available!r}."
            )
        return matches


def resolve_asset_row_groups(
    observation_table: pd.DataFrame,
    asset_table: pd.DataFrame,
    selector: AssetRowSelector,
) -> tuple[ResolvedAssetRowGroup, ...]:
    """Resolve observations in their existing order without mutating either table."""

    required_observation_columns = {"snip_id"}
    required_asset_columns = {
        "snip_id",
        "snip_product_key",
        "z_index",
        "processed_snip_path",
        "is_valid_snip",
    }
    missing_observation = sorted(
        required_observation_columns - set(observation_table.columns)
    )
    missing_asset = sorted(required_asset_columns - set(asset_table.columns))
    if missing_observation:
        raise AssetSelectionError(
            f"Observation table is missing required selector columns: {missing_observation}."
        )
    if missing_asset:
        raise AssetSelectionError(
            f"Asset table is missing required selector columns: {missing_asset}."
        )

    duplicate_observations = observation_table["snip_id"].duplicated(keep=False)
    if duplicate_observations.any():
        offending = (
            observation_table.loc[duplicate_observations, "snip_id"]
            .astype(str)
            .tolist()
        )
        raise AssetSelectionError(
            f"Observation table contains duplicate snip_id values: {offending!r}."
        )

    candidates_by_snip_id: dict[str, list[int]] = {}
    for asset_position, raw_snip_id in enumerate(asset_table["snip_id"].tolist()):
        if not isinstance(raw_snip_id, str) or not raw_snip_id:
            raise AssetSelectionError(
                f"Asset table row position {asset_position} has non-string or empty "
                f"snip_id={raw_snip_id!r}; IDs are required opaque strings."
            )
        candidates_by_snip_id.setdefault(raw_snip_id, []).append(asset_position)

    groups: list[ResolvedAssetRowGroup] = []
    for observation_position, raw_snip_id in enumerate(
        observation_table["snip_id"].tolist()
    ):
        if not isinstance(raw_snip_id, str) or not raw_snip_id:
            raise AssetSelectionError(
                f"Observation table row position {observation_position} has non-string or empty "
                f"snip_id={raw_snip_id!r}; IDs are required opaque strings."
            )
        snip_id = raw_snip_id
        candidate_positions = candidates_by_snip_id.get(snip_id, ())
        selected = tuple(
            int(position)
            for position in selector.select_row_positions(
                snip_id=snip_id,
                candidate_row_positions=candidate_positions,
                asset_table=asset_table,
            )
        )
        invalid_positions = [
            position
            for position in selected
            if position not in candidate_positions
            or position < 0
            or position >= len(asset_table)
        ]
        if invalid_positions:
            raise AssetSelectionError(
                f"Asset policy {selector.name!r} returned rows {invalid_positions!r} that do not "
                f"belong to snip_id={snip_id!r}."
            )
        groups.append(
            ResolvedAssetRowGroup(
                observation_row_position=observation_position,
                snip_id=snip_id,
                asset_row_positions=selected,
            )
        )
    return tuple(groups)


def selected_asset_paths(
    asset_table: pd.DataFrame, groups: Sequence[ResolvedAssetRowGroup]
) -> tuple[tuple[Path, ...], ...]:
    """Diagnostic helper retaining the ordered-row-list shape of the selector seam."""

    return tuple(
        tuple(
            Path(str(asset_table.iloc[position]["processed_snip_path"]))
            for position in group.asset_row_positions
        )
        for group in groups
    )
