"""Builder for the `physical_embryo_registry` product — the identity-origination boundary.

Given a per-well ``frame_masks`` table (the DETECTED/tracked set — NOT ``valid_masks``;
the registry lives upstream of the valid/invalid QC split so it never inherits a QC
filter), mint ONE row per distinct ``(well_id, track_id)``. The mint chain runs once
per animal here — not once per mask, the way the legacy snip crop loop did it.

The mint chain (named functions, no inline arithmetic), relocated out of snip_processing:

    raw_track_index    = parse_embryo_local_track_id(track_id)         # "..._track0000" → 0
    local_embryo_index = track_index_to_embryo_index(raw_track_index)  # 0 → 1 (one-based)
    physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.physical_embryo_registry.physical_embryo_registry_contract import (
    PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS,
    empty_physical_embryo_registry,
)
from data_pipeline.segmentation.physical_embryo_registry.validate_physical_embryo_registry import (
    validate_physical_embryo_registry,
)
from data_pipeline.shared.identifiers import (
    build_physical_embryo_id,
    parse_embryo_local_track_id,
    track_index_to_embryo_index,
)


def build_physical_embryo_registry(frame_masks: pd.DataFrame) -> pd.DataFrame:
    """Mint one registry row per distinct ``(well_id, track_id)`` in ``frame_masks``.

    No-mask placeholder rows (``track_id`` is NA) carry no tracked entity and are
    dropped. All other detected tracks are registered regardless of ``is_valid_mask`` —
    discovery is a tracking fact, not a quality verdict. The result is validated before
    return (fail loud at the boundary).
    """
    required = ["well_id", "track_id", "experiment_id", "track_id_source"]
    missing = [col for col in required if col not in frame_masks.columns]
    if missing:
        raise ValueError(
            f"build_physical_embryo_registry: frame_masks missing required column(s): "
            f"{', '.join(missing)}"
        )

    detected = frame_masks[frame_masks["track_id"].notna()].copy()
    if detected.empty:
        return empty_physical_embryo_registry()

    # One row per distinct animal: distinct (well_id, track_id), carrying provenance.
    distinct = (
        detected[required]
        .drop_duplicates(subset=["well_id", "track_id"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, object]] = []
    for _, row in distinct.iterrows():
        well_id = str(row["well_id"])
        track_id = str(row["track_id"])
        raw_track_index = parse_embryo_local_track_id(track_id)
        local_embryo_index = track_index_to_embryo_index(raw_track_index)
        physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
        rows.append(
            {
                "physical_embryo_id": physical_embryo_id,
                "experiment_id": str(row["experiment_id"]),
                "well_id": well_id,
                "local_embryo_index": local_embryo_index,
                "track_id": track_id,
                "track_id_source": str(row["track_id_source"]),
            }
        )

    registry = pd.DataFrame(rows, columns=PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS)
    validate_physical_embryo_registry(registry)
    return registry


def merge_physical_embryo_registry(shards: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-well registry shards and ENFORCE global physical_embryo_id uniqueness.

    Per-well minting is independently correct (``local_embryo_index`` is well-scoped),
    so the merge is a pure concat — but the merged table's validator is what turns the
    by-construction global-uniqueness invariant from a hope into a promise.
    """
    if not shards:
        return empty_physical_embryo_registry()
    merged = pd.concat(shards, ignore_index=True)
    validate_physical_embryo_registry(merged)
    return merged
