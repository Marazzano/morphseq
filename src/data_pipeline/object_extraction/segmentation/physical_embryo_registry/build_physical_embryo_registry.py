"""Builder for the `physical_embryo_registry` product — the identity-origination boundary.

Given a per-well ``frame_masks`` table (the DETECTED/tracked set — NOT ``valid_masks``;
the registry lives upstream of the valid/invalid QC split so it never inherits a QC
filter) and the collection PROVENANCE artifact (for the ``n_sources`` merge count), mint ONE
row per distinct animal. The mint chain runs once per animal here — not once per mask, the
way the legacy snip crop loop did it.

The mint chain (named functions, no inline arithmetic), relocated out of snip_processing:

    raw_track_index    = parse_embryo_local_track_id(track_id)         # "..._track0000" → 0
    local_embryo_index = track_index_to_embryo_index(raw_track_index)  # 0 → 1 (one-based)
    physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)

Merge policy (LOCKED 2026-07-25 — see EXPERIMENT_GROUP_PLATE_MODEL.md "physical_embryo_id
merge policy"). Snapshot acquisitions merged under one well arrive at the tracker as
consecutive ``time_index`` values, so tracking runs over ONE time-ordered series per well —
there is no cross-source ``track_id`` collision. The only fact that cannot be re-derived
downstream is HOW MANY raw acquisitions were merged; the provenance artifact states that as
``len(sources)``. Per well, with ``n_tracks`` = distinct ``track_id``:

    n_sources == 1                    → NORMAL    one physical_embryo_id per track (legacy)
    n_sources > 1  AND  n_tracks == 1 → BRIDGE    ONE physical_embryo_id across timepoints
    n_sources > 1  AND  n_tracks > 1  → FRACTURE  disjoint _e blocks per source (no guessing)

Grammar is UNCHANGED (``physical_embryo_id = {well_id}_e{NN}``, one-based). FRACTURE just
offsets the ``_e`` index per source. Source membership per track is derived from the
snapshot fact that each source is one ``time_index`` block: group the well's tracks by
their ``time_index`` into ``n_sources`` groups.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.object_extraction.segmentation.physical_embryo_registry.physical_embryo_registry_contract import (
    EmbryoMergePolicy,
    PHYSICAL_EMBRYO_REGISTRY_REQUIRED_COLUMNS,
    empty_physical_embryo_registry,
)
from data_pipeline.object_extraction.segmentation.physical_embryo_registry.validate_physical_embryo_registry import (
    validate_physical_embryo_registry,
)
from data_pipeline.shared.identifiers import (
    build_physical_embryo_id,
    parse_embryo_local_track_id,
    track_index_to_embryo_index,
)

# frame_inventory column carrying the per-well count of merged raw acquisitions (Agent B's
# shared contract: integer >= 1, constant within a well_id; 1 = single acquisition).
N_SOURCES_COLUMN = "n_sources"


# ─────────────────────────────────────────────────────────────────────────────────────
# Per-well mint — the merge policy branch
# ─────────────────────────────────────────────────────────────────────────────────────
def _resolve_merge_policy(n_sources: int, n_tracks: int) -> EmbryoMergePolicy:
    """The single visible policy decision (see EmbryoMergePolicy / the model spec)."""
    if n_sources <= 1:
        return EmbryoMergePolicy.NORMAL
    if n_tracks == 1:
        return EmbryoMergePolicy.BRIDGE
    return EmbryoMergePolicy.FRACTURE


def _mint_well_rows(
    well_id: str,
    well_tracks: pd.DataFrame,
    n_sources: int,
) -> list[dict[str, object]]:
    """Assign ``physical_embryo_id`` to every distinct track of one well under the policy.

    ``well_tracks`` is the well's distinct tracks (one row per ``track_id``), each carrying
    ``experiment_id``, ``track_id``, ``track_id_source`` and ``time_index`` (its source
    block, used only for the FRACTURE offset). ``n_sources`` comes from collection provenance.
    """
    n_tracks = int(well_tracks["track_id"].nunique())
    policy = _resolve_merge_policy(n_sources, n_tracks)

    if policy is EmbryoMergePolicy.NORMAL:
        # Legacy path: local_embryo_index derives from the track index, exactly as before.
        return [
            _row(
                well_id,
                row,
                track_index_to_embryo_index(parse_embryo_local_track_id(str(row["track_id"]))),
                policy,
                n_sources,
            )
            for _, row in well_tracks.iterrows()
        ]

    if policy is EmbryoMergePolicy.BRIDGE:
        # ONE animal spanning the well's timepoints: every row shares local_embryo_index 1,
        # hence one physical_embryo_id (like a timelapse — the only correspondence possible).
        return [_row(well_id, row, 1, policy, n_sources) for _, row in well_tracks.iterrows()]

    # FRACTURE: disjoint _e blocks per source so every animal gets a distinct id and none is
    # claimed to span time. Each source is one time_index block; order sources by time_index,
    # offset each source's block past the previous. Within a source, order animals by their
    # tracker index for determinism.
    local_index_by_track: dict[str, int] = {}
    offset = 0
    for _time_index, source_tracks in well_tracks.groupby("time_index", sort=True):
        ordered = sorted(
            (str(r["track_id"]) for _, r in source_tracks.iterrows()),
            key=parse_embryo_local_track_id,
        )
        for position, track_id in enumerate(ordered):
            local_index_by_track[track_id] = offset + position + 1  # one-based, past prior sources
        offset += len(ordered)

    return [
        _row(well_id, row, local_index_by_track[str(row["track_id"])], policy, n_sources)
        for _, row in well_tracks.iterrows()
    ]


def _row(
    well_id: str,
    row: pd.Series,
    local_embryo_index: int,
    policy: EmbryoMergePolicy,
    n_sources: int,
) -> dict[str, object]:
    physical_embryo_id = build_physical_embryo_id(well_id, local_embryo_index)
    return {
        "physical_embryo_id": physical_embryo_id,
        "experiment_id": str(row["experiment_id"]),
        "well_id": well_id,
        "local_embryo_index": local_embryo_index,
        "track_id": str(row["track_id"]),
        "track_id_source": str(row["track_id_source"]),
        "merge_policy": policy.value,
        "n_sources": int(n_sources),
    }


# ─────────────────────────────────────────────────────────────────────────────────────
# Builder + merge
# ─────────────────────────────────────────────────────────────────────────────────────


def _n_sources_from_provenance(collection_provenance: dict) -> int:
    """How many raw acquisitions merged into this experiment, per its provenance artifact.

    ``len(sources)`` IS the count: every experiment declares its sources (a single experiment is a
    collection of ONE), so this needs no default and no legacy backfill.
    """
    sources = collection_provenance.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            "build_physical_embryo_registry: collection provenance declares no 'sources'. Every "
            "experiment's provenance artifact lists its sources (a single experiment is a "
            "collection of ONE); n_sources is len(sources)."
        )
    return len(sources)


def build_physical_embryo_registry(
    frame_masks: pd.DataFrame,
    collection_provenance: dict,
) -> pd.DataFrame:
    """Mint one registry row per distinct ``(well_id, track_id)`` in ``frame_masks``.

    No-mask placeholder rows (``track_id`` is NA) carry no tracked entity and are
    dropped. All other detected tracks are registered regardless of ``is_valid_mask`` —
    discovery is a tracking fact, not a quality verdict. The result is validated before
    return (fail loud at the boundary).

    ``n_sources`` — how many raw acquisitions merged into this experiment — selects the
    ``EmbryoMergePolicy`` (NORMAL / BRIDGE / FRACTURE). It comes from the COLLECTION PROVENANCE
    artifact (``len(sources)``), which is its owner and states it authoritatively.

    It is deliberately NOT read from ``frame_inventory``. n_sources is an experiment/well-grain
    constant, so carrying it on every frame row made frame_inventory a courier for a fact it does
    not own — and every experiment now declares a provenance artifact (a single experiment is a
    collection of ONE source), so the count is always available at its source. A single experiment
    is ``n_sources == 1`` and mints exactly as it always has.
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

    # The merge count is an experiment-level declared fact; every well of this experiment shares it.
    n_sources = _n_sources_from_provenance(collection_provenance)

    # One row per distinct animal: distinct (well_id, track_id), carrying provenance +
    # time_index (kept only for the FRACTURE source-block offset). A track sits in one
    # time_index block for the snapshot MVP; take its first frame's time_index.
    carry = required + (["time_index"] if "time_index" in detected.columns else [])
    distinct = (
        detected[carry]
        .drop_duplicates(subset=["well_id", "track_id"])
        .reset_index(drop=True)
    )
    if "time_index" not in distinct.columns:
        distinct["time_index"] = 0

    rows: list[dict[str, object]] = []
    for well_id, well_tracks in distinct.groupby("well_id", sort=False):
        well_id = str(well_id)
        # No per-well coverage check needed: n_sources is an experiment-level declared fact, so it
        # applies to every well of the experiment by construction.
        rows.extend(_mint_well_rows(well_id, well_tracks, n_sources))

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
