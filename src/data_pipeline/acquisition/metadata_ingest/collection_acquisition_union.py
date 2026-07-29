"""Collection ACQ UNION — assemble ONE acquisition inventory from N one-read-per-source reads.

This module owns the **acquisition union** seam of the experiment-collection / plate model
(see ``docs/EXPERIMENT_GROUP_PLATE_MODEL.md``, sections "Two kinds of merge", "age escape
hatch", and the ACQ UNION block). It implements **Merge-A (identity), NOT Merge-B (pixels)**:

    experiment_id = {collection}_{plate_token}                    (ONE id — processed together)
      frames from 20260607_plate01_t45hpf/ → time_index block 0, start_age_hpf 45   (read alone)
      frames from 20260608_plate01_t72hpf/ → time_index block 1, start_age_hpf 72   (read alone)
      well_id = {coll}_{plate}_A01   (shared across timepoints)

The relaxed invariant (spec §"The invariant relaxes mildly"):

    "one experiment = one inventory, ASSEMBLED from one-read-per-source."

Each source is read EXACTLY ONCE (this module NEVER fuses pixels — it consumes the per-source
inventory the scope extractor already built). The union:

  1. Enumerates the plate's sources.
  2. Reads each source independently — via an injected ``read_source`` callable so the caller
     (and tests) control the one-read-per-source contract; this module does no ND2/TIFF I/O.
  3. Stamps each source with a distinct ``time_index`` BLOCK (ordered by ``declared_hpf`` — each
     source must declare a DISTINCT age, else the order is ambiguous and we raise) and records
     ``start_age_hpf`` = ``parse_declared_hpf`` per source. A source that is itself a multi-timepoint
     timelapse keeps its internal time ordering inside its block, so time_index stays globally
     unique across the union.
  4. Restamps ``experiment_id`` → ``{coll}_{plate}`` on every row, and (where the scope resolves
     ``well_id`` at ingest, e.g. Keyence) REBUILDS ``well_id`` off the unioned experiment_id so
     A01@t45 and A01@t72 share ``well_id = {coll}_{plate}_A01``.
  5. UNIONs the per-source frames under the single experiment_id and stamps ``n_sources`` — the
     per-well COUNT of merged raw acquisitions — on every row.

**What survives the seam is a COUNT, not a source label.** SAM2 tracks over time across the merged
well exactly like a timelapse, so no downstream stage needs to know WHICH source a frame came from.
The only fact that cannot be re-derived downstream is HOW MANY raw acquisitions were merged into a
well (a merged snapshot's ``time_index`` 0,1 is indistinguishable from a timelapse's first two
frames). So the union records ``n_sources`` per well (see docs/EXPERIMENT_GROUP_PLATE_MODEL.md
"physical_embryo_id merge policy"). Per-frame source LABELS (``source_child`` / ``source_scope`` /
``source_time_index``) are deliberately NOT emitted — they existed only to feed the now-removed
physical_embryo bridge. Each scope's own ``source_*_path`` audit column rides through untouched (it
is the scope's I/O provenance, part of that scope's acquisition contract — NOT a merge source label).

Import direction: MAY import shared identifiers; MUST NOT import stages, Snakemake/tasks, stitch,
or scope-specific extraction logic (it consumes their OUTPUT, not their code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.collection_merge_primitives import (
    rebuild_well_id_for_plate,
    remap_source_time_indices,
)
from data_pipeline.shared.identifiers.constructors import sanitize_experiment_id
from data_pipeline.shared.identifiers.parsers import (
    compose_collection_experiment_id,
    parse_declared_hpf,
    parse_plate_token,
)

# Columns this union ADDS on top of whatever a scope's acquisition inventory already carries.
# The scope core/Tier-2 schema is untouched; these are the merge's own annotations. Per-frame
# source LABELS are intentionally absent — only a per-well COUNT (n_sources) survives the seam.
UNION_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "n_sources",         # PROVENANCE: per-well count of merged raw acquisitions (>=1; constant per well)
)

# The age escape hatch (age-axis annotation, not a source label): declared age per time_index block.
UNION_TIME_AXIS_COLUMNS: tuple[str, ...] = (
    "start_age_hpf",     # declared age at this source's time_index block (age escape hatch)
)

@dataclass(frozen=True)
class PlateSource:
    """One raw source of a plate — a folder/file under the _coll dir, read exactly once.

    ``source_id`` is the raw source name (``{date}_{plate_token}[_{event_label}]``) from which
    plate_token and declared_hpf are parsed. ``scope`` is the microscope label (provenance).

    The name's date prefix is NOT part of this class's behavior: ordering keys on the declared age
    alone (see ``sort_key``), so acquisition-date trivia never leaks into identity.
    """

    source_id: str
    scope: str

    @property
    def declared_hpf(self) -> int | None:
        return parse_declared_hpf(self.source_id)

    @property
    def plate_token(self) -> str:
        return parse_plate_token(self.source_id)

    def sort_key(self) -> tuple[int, int]:
        """Order sources by ``declared_hpf`` — the declared age IS the temporal order.

        ``(has_declared_hpf_flag, declared_hpf)``: sources WITH a declared age come first in age
        order; sources with no declared age (``sci``/absent) sort after.

        This key IS the rule that assigns ``source_ordinal``, so it must identify each source
        UNAMBIGUOUSLY. The acquisition date is deliberately NOT part of it: the declared age is the
        biological coordinate the collection is built on, and ordering by a filename date would leak
        acquisition trivia into identity. Two sources declaring the SAME age are therefore ambiguous
        — ``assert_source_order_unambiguous`` raises rather than guessing.
        """
        hpf = self.declared_hpf
        return (0, hpf) if hpf is not None else (1, 0)


def assert_source_order_unambiguous(sources: Sequence[PlateSource]) -> None:
    """Fail loud if two sources of one plate declare the SAME age.

    ``declared_hpf`` is the rule that assigns ``source_ordinal``, and that ordinal is load-bearing:
    the classify artifact's age map is keyed by it, and every source-aware join (scope metadata,
    acquisition inventory, position mapping) uses it. Two sources declaring the same age give no
    declared basis for which comes first, so ``sorted`` — being stable — would fall back to whatever
    order the filesystem glob returned. The same data could then get different ordinals on different
    runs, silently repointing ages and position mappings at the wrong source.

    We refuse to guess. Inventing a tiebreaker (a filename date, the child name) would make the run
    reproducible while still being an arbitrary choice about acquisition order that only a human can
    make — so this raises and names the colliding children instead.
    """
    by_hpf: dict[int | None, list[str]] = {}
    for source in sources:
        by_hpf.setdefault(source.declared_hpf, []).append(source.source_id)

    collisions = {hpf: names for hpf, names in by_hpf.items() if len(names) > 1}
    if collisions:
        detail = "; ".join(
            f"declared_hpf={hpf!r}: {sorted(names)}" for hpf, names in sorted(
                collisions.items(), key=lambda kv: (kv[0] is None, kv[0])
            )
        )
        raise ValueError(
            "collection_acquisition_union: source order is AMBIGUOUS — two or more sources of one "
            f"plate declare the same age ({detail}). source_ordinal is assigned by declared_hpf, "
            "and it keys the age map plus every source-aware join, so each source must declare a "
            "DISTINCT t<NN>hpf. Fix the raw child names so their declared ages distinguish them; "
            "the pipeline will not infer acquisition order from directory order."
        )


# The experiment/well_id restamp lives in `collection_merge_primitives.rebuild_well_id_for_plate` — ONE
# implementation shared with the scope-metadata union's Keyence re-key, so the two cannot diverge on
# how a source-bound well_id becomes plate-bound.


def union_collection_acquisition_inventories(
    *,
    collection_name: str,
    sources: Sequence[PlateSource],
    read_source: Callable[[PlateSource, str], pd.DataFrame],
) -> pd.DataFrame:
    """Assemble ONE acquisition inventory for a ``{coll}_{plate}`` plate from N source reads.

    Args:
        collection_name: the ``_coll`` collection directory name (namespaces the plate id).
        sources: the plate's sources — MUST all share one plate_token (asserted). Order is
            irrelevant; this function sorts by ``declared_hpf`` then ``date`` to assign time_index
            blocks.
        read_source: injected one-read-per-source reader. Called EXACTLY ONCE per source as
            ``read_source(source, experiment_id) -> per_source_inventory_df``. The returned frame
            is the scope's own (already-validated) acquisition inventory for that source. Injecting
            it keeps this module free of ND2/TIFF I/O and lets tests supply tiny synthetic frames —
            it is the seam that enforces "each source read exactly once".

    Returns:
        The unioned acquisition inventory under a single ``experiment_id = {coll}_{plate}``:
        rows from every source concatenated, ``time_index`` made globally distinct per source
        (block-offset by source ordinal), ``well_id`` shared across sources, ``start_age_hpf`` per
        time_index block, and ``n_sources`` (the per-well merge count) stamped on every row. Per-frame
        source LABELS are NOT emitted — the count is the only merge fact that survives the seam.

    The union performs Merge-A only: it concatenates independent reads, it NEVER fuses pixels or
    reconciles a shared elapsed clock (the age escape hatch handles stage via ``start_age_hpf`` per
    source). Each source keeps its own ``elapsed_time_s`` (rebased per position within that read).
    """
    if not sources:
        raise ValueError(
            f"collection_acquisition_union: no sources given for collection "
            f"{collection_name!r}; a plate must have at least one source."
        )

    # All sources of ONE plate must share the plate_token — otherwise this is not one experiment.
    plate_tokens = {s.plate_token for s in sources}
    if len(plate_tokens) != 1:
        raise ValueError(
            f"collection_acquisition_union: sources span multiple plate tokens {sorted(plate_tokens)} "
            f"for collection {collection_name!r}. Group children by plate_token BEFORE calling the "
            "union — one call assembles ONE plate's experiment_id."
        )

    # The plate token IS the id (MERGE model). Mint it once via the shared constructor (never here).
    experiment_id = compose_collection_experiment_id(collection_name, sources[0].source_id)

    # source_ordinal is assigned by declared age, so the ages must be distinct. Fail before any
    # read: an ambiguous ordinal would silently repoint the age map and every source-aware join.
    assert_source_order_unambiguous(sources)

    ordered = sorted(sources, key=PlateSource.sort_key)
    n_sources = len(ordered)  # per-well merge count — the only source fact that survives the seam.

    unioned_parts: list[pd.DataFrame] = []
    time_index_block_offset = 0
    seen_source_ids: set[str] = set()
    # time_index value -> source ordinal that owns it; guards block disjointness WITHOUT a per-frame
    # source label (which we no longer emit). Two sources sharing a time_index would collide here.
    time_index_owner: dict[int, int] = {}

    for source_ordinal, source in enumerate(ordered):
        if source.source_id in seen_source_ids:
            raise ValueError(
                f"collection_acquisition_union: source {source.source_id!r} appears twice "
                f"for {experiment_id!r}. Each source must be read exactly once."
            )
        seen_source_ids.add(source.source_id)

        # ── ONE read per source ────────────────────────────────────────────────────────────────
        per_source = read_source(source, experiment_id)
        if not isinstance(per_source, pd.DataFrame):
            raise TypeError(
                f"collection_acquisition_union: read_source for {source.source_id!r} returned "
                f"{type(per_source).__name__}, expected a pandas DataFrame."
            )
        if per_source.empty:
            raise ValueError(
                f"collection_acquisition_union: read_source for {source.source_id!r} returned an "
                "empty inventory — a source with no frames cannot contribute a timepoint."
            )
        if "time_index" not in per_source.columns:
            raise ValueError(
                f"collection_acquisition_union: per-source inventory for {source.source_id!r} is "
                "missing required column 'time_index'."
            )

        part = rebuild_well_id_for_plate(per_source, experiment_id)

        # Place this source's frames on the merged time axis via the SHARED helper (the scope
        # metadata union calls the same one, so the two artifacts cannot drift on what time_index
        # means). Snapshot → one slot; timelapse → a contiguous block of its own. raw_time_index
        # preserves the source-native value; the merged axis is dense even if the source's own
        # numbering is sparse or 1-based.
        part, time_index_block_offset = remap_source_time_indices(
            part,
            time_index_block_offset,
            scope_label=f"collection_acquisition_union[{source.source_id}]",
        )
        # The helper records the source-native value it remapped from, so the native→merged
        # relation is recoverable without re-reading the pre-remap frame.
        native_to_merged = dict(zip(part["raw_time_index"], part["time_index"]))

        # ``time_index_claimed`` is the raw time atom inside the acquisition-inventory CELL KEY, so
        # it must follow the SAME remap — otherwise two single-snapshot sources (both claiming 0)
        # collide on the cell key after the union. Mapped through the same native→merged relation
        # rather than offset arithmetic, so sparse source numbering stays consistent with
        # time_index. Absent for scopes that don't carry the atom.
        if "time_index_claimed" in part.columns:
            claimed = pd.to_numeric(part["time_index_claimed"], errors="raise").astype(int)
            unmapped = sorted(set(claimed) - set(native_to_merged))
            if unmapped:
                raise ValueError(
                    f"collection_acquisition_union: source {source.source_id!r} has "
                    f"time_index_claimed value(s) {unmapped} that do not appear in its time_index "
                    f"{sorted(native_to_merged)}. The raw time atom must track the frame axis it "
                    "belongs to; a divergence means the per-source inventory is inconsistent."
                )
            part["time_index_claimed"] = claimed.map(native_to_merged).astype(int)

        # Age escape hatch: this source's declared age is its start_age_hpf (None → NaN, honestly).
        part["start_age_hpf"] = source.declared_hpf

        # WHICH SOURCE — the machine join key, the same value the classify artifact and the scope
        # metadata union carry (both derive it from PlateSource.sort_key ordering). Distinct from
        # time_index: this source may span a whole block of merged timepoints.
        part["source_ordinal"] = source_ordinal

        # Record which source ordinal owns each unioned time_index (disjointness guard below).
        for tv in part["time_index"].unique():
            time_index_owner.setdefault(int(tv), source_ordinal)

        unioned_parts.append(part)
        # `time_index_block_offset` was already advanced to the next free merged slot by the
        # remapper (its block width is the DISTINCT timepoint count, so a sparse source leaves no
        # hole for the following source).

    unioned = pd.concat(unioned_parts, ignore_index=True)

    # n_sources is a per-well fact carried on every frame row — constant across the whole union
    # (grouping by plate = one experiment). See frame_inventory_contract.FRAME_INVENTORY_PROVENANCE.
    unioned["n_sources"] = n_sources

    # Sanity: the block-offset scheme must give every source a disjoint band of time_index values.
    _assert_time_index_blocks_disjoint(
        unioned, time_index_owner=time_index_owner, experiment_id=experiment_id
    )
    return unioned


def _assert_time_index_blocks_disjoint(
    df: pd.DataFrame, *, time_index_owner: dict[int, int], experiment_id: str
) -> None:
    """Fail loud unless every unioned ``time_index`` was claimed by exactly ONE source block.

    Guards the core invariant with the source-ordinal map built during the union (no per-frame source
    LABEL is emitted anymore). Every unioned time_index value must map to a single source ordinal; a
    value never recorded means a frame carries a time_index no block claimed — either would mean two
    independent reads collapsed onto one timepoint slot.
    """
    unclaimed = sorted({int(tv) for tv in df["time_index"].unique()} - set(time_index_owner))
    if unclaimed:
        raise ValueError(
            f"collection_acquisition_union: {experiment_id!r} has unioned time_index value(s) "
            f"{unclaimed} not claimed by any source block — the per-source time_index blocks "
            "overlap or were mis-offset. This must never happen; it means two independent reads "
            "collapsed onto one timepoint slot."
        )
