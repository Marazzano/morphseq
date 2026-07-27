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

  1. Enumerates the plate's source children.
  2. Reads each source independently — via an injected ``read_source`` callable so the caller
     (and tests) control the one-read-per-source contract; this module does no ND2/TIFF I/O.
  3. Stamps each source with a distinct ``time_index`` BLOCK (ordinal across sources, ordered by
     ``declared_hpf`` then ``date``) and records ``start_age_hpf`` = ``parse_declared_hpf`` per
     source. A source that is itself a multi-timepoint timelapse keeps its internal time ordering
     inside its block, so time_index stays globally unique across the union.
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

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from data_pipeline.shared.identifiers.constructors import build_well_id, sanitize_experiment_id
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

# The 8-digit date is the split anchor of a collection child name; kept only for source ordering.
_CHILD_DATE_RE = re.compile(r"^(\d{8})_")


def _parse_child_date(child_name: str) -> str:
    """Return the 8-digit acquisition-date prefix of a collection child name (ordering only).

    ``"20260607_plate01_t45hpf" -> "20260607"``. Fails loud on a non-conforming name so a
    mis-shaped child cannot silently sort to an arbitrary position.
    """
    match = _CHILD_DATE_RE.match(str(child_name).strip())
    if not match:
        raise ValueError(
            f"collection_acquisition_union: cannot parse acquisition date from child "
            f"{child_name!r}. Expected a name like '{{date}}_{{plate_token}}[_{{event_label}}]' "
            "with date = 8 digits (e.g. 20260607_plate01_t45hpf)."
        )
    return match.group(1)


@dataclass(frozen=True)
class SourceChild:
    """One raw source of a plate — a collection child folder/file to be read exactly once.

    ``child_name`` is the raw child name (``{date}_{plate_token}[_{event_label}]``) from which
    plate_token, date and declared_hpf are parsed. ``scope`` is the microscope label (provenance).
    """

    child_name: str
    scope: str

    @property
    def date(self) -> str:
        return _parse_child_date(self.child_name)

    @property
    def declared_hpf(self) -> int | None:
        return parse_declared_hpf(self.child_name)

    @property
    def plate_token(self) -> str:
        return parse_plate_token(self.child_name)

    def sort_key(self) -> tuple[int, int, str]:
        """Order sources by declared_hpf then date (undeclared age sorts LAST, stable by date).

        ``(has_declared_hpf_flag, declared_hpf, date)``: sources WITH a declared age come first in
        age order; sources with no declared age (``sci``/absent) sort after, by date.
        """
        hpf = self.declared_hpf
        return (0, hpf, self.date) if hpf is not None else (1, 0, self.date)


def _restamp_experiment_and_well_id(
    df: pd.DataFrame, *, experiment_id: str
) -> pd.DataFrame:
    """Restamp ``experiment_id`` on every row and REBUILD ``well_id`` off it when present.

    A scope that resolves ``well_id`` at ingest (Keyence: ``build_well_id(per_source_exp, well)``)
    carries a well_id bound to the PER-SOURCE experiment_id. To make A01@t45 and A01@t72 share one
    well_id we rebuild it off the unioned experiment_id using the scope's own ``well_index`` (the
    raw well label, which is source-independent). A scope that has not resolved a well yet (YX1 —
    ``well_id`` attached later by the position→well mapping) has no ``well_id`` column, so nothing
    is rebuilt and the later mapping simply sees the unioned experiment_id.
    """
    out = df.copy()
    out["experiment_id"] = experiment_id
    if "well_id" in out.columns and "well_index" in out.columns:
        out["well_id"] = out["well_index"].map(
            lambda w: build_well_id(experiment_id, w)
        )
    return out


def union_collection_acquisition_inventories(
    *,
    collection_name: str,
    sources: Sequence[SourceChild],
    read_source: Callable[[SourceChild, str], pd.DataFrame],
) -> pd.DataFrame:
    """Assemble ONE acquisition inventory for a ``{coll}_{plate}`` plate from N source reads.

    Args:
        collection_name: the ``_coll`` collection directory name (namespaces the plate id).
        sources: the plate's source children — MUST all share one plate_token (asserted). Order is
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
            f"{collection_name!r}; a plate must have at least one source child."
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
    experiment_id = compose_collection_experiment_id(collection_name, sources[0].child_name)

    ordered = sorted(sources, key=SourceChild.sort_key)
    n_sources = len(ordered)  # per-well merge count — the only source fact that survives the seam.

    unioned_parts: list[pd.DataFrame] = []
    time_index_block_offset = 0
    seen_children: set[str] = set()
    # time_index value -> source ordinal that owns it; guards block disjointness WITHOUT a per-frame
    # source label (which we no longer emit). Two sources sharing a time_index would collide here.
    time_index_owner: dict[int, int] = {}

    for source_ordinal, source in enumerate(ordered):
        if source.child_name in seen_children:
            raise ValueError(
                f"collection_acquisition_union: source child {source.child_name!r} appears twice "
                f"for {experiment_id!r}. Each source must be read exactly once."
            )
        seen_children.add(source.child_name)

        # ── ONE read per source ────────────────────────────────────────────────────────────────
        per_source = read_source(source, experiment_id)
        if not isinstance(per_source, pd.DataFrame):
            raise TypeError(
                f"collection_acquisition_union: read_source for {source.child_name!r} returned "
                f"{type(per_source).__name__}, expected a pandas DataFrame."
            )
        if per_source.empty:
            raise ValueError(
                f"collection_acquisition_union: read_source for {source.child_name!r} returned an "
                "empty inventory — a source with no frames cannot contribute a timepoint."
            )
        if "time_index" not in per_source.columns:
            raise ValueError(
                f"collection_acquisition_union: per-source inventory for {source.child_name!r} is "
                "missing required column 'time_index'."
            )

        part = _restamp_experiment_and_well_id(per_source, experiment_id=experiment_id)

        # Block-offset the unioned time_index so sources never collide (snapshot → one block;
        # timelapse → a contiguous block of its own). The source's original per-source time_index is
        # NOT preserved as a column — SAM2 tracks over the unioned block, no source label survives.
        original_time_index = pd.to_numeric(part["time_index"], errors="raise").astype(int)
        part["time_index"] = original_time_index + time_index_block_offset

        # Age escape hatch: this source's declared age is its start_age_hpf (None → NaN, honestly).
        part["start_age_hpf"] = source.declared_hpf

        # Record which source ordinal owns each unioned time_index (disjointness guard below).
        for tv in part["time_index"].unique():
            time_index_owner.setdefault(int(tv), source_ordinal)

        unioned_parts.append(part)
        # Next source starts one past this source's max time_index (its full block width).
        time_index_block_offset = int(original_time_index.max()) + 1 + time_index_block_offset

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
