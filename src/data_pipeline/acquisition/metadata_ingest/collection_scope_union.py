"""Collection SCOPE UNION — one experiment-level scope_metadata assembled from N per-source reads.

This module owns **Step 1** of the collection worklist (see ``docs/COLLECTION_STEP_BY_STEP.md``).
It is the sibling of ``collection_acquisition_union`` (which unions the ACQUISITION inventory);
this one unions the **scope metadata**, the table ``map_positions`` + ``apply`` consume.

Why a REAL union (and not the unioned acquisition inventory written into the scope_csv slot):
``scope_metadata`` and ``acquisition_inventory`` are NOT interchangeable. Each keeps the columns
ITS consumer ingests:

  * ``scope_metadata``       → ``map_positions`` + ``apply``: per-position GEOMETRY
    (``raw_position_label``, ``x_um``, ``y_um``) + ``channel`` (the clean token) + ``raw_channel_name``.
  * ``acquisition_inventory`` → ``build_keyence_stitch_map`` + ``materialize``: image-level
    ``source_path`` + the acquisition axes.

Dumping the acquisition inventory into the scope_csv slot produced a table with no
``channel``/geometry columns, which is what crashed ``apply``. Both artifacts originate from the
SAME per-source read; each keeps its own columns.

**Acquisition facts DIFFER per source — never assume shared.** Each source is its own acquisition
with its OWN ``x_um``/``y_um`` (the plate is re-seated across the gap → a different stage frame), its
OWN calibration (``micrometers_per_pixel``, image dims) and its OWN timing (``absolute_start_time``,
``frame_interval_s``). There is no "the plate's scope metadata" — there is each source's, and they
legitimately differ. So this union is **per-source-lossless**: it CONCATENATES each source's real
rows and TAGS them, and it must never dedup/collapse geometry as if it were shared::

    unioned scope_metadata:
      time_index 0 → t28's rows (t28's OWN x/y, calibration, timing)
      time_index 1 → t52's rows (t52's OWN x/y, calibration, timing)   ← distinct, preserved

Step 2 recovers each source's block with ``groupby("time_index")`` — nothing is reconstructed,
because nothing was collapsed.

**Canonical source identity** (the keystone — the same columns in every source-aware artifact):

    source_ordinal ← WHICH SOURCE (0, 1, 2 …). The machine JOIN KEY; keys the age map.
    source_id      ← the source child name (e.g. 20250622_plate01_t28hpf). Readable provenance.
    source_path    ← the raw path (provenance + pixel access).

These are READ from the collection-classify artifact (``sources[].source_ordinal`` / ``.file`` /
``.raw_path``) — this module never re-globs the ``_coll`` dir.

**Time is separate from identity.** ``source_ordinal`` says which source; ``time_index`` says which
merged frame. One ``source_ordinal`` can span MANY ``time_index`` values (a timelapse source), so
they are different columns and neither is derived from the other. Each source's own frame numbering
is placed on the merged axis by the SHARED ``remap_source_time_indices`` helper, which the
acquisition union also calls — so both artifacts agree on what ``time_index`` means:

    raw_time_index ← the source-native frame index, unchanged (audit)
    time_index     ← the canonical merged frame index (contiguous across the plate)

Import direction: MAY import shared identifiers + the collection artifact reader whose OUTPUT it
consumes; MUST NOT import stages, Snakemake/tasks, or scope-specific extraction logic (the per-source
reader is INJECTED, exactly like the acquisition union's ``read_source``).
"""

from __future__ import annotations

from typing import Callable, Sequence

import pandas as pd

from data_pipeline.acquisition.metadata_ingest.collection_time_axis import (
    remap_source_time_indices,
)

# The canonical source-identity columns this union stamps on every row. `source_ordinal` is the
# JOIN KEY (which source); `source_id` is readable identity; `source_path` is provenance/pixel
# access. All three ride in every source-aware artifact so the collection artifact,
# scope_metadata, position_mapping and acquisition_inventory all speak the SAME source key.
#
# NOTE `time_index` is deliberately NOT here: it is the merged FRAME coordinate, not source
# identity. One source_ordinal can span many time_index values (a timelapse source).
SOURCE_IDENTITY_COLUMNS: tuple[str, ...] = ("source_id", "source_path", "source_ordinal")


def union_collection_scope_metadata(
    *,
    experiment_id: str,
    sources: Sequence[dict],
    read_source: Callable[[dict, str], pd.DataFrame],
    rekey_to_plate: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Assemble ONE experiment-level scope_metadata from N per-source scope reads.

    Args:
        experiment_id: the merged ``{collection}_{plate_token}`` id (the PLATE). Every row is
            restamped with it so downstream sees one experiment.
        sources: the collection artifact's ``sources`` provenance records — each a dict with
            ``file`` / ``raw_path`` / ``source_ordinal`` (``declared_hpf`` rides along unused here).
            Read from the artifact; NOT re-globbed and NOT re-derived. Processed in ordinal order.
        read_source: injected one-read-per-source reader, called EXACTLY ONCE per source as
            ``read_source(source_record, experiment_id) -> per_source_scope_metadata_df``. Injecting
            it keeps this module free of TIFF/ND2 I/O and lets tests supply tiny synthetic frames.
        rekey_to_plate: injected PER-SCOPE adapter, applied to each source's block as
            ``rekey_to_plate(block, experiment_id)``. Scopes differ in WHAT identity they resolve at
            ingest, so re-keying source-bound ids to the plate is a per-scope concern and does NOT
            belong in this scope-agnostic union: Keyence mints ``well_index``/``well_id``/``image_id``
            from the id it is handed (all source-bound → must be re-keyed), while YX1 mints none of
            them (``well_id`` is attached later by ``apply`` → nothing to re-key). ``None`` = no
            re-key needed (the YX1 shape).

    Returns:
        The concatenated scope_metadata: every source's own rows (its own geometry, calibration and
        timing) stamped with ``source_id`` / ``source_path`` / ``time_index``, under one
        ``experiment_id``. Per-source-lossless by construction.

    Raises:
        ValueError: if ``sources`` is empty, a record is missing a required key, a per-source read
            comes back empty, or two sources claim the same ``time_index``.
    """
    if not sources:
        raise ValueError(
            f"collection_scope_union: no sources given for {experiment_id!r}. A collection plate "
            "must have at least one source; read them from the collection-classify artifact's "
            "'sources' (never re-glob the _coll dir)."
        )

    parts: list[pd.DataFrame] = []
    seen_source_ordinal: dict[int, str] = {}
    # The merged time axis grows as sources are appended; each source's block starts here.
    running_time_offset = 0

    # Sources are processed in ARTIFACT ORDER (by source_ordinal), so the merged time axis follows
    # the same ordering the acquisition union uses — that is what keeps the two artifacts in step.
    for record in sorted(sources, key=lambda r: int(r["source_ordinal"])):
        missing = [
            k for k in ("file", "raw_path", "source_ordinal") if k not in record
        ]
        if missing:
            raise ValueError(
                f"collection_scope_union: source record {record!r} is missing {missing}. Each "
                "record comes from the collection-classify artifact and must carry "
                "file/raw_path/source_ordinal."
            )

        source_id = str(record["file"])
        source_path = str(record["raw_path"])
        source_ordinal = int(record["source_ordinal"])

        # source_ordinal is ASSIGNED by the artifact, so a collision means the artifact is
        # malformed — fail here rather than silently merging two acquisitions into one source.
        if source_ordinal in seen_source_ordinal:
            raise ValueError(
                f"collection_scope_union: sources {seen_source_ordinal[source_ordinal]!r} and "
                f"{source_id!r} both claim source_ordinal={source_ordinal} for {experiment_id!r}. "
                "Each source must own a distinct ordinal (the artifact assigns it)."
            )
        seen_source_ordinal[source_ordinal] = source_id

        # ── ONE read per source ────────────────────────────────────────────────────────────────
        per_source = read_source(record, experiment_id)
        if not isinstance(per_source, pd.DataFrame):
            raise TypeError(
                f"collection_scope_union: read_source for {source_id!r} returned "
                f"{type(per_source).__name__}, expected a pandas DataFrame."
            )
        if per_source.empty:
            raise ValueError(
                f"collection_scope_union: read_source for {source_id!r} returned empty scope "
                "metadata — a source with no positions cannot contribute a timepoint."
            )

        part = per_source.copy()
        # The PLATE id on every row: the collection is ONE experiment.
        part["experiment_id"] = experiment_id

        # Per-scope re-key: whatever identity THIS scope minted against the source id gets rebound
        # to the plate. What needs rebinding differs by scope, so the scope owns the rule (see the
        # rekey_to_plate arg); this union stays scope-agnostic.
        if rekey_to_plate is not None:
            part = rekey_to_plate(part, experiment_id)

        # NAME COLLISION, resolved explicitly. The scope extractors already emit a per-row
        # `source_file` = the IMAGE path (one tiff/plane per row) — pure provenance, no consumers.
        # The collection's source identity needs a per-SOURCE path (the child dir), and the worklist
        # calls that `source_path`. Those are different grains, so we keep both under exact names:
        #   source_image_path ← the per-row image path (renamed from the extractor's `source_file`)
        #   source_path       ← the per-SOURCE child dir (stamped below)
        # Renaming the extractors' column outright is a separate cleanup (it would touch YX1 +
        # fixtures); doing it here keeps the grain distinction honest without that churn.
        if "source_file" in part.columns:
            part = part.rename(columns={"source_file": "source_image_path"})
        # Stamp the canonical source identity. `source_ordinal` is WHICH SOURCE (the join key and
        # the age map's key); `source_id`/`source_path` are provenance.
        part["source_id"] = source_id
        part["source_path"] = source_path
        part["source_ordinal"] = source_ordinal

        # Place this source's own frames on the merged time axis: raw_time_index keeps the
        # source-native value, time_index becomes the merged coordinate. The SHARED helper does it
        # (the acquisition union calls the same one) so the two artifacts cannot drift on what
        # time_index means. A timelapse source therefore spans a contiguous RANGE of time_index,
        # exactly as the worklist specifies — it is NOT collapsed onto the source ordinal.
        part, running_time_offset = remap_source_time_indices(
            part, running_time_offset, scope_label=f"collection_scope_union[{source_id}]"
        )

        parts.append(part)

    # Concat, not merge: each block keeps its own acquisition facts. Columns present in only some
    # sources (scope drift) become NaN for the others rather than dropping a source's real data.
    unioned = pd.concat(parts, ignore_index=True, sort=False)
    return unioned
