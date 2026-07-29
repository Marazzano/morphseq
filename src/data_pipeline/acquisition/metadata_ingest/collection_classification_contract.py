"""Collection-classify artifact contract — the single source of truth for "what a collection is".

This is the schema + authoritative validator for the small per-experiment JSON emitted by the
early classify step (``collection_classification.classify_experiment``). It is NOT a table — it is
one JSON object per experiment — so the contract is proportionate: it pins the required keys, their
types, and the ``start_age_by_time_index`` map shape (stringified-int keys → int/None ages). It is
still the ONE authoritative validator for this artifact (philosophy doc: one contract → one
validator).

Shape (see docs/EXPERIMENT_GROUP_PLATE_MODEL.md, "CLASSIFY ONCE, CONSUME EVERYWHERE")::

    { "experiment_id": "chem28c_coll_plate01",
      "is_collection": true,
      "sources": [
        {"file": "20250622_plate01_t28hpf", "raw_path": ".../t28hpf", "declared_hpf": 28, "time_index": 0},
        {"file": "20250623_plate01_t52hpf", "raw_path": ".../t52hpf", "declared_hpf": 52, "time_index": 1}
      ],
      "start_age_by_time_index": {"0": 28, "1": 52} }

Each ``sources`` record is the on-disk PROVENANCE for one source file — the single source of truth
disk-touching steps read instead of re-globbing the ``_coll`` dir: ``file`` (child name), ``raw_path``
(absolute), ``declared_hpf`` (int or None), and ``time_index`` (the union block ordinal).

For a NON-collection experiment the payload is inert (consumers ignore it and behave as today)::

    { "experiment_id": "20250912",
      "is_collection": false,
      "sources": [],
      "start_age_by_time_index": {} }

``start_age_by_time_index`` keys are the union ``time_index`` block-start ordinals (stringified for
JSON) and values are the declared age in hpf (int), or ``None`` when a source declared no age (honest
absence — not omitted). The ``time_index`` keys are consistent WITH the acquisition union's ordering
by construction; that guarantee lives in the producer (``collection_classification``), which reuses
``PlateSource.sort_key`` — this contract only checks the key/value SHAPE.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────────────
# Contract — required keys + payload types
# ─────────────────────────────────────────────────────────────────────────────────────

REQUIRED_COLLECTION_CLASSIFICATION_KEYS: tuple[str, ...] = (
    "experiment_id",
    "is_collection",
    "sources",
    "start_age_by_source_ordinal",
    # TODO(collection-legacy-age-map): drop once every consumer reads
    # start_age_by_source_ordinal. Kept required (not merely tolerated) so the artifact stays
    # readable by not-yet-migrated consumers; its keys are source ordinals despite the name.
    "start_age_by_time_index",
)


def validate_collection_classification(
    payload: dict,
    *,
    scope_label: str = "collection_classification",
) -> None:
    """Fail loud unless ``payload`` follows the collection-classify artifact contract.

    Checks the required keys exist, ``experiment_id`` is a non-empty string, ``is_collection`` is a
    real bool, ``sources`` is a list of strings, and ``start_age_by_time_index`` maps stringified-int
    keys to int-or-None ages. Consistency of the ``time_index`` keys WITH the acquisition union
    ordering is the producer's guarantee (it reuses ``PlateSource.sort_key``); this validator pins
    the SHAPE, not the ordering.

    Raises:
        ValueError: on any contract violation, with a message that names the fix.
    """
    for key in REQUIRED_COLLECTION_CLASSIFICATION_KEYS:
        if key not in payload:
            raise ValueError(
                f"[{scope_label}] missing required key {key!r}. Expected keys: "
                f"{list(REQUIRED_COLLECTION_CLASSIFICATION_KEYS)}."
            )

    experiment_id = payload["experiment_id"]
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError(
            f"[{scope_label}] 'experiment_id' must be a non-empty string, got {experiment_id!r}."
        )

    is_collection = payload["is_collection"]
    if not isinstance(is_collection, bool):
        raise ValueError(
            f"[{scope_label}] 'is_collection' must be a bool, got {type(is_collection).__name__} "
            f"({is_collection!r}). The classify step DECIDES collection-ness once; downstream "
            "consumes this bool, never re-derives it."
        )

    sources = payload["sources"]
    if not isinstance(sources, list):
        raise ValueError(
            f"[{scope_label}] 'sources' must be a list of per-source provenance records, got "
            f"{type(sources).__name__}."
        )
    # `source_ordinal` is WHICH SOURCE — the machine join key and the age map's key. `time_index`
    # is retained as a legacy alias holding the same value (see the TODO on the required keys).
    _SOURCE_RECORD_KEYS = (
        "file",
        "raw_path",
        "declared_hpf",
        "source_ordinal",
        "time_index",
    )
    for i, rec in enumerate(sources):
        if not isinstance(rec, dict):
            raise ValueError(
                f"[{scope_label}] 'sources[{i}]' must be a provenance record dict with keys "
                f"{_SOURCE_RECORD_KEYS}, got {type(rec).__name__}."
            )
        missing = [k for k in _SOURCE_RECORD_KEYS if k not in rec]
        if missing:
            raise ValueError(
                f"[{scope_label}] 'sources[{i}]' missing keys {missing}. Each source record needs "
                f"{_SOURCE_RECORD_KEYS} (the on-disk provenance a disk step reads)."
            )
        if not isinstance(rec["file"], str) or not isinstance(rec["raw_path"], str):
            raise ValueError(
                f"[{scope_label}] 'sources[{i}]' file/raw_path must be strings, got "
                f"file={rec['file']!r}, raw_path={rec['raw_path']!r}."
            )
        for ordinal_key in ("source_ordinal", "time_index"):
            value = rec[ordinal_key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"[{scope_label}] 'sources[{i}].{ordinal_key}' must be an int (the source "
                    f"ordinal — WHICH source, not a frame index), got {value!r}."
                )
        if rec["source_ordinal"] != rec["time_index"]:
            raise ValueError(
                f"[{scope_label}] 'sources[{i}]' has source_ordinal={rec['source_ordinal']!r} but "
                f"time_index={rec['time_index']!r}. `time_index` is a LEGACY ALIAS of "
                "source_ordinal in this artifact and must hold the same value; the merged frame "
                "coordinate lives in the unioned tables, not here."
            )
        if rec["declared_hpf"] is not None and not isinstance(rec["declared_hpf"], int):
            raise ValueError(
                f"[{scope_label}] 'sources[{i}].declared_hpf' must be int or None, got "
                f"{rec['declared_hpf']!r}."
            )

    # Both age maps are keyed by SOURCE ORDINAL. (The legacy name says time_index; its keys are
    # ordinals all the same — that mismatch is why it is being retired.)
    age_map = payload["start_age_by_source_ordinal"]
    legacy_age_map = payload["start_age_by_time_index"]
    for map_name, candidate in (
        ("start_age_by_source_ordinal", age_map),
        ("start_age_by_time_index", legacy_age_map),
    ):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"[{scope_label}] {map_name!r} must be a dict "
                "{stringified source_ordinal: age_hpf}, got "
                f"{type(candidate).__name__}."
            )
        for ordinal_key, age in candidate.items():
            if not (isinstance(ordinal_key, str) and ordinal_key.lstrip("-").isdigit()):
                raise ValueError(
                    f"[{scope_label}] {map_name!r} keys must be stringified ints (the source "
                    f"ordinals), got key {ordinal_key!r}."
                )
            if age is not None and not isinstance(age, int):
                raise ValueError(
                    f"[{scope_label}] {map_name}[{ordinal_key!r}] must be an int hpf or None "
                    f"(undeclared age), got {type(age).__name__} ({age!r})."
                )
    if age_map != legacy_age_map:
        raise ValueError(
            f"[{scope_label}] 'start_age_by_time_index' must mirror "
            f"'start_age_by_source_ordinal' while it remains for compatibility; got "
            f"{legacy_age_map!r} vs {age_map!r}."
        )

    # A single (non-collection) experiment carries no sources and no age map — consumers ignore it.
    if not is_collection and (sources or age_map or legacy_age_map):
        raise ValueError(
            f"[{scope_label}] {experiment_id!r} is_collection=False but carries "
            f"sources={sources!r} / start_age_by_source_ordinal={age_map!r}. A non-collection "
            "payload must be inert (empty sources + empty age maps)."
        )
