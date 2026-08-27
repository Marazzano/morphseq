"""Experiment COLLECTION expansion — the plural counterpart to ``resolve_experiment_id``.

A collection is a coarse *run handle* that must be expanded into a flat list of
``experiment_id``s before anything runs. Both the SGE array submitter and the
pipeline need this same expansion (a plate ≈ an experiment_id to the pipeline), so
it is ONE shared primitive, not an SGE hack.

    mixed input (ids + collections) ──▶ resolve_experiment_ids ──▶ flat experiment_id list

DRY: this wrapper only decides *which folders to feed* the singular resolver / the
collection composer. It contains NO id-construction logic — every id is minted by
``shared/identifiers`` (``compose_collection_experiment_id``) or the singular
``resolve_experiment_id``. Nothing can drift.

See docs/EXPERIMENT_GROUP_PLATE_MODEL.md ("Collection as a RUN TARGET").
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.acquisition.metadata_ingest.experiment_identity import (
    resolve_experiment_id,
)
from data_pipeline.shared.identifiers import (
    compose_collection_experiment_id,
    is_collection,
    is_collection_plate_id,
    parse_collection_name_from_plate_id,
    parse_plate_token,
)


def resolve_experiment_ids(
    entries: list[str],
    raw_root: str | Path,
    microscope: str = "Keyence",
) -> list[str]:
    """Expand a mixed list of experiment_ids and collections into a flat id list.

    For each entry:
      - if it ends in ``_coll`` (``is_collection``), it is a COLLECTION: glob its raw
        children ``raw_root/<entry>/*``, group them by ``plate_token``, and yield one
        ``{collection}_{plate_token}`` experiment_id per DISTINCT plate. The different
        ``t<NN>hpf`` events of a plate MERGE into one id (they share the plate token).
      - else it is a literal ``experiment_id`` → passed through the singular
        ``resolve_experiment_id`` unchanged.

    Results are DEDUPED while PRESERVING first-seen order.

    ``microscope`` only affects passthrough entries (how the singular resolver reads a
    path); collection expansion is scope-agnostic (children are matched by plate token,
    per the MVP scope rule — a collection lives inside one scope dir).
    """
    root = Path(raw_root)
    resolved: list[str] = []
    seen: set[str] = set()

    def _emit(experiment_id: str) -> None:
        if experiment_id not in seen:
            seen.add(experiment_id)
            resolved.append(experiment_id)

    for entry in entries:
        name = str(entry).strip()
        if not name:
            continue

        if is_collection(name):
            for experiment_id in _expand_collection(name, root):
                _emit(experiment_id)
        else:
            _emit(resolve_experiment_id(name, microscope, explicit_experiment_id=name))

    return resolved


def _discover_collection_plates(
    collection_name: str, raw_root: Path, *, scope_label: str
) -> dict[str, list[str]]:
    """THE single filesystem interpretation of a ``_coll`` dir: plate experiment_id → its sources.

    This is the ONE authority. Both public callers — flat run-target resolution
    (``resolve_experiment_ids``) and per-plate source discovery (``discover_plate_sources``) — go
    through it, so they cannot disagree about child filtering, parsing, grouping, or ordering. They
    are two callers of one implementation, not two siblings that happen to agree today.

    Filesystem rules, stated once:
      * Keyence sources are DIRECTORIES (the dir name carries the plate token); YX1 sources are
        ``.nd2`` FILES (the stem carries it). Hence ``name`` for dirs, ``stem`` for files.
      * A child that carries no plate token (a stray sidecar, ``Thumbs.db``, notes) is SKIPPED.
      * Iteration is over ``sorted(...)`` so grouping and ordering are deterministic.

    Name interpretation is delegated to ``shared/identifiers`` (``parse_plate_token``,
    ``compose_collection_experiment_id``) — this layer reads the filesystem, identifiers read names.

    Returns ``{experiment_id: [source_id, ...]}``, insertion-ordered by first-seen plate.
    """
    collection_dir = raw_root / collection_name
    if not collection_dir.is_dir():
        raise ValueError(
            f"{scope_label}: collection {collection_name!r} is marked _coll but "
            f"{collection_dir!s} is not a directory. Cannot discover its plates."
        )

    plates: dict[str, list[str]] = {}
    for child in sorted(collection_dir.iterdir()):
        source_id = child.name if child.is_dir() else child.stem
        try:
            parse_plate_token(source_id)
            experiment_id = compose_collection_experiment_id(collection_name, source_id)
        except ValueError:
            continue  # no plate token — not a source of this collection
        plates.setdefault(experiment_id, []).append(source_id)

    if not plates:
        raise ValueError(
            f"{scope_label}: collection {collection_name!r} at {collection_dir!s} contains no "
            "children matching the collection grammar {date}_{plate_token}[_{event_label}]. "
            "Nothing to discover."
        )
    return plates


def discover_plate_sources(experiment_id: str, raw_root: Path) -> tuple[str, list[str]]:
    """Given a ``{collection}_{plate_token}`` experiment_id, return the sources that MERGE into it.

    The INVERSE of ``compose_collection_experiment_id``. Used ONLY when building the collection
    provenance artifact — every later step reads that artifact instead of re-discovering (the
    keystone rule).

    Returns ``(collection_name, [source_id, ...])`` deterministically ordered. Fails loud if the id
    is not a collection plate id, the dir is missing, or no source composes to it.
    """
    if not is_collection_plate_id(experiment_id):
        raise ValueError(
            f"discover_plate_sources: {experiment_id!r} is not a collection plate id "
            f"(expected '{{collection}}_coll_{{plate_token}}'). Not a merged collection experiment."
        )
    collection_name = parse_collection_name_from_plate_id(experiment_id)
    plates = _discover_collection_plates(
        collection_name, raw_root, scope_label="discover_plate_sources"
    )

    sources = plates.get(experiment_id)
    if not sources:
        raise ValueError(
            f"discover_plate_sources: no sources under {raw_root / collection_name!s} compose to "
            f"{experiment_id!r}. Discovered plates: {sorted(plates)}. Check the plate token in the "
            "source names."
        )
    return collection_name, sources


def _expand_collection(collection_name: str, raw_root: Path) -> list[str]:
    """Expand one ``_coll`` directory into its distinct plate experiment_ids.

    A thin view over the ONE discovery authority — the plate ids ARE its keys, so run-target
    resolution and provenance generation see identical membership by construction.
    """
    return list(
        _discover_collection_plates(
            collection_name, raw_root, scope_label="resolve_experiment_ids"
        )
    )
