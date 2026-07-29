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


def find_collection_plate_sources(experiment_id: str, raw_root: Path) -> tuple[str, list[str]]:
    """Given a ``{collection}_{plate_token}`` experiment_id, return its source children.

    The INVERSE of ``compose_collection_experiment_id``: acquisition ingest needs, for one
    merged plate experiment, the list of raw source-child names (each a
    ``{date}_{plate_token}[_{event_label}]`` folder/file) that MERGE into it. Globs the
    ``_coll`` dir and keeps the children whose composed id equals ``experiment_id``.

    Returns ``(collection_name, [source_id, ...])`` sorted deterministically. The
    membership test reuses ``compose_collection_experiment_id`` so it can never drift from
    how ids were minted (DRY). Fails loud if the id is not a collection plate id or no
    source children match.
    """
    if not is_collection_plate_id(experiment_id):
        raise ValueError(
            f"find_collection_plate_sources: {experiment_id!r} is not a collection plate id "
            f"(expected '{{collection}}_coll_{{plate_token}}'). Not a merged collection experiment."
        )
    collection_name = parse_collection_name_from_plate_id(experiment_id)
    collection_dir = raw_root / collection_name
    if not collection_dir.is_dir():
        raise ValueError(
            f"find_collection_plate_sources: collection dir {collection_dir!s} for "
            f"{experiment_id!r} is not a directory."
        )

    children: list[str] = []
    for child in sorted(collection_dir.iterdir()):
        source_id = child.name if child.is_dir() else child.stem
        try:
            composed = compose_collection_experiment_id(collection_name, source_id)
        except ValueError:
            continue  # child lacks a plate token (stray sidecar) — skip
        if composed == experiment_id:
            children.append(source_id)

    if not children:
        raise ValueError(
            f"find_collection_plate_sources: no source children under {collection_dir!s} "
            f"compose to {experiment_id!r}. Check the plate token in the child names."
        )
    return collection_name, children


def _expand_collection(collection_name: str, raw_root: Path) -> list[str]:
    """Expand one ``_coll`` directory into its distinct plate experiment_ids.

    Globs ``raw_root/<collection_name>/*`` (children are folders on Keyence, files on
    YX1 — both carry a plate token in their own name), groups by ``plate_token``, and
    mints one ``{collection}_{plate_token}`` id per distinct plate via the shared
    composer. Order follows first-seen plate token (sorted children → deterministic).
    """
    collection_dir = raw_root / collection_name
    if not collection_dir.is_dir():
        raise ValueError(
            f"resolve_experiment_ids: collection {collection_name!r} is marked _coll but "
            f"{collection_dir!s} is not a directory. Cannot expand its plates."
        )

    experiment_ids: list[str] = []
    seen: set[str] = set()
    for child in sorted(collection_dir.iterdir()):
        # Keyence children are folders (name IS the child name); YX1 children are .nd2
        # files (the stem carries the token). Use the stem for files, name for dirs.
        source_id = child.name if child.is_dir() else child.stem
        # Skip children that do not carry a plate token (e.g. stray sidecar files).
        try:
            parse_plate_token(source_id)
        except ValueError:
            continue
        experiment_id = compose_collection_experiment_id(collection_name, source_id)
        if experiment_id not in seen:
            seen.add(experiment_id)
            experiment_ids.append(experiment_id)

    if not experiment_ids:
        raise ValueError(
            f"resolve_experiment_ids: collection {collection_name!r} at {collection_dir!s} "
            "contains no children matching the collection grammar "
            "{date}_{plate_token}[_{event_label}]. Nothing to expand."
        )
    return experiment_ids
