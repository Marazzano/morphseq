"""Collection classification — the EARLY-DAG "what is this experiment?" fact (CLASSIFY ONCE).

The governing principle (docs/EXPERIMENT_GROUP_PLATE_MODEL.md, "CLASSIFY ONCE, CONSUME EVERYWHERE"):
decide at the START of the DAG whether an experiment is a collection, and emit that DECLARED fact as
one small per-experiment JSON artifact. Every downstream step that must branch reads THIS artifact —
it never re-infers collection-ness from ``experiment_id``, ``n_sources``, nulls, or the filesystem.

This module is the CREATE site (the one place allowed to derive), producing::

    { "experiment_id": "chem28c_coll_plate01",
      "is_collection": true,
      "sources": ["20250622_plate01_t28hpf", "20250623_plate01_t52hpf"],
      "start_age_by_time_index": {"0": 28, "1": 52} }

For a single (non-collection) experiment the payload is inert (``is_collection: false``, empty
sources + age map) so consumers behave exactly as today.

The age map RIDES IN this artifact (no separate age product — see the spec's "Age" section): the
classify step already parses the sources' ``t<NN>hpf`` tokens to decide ``is_collection``, so the
``time_index → start_age_hpf`` map falls out for free.

CRITICAL correctness point — ``time_index`` must match the acquisition union. The union
(``collection_acquisition_union``) orders sources by ``SourceChild.sort_key`` (declared_hpf, then
date) and assigns each a ``time_index`` BLOCK ordinal. This module REUSES that exact ``SourceChild``
+ ``sort_key`` so the ``time_index`` keys here are identical to the ``time_index`` values the union
stamps — the age lookup downstream can never drift from acquisition. (MVP snapshot: each source is
one frame → one ``time_index`` per source → block start == ordinal == the union's ``time_index``.)

Import direction: MAY import shared identifiers + the collection helpers whose OUTPUT it consumes;
it MUST NOT import stages, Snakemake/tasks, or orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import SourceChild
from data_pipeline.acquisition.metadata_ingest.collection_classification_contract import (
    validate_collection_classification,
)
from data_pipeline.acquisition.metadata_ingest.experiment_collection import (
    find_collection_plate_sources,
)
from data_pipeline.shared.identifiers import is_collection_plate_id


# ─────────────────────────────────────────────────────────────────────────────────────
# Classify — derive the declared fact (the ONE place allowed to derive)
# ─────────────────────────────────────────────────────────────────────────────────────

def classify_experiment(
    experiment_id: str,
    raw_root: str | Path,
    microscope: str = "Keyence",
) -> dict:
    """Classify one experiment into the collection-classify payload (the declared fact).

    Args:
        experiment_id: the experiment to classify — a merged ``{collection}_coll_{plate_token}`` id
            or a legacy single-experiment id.
        raw_root: raw image root containing the ``_coll`` dir (only read for a collection, to find
            its source children).
        microscope: microscope label — provenance carried onto each ``SourceChild`` (the ordering
            uses only the child NAME, so this does not change ``time_index`` assignment).

    Returns:
        The validated payload dict (see ``collection_classification_contract``): for a collection,
        ``is_collection=True`` with its ``sources`` and ``start_age_by_time_index`` (time_index →
        declared hpf, None where undeclared); for a single experiment, the inert
        ``is_collection=False`` payload.
    """
    experiment_id = str(experiment_id).strip()

    # is_collection is DECIDED here (the create site) via the identity grammar — never re-derived.
    if not is_collection_plate_id(experiment_id):
        payload = {
            "experiment_id": experiment_id,
            "is_collection": False,
            "sources": [],
            "start_age_by_time_index": {},
        }
        validate_collection_classification(payload)
        return payload

    collection_name, child_names = find_collection_plate_sources(experiment_id, Path(raw_root))
    collection_dir = Path(raw_root) / collection_name

    # REUSE the acquisition union's ordering so time_index here == time_index in the inventory.
    ordered_sources = sorted(
        (SourceChild(child_name=name, scope=microscope) for name in child_names),
        key=SourceChild.sort_key,
    )

    # One PROVENANCE RECORD per source: file + on-disk raw_path + declared_hpf + time_index. This is
    # the single source of truth every disk-touching step reads (it never re-globs the _coll dir).
    # time_index is the block ordinal of the union's ordering — identical to the acquisition
    # inventory's time_index by construction.
    sources = [
        {
            "file": source.child_name,
            "raw_path": str(collection_dir / source.child_name),
            "declared_hpf": source.declared_hpf,
            "time_index": time_index,
        }
        for time_index, source in enumerate(ordered_sources)
    ]
    start_age_by_time_index = {
        str(rec["time_index"]): rec["declared_hpf"] for rec in sources
    }

    payload = {
        "experiment_id": experiment_id,
        "is_collection": True,
        "sources": sources,
        "start_age_by_time_index": start_age_by_time_index,
    }
    validate_collection_classification(payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────────────
# Write — persist the declared fact as the artifact downstream consumes
# ─────────────────────────────────────────────────────────────────────────────────────

def write_collection_classification(
    *,
    experiment_id: str,
    raw_root: str | Path,
    microscope: str,
    output_json: Path,
) -> dict:
    """Classify ``experiment_id`` and write its collection-classify artifact to ``output_json``.

    Returns the (validated) payload it wrote. The path comes from ``paths.py`` at the call site;
    this writer only owns the classify + serialization.
    """
    payload = classify_experiment(experiment_id, raw_root, microscope)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_collection_classification(input_json: str | Path) -> dict:
    """Read + validate a collection-classify artifact (the consume boundary).

    Downstream steps call this to CONSUME the declared fact; it re-validates the shape so a
    hand-edited or corrupt artifact fails loud where it is read, not deep in a consumer.
    """
    payload = json.loads(Path(input_json).read_text())
    validate_collection_classification(payload)
    return payload
