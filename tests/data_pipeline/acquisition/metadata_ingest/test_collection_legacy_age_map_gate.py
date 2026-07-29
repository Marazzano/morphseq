"""Removal gate for the legacy ``start_age_by_time_index`` field.

The collection-classify artifact carries the age map twice while consumers migrate:

    start_age_by_source_ordinal   CANONICAL — keys are source ordinals (what ages vary over)
    start_age_by_time_index       LEGACY alias, same values, misleading name

The legacy name is actively wrong once any source is a timelapse: its keys are SOURCE ORDINALS,
so looking up a merged frame ``time_index`` in it silently reads a neighbouring source's age (or
raises). It must go.

This test is the GATE: it fails once no production consumer reads the legacy field, which is the
signal that ``start_age_by_time_index`` can be deleted from the producer + contract. When this
test fails with "legacy field is no longer read", do the removal and delete this file.

See TODO(collection-legacy-age-map) in collection_provenance.py.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/test_collection_legacy_age_map_gate.py
"""

from __future__ import annotations

import re
from pathlib import Path

_LEGACY_FIELD = "start_age_by_time_index"

# The producer + its contract necessarily still NAME the legacy field (they emit/validate it), so
# they are not evidence of a consumer. Everything else under src/ that reads it is.
_PRODUCER_FILES = {
    "collection_provenance.py",
    "collection_provenance_contract.py",
}

# A READ looks like `payload["start_age_by_time_index"]` or `.get("start_age_by_time_index")`.
# A mere mention in a docstring/comment is not a read.
_READ_RE = re.compile(
    r"""(\[\s*["']start_age_by_time_index["']\s*\]|\.get\(\s*["']start_age_by_time_index["'])"""
)


def _src_root() -> Path:
    # tests/data_pipeline/acquisition/metadata_ingest/ -> repo root -> src/
    return Path(__file__).resolve().parents[4] / "src"


def _legacy_readers() -> list[str]:
    readers = []
    for py in _src_root().rglob("*.py"):
        if py.name in _PRODUCER_FILES:
            continue
        if _READ_RE.search(py.read_text()):
            readers.append(str(py.relative_to(_src_root())))
    return sorted(readers)


def test_legacy_age_map_consumers_are_tracked():
    """Pins WHICH consumers still read the legacy field, so the set can only shrink.

    When this list becomes empty the gate below fires and the field can be deleted.
    """
    expected = {
        # Reads the legacy map only as a FALLBACK when the canonical field is absent
        # (compute._start_age_hpf_for_snip). Removing the fallback is the last migration step.
        "data_pipeline/feature_extraction/stage_predictions/compute.py",
    }
    assert set(_legacy_readers()) == expected, (
        "The set of legacy start_age_by_time_index readers changed. If you MIGRATED a consumer, "
        "remove it from `expected` here. If you ADDED one, don't — read "
        "start_age_by_source_ordinal instead (the legacy name's keys are source ordinals, so it "
        "is wrong for any timelapse source)."
    )


def test_gate_fires_once_legacy_field_is_unread():
    """The actual removal gate.

    While any consumer still reads the legacy field this test passes trivially. The moment none
    does, it FAILS with instructions — delete `start_age_by_time_index` from the producer and the
    contract, drop the fallback branch, and delete this file.
    """
    readers = _legacy_readers()
    assert readers, (
        f"The legacy field {_LEGACY_FIELD!r} is no longer read by any consumer under src/. "
        "REMOVE IT NOW: delete it from collection_provenance.py (both payload branches), from "
        "REQUIRED_COLLECTION_PROVENANCE_KEYS + the mirror check in "
        "collection_provenance_contract.py, and delete this test file. "
        "See TODO(collection-legacy-age-map)."
    )
