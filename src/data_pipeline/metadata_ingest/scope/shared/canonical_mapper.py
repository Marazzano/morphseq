"""Generic scope-dialect → canonical applier.

One small reusable function that translates a scope-native raw value into a canonical token using a
scope-specific mapping DICT and a canonical vocabulary. It enforces BOTH failure modes at the moment
the value is minted, so scope sludge never leaks past the adapter:

  1. the raw value is unmapped for this scope, or
  2. the mapping points at a value outside the canonical vocabulary.

Exact matching only — no lowercase / substring / regex inference. A new microscope label must be
NOTICED and explicitly mapped; that is annoying once but prevents silent false mappings forever.
Doctrine: *mappings translate dialect, vocabularies define language, validators guard contracts*
(``specs/acquisition_inventory_schema_policy.md``). The same applier serves any raw→canonical mapping
(channel today; scope→detector channel later) — different dict + vocabulary, same glue.
"""

from __future__ import annotations

from typing import Collection, Mapping


def apply_canonical_mapping(
    raw_value: str,
    mapping: Mapping[str, str],
    *,
    vocabulary: Collection[str],
    field: str,
    scope_name: str,
) -> str:
    """Translate ``raw_value`` to its canonical token via ``mapping``; fail loud on either failure mode."""
    if raw_value not in mapping:
        raise ValueError(
            f"{scope_name} has no {field} mapping for raw value {raw_value!r}. "
            f"Add it to the {scope_name} {field} mapping (exact-match only — new labels must be "
            "mapped explicitly)."
        )
    canonical_value = mapping[raw_value]
    if canonical_value not in vocabulary:
        raise ValueError(
            f"{scope_name} maps raw {field} value {raw_value!r} to {canonical_value!r}, but that is "
            f"not in the canonical {field} vocabulary {sorted(vocabulary)}. Fix the mapping target, "
            "or deliberately extend the vocabulary."
        )
    return canonical_value
