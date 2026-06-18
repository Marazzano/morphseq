"""The canonical channel vocabulary — the pipeline's channel "language".

Scope-native channel dialect (e.g. YX1 ``"EYES - Dia"``, Keyence ``"Brightfield"``) is translated to
this canonical vocabulary by the scope adapters (``scope/<scope>/mappings.py`` +
``scope/shared/canonical_mapper.py``). This module owns ONLY the canonical language + its validator;
the per-scope dialect maps live with their scope. Doctrine: *mappings translate dialect, vocabularies
define language, validators guard contracts* (``specs/acquisition_inventory_schema_policy.md``).
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────────────────────
# Canonical Vocabulary — the accepted channel_id tokens (the language every scope must map INTO)
# ─────────────────────────────────────────────────────────────────────────────────────────────

# Valid standardized channel names (the canonical channel_id vocabulary).
VALID_CHANNEL_NAMES = [
    'BF',      # Brightfield
    'GFP',     # Green fluorescent protein
    'RFP',     # Red fluorescent protein
    'BFP',     # Blue fluorescent protein
    'CFP',     # Cyan fluorescent protein
    'YFP',     # Yellow fluorescent protein
]

# Brightfield channel identifiers (for special processing).
BRIGHTFIELD_CHANNELS = ['BF']


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Validation — fail loud on a channel_id outside the canonical vocabulary
# ─────────────────────────────────────────────────────────────────────────────────────────────


def validate_channel_id(channel_id: str) -> str:
    """Assert ``channel_id`` is in the canonical vocabulary; return it unchanged.

    The vocabulary owner exposes its own validator (mirrors
    ``shared/identifiers/validators.py::validate_well_id``). ``channel_id`` is a controlled-vocabulary
    token, not a composed identifier, so it lives with the channel domain — not ``identifiers/``.
    The scope adapter already guarantees this at mint time; this is the contract-time net for
    hand-edited / stale tables.
    """
    text = str(channel_id)
    if text not in VALID_CHANNEL_NAMES:
        raise ValueError(
            f"channel_id {channel_id!r} is not in the canonical vocabulary "
            f"{sorted(VALID_CHANNEL_NAMES)}. A scope must map its raw channel name to an accepted "
            "channel_id (or the vocabulary must be deliberately extended)."
        )
    return text
