"""The canonical channel vocabulary — the pipeline's channel "language".

Scope-native channel dialect (e.g. YX1 ``"EYES - Dia"``, Keyence ``CH1``) is translated to this
canonical vocabulary by the scope adapters (``scope/<scope>/channel_map.py`` +
``scope/shared/canonical_mapper.py``). This module owns ONLY the canonical language + its validator;
the per-scope dialect maps live with their scope. Doctrine: *mappings translate dialect, vocabularies
define language, validators guard contracts* (``specs/acquisition_inventory_schema_policy.md``).

This module is cross-cutting (imported by every scope adapter AND by ``shared/identifiers/parsers.py``)
so it lives in ``shared/``, not inside any one stage.
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
# Display colors — how a channel is RENDERED, part of the same vocabulary
# ─────────────────────────────────────────────────────────────────────────────────────────────

# DISPLAY ONLY. Materialized image products stay single-channel INTENSITY data; nothing here
# changes stored pixels. A fluorescence product like RFP__projection__max is a quantitative
# measurement (native resolution, uint16, un-inverted) — baking a colormap into it would triple its
# size and destroy that. Color is applied when an image is shown, never when it is written.
#
# This lives beside the vocabulary rather than in an analysis/viz package because "RFP renders red"
# is a fact about what the token MEANS, keyed on the canonical channel_id — the same reason
# VALID_CHANNEL_NAMES lives here. Splitting the vocabulary across two trees is how the two halves
# drift apart. Keyed by FLUOROPHORE color, so the mapping states physics, not a naming convention.
#
# Known collisions, recorded rather than fixed: YFP's amber is close to the analysis-side genotype
# color for 'heterozygous' (#F7B267), and CFP/BFP/GFP sit near each other in colorblind space — if
# three fluorescent channels are ever plotted together, that trio needs a re-check.
CHANNEL_ID_COLORS: dict[str, str] = {
    'BF':  '#4D4D4D',   # Neutral gray — brightfield is not a fluorophore
    'GFP': '#2CA02C',   # Green
    'RFP': '#E63946',   # Imperial red
    'BFP': '#3B76D9',   # Blue
    'CFP': '#17BECF',   # Cyan
    'YFP': '#E8B92E',   # Amber — darkened from pure yellow, which is illegible on white
}


def color_for_channel_id(channel_id: str) -> str:
    """Return the display hex color for ``channel_id``; fail loud if it has none.

    Mirrors ``validate_channel_id``: validate first, then look up. Callers must not index
    ``CHANNEL_ID_COLORS`` directly — a raw lookup turns a vocabulary gap into a bare ``KeyError``
    at render time, with no statement of what went wrong or how to fix it.
    """
    text = validate_channel_id(channel_id)
    if text not in CHANNEL_ID_COLORS:
        raise ValueError(
            f"channel_id {channel_id!r} is in the canonical vocabulary but has no display color. "
            f"Add it to CHANNEL_ID_COLORS (colored channels: {sorted(CHANNEL_ID_COLORS)})."
        )
    return CHANNEL_ID_COLORS[text]


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
