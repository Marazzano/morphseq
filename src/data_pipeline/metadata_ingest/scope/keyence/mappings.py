"""Keyence scope dialect → canonical mappings (DATA).

Pure data: raw BZ-X channel strings → canonical ``channel_id`` tokens. Applied via
``scope/shared/canonical_mapper.apply_canonical_mapping`` against
``schemas/channel_normalization.VALID_CHANNEL_NAMES``. Adding a raw channel is a tiny reviewable diff;
the map-integrity test rejects a target that is not in the canonical vocabulary. (Shaped so a
config-loaded source could replace this literal later without touching the applier.)
"""

from __future__ import annotations

KEYENCE_CHANNEL_MAP: dict[str, str] = {
    "Brightfield": "BF",
    "GFP": "GFP",
    "RFP": "RFP",
    "mCherry": "RFP",
}
