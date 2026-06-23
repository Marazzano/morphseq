"""Keyence scope dialect → canonical mappings (DATA).

Pure data: the canonical ``channel_id`` mapping for Keyence. Applied via
``scope/shared/canonical_mapper.apply_canonical_mapping`` against
``schemas/channel_normalization.VALID_CHANNEL_NAMES`` (the map-integrity test rejects a target that is
not in the canonical vocabulary).

Keyence BZ-X embeds channel info in PROPRIETARY XML that is only reliably readable inside Keyence's
own software — the scraped channel NAME is often absent or untrustworthy. The one signal that is
ALWAYS on disk is the filename ``CH#`` index. So Keyence anchors ``channel_id`` on the integer channel
index via ``KEYENCE_CHANNEL_INDEX_MAP`` — one map, used by BOTH the acquisition inventory and the
(legacy) FF scope-metadata extractor. There is deliberately no name-keyed map: a name map would be a
second, unreliable mechanism for the same decision.

Keep the map MINIMAL — only the channels real experiments actually contain. Today's Keyence data is
brightfield-only, so ``CH1 → BF``. An unmapped index FAILS LOUD at the mapping boundary (via
``apply_canonical_mapping``) — the fix is to add the real channel here, never to guess (defaulting an
unknown ``CH#`` to BF would silently mislabel a true GFP/RFP plane). Deliberate by design. (Shaped so
a config-loaded source could replace this literal later without touching the applier.)
"""

from __future__ import annotations

KEYENCE_CHANNEL_INDEX_MAP: dict[int, str] = {
    1: "BF",
}
