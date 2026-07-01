"""Keyence's channel map: raw CH# channel index -> canonical channel_id. Self-checked on import.

Keyence BZ-X embeds channel info in PROPRIETARY XML that is only reliably readable inside Keyence's
own software — the scraped channel NAME is often absent or untrustworthy. The one signal that is
ALWAYS on disk is the filename ``CH#`` index. So this map is keyed on that integer index, used by
BOTH the acquisition inventory and the (legacy) FF scope-metadata extractor. There is deliberately
no name-keyed map: a name map would be a second, unreliable mechanism for the same decision.

Keep the map MINIMAL — only the channels real experiments actually contain. Today's Keyence data is
brightfield-only, so ``CH1 -> BF``. An unmapped index fails loud at ``to_canonical()`` — the fix is
to add the real channel here, never to guess (defaulting an unknown ``CH#`` to BF would silently
mislabel a true GFP/RFP plane). Deliberate by design.
"""

from __future__ import annotations

from data_pipeline.metadata_ingest.scope.shared.channel_map_contract import ScopeChannelMap

KEYENCE_CHANNEL_INDEX_MAP = ScopeChannelMap(
    scope_name="Keyence",
    raw_key_to_channel_id={
        1: "BF",
    },
)
