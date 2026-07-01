"""YX1's channel map: raw ND2 channel names -> canonical channel_id. Self-checked on import.

YX1 embeds the channel NAME directly and reliably in ND2 metadata, so the map is keyed on that
raw name string. (Contrast Keyence, which anchors on a channel INDEX instead — see
``scope/keyence/channel_map.py``.)
"""

from __future__ import annotations

from data_pipeline.metadata_ingest.scope.shared.channel_map_contract import ScopeChannelMap

YX1_CHANNEL_MAP = ScopeChannelMap(
    scope_name="YX1",
    raw_key_to_channel_id={
        "Empty":       "BF",
        "EYES - Dia":  "BF",
        "EYES - GFP":  "GFP",
        "EYES - RFP":  "RFP",
    },
)
