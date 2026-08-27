"""Channel queries over ACQUISITION FACTS — read the recorded index, never re-derive it.

Layering (why this is not in ``shared/channel_vocabulary.py``):

    shared/channel_vocabulary.py   the LANGUAGE  — which channel_id tokens exist
    scope/<scope>/channel_map.py   the DIALECT   — raw_channel_name -> channel_id
    THIS MODULE                    the FACTS     — channel_id -> channel_index in ONE acquisition

The meaning of ``"BF"`` is global vocabulary. That ``"BF"`` is array index 0 in one ND2 and index 1
in another is file-specific acquisition metadata, so a DataFrame query over inventory columns belongs
here, next to ``acquisition_checks.assert_channel_mapping_consistent`` (which enforces the 1:1:1
triple at mint time), not inside the module that defines the language.

**The rule about channel indices.** Supported pipeline consumers MUST use the recorded inventory
mapping (``channel_index`` / ``raw_channel_name`` / ``channel_id``, minted once by the scope adapter).
A legacy reader that has no inventory may derive an index only by translating raw names through the
scope's ONE canonical ``channel_map.py`` — never through private aliases or an environment override.
The failure this prevents: a private name-matcher knew ``"BF"``/``"EYES - Dia"``/``"Empty"`` but not
``"BF-no bin"``, so it could not resolve a real fluorescence plate at all.
"""

from __future__ import annotations

import pandas as pd

CHANNEL_ID_COLUMN = "channel_id"
CHANNEL_INDEX_COLUMN = "channel_index"


def resolve_channel_index(inventory: pd.DataFrame, channel_id: str) -> int:
    """Return the ``channel_index`` this acquisition recorded for ``channel_id``.

    Args:
        inventory: acquisition-inventory rows carrying the minted channel triple (typically already
            narrowed to one well/source).
        channel_id: the canonical token to resolve (e.g. ``"BF"``).

    Raises:
        ValueError: if the channel columns are absent, ``channel_id`` is not in this acquisition (the
            message lists what is), or it maps to more than one index.
    """
    required = {CHANNEL_ID_COLUMN, CHANNEL_INDEX_COLUMN}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(
            f"Acquisition inventory is missing channel columns: {sorted(missing)}. The scope adapter "
            "records channel_index/raw_channel_name/channel_id; read it rather than matching names."
        )

    matches = (
        inventory.loc[
            inventory[CHANNEL_ID_COLUMN].astype(str) == str(channel_id),
            CHANNEL_INDEX_COLUMN,
        ]
        .drop_duplicates()
    )

    if matches.empty:
        available = sorted(inventory[CHANNEL_ID_COLUMN].dropna().astype(str).unique())
        raise ValueError(
            f"Channel {channel_id!r} is absent from this acquisition. Available channels: "
            f"{available}. Either the requested product names a channel this file does not contain, "
            "or the scope's channel_map.py is missing a raw-name mapping (add it there)."
        )

    if len(matches) != 1:
        raise ValueError(
            f"Channel {channel_id!r} maps to multiple indices: {sorted(matches.tolist())}. The "
            "channel triple must be 1:1:1 — fix the producer (assert_channel_mapping_consistent)."
        )

    return int(matches.iloc[0])
