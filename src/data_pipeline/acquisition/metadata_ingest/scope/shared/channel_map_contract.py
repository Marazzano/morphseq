"""The self-validating contract for a scope's channel map.

A "channel map" answers one question for one microscope: given the raw channel
identifier that scope writes to disk (its "dialect" — e.g. YX1's raw channel NAME,
or Keyence's raw channel INDEX), what is our canonical channel_id?

This matters because channel_id is STICKY: it gets embedded inside image_id, which
identifies every frame for the rest of the pipeline. A wrong channel_id here becomes
a wrong image_id everywhere downstream. So the map checks itself the moment it is
built — a bad map fails loudly at construction, not silently three stages later.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from data_pipeline.shared.channel_vocabulary import VALID_CHANNEL_NAMES

# The raw dialect key can be a string (YX1's raw channel name, e.g. "EYES - Dia")
# or an int (Keyence's raw CH# index, e.g. 1). ScopeChannelMap works with either.
RawDialectKey = TypeVar("RawDialectKey")


class ScopeChannelMap(Generic[RawDialectKey]):
    """One microscope's raw-dialect-key -> canonical-channel_id lookup, self-checked.

    Build one like this (one entry per line, reads like a dictionary):

        YX1_CHANNEL_MAP = ScopeChannelMap(
            scope_name="YX1",
            raw_key_to_channel_id={
                "Empty":       "BF",
                "EYES - Dia":  "BF",
                "EYES - GFP":  "GFP",
                "EYES - RFP":  "RFP",
            },
        )

    Then translate a raw dialect key with:

        canonical_channel_id = YX1_CHANNEL_MAP.to_canonical("EYES - Dia")   # -> "BF"
    """

    def __init__(self, scope_name: str, raw_key_to_channel_id: dict[RawDialectKey, str]) -> None:
        # Keep the scope name so error messages can say WHICH microscope is misconfigured.
        self.scope_name = scope_name

        # Store our own copy so nobody can change the map after we have checked it.
        self.raw_key_to_channel_id = dict(raw_key_to_channel_id)

        # Check the whole map right now, at construction time. If anything is wrong,
        # we raise here and the program stops — better than handing back a broken map.
        self._check_every_target_is_a_canonical_channel_id()

    def _check_every_target_is_a_canonical_channel_id(self) -> None:
        """Every value the map points AT must be a real canonical channel_id.

        Example of a mistake this catches: mapping "EYES - Cy5" -> "Cy5" when "Cy5"
        is not (yet) a canonical channel. That would mint a channel_id nothing else
        in the pipeline understands.
        """
        canonical_channel_ids = set(VALID_CHANNEL_NAMES)

        for raw_key, channel_id in self.raw_key_to_channel_id.items():
            if channel_id not in canonical_channel_ids:
                raise ValueError(
                    f"[{self.scope_name} channel map] The raw channel {raw_key!r} is "
                    f"mapped to {channel_id!r}, but {channel_id!r} is not a canonical "
                    f"channel_id.\n"
                    f"  Canonical channel_ids are: {sorted(canonical_channel_ids)}\n"
                    f"  To fix: either map {raw_key!r} to one of those, or — if "
                    f"{channel_id!r} is a genuinely new channel — add it to "
                    f"VALID_CHANNEL_NAMES in shared/channel_vocabulary.py first.\n"
                    f"  Do NOT map to a non-canonical token: channel_id is embedded in "
                    f"image_id, so a wrong value here mislabels every downstream frame."
                )

    def to_canonical(self, raw_key: RawDialectKey) -> str:
        """Translate one raw scope dialect key into its canonical channel_id.

        Fails loudly if this scope has never declared how to handle that raw key —
        guessing would risk silently mislabeling a real channel.
        """
        if raw_key not in self.raw_key_to_channel_id:
            known_raw_keys = sorted(self.raw_key_to_channel_id.keys(), key=str)
            raise ValueError(
                f"[{self.scope_name} channel map] No mapping for raw channel "
                f"{raw_key!r}.\n"
                f"  Raw channels this scope knows how to translate: {known_raw_keys}\n"
                f"  To fix: add {raw_key!r} -> <canonical channel_id> to this "
                f"scope's channel_map.py. Never guess a default — an unmapped channel "
                f"is a real gap, not a brightfield frame."
            )
        return self.raw_key_to_channel_id[raw_key]
