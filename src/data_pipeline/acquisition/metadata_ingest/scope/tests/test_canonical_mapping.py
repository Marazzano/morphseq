"""Tests for scope dialect -> canonical channel mapping.

ScopeChannelMap owns its own validation (it self-checks at construction and fails loud
in ``to_canonical``). These tests verify that guard EXISTS and behaves correctly — they
do not recompute the check themselves. Mirrors how snip_qc tests check the verdict, not
reimplement it.
"""

from __future__ import annotations

import pytest

from data_pipeline.acquisition.metadata_ingest.scope.keyence.channel_map import KEYENCE_CHANNEL_INDEX_MAP
from data_pipeline.acquisition.metadata_ingest.scope.shared.channel_map_contract import ScopeChannelMap
from data_pipeline.acquisition.metadata_ingest.scope.yx1.channel_map import YX1_CHANNEL_MAP
from data_pipeline.shared.channel_vocabulary import validate_channel_id


def test_to_canonical_returns_expected_value():
    assert YX1_CHANNEL_MAP.to_canonical("EYES - Dia") == "BF"


def test_yx1_celesta_473_maps_to_gfp():
    assert YX1_CHANNEL_MAP.to_canonical("Celesta 473") == "GFP"


def test_to_canonical_rejects_unmapped_raw_value():
    with pytest.raises(ValueError, match="No mapping for raw channel"):
        YX1_CHANNEL_MAP.to_canonical("EYES - Cy5")


def test_construction_rejects_non_canonical_target():
    # The map rejects a bad target at construction — not the test.
    with pytest.raises(ValueError, match="not a canonical channel_id"):
        ScopeChannelMap("YX1", {"EYES - Cy5": "Cy5"})


@pytest.mark.parametrize("channel_map", [YX1_CHANNEL_MAP, KEYENCE_CHANNEL_INDEX_MAP])
def test_scope_channel_maps_are_already_valid(channel_map):
    # If construction succeeded, every target is canonical by definition. This just
    # confirms both real scope maps are live ScopeChannelMap instances (the guard ran).
    assert isinstance(channel_map, ScopeChannelMap)


def test_keyence_channel_map_is_keyed_on_integer_index():
    assert KEYENCE_CHANNEL_INDEX_MAP.to_canonical(1) == "BF"


def test_validate_channel_id_accepts_known_channel_id():
    assert validate_channel_id("BF") == "BF"


def test_validate_channel_id_rejects_unknown_channel_id():
    with pytest.raises(ValueError, match="not in the canonical vocabulary"):
        validate_channel_id("Cy5")
