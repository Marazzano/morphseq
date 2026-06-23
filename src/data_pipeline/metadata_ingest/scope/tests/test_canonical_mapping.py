"""Tests for scope dialect -> canonical channel mapping."""

from __future__ import annotations

import pytest

from data_pipeline.metadata_ingest.scope.keyence.mappings import KEYENCE_CHANNEL_INDEX_MAP
from data_pipeline.metadata_ingest.scope.shared.canonical_mapper import apply_canonical_mapping
from data_pipeline.metadata_ingest.scope.yx1.mappings import YX1_CHANNEL_MAP
from data_pipeline.schemas.channel_normalization import VALID_CHANNEL_NAMES, validate_channel_id


def test_apply_canonical_mapping_returns_expected_value():
    assert (
        apply_canonical_mapping(
            "EYES - Dia",
            YX1_CHANNEL_MAP,
            vocabulary=VALID_CHANNEL_NAMES,
            field="channel_id",
            scope_name="YX1",
        )
        == "BF"
    )


def test_apply_canonical_mapping_rejects_unmapped_raw_value():
    with pytest.raises(ValueError, match="YX1 has no channel_id mapping"):
        apply_canonical_mapping(
            "EYES - Cy5",
            YX1_CHANNEL_MAP,
            vocabulary=VALID_CHANNEL_NAMES,
            field="channel_id",
            scope_name="YX1",
        )


def test_apply_canonical_mapping_rejects_non_canonical_target():
    with pytest.raises(ValueError, match="not in the canonical channel_id vocabulary"):
        apply_canonical_mapping(
            "EYES - Cy5",
            {"EYES - Cy5": "Cy5"},
            vocabulary=VALID_CHANNEL_NAMES,
            field="channel_id",
            scope_name="YX1",
        )


@pytest.mark.parametrize("mapping", [YX1_CHANNEL_MAP, KEYENCE_CHANNEL_INDEX_MAP])
def test_scope_channel_maps_target_valid_channel_ids(mapping):
    invalid_targets = sorted(set(mapping.values()) - set(VALID_CHANNEL_NAMES))
    assert invalid_targets == []


def test_validate_channel_id_accepts_known_channel_id():
    assert validate_channel_id("BF") == "BF"


def test_validate_channel_id_rejects_unknown_channel_id():
    with pytest.raises(ValueError, match="not in the canonical vocabulary"):
        validate_channel_id("Cy5")
