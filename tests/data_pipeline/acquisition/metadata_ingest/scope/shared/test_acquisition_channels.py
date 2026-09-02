"""Tests for channel_id → channel_index resolution.

``channel_index`` is a FACT OF THE FILE: the scope adapter mints the
``channel_index / raw_channel_name / channel_id`` triple once into the acquisition inventory, and
readers LOOK IT UP. Re-deriving an index by matching channel NAMES at read time invents a second
channel vocabulary that drifts from the scope's ``channel_map.py`` — which is exactly what happened:
a private name-matcher knew "BF"/"EYES - Dia"/"Empty" but not "BF-no bin", so it could not resolve a
real fluorescence plate at all and told callers to set an env var instead.

Run: PYTHONPATH=src pytest tests/data_pipeline/acquisition/metadata_ingest/scope/shared/test_acquisition_channels.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.acquisition.metadata_ingest.scope.shared.acquisition_checks import (
    assert_channel_mapping_consistent,
)

from data_pipeline.acquisition.metadata_ingest.scope.shared.acquisition_channels import (
    resolve_channel_index,
)

# The REAL triple minted for the pbx fluorescence collection — the file the old matcher choked on.
PBX_TRIPLE = pd.DataFrame(
    [
        {"channel_index": 0, "raw_channel_name": "BF-no bin", "channel_id": "BF"},
        {"channel_index": 1, "raw_channel_name": "tdtomato", "channel_id": "RFP"},
    ]
)


def test_resolves_the_real_pbx_channels():
    assert resolve_channel_index(PBX_TRIPLE, "BF") == 0
    assert resolve_channel_index(PBX_TRIPLE, "RFP") == 1


def test_index_is_read_not_guessed_from_order():
    """The recorded index wins even when it does not match row order or alphabetical order."""
    shuffled = pd.DataFrame(
        [
            {"channel_index": 3, "raw_channel_name": "tdtomato", "channel_id": "RFP"},
            {"channel_index": 1, "raw_channel_name": "BF-no bin", "channel_id": "BF"},
        ]
    )
    assert resolve_channel_index(shuffled, "BF") == 1
    assert resolve_channel_index(shuffled, "RFP") == 3


def test_many_rows_per_channel_collapse_to_one_index():
    # Real inventories carry one row per (position, z, channel) — the triple repeats.
    repeated = pd.concat([PBX_TRIPLE] * 50, ignore_index=True)
    assert resolve_channel_index(repeated, "BF") == 0


def test_absent_channel_fails_loud_and_lists_what_is_available():
    with pytest.raises(ValueError, match="is absent from this acquisition"):
        resolve_channel_index(PBX_TRIPLE, "GFP")
    # The message must name the alternatives, not suggest an env-var workaround.
    with pytest.raises(ValueError, match=r"Available channels: \['BF', 'RFP'\]"):
        resolve_channel_index(PBX_TRIPLE, "GFP")


def test_ambiguous_mapping_fails_loud():
    broken = pd.DataFrame(
        [
            {"channel_index": 0, "raw_channel_name": "BF-no bin", "channel_id": "BF"},
            {"channel_index": 1, "raw_channel_name": "EYES - Dia", "channel_id": "BF"},
        ]
    )
    with pytest.raises(ValueError, match="maps to multiple indices"):
        resolve_channel_index(broken, "BF")


def test_missing_columns_fail_loud():
    with pytest.raises(ValueError, match="missing channel columns"):
        resolve_channel_index(PBX_TRIPLE.drop(columns=["channel_index"]), "BF")


def test_unknown_token_is_reported_as_absent():
    with pytest.raises(ValueError, match="is absent from this acquisition"):
        resolve_channel_index(PBX_TRIPLE, "tdTomato")


def test_two_channel_ids_sharing_ONE_index_is_caught_by_the_producer():
    """The 1:1:1 invariant this resolver alone cannot see.

    Resolving each channel_id independently looks fine here — BF->0 and RFP->0 both "work" — yet both
    point at the same array channel, so one product would silently receive the other's pixels. Only a
    whole-table check catches it, which is why the PRODUCER must run
    assert_channel_mapping_consistent before writing the inventory.
    """
    collided = pd.DataFrame(
        [
            {"channel_index": 0, "raw_channel_name": "BF-no bin", "channel_id": "BF"},
            {"channel_index": 0, "raw_channel_name": "tdtomato", "channel_id": "RFP"},
        ]
    )
    # Per-id resolution cannot detect it...
    assert resolve_channel_index(collided, "BF") == 0
    assert resolve_channel_index(collided, "RFP") == 0
    # ...but the producer's whole-table assertion must reject it.
    with pytest.raises(ValueError, match="inconsistent channel mapping"):
        assert_channel_mapping_consistent(
            collided, normalized_column="channel_id", scope_label="test"
        )
