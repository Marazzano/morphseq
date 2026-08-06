from __future__ import annotations

import pandas as pd

from data_pipeline.acquisition.seahub.reconciliation import (
    apply_inclusion_policy,
    reconcile_seahub_metadata,
)
from data_pipeline.acquisition.seahub.stages import stage_label_to_hpf


def test_owner_stage_crosswalk_includes_12_somite_interpolation():
    hpf, method, label = stage_label_to_hpf("12s")
    assert hpf == 15.0
    assert method == "stage_crosswalk"
    assert label == "12-somite"


def test_reconciliation_uses_collection_prefix_experiment_correction():
    images = pd.DataFrame(
        [
            {
                "image_id": "fov-1",
                "image_role": "eight_embryo_fov",
                "experiment_id": "GENE2",
                "stage_source_label": "12s",
                "stage_addition_source_label": pd.NA,
                "perturbation_parsed": "ctrl-inj",
                "perturbation_key": "ctrl|inj",
                "metadata_match_status": "unmatched_stage",
            }
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "metadata_row_number": 2,
                "expt": "GENE99",
                "collection_name": "GENE2_ctrl-inj_12s",
                "perturbation": "inj_ctrl",
                "stage_collected": "12-somite",
                "stage_addition": pd.NA,
            }
        ]
    )

    row = reconcile_seahub_metadata(images, metadata).iloc[0]
    assert row["metadata_match_status"] == "exact"
    assert row["stage_hpf"] == 15.0
    assert row["stage_match_method"] == "stage_crosswalk"
    assert row["metadata_experiment_original"] == "GENE99"
    assert row["metadata_experiment_effective"] == "GENE2"
    assert bool(row["metadata_experiment_corrected"])
    assert (
        row["metadata_experiment_correction_source"]
        == "collection_name_prefix"
    )


def test_existing_success_is_preserved_but_receives_audit_fields():
    images = pd.DataFrame(
        [
            {
                "image_id": "fov-1",
                "image_role": "eight_embryo_fov",
                "experiment_id": "GENE2",
                "stage_source_label": "24hpf",
                "stage_addition_source_label": pd.NA,
                "metadata_match_status": "exact",
                "metadata_row_number": 77,
                "metadata_expt": "GENE2",
                "metadata_collection_name": "GENE2_ctrl_24hpf",
                "metadata_stage_collected": "24hpf",
            }
        ]
    )
    unrelated_metadata = pd.DataFrame(
        [
            {
                "metadata_row_number": 2,
                "expt": "GENE3",
                "collection_name": "GENE3_ctrl_24hpf",
                "perturbation": "ctrl",
                "stage_collected": "24hpf",
            }
        ]
    )
    row = reconcile_seahub_metadata(images, unrelated_metadata).iloc[0]
    assert row["metadata_match_status"] == "exact"
    assert row["metadata_row_number"] == 77
    assert row["metadata_experiment_effective"] == "GENE2"
    assert row["metadata_stage_collected_normalized"] == "24hpf"


def test_inclusion_passes_stage_and_condition_failures_only():
    base = {
        "image_role": "eight_embryo_fov",
        "experiment_id": "GENE2",
        "image_path": "/data/GENE2/images/fov.jpg",
        "relative_path": "GENE2/images/fov.jpg",
        "excluded_path": float("nan"),
        "read_error": pd.NA,
    }
    rows = [
        {**base, "image_id": "stage", "metadata_match_status": "unmatched_stage"},
        {
            **base,
            "image_id": "condition",
            "metadata_match_status": "unmatched_condition",
        },
        {
            **base,
            "image_id": "duplicate",
            "metadata_match_status": "exact_duplicate",
        },
        {
            **base,
            "image_id": "abandoned",
            "relative_path": "GENE2/abandoned/fov.jpg",
            "metadata_match_status": "exact",
        },
    ]
    policy = apply_inclusion_policy(pd.DataFrame(rows)).set_index("image_id")
    assert bool(policy.loc["stage", "include_for_seahub"])
    assert bool(policy.loc["condition", "include_for_seahub"])
    assert not bool(policy.loc["duplicate", "include_for_seahub"])
    assert (
        policy.loc["duplicate", "exclusion_reason"]
        == "duplicate_or_ambiguous_metadata"
    )
    assert not bool(policy.loc["abandoned", "include_for_seahub"])
    assert policy.loc["abandoned", "exclusion_reason"] == "abandoned"


def test_inclusion_records_contract_invalid_rows_without_stopping_corpus():
    base = {
        "image_role": "eight_embryo_fov",
        "experiment_id": "GENE2",
        "image_path": "/data/fov.jpg",
        "relative_path": "GENE2/images/fov.jpg",
        "excluded_path": False,
        "read_error": pd.NA,
        "metadata_match_status": "exact",
    }
    rows = [
        {**base, "image_id": "valid"},
        {**base, "image_id": pd.NA},
        {**base, "image_id": "duplicate"},
        {**base, "image_id": "duplicate"},
        {**base, "image_id": "bad-status", "metadata_match_status": pd.NA},
    ]

    policy = apply_inclusion_policy(pd.DataFrame(rows))
    assert bool(policy.iloc[0]["include_for_seahub"])
    assert (
        policy.iloc[1]["exclusion_reason"]
        == "contract_invalid_missing_source_fov_id"
    )
    assert set(policy.iloc[2:4]["exclusion_reason"]) == {
        "contract_invalid_duplicate_source_fov_id"
    }
    assert (
        policy.iloc[4]["exclusion_reason"]
        == "contract_invalid_metadata_match_status"
    )
