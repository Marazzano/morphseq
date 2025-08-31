from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.build.build_utils import reconstruct_perturbation_key_from_df02


def test_reconstruct_perturbation_key_from_df02_majority(tmp_path: Path):
    root = tmp_path / "proj"
    df_dir = root / "metadata" / "combined_metadata_files"
    df_dir.mkdir(parents=True)

    # Build04-style df02 with duplicated rows and slight disagreements
    df02 = pd.DataFrame(
        {
            "master_perturbation": ["atf6", "atf6", "inj-ctrl", "inj-ctrl", "EM"],
            "short_pert_name": ["atf6", "atf6", "inj-ctrl", "inj-ctrl", "EM"],
            "phenotype": ["unknown", "unknown", "wt", "wt", "wt"],
            "control_flag": [False, False, True, True, True],
            "pert_type": ["CRISPR", "CRISPR", "control", "control", "medium"],
            "background": ["wik", "wik", "wik", "wik", "wik"],
        }
    )
    df02_path = df_dir / "embryo_metadata_df02.csv"
    df02.to_csv(df02_path, index=False)

    key_df = reconstruct_perturbation_key_from_df02(root=str(root), df02_path=str(df02_path))

    # File written
    out_path = root / "metadata" / "perturbation_name_key.csv"
    assert out_path.exists()

    # Mapping content
    assert set(key_df.columns) == {
        "master_perturbation",
        "short_pert_name",
        "phenotype",
        "control_flag",
        "pert_type",
        "background",
    }
    # 3 unique perts
    assert set(key_df["master_perturbation"]) == {"atf6", "inj-ctrl", "EM"}
    # Majority vote preserved
    atf6 = key_df.set_index("master_perturbation").loc["atf6"].to_dict()
    assert atf6["phenotype"] == "unknown"
    assert atf6["control_flag"] is False

