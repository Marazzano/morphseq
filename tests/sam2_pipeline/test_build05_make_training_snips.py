from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import skimage.io as io

from src.build.build05_make_training_snips import make_image_snips


def _write_jpg(p: Path, h: int = 12, w: int = 16):
    p.parent.mkdir(parents=True, exist_ok=True)
    img = (np.random.rand(h, w) * 255).astype(np.uint8)
    io.imsave(str(p), img, check_contrast=False)


def test_make_image_snips_minimal(tmp_path: Path):
    root = tmp_path / "proj"
    # Input snips
    date = "20250612_30hpf_ctrl_atf6"
    snip_ids = ["20250612_30hpf_ctrl_atf6_C12_e00_t0000", "20250612_30hpf_ctrl_atf6_E06_e01_t0001"]
    snip_files = [f"{sid}.jpg" for sid in snip_ids]

    # Create input snip images where Build05 expects them
    for f in snip_files:
        _write_jpg(root / "training_data" / "bf_embryo_snips" / date / f)

    # Minimal df02
    df02 = pd.DataFrame(
        {
            "snip_id": snip_ids,
            "embryo_id": ["e0", "e1"],
            "experiment_date": [date, date],
            "use_embryo_flag": [True, True],
            # Optional label columns (not used when label_var=None)
            "short_pert_name": ["inj-ctrl", "atf6"],
            "phenotype": ["wt", "unknown"],
            "background": ["wik", "wik"],
        }
    )
    df_dir = root / "metadata" / "combined_metadata_files" / "curation"
    df_dir.mkdir(parents=True, exist_ok=True)
    df02_path = df_dir.parent / "embryo_metadata_df02.csv"
    df02.to_csv(df02_path, index=False)

    # Minimal curation files expected by Build05
    curation_df = pd.DataFrame(
        {
            "snip_id": snip_ids,
            "manual_stage_hpf": [np.nan, np.nan],
            "use_embryo_manual": [np.nan, np.nan],
            "manual_update_flag": [0, 0],
        }
    )
    curation_df.to_csv(df_dir / "curation_df.csv", index=False)

    emb_curation_df = pd.DataFrame(
        {
            "embryo_id": ["e0", "e1"],
            "short_pert_name": ["inj-ctrl", "atf6"],
            "phenotype": ["wt", "unknown"],
            "phenotype_orig": ["wt", "unknown"],
            "background": ["wik", "wik"],
            "manual_update_flag": [0, 0],
            "use_embryo_flag_manual": [np.nan, np.nan],
        }
    )
    emb_curation_df.to_csv(df_dir / "embryo_curation_df.csv", index=False)

    # Run
    make_image_snips(root=str(root), train_name="unit_train", label_var=None, rs_factor=1.0, overwrite_flag=True)

    # Outputs
    train_root = root / "training_data" / "unit_train"
    assert (train_root / "embryo_metadata_df_train.csv").exists()

    # Images organized under images/0/
    for f in snip_files:
        assert (train_root / "images" / "0" / f).exists()

