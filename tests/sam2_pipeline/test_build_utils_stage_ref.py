from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.build.build_utils import generate_stage_ref_from_df01


def test_generate_stage_ref_from_df01_minimal(tmp_path: Path):
    root = tmp_path / "proj"
    df_dir = root / "metadata" / "combined_metadata_files"
    df_dir.mkdir(parents=True)

    # Minimal df01 with required columns
    df01 = pd.DataFrame(
        {
            "experiment_date": ["20250101", "20250101", "20250101", "20250101"],
            "predicted_stage_hpf": [26.2, 30.1, 34.9, 30.0],
            "surface_area_um": [4e5, 6e5, 8e5, 5.5e5],
            "use_embryo_flag": [1, 1, 1, 0],
        }
    )
    df01_path = df_dir / "embryo_metadata_df01.csv"
    df01.to_csv(df01_path, index=False)

    stage_df, params_df = generate_stage_ref_from_df01(root=str(root), df01_path=str(df01_path), quantile=0.95, max_stage=48)

    # Files written
    assert (root / "metadata" / "stage_ref_df.csv").exists()
    assert (root / "metadata" / "stage_ref_params.csv").exists()

    # Basic schema checks
    assert list(stage_df.columns) == ["sa_um", "stage_hpf"]
    assert len(stage_df) == 49  # 0..48 inclusive
    assert set(params_df.columns) == {"offset", "sa_max", "hill_coeff", "inflection_point"}

