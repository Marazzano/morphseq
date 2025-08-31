from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.build.build03A_process_images import segment_wells_sam2_csv


def test_segment_wells_sam2_csv_basic(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir(parents=True)

    exp = "20250612_30hpf_ctrl_atf6"
    sam2_csv = root / f"sam2_metadata_{exp}.csv"

    df = pd.DataFrame(
        {
            "experiment_id": [exp, exp],
            "video_id": [f"{exp}_C12", f"{exp}_E06"],
            "image_id": [f"{exp}_C12_ch00_t0000", f"{exp}_E06_ch00_t0001"],
            "embryo_id": [f"{exp}_C12_e00", f"{exp}_E06_e01"],
            "frame_index": [0, 1],
            "exported_mask_path": ["mask_C12_t0000_m1.png", "mask_E06_t0001_m1.png"],
            "bbox_x_min": [10.0, 20.0],
            "bbox_x_max": [30.0, 45.0],
            "bbox_y_min": [5.0, 15.0],
            "bbox_y_max": [25.0, 40.0],
            "start_age_hpf": [30.0, 30.0],
            "Time Rel (s)": [0.0, 3600.0],
            "temperature": [28.5, 28.5],
        }
    )
    df.to_csv(sam2_csv, index=False)

    out = segment_wells_sam2_csv(root=str(root), exp_name=exp, sam2_csv_path=str(sam2_csv))

    assert {"xpos", "ypos", "region_label", "predicted_stage_hpf", "well", "time_int"}.issubset(out.columns)
    # Derived positions
    assert out.loc[out["well"] == "C12", "xpos"].iloc[0] == (10.0 + 30.0) / 2
    assert out.loc[out["well"] == "E06", "ypos"].iloc[0] == (15.0 + 40.0) / 2
    # Derived region label from embryo_id suffix
    assert int(out.loc[out["well"] == "C12", "region_label"].iloc[0]) == 0
    assert int(out.loc[out["well"] == "E06", "region_label"].iloc[0]) == 1

