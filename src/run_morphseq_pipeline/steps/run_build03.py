from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.run_morphseq_pipeline.utils import ensure_parent, sample_tracked_df

from src.build.build03A_process_images import (
    segment_wells_sam2_csv,
    segment_wells,
    compile_embryo_stats,
    extract_embryo_snips,
)


def run_build03(
    root: str | Path,
    exp: str,
    sam2_csv: str | None = None,
    by_embryo: int | None = None,
    frames_per_embryo: int | None = None,
    max_samples: int | None = None,
    n_workers: int = 1,
    df01_out: str | Path = "metadata/combined_metadata_files/embryo_metadata_df01.csv",
) -> Path:
    """Run Build03A with either SAM2 CSV bridge or legacy segmentation metadata.
    
    Auto-discovers SAM2 CSV if present at:
    <data_root>/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv

    Writes embryo_metadata_df01.csv to the combined metadata folder by default.
    """
    root = Path(root)

    # 1) Auto-discover SAM2 CSV if not explicitly provided
    if sam2_csv is None:
        auto_csv = root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp}.csv" 
        if auto_csv.exists():
            sam2_csv = str(auto_csv.absolute())  # Use absolute path to avoid doubling
            print(f"🔍 Auto-discovered SAM2 CSV: {sam2_csv}")
        else:
            print(f"🔍 No SAM2 CSV found at {auto_csv}, using legacy segmentation")

    # 2) Load tracking/metadata
    if sam2_csv:
        sam2_csv_path = Path(sam2_csv)
        if not sam2_csv_path.is_absolute():
            sam2_csv_path = root / sam2_csv_path
        print(f"🎯 Using SAM2 segmentation with CSV: {sam2_csv_path}")
        tracked_df = segment_wells_sam2_csv(root=root, exp_name=exp, sam2_csv_path=sam2_csv_path)
    else:
        # Legacy path will expect precomputed master metadata and masks
        print(f"🏗️ Using legacy segmentation for experiment: {exp}")
        tracked_df = segment_wells(root=root, exp_name=exp)

    # 3) Optional subset for validation/dev
    tracked_df = sample_tracked_df(
        tracked_df, by_embryo=by_embryo, frames_per_embryo=frames_per_embryo, max_samples=max_samples
    )

    # 4) Compile stats + QC flags
    stats_df = compile_embryo_stats(root=str(root), tracked_df=tracked_df, n_workers=n_workers)

    # 5) Extract snips
    extract_embryo_snips(root=str(root), stats_df=stats_df, n_workers=n_workers)

    # 6) Write df01 where Build04 expects it
    out_path = Path(df01_out)
    if not out_path.is_absolute():
        out_path = root / out_path
    ensure_parent(out_path)
    stats_df.to_csv(out_path, index=False)
    print(f"✔️  Wrote df01 to {out_path}")
    return out_path
