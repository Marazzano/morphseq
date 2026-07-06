"""analysis_ready entrypoint — thin filesystem adapter.

Reads the merged feature tables + merged snip_qc + merged latents + plate_metadata, runs the
fan-in join, and writes the assembled analysis_ready table (parquet — the wide z_mu_* block makes
parquet the right container). analysis_ready is an OPTIONAL downstream product; snip_qc remains the
proven through-line terminal.
"""

from __future__ import annotations

from pathlib import Path

from .assemble import read_and_assemble


def run_analysis_ready(
    *,
    curvature_csv: Path,
    stage_predictions_csv: Path,
    mask_geometry_csv: Path,
    pose_kinematics_csv: Path,
    fraction_alive_csv: Path,
    latents_parquet: Path,
    snip_qc_parquet: Path,
    plate_metadata_csv: Path,
    output_parquet: Path,
) -> None:
    df = read_and_assemble(
        snip_qc_path=snip_qc_parquet,
        feature_paths={
            "curvature_metrics": curvature_csv,
            "stage_predictions": stage_predictions_csv,
            "mask_geometry": mask_geometry_csv,
            "pose_kinematics": pose_kinematics_csv,
            "fraction_alive": fraction_alive_csv,
        },
        latents_path=latents_parquet,
        plate_metadata_csv=plate_metadata_csv,
    )

    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
