from __future__ import annotations
from pathlib import Path

from src.build.build01A_compile_keyence_torch import build_ff_from_keyence, stitch_ff_from_keyence
from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1


def run_build01(
    root: str | Path,
    exp: str | None,
    microscope: str,
    metadata_only: bool = False,
    overwrite: bool = False,
) -> None:
    """Run Build01 for Keyence or YX1.

    Notes:
    - Writes per-experiment built metadata to `metadata/built_metadata_files/{exp}_metadata.csv`.
    - Writes stitched FF images under `built_image_data/stitched_FF_images/{exp}/`.
    """
    if not exp:
        raise SystemExit("--exp is required for build01")
    root = Path(root)

    if microscope == "keyence":
        build_ff_from_keyence(
            data_root=str(root), repo_root=str(root), exp_name=exp,
            metadata_only=metadata_only, overwrite=overwrite,
        )
        # For Keyence runs, stitching is a separate step
        stitch_ff_from_keyence(data_root=str(root), exp_name=exp, overwrite=overwrite)
    elif microscope == "yx1":
        build_ff_from_yx1(
            data_root=str(root), repo_root=str(root), exp_name=exp,
            metadata_only=metadata_only, overwrite=overwrite,
        )
    print("✔️  Build01 complete.")
