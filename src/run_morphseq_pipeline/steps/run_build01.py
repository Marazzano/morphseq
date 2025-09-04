from __future__ import annotations
from pathlib import Path

from src.build.build01A_compile_keyence_torch import build_ff_from_keyence, stitch_ff_from_keyence
from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1


def validate_build01_inputs(root: Path, exp: str, microscope: str) -> None:
    """Validate that all required inputs exist for Build01.
    
    Args:
        root: Data root directory
        exp: Experiment name
        microscope: Microscope type ('Keyence' or 'YX1')

    Raises:
        FileNotFoundError: If required files are missing with detailed guidance
    """
    errors = []
    
    # Check raw image data exists
    raw_data_dir = root / "raw_image_data" / microscope / exp
    if not raw_data_dir.exists():
        errors.append(f"Raw image data directory not found: {raw_data_dir}")
        errors.append(f"Expected structure: <data-root>/raw_image_data/{microscope.upper()}/{exp}/")
    else:
        # Check if directory has image files
        if microscope == "keyence":
            image_files = list(raw_data_dir.glob("**/*.VGI")) + list(raw_data_dir.glob("**/*.vgi"))
            if not image_files:
                errors.append(f"No Keyence VGI files found in: {raw_data_dir}")
        elif microscope == "yx1":
            image_files = list(raw_data_dir.glob("**/*.nd2"))
            if not image_files:
                errors.append(f"No YX1 ND2 files found in: {raw_data_dir}")
    
    # Check well/plate metadata Excel file exists in either location
    well_meta_dir = root / "metadata" / "well_metadata"
    plate_meta_dir = root / "metadata" / "plate_metadata"

    # Prefer well_metadata if present, otherwise fall back to plate_metadata
    if well_meta_dir.exists():
        metadata_dir = well_meta_dir
    else:
        metadata_dir = plate_meta_dir

    well_metadata_file = metadata_dir / f"{exp}_well_metadata.xlsx"

    if not (well_meta_dir.exists() or plate_meta_dir.exists()):
        errors.append(f"❌ No metadata directory found. Expected one of: {well_meta_dir} or {plate_meta_dir}")
    elif not well_metadata_file.exists():
        errors.append(f"❌ Required metadata file not found: {well_metadata_file}")
        errors.append(f"   This Excel file should contain experimental metadata sheets")
    else:
        # Basic validation - just check that required sheets exist
        try:
            import pandas as pd
            with pd.ExcelFile(well_metadata_file) as xlf:
                required_sheets = ["medium", "genotype", "start_age_hpf", "temperature"]
                missing_sheets = [s for s in required_sheets if s not in xlf.sheet_names]

                if missing_sheets:
                    errors.append(f"❌ Missing required sheets in {well_metadata_file.name}: {', '.join(missing_sheets)}")

        except Exception as e:
            errors.append(f"❌ Cannot read Excel file {well_metadata_file}: {e}")
    
    if errors:
        error_message = "❌ Build01 validation failed:\n\n" + "\n".join(f"  • {error}" for error in errors)
        raise FileNotFoundError(error_message)


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
    
    # Validate all required inputs exist before processing
    print("🔍 Validating Build01 inputs...")
    try:
        validate_build01_inputs(root, exp, microscope)
        print("✅ Build01 validation passed")
    except FileNotFoundError as e:
        print(str(e))
        raise SystemExit(1) from e

    if microscope == "Keyence":
        build_ff_from_keyence(
            data_root=str(root), repo_root=str(root), exp_name=exp,
            metadata_only=metadata_only, overwrite=overwrite,
        )
        # For Keyence runs, stitching is a separate step
        stitch_ff_from_keyence(data_root=str(root), exp_name=exp, overwrite=overwrite)
    elif microscope == "YX1":
        build_ff_from_yx1(
            data_root=str(root), repo_root=str(root), exp_name=exp,
            metadata_only=metadata_only, overwrite=overwrite,
        )
    print("✔️  Build01 complete.")
