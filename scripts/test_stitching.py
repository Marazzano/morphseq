#!/usr/bin/env python3
"""
Standalone test script for image stitching (YX1 and Keyence microscopes).

Quick validation of GPU/CPU stitching functionality with timing and output checks.

Usage:
    python scripts/test_stitching.py --experiment test_yx1_001 --microscope YX1 --device cuda
    python scripts/test_stitching.py --experiment test_keyence_001 --microscope Keyence
"""

import sys
import argparse
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import pandas as pd
from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1
from src.build.build01A_compile_keyence_torch import build_ff_from_keyence, stitch_ff_from_keyence


def check_gpu_info():
    """Report GPU availability and details."""
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)

    if torch.cuda.is_available():
        print(f"✓ CUDA available: True")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

        # Memory info
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  Memory allocated: {mem_allocated:.2f} GB")
            print(f"  Memory reserved: {mem_reserved:.2f} GB")
    else:
        print(f"✗ CUDA available: False (CPU mode)")
    print("="*60 + "\n")


def validate_yx1_outputs(data_root: Path, exp_name: str, n_frames: int = 3):
    """Validate YX1 stitched outputs exist and have correct properties."""
    print("\n" + "-"*60)
    print("VALIDATING YX1 OUTPUTS")
    print("-"*60)

    stitch_dir = data_root / "built_image_data" / "stitched_FF_images" / exp_name

    # Check metadata
    meta_file = data_root / "metadata" / "built_metadata_files" / f"{exp_name}_metadata.csv"
    if not meta_file.exists():
        print(f"✗ Metadata file missing: {meta_file}")
        return False

    meta_df = pd.read_csv(meta_file)
    print(f"✓ Metadata file exists: {len(meta_df)} rows")

    # Check stitched images
    stitch_files = sorted(stitch_dir.glob("*_stitch.jpg"))
    print(f"✓ Found {len(stitch_files)} stitched images")

    if len(stitch_files) == 0:
        print(f"✗ No stitched images found in {stitch_dir}")
        return False

    # Check image properties
    import skimage.io as skio
    for img_path in stitch_files[:3]:  # Check first 3
        img = skio.imread(img_path)
        print(f"  {img_path.name}: shape={img.shape}, dtype={img.dtype}")

    print("-"*60)
    return True


def validate_keyence_outputs(data_root: Path, exp_name: str, n_frames: int = 3):
    """Validate Keyence stitched outputs exist and have correct properties."""
    print("\n" + "-"*60)
    print("VALIDATING KEYENCE OUTPUTS")
    print("-"*60)

    ff_dir = data_root / "built_image_data" / "Keyence" / "FF_images" / exp_name
    stitch_dir = data_root / "built_image_data" / "stitched_FF_images" / exp_name

    # Check metadata
    meta_file = data_root / "metadata" / "built_metadata_files" / f"{exp_name}_metadata.csv"
    if not meta_file.exists():
        print(f"✗ Metadata file missing: {meta_file}")
        return False

    meta_df = pd.read_csv(meta_file)
    print(f"✓ Metadata file exists: {len(meta_df)} rows")

    # Check FF tiles
    ff_folders = sorted(ff_dir.glob("ff_*"))
    print(f"✓ Found {len(ff_folders)} FF tile folders")

    if len(ff_folders) > 0:
        first_folder = ff_folders[0]
        tiles = sorted(first_folder.glob("*.jpg"))
        print(f"  Example: {first_folder.name} has {len(tiles)} tiles")

    # Check stitched images
    stitch_files = sorted(stitch_dir.glob("*_stitch.jpg"))
    print(f"✓ Found {len(stitch_files)} stitched images")

    if len(stitch_files) == 0:
        print(f"✗ No stitched images found in {stitch_dir}")
        return False

    # Check image properties
    import skimage.io as skio
    for img_path in stitch_files[:3]:  # Check first 3
        img = skio.imread(img_path)
        print(f"  {img_path.name}: shape={img.shape}, dtype={img.dtype}")

    print("-"*60)
    return True


def test_yx1_stitching(data_root: Path, exp_name: str, device: str, n_frames: int = 3):
    """Test YX1 stitching pipeline."""
    print("\n" + "="*60)
    print(f"TESTING YX1 STITCHING: {exp_name}")
    print(f"Device: {device}")
    print("="*60)

    # Time the operation
    start_time = time.time()

    try:
        build_ff_from_yx1(
            data_root=data_root,
            repo_root=REPO_ROOT,
            exp_name=exp_name,
            device=device,
            n_workers=1,
            overwrite=True,
            metadata_only=False
        )

        elapsed = time.time() - start_time
        print(f"\n✓ YX1 stitching completed in {elapsed:.2f} seconds")

        # Validate outputs
        success = validate_yx1_outputs(data_root, exp_name, n_frames)

        return success

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ YX1 stitching failed after {elapsed:.2f} seconds")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keyence_stitching(data_root: Path, exp_name: str, n_frames: int = 3):
    """Test Keyence stitching pipeline."""
    print("\n" + "="*60)
    print(f"TESTING KEYENCE STITCHING: {exp_name}")
    print("="*60)

    # Time the FF operation
    start_time = time.time()

    try:
        # Step 1: Build FF images
        print("\nStep 1: Building FF images...")
        build_ff_from_keyence(
            data_root=data_root,
            repo_root=REPO_ROOT,
            exp_name=exp_name,
            overwrite=True,
            metadata_only=False
        )

        ff_elapsed = time.time() - start_time
        print(f"✓ FF building completed in {ff_elapsed:.2f} seconds")

        # Step 2: Stitch FF images
        print("\nStep 2: Stitching FF images...")
        stitch_start = time.time()

        stitch_ff_from_keyence(
            data_root=data_root,
            exp_name=exp_name,
            n_workers=1,
            overwrite=True
        )

        stitch_elapsed = time.time() - stitch_start
        total_elapsed = time.time() - start_time

        print(f"✓ Stitching completed in {stitch_elapsed:.2f} seconds")
        print(f"✓ Total pipeline time: {total_elapsed:.2f} seconds")

        # Validate outputs
        success = validate_keyence_outputs(data_root, exp_name, n_frames)

        return success

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Keyence stitching failed after {elapsed:.2f} seconds")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test image stitching for YX1 or Keyence microscopes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test YX1 with GPU
  python scripts/test_stitching.py --experiment test_yx1_001 --microscope YX1 --device cuda

  # Test YX1 with CPU
  python scripts/test_stitching.py --experiment test_yx1_001 --microscope YX1 --device cpu

  # Test Keyence (auto-detects GPU)
  python scripts/test_stitching.py --experiment test_keyence_001 --microscope Keyence

  # Custom data root
  python scripts/test_stitching.py --experiment test_yx1_001 --microscope YX1 \\
      --data-root /path/to/data
        """
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (e.g., test_yx1_001, test_keyence_001)"
    )

    parser.add_argument(
        "--microscope",
        required=True,
        choices=["YX1", "Keyence"],
        help="Microscope type"
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device for computation (YX1 only; default: auto-detect)"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "test_data" / "real_subset_yx1" if "--microscope YX1" in " ".join(sys.argv)
                else REPO_ROOT / "test_data" / "real_subset_keyence",
        help="Data root directory (default: test_data/real_subset_<microscope>)"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=3,
        help="Number of frames to process (default: 3)"
    )

    args = parser.parse_args()

    # Auto-detect data root based on microscope if not specified
    if args.data_root == REPO_ROOT / "test_data" / "real_subset_yx1":
        if args.microscope == "Keyence":
            args.data_root = REPO_ROOT / "test_data" / "real_subset_keyence"

    # Auto-detect device if not specified (YX1)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*60)
    print("STITCHING TEST SCRIPT")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Microscope: {args.microscope}")
    print(f"Data root: {args.data_root}")
    print(f"Frames to process: {args.frames}")
    if args.microscope == "YX1":
        print(f"Device: {args.device}")
    print("="*60)

    # Check GPU info
    check_gpu_info()

    # Run appropriate test
    if args.microscope == "YX1":
        success = test_yx1_stitching(
            data_root=args.data_root,
            exp_name=args.experiment,
            device=args.device,
            n_frames=args.frames
        )
    else:  # Keyence
        success = test_keyence_stitching(
            data_root=args.data_root,
            exp_name=args.experiment,
            n_frames=args.frames
        )

    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    if success:
        print("✓✓✓ All tests PASSED ✓✓✓")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("✗✗✗ Tests FAILED ✗✗✗")
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
