#!/usr/bin/env python3
"""
Convert DLMA Detectron2 Predictions to NumPy Format

This utility converts Detectron2 pickle files to lightweight NumPy .npz archives,
decoupling model inference from downstream feature extraction.

Usage:
    # Convert single file
    python convert_dlma_to_npz.py predictions/zebrafish_001_dlma.pkl

    # Convert entire directory
    python convert_dlma_to_npz.py predictions/ --output-dir predictions_npz/

    # Batch convert with pattern
    python convert_dlma_to_npz.py predictions/ --pattern "*_dlma.pkl"

Benefits:
    - No Detectron2/PyTorch required for feature extraction
    - Smaller file sizes (~10x reduction)
    - Faster loading times
    - Pure NumPy/scikit-image dependency

Author: morphseq team
Date: 2025-11-15
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional
import sys


def convert_detectron2_to_npz(
    pickle_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> Path:
    """
    Convert Detectron2 pickle output to NumPy .npz format.

    Args:
        pickle_path: Path to Detectron2 .pkl file
        output_path: Optional output path (default: same name with .npz)
        verbose: Print conversion details

    Returns:
        Path to created .npz file

    Raises:
        ImportError: If Detectron2/torch not available
        FileNotFoundError: If pickle file doesn't exist
    """
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    # Import here so error message is clear
    try:
        import numpy as np
        import torch
    except ImportError as e:
        print("ERROR: This conversion script requires PyTorch", file=sys.stderr)
        print("Install with: pip install torch", file=sys.stderr)
        raise

    if verbose:
        print(f"Loading: {pickle_path}")

    # Load Detectron2 output
    with open(pickle_path, 'rb') as f:
        outputs = pickle.load(f)

    # Extract instances to CPU
    instances = outputs['instances'].to('cpu')

    # Convert to numpy
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    # Determine output path
    if output_path is None:
        output_path = pickle_path.with_suffix('.npz')

    # Save as compressed NumPy archive
    np.savez_compressed(
        output_path,
        masks=masks,
        classes=classes,
        scores=scores,
        boxes=boxes,
        image_height=instances.image_size[0],
        image_width=instances.image_size[1],
    )

    if verbose:
        pkl_size = pickle_path.stat().st_size / 1024 / 1024
        npz_size = output_path.stat().st_size / 1024 / 1024
        reduction = (1 - npz_size / pkl_size) * 100
        print(f"Saved: {output_path}")
        print(f"  Detections: {len(masks)}")
        print(f"  Image size: {instances.image_size[0]}×{instances.image_size[1]}")
        print(f"  File size: {pkl_size:.2f} MB → {npz_size:.2f} MB ({reduction:.1f}% reduction)")

    return output_path


def convert_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    pattern: str = "*.pkl",
    verbose: bool = True,
) -> list[Path]:
    """
    Convert all Detectron2 pickle files in a directory.

    Args:
        input_dir: Directory containing .pkl files
        output_dir: Output directory (default: same as input)
        pattern: Glob pattern for pickle files
        verbose: Print conversion details

    Returns:
        List of created .npz files
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    # Find all matching files
    pickle_files = list(input_dir.glob(pattern))

    if not pickle_files:
        print(f"WARNING: No files matching '{pattern}' found in {input_dir}", file=sys.stderr)
        return []

    if verbose:
        print(f"Found {len(pickle_files)} files to convert")
        print()

    # Create output directory if specified
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each file
    converted_files = []
    for i, pkl_path in enumerate(pickle_files, 1):
        if verbose:
            print(f"[{i}/{len(pickle_files)}]", end=" ")

        try:
            # Determine output path
            if output_dir is not None:
                out_path = output_dir / pkl_path.with_suffix('.npz').name
            else:
                out_path = None

            # Convert
            npz_path = convert_detectron2_to_npz(pkl_path, out_path, verbose=verbose)
            converted_files.append(npz_path)

            if verbose:
                print()

        except Exception as e:
            print(f"ERROR converting {pkl_path}: {e}", file=sys.stderr)
            continue

    if verbose:
        print()
        print(f"Successfully converted {len(converted_files)}/{len(pickle_files)} files")

    return converted_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert DLMA Detectron2 predictions to NumPy format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_dlma_to_npz.py predictions/zebrafish_001_dlma.pkl

  # Convert directory (in-place)
  python convert_dlma_to_npz.py predictions/

  # Convert to different directory
  python convert_dlma_to_npz.py predictions/ --output-dir predictions_npz/

  # Custom pattern
  python convert_dlma_to_npz.py data/ --pattern "*_output.pkl"
        """
    )

    parser.add_argument(
        'input',
        type=Path,
        help='Input .pkl file or directory containing .pkl files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: same as input)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.pkl',
        help='Glob pattern for finding pickle files in directory (default: *.pkl)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    try:
        if args.input.is_file():
            # Convert single file
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.output_dir / args.input.with_suffix('.npz').name
            else:
                output_path = None

            convert_detectron2_to_npz(args.input, output_path, verbose=verbose)

        elif args.input.is_dir():
            # Convert directory
            convert_directory(
                args.input,
                args.output_dir,
                args.pattern,
                verbose=verbose
            )

        else:
            print(f"ERROR: {args.input} is neither a file nor directory", file=sys.stderr)
            sys.exit(1)

    except ImportError:
        print("\nThis conversion script must run in an environment with Detectron2 installed.", file=sys.stderr)
        print("After conversion, feature extraction only needs NumPy.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
