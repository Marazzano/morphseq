#!/usr/bin/env python3
"""
DLMA Inference Example for morphseq Integration

This script demonstrates:
1. Running DLMA inference on zebrafish images
2. Saving predictions in morphseq-compatible format
3. Computing mask area percentages
4. Analyzing results

Usage:
    python dlma_inference_example.py --image_dir data/images --output_dir data/dlma_predictions

Requirements:
    - DLMA repository cloned
    - Detectron2 installed
    - Model weights downloaded

Author: morphseq team
Date: 2025-11-15
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List
import warnings

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import morphseq integration
from zebrafish_morphometric_analysis import (
    compute_mask_area_percentages,
    extract_morphometric_features_batch,
    summarize_morphometric_features,
    DLMA_ALL_CLASSES,
)


def setup_dlma_predictor(model_weights_path: str, device: str = 'cuda'):
    """
    Set up DLMA/Detectron2 predictor.

    Args:
        model_weights_path: Path to DLMA model weights (.pth file)
        device: 'cuda' or 'cpu'

    Returns:
        Detectron2 DefaultPredictor instance
    """
    try:
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
    except ImportError:
        raise ImportError(
            "Detectron2 not installed. Please install following DLMA_INSTALLATION.md"
        )

    # Create config
    cfg = get_cfg()

    # Load base Mask R-CNN config
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )

    # DLMA-specific settings
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # 8 organs + 8 phenotypes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection confidence threshold
    cfg.MODEL.DEVICE = device

    # Create predictor
    predictor = DefaultPredictor(cfg)

    return predictor


def run_dlma_inference_single(
    predictor,
    image_path: Path,
    output_dir: Path,
    save_format: str = 'pickle',
) -> Dict:
    """
    Run DLMA inference on a single image.

    Args:
        predictor: Detectron2 predictor
        image_path: Path to input image
        output_dir: Directory to save predictions
        save_format: 'pickle' or 'numpy'

    Returns:
        Dictionary with prediction results
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Run inference
    outputs = predictor(image)

    # Extract results
    instances = outputs['instances'].to('cpu')
    masks = instances.pred_masks.numpy()  # (N, H, W)
    classes = instances.pred_classes.numpy()  # (N,)
    scores = instances.scores.numpy()  # (N,)
    boxes = instances.pred_boxes.tensor.numpy()  # (N, 4)

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = image_path.stem

    if save_format == 'pickle':
        output_path = output_dir / f"{image_stem}_dlma.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(outputs, f)

    elif save_format == 'numpy':
        output_path = output_dir / f"{image_stem}_dlma.npz"
        np.savez(
            output_path,
            masks=masks,
            classes=classes,
            scores=scores,
            boxes=boxes,
        )

    # Return summary
    result = {
        'image_path': str(image_path),
        'n_detections': len(masks),
        'output_path': str(output_path),
        'detected_classes': [DLMA_ALL_CLASSES.get(c, f'unknown_{c}') for c in classes],
        'scores': scores.tolist(),
    }

    return result


def run_dlma_batch_inference(
    image_dir: Path,
    output_dir: Path,
    model_weights_path: str,
    device: str = 'cuda',
    file_extensions: List[str] = ['.png', '.jpg', '.tif', '.tiff'],
) -> pd.DataFrame:
    """
    Run DLMA inference on a batch of images.

    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save predictions
        model_weights_path: Path to DLMA model weights
        device: 'cuda' or 'cpu'
        file_extensions: List of image file extensions to process

    Returns:
        DataFrame with inference results
    """
    # Setup predictor
    print("Setting up DLMA predictor...")
    predictor = setup_dlma_predictor(model_weights_path, device)

    # Find images
    image_paths = []
    for ext in file_extensions:
        image_paths.extend(image_dir.glob(f'*{ext}'))

    print(f"Found {len(image_paths)} images to process")

    # Run inference
    results = []
    for image_path in tqdm(image_paths, desc="Running DLMA inference"):
        try:
            result = run_dlma_inference_single(
                predictor,
                image_path,
                output_dir,
                save_format='pickle',
            )
            results.append(result)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': str(image_path),
                'error': str(e),
            })

    return pd.DataFrame(results)


def compute_area_percentages_from_predictions(
    prediction_path: Path,
    embryo_mask_path: Path = None,
) -> Dict[str, float]:
    """
    Compute mask area percentages from saved DLMA predictions.

    Args:
        prediction_path: Path to .pkl or .npz prediction file
        embryo_mask_path: Optional path to total embryo mask

    Returns:
        Dictionary with area percentages
    """
    # Load predictions
    if prediction_path.suffix == '.pkl':
        with open(prediction_path, 'rb') as f:
            outputs = pickle.load(f)
        instances = outputs['instances'].to('cpu')
        masks = instances.pred_masks.numpy()
        classes = instances.pred_classes.numpy()

    elif prediction_path.suffix == '.npz':
        data = np.load(prediction_path)
        masks = data['masks']
        classes = data['classes']

    else:
        raise ValueError(f"Unsupported format: {prediction_path.suffix}")

    # Load embryo mask if provided
    embryo_mask = None
    if embryo_mask_path and embryo_mask_path.exists():
        embryo_mask = cv2.imread(str(embryo_mask_path), cv2.IMREAD_GRAYSCALE)

    # Compute percentages
    percentages = compute_mask_area_percentages(masks, classes, embryo_mask)

    return percentages


def visualize_detections(
    image_path: Path,
    prediction_path: Path,
    output_path: Path,
    show_labels: bool = True,
):
    """
    Visualize DLMA detections on the original image.

    Args:
        image_path: Path to input image
        prediction_path: Path to prediction file
        output_path: Path to save visualization
        show_labels: Whether to show class labels
    """
    try:
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
    except ImportError:
        warnings.warn("Detectron2 not available, skipping visualization")
        return

    # Load image
    image = cv2.imread(str(image_path))

    # Load predictions
    with open(prediction_path, 'rb') as f:
        outputs = pickle.load(f)

    # Setup metadata with class names
    metadata = MetadataCatalog.get("dlma_dataset")
    metadata.thing_classes = [DLMA_ALL_CLASSES.get(i, f'class_{i}') for i in range(17)]

    # Visualize
    visualizer = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        instance_mode=ColorMode.IMAGE,
    )

    vis = visualizer.draw_instance_predictions(outputs['instances'].to('cpu'))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(str(output_path))


def example_full_pipeline(
    image_dir: Path,
    output_base_dir: Path,
    model_weights_path: str,
    tracking_csv_path: Path = None,
):
    """
    Example of full DLMA + morphseq integration pipeline.

    Args:
        image_dir: Directory with zebrafish images
        output_base_dir: Base directory for outputs
        model_weights_path: Path to DLMA model weights
        tracking_csv_path: Optional tracking CSV (if None, will create from images)
    """
    print("=" * 80)
    print("DLMA + morphseq Integration Pipeline")
    print("=" * 80)

    # Setup directories
    predictions_dir = output_base_dir / "dlma_predictions"
    features_dir = output_base_dir / "morphometric_features"
    viz_dir = output_base_dir / "visualizations"

    predictions_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run DLMA inference
    print("\nStep 1: Running DLMA inference...")
    inference_results = run_dlma_batch_inference(
        image_dir=image_dir,
        output_dir=predictions_dir,
        model_weights_path=model_weights_path,
        device='cuda',
    )
    inference_results.to_csv(features_dir / "dlma_inference_summary.csv", index=False)
    print(f"Processed {len(inference_results)} images")

    # Step 2: Create tracking DataFrame if not provided
    if tracking_csv_path is None or not tracking_csv_path.exists():
        print("\nStep 2: Creating tracking DataFrame...")
        tracking_df = pd.DataFrame({
            'snip_id': [Path(p).stem for p in inference_results['image_path']],
            'image_id': [Path(p).stem for p in inference_results['image_path']],
            'um_per_pixel': 1.0,  # Update with actual value from microscope calibration
        })
    else:
        print(f"\nStep 2: Loading tracking DataFrame from {tracking_csv_path}...")
        tracking_df = pd.read_csv(tracking_csv_path)

    # Step 2.5: Convert predictions to NumPy format (optional but recommended)
    print("\nStep 2.5: Converting predictions to NumPy format...")
    from convert_dlma_to_npz import convert_directory
    try:
        convert_directory(predictions_dir, verbose=False)
        print("  ✓ Converted to .npz format (no Detectron2 needed for feature extraction)")
        prediction_format_to_use = 'numpy'
    except Exception as e:
        print(f"  Warning: Could not convert to .npz: {e}")
        print("  Will use Detectron2 format (requires Detectron2 installed)")
        prediction_format_to_use = 'detectron2'

    # Step 3: Extract morphometric features
    print("\nStep 3: Extracting morphometric features...")
    features_df = extract_morphometric_features_batch(
        tracking_df=tracking_df,
        dlma_predictions_dir=predictions_dir,
        prediction_format=prediction_format_to_use,
    )
    features_df.to_csv(features_dir / "morphometric_features.csv", index=False)
    print(f"Extracted features for {len(features_df)} snips")

    # Step 4: Summarize results
    print("\nStep 4: Summarizing morphometric features...")
    summary_df = summarize_morphometric_features(features_df)
    summary_df.to_csv(features_dir / "morphometric_summary.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal images processed: {len(features_df)}")
    print(f"\nMean area percentages (for detected organs/phenotypes):")

    for col in features_df.columns:
        if col.endswith('_area_pct'):
            class_name = col.replace('_area_pct', '')
            values = features_df[col].dropna()
            detected = (values > 0).sum()
            if detected > 0:
                mean_pct = values[values > 0].mean()
                print(f"  {class_name:30s}: {mean_pct:6.2f}% (detected in {detected}/{len(features_df)} images)")

    print("\n" + "=" * 80)
    print(f"Results saved to: {features_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run DLMA inference and compute morphometric features"
    )
    parser.add_argument(
        '--image_dir',
        type=Path,
        required=True,
        help='Directory containing zebrafish images'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for predictions and features'
    )
    parser.add_argument(
        '--model_weights',
        type=str,
        required=True,
        help='Path to DLMA model weights (.pth file)'
    )
    parser.add_argument(
        '--tracking_csv',
        type=Path,
        default=None,
        help='Optional tracking CSV with snip_id and metadata'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )

    args = parser.parse_args()

    # Run full pipeline
    example_full_pipeline(
        image_dir=args.image_dir,
        output_base_dir=args.output_dir,
        model_weights_path=args.model_weights,
        tracking_csv_path=args.tracking_csv,
    )


if __name__ == "__main__":
    main()
