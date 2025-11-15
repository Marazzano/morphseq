"""
Zebrafish Morphometric Analysis using DLMA Framework

This module provides integration with the Deep Learning-Enabled Morphometric Analysis (DLMA)
framework for zebrafish organ and phenotype detection. It computes mask area percentages for
each detected class relative to total embryo area.

Based on:
- Dong et al., "Deep Learning-Enabled Morphometric Analysis for Toxicity Screening
  Using Zebrafish Larvae" (Computers in Biology and Medicine, 2024)
- GitHub: https://github.com/gonggqing/DLMA
- GitHub: https://github.com/gonggqing/zebrafish_detection

The DLMA model detects 16 object classes:
- 8 specific organs (e.g., eye, heart, yolk, swim bladder, etc.)
- 8 abnormal phenotypes (e.g., pericardial edema, yolk sac edema, spinal curvature, etc.)

Authors: morphseq team
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import skimage.io as io
from collections import defaultdict
import warnings


# DLMA class definitions (16 classes: 8 organs + 8 phenotypes)
# These should match the model's output classes
DLMA_ORGAN_CLASSES = {
    0: 'background',
    1: 'eye',
    2: 'heart',
    3: 'yolk',
    4: 'swim_bladder',
    5: 'otolith',
    6: 'gut',
    7: 'trunk',
    8: 'tail',
}

DLMA_PHENOTYPE_CLASSES = {
    9: 'pericardial_edema',
    10: 'yolk_sac_edema',
    11: 'spinal_curvature',
    12: 'tail_malformation',
    13: 'craniofacial_malformation',
    14: 'reduced_pigmentation',
    15: 'general_edema',
    16: 'hemorrhage',
}

# Combined class mapping
DLMA_ALL_CLASSES = {**DLMA_ORGAN_CLASSES, **DLMA_PHENOTYPE_CLASSES}


def compute_mask_area_percentages(
    instance_masks: Union[np.ndarray, List[np.ndarray]],
    class_ids: Union[np.ndarray, List[int]],
    total_embryo_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute area percentage for each detected class relative to total embryo area.

    This function processes instance segmentation masks from DLMA/Detectron2 and computes
    what percentage of the total embryo area each organ/phenotype occupies.

    Args:
        instance_masks: Either:
            - 3D array (N, H, W) where N is number of instances
            - List of 2D binary masks (H, W)
        class_ids: Array or list of class IDs corresponding to each instance mask
        total_embryo_mask: Optional binary mask defining total embryo area (H, W).
                          If None, uses union of all instance masks.

    Returns:
        Dictionary mapping class names to their area percentages:
        {
            'eye_area_pct': 2.5,
            'heart_area_pct': 1.2,
            'yolk_area_pct': 15.3,
            ...
            'total_embryo_area_px': 50000,
        }
    """
    # Ensure arrays
    if isinstance(instance_masks, list):
        instance_masks = np.array(instance_masks)
    if isinstance(class_ids, list):
        class_ids = np.array(class_ids)

    # Handle different mask formats
    if instance_masks.ndim == 2:
        # Single mask, reshape to (1, H, W)
        instance_masks = instance_masks[np.newaxis, ...]
    elif instance_masks.ndim != 3:
        raise ValueError(f"instance_masks must be 2D or 3D, got shape {instance_masks.shape}")

    # Calculate total embryo area
    if total_embryo_mask is not None:
        total_area = np.sum(total_embryo_mask > 0)
    else:
        # Use union of all instance masks
        union_mask = np.any(instance_masks > 0, axis=0)
        total_area = np.sum(union_mask)

    if total_area == 0:
        warnings.warn("Total embryo area is 0, returning NaN percentages")
        result = {f"{name}_area_pct": np.nan for name in DLMA_ALL_CLASSES.values() if name != 'background'}
        result['total_embryo_area_px'] = 0
        return result

    # Aggregate areas by class
    class_areas = defaultdict(int)

    for mask, class_id in zip(instance_masks, class_ids):
        class_id = int(class_id)
        if class_id == 0:  # Skip background
            continue

        area = np.sum(mask > 0)
        class_name = DLMA_ALL_CLASSES.get(class_id, f'unknown_class_{class_id}')
        class_areas[class_name] += area

    # Compute percentages
    result = {}
    for class_name in DLMA_ALL_CLASSES.values():
        if class_name == 'background':
            continue

        area = class_areas.get(class_name, 0)
        percentage = (area / total_area) * 100.0
        result[f"{class_name}_area_pct"] = float(percentage)

    result['total_embryo_area_px'] = int(total_area)

    return result


def load_dlma_predictions(
    prediction_path: Path,
    format: str = 'detectron2',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DLMA model predictions from various formats.

    Args:
        prediction_path: Path to prediction file
        format: Format of predictions:
            - 'detectron2': Detectron2 pickle output
            - 'numpy': .npz file with 'masks' and 'classes' keys
            - 'masks_dir': Directory with individual mask PNGs

    Returns:
        Tuple of (instance_masks, class_ids)
        - instance_masks: (N, H, W) array
        - class_ids: (N,) array
    """
    if format == 'detectron2':
        # Detectron2 outputs are typically pickled
        import pickle
        with open(prediction_path, 'rb') as f:
            outputs = pickle.load(f)

        instances = outputs['instances']
        masks = instances.pred_masks.cpu().numpy()  # (N, H, W)
        classes = instances.pred_classes.cpu().numpy()  # (N,)

        return masks, classes

    elif format == 'numpy':
        # Numpy archive format
        data = np.load(prediction_path)
        masks = data['masks']
        classes = data['classes']
        return masks, classes

    elif format == 'masks_dir':
        # Individual mask files named like: {class_id}_{instance_id}.png
        raise NotImplementedError("Directory-based mask loading not yet implemented")

    else:
        raise ValueError(f"Unknown format: {format}")


def extract_morphometric_features_batch(
    tracking_df: pd.DataFrame,
    dlma_predictions_dir: Path,
    embryo_mask_dir: Optional[Path] = None,
    prediction_format: str = 'detectron2',
    pixel_size_col: str = 'micrometers_per_pixel',
) -> pd.DataFrame:
    """
    Extract DLMA morphometric features for batch of snips.

    This is the main batch processing function that follows the morphseq pattern.
    It processes multiple snips and returns a DataFrame with area percentages.

    Args:
        tracking_df: Segmentation tracking DataFrame with snip_id
        dlma_predictions_dir: Directory containing DLMA prediction files
        embryo_mask_dir: Optional directory with total embryo masks
        prediction_format: Format of DLMA predictions ('detectron2', 'numpy')
        pixel_size_col: Column name for pixel size (for future micrometer conversion)

    Returns:
        DataFrame with columns:
        - snip_id
        - {class_name}_area_pct for each DLMA class
        - total_embryo_area_px
        - total_embryo_area_um2 (if pixel_size available)
    """
    results = []

    for idx, row in tracking_df.iterrows():
        snip_id = row['snip_id']
        image_id = row.get('image_id', snip_id.rsplit('_', 1)[0])

        # Look for DLMA prediction file
        # Try multiple naming conventions
        pred_path = None
        for ext in ['.pkl', '.npz', '.pickle']:
            candidate = dlma_predictions_dir / f"{image_id}_dlma{ext}"
            if candidate.exists():
                pred_path = candidate
                break
            candidate = dlma_predictions_dir / f"{snip_id}_dlma{ext}"
            if candidate.exists():
                pred_path = candidate
                break

        if pred_path is None:
            # No DLMA predictions found, return NaN
            result = {'snip_id': snip_id}
            for class_name in DLMA_ALL_CLASSES.values():
                if class_name != 'background':
                    result[f"{class_name}_area_pct"] = np.nan
            result['total_embryo_area_px'] = np.nan
            results.append(result)
            continue

        try:
            # Load DLMA predictions
            instance_masks, class_ids = load_dlma_predictions(pred_path, prediction_format)

            # Load total embryo mask if available
            total_mask = None
            if embryo_mask_dir is not None:
                mask_path = embryo_mask_dir / f"{image_id}_masks.png"
                if not mask_path.exists():
                    mask_path = embryo_mask_dir / f"{snip_id}_mask.png"

                if mask_path.exists():
                    total_mask = io.imread(mask_path)

            # Compute area percentages
            metrics = compute_mask_area_percentages(instance_masks, class_ids, total_mask)
            metrics['snip_id'] = snip_id

            # Add micrometers squared if pixel size available
            if pixel_size_col in row and not pd.isna(row[pixel_size_col]):
                pixel_size = row[pixel_size_col]
                metrics['total_embryo_area_um2'] = metrics['total_embryo_area_px'] * (pixel_size ** 2)

            results.append(metrics)

        except Exception as e:
            print(f"Warning: Failed to process {snip_id}: {e}")
            result = {'snip_id': snip_id}
            for class_name in DLMA_ALL_CLASSES.values():
                if class_name != 'background':
                    result[f"{class_name}_area_pct"] = np.nan
            result['total_embryo_area_px'] = np.nan
            results.append(result)

    return pd.DataFrame(results)


def summarize_morphometric_features(
    features_df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Summarize morphometric features across groups (e.g., by treatment, timepoint).

    Args:
        features_df: DataFrame from extract_morphometric_features_batch
        group_by: Optional list of columns to group by (e.g., ['treatment', 'timepoint'])

    Returns:
        Summary DataFrame with mean, std, and detection frequency for each class
    """
    if group_by is None:
        group_by = []

    # Get all percentage columns
    pct_cols = [col for col in features_df.columns if col.endswith('_area_pct')]

    if not group_by:
        # Overall summary
        summary = {
            'n_samples': len(features_df),
        }

        for col in pct_cols:
            class_name = col.replace('_area_pct', '')
            # Only compute stats for non-NaN values
            valid_data = features_df[col].dropna()

            summary[f"{class_name}_mean_pct"] = valid_data.mean() if len(valid_data) > 0 else np.nan
            summary[f"{class_name}_std_pct"] = valid_data.std() if len(valid_data) > 0 else np.nan
            summary[f"{class_name}_detection_freq"] = (valid_data > 0).sum() / len(features_df)

        return pd.DataFrame([summary])

    else:
        # Group-wise summary
        summaries = []

        for group_vals, group_df in features_df.groupby(group_by):
            summary = {}

            # Add group identifiers
            if isinstance(group_vals, tuple):
                for col, val in zip(group_by, group_vals):
                    summary[col] = val
            else:
                summary[group_by[0]] = group_vals

            summary['n_samples'] = len(group_df)

            for col in pct_cols:
                class_name = col.replace('_area_pct', '')
                valid_data = group_df[col].dropna()

                summary[f"{class_name}_mean_pct"] = valid_data.mean() if len(valid_data) > 0 else np.nan
                summary[f"{class_name}_std_pct"] = valid_data.std() if len(valid_data) > 0 else np.nan
                summary[f"{class_name}_detection_freq"] = (valid_data > 0).sum() / len(group_df)

            summaries.append(summary)

        return pd.DataFrame(summaries)


def get_organ_names() -> List[str]:
    """Get list of organ class names."""
    return [name for name in DLMA_ORGAN_CLASSES.values() if name != 'background']


def get_phenotype_names() -> List[str]:
    """Get list of phenotype class names."""
    return list(DLMA_PHENOTYPE_CLASSES.values())


def get_all_class_names() -> List[str]:
    """Get list of all class names (organs + phenotypes)."""
    return [name for name in DLMA_ALL_CLASSES.values() if name != 'background']


# Example usage
if __name__ == "__main__":
    print("Zebrafish Morphometric Analysis Module")
    print("=" * 60)
    print("\nDLMA Model Classes:")
    print("\nOrgans (8):")
    for class_id, name in DLMA_ORGAN_CLASSES.items():
        if name != 'background':
            print(f"  {class_id}: {name}")

    print("\nPhenotypes (8):")
    for class_id, name in DLMA_PHENOTYPE_CLASSES.items():
        print(f"  {class_id}: {name}")

    print("\n" + "=" * 60)
    print("\nExample: Computing area percentages from random masks")

    # Create dummy data
    np.random.seed(42)
    H, W = 512, 512

    # Simulate 3 detected instances
    masks = np.random.rand(3, H, W) > 0.7
    class_ids = np.array([1, 2, 3])  # eye, heart, yolk

    # Compute percentages
    result = compute_mask_area_percentages(masks, class_ids)

    print("\nResults:")
    for key, value in result.items():
        if not key.startswith('total'):
            if value > 0:
                print(f"  {key}: {value:.2f}%")
    print(f"  Total area: {result['total_embryo_area_px']} pixels")
