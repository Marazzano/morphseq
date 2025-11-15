#!/usr/bin/env python3
"""
DLMA Output Format Examples - Practical Demonstrations

This script shows concrete examples of what DLMA outputs look like at each stage
and how to work with different data formats.

Author: morphseq team
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List

# ============================================================================
# EXAMPLE 1: Raw Detectron2 Output Structure
# ============================================================================

def show_detectron2_output_structure():
    """
    Demonstrates what a raw Detectron2 output looks like (simulated).
    """
    print("=" * 80)
    print("EXAMPLE 1: Detectron2 Raw Output Structure")
    print("=" * 80)

    # Simulated output from: outputs = predictor(image)
    print("\nAfter running: outputs = predictor(image)\n")

    # This is what you get:
    simulated_output = {
        'instances': {
            'num_instances': 5,
            'image_height': 512,
            'image_width': 512,
            'pred_boxes': np.array([
                [120.5, 180.3, 145.2, 210.8],  # eye bbox
                [115.0, 215.5, 135.0, 240.2],  # heart bbox
                [110.0, 250.0, 180.5, 320.8],  # yolk bbox
                [125.0, 295.0, 145.0, 310.0],  # swim bladder bbox
                [100.0, 180.0, 200.0, 450.0],  # trunk bbox
            ]),
            'scores': np.array([0.985, 0.923, 0.978, 0.856, 0.912]),
            'pred_classes': np.array([1, 2, 3, 4, 7]),  # eye, heart, yolk, swim_bladder, trunk
            # pred_masks would be (5, 512, 512) boolean arrays - too large to show
            'pred_masks_shape': (5, 512, 512),
        }
    }

    print("outputs['instances']:")
    print(f"  Number of detections: {simulated_output['instances']['num_instances']}")
    print(f"  Image size: {simulated_output['instances']['image_height']}x{simulated_output['instances']['image_width']}")
    print()

    print("  pred_boxes (bounding boxes):")
    print(f"    Shape: {simulated_output['instances']['pred_boxes'].shape}")
    print("    Format: [x1, y1, x2, y2] for each detection")
    for i, bbox in enumerate(simulated_output['instances']['pred_boxes']):
        print(f"      Detection {i}: {bbox}")
    print()

    print("  scores (confidence):")
    print(f"    Shape: {simulated_output['instances']['scores'].shape}")
    for i, score in enumerate(simulated_output['instances']['scores']):
        print(f"      Detection {i}: {score:.3f}")
    print()

    print("  pred_classes (class IDs):")
    print(f"    Shape: {simulated_output['instances']['pred_classes'].shape}")
    class_names = {1: 'eye', 2: 'heart', 3: 'yolk', 4: 'swim_bladder', 7: 'trunk'}
    for i, class_id in enumerate(simulated_output['instances']['pred_classes']):
        print(f"      Detection {i}: {class_id} ({class_names[class_id]})")
    print()

    print("  pred_masks (segmentation masks):")
    print(f"    Shape: {simulated_output['instances']['pred_masks_shape']}")
    print("    Format: Binary masks (0/1) for each detection")
    print("    Example for detection 0 (eye):")
    print("      Simulated mask area: 1250 pixels")
    print()


# ============================================================================
# EXAMPLE 2: Extracting Data from Detectron2 Output
# ============================================================================

def show_data_extraction():
    """
    Shows how to extract usable data from Detectron2 output.
    """
    print("=" * 80)
    print("EXAMPLE 2: Extracting Data from Detectron2 Output")
    print("=" * 80)

    # Simulated data
    masks = np.array([
        np.random.rand(512, 512) > 0.98,  # eye mask (small)
        np.random.rand(512, 512) > 0.985,  # heart mask (small)
        np.random.rand(512, 512) > 0.95,   # yolk mask (larger)
    ])
    classes = np.array([1, 2, 3])
    scores = np.array([0.985, 0.923, 0.978])
    boxes = np.array([
        [120.5, 180.3, 145.2, 210.8],
        [115.0, 215.5, 135.0, 240.2],
        [110.0, 250.0, 180.5, 320.8],
    ])

    print("\nCode to extract data:")
    print("""
    # Move to CPU and convert to numpy
    instances = outputs['instances'].to('cpu')

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    masks = instances.pred_masks.numpy()
    """)

    print("\nExtracted data:")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Classes shape: {classes.shape}")
    print(f"  Masks shape: {masks.shape}")
    print()

    print("Per-detection information:")
    class_names = {1: 'eye', 2: 'heart', 3: 'yolk'}
    for i in range(len(classes)):
        area = masks[i].sum()
        print(f"\n  Detection {i+1}:")
        print(f"    Class: {class_names[classes[i]]} (ID: {classes[i]})")
        print(f"    Confidence: {scores[i]:.3f}")
        print(f"    Bounding box: [{boxes[i][0]:.1f}, {boxes[i][1]:.1f}, {boxes[i][2]:.1f}, {boxes[i][3]:.1f}]")
        print(f"    Mask area: {area} pixels")
    print()


# ============================================================================
# EXAMPLE 3: Converting to CSV Format
# ============================================================================

def show_csv_conversion():
    """
    Demonstrates converting Detectron2 output to CSV format.
    """
    print("=" * 80)
    print("EXAMPLE 3: Converting to CSV Format (fishutil/fishclass style)")
    print("=" * 80)

    # Simulated detection data
    data = {
        'image_id': ['zebrafish_001'] * 5,
        'detection_id': [0, 1, 2, 3, 4],
        'class_id': [1, 2, 3, 4, 7],
        'class_name': ['eye', 'heart', 'yolk', 'swim_bladder', 'trunk'],
        'confidence': [0.985, 0.923, 0.978, 0.856, 0.912],
        'bbox_x1': [120.5, 115.0, 110.0, 125.0, 100.0],
        'bbox_y1': [180.3, 215.5, 250.0, 295.0, 180.0],
        'bbox_x2': [145.2, 135.0, 180.5, 145.0, 200.0],
        'bbox_y2': [210.8, 240.2, 320.8, 310.0, 450.0],
        'area_px': [1250, 890, 5600, 425, 12750],
        'perimeter_px': [142.5, 98.2, 280.3, 78.5, 520.8],
        'centroid_x': [132.8, 125.0, 145.2, 135.0, 150.0],
        'centroid_y': [195.5, 227.8, 285.4, 302.5, 315.0],
        'length': [24.7, 20.0, 70.5, 20.0, 100.0],
        'width': [20.5, 18.5, 65.2, 15.0, 50.0],
        'aspect_ratio': [1.20, 1.08, 1.08, 1.33, 2.00],
    }

    df = pd.DataFrame(data)

    print("\nCSV Format (one row per detection):")
    print(df.to_string(index=False))
    print()

    print("Summary statistics:")
    print(f"  Total detections: {len(df)}")
    print(f"  Unique classes: {df['class_name'].nunique()}")
    print(f"  Total area: {df['area_px'].sum()} pixels")
    print()

    print("Area by organ:")
    organ_summary = df.groupby('class_name')['area_px'].sum().sort_values(ascending=False)
    for organ, area in organ_summary.items():
        pct = (area / df['area_px'].sum()) * 100
        print(f"  {organ:20s}: {area:6d} pixels ({pct:5.1f}%)")
    print()


# ============================================================================
# EXAMPLE 4: morphseq Integration Format
# ============================================================================

def show_morphseq_format():
    """
    Demonstrates the morphseq integration output format.
    """
    print("=" * 80)
    print("EXAMPLE 4: morphseq Integration Format (Area Percentages)")
    print("=" * 80)

    # Example data for 3 zebrafish at different timepoints
    data = {
        'snip_id': ['zebrafish_001_t0', 'zebrafish_001_t1', 'zebrafish_002_t0'],
        'eye_area_pct': [2.50, 2.55, 2.60],
        'heart_area_pct': [1.78, 1.80, 1.85],
        'yolk_area_pct': [11.20, 11.00, 11.50],
        'swim_bladder_area_pct': [0.85, 0.88, 0.90],
        'otolith_area_pct': [0.42, 0.45, 0.48],
        'gut_area_pct': [3.20, 3.25, 3.30],
        'trunk_area_pct': [25.50, 25.80, 26.00],
        'tail_area_pct': [45.30, 45.10, 44.80],
        'pericardial_edema_area_pct': [2.10, 0.00, 0.00],
        'yolk_sac_edema_area_pct': [0.00, 0.00, 0.00],
        'spinal_curvature_area_pct': [0.00, 0.00, 0.00],
        'tail_malformation_area_pct': [0.00, 0.00, 0.00],
        'craniofacial_malformation_area_pct': [0.00, 0.00, 0.00],
        'reduced_pigmentation_area_pct': [0.00, 0.00, 0.00],
        'general_edema_area_pct': [0.00, 0.00, 0.00],
        'hemorrhage_area_pct': [0.00, 0.00, 0.00],
        'total_embryo_area_px': [50000, 51200, 50500],
        'total_embryo_area_um2': [125000.0, 128000.0, 126250.0],
    }

    df = pd.DataFrame(data)

    print("\nmorphseq Format (one row per snip_id, all classes present):")
    print()

    # Show first few columns
    display_cols = ['snip_id', 'eye_area_pct', 'heart_area_pct', 'yolk_area_pct',
                    'pericardial_edema_area_pct', 'total_embryo_area_px']
    print("First few columns:")
    print(df[display_cols].to_string(index=False))
    print()

    print("Key features of this format:")
    print("  ✓ One row per snip_id (image/timepoint)")
    print("  ✓ All 16 classes present (even if 0)")
    print("  ✓ Values are percentages of total embryo area")
    print("  ✓ Includes total area in pixels and micrometers")
    print("  ✓ Multiple instances of same class are aggregated")
    print()

    print("Example analysis:")
    print(f"  Zebrafish 001 at t0:")
    print(f"    - Total area: {df.loc[0, 'total_embryo_area_px']} pixels = {df.loc[0, 'total_embryo_area_um2']:.0f} μm²")
    print(f"    - Yolk: {df.loc[0, 'yolk_area_pct']:.2f}% of total")
    print(f"    - Yolk absolute area: {df.loc[0, 'yolk_area_pct'] / 100 * df.loc[0, 'total_embryo_area_um2']:.0f} μm²")
    print(f"    - Has pericardial edema: {df.loc[0, 'pericardial_edema_area_pct'] > 0}")
    print()


# ============================================================================
# EXAMPLE 5: Format Comparison Side-by-Side
# ============================================================================

def show_format_comparison():
    """
    Shows the same detection results in all three formats.
    """
    print("=" * 80)
    print("EXAMPLE 5: Same Data in Different Formats")
    print("=" * 80)

    print("\nScenario: Zebrafish with 3 detections (eye, heart, yolk)")
    print()

    # -------------------------
    # Format 1: Detectron2
    # -------------------------
    print("FORMAT 1: Detectron2 (Python dict)")
    print("-" * 80)
    print("""
    outputs = {
        'instances': {
            'pred_classes': [1, 2, 3],              # eye, heart, yolk
            'scores': [0.985, 0.923, 0.978],
            'pred_boxes': [
                [120.5, 180.3, 145.2, 210.8],      # eye bbox
                [115.0, 215.5, 135.0, 240.2],      # heart bbox
                [110.0, 250.0, 180.5, 320.8],      # yolk bbox
            ],
            'pred_masks': (3, 512, 512),            # 3 binary masks
        }
    }

    # Mask areas (after counting pixels):
    # Eye: 1250 pixels
    # Heart: 890 pixels
    # Yolk: 5600 pixels
    """)

    # -------------------------
    # Format 2: CSV
    # -------------------------
    print("\nFORMAT 2: CSV (tabular)")
    print("-" * 80)

    csv_data = pd.DataFrame({
        'image_id': ['zebrafish_001', 'zebrafish_001', 'zebrafish_001'],
        'class_name': ['eye', 'heart', 'yolk'],
        'confidence': [0.985, 0.923, 0.978],
        'area_px': [1250, 890, 5600],
        'bbox_x1': [120.5, 115.0, 110.0],
        'bbox_y1': [180.3, 215.5, 250.0],
    })
    print(csv_data.to_string(index=False))
    print()

    # -------------------------
    # Format 3: morphseq
    # -------------------------
    print("\nFORMAT 3: morphseq (aggregated, percentage-based)")
    print("-" * 80)

    total_area = 50000  # pixels
    morphseq_data = pd.DataFrame({
        'snip_id': ['zebrafish_001'],
        'eye_area_pct': [1250 / total_area * 100],
        'heart_area_pct': [890 / total_area * 100],
        'yolk_area_pct': [5600 / total_area * 100],
        'swim_bladder_area_pct': [0.0],
        'total_embryo_area_px': [total_area],
    })
    print(morphseq_data.to_string(index=False))
    print()

    print("Comparison:")
    print("  Detectron2:  Rich, includes masks, large files, best for visualization")
    print("  CSV:         Tabular, instance-level, medium files, best for analysis")
    print("  morphseq:    Aggregated, percentage-based, small files, best for pipelines")
    print()


# ============================================================================
# EXAMPLE 6: Working with Actual Data
# ============================================================================

def show_practical_usage():
    """
    Shows practical code examples for working with each format.
    """
    print("=" * 80)
    print("EXAMPLE 6: Practical Usage Examples")
    print("=" * 80)

    print("\n1. Loading Detectron2 pickle file:")
    print("-" * 80)
    print("""
    import pickle

    with open('zebrafish_001_dlma.pkl', 'rb') as f:
        outputs = pickle.load(f)

    instances = outputs['instances'].to('cpu')
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    # Calculate areas
    for i, (mask, class_id) in enumerate(zip(masks, classes)):
        area = mask.sum()
        print(f"Detection {i}: Class {class_id}, Area: {area} pixels")
    """)

    print("\n2. Loading CSV file:")
    print("-" * 80)
    print("""
    import pandas as pd

    df = pd.read_csv('dlma_results.csv')

    # Get all eyes
    eyes = df[df['class_name'] == 'eye']
    mean_eye_area = eyes['area_px'].mean()

    # Find abnormal embryos
    abnormal = df[df['class_id'] >= 9]  # Phenotype classes
    """)

    print("\n3. Loading morphseq format:")
    print("-" * 80)
    print("""
    import pandas as pd

    features = pd.read_csv('morphometric_features.csv')

    # Calculate yolk area in micrometers
    features['yolk_um2'] = (
        features['yolk_area_pct'] / 100 *
        features['total_embryo_area_um2']
    )

    # Find embryos with phenotypes
    phenotype_cols = [col for col in features.columns
                      if 'edema' in col or 'malformation' in col]
    features['has_phenotype'] = features[phenotype_cols].gt(0).any(axis=1)
    """)

    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    show_detectron2_output_structure()
    print("\n" * 2)

    show_data_extraction()
    print("\n" * 2)

    show_csv_conversion()
    print("\n" * 2)

    show_morphseq_format()
    print("\n" * 2)

    show_format_comparison()
    print("\n" * 2)

    show_practical_usage()

    print("=" * 80)
    print("END OF EXAMPLES")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - DLMA_OUTPUT_FORMATS.md")
    print("  - DLMA_INSTALLATION.md")
    print("  - README_DLMA_INTEGRATION.md")


if __name__ == "__main__":
    main()
