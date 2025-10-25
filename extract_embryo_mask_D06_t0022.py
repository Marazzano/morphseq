"""
Extract and analyze mask for specific embryo: 20251017_part2_D06_ch00_t0022

This script extracts the mask for the specified embryo image and applies
the curvature analysis methods from embryo_curvature_analysis.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import UnivariateSpline, splprep, splev
from scipy.ndimage import distance_transform_edt
from skimage import io, morphology, measure
from skimage.filters import gaussian
from sklearn.decomposition import PCA
import cv2
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import the curvature analyzer
import sys
sys.path.append('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022')
from embryo_curvature_analysis import EmbryoCurvatureAnalyzer


def main():
    # Define paths
    image_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/raw_data_organized/20251017_part2/images/20251017_part2_D06/20251017_part2_D06_ch00_t0022.jpg")
    mask_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/exported_masks/20251017_part2/masks/20251017_part2_D06_ch00_t0022_masks_emnum_1.png")
    
    print(f"Image path: {image_path}")
    print(f"Image exists: {image_path.exists()}")
    print(f"\nMask path: {mask_path}")
    print(f"Mask exists: {mask_path.exists()}")
    
    if not mask_path.exists():
        print("\nERROR: Mask file not found!")
        return
    
    if not image_path.exists():
        print("\nERROR: Image file not found!")
        return
    
    # Load the original image for visualization
    print("\nLoading image...")
    image = io.imread(image_path)
    print(f"Image shape: {image.shape}")
    
    # Load and analyze the mask
    print("\nLoading and analyzing mask...")
    analyzer = EmbryoCurvatureAnalyzer(mask_path)
    print(f"Mask shape: {analyzer.mask.shape}")
    print(f"Mask sum (area): {analyzer.mask.sum()} pixels")
    
    # Create output directory
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251024")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save the raw mask
    mask_output_path = output_dir / "embryo_D06_t0022_mask.png"
    io.imsave(mask_output_path, (analyzer.mask * 255).astype(np.uint8))
    print(f"\nSaved mask to: {mask_output_path}")
    
    # Save mask as numpy array
    mask_npy_path = output_dir / "embryo_D06_t0022_mask.npy"
    np.save(mask_npy_path, analyzer.mask)
    print(f"Saved mask array to: {mask_npy_path}")
    
    # Visualize the image with mask overlay
    print("\nCreating image + mask visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(analyzer.mask, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(analyzer.mask, cmap='Reds', alpha=0.4)
    axes[2].set_title('Image + Mask Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    overlay_path = output_dir / "embryo_D06_t0022_image_mask_overlay.png"
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    print(f"Saved overlay to: {overlay_path}")
    plt.close()
    
    # Run curvature analysis with all 4 methods
    print("\n" + "="*60)
    print("Running Curvature Analysis")
    print("="*60)
    
    print("\nMethod 1: Skeletonization...")
    skeleton, points1 = analyzer.method1_skeletonize(visualize=False)
    
    print("Method 2: Distance Transform Ridge...")
    dist_map, points2 = analyzer.method2_distance_transform_ridge(visualize=False)
    
    print("Method 3: PCA-Based Slicing...")
    axis, points3 = analyzer.method3_pca_slicing(visualize=False)
    
    print("Method 4: Contour-Based...")
    contours, points4 = analyzer.method4_contour_based(visualize=False)
    
    # Compare all methods
    print("\nCreating comparison figure...")
    fig = analyzer.compare_all_methods(
        save_path=output_dir / "embryo_D06_t0022_curvature_comparison.png"
    )
    plt.close()
    
    # Create detailed visualization with image
    print("\nCreating detailed analysis with original image...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Image with different centerline methods
    methods_data = [
        ("Skeletonization", points1),
        ("Distance Ridge", points2),
        ("PCA Slicing", points3),
    ]
    
    for idx, (name, points) in enumerate(methods_data):
        axes[0, idx].imshow(image, cmap='gray')
        axes[0, idx].imshow(analyzer.mask, cmap='Reds', alpha=0.3)
        
        if len(points) > 0:
            axes[0, idx].plot(points[:, 0], points[:, 1], 'cyan', linewidth=2, label='Centerline')
            axes[0, idx].scatter(points[0, 0], points[0, 1], c='lime', s=100, marker='o', label='Start', zorder=5)
            axes[0, idx].scatter(points[-1, 0], points[-1, 1], c='blue', s=100, marker='s', label='End', zorder=5)
        
        axes[0, idx].set_title(f'{name}\nCenterline Extraction')
        axes[0, idx].legend(fontsize=8)
        axes[0, idx].axis('off')
    
    # Bottom row: Curvature plots
    for idx, (name, points) in enumerate(methods_data):
        if len(points) > 3:
            arc_length, curvature = analyzer.compute_curvature(points)
            if len(arc_length) > 0:
                axes[1, idx].plot(arc_length, curvature, 'b-', linewidth=2)
                axes[1, idx].set_xlabel('Arc Length (pixels)', fontsize=10)
                axes[1, idx].set_ylabel('Curvature (1/pixels)', fontsize=10)
                axes[1, idx].set_title(f'{name}\nCurvature Profile', fontsize=10)
                axes[1, idx].grid(True, alpha=0.3)
                
                # Add statistics
                mean_curv = np.mean(curvature)
                max_curv = np.max(curvature)
                axes[1, idx].text(0.05, 0.95, f'Mean: {mean_curv:.4f}\nMax: {max_curv:.4f}',
                                transform=axes[1, idx].transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                fontsize=8)
    
    plt.suptitle('Embryo 20251017_part2_D06 at t=0022 - Curvature Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    detailed_path = output_dir / "embryo_D06_t0022_detailed_analysis.png"
    plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed analysis to: {detailed_path}")
    plt.close()
    
    # Save centerline data
    print("\nSaving centerline data...")
    for name, points in [("skeleton", points1), ("ridge", points2), ("pca", points3), ("contour", points4)]:
        if len(points) > 0:
            centerline_path = output_dir / f"embryo_D06_t0022_centerline_{name}.npy"
            np.save(centerline_path, points)
            print(f"  - Saved {name} centerline to: {centerline_path}")
            
            # Also save curvature data
            arc_length, curvature = analyzer.compute_curvature(points)
            if len(arc_length) > 0:
                curvature_data = np.column_stack([arc_length, curvature])
                curvature_path = output_dir / f"embryo_D06_t0022_curvature_{name}.csv"
                pd.DataFrame(curvature_data, columns=['arc_length', 'curvature']).to_csv(
                    curvature_path, index=False
                )
                print(f"  - Saved {name} curvature to: {curvature_path}")
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nAll files saved to: {output_dir}")
    print("\nFiles created:")
    print(f"  1. Mask PNG: embryo_D06_t0022_mask.png")
    print(f"  2. Mask NPY: embryo_D06_t0022_mask.npy")
    print(f"  3. Image+Mask overlay: embryo_D06_t0022_image_mask_overlay.png")
    print(f"  4. Curvature comparison: embryo_D06_t0022_curvature_comparison.png")
    print(f"  5. Detailed analysis: embryo_D06_t0022_detailed_analysis.png")
    print(f"  6. Centerline data (4 methods): embryo_D06_t0022_centerline_*.npy")
    print(f"  7. Curvature profiles (4 methods): embryo_D06_t0022_curvature_*.csv")


if __name__ == "__main__":
    main()
