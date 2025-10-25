"""
Diagnose Mask Issues for Failed Geodesic/Curvature Analysis

Investigates specific embryos to identify morphological problems:
- Holes in masks
- Disconnected components
- Spindly appendages
- Skeleton quality
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


# Import mask cleaning function
def clean_embryo_mask(mask: np.ndarray, verbose: bool = False):
    """
    Clean embryo mask to remove artifacts before centerline extraction.

    Cleaning steps:
    1. Fill holes
    2. Morphological closing to connect nearby parts
    3. Keep largest connected component only
    4. Adaptive morphological opening to remove spindly protrusions
    """
    original_area = mask.sum()

    # Count initial components
    initial_labeled = measure.label(mask)
    n_components_initial = initial_labeled.max()

    # Step 1: Fill holes
    filled = ndimage.binary_fill_holes(mask)
    holes_filled = filled.sum() - mask.sum()

    # Step 2: Morphological CLOSING to connect nearby parts
    closing_radius = 5  # Connect parts within ~5 pixels
    selem_close = morphology.disk(closing_radius)
    closed = morphology.binary_closing(filled, selem_close)

    # Check if closing connected components
    closed_labeled = measure.label(closed)
    n_components_after_closing = closed_labeled.max()

    # Step 3: Keep largest connected component
    labeled = measure.label(closed)
    if labeled.max() > 1:
        component_sizes = [(i, np.sum(labeled == i)) for i in range(1, labeled.max() + 1)]
        largest_label = max(component_sizes, key=lambda x: x[1])[0]
        closed = (labeled == largest_label)
        n_components_removed = labeled.max() - 1
    else:
        n_components_removed = 0

    # Step 4: Adaptive morphological opening
    props = measure.regionprops(measure.label(closed))[0]
    perimeter = props.perimeter
    adaptive_radius = max(3, int(perimeter / 100))
    solidity_before = props.solidity

    selem_open = morphology.disk(adaptive_radius)
    cleaned = morphology.binary_opening(closed, selem_open)

    # Get final stats
    final_props = measure.regionprops(measure.label(cleaned))[0]
    solidity_after = final_props.solidity
    area_after = cleaned.sum()

    cleaning_stats = {
        'original_area': original_area,
        'holes_filled': holes_filled,
        'n_components_initial': n_components_initial,
        'n_components_after_closing': n_components_after_closing,
        'n_components_removed': n_components_removed,
        'closing_radius': closing_radius,
        'adaptive_radius': adaptive_radius,
        'solidity_before': solidity_before,
        'solidity_after': solidity_after,
        'area_after': area_after,
        'area_removed': original_area - area_after,
        'area_removed_pct': (original_area - area_after) / original_area * 100
    }

    return cleaned, cleaning_stats


def analyze_mask_quality(mask: np.ndarray, snip_id: str):
    """
    Analyze mask for morphological issues.

    Returns dict with diagnostics.
    """
    results = {
        'snip_id': snip_id,
        'total_area': mask.sum(),
        'shape': mask.shape
    }

    # 1. Check for holes
    filled = ndimage.binary_fill_holes(mask)
    hole_area = filled.sum() - mask.sum()
    results['hole_area'] = hole_area
    results['hole_fraction'] = hole_area / mask.sum() if mask.sum() > 0 else 0

    # 2. Check connected components
    labeled = measure.label(mask)
    n_components = labeled.max()
    results['n_components'] = n_components

    if n_components > 1:
        component_sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
        results['largest_component_fraction'] = max(component_sizes) / mask.sum()
        results['component_sizes'] = component_sizes
    else:
        results['largest_component_fraction'] = 1.0
        results['component_sizes'] = [mask.sum()]

    # 3. Check skeleton quality
    try:
        skeleton = morphology.skeletonize(mask)
        results['skeleton_length'] = skeleton.sum()

        # Count branch points (pixels with >2 neighbors)
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
        branch_points = np.sum((skeleton > 0) & (neighbor_count > 2))
        results['branch_points'] = branch_points

        # Count endpoints (pixels with 1 neighbor)
        endpoints = np.sum((skeleton > 0) & (neighbor_count == 1))
        results['endpoints'] = endpoints

        results['skeleton_success'] = True
    except Exception as e:
        results['skeleton_success'] = False
        results['skeleton_error'] = str(e)

    # 4. Morphological metrics
    props = measure.regionprops(measure.label(mask))[0]
    results['eccentricity'] = props.eccentricity
    results['solidity'] = props.solidity
    results['extent'] = props.extent

    return results


def visualize_mask_diagnostics(mask: np.ndarray, snip_id: str, diagnostics: dict, ax=None):
    """
    Create diagnostic visualization showing:
    - Original mask
    - Filled mask (highlighting holes)
    - Skeleton
    - Connected components
    """
    if ax is None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        axes = ax

    # Original mask
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(f'{snip_id[:30]}\nOriginal Mask\nArea: {diagnostics["total_area"]:,} px',
                         fontsize=9)
    axes[0].axis('off')

    # Filled mask (holes highlighted)
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask

    # Create RGB image to show holes in red
    display_img = np.zeros((*mask.shape, 3))
    display_img[mask > 0] = [1, 1, 1]  # White for mask
    display_img[holes > 0] = [1, 0, 0]  # Red for holes

    axes[1].imshow(display_img)
    axes[1].set_title(f'Holes (red)\nHole area: {diagnostics["hole_area"]:,} px\n' +
                         f'Hole fraction: {diagnostics["hole_fraction"]:.2%}',
                         fontsize=9)
    axes[1].axis('off')

    # Skeleton with branch points and endpoints
    if diagnostics['skeleton_success']:
        skeleton = morphology.skeletonize(mask)

        # Find branch points and endpoints
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
        branch_points = (skeleton > 0) & (neighbor_count > 2)
        endpoints = (skeleton > 0) & (neighbor_count == 1)

        # Create RGB visualization
        skel_img = np.zeros((*mask.shape, 3))
        skel_img[mask > 0] = [0.3, 0.3, 0.3]  # Gray background
        skel_img[skeleton > 0] = [1, 1, 1]  # White skeleton
        skel_img[branch_points] = [1, 1, 0]  # Yellow branch points
        skel_img[endpoints] = [0, 1, 0]  # Green endpoints

        axes[2].imshow(skel_img)
        axes[2].set_title(f'Skeleton\nLength: {diagnostics["skeleton_length"]} px\n' +
                            f'Branch pts: {diagnostics["branch_points"]}, ' +
                            f'Endpoints: {diagnostics["endpoints"]}',
                            fontsize=9)
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, f'Skeleton Failed\n{diagnostics.get("skeleton_error", "")}',
                       ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')

    # Connected components
    labeled = measure.label(mask)
    n_components = labeled.max()

    if n_components == 1:
        axes[3].imshow(labeled, cmap='gray')
        axes[3].set_title(f'Connected Components: {n_components}\n(Single component - Good!)',
                            fontsize=9)
    else:
        # Color each component differently
        axes[3].imshow(labeled, cmap='tab20')
        component_sizes = diagnostics['component_sizes']
        title = f'Connected Components: {n_components}\n'
        title += f'Largest: {max(component_sizes):,} px ({diagnostics["largest_component_fraction"]:.1%})\n'
        title += f'Sizes: {component_sizes[:5]}'  # Show first 5
        axes[3].set_title(title, fontsize=9)
    axes[3].axis('off')


def main():
    """Main diagnostic execution."""
    # Embryos to diagnose
    problem_embryos = [
        '20251017_part2_B05_e01_t0037',  # User mentioned - might have spindly parts/holes
        '20251017_part2_G07_e01_t0013',  # User asked about this one
        '20251017_part2_B04_e01_t0039',  # User just mentioned - geodesic spline didn't work
    ]

    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part2.csv")
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251024")

    print("="*80)
    print("MASK QUALITY DIAGNOSTICS")
    print("="*80)

    # Load data
    print(f"\nLoading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Process each problem embryo
    diagnostics_list = []

    for snip_id in problem_embryos:
        print(f"\n{'='*80}")
        print(f"Analyzing: {snip_id}")
        print('='*80)

        # Find in dataframe
        row = df[df['snip_id'] == snip_id]
        if len(row) == 0:
            print(f"  ERROR: {snip_id} not found in CSV!")
            continue

        row = row.iloc[0]

        # Decode mask
        mask_rle = row['mask_rle']
        mask = decode_mask_rle({
            'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
            'counts': mask_rle
        })
        mask = np.ascontiguousarray(mask.astype(np.uint8))

        # Analyze BEFORE cleaning
        print(f"\n  BEFORE CLEANING:")
        diagnostics_before = analyze_mask_quality(mask, snip_id + "_before")
        print(f"    Total Area: {diagnostics_before['total_area']:,} px")
        print(f"    Hole Area: {diagnostics_before['hole_area']:,} px ({diagnostics_before['hole_fraction']:.2%})")
        print(f"    Connected Components: {diagnostics_before['n_components']}")
        if diagnostics_before['skeleton_success']:
            print(f"    Branch Points: {diagnostics_before['branch_points']}, Endpoints: {diagnostics_before['endpoints']}")
        print(f"    Solidity: {diagnostics_before['solidity']:.3f}")

        # CLEAN MASK
        print(f"\n  CLEANING MASK...")
        mask_cleaned, cleaning_stats = clean_embryo_mask(mask, verbose=False)
        print(f"    Holes filled: {cleaning_stats['holes_filled']} px")
        print(f"    Components removed: {cleaning_stats['n_components_removed']}")
        print(f"    Adaptive radius: {cleaning_stats['adaptive_radius']} px")
        print(f"    Area removed: {cleaning_stats['area_removed']:,} px ({cleaning_stats['area_removed_pct']:.1f}%)")

        # Analyze AFTER cleaning
        print(f"\n  AFTER CLEANING:")
        diagnostics_after = analyze_mask_quality(mask_cleaned, snip_id + "_after")
        print(f"    Total Area: {diagnostics_after['total_area']:,} px")
        print(f"    Connected Components: {diagnostics_after['n_components']}")
        if diagnostics_after['skeleton_success']:
            print(f"    Branch Points: {diagnostics_after['branch_points']}, Endpoints: {diagnostics_after['endpoints']}")
        print(f"    Solidity: {diagnostics_after['solidity']:.3f}")

        # Print improvement
        if diagnostics_before['skeleton_success'] and diagnostics_after['skeleton_success']:
            branch_reduction = diagnostics_before['branch_points'] - diagnostics_after['branch_points']
            print(f"\n  IMPROVEMENT:")
            print(f"    Branch points: {diagnostics_before['branch_points']} → {diagnostics_after['branch_points']} ({branch_reduction} removed)")
            print(f"    Solidity: {diagnostics_before['solidity']:.3f} → {diagnostics_after['solidity']:.3f}")

        # Store both for visualization
        diagnostics_before['mask'] = mask
        diagnostics_after['mask'] = mask_cleaned
        diagnostics_after['mask_original'] = mask
        diagnostics_list.append(diagnostics_after)

    # Create visualization
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print('='*80)

    n_valid = len([d for d in diagnostics_list if 'mask' in d])
    if n_valid == 0:
        print("No valid masks to visualize!")
        return

    fig, axes = plt.subplots(n_valid, 4, figsize=(16, 4*n_valid))

    # Handle different array shapes
    if n_valid == 1:
        axes = axes.reshape(1, -1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    valid_idx = 0
    for i, diag in enumerate(diagnostics_list):
        if 'mask' in diag:
            visualize_mask_diagnostics(diag['mask'], diag['snip_id'], diag, ax=axes[valid_idx])
            valid_idx += 1

    plt.tight_layout()
    output_path = output_dir / "mask_diagnostics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Save diagnostics to CSV
    diag_df = pd.DataFrame([{k: v for k, v in d.items() if k != 'mask'}
                            for d in diagnostics_list])
    diag_csv = output_dir / "mask_diagnostics.csv"
    diag_df.to_csv(diag_csv, index=False)
    print(f"Saved: {diag_csv}")

    print(f"\n{'='*80}")
    print("DIAGNOSTICS COMPLETE")
    print('='*80)


if __name__ == "__main__":
    main()
