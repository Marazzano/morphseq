"""
Compare PCA vs Geodesic Spline Extraction Methods

Analyzes when PCA and Geodesic methods produce different centerlines/splines,
and what morphological characteristics predict these differences.

Key features:
1. Dummy smart cleaning function (placeholder for future implementation)
2. Computes both PCA and Geodesic splines
3. Measures spline differences (mean distance, max distance, Hausdorff)
4. Records morphology metrics for decision rule analysis
5. Outputs CSV for automatic QC detection
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
from tqdm import tqdm
import multiprocessing as mp
import os
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle

# Import existing analyzer classes
sys.path.insert(0, str(Path(__file__).parent.parent / "20251022"))
from geodesic_bspline_smoothing import GeodesicBSplineAnalyzer
from test_pca_smoothing import PCACurvatureAnalyzer


def smart_mask_cleaning(mask: np.ndarray, verbose: bool = False):
    """
    DUMMY FUNCTION: Smart mask cleaning based on morphology metrics.

    TODO: Replace with actual implementation once thresholds are determined
    from morphology distribution analysis.

    Current behavior: Returns original mask unchanged.

    Future behavior:
    - Compute morphology metrics (solidity, circularity, etc.)
    - Decide if opening is needed based on thresholds
    - Apply appropriate cleaning operations
    - Return cleaned mask

    Args:
        mask: Binary mask array
        verbose: Print debug info

    Returns:
        cleaned_mask: Cleaned binary mask (currently just returns input)
        cleaning_applied: Boolean indicating if cleaning was applied (currently False)
    """
    if verbose:
        print("  [DUMMY] smart_mask_cleaning: No cleaning applied (placeholder)")

    # TODO: Implement smart cleaning logic
    # Example structure:
    # props = measure.regionprops(measure.label(mask))[0]
    # needs_opening = (props.solidity < 0.85) or (circularity > 10)
    # if needs_opening:
    #     mask = apply_opening(mask)
    #     return mask, True

    return mask.copy(), False


def compute_morphology_metrics(mask: np.ndarray):
    """
    Compute morphological metrics for a mask.

    Returns dict with metrics used for decision rules.
    """
    props = measure.regionprops(measure.label(mask))[0]

    metrics = {
        'area': props.area,
        'perimeter': props.perimeter,
        'solidity': props.solidity,
        'extent': props.extent,
        'eccentricity': props.eccentricity,
        'circularity': (props.perimeter ** 2) / (4 * np.pi * props.area),
        'perimeter_area_ratio': props.perimeter / np.sqrt(props.area),
    }

    return metrics


def compute_spline_differences(spline1_x, spline1_y, spline2_x, spline2_y):
    """
    Compute differences between two splines.

    Args:
        spline1_x, spline1_y: First spline coordinates
        spline2_x, spline2_y: Second spline coordinates

    Returns:
        dict with difference metrics:
        - mean_distance: Average Euclidean distance
        - max_distance: Maximum distance
        - hausdorff_distance: Hausdorff distance (symmetric)
    """
    if len(spline1_x) == 0 or len(spline2_x) == 0:
        return {
            'mean_distance': np.nan,
            'max_distance': np.nan,
            'hausdorff_distance': np.nan,
        }

    # Stack into (N, 2) arrays
    spline1 = np.column_stack([spline1_x, spline1_y])
    spline2 = np.column_stack([spline2_x, spline2_y])

    # Resample both splines to same number of points for point-wise comparison
    n_points = min(len(spline1), len(spline2), 100)

    # Resample spline1
    u1 = np.linspace(0, 1, len(spline1))
    u1_new = np.linspace(0, 1, n_points)
    spline1_resampled_x = np.interp(u1_new, u1, spline1[:, 0])
    spline1_resampled_y = np.interp(u1_new, u1, spline1[:, 1])
    spline1_resampled = np.column_stack([spline1_resampled_x, spline1_resampled_y])

    # Resample spline2
    u2 = np.linspace(0, 1, len(spline2))
    u2_new = np.linspace(0, 1, n_points)
    spline2_resampled_x = np.interp(u2_new, u2, spline2[:, 0])
    spline2_resampled_y = np.interp(u2_new, u2, spline2[:, 1])
    spline2_resampled = np.column_stack([spline2_resampled_x, spline2_resampled_y])

    # Point-wise distances
    distances = np.linalg.norm(spline1_resampled - spline2_resampled, axis=1)

    # Hausdorff distance (symmetric)
    hausdorff_1to2 = directed_hausdorff(spline1, spline2)[0]
    hausdorff_2to1 = directed_hausdorff(spline2, spline1)[0]
    hausdorff = max(hausdorff_1to2, hausdorff_2to1)

    return {
        'mean_distance': float(np.mean(distances)),
        'max_distance': float(np.max(distances)),
        'hausdorff_distance': float(hausdorff),
    }


def compare_methods_single_embryo(mask: np.ndarray, snip_id: str, um_per_pixel: float = 1.0):
    """
    Compare PCA and Geodesic methods for a single embryo.

    Returns dict with all metrics, or None if either method fails.
    """
    result = {'snip_id': snip_id}

    try:
        # 1. Apply dummy smart cleaning
        cleaned_mask, cleaning_applied = smart_mask_cleaning(mask, verbose=False)
        result['cleaning_applied'] = cleaning_applied

        # 2. Compute morphology metrics
        morphology = compute_morphology_metrics(cleaned_mask)
        result.update(morphology)

        # 3. Extract PCA spline
        try:
            pca_analyzer = PCACurvatureAnalyzer(cleaned_mask, um_per_pixel=um_per_pixel)
            centerline_pca = pca_analyzer.extract_centerline_pca(n_slices=100)
            arc_pca, curv_pca, spline_pca_x, spline_pca_y = pca_analyzer.compute_curvature(
                centerline_pca, smoothing=5.0
            )
            result['pca_success'] = True
            result['pca_n_points'] = len(spline_pca_x)
        except Exception as e:
            result['pca_success'] = False
            result['pca_error'] = str(e)
            result['pca_n_points'] = 0
            spline_pca_x = np.array([])
            spline_pca_y = np.array([])

        # 4. Extract Geodesic spline
        try:
            geo_analyzer = GeodesicBSplineAnalyzer(
                cleaned_mask, um_per_pixel=um_per_pixel, bspline_smoothing=5.0
            )
            centerline_geo, endpoints_geo, skeleton_geo = geo_analyzer.extract_centerline()
            arc_geo, curv_geo, spline_geo_x, spline_geo_y = geo_analyzer.compute_curvature(centerline_geo)
            result['geodesic_success'] = True
            result['geodesic_n_points'] = len(spline_geo_x)
        except Exception as e:
            result['geodesic_success'] = False
            result['geodesic_error'] = str(e)
            result['geodesic_n_points'] = 0
            spline_geo_x = np.array([])
            spline_geo_y = np.array([])

        # 5. Compare splines if both succeeded
        if result['pca_success'] and result['geodesic_success']:
            differences = compute_spline_differences(
                spline_pca_x, spline_pca_y,
                spline_geo_x, spline_geo_y
            )
            result.update(differences)
            result['both_methods_succeeded'] = True
        else:
            result['both_methods_succeeded'] = False
            result['mean_distance'] = np.nan
            result['max_distance'] = np.nan
            result['hausdorff_distance'] = np.nan

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def process_single_row(row_data):
    """
    Process a single row for parallel execution.

    Args:
        row_data: Tuple of (snip_id, mask_rle, mask_height_px, mask_width_px)

    Returns:
        Result dict from compare_methods_single_embryo or None
    """
    snip_id, mask_rle, mask_height_px, mask_width_px = row_data

    try:
        # Decode mask
        mask = decode_mask_rle({
            'size': [int(mask_height_px), int(mask_width_px)],
            'counts': mask_rle
        })
        mask = np.ascontiguousarray(mask.astype(np.uint8))

        # Compare methods
        result = compare_methods_single_embryo(mask, snip_id, um_per_pixel=1.0)
        return result

    except Exception as e:
        return {'snip_id': snip_id, 'error': str(e)}


def analyze_dataset(csv_path: Path, n_samples: int = 100, random_seed: int = 42, n_jobs: int = None):
    """
    Analyze random sample from dataset with parallel processing.

    Args:
        csv_path: Path to CSV with mask data
        n_samples: Number of random embryos to analyze
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (default: n_cpus - 1)

    Returns:
        DataFrame with all results
    """
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    print(f"Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Random sample
    np.random.seed(random_seed)
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=random_seed)

    print(f"Comparing PCA vs Geodesic for {len(sample_df)} embryos...")
    print(f"Using {n_jobs} parallel workers (CPU count: {mp.cpu_count()})")

    # Prepare data for parallel processing
    row_data_list = [
        (row['snip_id'], row['mask_rle'], row['mask_height_px'], row['mask_width_px'])
        for _, row in sample_df.iterrows()
    ]

    # Process in parallel with progress bar
    with mp.Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_single_row, row_data_list),
            total=len(row_data_list),
            desc="Processing embryos"
        ))

    # Filter out None results
    results = [r for r in results if r is not None]

    results_df = pd.DataFrame(results)

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print('='*80)

    n_total = len(results_df)
    n_pca_success = results_df['pca_success'].sum()
    n_geo_success = results_df['geodesic_success'].sum()
    n_both_success = results_df['both_methods_succeeded'].sum()

    print(f"Total embryos analyzed: {n_total}")
    print(f"PCA succeeded: {n_pca_success}/{n_total} ({100*n_pca_success/n_total:.1f}%)")
    print(f"Geodesic succeeded: {n_geo_success}/{n_total} ({100*n_geo_success/n_total:.1f}%)")
    print(f"Both succeeded: {n_both_success}/{n_total} ({100*n_both_success/n_total:.1f}%)")

    if n_both_success > 0:
        both_df = results_df[results_df['both_methods_succeeded']]
        print(f"\nSpline Differences (n={len(both_df)}):")
        print(f"  Mean distance:")
        print(f"    Mean: {both_df['mean_distance'].mean():.2f} px")
        print(f"    Median: {both_df['mean_distance'].median():.2f} px")
        print(f"    Max: {both_df['mean_distance'].max():.2f} px")
        print(f"  Hausdorff distance:")
        print(f"    Mean: {both_df['hausdorff_distance'].mean():.2f} px")
        print(f"    Median: {both_df['hausdorff_distance'].median():.2f} px")
        print(f"    Max: {both_df['hausdorff_distance'].max():.2f} px")

    return results_df


def generate_decision_rules(results_df: pd.DataFrame, output_dir: Path):
    """
    Analyze correlations between morphology metrics and spline differences
    to generate decision rules for when to use Geodesic vs PCA.

    Args:
        results_df: DataFrame with comparison results
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*80}")
    print("DECISION RULE ANALYSIS")
    print('='*80)

    # Filter to embryos where both methods succeeded
    both_df = results_df[results_df['both_methods_succeeded']].copy()

    if len(both_df) == 0:
        print("No embryos where both methods succeeded. Cannot generate decision rules.")
        return

    # Define "large difference" threshold (e.g., mean distance > 10 px)
    large_diff_threshold = both_df['mean_distance'].quantile(0.75)  # Top 25% of differences
    both_df['large_difference'] = both_df['mean_distance'] > large_diff_threshold

    print(f"\nLarge difference threshold: mean_distance > {large_diff_threshold:.2f} px")
    print(f"Embryos with large differences: {both_df['large_difference'].sum()}/{len(both_df)}")

    # Compute correlations
    morphology_metrics = ['solidity', 'circularity', 'extent', 'eccentricity', 'perimeter_area_ratio']
    difference_metrics = ['mean_distance', 'max_distance', 'hausdorff_distance']

    print(f"\n{'='*80}")
    print("CORRELATIONS: Morphology vs Spline Differences")
    print('='*80)

    for diff_metric in difference_metrics:
        print(f"\n{diff_metric.upper()}:")
        for morph_metric in morphology_metrics:
            corr = both_df[morph_metric].corr(both_df[diff_metric])
            print(f"  {morph_metric:25s}: {corr:+.3f}")

    # Identify best predictors
    print(f"\n{'='*80}")
    print("BEST PREDICTORS OF LARGE DIFFERENCES")
    print('='*80)

    large_diff_df = both_df[both_df['large_difference']]
    small_diff_df = both_df[~both_df['large_difference']]

    print(f"\nMorphology Metric Comparison:")
    print(f"{'Metric':<25s} | {'Small Diff (n={})'.format(len(small_diff_df)):^20s} | {'Large Diff (n={})'.format(len(large_diff_df)):^20s}")
    print('-'*70)

    for metric in morphology_metrics:
        small_mean = small_diff_df[metric].mean()
        large_mean = large_diff_df[metric].mean()
        diff_pct = 100 * (large_mean - small_mean) / small_mean if small_mean != 0 else 0

        print(f"{metric:<25s} | {small_mean:^20.3f} | {large_mean:^20.3f} ({diff_pct:+.1f}%)")

    # Generate threshold recommendations
    print(f"\n{'='*80}")
    print("THRESHOLD RECOMMENDATIONS")
    print('='*80)
    print("\nSuggested thresholds for detecting when Geodesic is preferred over PCA:")

    for metric in morphology_metrics:
        # Find threshold that best separates large vs small differences
        threshold_candidates = np.percentile(both_df[metric], [10, 25, 50, 75, 90])

        best_threshold = None
        best_separation = 0

        for threshold in threshold_candidates:
            if metric in ['solidity', 'extent']:
                # Lower values indicate problems
                predicted_large = both_df[metric] < threshold
            else:
                # Higher values indicate problems
                predicted_large = both_df[metric] > threshold

            # Measure separation (difference in mean_distance)
            predicted_large_mean = both_df[predicted_large]['mean_distance'].mean() if predicted_large.sum() > 0 else 0
            predicted_small_mean = both_df[~predicted_large]['mean_distance'].mean() if (~predicted_large).sum() > 0 else 0
            separation = abs(predicted_large_mean - predicted_small_mean)

            if separation > best_separation:
                best_separation = separation
                best_threshold = threshold

        if best_threshold is not None:
            if metric in ['solidity', 'extent']:
                print(f"  {metric} < {best_threshold:.3f} → Consider Geodesic")
            else:
                print(f"  {metric} > {best_threshold:.3f} → Consider Geodesic")

    print(f"\n{'='*80}")


def main():
    """Main execution."""
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part1.csv")
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027")

    print("="*80)
    print("PCA VS GEODESIC SPLINE COMPARISON (PARALLELIZED)")
    print("="*80)

    # Analyze dataset with parallelization (n_jobs defaults to cpu_count - 1)
    results_df = analyze_dataset(csv_path, n_samples=100, random_seed=42, n_jobs=None)

    # Save results
    output_csv = output_dir / "pca_vs_geodesic_comparison.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # Generate decision rules
    generate_decision_rules(results_df, output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
