#!/usr/bin/env python3
"""
Debug SA Outlier Detection

Visualizes exactly what's happening with SA outlier flagging to understand
why seemingly normal embryos are being flagged.

Usage:
    python debug_sa_outlier.py --exp 20250529_36hpf_ctrl_atf6
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.build.build04_perform_embryo_qc import _sa_qc_with_fallback, _validate_and_prepare_inputs


def load_build03_data(root: Path, exp: str) -> pd.DataFrame:
    """Load Build03 CSV for the experiment."""
    build03_csv = root / "metadata" / "build03_output" / f"expr_embryo_metadata_{exp}.csv"
    if not build03_csv.exists():
        raise FileNotFoundError(f"Build03 CSV not found: {build03_csv}")

    df = pd.read_csv(build03_csv)
    print(f"📊 Loaded {len(df)} rows from {build03_csv}")
    return df


def prepare_data_for_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data same as Build04 does."""
    # Map columns to expected names
    if "area_um2" in df.columns:
        df["surface_area_um"] = df["area_um2"]
        print("✅ Mapped surface_area_um from area_um2")

    if "genotype" in df.columns:
        df["phenotype"] = df["genotype"]
        print("✅ Mapped phenotype from genotype")

    if "chem_perturbation" in df.columns:
        df["short_pert_name"] = df["chem_perturbation"]
        print("✅ Mapped short_pert_name from chem_perturbation")

    # Add default control_flag
    df["control_flag"] = True
    print("✅ Added default control_flag: True")

    # Validate and prepare inputs
    df = _validate_and_prepare_inputs(df)

    return df


def debug_sa_qc(df: pd.DataFrame, stage_ref_path: Path) -> tuple:
    """Run SA QC and extract debug information."""
    print("\n🔍 Running SA QC with debug info...")

    # Store original data
    orig_df = df.copy()

    # Run SA QC - this will modify df with sa_outlier_flag
    result_df = _sa_qc_with_fallback(
        df=df,
        stage_ref_path=stage_ref_path,
        sg_window=5,
        sg_poly=2,
        percentile=95.0,
        bin_step=0.5,
        hpf_window=0.75,
        min_embryos=2,
        margin_k=1.40,
        calibrate_scale=True,
    )

    # Extract debug info
    debug_info = {
        'total_embryos': len(result_df),
        'flagged_count': result_df['sa_outlier_flag'].sum(),
        'flagged_pct': 100.0 * result_df['sa_outlier_flag'].sum() / len(result_df),
    }

    # Identify reference embryos (controls)
    use_mask = result_df.get("use_embryo_flag", True)
    if isinstance(use_mask, bool):
        use_mask = pd.Series([use_mask] * len(result_df), index=result_df.index)
    use_mask = use_mask.astype(bool)
    ph_wt = (result_df["phenotype"].astype(str).str.lower() == "wt") if ("phenotype" in result_df.columns) else pd.Series(False, index=result_df.index)
    ctrl = result_df["control_flag"].astype(bool) if ("control_flag" in result_df.columns) else pd.Series(False, index=result_df.index)
    ref_mask = (ph_wt | ctrl) & use_mask

    debug_info['ref_count'] = ref_mask.sum()
    debug_info['ref_mask'] = ref_mask

    print(f"📈 Debug Summary:")
    print(f"   Total embryos: {debug_info['total_embryos']}")
    print(f"   Flagged as outliers: {debug_info['flagged_count']} ({debug_info['flagged_pct']:.1f}%)")
    print(f"   Reference embryos: {debug_info['ref_count']}")

    return result_df, debug_info


def calculate_thresholds(df: pd.DataFrame, stage_ref_path: Path) -> np.ndarray:
    """Calculate the actual thresholds used for each data point."""
    print("\n🎯 Calculating thresholds used...")

    # This replicates the stage_ref fallback logic from _sa_qc_with_fallback
    ref_df = pd.read_csv(stage_ref_path)

    import scipy.interpolate
    sa_of_stage = scipy.interpolate.interp1d(
        ref_df["stage_hpf"].to_numpy(),
        ref_df["sa_um"].to_numpy(),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    stage_vals = df["predicted_stage_hpf"].to_numpy()
    sa_ref = sa_of_stage(stage_vals)

    # Calculate calibration scale
    scale = 1.0
    valid = (~np.isnan(sa_ref)) & (~df["surface_area_um"].isna())
    if valid.sum() >= 5:
        ratio = (df.loc[valid, "surface_area_um"].to_numpy() / sa_ref[valid])
        high = np.quantile(ratio, 0.975)
        ratio = ratio[ratio <= high]
        if len(ratio) > 0:
            scale = float(np.median(ratio))

    thresholds = scale * 1.40 * sa_ref  # margin_k = 1.40 (matches build04)

    print(f"   Calibration scale: {scale:.3f}")
    print(f"   Margin multiplier: 1.40")
    print(f"   Threshold range: {np.nanmin(thresholds):.1f} - {np.nanmax(thresholds):.1f}")

    return thresholds, scale


def plot_sa_outliers(df: pd.DataFrame, debug_info: dict, thresholds: np.ndarray, scale: float, exp: str):
    """Create visualization of SA outlier detection."""
    plt.figure(figsize=(12, 8))

    # Color by status
    colors = []
    labels = []

    for idx, row in df.iterrows():
        if debug_info['ref_mask'].iloc[idx]:
            colors.append('green')
            labels.append('Control/Reference')
        elif row['sa_outlier_flag']:
            colors.append('red')
            labels.append('SA Outlier')
        else:
            colors.append('blue')
            labels.append('Normal')

    # Scatter plot
    plt.scatter(
        df['predicted_stage_hpf'],
        df['surface_area_um'],
        c=colors,
        alpha=0.7,
        s=60
    )

    # Plot threshold line
    stage_range = np.linspace(df['predicted_stage_hpf'].min(), df['predicted_stage_hpf'].max(), 100)
    valid_mask = ~np.isnan(thresholds)
    if valid_mask.any():
        # Sort by stage for line plot
        sorted_idx = np.argsort(df['predicted_stage_hpf'].to_numpy())
        sorted_stages = df['predicted_stage_hpf'].to_numpy()[sorted_idx]
        sorted_thresholds = thresholds[sorted_idx]
        valid_sorted = ~np.isnan(sorted_thresholds)

        plt.plot(
            sorted_stages[valid_sorted],
            sorted_thresholds[valid_sorted],
            'green',
            linewidth=2,
            alpha=0.8,
            label=f'SA Threshold (scale={scale:.3f})'
        )

    # Annotate outliers
    outliers = df[df['sa_outlier_flag']]
    for idx, row in outliers.iterrows():
        if 'embryo_id' in df.columns:
            label = row['embryo_id']
        elif 'snip_id' in df.columns:
            label = row['snip_id']
        else:
            label = f"Row {idx}"

        plt.annotate(
            label,
            (row['predicted_stage_hpf'], row['surface_area_um']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Normal'),
        Patch(facecolor='red', label='SA Outlier'),
        Patch(facecolor='green', label='Control/Reference'),
        plt.Line2D([0], [0], color='green', linewidth=2, label=f'Threshold (scale={scale:.3f})')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.xlabel('Predicted Stage (hpf)')
    plt.ylabel('Surface Area (µm²)')
    plt.title(f'SA Outlier Detection Debug - {exp}\n'
              f'{debug_info["flagged_count"]}/{debug_info["total_embryos"]} '
              f'({debug_info["flagged_pct"]:.1f}%) flagged as outliers')
    plt.grid(True, alpha=0.3)

    # Save plot
    out_path = f"debug_sa_outliers_{exp}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📊 Plot saved to: {out_path}")


def print_detailed_outliers(df: pd.DataFrame):
    """Print detailed info about flagged outliers."""
    outliers = df[df['sa_outlier_flag']]

    if len(outliers) == 0:
        print("\n✅ No outliers detected!")
        return

    print(f"\n🚨 Detailed outlier information ({len(outliers)} embryos):")
    print("-" * 80)

    for idx, row in outliers.iterrows():
        embryo_id = row.get('embryo_id', f'Row_{idx}')
        stage = row['predicted_stage_hpf']
        sa = row['surface_area_um']

        print(f"• {embryo_id}")
        print(f"  Stage: {stage:.2f} hpf")
        print(f"  Surface Area: {sa:.1f} µm²")
        print()


def main():
    parser = argparse.ArgumentParser(description="Debug SA outlier detection")
    parser.add_argument("--exp", required=True, help="Experiment name")
    parser.add_argument("--root", default="morphseq_playground", help="Data root directory")
    args = parser.parse_args()

    root = Path(args.root)
    exp = args.exp

    print(f"🔍 Debugging SA outlier detection for: {exp}")
    print(f"📁 Data root: {root}")

    # Load data
    df = load_build03_data(root, exp)
    df = prepare_data_for_qc(df)

    # Check for stage_ref
    stage_ref = root / "metadata" / "stage_ref_df.csv"
    if not stage_ref.exists():
        print(f"❌ Stage reference not found: {stage_ref}")
        return 1

    print(f"📊 Using stage reference: {stage_ref}")

    # Run SA QC with debugging
    result_df, debug_info = debug_sa_qc(df, stage_ref)

    # Calculate actual thresholds used
    thresholds, scale = calculate_thresholds(result_df, stage_ref)

    # Print detailed outlier info
    print_detailed_outliers(result_df)

    # Create visualization
    plot_sa_outliers(result_df, debug_info, thresholds, scale, exp)

    print("\n✅ Debug analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())