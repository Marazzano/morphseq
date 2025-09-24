#!/usr/bin/env python3
"""
Test script for bootstrap spline functionality.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the morphseq src directory to Python path
morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
sys.path.insert(0, os.path.join(morphseq_root, "src"))

# Import required functions
from functions.improved_build_splines import build_splines_and_segments_with_bootstrap
from functions.spline_fitting_v2 import spline_fit_wrapper

def main():
    # Set up directories
    results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250920"
    data_dir = os.path.join(results_dir, "data")
    plot_dir = os.path.join(results_dir, "plots")
    
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plot directory: {plot_dir}")
    
    # Create directories if they don't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load the data
    input_file = os.path.join(data_dir, "tricane_tritraition_phenotype_analysis_input.csv")
    print(f"Loading data from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    print(f"Loaded data shape: {df.shape}")
    
    # Check what columns are available
    print(f"\nColumns in data:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Check for z_mu_b columns
    z_mu_b_cols = [col for col in df.columns if col.startswith('z_mu_b_')]
    print(f"\nFound {len(z_mu_b_cols)} z_mu_b columns")
    if len(z_mu_b_cols) > 0:
        print(f"First few z_mu_b columns: {z_mu_b_cols[:5]}")
    
    # Check available conditions
    if 'chem_n_genotype' in df.columns:
        print(f"\nAvailable chem_n_genotype conditions:")
        condition_counts = df['chem_n_genotype'].value_counts()
        for condition, count in condition_counts.head(10).items():
            print(f"  {condition}: {count}")
    
    # Test with a subset of conditions
    test_conditions = []
    
    # Look for tricane conditions
    available_conditions = df['chem_n_genotype'].unique()
    tricane_conditions = [cond for cond in available_conditions if 'tri_' in str(cond)]
    
    if len(tricane_conditions) >= 2:
        test_conditions = tricane_conditions[:3]  # Take first 3 tricane conditions
    else:
        # Fallback to any available conditions
        test_conditions = list(available_conditions)[:3]
    
    print(f"\nTesting with conditions: {test_conditions}")
    
    # Filter data for testing
    test_df = df[df['chem_n_genotype'].isin(test_conditions)].copy()
    print(f"Test data shape: {test_df.shape}")
    
    if test_df.empty:
        print("Error: No data after filtering for test conditions")
        return
    
    try:
        # Test bootstrap spline fitting
        print(f"\n{'='*50}")
        print("TESTING BOOTSTRAP SPLINE FITTING")
        print(f"{'='*50}")
        
        pert_splines, df_augmented, segment_info_df = build_splines_and_segments_with_bootstrap(
            df=test_df,
            model_index=76,  # Test model index
            spline_fit_wrapper=spline_fit_wrapper,
            save_dir=data_dir,
            comparisons=test_conditions,
            group_by_col="chem_n_genotype",
            stage_col="predicted_stage_hpf",
            # Bootstrap parameters
            bandwidth=1,
            h= .5,  # Step size parameter (None = automatic)
            n_boots=5,  # Small number for testing
            boot_size=500,  # Small size for testing
            n_spline_points=100,  # Fewer points for testing
            time_window=2,
            # Segmentation parameters
            k=20  # Fewer segments for testing
        )
        
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        print(f"Bootstrap splines shape: {pert_splines.shape}")
        print(f"Augmented data shape: {df_augmented.shape}")
        print(f"Segment info shape: {segment_info_df.shape}")
        
        if not pert_splines.empty:
            print(f"\nSplines columns: {list(pert_splines.columns)}")
            
            # Check for uncertainty columns
            uncertainty_cols = [col for col in pert_splines.columns if col.endswith('_se')]
            print(f"Uncertainty columns found: {uncertainty_cols}")
            
            # Show conditions in splines
            print(f"Conditions in splines: {pert_splines['chem_n_genotype'].unique()}")
            
            # Save results
            splines_output = os.path.join(data_dir, "test_bootstrap_splines.csv")
            pert_splines.to_csv(splines_output, index=False)
            print(f"Saved test splines to: {splines_output}")
            
            print(f"\n{'='*50}")
            print("SUCCESS: Bootstrap spline fitting completed!")
            print(f"{'='*50}")
            
        else:
            print("Warning: No splines were generated")
            
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR: {e}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()