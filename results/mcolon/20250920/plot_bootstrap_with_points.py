#!/usr/bin/env python3
"""
Plotting script for bootstrap splines with original data points.
This version applies PCA to the original data to match the spline coordinates.
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# Add the morphseq src directory to Python path
morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
sys.path.insert(0, os.path.join(morphseq_root, "src"))

def apply_pca_to_points(df, z_mu_biological_columns, n_components=3):
    """
    Apply PCA to data points to get PCA coordinates that match the splines.
    """
    print(f"Applying PCA to {len(df)} data points with {len(z_mu_biological_columns)} features...")
    
    # Remove NaN values
    df_clean = df.dropna(subset=z_mu_biological_columns).copy()
    print(f"After removing NaN: {len(df_clean)} points remaining")
    
    if df_clean.empty:
        print("No valid data points after removing NaN values")
        return df
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_clean[z_mu_biological_columns])
    
    # Add PCA columns
    for i in range(n_components):
        df_clean[f'PCA_{i:02d}_bio'] = pca_result[:, i]
    
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance ratios: {explained_variance}")
    
    return df_clean

def plot_splines_with_original_points(
    pert_splines,
    df_points,
    fit_cols=None,
    group_by_col="chem_n_genotype",
    save_dir=None,
    filename="bootstrap_splines_with_points.html",
    show_uncertainty=True,
    uncertainty_opacity=0.2,
    spline_width=8,
    point_opacity=0.3,
    point_size=2,
    max_points_per_group=1000,
    title="Bootstrap Splines with Original Data Points"
):
    """
    Plot bootstrap splines with original data points overlaid.
    """
    # Auto-detect PCA columns
    if fit_cols is None:
        fit_cols = [col for col in pert_splines.columns if col.startswith("PCA_") and col.endswith("_bio")]
    
    if len(fit_cols) < 3:
        raise ValueError(f"Need at least 3 PCA columns. Found: {fit_cols}")
    
    print(f"Using PCA columns: {fit_cols}")
    
    # Check for uncertainty columns
    se_cols = [col + "_se" for col in fit_cols]
    has_uncertainty = all(col in pert_splines.columns for col in se_cols)
    
    if show_uncertainty and not has_uncertainty:
        print("Warning: Uncertainty columns not found. Disabling uncertainty display.")
        show_uncertainty = False
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Plotly
    unique_groups = pert_splines[group_by_col].unique()
    
    print(f"Plotting {len(unique_groups)} groups: {unique_groups}")
    
    # Plot original data points first (so they appear behind splines)
    if df_points is not None and all(col in df_points.columns for col in fit_cols):
        print("Adding original data points...")
        
        for i, group in enumerate(unique_groups):
            if group in df_points[group_by_col].values:
                group_points = df_points[df_points[group_by_col] == group].copy()
                
                # Downsample if too many points
                if len(group_points) > max_points_per_group:
                    group_points = group_points.sample(n=max_points_per_group, random_state=42)
                    print(f"Downsampled {group} points to {max_points_per_group}")
                
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter3d(
                    x=group_points[fit_cols[0]],
                    y=group_points[fit_cols[1]],
                    z=group_points[fit_cols[2]],
                    mode='markers',
                    name=f'{group} (data)',
                    marker=dict(
                        size=point_size,
                        color=color,
                        opacity=point_opacity
                    ),
                    showlegend=True
                ))
                
                print(f"Added {len(group_points)} points for {group}")
    
    # Plot bootstrap splines
    print("Adding bootstrap splines...")
    for i, group in enumerate(unique_groups):
        group_data = pert_splines[pert_splines[group_by_col] == group].copy()
        
        if group_data.empty:
            continue
        
        color = colors[i % len(colors)]
        
        # Sort by spline_point_index if available
        if 'spline_point_index' in group_data.columns:
            group_data = group_data.sort_values('spline_point_index')
        
        # Main spline line
        fig.add_trace(go.Scatter3d(
            x=group_data[fit_cols[0]],
            y=group_data[fit_cols[1]],
            z=group_data[fit_cols[2]],
            mode='lines+markers',
            name=f'{group} (spline)',
            line=dict(color=color, width=spline_width),
            marker=dict(size=4, color=color),
            showlegend=True
        ))
        
        # Add uncertainty bands if available
        if show_uncertainty and has_uncertainty:
            # Upper bound
            upper_x = group_data[fit_cols[0]] + group_data[se_cols[0]]
            upper_y = group_data[fit_cols[1]] + group_data[se_cols[1]]
            upper_z = group_data[fit_cols[2]] + group_data[se_cols[2]]
            
            # Lower bound  
            lower_x = group_data[fit_cols[0]] - group_data[se_cols[0]]
            lower_y = group_data[fit_cols[1]] - group_data[se_cols[1]]
            lower_z = group_data[fit_cols[2]] - group_data[se_cols[2]]
            
            # Add uncertainty cloud
            fig.add_trace(go.Scatter3d(
                x=upper_x, y=upper_y, z=upper_z,
                mode='markers',
                name=f'{group} (+SE)',
                marker=dict(size=1, color=color, opacity=uncertainty_opacity),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter3d(
                x=lower_x, y=lower_y, z=lower_z,
                mode='markers', 
                name=f'{group} (-SE)',
                marker=dict(size=1, color=color, opacity=uncertainty_opacity),
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=fit_cols[0],
            yaxis_title=fit_cols[1],
            zaxis_title=fit_cols[2],
            aspectmode='data'
        ),
        width=1400,
        height=1000,
        title=title,
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    # Save the plot
    if save_dir:
        save_path = os.path.join(save_dir, filename)
        fig.write_html(save_path)
        print(f"Plot saved to: {save_path}")
    
    return fig

def main():
    # Set up directories
    results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250920"
    data_dir = os.path.join(results_dir, "data")
    plot_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plot directory: {plot_dir}")
    
    # Load bootstrap splines
    splines_file = os.path.join(data_dir, "test_bootstrap_splines.csv")
    print(f"\\nLoading bootstrap splines from: {splines_file}")
    
    if not os.path.exists(splines_file):
        print(f"Error: Splines file not found: {splines_file}")
        print("Run test_bootstrap_splines.py first to generate the spline data.")
        return
    
    pert_splines = pd.read_csv(splines_file)
    print(f"Loaded splines shape: {pert_splines.shape}")
    print(f"Spline conditions: {pert_splines['chem_n_genotype'].unique()}")
    
    # Load original data points
    points_file = os.path.join(data_dir, "tricane_tritraition_phenotype_analysis_input.csv")
    print(f"\\nLoading original data from: {points_file}")
    
    if not os.path.exists(points_file):
        print(f"Error: Points file not found: {points_file}")
        return
    
    df_full = pd.read_csv(points_file, low_memory=False)
    print(f"Loaded full data shape: {df_full.shape}")
    
    # Filter to same conditions as splines
    spline_conditions = pert_splines['chem_n_genotype'].unique()
    df_points = df_full[df_full['chem_n_genotype'].isin(spline_conditions)].copy()
    print(f"Filtered to spline conditions: {df_points.shape}")
    
    # Find z_mu_b columns for PCA
    z_mu_b_cols = [col for col in df_points.columns if col.startswith('z_mu_b_')]
    print(f"Found {len(z_mu_b_cols)} z_mu_b columns for PCA")
    
    if len(z_mu_b_cols) == 0:
        print("Error: No z_mu_b columns found for PCA")
        return
    
    # Apply PCA to match spline coordinates
    print(f"\\n{'='*50}")
    print("APPLYING PCA TO ORIGINAL DATA POINTS")
    print(f"{'='*50}")
    
    df_points_pca = apply_pca_to_points(
        df=df_points,
        z_mu_biological_columns=z_mu_b_cols,
        n_components=3
    )
    
    if df_points_pca.empty:
        print("Error: No valid data after PCA")
        return
    
    print(f"PCA applied. Final points shape: {df_points_pca.shape}")
    
    # Create comprehensive plot with points and splines
    print(f"\\n{'='*50}")
    print("CREATING PLOT WITH ORIGINAL POINTS AND BOOTSTRAP SPLINES")
    print(f"{'='*50}")
    
    try:
        fig = plot_splines_with_original_points(
            pert_splines=pert_splines,
            df_points=df_points_pca,
            group_by_col="chem_n_genotype",
            save_dir=plot_dir,
            filename="bootstrap_splines_with_original_points.html",
            show_uncertainty=True,
            uncertainty_opacity=0.2,
            spline_width=10,
            point_opacity=0.3,
            point_size=2,
            max_points_per_group=800,  # Limit points for better visualization
            title="Bootstrap Splines with Original Data Points and Uncertainty"
        )
        
        # Also create a version without uncertainty for clearer view of fit
        fig_simple = plot_splines_with_original_points(
            pert_splines=pert_splines,
            df_points=df_points_pca,
            group_by_col="chem_n_genotype",
            save_dir=plot_dir,
            filename="bootstrap_splines_with_points_simple.html",
            show_uncertainty=False,
            spline_width=12,
            point_opacity=0.4,
            point_size=2,
            max_points_per_group=800,
            title="Bootstrap Splines with Original Data Points (No Uncertainty)"
        )
        
        print(f"\\n{'='*50}")
        print("SUCCESS: Plots with original data points created!")
        print(f"{'='*50}")
        print(f"Plots saved to: {plot_dir}")
        print("Files created:")
        print("  - bootstrap_splines_with_original_points.html (with uncertainty)")
        print("  - bootstrap_splines_with_points_simple.html (clean view)")
        
    except Exception as e:
        print(f"\\n{'='*50}")
        print(f"ERROR: {e}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()