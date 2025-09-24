#!/usr/bin/env python3
"""
Plotting script for bootstrap splines with uncertainty visualization.
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add the morphseq src directory to Python path
morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
sys.path.insert(0, os.path.join(morphseq_root, "src"))

def plot_bootstrap_splines_3d(
    pert_splines,
    df_points=None,
    fit_cols=None,
    group_by_col="chem_n_genotype",
    save_dir=None,
    filename="bootstrap_splines_3d.html",
    show_uncertainty=True,
    uncertainty_opacity=0.3,
    spline_width=6,
    point_opacity=0.6,
    title="Bootstrap Splines with Uncertainty"
):
    """
    Create 3D plot of bootstrap splines with uncertainty bands.
    
    Parameters
    ----------
    pert_splines : pd.DataFrame
        DataFrame with spline points and uncertainty estimates
    df_points : pd.DataFrame, optional
        Original data points to plot alongside splines
    fit_cols : list
        PCA column names to plot (default: auto-detect)
    group_by_col : str
        Column to group/color by
    save_dir : str, optional
        Directory to save plot
    filename : str
        Filename for saved plot
    show_uncertainty : bool
        Whether to show uncertainty bands
    uncertainty_opacity : float
        Transparency for uncertainty bands
    spline_width : int
        Width of spline lines
    point_opacity : float
        Opacity for data points
    title : str
        Plot title
    """
    
    # Auto-detect PCA columns if not provided
    if fit_cols is None:
        fit_cols = [col for col in pert_splines.columns if col.startswith("PCA_") and col.endswith("_bio")]
        if len(fit_cols) < 3:
            # Fallback to standard PCA columns
            potential_cols = [col for col in pert_splines.columns if col.startswith("PCA_") and not col.endswith("_se")]
            fit_cols = potential_cols[:3]
    
    if len(fit_cols) < 3:
        raise ValueError(f"Need at least 3 PCA columns for 3D plotting. Found: {fit_cols}")
    
    print(f"Using PCA columns: {fit_cols}")
    
    # Check for uncertainty columns
    se_cols = [col + "_se" for col in fit_cols]
    has_uncertainty = all(col in pert_splines.columns for col in se_cols)
    
    if show_uncertainty and not has_uncertainty:
        print("Warning: Uncertainty columns not found. Disabling uncertainty display.")
        show_uncertainty = False
    elif show_uncertainty:
        print(f"Found uncertainty columns: {se_cols}")
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Plotly
    unique_groups = pert_splines[group_by_col].unique()
    
    # Plot data points first (if provided)
    if df_points is not None:
        print("Adding original data points...")
        
        # Check if df_points has the same PCA columns
        point_fit_cols = fit_cols.copy()
        if not all(col in df_points.columns for col in fit_cols):
            # Try mapping from standard PCA format
            point_fit_cols = [f"PCA_{i+1}" for i in range(3)]
            if not all(col in df_points.columns for col in point_fit_cols):
                print("Warning: Could not find matching PCA columns in data points")
                df_points = None
        
        if df_points is not None:
            for i, group in enumerate(unique_groups):
                if group in df_points[group_by_col].values:
                    group_points = df_points[df_points[group_by_col] == group]
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Scatter3d(
                        x=group_points[point_fit_cols[0]],
                        y=group_points[point_fit_cols[1]],
                        z=group_points[point_fit_cols[2]],
                        mode='markers',
                        name=f'{group} (points)',
                        marker=dict(
                            size=2,
                            color=color,
                            opacity=point_opacity
                        ),
                        showlegend=True
                    ))
    
    # Plot splines
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
        
        # Add uncertainty bands if available and requested
        if show_uncertainty and has_uncertainty:
            print(f"Adding uncertainty bands for {group}...")
            
            # Upper bound points
            upper_x = group_data[fit_cols[0]] + group_data[se_cols[0]]
            upper_y = group_data[fit_cols[1]] + group_data[se_cols[1]]
            upper_z = group_data[fit_cols[2]] + group_data[se_cols[2]]
            
            # Lower bound points
            lower_x = group_data[fit_cols[0]] - group_data[se_cols[0]]
            lower_y = group_data[fit_cols[1]] - group_data[se_cols[1]]
            lower_z = group_data[fit_cols[2]] - group_data[se_cols[2]]
            
            # Add uncertainty cloud (upper bound)
            fig.add_trace(go.Scatter3d(
                x=upper_x,
                y=upper_y,
                z=upper_z,
                mode='markers',
                name=f'{group} (+SE)',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=uncertainty_opacity,
                    symbol='circle-open'
                ),
                showlegend=False
            ))
            
            # Add uncertainty cloud (lower bound)
            fig.add_trace(go.Scatter3d(
                x=lower_x,
                y=lower_y,
                z=lower_z,
                mode='markers',
                name=f'{group} (-SE)',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=uncertainty_opacity,
                    symbol='circle-open'
                ),
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
        width=1200,
        height=800,
        title=title,
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor="Black",
            borderwidth=1
        )
    )

    # Save plot if directory specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.write_html(save_path)
        print(f"Bootstrap spline plot saved to: {save_path}")

    return fig


def plot_uncertainty_magnitude(
    pert_splines,
    fit_cols=None,
    group_by_col="chem_n_genotype",
    save_dir=None,
    filename="uncertainty_magnitude.html"
):
    """
    Plot the magnitude of uncertainty along each spline.
    """
    
    if fit_cols is None:
        fit_cols = [col for col in pert_splines.columns if col.startswith("PCA_") and col.endswith("_bio")]
        if len(fit_cols) < 3:
            potential_cols = [col for col in pert_splines.columns if col.startswith("PCA_") and not col.endswith("_se")]
            fit_cols = potential_cols[:3]
    
    se_cols = [col + "_se" for col in fit_cols]
    has_uncertainty = all(col in pert_splines.columns for col in se_cols)
    
    if not has_uncertainty:
        print("No uncertainty columns found for uncertainty magnitude plot")
        return None
    
    # Calculate total uncertainty magnitude
    pert_splines_copy = pert_splines.copy()
    pert_splines_copy['total_uncertainty'] = np.sqrt(
        pert_splines_copy[se_cols[0]]**2 + 
        pert_splines_copy[se_cols[1]]**2 + 
        pert_splines_copy[se_cols[2]]**2
    )
    
    # Create subplots
    unique_groups = pert_splines_copy[group_by_col].unique()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Total Uncertainty Magnitude",
            f"{fit_cols[0]} Uncertainty",
            f"{fit_cols[1]} Uncertainty", 
            f"{fit_cols[2]} Uncertainty"
        ],
        specs=[[{"colspan": 2}, None],
               [{}, {}]]
    )
    
    colors = px.colors.qualitative.Plotly
    
    for i, group in enumerate(unique_groups):
        group_data = pert_splines_copy[pert_splines_copy[group_by_col] == group].copy()
        if group_data.empty:
            continue
            
        color = colors[i % len(colors)]
        
        # Sort by spline_point_index if available
        if 'spline_point_index' in group_data.columns:
            group_data = group_data.sort_values('spline_point_index')
            x_axis = group_data['spline_point_index']
            x_title = "Spline Point Index"
        else:
            x_axis = range(len(group_data))
            x_title = "Point Number"
        
        # Total uncertainty
        fig.add_trace(
            go.Scatter(x=x_axis, y=group_data['total_uncertainty'],
                      mode='lines+markers', name=f'{group}',
                      line=dict(color=color),
                      legendgroup=group),
            row=1, col=1
        )
        
        # Individual component uncertainties
        for j, se_col in enumerate(se_cols):
            row_num = 2
            col_num = j + 1 if j < 2 else 2
            
            fig.add_trace(
                go.Scatter(x=x_axis, y=group_data[se_col],
                          mode='lines+markers', name=f'{group}',
                          line=dict(color=color),
                          showlegend=False,
                          legendgroup=group),
                row=row_num, col=col_num
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        title="Bootstrap Uncertainty Analysis",
        showlegend=True
    )
    
    # Update x-axis titles
    fig.update_xaxes(title_text=x_title, row=1, col=1)
    fig.update_xaxes(title_text=x_title, row=2, col=1)
    fig.update_xaxes(title_text=x_title, row=2, col=2)
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Total Uncertainty", row=1, col=1)
    fig.update_yaxes(title_text="Standard Error", row=2, col=1)
    fig.update_yaxes(title_text="Standard Error", row=2, col=2)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.write_html(save_path)
        print(f"Uncertainty analysis plot saved to: {save_path}")
    
    return fig


def main():
    # Set up directories
    results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250920"
    data_dir = os.path.join(results_dir, "data")
    plot_dir = os.path.join(results_dir, "plots")
    
    print(f"Results directory: {results_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Plot directory: {plot_dir}")
    
    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load the bootstrap splines data
    splines_file = os.path.join(data_dir, "test_bootstrap_splines.csv")
    print(f"Loading bootstrap splines from: {splines_file}")
    
    if not os.path.exists(splines_file):
        print(f"Error: Splines file not found: {splines_file}")
        print("Run test_bootstrap_splines.py first to generate the spline data.")
        return
    
    pert_splines = pd.read_csv(splines_file)
    print(f"Loaded splines shape: {pert_splines.shape}")
    print(f"Splines columns: {list(pert_splines.columns)}")
    print(f"Conditions: {pert_splines['chem_n_genotype'].unique()}")
    
    # Load original data points (optional)
    points_file = os.path.join(data_dir, "tricane_tritraition_phenotype_analysis_input.csv")
    df_points = None
    
    if os.path.exists(points_file):
        print(f"Loading original data points from: {points_file}")
        df_full = pd.read_csv(points_file)
        
        # Filter to same conditions as splines
        spline_conditions = pert_splines['chem_n_genotype'].unique()
        df_points = df_full[df_full['chem_n_genotype'].isin(spline_conditions)].copy()
        print(f"Filtered points shape: {df_points.shape}")
        
        # Check what PCA columns are available in the points data
        pca_cols_in_points = [col for col in df_points.columns if 'PCA' in col]
        print(f"PCA columns in points data: {pca_cols_in_points}")
        
        # If we have PCA_00_bio format in splines but PCA_1 format in points, add them
        if 'PCA_00_bio' in pert_splines.columns and 'PCA_1' in df_points.columns:
            df_points['PCA_00_bio'] = df_points['PCA_1']
            df_points['PCA_01_bio'] = df_points['PCA_2'] 
            df_points['PCA_02_bio'] = df_points['PCA_3']
            print("Added bio format PCA columns to points data")
        elif 'PCA_00_bio' in pert_splines.columns and 'PCA_00_bio' not in df_points.columns:
            # Need to apply PCA to original data points
            print("Applying PCA to original data points...")
            
            # Get z_mu_b columns
            z_mu_b_cols = [col for col in df_points.columns if col.startswith('z_mu_b_')]
            print(f"Found {len(z_mu_b_cols)} z_mu_b columns for PCA")
            
            if len(z_mu_b_cols) > 0:
                # Apply PCA
                from sklearn.decomposition import PCA
                
                # Clean data
                df_clean = df_points.dropna(subset=z_mu_b_cols).copy()
                print(f"After removing NaN rows: {len(df_clean)} points remaining")
                
                if len(df_clean) > 0:
                    # Fit PCA
                    pca = PCA(n_components=3)
                    pca_coords = pca.fit_transform(df_clean[z_mu_b_cols])
                    
                    # Add PCA columns to match spline format
                    df_clean['PCA_00_bio'] = pca_coords[:, 0]
                    df_clean['PCA_01_bio'] = pca_coords[:, 1] 
                    df_clean['PCA_02_bio'] = pca_coords[:, 2]
                    
                    # Update df_points
                    df_points = df_clean
                    
                    print(f"Applied PCA. Explained variance: {pca.explained_variance_ratio_}")
                    print(f"PCA points shape: {df_points.shape}")
                else:
                    print("Warning: No clean data available for PCA")
                    df_points = None
            else:
                print("Warning: No z_mu_b columns found for PCA")
                df_points = None
    else:
        print("Original data points file not found. Plotting splines only.")
    
    try:
        # Create 3D plot with uncertainty
        print(f"\n{'='*50}")
        print("CREATING 3D BOOTSTRAP SPLINE PLOT")
        print(f"{'='*50}")
        
        fig_3d = plot_bootstrap_splines_3d(
            pert_splines=pert_splines,
            df_points=df_points,
            group_by_col="chem_n_genotype",
            save_dir=plot_dir,
            filename="bootstrap_splines_3d_with_uncertainty.html",
            show_uncertainty=True,
            uncertainty_opacity=0.3,
            spline_width=8,
            point_opacity=0.4,
            title="Bootstrap Splines with Uncertainty Estimates"
        )
        
        # Create uncertainty analysis plot
        print(f"\n{'='*50}")
        print("CREATING UNCERTAINTY ANALYSIS PLOT")
        print(f"{'='*50}")
        
        fig_uncertainty = plot_uncertainty_magnitude(
            pert_splines=pert_splines,
            group_by_col="chem_n_genotype",
            save_dir=plot_dir,
            filename="bootstrap_uncertainty_analysis.html"
        )
        
        # Create a simpler 3D plot without uncertainty for comparison
        print(f"\n{'='*50}")
        print("CREATING SIMPLIFIED 3D SPLINE PLOT")
        print(f"{'='*50}")
        
        fig_simple = plot_bootstrap_splines_3d(
            pert_splines=pert_splines,
            df_points=df_points,
            group_by_col="chem_n_genotype", 
            save_dir=plot_dir,
            filename="bootstrap_splines_3d_simple.html",
            show_uncertainty=False,  # No uncertainty bands
            spline_width=10,
            point_opacity=0.5,
            title="Bootstrap Splines (Mean Only)"
        )
        
        print(f"\n{'='*50}")
        print("SUCCESS: All plots created!")
        print(f"{'='*50}")
        print(f"Plots saved to: {plot_dir}")
        print("Files created:")
        print("  - bootstrap_splines_3d_with_uncertainty.html")
        print("  - bootstrap_uncertainty_analysis.html") 
        print("  - bootstrap_splines_3d_simple.html")
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR: {e}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()