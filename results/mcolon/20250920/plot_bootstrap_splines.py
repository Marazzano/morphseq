#!/usr/bin/env python3
"""
Script to plot bootstrap splines with uncertainty estimates
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Add the morphseq src directory to the path
morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
sys.path.insert(0, os.path.join(morphseq_root, "src", "functions"))

# Import the plotting function from improved_build_splines
from improved_build_splines import plot_bootstrap_splines_with_uncertainty

def load_test_data():
    """Load the test data and bootstrap splines"""
    results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250920"
    data_dir = os.path.join(results_dir, "data")
    
    # Load the original data
    data_file = os.path.join(data_dir, "tricane_tritraition_phenotype_analysis_input.csv")
    df = pd.read_csv(data_file)
    print(f"Loaded original data: {df.shape}")
    
    # Load the bootstrap splines
    splines_file = os.path.join(data_dir, "test_bootstrap_splines.csv")
    splines_df = pd.read_csv(splines_file)
    print(f"Loaded bootstrap splines: {splines_df.shape}")
    
    return df, splines_df

def create_enhanced_3d_plot(df_points, df_splines, save_dir):
    """Create an enhanced 3D plot with bootstrap uncertainty"""
    
    # Filter the points data to match the splines conditions
    conditions = df_splines['chem_n_genotype'].unique()
    df_points_filtered = df_points[df_points['chem_n_genotype'].isin(conditions)].copy()
    
    # Add PCA columns to points data if they don't exist
    if 'PCA_00_bio' not in df_points_filtered.columns:
        print("Adding PCA columns to points data...")
        # Auto-detect z_mu_b columns
        z_mu_b_cols = [col for col in df_points_filtered.columns if col.startswith('z_mu_b_')]
        print(f"Using {len(z_mu_b_cols)} z_mu_b columns for PCA")
        
        from sklearn.decomposition import PCA
        
        # Clean data and apply PCA
        df_clean = df_points_filtered.dropna(subset=z_mu_b_cols)
        pca = PCA(n_components=3)
        pca_values = pca.fit_transform(df_clean[z_mu_b_cols])
        
        # Add PCA columns
        pca_cols = ['PCA_00_bio', 'PCA_01_bio', 'PCA_02_bio']
        for i, col in enumerate(pca_cols):
            df_clean[col] = pca_values[:, i]
        
        df_points_filtered = df_clean
        print(f"Applied PCA to points data. Explained variance: {pca.explained_variance_ratio_}")
    
    # Create the figure
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Plotly
    
    # PCA columns to plot
    pca_cols = ['PCA_00_bio', 'PCA_01_bio', 'PCA_02_bio']
    
    for i, condition in enumerate(conditions):
        color = colors[i % len(colors)]
        
        # Plot the data points
        points_data = df_points_filtered[df_points_filtered['chem_n_genotype'] == condition]
        
        # Sample points if too many (for performance)
        if len(points_data) > 1000:
            points_data = points_data.sample(n=1000, random_state=42)
        
        fig.add_trace(go.Scatter3d(
            x=points_data[pca_cols[0]],
            y=points_data[pca_cols[1]],
            z=points_data[pca_cols[2]],
            mode='markers',
            marker=dict(
                size=3,
                color=color,
                opacity=0.6
            ),
            name=f'{condition} (points)',
            showlegend=True
        ))
        
        # Plot the spline
        spline_data = df_splines[df_splines['chem_n_genotype'] == condition]
        
        fig.add_trace(go.Scatter3d(
            x=spline_data[pca_cols[0]],
            y=spline_data[pca_cols[1]],
            z=spline_data[pca_cols[2]],
            mode='lines',
            line=dict(
                color=color,
                width=8
            ),
            name=f'{condition} (spline)',
            showlegend=True
        ))
        
        # Add uncertainty bands
        se_cols = [col + '_se' for col in pca_cols]
        if all(col in spline_data.columns for col in se_cols):
            # Upper bound
            upper_x = spline_data[pca_cols[0]] + spline_data[se_cols[0]]
            upper_y = spline_data[pca_cols[1]] + spline_data[se_cols[1]]
            upper_z = spline_data[pca_cols[2]] + spline_data[se_cols[2]]
            
            # Lower bound
            lower_x = spline_data[pca_cols[0]] - spline_data[se_cols[0]]
            lower_y = spline_data[pca_cols[1]] - spline_data[se_cols[1]]
            lower_z = spline_data[pca_cols[2]] - spline_data[se_cols[2]]
            
            # Add uncertainty cloud
            fig.add_trace(go.Scatter3d(
                x=upper_x,
                y=upper_y,
                z=upper_z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.3
                ),
                name=f'{condition} (+SE)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter3d(
                x=lower_x,
                y=lower_y,
                z=lower_z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.3
                ),
                name=f'{condition} (-SE)',
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=pca_cols[0],
            yaxis_title=pca_cols[1],
            zaxis_title=pca_cols[2],
            aspectmode='data'
        ),
        width=1200,
        height=800,
        title="Bootstrap Splines with Uncertainty Estimates - Tricane Titration",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "bootstrap_splines_3d_plot.html")
    fig.write_html(save_path)
    print(f"3D plot saved to: {save_path}")
    
    return fig

def create_uncertainty_analysis_plot(df_splines, save_dir):
    """Create plots to analyze the uncertainty estimates"""
    
    # Calculate average uncertainty for each condition
    pca_cols = ['PCA_00_bio', 'PCA_01_bio', 'PCA_02_bio']
    se_cols = [col + '_se' for col in pca_cols]
    
    uncertainty_stats = []
    for condition in df_splines['chem_n_genotype'].unique():
        condition_data = df_splines[df_splines['chem_n_genotype'] == condition]
        
        for i, (pca_col, se_col) in enumerate(zip(pca_cols, se_cols)):
            uncertainty_stats.append({
                'condition': condition,
                'pca_component': f'PC{i+1}',
                'mean_uncertainty': condition_data[se_col].mean(),
                'max_uncertainty': condition_data[se_col].max(),
                'min_uncertainty': condition_data[se_col].min()
            })
    
    uncertainty_df = pd.DataFrame(uncertainty_stats)
    
    # Create bar plot of mean uncertainties
    fig = px.bar(
        uncertainty_df,
        x='condition',
        y='mean_uncertainty',
        color='pca_component',
        title='Mean Bootstrap Uncertainty by Condition and PCA Component',
        labels={'mean_uncertainty': 'Mean Standard Error', 'condition': 'Tricane Condition'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        width=800,
        height=600
    )
    
    # Save uncertainty plot
    uncertainty_save_path = os.path.join(save_dir, "uncertainty_analysis.html")
    fig.write_html(uncertainty_save_path)
    print(f"Uncertainty analysis plot saved to: {uncertainty_save_path}")
    
    return fig, uncertainty_df

def create_spline_trajectory_plot(df_splines, save_dir):
    """Create 2D projections of the spline trajectories"""
    
    pca_cols = ['PCA_00_bio', 'PCA_01_bio', 'PCA_02_bio']
    
    # Create subplot with 3 2D projections
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['PC1 vs PC2', 'PC1 vs PC3', 'PC2 vs PC3'],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    colors = px.colors.qualitative.Plotly
    
    # Define the projections
    projections = [
        (pca_cols[0], pca_cols[1]),  # PC1 vs PC2
        (pca_cols[0], pca_cols[2]),  # PC1 vs PC3
        (pca_cols[1], pca_cols[2])   # PC2 vs PC3
    ]
    
    for i, condition in enumerate(df_splines['chem_n_genotype'].unique()):
        color = colors[i % len(colors)]
        condition_data = df_splines[df_splines['chem_n_genotype'] == condition]
        
        for j, (x_col, y_col) in enumerate(projections):
            # Add main spline
            fig.add_trace(
                go.Scatter(
                    x=condition_data[x_col],
                    y=condition_data[y_col],
                    mode='lines+markers',
                    name=condition if j == 0 else None,  # Only show legend for first subplot
                    showlegend=(j == 0),
                    line=dict(color=color, width=3),
                    marker=dict(size=4, color=color)
                ),
                row=1, col=j+1
            )
            
            # Add uncertainty if available
            x_se_col = x_col + '_se'
            y_se_col = y_col + '_se'
            
            if x_se_col in condition_data.columns and y_se_col in condition_data.columns:
                # Add error bars or uncertainty region
                for _, point in condition_data.iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[point[x_col] - point[x_se_col], point[x_col] + point[x_se_col]],
                            y=[point[y_col], point[y_col]],
                            mode='lines',
                            line=dict(color=color, width=1),
                            opacity=0.3,
                            showlegend=False
                        ),
                        row=1, col=j+1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[point[x_col], point[x_col]],
                            y=[point[y_col] - point[y_se_col], point[y_col] + point[y_se_col]],
                            mode='lines',
                            line=dict(color=color, width=1),
                            opacity=0.3,
                            showlegend=False
                        ),
                        row=1, col=j+1
                    )
    
    # Update layout
    fig.update_layout(
        title="Spline Trajectories with Uncertainty - 2D Projections",
        width=1400,
        height=500
    )
    
    # Save the plot
    trajectory_save_path = os.path.join(save_dir, "spline_trajectories_2d.html")
    fig.write_html(trajectory_save_path)
    print(f"2D trajectory plot saved to: {trajectory_save_path}")
    
    return fig

def main():
    """Main function to create all plots"""
    print("="*50)
    print("TESTING BOOTSTRAP SPLINE PLOTTING")
    print("="*50)
    
    # Set up directories
    results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250920"
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df_points, df_splines = load_test_data()
    
    print(f"Splines data shape: {df_splines.shape}")
    print(f"Splines columns: {list(df_splines.columns)}")
    print(f"Conditions: {df_splines['chem_n_genotype'].unique()}")
    
    # Create plots
    print("\n1. Creating 3D plot with uncertainty...")
    fig_3d = create_enhanced_3d_plot(df_points, df_splines, plot_dir)
    
    print("\n2. Creating uncertainty analysis...")
    fig_uncertainty, uncertainty_stats = create_uncertainty_analysis_plot(df_splines, plot_dir)
    
    print("\n3. Creating 2D trajectory plots...")
    fig_2d = create_spline_trajectory_plot(df_splines, plot_dir)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("UNCERTAINTY SUMMARY")
    print("="*50)
    print(uncertainty_stats.groupby('condition')['mean_uncertainty'].mean().sort_values(ascending=False))
    
    print("\n" + "="*50)
    print("SUCCESS: All plots created!")
    print("="*50)
    print(f"Check the plots directory: {plot_dir}")
    
    return df_points, df_splines, fig_3d, fig_uncertainty, fig_2d

if __name__ == "__main__":
    df_points, df_splines, fig_3d, fig_uncertainty, fig_2d = main()