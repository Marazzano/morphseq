"""
Tutorial 05: Spline Fits for Projection-Derived Groups

Fits splines for cluster groups derived from the combined projection
and saves the spline coordinates for downstream comparison.

Inputs:
- Combined projection assignments from Tutorial 04
- Source trajectories from 20260122 + 20260124
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "05"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECTION_DIR = OUTPUT_DIR / "figures" / "04" / "projection_results"
PROJECTION_CSV = PROJECTION_DIR / "combined_projection_bootstrap.csv"

SOURCE_EXPERIMENTS = ["20260122", "20260124"]
meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"

print("=" * 80)
print("TUTORIAL 05: SPLINES FOR PROJECTION GROUPS")
print("=" * 80)

if not PROJECTION_CSV.exists():
    raise FileNotFoundError(f"Missing projection file: {PROJECTION_CSV}")

print(f"Loading projection assignments: {PROJECTION_CSV}")
proj = pd.read_csv(PROJECTION_CSV, low_memory=False)

print("Loading source trajectories...")
source_dfs = []
for exp_id in SOURCE_EXPERIMENTS:
    df_exp = pd.read_csv(meta_dir / f"qc_staged_{exp_id}.csv", low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp_id
    source_dfs.append(df_exp)

df = pd.concat(source_dfs, ignore_index=True)

# Merge cluster labels
proj_cols = ["embryo_id", "cluster_label", "membership"]
proj = proj[proj_cols].drop_duplicates(subset="embryo_id")
df = df.merge(proj, on="embryo_id", how="inner")
df = df[df["cluster_label"].notna()].copy()

print(f"Trajectories with labels: {df['embryo_id'].nunique()} embryos")
print(f"Cluster labels: {sorted(df['cluster_label'].unique())}")

# Choose coordinates for spline fitting
candidate_coords = ["baseline_deviation_normalized", "total_length_um"]
coords = [c for c in candidate_coords if c in df.columns]
if len(coords) < 2:
    raise ValueError(
        f"Need at least 2 coordinate columns for spline fitting. "
        f"Found: {coords}"
    )

print(f"Using coordinates: {coords}")

from analyze.spline_fitting import spline_fit_wrapper

print("\nFitting splines by cluster_label...")
spline_df = spline_fit_wrapper(
    df,
    group_by="cluster_label",
    pca_cols=coords,
    stage_col="predicted_stage_hpf",
    n_bootstrap=50,
    bootstrap_size=2500,
    n_spline_points=200,
    time_window=2,
)

print(f"Fitted splines for {spline_df['cluster_label'].nunique()} clusters")

csv_path = RESULTS_DIR / "05_projection_splines_by_cluster.csv"
pkl_path = RESULTS_DIR / "05_projection_splines_by_cluster.pkl"
spline_df.to_csv(csv_path, index=False)
print(f"Saved spline table: {csv_path}")

import pickle
with open(pkl_path, "wb") as f:
    pickle.dump(spline_df, f)
print(f"Saved spline pickle: {pkl_path}")

# Quick 2D visualization per cluster
clusters = sorted(df["cluster_label"].unique())
n_cols = 2
n_rows = int(np.ceil(len(clusters) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

for idx, cluster in enumerate(clusters):
    r, c = divmod(idx, n_cols)
    ax = axes[r][c]
    df_c = df[df["cluster_label"] == cluster]
    spline_c = spline_df[spline_df["cluster_label"] == cluster]

    ax.scatter(
        df_c[coords[0]],
        df_c[coords[1]],
        s=6,
        alpha=0.2,
        label="points"
    )
    if not spline_c.empty:
        ax.plot(
            spline_c[coords[0]],
            spline_c[coords[1]],
            color="red",
            linewidth=2,
            label="spline"
        )
    ax.set_title(cluster)
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])

for idx in range(len(clusters), n_rows * n_cols):
    r, c = divmod(idx, n_cols)
    axes[r][c].axis("off")

fig.suptitle("Spline Fits per Cluster (Projection Labels)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = FIGURES_DIR / "05_projection_splines_by_cluster.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved spline plot: {fig_path}")
plt.close(fig)

print("\n✓ Tutorial 05 complete.")
