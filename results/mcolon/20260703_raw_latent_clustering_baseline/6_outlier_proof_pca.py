"""
6_outlier_proof_pca.py
------------------------
Proves that 20260304_E03_e02 @ 72hpf is a genuine outlier in the ORIGINAL
80-dim raw z_mu_b feature space -- not an artifact introduced by UMAP init
or the condensation solver.

Method: take every (embryo, time_bin) row as an independent point in the
raw 80-dim z_mu_b space (no UMAP, no condensation). Find the target row's
10 nearest neighbors by Euclidean distance IN THAT ORIGINAL 80-dim SPACE.
Then fit a 3D PCA on all rows (all embryos, all time points) purely for
visualization, and color:
  - the target embryo's 72hpf row: gold star
  - its 10 nearest neighbors (in 80-dim space): green
  - every other row: red

If the neighbors (green) are genuinely far from the target in the 3D PCA
view too, and the target sits apart from the bulk of red points, that
confirms the isolation is real in the original space, not a UMAP artifact
-- PCA is a global linear projection with no per-bin alignment, so it
can't introduce the kind of chain-drift discontinuity aligned_umap_init can.

Output:
  figures/outlier_proof_pca3d.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

_HERE = Path(__file__).resolve().parent
TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"

TARGET_EMBRYO = "20260304_E03_e02"
TARGET_TIME_BIN = 72.0
N_NEIGHBORS = 10


def main() -> None:
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    print(f"Loaded {len(binned)} (embryo, time_bin) rows, {len(z_cols)} raw z_mu_b dims")

    X = binned[z_cols].values.astype(float)

    target_mask = (binned["embryo_id"] == TARGET_EMBRYO) & (binned["time_bin"] == TARGET_TIME_BIN)
    target_idx = np.where(target_mask.values)[0]
    if len(target_idx) == 0:
        raise ValueError(f"Target row {TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf not found")
    target_idx = target_idx[0]
    print(f"Target row index: {target_idx} ({TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf)")

    # ── Nearest neighbors in the ORIGINAL 80-dim space (no UMAP, no condensation) ──
    target_vec = X[target_idx : target_idx + 1, :]
    dists = cdist(target_vec, X)[0]
    dists[target_idx] = np.inf  # exclude self
    neighbor_order = np.argsort(dists)
    neighbor_idx = neighbor_order[:N_NEIGHBORS]

    print(f"\n{N_NEIGHBORS} nearest neighbors in raw 80-dim z_mu_b space:")
    for i in neighbor_idx:
        print(f"  dist={dists[i]:.3f}  {binned.iloc[i]['embryo_id']}  t={binned.iloc[i]['time_bin']}  "
              f"genotype={binned.iloc[i].get('genotype', '?')}")

    all_pairwise_mean = cdist(X, X)
    iu = np.triu_indices(len(X), k=1)
    global_mean_dist = all_pairwise_mean[iu].mean()
    print(f"\nFor context -- mean pairwise distance across ALL {len(X)} rows in raw 80-dim space: "
          f"{global_mean_dist:.3f}")
    print(f"Target's nearest neighbor distance: {dists[neighbor_idx[0]]:.3f} "
          f"({dists[neighbor_idx[0]] / global_mean_dist:.2f}x the global mean)")

    # ── 3D PCA on ALL rows, all embryos, all time points ────────────────────
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: PC1={var_explained[0]:.1%} PC2={var_explained[1]:.1%} PC3={var_explained[2]:.1%} "
          f"(total {var_explained.sum():.1%})")

    # ── Color assignment ─────────────────────────────────────────────────────
    color = np.full(len(X), "red", dtype=object)
    color[neighbor_idx] = "green"
    size = np.full(len(X), 3.0)
    size[neighbor_idx] = 6.0

    fig = go.Figure()

    # red = everyone else
    other_mask = np.ones(len(X), dtype=bool)
    other_mask[neighbor_idx] = False
    other_mask[target_idx] = False
    fig.add_trace(go.Scatter3d(
        x=coords[other_mask, 0], y=coords[other_mask, 1], z=coords[other_mask, 2],
        mode="markers",
        marker=dict(size=3, color="red", opacity=0.35),
        name="all other rows",
        text=[f"{binned.iloc[i]['embryo_id']} t={binned.iloc[i]['time_bin']}" for i in np.where(other_mask)[0]],
        hovertemplate="%{text}<extra></extra>",
    ))

    # green = 10 nearest neighbors in raw 80-dim space
    fig.add_trace(go.Scatter3d(
        x=coords[neighbor_idx, 0], y=coords[neighbor_idx, 1], z=coords[neighbor_idx, 2],
        mode="markers",
        marker=dict(size=7, color="green", opacity=0.95, line=dict(width=1, color="darkgreen")),
        name=f"{N_NEIGHBORS} nearest neighbors (raw 80-dim space)",
        text=[f"{binned.iloc[i]['embryo_id']} t={binned.iloc[i]['time_bin']} dist={dists[i]:.3f}" for i in neighbor_idx],
        hovertemplate="%{text}<extra></extra>",
    ))

    # gold star = the target itself
    fig.add_trace(go.Scatter3d(
        x=coords[[target_idx], 0], y=coords[[target_idx], 1], z=coords[[target_idx], 2],
        mode="markers",
        marker=dict(size=12, color="gold", symbol="diamond", line=dict(width=2, color="black")),
        name=f"TARGET: {TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf",
        text=[f"{TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf (TARGET)"],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        title=(
            f"<b>Outlier proof: {TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf in raw 80-dim z_mu_b space</b><br>"
            f"<sup>3D PCA of ALL {len(X)} (embryo, time_bin) rows, all embryos, all time points. "
            f"Nearest neighbor dist = {dists[neighbor_idx[0]]:.2f} vs. global mean pairwise dist = "
            f"{global_mean_dist:.2f} ({dists[neighbor_idx[0]]/global_mean_dist:.2f}x). "
            f"PCA var. explained: {var_explained.sum():.1%}</sup>"
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var_explained[0]:.1%})",
            yaxis_title=f"PC2 ({var_explained[1]:.1%})",
            zaxis_title=f"PC3 ({var_explained[2]:.1%})",
        ),
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        height=800,
        margin=dict(l=0, r=0, t=100, b=0),
    )

    out = FIGURES / "outlier_proof_pca3d.html"
    fig.write_html(str(out), include_plotlyjs="inline")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
