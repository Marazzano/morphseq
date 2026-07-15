"""
14_outlier_trace_neighbors.py
------------------------------
Evolves 6_outlier_proof_pca.py: answers the spatial question directly --
where does the outlier's WHOLE trajectory and its neighborhood sit
relative to the bulk of the population? -- via ONE shared PCA scene with
a 3-item dropdown that switches the GREEN-highlighted trajectory set.

Method
------
1. NN space = RAW 80-dim z_mu_b (Euclidean), matching script 6 exactly.
   Recompute the target row's (72hpf) nearest neighbors, excluding self.
   Take two discrete tiers: top-5 and top-15 (by ROW). A neighbor row is
   one (embryo, time_bin); we dedupe each tier to its unique embryo_ids
   and highlight each embryo's WHOLE trajectory.
2. ONE PCA fit on all 2871 rows, all embryos/time bins (as script 6).
3. Draw FULL TIME TRAJECTORIES (lines through each embryo's time_bin
   points, in time order) for the target embryo + every neighbor embryo.
   Everyone else (the bulk) stays a light-gray POINT cloud for readability.
4. A dropdown (updatemenus) with 3 items -- [Target] / [Top 5] / [Top 15]
   -- toggles which embryos' full traces render GREEN (popping over
   everything); non-highlighted neighbor traces keep their tier color
   (orange). No PCA refit per item, no separate scenes: the dropdown only
   flips per-trace visibility across the shared scene. Tier rings + the
   target 72hpf star are kept as reference markers.

Output
------
  figures/outlier_trace_neighbors.html
  tables/outlier_neighbors_ranked.csv (rank, distance, embryo_id,
    time_bin, genotype, experiment_id, tier) -- ALL rows ranked by
    distance to target, consumed downstream by the gallery.
"""
from __future__ import annotations

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
TIER1_N = 5
TIER2_N = 15

GREEN = "#2ca02c"
TIER_COLOR = "#ff7f0e"   # orange -- neighbor traces not currently highlighted green
TARGET_BASE = "#7f2fbf" # purple -- target trace when not the green-highlighted set
BULK_COLOR = "#c9c9c9"  # light gray point cloud for the readable bulk background


def main() -> None:
    df = pd.read_csv(TABLES / "pbx_binned_zmub_with_wt.csv", low_memory=False)
    z_cols = [c for c in df.columns if c.startswith("z_mu_b")]
    assert len(z_cols) == 80, f"expected 80 z_mu_b dims, got {len(z_cols)}"
    print(f"Loaded {len(df)} (embryo, time_bin) rows, {len(z_cols)} raw z_mu_b dims")

    X = df[z_cols].values.astype(float)

    target_mask = (df["embryo_id"] == TARGET_EMBRYO) & (df["time_bin"] == TARGET_TIME_BIN)
    target_idx_arr = np.where(target_mask.values)[0]
    if len(target_idx_arr) == 0:
        raise ValueError(f"Target row {TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf not found")
    target_idx = target_idx_arr[0]
    print(f"Target row index: {target_idx} ({TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf)")

    trace_mask = (df["embryo_id"] == TARGET_EMBRYO).values
    trace_idx = np.where(trace_mask)[0]
    trace_order = np.argsort(df.iloc[trace_idx]["time_bin"].values)
    trace_idx = trace_idx[trace_order]
    print(f"Target embryo full trajectory: {len(trace_idx)} rows, "
          f"time_bin range {df.iloc[trace_idx]['time_bin'].min()}-{df.iloc[trace_idx]['time_bin'].max()}")

    # ── Nearest neighbors in the ORIGINAL 80-dim space (no UMAP, no condensation) ──
    target_vec = X[target_idx : target_idx + 1, :]
    dists = cdist(target_vec, X)[0]
    dists_for_rank = dists.copy()
    dists_for_rank[target_idx] = np.inf  # exclude self from neighbor ranking
    neighbor_order = np.argsort(dists_for_rank)

    tier1_idx = neighbor_order[:TIER1_N]
    tier2_idx = neighbor_order[:TIER2_N]  # top-15 is a superset of top-5

    print(f"\nTop-{TIER1_N} nearest neighbors in raw 80-dim z_mu_b space:")
    for i in tier1_idx:
        row = df.iloc[i]
        print(f"  dist={dists[i]:.3f}  {row['embryo_id']}  t={row['time_bin']}  genotype={row.get('genotype', '?')}")

    print(f"\nTop-{TIER2_N} nearest neighbors (includes top-{TIER1_N} above), ranks 6-15:")
    for i in tier2_idx[TIER1_N:]:
        row = df.iloc[i]
        print(f"  dist={dists[i]:.3f}  {row['embryo_id']}  t={row['time_bin']}  genotype={row.get('genotype', '?')}")

    all_pairwise = cdist(X, X)
    iu = np.triu_indices(len(X), k=1)
    global_mean_dist = all_pairwise[iu].mean()
    print(f"\nFor context -- mean pairwise distance across ALL {len(X)} rows in raw 80-dim space: "
          f"{global_mean_dist:.3f}")
    print(f"Target's nearest neighbor distance: {dists[neighbor_order[0]]:.3f} "
          f"({dists[neighbor_order[0]] / global_mean_dist:.2f}x the global mean)")

    # ── Ranked neighbor table (ALL rows, complete schema for downstream reuse) ──
    rank_order = np.argsort(dists_for_rank)  # target itself sorts last (inf)
    n_rows = len(df)
    rank_df = pd.DataFrame({
        "rank": np.arange(1, n_rows + 1),
        "distance": dists_for_rank[rank_order],
        "embryo_id": df.iloc[rank_order]["embryo_id"].values,
        "time_bin": df.iloc[rank_order]["time_bin"].values,
        "genotype": df.iloc[rank_order]["genotype"].values,
        "experiment_id": df.iloc[rank_order]["experiment_id"].values,
    })
    # fix target's own distance/rank bookkeeping: target is excluded from neighbor
    # ranking (inf) but must still appear in the complete table with distance 0
    # and be labeled distinctly rather than falling out to "other".
    rank_df.loc[rank_df["distance"] == np.inf, "distance"] = 0.0

    tier = np.full(n_rows, "other", dtype=object)
    rank_positions = np.argsort(rank_order)  # position of original row i in the ranked table
    tier[rank_positions[tier2_idx]] = "top15"
    tier[rank_positions[tier1_idx]] = "top5"
    tier[rank_positions[target_idx]] = "target"
    rank_df["tier"] = tier

    rank_csv = TABLES / "outlier_neighbors_ranked.csv"
    rank_df.to_csv(rank_csv, index=False)
    print(f"\nSaved ranked neighbor table ({n_rows} rows) -> {rank_csv}")

    # ── PCA on ALL rows, all embryos, all time points ───────────────────────
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: PC1={var_explained[0]:.1%} PC2={var_explained[1]:.1%} "
          f"PC3={var_explained[2]:.1%} (total {var_explained.sum():.1%})")

    # ── Dedupe each tier to UNIQUE embryo_ids (a neighbor row is one (embryo,
    #    time_bin); we highlight that embryo's WHOLE trajectory) ──────────────
    def unique_embryos(idxs):
        # preserve first-encountered (nearest-rank) order
        seen, out = set(), []
        for i in idxs:
            emb = df.iloc[i]["embryo_id"]
            if emb not in seen:
                seen.add(emb)
                out.append(emb)
        return out

    target_embryos = [TARGET_EMBRYO]
    top5_embryos = unique_embryos(tier1_idx)
    top15_embryos = unique_embryos(tier2_idx)

    print(f"\nUnique embryos per highlight set:")
    print(f"  Target : {len(target_embryos)} -> {target_embryos}")
    print(f"  Top 5  : {len(top5_embryos)} unique embryos (from {TIER1_N} rows) -> {top5_embryos}")
    print(f"  Top 15 : {len(top15_embryos)} unique embryos (from {TIER2_N} rows) -> {top15_embryos}")

    # union of every embryo that ever needs a full trajectory line drawn
    # (target + all top-15 neighbor embryos). Everyone else = bulk points.
    relevant_embryos = list(dict.fromkeys(target_embryos + top15_embryos))
    relevant_mask = df["embryo_id"].isin(relevant_embryos).values
    bulk_mask = ~relevant_mask
    print(f"\nRelevant (trajectory-drawn) embryos: {len(relevant_embryos)}; "
          f"bulk points: {bulk_mask.sum()} rows")

    hover_text = [
        f"{df.iloc[i]['embryo_id']} t={df.iloc[i]['time_bin']} genotype={df.iloc[i].get('genotype', '?')} "
        f"dist={dists[i]:.3f}"
        for i in range(n_rows)
    ]

    # per-embryo row indices sorted in time order (for connected trajectories)
    def embryo_time_order(emb):
        idx = np.where(df["embryo_id"].values == emb)[0]
        return idx[np.argsort(df.iloc[idx]["time_bin"].values)]

    fig = go.Figure()

    # ── Base trace 0: the bulk point cloud (always visible, never toggled) ────
    fig.add_trace(go.Scatter3d(
        x=coords[bulk_mask, 0], y=coords[bulk_mask, 1], z=coords[bulk_mask, 2],
        mode="markers",
        marker=dict(size=2.5, color=BULK_COLOR, opacity=0.45),
        name="bulk (all other rows)",
        text=[hover_text[i] for i in np.where(bulk_mask)[0]],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
    ))

    # ── One full-trajectory line trace per relevant embryo. Color is set per
    #    dropdown state via a visibility array: we add TWO traces per embryo
    #    (a GREEN copy and a BASE-color copy) and toggle which is visible. ────
    #    order matters for the visibility bookkeeping below.
    green_trace_specs = []  # (trace_index, embryo)
    base_trace_specs = []   # (trace_index, embryo)

    def add_embryo_pair(emb):
        eidx = embryo_time_order(emb)
        is_target = (emb == TARGET_EMBRYO)
        base_color = TARGET_BASE if is_target else TIER_COLOR
        label = f"{emb}" + (" (TARGET)" if is_target else "")
        common = dict(
            x=coords[eidx, 0], y=coords[eidx, 1], z=coords[eidx, 2],
            mode="lines+markers",
            text=[hover_text[i] for i in eidx],
            hovertemplate="%{text}<extra></extra>",
        )
        # GREEN copy (highlighted)
        fig.add_trace(go.Scatter3d(
            **common,
            line=dict(color=GREEN, width=7),
            marker=dict(size=4, color=GREEN, line=dict(width=0.5, color="darkgreen")),
            name=f"{label} [green]",
            legendgroup=emb, showlegend=False, visible=False,
        ))
        green_trace_specs.append((len(fig.data) - 1, emb))
        # BASE-color copy
        fig.add_trace(go.Scatter3d(
            **common,
            line=dict(color=base_color, width=3),
            marker=dict(size=3, color=base_color),
            name=f"{label}",
            legendgroup=emb, showlegend=False, visible=False,
        ))
        base_trace_specs.append((len(fig.data) - 1, emb))

    for emb in relevant_embryos:
        add_embryo_pair(emb)

    # ── Reference overlays: tier rings + target star (always visible) ─────────
    fig.add_trace(go.Scatter3d(
        x=coords[tier2_idx, 0], y=coords[tier2_idx, 1], z=coords[tier2_idx, 2],
        mode="markers",
        marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(width=2, color="#1b9e77"), symbol="circle"),
        name=f"top-{TIER2_N} NN rows (ring)",
        text=[hover_text[i] for i in tier2_idx],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=coords[tier1_idx, 0], y=coords[tier1_idx, 1], z=coords[tier1_idx, 2],
        mode="markers",
        marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=3, color="#d95f02"), symbol="circle"),
        name=f"top-{TIER1_N} NN rows (ring)",
        text=[hover_text[i] for i in tier1_idx],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=coords[[target_idx], 0], y=coords[[target_idx], 1], z=coords[[target_idx], 2],
        mode="markers",
        marker=dict(size=14, color="gold", symbol="diamond", line=dict(width=2, color="black")),
        name=f"TARGET {TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf",
        text=[hover_text[target_idx]],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
    ))

    total_traces = len(fig.data)

    # ── Build per-dropdown visibility arrays ──────────────────────────────────
    # Every trace is visible EXCEPT the per-embryo green/base pairs, which we
    # switch: for a given highlight set, that set's embryos show GREEN copy;
    # all OTHER relevant embryos show BASE copy. Bulk + rings + star always on.
    green_idx_by_emb = {emb: ti for ti, emb in green_trace_specs}
    base_idx_by_emb = {emb: ti for ti, emb in base_trace_specs}
    pair_trace_indices = set(green_idx_by_emb.values()) | set(base_idx_by_emb.values())

    def visibility_for(highlight_embryos):
        vis = [True] * total_traces
        # start: hide all pair traces, then selectively turn on green vs base
        for ti in pair_trace_indices:
            vis[ti] = False
        hl = set(highlight_embryos)
        for emb in relevant_embryos:
            if emb in hl:
                vis[green_idx_by_emb[emb]] = True   # green copy on
            else:
                vis[base_idx_by_emb[emb]] = True    # base-color copy on
        return vis

    vis_target = visibility_for(target_embryos)
    vis_top5 = visibility_for(top5_embryos)
    vis_top15 = visibility_for(top15_embryos)

    # apply the default (Target) state to the actual traces
    for ti, v in enumerate(vis_target):
        fig.data[ti].visible = v

    def _title(active):
        return (
            f"<b>Outlier trace + neighborhood: {TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf, raw 80-dim z_mu_b space</b><br>"
            f"<sup>ONE PCA fit, all {n_rows} rows. Dropdown highlights a set of embryos' FULL trajectories in "
            f"GREEN over the shared scene. Bulk = gray points; other neighbor embryos = orange traces; target base "
            f"= purple. Rings = NN rows, gold star = target 72hpf. NN dist = {dists[neighbor_order[0]]:.2f} vs. "
            f"global mean pairwise = {global_mean_dist:.2f} ({dists[neighbor_order[0]]/global_mean_dist:.2f}x). "
            f"PCA var. explained (3 PCs): {var_explained.sum():.1%}. "
            f"[active highlight: {active}]</sup>"
        )

    fig.update_layout(
        title=_title(f"Target ({len(target_embryos)} embryo)"),
        scene=dict(
            xaxis_title=f"PC1 ({var_explained[0]:.1%})",
            yaxis_title=f"PC2 ({var_explained[1]:.1%})",
            zaxis_title=f"PC3 ({var_explained[2]:.1%})",
        ),
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        height=900,
        margin=dict(l=0, r=0, t=120, b=0),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.01, y=1.0, xanchor="left", yanchor="top",
                buttons=[
                    dict(
                        label=f"Target ({len(target_embryos)} embryo)",
                        method="update",
                        args=[{"visible": vis_target},
                              {"title": _title(f"Target ({len(target_embryos)} embryo)")}],
                    ),
                    dict(
                        label=f"Top 5 ({len(top5_embryos)} embryos)",
                        method="update",
                        args=[{"visible": vis_top5},
                              {"title": _title(f"Top 5 ({len(top5_embryos)} embryos)")}],
                    ),
                    dict(
                        label=f"Top 15 ({len(top15_embryos)} embryos)",
                        method="update",
                        args=[{"visible": vis_top15},
                              {"title": _title(f"Top 15 ({len(top15_embryos)} embryos)")}],
                    ),
                ],
            )
        ],
    )

    out = FIGURES / "outlier_trace_neighbors.html"
    fig.write_html(str(out), include_plotlyjs="inline")
    print(f"\nSaved -> {out}")
    print(f"Total traces: {total_traces} (1 bulk + {len(relevant_embryos)}x2 embryo pairs + 3 overlays)")


if __name__ == "__main__":
    main()
