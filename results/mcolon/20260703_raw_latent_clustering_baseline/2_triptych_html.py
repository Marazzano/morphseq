"""
2_triptych_html.py  ← GATE
--------------------------
Dump the post-condensation coordinates directly into ONE self-contained Plotly
HTML with three side-by-side 3D panels (x, y, time), colored by genotype.

The clustering method is always the SAME: distance/graph over features -> UMAP
init -> trajectory-condensation dynamics. What differs between arms is only what
goes INTO UMAP (raw z_mu_b vs. classifier margins). There is no k-means/Leiden
step anywhere in this pipeline.

Panels:
  1. raw z_mu_b — UMAP init (x0, pre-condensation)
  2. raw z_mu_b — after condensation
  3. margin baseline — after condensation (existing PBX 20260407 run)

Coordinates are embedded inline (include_plotlyjs="inline"), so the file renders
standalone with no iframes and no external assets.

Output:
  figures/triptych.html
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

FIGURES = _HERE / "figures"

RAW_NPZ = FIGURES / "condensed_raw_zmub" / "condensed_positions.npz"
# Baseline = the RAW condensation the final principal-tree run actually consumed.
# Pinned by results/.../principal_tree/raw_5class_n10_mu03/metadata.json -> input_npz.
# (NOT the shrunk/margin variant — that was a wrong path in the first pass.)
MARGIN_NPZ = (
    _REPO
    / "results/mcolon/20260407_pbx_analysis_cont/results/positioning/trajectory"
    / "combined_raw_condensation_5class_bin4_perm500"
    / "condensed_positions.npz"
)

GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd",
    "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}
LABEL_MAP = {
    "inj_ctrl": "inj. ctrl",
    "pbx1b_crispant": "pbx1b",
    "pbx4_crispant": "pbx4",
    "pbx1b_pbx4_crispant": "pbx1b+4",
    "wik_ab": "wik-ab",
}


def _add_panel(fig, col, positions, mask, time_values, labels, embryo_ids, show_legend):
    """Add one 3D scatter panel (x, y, time) colored by genotype, one trace per genotype."""
    N, T, _ = positions.shape
    # time coordinate broadcast to every (embryo, bin)
    tgrid = np.broadcast_to(time_values[None, :], (N, T))
    lgrid = np.broadcast_to(np.asarray(labels)[:, None], (N, T))
    igrid = np.broadcast_to(np.asarray(embryo_ids)[:, None], (N, T))

    for geno, color in GENOTYPE_COLORS.items():
        sel = mask & (lgrid == geno)
        if not sel.any():
            continue
        fig.add_trace(
            go.Scatter3d(
                x=positions[..., 0][sel],
                y=positions[..., 1][sel],
                z=tgrid[sel],
                mode="markers",
                marker=dict(size=2.5, color=color, opacity=0.75),
                name=LABEL_MAP.get(geno, geno),
                legendgroup=geno,
                showlegend=show_legend,
                text=igrid[sel],
                hovertemplate="%{text}<br>hpf=%{z}<extra></extra>",
            ),
            row=1,
            col=col,
        )


def main() -> None:
    raw = np.load(RAW_NPZ, allow_pickle=True)
    margin = np.load(MARGIN_NPZ, allow_pickle=True)

    panels = [
        ("raw z_mu_b — UMAP init (pre-condensation)",
         raw["x0"], raw["mask"], raw["time_values"], raw["labels"], raw["embryo_ids"]),
        ("raw z_mu_b — condensed",
         raw["positions"], raw["mask"], raw["time_values"], raw["labels"], raw["embryo_ids"]),
        ("PBX baseline — condensed (combined_raw 5class, final tree input)",
         margin["positions"], margin["mask"], margin["time_values"], margin["labels"], margin["embryo_ids"]),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.02,
    )

    for i, (_, pos, mask, tv, labels, eids) in enumerate(panels, start=1):
        _add_panel(fig, i, pos, mask, tv, labels, eids, show_legend=(i == 1))

    scene = dict(
        xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="hpf",
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
    )
    fig.update_layout(
        title="<b>PBX raw-latent condensation baseline</b> — "
              "same solver, different UMAP input (raw z_mu_b vs. PBX combined_raw baseline)",
        template="plotly_white",
        scene=scene, scene2=scene, scene3=scene,
        legend=dict(title="genotype", itemsizing="constant"),
        height=650,
        margin=dict(l=0, r=0, t=70, b=0),
    )

    out = FIGURES / "triptych.html"
    fig.write_html(str(out), include_plotlyjs="inline")
    print(f"Wrote self-contained triptych -> {out}")


if __name__ == "__main__":
    main()
