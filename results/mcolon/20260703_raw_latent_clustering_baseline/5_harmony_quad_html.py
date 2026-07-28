"""
5_harmony_quad_html.py
------------------------
Visual verification of the Harmony-corrected condensation. Same format as
2_triptych_html.py (self-contained inline Plotly, one 3D scatter per panel,
colored by genotype, one trace per genotype so the legend toggles cleanly),
extended to 4 panels so the Harmony-corrected arm can be checked by eye
against the raw and margin arms:

  1. raw z_mu_b            — condensed (pre-correction, shows the "columns")
  2. harmony_theta2         — condensed (recommended theta from diagnostic)
  3. margin baseline        — condensed (existing PBX 20260407 run)
  4. harmony_theta2         — UMAP init (pre-condensation), for reference

Output:
  figures/harmony_quad.html
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
HARMONY_NPZ = FIGURES / "condensed_harmony_theta2" / "condensed_positions.npz"
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


def _add_panel(fig, row, col, positions, mask, time_values, labels, embryo_ids, show_legend):
    N, T, _ = positions.shape
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
            row=row,
            col=col,
        )


def main() -> None:
    raw = np.load(RAW_NPZ, allow_pickle=True)
    harmony = np.load(HARMONY_NPZ, allow_pickle=True)
    margin = np.load(MARGIN_NPZ, allow_pickle=True)

    panels = [
        ("raw z_mu_b — condensed (pre-correction)",
         raw["positions"], raw["mask"], raw["time_values"], raw["labels"], raw["embryo_ids"]),
        ("harmony theta=2 — condensed (recommended)",
         harmony["positions"], harmony["mask"], harmony["time_values"], harmony["labels"], harmony["embryo_ids"]),
        ("PBX margin baseline — condensed",
         margin["positions"], margin["mask"], margin["time_values"], margin["labels"], margin["embryo_ids"]),
        ("harmony theta=2 — UMAP init (pre-condensation)",
         harmony["x0"], harmony["mask"], harmony["time_values"], harmony["labels"], harmony["embryo_ids"]),
    ]

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "scene"}] * 4],
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.02,
    )

    for i, (_, pos, mask, tv, labels, eids) in enumerate(panels, start=1):
        _add_panel(fig, 1, i, pos, mask, tv, labels, eids, show_legend=(i == 1))

    scene = dict(
        xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="hpf",
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
    )
    fig.update_layout(
        title="<b>Harmony-corrected condensation vs. raw and margin</b> — "
              "same solver, different UMAP input (raw z_mu_b / harmony-corrected z_mu_b / margin baseline)",
        template="plotly_white",
        scene=scene, scene2=scene, scene3=scene, scene4=scene,
        legend=dict(title="genotype", itemsizing="constant"),
        height=650,
        margin=dict(l=0, r=0, t=70, b=0),
    )

    out = FIGURES / "harmony_quad.html"
    fig.write_html(str(out), include_plotlyjs="inline")
    print(f"Wrote self-contained quad view -> {out}")


if __name__ == "__main__":
    main()
