"""
28_raw_overlap_and_3d_umap.py
-----------------------------
Two raw-z_mu_b diagnostics:

(A) Drop 20251207 entirely; embed ONLY 20260304 + 20260306 from raw z_mu_b and
    measure cross-batch mixing per bin. Question: do ANY two raw experiments
    overlap, or is the raw data fundamentally too different regardless of which
    batch is present? (20260304 enters 20hpf, 20260306 enters 72hpf, so they only
    co-occur around 72-76hpf -- that narrow window is the honest test.)

(B) One 3D UMAP over every observed (embryo,time) raw z_mu_b vector, rendered as a
    single Plotly HTML with a dropdown to recolor by experiment / genotype / time.

Outputs (figures/raw_overlap_3d/):
  drop_20251207_mixing.txt
  whole_embryo_3d_umap.html
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE / "numba"))
for _d in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
    Path(os.environ[_d]).mkdir(parents=True, exist_ok=True)

import importlib.util
import numpy as np
import pandas as pd
import plotly.graph_objects as go

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

_spec = importlib.util.spec_from_file_location("_baseline", _HERE / "1_cluster_raw_latent.py")
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
TABLES = _baseline.TABLES
RANDOM_STATE = _baseline.RANDOM_STATE

OUT_DIR = _HERE / "figures" / "raw_overlap_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_COLORS = {"20251207": "#6A3D9A", "20260304": "#1B9E77", "20260306": "#D95F02"}
GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC", "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd", "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def _load():
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z = [c for c in binned.columns if "z_mu_b" in c]
    feat, mask, eids, tv, lab = _baseline._pivot_to_tensor(binned, z)
    batch = np.array([str(e).split("_")[0] for e in eids], dtype=object)
    return feat, mask, eids, tv, lab, batch, z


def cross_batch_overlap_feat(X, b, k=10):
    """Mean fraction of each point's k-NN (in FEATURE space) that are a different
    batch. X: (n,K) observed feature rows for one bin; b: batch per row."""
    if X.shape[0] < k + 2 or len(set(b.tolist())) < 2:
        return (0.0 if len(set(b.tolist())) < 2 else float("nan")), X.shape[0]
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    fracs = [np.mean(b[np.argsort(d[i])[:k]] != b[i]) for i in range(X.shape[0])]
    return float(np.mean(fracs)), X.shape[0]


def part_a_drop_20251207(feat, mask, tv, batch):
    """Feature-space cross-batch mixing per bin, using ONLY 20260304 + 20260306."""
    keep = np.isin(batch, ["20260304", "20260306"])
    lines = ["(A) Raw z_mu_b FEATURE-space cross-batch mixing, 20251207 DROPPED",
             "    (only 20260304 + 20260306; they co-occur ~72-76hpf)", "",
             " t   hpf   n   batches_present         feat_mix_10nn"]
    T = mask.shape[1]
    for t in range(T):
        rows = np.flatnonzero(mask[:, t] & keep)
        if rows.size == 0:
            continue
        bp = sorted(set(batch[rows].tolist()))
        X = feat[rows, t, :]
        finite = ~np.isnan(X).any(axis=1)
        X, b = X[finite], batch[rows][finite]
        mix, n = cross_batch_overlap_feat(X, b)
        tag = "  <- BOTH present" if len(bp) > 1 else ""
        lines.append(f"{t:2d}  {tv[t]:4.0f}  {n:3d}  {str(bp):22s}  {mix:5.2f}{tag}")
    report = "\n".join(lines)
    (OUT_DIR / "drop_20251207_mixing.txt").write_text(report)
    print(report)


def _observed_points(feat, mask):
    ei, ti = np.nonzero(mask)
    X = feat[ei, ti, :]
    finite = ~np.isnan(X).any(axis=1)
    return ei[finite], ti[finite], X[finite]


def _render_3d(emb, ei, ti, eids, tv, lab, batch, *, title, axis_prefix, out_path):
    """Render a 3D embedding as one Plotly HTML with an experiment/genotype/time menu."""
    exp = batch[ei]
    gen = np.array([str(lab[e]) for e in ei])
    time_hpf = tv[ti]
    hover = [f"{eids[e]}<br>{tv[t]:.0f}hpf<br>{lab[e]}" for e, t in zip(ei, ti)]

    def disc_traces(cats, cmap):
        traces = []
        for c in sorted(set(cats.tolist())):
            m = cats == c
            traces.append(go.Scatter3d(
                x=emb[m, 0], y=emb[m, 1], z=emb[m, 2], mode="markers",
                marker=dict(size=2.5, color=cmap.get(c, "#333333"), opacity=0.7),
                name=str(c), text=[hover[i] for i in np.flatnonzero(m)],
                hoverinfo="text", visible=True))
        return traces

    exp_traces = disc_traces(exp, EXPERIMENT_COLORS)
    gen_traces = disc_traces(gen, GENOTYPE_COLORS)
    time_trace = [go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2], mode="markers",
        marker=dict(size=2.5, color=time_hpf, colorscale="Viridis", opacity=0.7,
                    colorbar=dict(title="hpf")),
        name="time", text=hover, hoverinfo="text", visible=False)]

    all_traces = exp_traces + gen_traces + time_trace
    n_exp, n_gen = len(exp_traces), len(gen_traces)
    n_total = len(all_traces)

    def vis(start, count):
        v = [False] * n_total
        for i in range(start, start + count):
            v[i] = True
        return v

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=f"{axis_prefix}1", yaxis_title=f"{axis_prefix}2",
                   zaxis_title=f"{axis_prefix}3"),
        updatemenus=[dict(
            buttons=[
                dict(label="experiment", method="update", args=[{"visible": vis(0, n_exp)}]),
                dict(label="genotype", method="update", args=[{"visible": vis(n_exp, n_gen)}]),
                dict(label="time (hpf)", method="update",
                     args=[{"visible": vis(n_exp + n_gen, 1)}]),
            ],
            direction="down", showactive=True, x=0.02, xanchor="left", y=0.99, yanchor="top",
        )],
    )
    fig.write_html(str(out_path))
    print(f"Saved -> {out_path}")


def part_b_3d_umap(feat, mask, eids, tv, lab, batch):
    import umap
    ei, ti, X = _observed_points(feat, mask)
    print(f"3D UMAP on {X.shape[0]} points x {X.shape[1]} dims...")
    emb = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                    random_state=RANDOM_STATE).fit_transform(X)
    _render_3d(emb, ei, ti, eids, tv, lab, batch,
               title="Whole-embryo 3D UMAP on raw z_mu_b (color: menu)",
               axis_prefix="UMAP", out_path=OUT_DIR / "whole_embryo_3d_umap.html")


def part_c_3d_pca(feat, mask, eids, tv, lab, batch):
    """Linear PCA top-3 of raw z_mu_b. Unlike UMAP, a batch offset appears as literal
    displacement at true scale, so this tests whether 20251207's offset is a dominant
    LINEAR direction (=> strongly correctable) vs a small local one UMAP exaggerates."""
    from sklearn.decomposition import PCA
    ei, ti, X = _observed_points(feat, mask)
    pca = PCA(n_components=min(10, X.shape[1]), random_state=RANDOM_STATE).fit(X)
    emb = pca.transform(X)[:, :3]
    evr = pca.explained_variance_ratio_
    print(f"3D PCA on {X.shape[0]} points x {X.shape[1]} dims")
    print(f"  top-3 explained var: {evr[:3].round(3)}  (cum {evr[:3].sum():.3f}); "
          f"top-10 cum {evr[:10].sum():.3f}")

    # Is a batch axis among the top PCs? Correlate each of PC1-3 with an indicator of
    # 20251207 vs rest, and report per-batch centroid on PC1-3 (separation = offset).
    exp = batch[ei]
    is_1207 = (exp == "20251207").astype(float)
    lines = ["(C) PCA top-3 of raw z_mu_b: is 20251207's offset a dominant LINEAR axis?",
             f"  explained var PC1-3: {evr[:3].round(3).tolist()} (cum {evr[:3].sum():.3f})",
             "", "  per-batch centroid on PC1 / PC2 / PC3:"]
    for b in sorted(set(exp.tolist())):
        c = emb[exp == b].mean(axis=0)
        lines.append(f"    {b}: [{c[0]:7.2f} {c[1]:7.2f} {c[2]:7.2f}]  (n={int((exp==b).sum())})")
    lines.append("")
    lines.append("  |corr(PC_k, 20251207-indicator)|  (high => that PC IS ~a batch axis):")
    for k in range(3):
        r = abs(np.corrcoef(emb[:, k], is_1207)[0, 1])
        lines.append(f"    PC{k+1}: {r:.3f}")
    report = "\n".join(lines)
    (OUT_DIR / "pca3_batch_axis.txt").write_text(report)
    print(report)

    _render_3d(emb, ei, ti, eids, tv, lab, batch,
               title="Whole-embryo raw z_mu_b PCA top-3 (color: menu)",
               axis_prefix="PC", out_path=OUT_DIR / "whole_embryo_3d_pca.html")


def main():
    feat, mask, eids, tv, lab, batch, z = _load()
    print(f"Tensor {feat.shape}, batches={sorted(set(batch))}")
    part_a_drop_20251207(feat, mask, tv, batch)
    print()
    part_b_3d_umap(feat, mask, eids, tv, lab, batch)
    print()
    part_c_3d_pca(feat, mask, eids, tv, lab, batch)


if __name__ == "__main__":
    main()
