"""
make_multiview.py <positions_npz> <output_html> [title]
------------------------------------------------------
Render a condensed run as ONE multi-view time-slice HTML: dropdown to colour by
experiment / genotype / top-variance z_mu_b dims (true per-timepoint value on a
global colorbar). Reusable for any run's positions npz.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/net/trapnell/vol1/home/mdcolon/proj/morphseq/src")
import importlib.util
import numpy as np
import pandas as pd
import analyze.trajectory_condensation as tc

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_b", _HERE / "1_cluster_raw_latent.py")
_b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_b)

EXP = {"20251207_pbx": "#6A3D9A", "20260304": "#1B9E77", "20260306": "#D95F02"}
GEN = {"inj_ctrl": "#2166AC", "wik_ab": "#808080", "pbx1b_crispant": "#9467bd",
       "pbx4_crispant": "#F7B267", "pbx1b_pbx4_crispant": "#B2182B"}


def _exp(e):
    p = e.split("_"); return "_".join(p[:2]) if len(p) > 1 and p[1].isalpha() else p[0]


# Requested z_mu_b dims by COLUMN SUFFIX (the latent-dim number, as in
# 18_dim_expression_slices.py): 33 (suppressor), 71 (genotype gradient), 85.
# NB the suffix is NOT the array index: columns run z_mu_b_20..z_mu_b_99.
DIM_SUFFIXES = [33, 71, 85]


def main(npz_path, out_html, title, dim_suffixes=DIM_SUFFIXES):
    d = np.load(npz_path, allow_pickle=True)
    pos, mask, tv, ids, lab = d["positions"], d["mask"], d["time_values"], d["embryo_ids"], d["labels"]

    binned = pd.read_csv(_b.TABLES / "pbx_binned_zmub.csv", low_memory=False)
    # full ordered list of z_mu_b columns; map suffix -> array index
    zc = [c for c in binned.columns if "z_mu_b" in c]
    col_of = {int(c.split("z_mu_b_")[1].split("_")[0]): j for j, c in enumerate(zc)}
    feat, fmask, feids, ftv, flab = _b._pivot_to_tensor(binned, zc)
    fidx = {str(e): i for i, e in enumerate(feids)}
    feat = feat[np.array([fidx[str(e)] for e in ids])]  # align to run order

    exp = np.array([_exp(str(e)) for e in ids])
    views = [{"name": "experiment", "labels": exp, "color_map": EXP},
             {"name": "genotype", "labels": lab, "color_map": GEN}]
    for suf in dim_suffixes:
        if suf not in col_of:
            print(f"WARN: z_mu_b_{suf} not found; skipping")
            continue
        views.append({"name": f"z_mu_b_{suf}", "field": feat[:, :, col_of[suf]],
                      "colorscale": "RdBu_r", "clip_pct": 2})  # coolwarm-like, as 18_

    tc.time_slice_html(pos, mask, tv, embryo_ids=ids, views=views,
                       title=title, output_path=out_html)


if __name__ == "__main__":
    npz, out = sys.argv[1], sys.argv[2]
    ttl = sys.argv[3] if len(sys.argv) > 3 else Path(out).stem
    main(npz, out, ttl)
