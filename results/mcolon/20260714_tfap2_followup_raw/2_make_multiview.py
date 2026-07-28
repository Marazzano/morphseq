"""
2_make_multiview.py [output_html]
---------------------------------
Render the raw-latent tfap2 condensation as ONE multi-view time-slice HTML — the
primary deliverable. A "Colour by:" dropdown switches between:
  - genotype   (16 tfap2 genotypes; build_genotype_color_lookup)
  - experiment (5 experiments; from the per-embryo `experiments` array in the npz)
  - z_mu_b_<suffix> continuous views (true per-(embryo,time) value, global colorbar)

Adapted from 20260703_raw_latent_clustering_baseline/make_multiview.py, but reads
the experiment label from the npz `experiments` array (carried through by
0_load / 1_condense) rather than parsing it out of embryo_id prefixes — tfap2
embryo_ids do not cleanly encode the experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

import analyze.trajectory_condensation as tc  # noqa: E402
from analyze.viz.styling.color_utils import build_genotype_color_lookup  # noqa: E402

from common import DIM_SUFFIXES, EXPERIMENT_COLOR_MAP  # noqa: E402

TABLES = _HERE / "tables"
RUN_DIR = _HERE / "figures" / "condensed_raw_tfap2"


def _pivot_field_tensor(binned: pd.DataFrame, z_cols: list[str], embryo_ids: np.ndarray):
    """Build feature tensor (N_e,T,K) aligned to the run's embryo order.

    Returns (feat_aligned, col_of) where col_of maps a z_mu_b suffix (int) to its
    array index in z_cols.
    """
    feids = np.array(sorted(binned["embryo_id"].unique()))
    time_values = np.array(sorted(binned["time_bin"].unique()), dtype=float)
    N_e, T, K = len(feids), len(time_values), len(z_cols)
    eid_idx = {e: i for i, e in enumerate(feids)}
    t_idx = {t: i for i, t in enumerate(time_values)}

    feat = np.full((N_e, T, K), np.nan)
    for _, row in binned.iterrows():
        feat[eid_idx[row["embryo_id"]], t_idx[float(row["time_bin"])], :] = (
            row[z_cols].values.astype(float)
        )

    fidx = {str(e): i for i, e in enumerate(feids)}
    feat = feat[np.array([fidx[str(e)] for e in embryo_ids])]  # align to run order

    col_of = {int(c.split("z_mu_b_")[1].split("_")[0]): j for j, c in enumerate(z_cols)}
    return feat, col_of


def main(out_html: Path, dim_suffixes=DIM_SUFFIXES) -> None:
    npz_path = RUN_DIR / "condensed_positions.npz"
    d = np.load(npz_path, allow_pickle=True)
    pos, mask, tv = d["positions"], d["mask"], d["time_values"]
    ids, lab, exp = d["embryo_ids"], d["labels"], d["experiments"]

    # Continuous z_mu_b fields, aligned to run order.
    binned = pd.read_csv(TABLES / "tfap2_binned_zmub.csv", low_memory=False)
    zc = [c for c in binned.columns if "z_mu_b" in c]
    feat, col_of = _pivot_field_tensor(binned, zc, ids)

    genotype_cmap = build_genotype_color_lookup(sorted(set(lab.tolist())))

    views = [
        {"name": "genotype", "labels": lab, "color_map": genotype_cmap},
        {"name": "experiment", "labels": exp, "color_map": EXPERIMENT_COLOR_MAP},
    ]
    for suf in dim_suffixes:
        if suf not in col_of:
            print(f"WARN: z_mu_b_{suf} not found; skipping")
            continue
        views.append({
            "name": f"z_mu_b_{suf}",
            "field": feat[:, :, col_of[suf]],
            "colorscale": "RdBu_r",
            "clip_pct": 2,
        })

    tc.time_slice_html(
        pos, mask, tv, embryo_ids=ids, views=views,
        title="raw z_mu_b tfap2 condensation — 16 genotypes",
        output_path=out_html,
    )
    print(f"Saved multi-view viewer -> {out_html}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN_DIR / "multiview_time_slice.html"
    main(out)
