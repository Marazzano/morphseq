"""
10_unknown_cluster_shuffle.py
-----------------------------
Discovery stress-test: does cluster coherence survive when the two-cluster
IDENTITY is unknown / unstable across time slices?

In real phenotype discovery you find two modes per time slice but you don't know
which mode at bin t corresponds to which mode at bin t+1 -- the cross-time
identity is unknown. This tests what happens to direction-projection clustering
under that instability.

Reference already exists (script 9):
  B_split_2dir  : true CE/HTA labels, stable -> resolves cleanly (the encouraging case)
  A_pooled_1dir : one pooled direction -> severity smear, CE/HTA overlap

Two NEW conditions here (fit A-vs-WT + B-vs-WT per bin, project -> 2-D, condense):
  C_identity_flip : take the REAL CE/HTA membership as the two discovered modes,
                    but randomly SWAP which is 'A' vs 'B' INDEPENDENTLY at each
                    time bin. The split is real; the A/B naming is inconsistent
                    over time. -> does temporal identity-instability scramble it?
  D_random_null   : ignore CE/HTA; randomly assign each phenotype embryo to A or B
                    (per time bin). A meaningless split. -> null floor.

Multiview colored by TRUE phenotype_clean {CE, HTA, wildtype} so you can SEE
whether the discovered clusters track real biology.

Outputs (figures/b9d2_projection/{C_identity_flip,D_random_null}/).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE = Path("/tmp") / "morphseq_20260715_b9d2shuf_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE / "numba"))
for _d in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
    Path(os.environ[_d]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

import importlib.util
_spec = importlib.util.spec_from_file_location("_m9", _HERE / "9_direction_projection_clustering.py")
_m9 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m9)

OUT_ROOT = _HERE / "figures" / "b9d2_projection"
SEED = 0


def assign_ab_identity_flip(dfz, rng):
    """Real CE/HTA modes, but which is 'A' vs 'B' is randomly swapped PER TIME BIN.
    Returns a per-row 'grp' column in {A, B, WT}."""
    grp = pd.Series("WT", index=dfz.index, dtype=object)
    is_ce = dfz["pheno"] == "CE"
    is_hta = dfz["pheno"] == "HTA"
    for tb, idx in dfz.groupby("time_bin").groups.items():
        flip = rng.random() < 0.5  # per-bin coin flip: does CE map to A or B?
        sub = dfz.loc[idx]
        ce_idx = sub.index[sub["pheno"] == "CE"]
        hta_idx = sub.index[sub["pheno"] == "HTA"]
        if flip:
            grp.loc[ce_idx] = "A"; grp.loc[hta_idx] = "B"
        else:
            grp.loc[ce_idx] = "B"; grp.loc[hta_idx] = "A"
    return grp


def assign_ab_random_null(dfz, rng):
    """Ignore CE/HTA: each phenotype embryo randomly A or B, decided PER TIME BIN.
    Returns per-row 'grp' in {A, B, WT}."""
    grp = pd.Series("WT", index=dfz.index, dtype=object)
    pheno_mask = dfz["pheno"].isin(["CE", "HTA"])
    # decide per (embryo, bin) so grain matches the per-bin flip
    for (e, tb), idx in dfz[pheno_mask].groupby(["embryo_id", "time_bin"]).groups.items():
        grp.loc[idx] = "A" if rng.random() < 0.5 else "B"
    return grp


def run_condition(dfz, z_cols, grp_col, cond_name, title):
    df = dfz.copy()
    df["grp"] = grp_col
    vecs = _m9.fit_dirs(df, z_cols, [{"positive": "A", "negative": "WT"},
                                     {"positive": "B", "negative": "WT"}])
    comp_ids = ["A__vs__WT", "B__vs__WT"]
    feats, mask, eids, tvals, labels = _m9.build_features(dfz, z_cols, vecs, comp_ids)
    print(f"[{cond_name}] features {feats.shape}, mask cov {mask.mean():.1%}")
    out_dir = OUT_ROOT / cond_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _m9.save_scores(feats, mask, eids, tvals, labels, comp_ids, out_dir)
    _m9.condense_and_render(feats, mask, eids, tvals, labels, out_dir, title)


def main():
    dfz, z_cols = _m9.load_and_zscore()
    rng = np.random.default_rng(SEED)
    print(f"z-scored {len(dfz)} frames; embryos {dfz.embryo_id.nunique()}")

    # C: real 2-cluster split, identity flipped per time bin
    grp_c = assign_ab_identity_flip(dfz, rng)
    run_condition(dfz, z_cols, grp_c, "C_identity_flip",
                  "b9d2 UNKNOWN clusters — real CE/HTA modes, A/B identity flipped PER TIME BIN "
                  "(does coherence survive unstable identity?)")

    # D: fully random A/B partition (null floor)
    grp_d = assign_ab_random_null(dfz, rng)
    run_condition(dfz, z_cols, grp_d, "D_random_null",
                  "b9d2 NULL — fully random A/B partition per bin "
                  "(meaningless split: expect incoherent)")

    print("\nDone. figures/b9d2_projection/{C_identity_flip,D_random_null}/")


if __name__ == "__main__":
    main()
