"""
37_pbx_direction_fingerprints_zscored.py
-----------------------------------------
PBX analogue of the b9d2 z-normalized fingerprint
(20260715_b9d2_direction_decomposition/8_direction_fingerprints_zscored.py).

3 x 2 fingerprint (rows = pbx1b / pbx4 / double crispant, each vs inj_ctrl;
LEFT = z-normed embedding difference from inj_ctrl, RIGHT = signed classifier
coefficient on z-normed inputs; dims on Y natural order, time on X), on
WT(inj_ctrl)-z-normalized coordinates, with a shared color scale PER COLUMN
across the 3 rows so magnitude is comparable between crispants. NOT ablated.

inj_ctrl is the reference ("WT") for both the z-score and the difference.

Output: figures/pbx_direction_fingerprints_zscored_3x2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from analyze.classification.directions.extract import extract_classifier_directions

import importlib.util
_spec34 = importlib.util.spec_from_file_location("_z", _HERE / "34_zscore_classifier_select_clustering.py")
_z = importlib.util.module_from_spec(_spec34); _spec34.loader.exec_module(_z)

TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"
BIN_WIDTH = 4.0
CONTROL = "inj_ctrl"
CRISPANTS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
CRISP_TITLE = {"pbx1b_crispant": "pbx1b vs inj_ctrl",
               "pbx4_crispant": "pbx4 vs inj_ctrl",
               "pbx1b_pbx4_crispant": "double vs inj_ctrl"}


def dim_short(d):
    return d.replace("z_mu_b_", "").replace("_binned", "")


def signed_coef_long(bz, z_cols, crispant):
    df = bz[bz.genotype.isin([crispant, CONTROL])].copy()
    df["grp"] = np.where(df.genotype == CONTROL, "WT", "GRP")
    directions = extract_classifier_directions(
        df, class_col="grp", id_col="embryo_id", time_col="time_bin",
        comparisons=[{"positive": "GRP", "negative": "WT"}],
        features={"emb": z_cols}, bin_width=BIN_WIDTH,
        min_samples_per_group=2, min_samples_per_member=2, verbose=False,
    )
    rows = []
    for _, r in directions.metadata.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        tb = float(r["time_bin_center"]) - BIN_WIDTH / 2.0
        for name, w in zip(names, vec):
            rows.append(dict(crispant=crispant, time_bin=tb, dim=name, signed_coef=float(w)))
    return pd.DataFrame(rows)


def embedding_diff_long(bz, z_cols, crispant):
    ctrl = bz[bz.genotype == CONTROL]
    grp = bz[bz.genotype == crispant]
    rows = []
    for tb in sorted(set(grp.time_bin) & set(ctrl.time_bin)):
        g, w = grp[grp.time_bin == tb], ctrl[ctrl.time_bin == tb]
        if g.embryo_id.nunique() < 2 or w.embryo_id.nunique() < 2:
            continue
        gm = g.groupby("embryo_id")[z_cols].mean().mean()
        wm = w.groupby("embryo_id")[z_cols].mean().mean()
        for name in z_cols:
            rows.append(dict(crispant=crispant, time_bin=float(tb), dim=name,
                             embedding_diff=float(gm[name] - wm[name])))
    return pd.DataFrame(rows)


def _panel(ax, long_df, value_col, z_cols, time_bins, vmax, *, title, cbar_label, fig):
    piv = long_df.pivot(index="dim", columns="time_bin", values=value_col).reindex(
        index=z_cols, columns=time_bins)
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(z_cols)))
    ax.set_yticklabels([dim_short(c) for c in z_cols], fontsize=4.3)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = sorted([c for c in binned.columns if "z_mu_b" in c],
                    key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")))
    bz = _z.wt_zscore(binned, z_cols)
    print(f"z-normalized {len(bz)} rows, {len(z_cols)} dims")

    coef_all = pd.concat([signed_coef_long(bz, z_cols, g) for g in CRISPANTS], ignore_index=True)
    diff_all = pd.concat([embedding_diff_long(bz, z_cols, g) for g in CRISPANTS], ignore_index=True)

    time_bins = sorted(set(coef_all.time_bin) | set(diff_all.time_bin))
    diff_vmax = float(np.nanmax(np.abs(diff_all.embedding_diff.values)))
    coef_vmax = float(np.nanmax(np.abs(coef_all.signed_coef.values)))

    fig, axes = plt.subplots(len(CRISPANTS), 2, figsize=(22, 6.4 * len(CRISPANTS)),
                             gridspec_kw=dict(wspace=0.22))
    for r, crispant in enumerate(CRISPANTS):
        ax_diff, ax_coef = axes[r, 0], axes[r, 1]
        _panel(ax_diff, diff_all[diff_all.crispant == crispant], "embedding_diff",
               z_cols, time_bins, diff_vmax, fig=fig,
               title=f"{CRISP_TITLE[crispant]} — z-normed embedding difference (WT-noise units)",
               cbar_label="mean(GRP)-mean(ctrl), z-units\n(shared scale, all rows)")
        _panel(ax_coef, coef_all[coef_all.crispant == crispant], "signed_coef",
               z_cols, time_bins, coef_vmax, fig=fig,
               title=f"{CRISP_TITLE[crispant]} — classifier signed coef (z-normed inputs)",
               cbar_label="signed logistic coef\n(shared scale, all rows)")
        ax_diff.set_ylabel("z_mu_b dim (natural index order)", fontsize=10)
        for ax in (ax_diff, ax_coef):
            ax.set_xticks(range(len(time_bins)))
            ax.set_xticklabels([f"{int(t)}" for t in time_bins], fontsize=6.5, rotation=90)
        if r == len(CRISPANTS) - 1:
            ax_diff.set_xlabel("time_bin (hpf)"); ax_coef.set_xlabel("time_bin (hpf)")

    fig.suptitle(
        "PBX crispant phenotype fingerprints on Z-NORMALIZED coords (not ablated) — shared scale per column\n"
        "LEFT: z-normed embedding difference from inj_ctrl   |   RIGHT: classifier signed coef. "
        "Magnitude comparable across pbx1b/pbx4/double within each column.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "pbx_direction_fingerprints_zscored_3x2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
