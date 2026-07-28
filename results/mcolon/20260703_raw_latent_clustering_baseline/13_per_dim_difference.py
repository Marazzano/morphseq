"""
13_per_dim_difference.py
-------------------------
HEADLINE QUESTION: does the classifier (see HANDOFF_classifier_weight_investigation.md,
which found z_mu_b_33 and z_mu_b_71 dominate |coefficient| for every crispant
comparison) attend where the RAW z_mu_b latent actually moves over developmental
time, or does it light up on dims (like z_mu_b_33) that show little/no raw
change?

These are VAE latent dims -- raw magnitude is meaningful. This script does NOT
standardize/z-score. It builds two matched heatmap families per crispant
condition, dims in NATURAL INDEX ORDER (not ranked):

  1. RAW DELTA heatmap: mean(raw z_mu_b | crispant) - mean(raw z_mu_b | pooled
     control) per (dim, time_bin). Diverging colormap centered at 0 (+ = toward
     crispant, - = toward control).
  2. CLASSIFIER heatmap: |logistic-regression coefficient| per (dim, time_bin)
     for the same crispant vs inj_ctrl comparison (classifier_weight_long.csv;
     that table was fit vs inj_ctrl only, not the pooled control -- noted
     inline, doesn't change the qualitative read).

Placed side by side, one row per crispant, same dim order (y) and same
time_bin grid (x), so the reader can scan for whether classifier attention
tracks raw delta magnitude, or diverges (e.g. dim 33 hot in classifier panel,
flat in raw-delta panel; dim 71 showing a raw gradient over time).

Secondary/supplementary (kept from the original per-dim-difference-test plan,
NOT the headline): Cohen's d / Wasserstein / bootstrap-CI difference table,
ranked-overlay plot, histogram small-multiples, hot-dims zoom.

Outputs:
  tables/raw_delta_over_time.csv          headline data (long)
  figures/raw_delta_vs_classifier_over_time.png   HEADLINE figure
  tables/per_dim_difference.csv           supplementary (Cohen's d/Wasserstein/boot CI)
  figures/per_dim_difference_ranked.png   supplementary
  figures/per_dim_difference_over_time.png supplementary (Cohen's d heatmap only)
  figures/per_dim_histograms.png          supplementary
  figures/hot_dims_zoom.png               supplementary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
from analyze.classification.directions.extract import extract_classifier_directions

TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"
RAW_WT_CSV = TABLES / "pbx_binned_zmub_with_wt.csv"
CLF_WEIGHT_CSV = TABLES / "classifier_weight_long.csv"

CONTROL_GENOS = ("inj_ctrl", "wik_ab")
CRISPANTS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
GENO_LABEL = {
    "pbx1b_crispant": "pbx1b",
    "pbx4_crispant": "pbx4",
    "pbx1b_pbx4_crispant": "double",
}
HOT_DIMS = ("z_mu_b_33_binned", "z_mu_b_71_binned")
B_BOOT = 1000
RNG_SEED = 0
POOLED_LABEL = "pooled"
# classifier_weight_long.csv was fit with extract_classifier_directions,
# bin_width=4.0, reporting time_bin_center = raw_time_bin + bin_width/2
CLF_BIN_WIDTH = 4.0


# ───────────────────────── data loading ─────────────────────────

def load_data():
    df = pd.read_csv(RAW_WT_CSV, low_memory=False)
    z_cols = sorted(
        [c for c in df.columns if "z_mu_b" in c],
        key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")),
    )
    keep_genos = (*CONTROL_GENOS, *CRISPANTS)
    df = df[df.genotype.isin(keep_genos)].copy()
    df["group"] = np.where(df.genotype.isin(CONTROL_GENOS), "control", "crispant")
    return df, z_cols


def load_classifier_weights():
    clf = pd.read_csv(CLF_WEIGHT_CSV, low_memory=False)
    summary = (
        clf.groupby("dim")["abs_weight"]
        .agg(clf_mean_abs_weight="mean", clf_max_abs_weight="max")
        .reset_index()
    )
    return clf, summary


def compute_signed_classifier_coefs(df: pd.DataFrame, z_cols: list[str]) -> pd.DataFrame:
    """classifier_weight_long.csv only stores |coefficient| (script 10 takes
    abs(w) before saving). To compare SIGN against the raw-delta sign, refit
    extract_classifier_directions() ourselves (same call as script 10:
    positive=crispant, negative=inj_ctrl, bin_width=4.0) and keep the signed
    unit_coef per dim. Per fit.py's sign convention, the whole coefficient
    vector is oriented so dot(unit_coef, pos_mean - neg_mean) >= 0 (positive
    class = crispant here) -- i.e. + = toward crispant on net, matching the
    raw-delta sign convention, though an individual dim's sign is not
    guaranteed to match its own marginal centroid difference (that mismatch
    is exactly the suppressor-variable signature the handoff describes)."""
    comparisons = [{"positive": g, "negative": "inj_ctrl"} for g in CRISPANTS]
    directions = extract_classifier_directions(
        df, class_col="genotype", id_col="embryo_id", time_col="time_bin",
        comparisons=comparisons, features={"emb": z_cols}, bin_width=CLF_BIN_WIDTH,
        verbose=False,
    )
    meta = directions.metadata
    rows = []
    for _, r in meta.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        comp = r["comparison_id"].replace("__vs__inj_ctrl", "__vs__control")
        time_bin = r["time_bin_center"] - CLF_BIN_WIDTH / 2.0
        for name, w in zip(names, vec):
            rows.append(dict(comparison=comp, time_bin=time_bin, dim=name,
                              classifier_signed_coef=float(w)))
    return pd.DataFrame(rows)


# ───────────────────────── HEADLINE: raw delta over time ─────────────────────────

def compute_raw_delta_over_time(df: pd.DataFrame, z_cols: list[str]) -> pd.DataFrame:
    """Long table: comparison, time_bin (raw_time_bin, matches classifier
    time_bin_center - CLF_BIN_WIDTH/2), dim, raw_delta, n_control, n_crispant."""
    rows = []
    time_bins = sorted(df.time_bin.dropna().unique())
    control_df = df[df.group == "control"]

    for crispant in CRISPANTS:
        comp_label = f"{crispant}__vs__control"
        crispant_df = df[df.genotype == crispant]
        for tb in time_bins:
            c_tb = control_df[control_df.time_bin == tb]
            x_tb = crispant_df[crispant_df.time_bin == tb]
            if len(c_tb) == 0 or len(x_tb) == 0:
                continue
            for dim in z_cols:
                c_vals = c_tb[dim].dropna().values
                x_vals = x_tb[dim].dropna().values
                if len(c_vals) == 0 or len(x_vals) == 0:
                    continue
                delta = x_vals.mean() - c_vals.mean()
                rows.append(dict(
                    comparison=comp_label, time_bin=tb, dim=dim,
                    raw_delta=delta, n_control=len(c_vals), n_crispant=len(x_vals),
                ))
        print(f"[{comp_label}] raw delta computed for {len(time_bins)} time bins")

    return pd.DataFrame(rows)


def build_headline_table(raw_delta_df: pd.DataFrame, clf_long: pd.DataFrame,
                          signed_clf_df: pd.DataFrame) -> pd.DataFrame:
    """Join raw_delta (indexed by raw time_bin) to classifier |coef| (indexed
    by time_bin_center = raw_time_bin + CLF_BIN_WIDTH/2) on the nearest match,
    per comparison x dim, plus the re-derived SIGNED coefficient. Both
    classifier tables were fit vs inj_ctrl only (not pooled control) -- kept
    as-is per handoff, noted in column name."""
    clf = clf_long.copy()
    clf["comparison"] = clf["comparison_id"].str.replace("__vs__inj_ctrl", "__vs__control", regex=False)
    clf["time_bin"] = clf["time_bin_center"] - CLF_BIN_WIDTH / 2.0

    merged = raw_delta_df.merge(
        clf[["comparison", "time_bin", "dim", "abs_weight"]],
        on=["comparison", "time_bin", "dim"], how="left",
    ).rename(columns={"abs_weight": "classifier_abs_coef"})

    merged = merged.merge(
        signed_clf_df[["comparison", "time_bin", "dim", "classifier_signed_coef"]],
        on=["comparison", "time_bin", "dim"], how="left",
    )
    return merged


def dim_short(d: str) -> str:
    return d.replace("z_mu_b_", "").replace("_binned", "")


def fig_raw_delta_vs_classifier(headline_df: pd.DataFrame, z_cols: list[str]):
    """Side-by-side raw-delta | classifier-coef heatmaps, one row per crispant,
    dims in NATURAL INDEX ORDER on y, same time_bin grid on x for both panels."""
    time_bins = sorted(headline_df.time_bin.unique())
    n_rows = len(CRISPANTS)

    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 4.6 * n_rows),
                              gridspec_kw=dict(wspace=0.25))

    for row_i, crispant in enumerate(CRISPANTS):
        comp = f"{crispant}__vs__control"
        d = headline_df[headline_df.comparison == comp]

        raw_piv = d.pivot(index="dim", columns="time_bin", values="raw_delta")
        raw_piv = raw_piv.reindex(index=z_cols, columns=time_bins)
        clf_piv = d.pivot(index="dim", columns="time_bin", values="classifier_signed_coef")
        clf_piv = clf_piv.reindex(index=z_cols, columns=time_bins)

        ax_raw, ax_clf = axes[row_i, 0], axes[row_i, 1]

        vmax = np.nanmax(np.abs(raw_piv.values)) if np.isfinite(raw_piv.values).any() else 1.0
        im1 = ax_raw.imshow(raw_piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_raw.set_title(f"{GENO_LABEL[crispant]} vs control -- RAW DELTA "
                          "(crispant mean - control mean)", fontsize=11)
        fig.colorbar(im1, ax=ax_raw, shrink=0.85, label="raw z_mu_b delta\n(+ = toward crispant)")

        clf_vmax = np.nanmax(np.abs(clf_piv.values)) if np.isfinite(clf_piv.values).any() else 1.0
        im2 = ax_clf.imshow(clf_piv.values, aspect="auto", cmap="RdBu_r", vmin=-clf_vmax, vmax=clf_vmax)
        ax_clf.set_title(f"{GENO_LABEL[crispant]} vs inj_ctrl -- CLASSIFIER SIGNED coefficient",
                          fontsize=11)
        fig.colorbar(im2, ax=ax_clf, shrink=0.85,
                     label="signed logistic-regression coef\n(+ = toward crispant, net direction)")

        for ax in (ax_raw, ax_clf):
            ax.set_yticks(range(len(z_cols)))
            ax.set_yticklabels([dim_short(c) for c in z_cols], fontsize=4.3)
            ax.set_xticks(range(len(time_bins)))
            ax.set_xticklabels([f"{int(t)}" for t in time_bins], fontsize=6.5, rotation=90)
            ax.set_xlabel("time_bin (hpf, raw)")
            for i, c in enumerate(z_cols):
                if c in HOT_DIMS:
                    ax.get_yticklabels()[i].set_color("darkorange")
                    ax.get_yticklabels()[i].set_fontweight("bold")
                    ax.get_yticklabels()[i].set_fontsize(6.0)

        ax_raw.set_ylabel("raw z_mu_b dim (natural index order)")

    fig.suptitle(
        "Raw latent delta vs classifier signed coefficient, over developmental time\n"
        "dims kept in natural index order (not ranked); orange labels = classifier-hot dims "
        "z_mu_b_33 / z_mu_b_71 (from HANDOFF investigation); both panels use the SAME "
        "diverging scale convention (+ = toward crispant)\n"
        "Question: does the classifier's sign/magnitude (right) track where raw values actually "
        "move (left), or does dim 33 have nonzero classifier signal while its raw delta is ~0?",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES / "raw_delta_vs_classifier_over_time.png", dpi=130)
    plt.close(fig)


# ───────────────────────── supplementary: difference-test stats ─────────────────────────

def cohens_d(control_vals: np.ndarray, crispant_vals: np.ndarray) -> float:
    n1, n2 = len(control_vals), len(crispant_vals)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = control_vals.var(ddof=1), crispant_vals.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return np.nan
    return (crispant_vals.mean() - control_vals.mean()) / pooled_sd


def signed_wasserstein(control_vals: np.ndarray, crispant_vals: np.ndarray) -> float:
    if len(control_vals) < 2 or len(crispant_vals) < 2:
        return np.nan
    w = stats.wasserstein_distance(control_vals, crispant_vals)
    sign = np.sign(crispant_vals.mean() - control_vals.mean())
    return sign * w if sign != 0 else w


def cluster_bootstrap_ci(sub_df: pd.DataFrame, dim: str, n_boot: int, rng: np.random.Generator):
    control_rows = sub_df[sub_df.group == "control"]
    crispant_rows = sub_df[sub_df.group == "crispant"]
    control_by_embryo = {e: g[dim].dropna().values for e, g in control_rows.groupby("embryo_id")}
    crispant_by_embryo = {e: g[dim].dropna().values for e, g in crispant_rows.groupby("embryo_id")}
    control_ids = np.array(list(control_by_embryo.keys()))
    crispant_ids = np.array(list(crispant_by_embryo.keys()))
    if len(control_ids) < 2 or len(crispant_ids) < 2:
        return np.nan, np.nan

    boot_d = np.empty(n_boot)
    for b in range(n_boot):
        c_pick = rng.choice(control_ids, size=len(control_ids), replace=True)
        x_pick = rng.choice(crispant_ids, size=len(crispant_ids), replace=True)
        c_vals = np.concatenate([control_by_embryo[e] for e in c_pick])
        x_vals = np.concatenate([crispant_by_embryo[e] for e in x_pick])
        boot_d[b] = cohens_d(c_vals, x_vals)

    boot_d = boot_d[np.isfinite(boot_d)]
    if len(boot_d) < n_boot * 0.5:
        return np.nan, np.nan
    lo, hi = np.percentile(boot_d, [2.5, 97.5])
    return lo, hi


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def compute_for_slice(sub: pd.DataFrame, z_cols: list[str], rng: np.random.Generator,
                       comp_label: str, time_bin_label) -> list[dict]:
    rows = []
    control_vals_all = sub[sub.group == "control"]
    crispant_vals_all = sub[sub.group == "crispant"]
    for dim in z_cols:
        c_vals = control_vals_all[dim].dropna().values
        x_vals = crispant_vals_all[dim].dropna().values
        if len(c_vals) < 2 or len(x_vals) < 2:
            continue
        d = cohens_d(c_vals, x_vals)
        wass = signed_wasserstein(c_vals, x_vals)
        lo, hi = cluster_bootstrap_ci(sub, dim, B_BOOT, rng)
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(d):
            se = (hi - lo) / (2 * 1.96)
            p_approx = 2 * stats.norm.sf(abs(d) / se) if se > 0 else (0.0 if d != 0 else 1.0)
        else:
            p_approx = 1.0
        rows.append(dict(
            comparison=comp_label, time_bin=time_bin_label, dim=dim,
            cohens_d=d, wasserstein=wass, boot_ci_lo=lo, boot_ci_hi=hi,
            n_control=len(c_vals), n_crispant=len(x_vals), boot_p_approx=p_approx,
        ))
    return rows


def compute_difference_table_pooled_only(df: pd.DataFrame, z_cols: list[str],
                                          rng: np.random.Generator) -> pd.DataFrame:
    """Supplementary: pooled-across-bins Cohen's d / Wasserstein / bootstrap CI
    only (skips the full per-time-bin bootstrap grid to keep runtime reasonable
    now that the raw-delta-over-time heatmap is the headline)."""
    rows = []
    for crispant in CRISPANTS:
        comp_label = f"{crispant}__vs__control"
        sub_all = df[(df.genotype == crispant) | (df.group == "control")].copy()
        print(f"[{comp_label}] pooled bootstrap: n_control={len(sub_all[sub_all.group=='control'])}, "
              f"n_crispant={len(sub_all[sub_all.group=='crispant'])}")
        rows.extend(compute_for_slice(sub_all, z_cols, rng, comp_label, POOLED_LABEL))

    out = pd.DataFrame(rows)
    out["fdr_significant"] = False
    out["fdr_q"] = np.nan
    for comp, grp in out.groupby("comparison"):
        idx = grp.index
        q = bh_fdr(grp["boot_p_approx"].values)
        out.loc[idx, "fdr_q"] = q
        out.loc[idx, "fdr_significant"] = q < 0.05
    return out.drop(columns=["boot_p_approx"])


def fig_ranked_headline(pooled_df: pd.DataFrame, clf_summary: pd.DataFrame):
    comp = "pbx1b_pbx4_crispant__vs__control"
    d = pooled_df[pooled_df.comparison == comp].copy()
    d = d.merge(clf_summary, on="dim", how="left")
    d["abs_d"] = d.cohens_d.abs()
    d = d.sort_values("abs_d", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)

    fig, ax = plt.subplots(figsize=(11, 16))
    colors = np.where(d.cohens_d >= 0, "#B2182B", "#2166AC")
    y = np.arange(len(d))[::-1]

    ax.barh(y, d.cohens_d, color=colors, alpha=0.85, edgecolor="none", height=0.72, zorder=2)
    lo_err = (d.cohens_d - d.boot_ci_lo).clip(lower=0)
    hi_err = (d.boot_ci_hi - d.cohens_d).clip(lower=0)
    ax.errorbar(d.cohens_d, y, xerr=[lo_err, hi_err], fmt="none",
                ecolor="black", elinewidth=0.7, capsize=1.5, alpha=0.55, zorder=3)
    sig = d.fdr_significant.values
    ax.scatter(np.where(sig, d.cohens_d, np.nan), y, marker="*", color="black", s=40, zorder=4,
               label="BH-FDR sig. (q<0.05)")

    labels = [dim_short(row["dim"]) for _, row in d.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)

    for i, row in d.iterrows():
        if row["dim"] in HOT_DIMS:
            ax.get_yticklabels()[len(d) - 1 - i].set_color("darkorange")
            ax.get_yticklabels()[len(d) - 1 - i].set_fontweight("bold")
            ax.scatter([row["cohens_d"]], [y[i]], s=260, facecolors="none",
                       edgecolors="darkorange", linewidths=2.2, zorder=5)
            ax.annotate(f"rank {row['rank']}/80  |coef|={row['clf_mean_abs_weight']:.2f}",
                        xy=(row["cohens_d"], y[i]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color="darkorange", fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d  (pbx1b+pbx4 double crispant vs pooled control)\n"
                  "positive (red) = toward crispant  |  negative (blue) = toward control")
    ax.set_title(
        "Supplementary: all 80 raw z_mu_b dims ranked by marginal effect size (pooled across bins)\n"
        "orange rings = classifier-hot dims (z_mu_b_33, z_mu_b_71)", fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "per_dim_difference_ranked.png", dpi=140)
    plt.close(fig)
    return d


def fig_over_time_cohend(raw_delta_df: pd.DataFrame, df: pd.DataFrame, z_cols: list[str]):
    """Supplementary Cohen's d heatmap over time (cheap, no bootstrap needed)."""
    rows = []
    control_df = df[df.group == "control"]
    time_bins = sorted(df.time_bin.dropna().unique())
    for crispant in CRISPANTS:
        comp_label = f"{crispant}__vs__control"
        crispant_df = df[df.genotype == crispant]
        for tb in time_bins:
            c_tb = control_df[control_df.time_bin == tb]
            x_tb = crispant_df[crispant_df.time_bin == tb]
            if len(c_tb) == 0 or len(x_tb) == 0:
                continue
            for dim in z_cols:
                c_vals = c_tb[dim].dropna().values
                x_vals = x_tb[dim].dropna().values
                if len(c_vals) < 2 or len(x_vals) < 2:
                    continue
                rows.append(dict(comparison=comp_label, time_bin=tb, dim=dim,
                                  cohens_d=cohens_d(c_vals, x_vals)))
    d_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(len(CRISPANTS), 1, figsize=(13, 4.2 * len(CRISPANTS)), sharex=True)
    for ax, crispant in zip(axes, CRISPANTS):
        comp = f"{crispant}__vs__control"
        d = d_df[d_df.comparison == comp]
        piv = d.pivot(index="dim", columns="time_bin", values="cohens_d")
        piv = piv.reindex(index=z_cols, columns=time_bins)
        vmax = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1.0
        im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(z_cols)))
        ax.set_yticklabels([dim_short(c) for c in z_cols], fontsize=4.5)
        ax.set_xticks(range(len(time_bins)))
        ax.set_xticklabels([f"{int(t)}" for t in time_bins], fontsize=7, rotation=90)
        ax.set_title(f"{GENO_LABEL[crispant]} vs control -- Cohen's d per dim over time_bin (hpf)",
                     fontsize=10.5)
        fig.colorbar(im, ax=ax, label="Cohen's d (red=toward crispant)", shrink=0.8)
        for i, c in enumerate(z_cols):
            if c in HOT_DIMS:
                ax.get_yticklabels()[i].set_color("darkorange")
                ax.get_yticklabels()[i].set_fontweight("bold")

    axes[-1].set_xlabel("time_bin (hpf)")
    fig.suptitle("Supplementary: standardized (Cohen's d) per-dim difference over time\n"
                 "(orange labels = classifier-hot dims z_mu_b_33 / z_mu_b_71)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES / "per_dim_difference_over_time.png", dpi=130)
    plt.close(fig)


def fig_histograms(df: pd.DataFrame, z_cols: list[str]):
    n_dims = len(z_cols)
    ncols = 8
    nrows = int(np.ceil(n_dims / ncols))
    crispant = "pbx1b_pbx4_crispant"

    control_df = df[df.group == "control"]
    crispant_df = df[df.genotype == crispant]

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 1.6))
    axes = axes.ravel()
    for i, dim in enumerate(z_cols):
        ax = axes[i]
        c_vals = control_df[dim].dropna().values
        x_vals = crispant_df[dim].dropna().values
        lo = min(c_vals.min(), x_vals.min())
        hi = max(c_vals.max(), x_vals.max())
        bins = np.linspace(lo, hi, 25)
        ax.hist(c_vals, bins=bins, color="#2166AC", alpha=0.55, density=True, label="control")
        ax.hist(x_vals, bins=bins, color="#B2182B", alpha=0.55, density=True, label="double crispant")
        tag = dim_short(dim)
        is_hot = dim in HOT_DIMS
        ax.set_title(tag, fontsize=8, color="darkorange" if is_hot else "black",
                     fontweight="bold" if is_hot else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        if is_hot:
            for spine in ax.spines.values():
                spine.set_edgecolor("darkorange")
                spine.set_linewidth(2)

    for j in range(n_dims, len(axes)):
        axes[j].axis("off")

    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle("Supplementary: all 80 raw z_mu_b dims, control (blue) vs double-crispant (red), "
                 "pooled across time bins\n(orange border = classifier-hot dim)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "per_dim_histograms.png", dpi=130)
    plt.close(fig)


def fig_hot_dims_zoom(df: pd.DataFrame, pooled_ranked: pd.DataFrame, top_n: int = 6):
    top_marginal = [d for d in pooled_ranked["dim"].head(top_n) if d not in HOT_DIMS]
    zoom_dims = list(HOT_DIMS) + top_marginal
    zoom_dims = list(dict.fromkeys(zoom_dims))

    control_df = df[df.group == "control"]
    ncols = 3
    nrows = int(np.ceil(len(zoom_dims) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 3.6))
    axes = np.atleast_1d(axes).ravel()

    crispant_colors = {"pbx1b_crispant": "#f4a582", "pbx4_crispant": "#d6604d",
                        "pbx1b_pbx4_crispant": "#B2182B"}

    for ax, dim in zip(axes, zoom_dims):
        c_vals = control_df[dim].dropna().values
        all_vals = [c_vals] + [df.loc[df.genotype == g, dim].dropna().values for g in CRISPANTS]
        lo = min(v.min() for v in all_vals if len(v))
        hi = max(v.max() for v in all_vals if len(v))
        bins = np.linspace(lo, hi, 30)
        ax.hist(c_vals, bins=bins, color="#2166AC", alpha=0.45, density=True, label="control")
        for g in CRISPANTS:
            vals = df.loc[df.genotype == g, dim].dropna().values
            ax.hist(vals, bins=bins, histtype="step", linewidth=2,
                    color=crispant_colors[g], density=True, label=GENO_LABEL[g])
        is_hot = dim in HOT_DIMS
        title = dim_short(dim) + ("  [CLASSIFIER HOT]" if is_hot else "  [top marginal]")
        ax.set_title(title, fontsize=10, color="darkorange" if is_hot else "black", fontweight="bold")
        ax.set_xlabel(dim)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        if is_hot:
            for spine in ax.spines.values():
                spine.set_edgecolor("darkorange")
                spine.set_linewidth(2.2)

    for j in range(len(zoom_dims), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Supplementary zoom: classifier-hot dims (z_mu_b_33, z_mu_b_71) vs top marginal dims\n"
                 "control (filled blue) vs each crispant (colored outline), pooled across time bins",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES / "hot_dims_zoom.png", dpi=140)
    plt.close(fig)


# ───────────────────────── main ─────────────────────────

def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    df, z_cols = load_data()
    print(f"Loaded {len(df)} rows, {df.embryo_id.nunique()} embryos, {len(z_cols)} z_mu_b dims")
    print(df.groupby("genotype").embryo_id.nunique())

    clf_long, clf_summary = load_classifier_weights()
    print("\nClassifier weight summary (mean |coef| across all comparisons/bins), hot dims:")
    print(clf_summary[clf_summary.dim.isin(HOT_DIMS)])

    # ── HEADLINE ──
    print("\n[HEADLINE] Computing raw delta over time (crispant mean - control mean, per dim/bin)...")
    raw_delta_df = compute_raw_delta_over_time(df, z_cols)

    print("[HEADLINE] Refitting classifier directions to recover SIGNED coefficients "
          "(classifier_weight_long.csv only stores |coef|)...")
    signed_clf_df = compute_signed_classifier_coefs(df, z_cols)

    headline_df = build_headline_table(raw_delta_df, clf_long, signed_clf_df)
    headline_df.to_csv(TABLES / "raw_delta_over_time.csv", index=False)
    print(f"Saved tables/raw_delta_over_time.csv ({len(headline_df)} rows)")

    n_missing_clf = headline_df["classifier_abs_coef"].isna().sum()
    n_missing_signed = headline_df["classifier_signed_coef"].isna().sum()
    print(f"  (rows with no matching classifier |coef| bin: {n_missing_clf}/{len(headline_df)}; "
          f"no matching signed-coef bin: {n_missing_signed}/{len(headline_df)})")

    fig_raw_delta_vs_classifier(headline_df, z_cols)
    print("Saved figures/raw_delta_vs_classifier_over_time.png  [HEADLINE FIGURE]")

    # ── supplementary ──
    print("\n[supplementary] Computing pooled Cohen's d / Wasserstein / bootstrap CI "
          f"(B={B_BOOT})...")
    pooled_result = compute_difference_table_pooled_only(df, z_cols, rng)
    pooled_result.to_csv(TABLES / "per_dim_difference.csv", index=False)
    print(f"Saved tables/per_dim_difference.csv ({len(pooled_result)} rows)")

    ranked = fig_ranked_headline(pooled_result, clf_summary)
    print("Saved figures/per_dim_difference_ranked.png")

    fig_over_time_cohend(raw_delta_df, df, z_cols)
    print("Saved figures/per_dim_difference_over_time.png")

    fig_histograms(df, z_cols)
    print("Saved figures/per_dim_histograms.png")

    fig_hot_dims_zoom(df, ranked, top_n=6)
    print("Saved figures/hot_dims_zoom.png")

    # ── final report ──
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    comp = "pbx1b_pbx4_crispant__vs__control"
    for hot in HOT_DIMS:
        sub = headline_df[(headline_df.comparison == comp) & (headline_df.dim == hot)]
        if sub.empty:
            continue
        print(f"\n{hot} (double crispant vs control), over time:")
        print(f"  raw_delta range: [{sub.raw_delta.min():.3f}, {sub.raw_delta.max():.3f}], "
              f"mean={sub.raw_delta.mean():.3f}")
        print(f"  classifier |coef| range: [{sub.classifier_abs_coef.min():.3f}, "
              f"{sub.classifier_abs_coef.max():.3f}], mean={sub.classifier_abs_coef.mean():.3f}")
        s = sub.dropna(subset=["classifier_signed_coef"])
        print(f"  classifier signed coef range: [{s.classifier_signed_coef.min():.3f}, "
              f"{s.classifier_signed_coef.max():.3f}], mean={s.classifier_signed_coef.mean():.3f}")
        agree = np.sign(s.raw_delta) == np.sign(s.classifier_signed_coef)
        print(f"  sign agreement (raw_delta vs classifier_signed_coef): "
              f"{agree.sum()}/{len(agree)} bins ({100*agree.mean():.0f}%)")
        rank_row = ranked[ranked.dim == hot]
        if not rank_row.empty:
            r = rank_row.iloc[0]
            print(f"  pooled marginal rank (Cohen's d): {int(r['rank'])}/80, d={r['cohens_d']:.3f}")


if __name__ == "__main__":
    main()
