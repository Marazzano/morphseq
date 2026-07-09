"""
Scratch: raw-value box/strip plots for script 10's classifier-hot dims,
at the bin center nearest 72hpf, split by genotype and colored by experiment_id.

Uses the SAME time-binning as script 10 (extract_classifier_directions with
bin_width=4.0 on the raw time_bin column). Earlier drafts of this analysis
filtered to a raw time_bin==72 value with re-binning disabled, which is a
DIFFERENT bin than script 10's heatmap (script 10 lands on bin centers
70.0/74.0, not 72.0) and produced a different, wrong top-dim ranking
(z_mu_b_69 instead of the real z_mu_b_71). Hot dims are derived from the
pooled fit at the correct bin, not hardcoded.

With the fix, BOTH top dims show up in the wik_ab-vs-inj_ctrl negative control
(no CRISPR) as well as all 3 crispant-vs-inj_ctrl comparisons -- so this is no
longer a clean "one dim is CRISPR-specific, one is injection-artifact" split;
that was an artifact of the binning bug. See 11_batch_reproducibility_check.py for
the per-experiment reproducibility check this plot is a companion to.
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

TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"
RAW_WT_CSV = TABLES / "pbx_binned_zmub_with_wt.csv"

GENO_ORDER = ("wik_ab", "inj_ctrl", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
GENO_LABEL = {"wik_ab": "WT", "inj_ctrl": "inj_ctrl", "pbx1b_crispant": "pbx1b",
              "pbx4_crispant": "pbx4", "pbx1b_pbx4_crispant": "double"}
CRISPANTS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
NEG_CONTROL_COMPARISON = ("wik_ab", "inj_ctrl")
BIN_WIDTH = 4.0   # must match script 10

df = pd.read_csv(RAW_WT_CSV, low_memory=False)
z_cols_all = sorted([c for c in df.columns if "z_mu_b" in c],
                    key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")))
sub = df[df.genotype.isin(GENO_ORDER)].copy()

experiments = sorted(sub.experiment_id.unique())   # all 3: 20251207_pbx, 20260304, 20260306
exp_colors = dict(zip(experiments, plt.cm.tab10.colors))
N_GENO = len(GENO_ORDER)


def fit_directions(data, comparisons, bin_width=BIN_WIDTH):
    present = set(data.genotype.unique())
    usable = [c for c in comparisons if c["positive"] in present and c["negative"] in present]
    if not usable:
        return None
    try:
        return extract_classifier_directions(
            data, class_col="genotype", id_col="embryo_id", time_col="time_bin",
            comparisons=usable, features={"emb": z_cols_all}, bin_width=bin_width,
            min_samples_per_group=2, min_samples_per_member=2, verbose=False,
        )
    except Exception:
        return None


def abs_coef_at(directions, comparison_id, bin_center, dim):
    if directions is None:
        return np.nan
    meta = directions.metadata
    row = meta[(meta.comparison_id == comparison_id) & np.isclose(meta.time_bin_center, bin_center)]
    if row.empty:
        return np.nan
    row = row.iloc[0]
    vec = directions.vectors[row["vector_id"]]
    names = directions.feature_names[row["feature_set"]]
    return abs(float(vec[names.index(dim)]))


# ── derive the correct bin center nearest 72hpf, and the pooled top-2 hot dims,
# exactly as script 10 would (same fit call, same binning) ─────────────────────
crispant_comparisons = [{"positive": g, "negative": "inj_ctrl"} for g in CRISPANTS]
neg_comparison = [{"positive": NEG_CONTROL_COMPARISON[0], "negative": NEG_CONTROL_COMPARISON[1]}]
all_comparisons = crispant_comparisons + neg_comparison

pooled_all = fit_directions(sub, all_comparisons)
probe_bins = sorted(pooled_all.metadata.time_bin_center.unique())
near72 = min(probe_bins, key=lambda b: abs(b - 72))
print(f"[bin_width={BIN_WIDTH}] nearest bin center to 72 -> {near72}")

pooled_meta_at_bin = pooled_all.metadata[np.isclose(pooled_all.metadata.time_bin_center, near72)]
pooled_avg = {}
for _, r in pooled_meta_at_bin.iterrows():
    vec = pooled_all.vectors[r["vector_id"]]
    names = pooled_all.feature_names[r["feature_set"]]
    for name, w in zip(names, vec):
        pooled_avg.setdefault(name, []).append(abs(float(w)))
pooled_avg = pd.Series({k: np.mean(v) for k, v in pooled_avg.items()}).sort_values(ascending=False)
DIMS = tuple(pooled_avg.index[:2])
print(f"Top-2 pooled dims @ bin={near72}: {DIMS}")

DIM_TAG = {d: f"{d.replace('_binned','')}  (top-{i+1} pooled |coef| @ bin {near72})"
           for i, d in enumerate(DIMS)}
# strongest crispant (largest mean |coef| across DIMS) used for the bottom-row bars,
# alongside the negative control -- shows BOTH stories side by side per dim now
strongest_crispant = max(
    CRISPANTS,
    key=lambda g: np.mean([abs_coef_at(pooled_all, f"{g}__vs__inj_ctrl", near72, d) for d in DIMS]),
)
print(f"Strongest crispant comparison for bar panel: {strongest_crispant}")

# extract_classifier_directions reports time_bin_center = raw_time_bin + bin_width/2
# (see engine/data_prep.py), so the raw `time_bin` value matching `near72` is
# near72 - BIN_WIDTH/2, not near72 itself.
raw_time_bin_match = near72 - BIN_WIDTH / 2.0
at = sub[np.isclose(sub.time_bin, raw_time_bin_match)].copy()
print(f"raw time_bin == {raw_time_bin_match} -> n={len(at)}")


fig, axes = plt.subplots(2, 2, figsize=(15, 10.5),
                         gridspec_kw=dict(height_ratios=[2.3, 1]))
rng = np.random.default_rng(0)

for ax, dim in zip(axes[0], DIMS):
    box_data = []
    positions = []
    for i, g in enumerate(GENO_ORDER):
        vals = at.loc[at.genotype == g, dim].dropna().values
        box_data.append(vals)
        positions.append(i)

    bp = ax.boxplot(box_data, positions=positions, widths=0.55, showfliers=False,
                     patch_artist=True, zorder=2)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dddddd")
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # jittered points colored by experiment, with per-cell n annotation
    for i, g in enumerate(GENO_ORDER):
        gsub = at[at.genotype == g]
        for exp in experiments:
            vals = gsub.loc[gsub.experiment_id == exp, dim].dropna().values
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(i + jitter, vals, s=22, color=exp_colors[exp], alpha=0.75,
                       edgecolors="none", zorder=3,
                       label=exp if i == 0 else None)
        n_total = len(gsub[dim].dropna())
        ymax = at[dim].max()
        ax.text(i, ymax * 1.06, f"n={n_total}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(range(N_GENO)))
    ax.set_xticklabels([GENO_LABEL[g] for g in GENO_ORDER], fontsize=10)
    ax.axvspan(-0.5, 1.5, color="#f4a582", alpha=0.12, zorder=0)   # WT/inj_ctrl pair
    ax.axvspan(1.5, N_GENO - 0.5, color="#92c5de", alpha=0.12, zorder=0)    # crispants
    ax.set_ylabel(f"raw {dim} value")
    ax.set_title(DIM_TAG[dim], fontsize=10.5)
    ax.set_ylim(top=at[dim].max() * 1.18)
    ax.grid(alpha=0.25, axis="y")

axes[0][0].legend(title="experiment_id", loc="upper right", fontsize=8, title_fontsize=8)

# ── bottom row: does the classifier's |coef| for this dim survive pooling, for
# BOTH the negative control AND the strongest crispant comparison? Per-experiment
# fits (fit separately within each batch) vs one fit pooling all experiments. If
# pooled collapses relative to each individual experiment, the signal was
# inconsistent/batch-specific and pooling washes it out; if pooled stays as
# strong as each experiment, it's reproducible. ──
bar_comparisons = [
    ("neg-ctrl", NEG_CONTROL_COMPARISON, "#f4a582"),
    (GENO_LABEL[strongest_crispant], (strongest_crispant, "inj_ctrl"), "#92c5de"),
]

for ax, dim in zip(axes[1], DIMS):
    group_w = 0.38
    for gi, (tag, (pos, neg), face) in enumerate(bar_comparisons):
        cid = f"{pos}__vs__{neg}"
        xs, hs = [], []
        # fit each experiment on its FULL time range (bin_width=BIN_WIDTH, same as
        # pooled) so the fitted bin center is directly comparable to `near72`
        for exp in experiments:
            exp_data = sub[sub.experiment_id == exp]
            d = fit_directions(exp_data, [{"positive": pos, "negative": neg}])
            xs.append(exp)
            hs.append(abs_coef_at(d, cid, near72, dim))
        pooled_val = abs_coef_at(pooled_all, cid, near72, dim)
        xs.append("POOLED"); hs.append(pooled_val)

        n_bars = len(xs)
        base_x = np.arange(n_bars)
        offset = (gi - 0.5) * group_w
        heights = [0 if np.isnan(h) else h for h in hs]
        ax.bar(base_x + offset, heights, width=group_w * 0.92, color=face,
              edgecolor="k", linewidth=0.7, label=tag)
        for x, h in zip(base_x + offset, hs):
            label = "n/a" if np.isnan(h) else f"{h:.2f}"
            ax.text(x, (0 if np.isnan(h) else h) + 0.015, label, ha="center", fontsize=7.5,
                   rotation=90 if n_bars > 4 else 0)

    ax.set_xticks(np.arange(len(experiments) + 1))
    ax.set_xticklabels(list(experiments) + ["POOLED"], fontsize=8, rotation=15)
    ax.set_ylabel("|classifier coef|")
    ax.set_title(f"per-experiment vs pooled |coef({dim.replace('_binned','')})|", fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7.5)

fig.suptitle(f"Raw values (top) and classifier importance (bottom) of script 10's hot dims, "
             f"time_bin_center={near72} (bin_width={BIN_WIDTH}, matches script 10)\n"
             "shaded orange = negative-control pair (WT vs inj_ctrl, no CRISPR); "
             f"shaded blue = strongest crispant ({GENO_LABEL[strongest_crispant]}) vs inj_ctrl",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(FIGURES / "hot_dims_raw_vs_coef.png", dpi=140, bbox_inches="tight")
print("saved figures/hot_dims_raw_vs_coef.png")
