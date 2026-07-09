"""
Scratch: is the hot-dim pattern in script 10's heatmap a batch effect?

Uses the SAME time-binning as script 10 (extract_classifier_directions with
bin_width=4.0 on the raw time_bin column) -- do not filter to a raw time_bin
value and disable re-binning, that silently changes which bin center "72hpf"
actually refers to and changes which dims come out on top.

At the bin center nearest 72hpf, fits crispant-vs-inj_ctrl classifier
directions three ways:
  (a) within each experiment separately
  (b) pooled across all experiments (same fit script 10's heatmap uses)
Top-2 pooled dims are derived from the data, not hardcoded. Overlaid as line
plots over the 80 raw dims. If (a) per-experiment curves disagree wildly with
each other AND the pooled curve just happens to peak at the same dims, that's
consistent with batch driving the pooled result. If all curves (per-experiment
AND pooled) agree on the same hot dims, that's signal reproducible independent
of batch.
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
CRISPANTS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
GENO_LABEL = {"pbx1b_crispant": "pbx1b", "pbx4_crispant": "pbx4",
              "pbx1b_pbx4_crispant": "double", "wik_ab": "WT (neg ctrl)"}
NEG_CONTROL = "wik_ab"   # uninjected WT vs inj_ctrl: injection-procedure artifact check,
                         # NOT genotype/CRISPR. Should NOT light up the same dims as the
                         # crispant comparisons if 33/69 are real CRISPR-phenotype signal.
ALL_GROUPS = (*CRISPANTS, NEG_CONTROL)

# IMPORTANT: use the exact same time-binning scheme as script 10
# (extract_classifier_directions(..., bin_width=4.0) on the raw time_bin column)
# so "72hpf" here means the same bin script 10 reports. Script 10's bins land at
# 70.0/74.0 (not 72.0) -- snapping to a manually-filtered time_bin==72 subset and
# disabling re-binning (bin_width=1000) silently used a DIFFERENT bin than script
# 10, which produced a different top-dim ranking (69 vs the real 71). Never diverge
# the binning between scripts comparing the same "hot dims" result.
BIN_WIDTH = 4.0

df = pd.read_csv(RAW_WT_CSV, low_memory=False)
z_cols = sorted([c for c in df.columns if "z_mu_b" in c],
                key=lambda c: int(c.replace("z_mu_b_", "").replace("_binned", "")))
sub = df[df.genotype.isin((*ALL_GROUPS, "inj_ctrl"))].copy()

experiments = sorted(sub.experiment_id.unique())
comparisons = [{"positive": g, "negative": "inj_ctrl"} for g in ALL_GROUPS]

def fit(data, min_grp=2, min_mem=2):
    return extract_classifier_directions(
        data, class_col="genotype", id_col="embryo_id", time_col="time_bin",
        comparisons=comparisons, features={"emb": z_cols}, bin_width=BIN_WIDTH,
        min_samples_per_group=min_grp, min_samples_per_member=min_mem, verbose=True,
    )

# fit on the FULL (unfiltered) frame so extract_classifier_directions does its own
# 4hpf binning exactly like script 10, then find the resulting bin center nearest 72
_pooled_probe = fit(sub)
_probe_bins = sorted(_pooled_probe.metadata.time_bin_center.unique())
near72 = min(_probe_bins, key=lambda b: abs(b - 72))
print(f"[bin_width={BIN_WIDTH}] nearest bin center to 72 -> {near72}")

at = sub.copy()   # keep full time range; filtering to one bin happens post-fit

def to_frame(directions, tag, keep_bin=None):
    """keep_bin: if given, only rows whose time_bin_center == keep_bin are kept
    (so per-experiment fits use the SAME bin definition/center as the pooled fit,
    filtered after the fact rather than by pre-slicing raw time_bin values)."""
    meta = directions.metadata
    if keep_bin is not None:
        meta = meta[np.isclose(meta.time_bin_center, keep_bin)]
    rows = []
    for _, r in meta.iterrows():
        vec = directions.vectors[r["vector_id"]]
        names = directions.feature_names[r["feature_set"]]
        for name, w in zip(names, vec):
            rows.append(dict(source=tag, comparison_id=r["comparison_id"], dim=name, abs_weight=abs(float(w))))
    return pd.DataFrame(rows)

all_frames = []
all_frames.append(to_frame(_pooled_probe, "POOLED", keep_bin=near72))

for exp in experiments:
    exp_data = at[at.experiment_id == exp]
    present = set(exp_data.genotype.unique())
    exp_comparisons = [c for c in comparisons
                       if c["positive"] in present and c["negative"] in present]
    if not exp_comparisons:
        print(f"  skip {exp}: no comparisons available")
        continue
    try:
        d = extract_classifier_directions(
            exp_data, class_col="genotype", id_col="embryo_id", time_col="time_bin",
            comparisons=exp_comparisons, features={"emb": z_cols}, bin_width=BIN_WIDTH,
            min_samples_per_group=2, min_samples_per_member=2, verbose=True,
        )
        all_frames.append(to_frame(d, exp, keep_bin=near72))
    except Exception as e:
        print(f"  skip {exp}: {e}")

long_df = pd.concat(all_frames, ignore_index=True)
long_df.to_csv(TABLES / "batch_reproducibility_check.csv", index=False)

# derive the actual top-2 pooled dims at this bin (don't hardcode which dims are
# "hot" -- that changes with the binning, as this whole fix demonstrated)
pooled_avg = (long_df[long_df.source == "POOLED"].groupby("dim")["abs_weight"]
              .mean().sort_values(ascending=False))
top2 = list(pooled_avg.index[:2])
print(f"Top-2 pooled dims @ bin={near72}: {top2}")

# overlay plot: one panel per comparison (3 crispants + 1 neg control), one line per source
fig, axes = plt.subplots(len(ALL_GROUPS), 1, figsize=(14, 13), sharex=True)
colors = {"POOLED": "black"}
palette = plt.cm.tab10.colors
for i, exp in enumerate(experiments):
    colors[exp] = palette[i % len(palette)]

for ax, g in zip(axes, ALL_GROUPS):
    cid = f"{g}__vs__inj_ctrl"
    gsub = long_df[long_df.comparison_id == cid]
    for src, ssub in gsub.groupby("source"):
        ssub = ssub.set_index("dim").reindex(z_cols)
        lw = 3 if src == "POOLED" else 1.5
        ls = "-" if src == "POOLED" else "--"
        ax.plot(range(len(z_cols)), ssub.abs_weight.values, label=src,
                 color=colors.get(src, "gray"), lw=lw, ls=ls, alpha=0.9)
    title_tag = " *** NEGATIVE CONTROL (no CRISPR) ***" if g == NEG_CONTROL else ""
    ax.set_title(f"{GENO_LABEL[g]} vs inj_ctrl @ time_bin={near72}{title_tag}")
    ax.set_ylabel("|unit coef|")
    ax.legend(fontsize=7, ncol=4)
    for dim in top2:
        ax.axvline(z_cols.index(dim), color="red", ls=":", alpha=0.5)

axes[-1].set_xticks(range(0, len(z_cols), 5))
axes[-1].set_xticklabels([z_cols[i] for i in range(0, len(z_cols), 5)], rotation=90, fontsize=7)
axes[-1].set_xlabel("raw z_mu_b dim")
fig.suptitle(f"Per-experiment vs pooled classifier weight, time_bin_center={near72} "
             f"(bin_width={BIN_WIDTH}, matches script 10)\n"
             f"(red dashed lines mark {top2[0]} / {top2[1]}, the pooled-hot dims)")
fig.tight_layout()
fig.savefig(FIGURES / "batch_reproducibility_check.png", dpi=130)
print("saved figures/batch_reproducibility_check.png")
