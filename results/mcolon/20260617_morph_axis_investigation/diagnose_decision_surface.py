"""
DIAGNOSTIC (not a fix): map the system's continuous-vs-discrete decision surface
using b9d2 as a known-DISCRETE use case.

We do NOT change the core. This script calls the existing support-geometry
statistics (valley_depth / mst_max_edge / fiedler) and re-derives the support call
under a RANGE of two system-level knobs, to SEE what the system currently allows to
be called discrete and which design decisions move that line:

  KNOB 1 - normalize_shape whitening   : MAD (current) | trimmed-core | none(raw)
      The scale denominator applied before the statistics. A bimodal group's own
      spread includes the between-mode distance, so whitening can shrink a real gap
      into invisibility. We test three denominators.

  KNOB 2 - corroboration rule          : valley&graph (current) | valley|graph | valley-alone
      How the three statistics combine into "discrete". The current rule REQUIRES a
      density valley AND a graph statistic; looser rules are more permissive.

b9d2 is the probe because we KNOW the biological answer is discrete (CE is a
discrete phenotype; the CE/HTA split sharpens over developmental time). So a knob
setting that calls b9d2 continuous at 24/48 hpf is under-calling; one that calls WT
discrete would be over-calling. This is exploration of strengths/weaknesses, not
tuning: nothing here is written back into the core.

Outputs:
  plots/decision_surface_heatmap.png   - combined grid: (whitening x rule) rows x hpf cols,
                                         cell = call, annotated with the triggering stat's p
  plots/decision_surface_perknob.png   - small-multiples: each knob swept alone
  tables/decision_surface.csv          - every (whitening, rule, axis, hpf) -> call + p-values

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/diagnose_decision_surface.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RUN_DIR))

import support_geometry as sg  # noqa: E402  (we read its statistics; we do NOT edit it)

GENE14_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
REF_CSV = GENE14_DIR / "tables" / "reference_b9d2_clean.csv"
PLOT_DIR = RUN_DIR / "plots"
TABLE_DIR = RUN_DIR / "tables"
PLOT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config -- match the anchor scripts, but pool ALL labeled CE/HTA (label-free
# geometry test; zygosity is not a geometry knob so we keep the full arm present
# and expose only the two requested system knobs).
# ---------------------------------------------------------------------------

X_FEAT, Y_FEAT = "total_length_um", "baseline_deviation_normalized"
PHENO_LABELS = ["CE", "HTA"]
TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
TARGET_DESIGN_HPF = [14, 18, 24, 30, 48]
_bin_center = lambda h: float(int(h // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2)
BIN_CENTER_TO_HPF = {_bin_center(h): h for h in TARGET_DESIGN_HPF}
MIN_EMBRYOS = 10
N_RESAMPLE = 300
SEED = 42

# ---------------------------------------------------------------------------
# KNOB 1: whitening variants. Each maps 2-D points -> normalized 2-D points.
# We reuse the core MAD version verbatim; the others are diagnostic alternatives.
# ---------------------------------------------------------------------------

def whiten_mad(pts: np.ndarray) -> np.ndarray:
    """CURRENT core behavior (support_geometry.normalize_shape)."""
    return sg.normalize_shape(pts)


def whiten_trimmed(pts: np.ndarray) -> np.ndarray:
    """Scale by a within-CORE spread: MAD of the inner 50% (IQR-trimmed) points,
    so the between-mode distance can't inflate its own denominator. Still centers
    on the median and kills gross scale, but preserves a true gap's gap/width."""
    pts = np.asarray(pts, dtype=float)
    center = np.median(pts, axis=0)
    d = pts - center
    scale = np.empty(pts.shape[1])
    for a in range(pts.shape[1]):
        col = np.abs(d[:, a])
        lo, hi = np.percentile(col, [0, 50])  # inner-half absolute deviations
        core = col[(col >= lo) & (col <= hi)]
        s = np.median(core) * 1.4826
        scale[a] = s if s > 1e-12 else 1.0
    return d / scale


def whiten_none(pts: np.ndarray) -> np.ndarray:
    """No whitening: center on median only, keep native per-axis scale ratio by
    dividing by a SINGLE global scale (so KDE bandwidth is sane) but NOT per-axis.
    This is the 'let the real gap stand' extreme."""
    pts = np.asarray(pts, dtype=float)
    center = np.median(pts, axis=0)
    d = pts - center
    s = np.median(np.abs(d)) * 1.4826
    s = s if s > 1e-12 else 1.0
    return d / s


WHITENERS = {"MAD (current)": whiten_mad,
             "trimmed-core": whiten_trimmed,
             "raw (none)": whiten_none}

# ---------------------------------------------------------------------------
# KNOB 2: corroboration rules. Pure functions of the three p-values.
# ---------------------------------------------------------------------------

def rule_valley_and_graph(vp, mp, fp):
    """CURRENT core rule (support_geometry.support_call), including the relaxation."""
    graph = (fp < 0.05) or (mp < 0.05)
    strong = fp < 0.02
    return (vp < 0.05 and graph) or (vp < 0.12 and strong)


def rule_valley_or_graph(vp, mp, fp):
    return (vp < 0.05) or (mp < 0.05) or (fp < 0.05)


def rule_valley_alone(vp, mp, fp):
    return vp < 0.05


RULES = {"valley & graph (current)": rule_valley_and_graph,
         "valley | graph": rule_valley_or_graph,
         "valley alone": rule_valley_alone}

# ---------------------------------------------------------------------------
# Data prep -- one row per embryo, per time-bin, pooled labeled CE/HTA vs WT.
# ---------------------------------------------------------------------------

def _modal(x):
    m = x.dropna().mode()
    return m.iloc[0] if len(m) else np.nan


def load_binned():
    df = pd.read_csv(REF_CSV, low_memory=False)
    df = df.dropna(subset=[TIME_COL, X_FEAT, Y_FEAT, "phenotype_clean", "zygosity"]).copy()
    df["bc"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2
    ecol = "embryo_id" if df["physical_embryo_id"].isna().all() else "physical_embryo_id"
    agg = {X_FEAT: "mean", Y_FEAT: "mean",
           "phenotype_clean": _modal, "zygosity": _modal}
    e = df.groupby([ecol, "bc"]).agg(agg).reset_index().rename(columns={ecol: "embryo_id"})
    return e


def stat_pvalues(group_xy, wt_xy, whiten, rng):
    """Run the three core statistics with a chosen whitening, each vs a matched-N WT
    bootstrap null (same machinery as compute_support_geometry, but we inject the
    whitener rather than editing the core). Returns dict name -> (obs, pvalue)."""
    g = whiten(group_xy)
    w = whiten(wt_xy)
    n = len(g)
    draws = [rng.choice(len(w), size=n, replace=True) for _ in range(N_RESAMPLE)]
    out = {}
    for name, fn in sg.SUPPORT_STATISTICS.items():
        obs = fn(g)
        null = np.array([fn(w[idx]) for idx in draws])
        out[name] = (float(obs), float(np.mean(null >= obs)))
    return out

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep():
    e = load_binned()
    rows = []
    for bc, hpf in sorted(BIN_CENTER_TO_HPF.items()):
        sub = e[e["bc"] == bc]
        grp = sub[sub["phenotype_clean"].isin(PHENO_LABELS)]
        wt = sub[sub["zygosity"] == "wildtype"]
        if len(grp) < MIN_EMBRYOS or len(wt) < MIN_EMBRYOS:
            for wname in WHITENERS:
                for rname in RULES:
                    rows.append(dict(hpf=hpf, whitening=wname, rule=rname,
                                     n_group=len(grp), n_wt=len(wt),
                                     valley_p=np.nan, mst_p=np.nan, fiedler_p=np.nan,
                                     call="n/a"))
            continue
        gxy = grp[[X_FEAT, Y_FEAT]].values.astype(float)
        wxy = wt[[X_FEAT, Y_FEAT]].values.astype(float)
        for wname, whiten in WHITENERS.items():
            rng = np.random.default_rng(SEED)  # same draws across rules
            pv = stat_pvalues(gxy, wxy, whiten, rng)
            vp = pv["valley_depth"][1]
            mp = pv["mst_max_edge"][1]
            fp = pv["fiedler"][1]
            for rname, rule in RULES.items():
                call = "discrete" if rule(vp, mp, fp) else "continuous"
                rows.append(dict(hpf=hpf, whitening=wname, rule=rname,
                                 n_group=len(gxy), n_wt=len(wxy),
                                 valley_p=vp, mst_p=mp, fiedler_p=fp, call=call))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plot 1: combined heatmap  (whitening x rule) rows  x  hpf cols
# ---------------------------------------------------------------------------

CALL_VAL = {"continuous": 0, "discrete": 1, "n/a": np.nan}
CMAP = plt.cm.RdBu_r  # continuous(0)=blue, discrete(1)=red


def plot_heatmap(df):
    hpfs = TARGET_DESIGN_HPF
    combos = [(w, r) for w in WHITENERS for r in RULES]
    grid = np.full((len(combos), len(hpfs)), np.nan)
    annot = np.empty((len(combos), len(hpfs)), dtype=object)
    for i, (w, r) in enumerate(combos):
        for j, h in enumerate(hpfs):
            row = df[(df.whitening == w) & (df.rule == r) & (df.hpf == h)]
            if len(row) == 0 or row.iloc[0]["call"] == "n/a":
                annot[i, j] = ""
                continue
            rr = row.iloc[0]
            grid[i, j] = CALL_VAL[rr["call"]]
            annot[i, j] = f"v{rr.valley_p:.2f}\nf{rr.fiedler_p:.2f}"

    fig, ax = plt.subplots(figsize=(1.15 * len(hpfs) + 3, 0.7 * len(combos) + 2))
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(hpfs)))
    ax.set_xticklabels([f"{h} hpf" for h in hpfs])
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels([f"{w}\n{r}" for (w, r) in combos], fontsize=8)
    for i in range(len(combos)):
        for j in range(len(hpfs)):
            if annot[i, j]:
                val = grid[i, j]
                txt_call = "DISCRETE" if val == 1 else "cont."
                ax.text(j, i, f"{txt_call}\n{annot[i,j]}", ha="center", va="center",
                        fontsize=6.5, color="white" if not np.isnan(val) else "gray")
    ax.set_title("b9d2 decision surface — what the system ALLOWS to be called discrete\n"
                 "(known-discrete probe; red=discrete, blue=continuous; v=valley_p, f=fiedler_p)",
                 fontsize=10)
    # separators between whitening blocks
    for k in range(1, len(WHITENERS)):
        ax.axhline(k * len(RULES) - 0.5, color="k", linewidth=1.4)
    fig.tight_layout()
    out = PLOT_DIR / "decision_surface_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


# ---------------------------------------------------------------------------
# Plot 2: per-knob small-multiples (sweep one knob, hold the other at CURRENT)
# ---------------------------------------------------------------------------

CUR_WHITEN = "MAD (current)"
CUR_RULE = "valley & graph (current)"


def plot_perknob(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    # (a) sweep whitening, rule fixed at current
    ax = axes[0]
    for w in WHITENERS:
        sub = df[(df.whitening == w) & (df.rule == CUR_RULE)].sort_values("hpf")
        y = [CALL_VAL.get(c, np.nan) for c in sub["call"]]
        ax.plot(sub["hpf"], y, "o-", label=w, linewidth=2, markersize=8)
    ax.set_title(f"KNOB 1: whitening  (rule held = {CUR_RULE})", fontsize=9)
    _call_axis(ax)

    # (b) sweep rule, whitening fixed at current
    ax = axes[1]
    for r in RULES:
        sub = df[(df.whitening == CUR_WHITEN) & (df.rule == r)].sort_values("hpf")
        y = [CALL_VAL.get(c, np.nan) for c in sub["call"]]
        ax.plot(sub["hpf"], y, "s--", label=r, linewidth=2, markersize=8)
    ax.set_title(f"KNOB 2: corroboration rule  (whitening held = {CUR_WHITEN})", fontsize=9)
    _call_axis(ax)

    fig.suptitle("b9d2 call vs. each system knob swept alone (ground truth = DISCRETE at 24/48 hpf)",
                 fontsize=11)
    fig.tight_layout()
    out = PLOT_DIR / "decision_surface_perknob.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


def _call_axis(ax):
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["continuous", "DISCRETE"])
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("predicted stage (hpf)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Sweeping b9d2 decision surface (whitening x corroboration rule x hpf)...")
    df = sweep()
    csv = TABLE_DIR / "decision_surface.csv"
    df.to_csv(csv, index=False)
    print(f"Saved: {csv.relative_to(RUN_DIR)}  ({len(df)} rows)")

    # console summary: where does b9d2 read discrete?
    disc = df[df.call == "discrete"]
    print("\nSettings that call b9d2 DISCRETE:")
    if len(disc) == 0:
        print("  (none — the system calls b9d2 continuous under every swept setting)")
    else:
        for _, r in disc.iterrows():
            print(f"  {r.hpf:>2} hpf | {r.whitening:14s} | {r.rule:26s} "
                  f"| valley_p={r.valley_p:.3f} fiedler_p={r.fiedler_p:.3f}")

    plot_heatmap(df)
    plot_perknob(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
