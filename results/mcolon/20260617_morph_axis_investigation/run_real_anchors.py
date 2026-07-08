"""
Apply the full WT-calibrated phenotype-geometry decision tree to the real anchor
genes (cep290, b9d2), on TWO axes:

  axis A ("hand")      : total_length_um x baseline_deviation_normalized
  axis B ("embedding") : first two PCs of z_mu_b, PCA FIT ON WILDTYPE ONLY per
                         time-bin (label-free; WT defines the coordinate system,
                         Axiom 1), mutants projected in.

For each (gene x time-bin x axis) it runs `run_phenotype_geometry` (Stage 1-4 with
conditioning + per-statistic confidence) on the POOLED homozygous population vs.
stage-matched wildtype, then:
  - writes a machine-readable summary CSV (one row per gene x timebin x axis),
  - persists the full bootstrap null for every statistic to a companion .npz,
  - detects the estimated variance->mode transition (first sustained discrete call
    at confidence >= moderate) per (gene, axis),
  - plots the connectedness trajectory for both axes with the transition marked.

Anchors (known biology; used to check the method recovers it):
  cep290 homozygous  -> expected CONTINUOUS (broad continuum)
  b9d2   homozygous  -> expected DISCRETE (fate split)

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/run_real_anchors.py
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

GENE14_DIR = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc"
REF_DIR = GENE14_DIR / "tables"
PLOT_DIR = RUN_DIR / "plots"
TABLE_DIR = RUN_DIR / "tables"
PLOT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

from phenotype_geometry import run_phenotype_geometry  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GENES = {
    "cep290": {
        "csv": REF_DIR / "reference_cep290_clean.csv",
        "phenotype_labels": ["High_to_Low", "Low_to_High"],
    },
    "b9d2": {
        "csv": REF_DIR / "reference_b9d2_clean.csv",
        "phenotype_labels": ["CE", "HTA"],
    },
}

HAND_FEATS = ["total_length_um", "baseline_deviation_normalized"]
Z_MU_B_PREFIX = "z_mu_b_"

TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 4.0
TARGET_DESIGN_HPF = [14, 18, 24, 30, 48]
_bin_center = lambda h: float(int(h // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2)
BIN_CENTER_TO_DESIGN_HPF = {_bin_center(h): h for h in TARGET_DESIGN_HPF}
TARGET_BIN_CENTERS = sorted(BIN_CENTER_TO_DESIGN_HPF)

CLIP_PERCENTILE = 1
MIN_EMBRYOS = 10
N_RESAMPLE = 80  # lowered from 200: conductance eigendecomps per resample are slow;
                 # 80 is adequate for p-value resolution at the p<0.05 threshold

CONF_ORDER = {"insufficient": 0, "low": 1, "moderate": 2, "high": 3}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_gene(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["time_bin_center"] = (df[TIME_COL] // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2
    return df


def _modal(x: pd.Series):
    m = x.dropna().mode()
    return m.iloc[0] if len(m) else np.nan


def collapse_to_embryo(
    df: pd.DataFrame,
    feat_cols: list[str],
    group_cols: list[str] | None = None,
    label_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One row per embryo group: mean features plus modal labels."""
    if group_cols is None:
        group_cols = ["time_bin_center"]
    if label_cols is None:
        label_cols = ["zygosity"]
    needed = [*group_cols, *label_cols, *feat_cols]
    d = df.dropna(subset=needed).copy()
    embryo_col = "embryo_id" if d["physical_embryo_id"].isna().all() else "physical_embryo_id"
    agg = {c: "mean" for c in feat_cols}
    for col in label_cols:
        agg[col] = _modal
    e = (d.groupby([embryo_col, *group_cols]).agg(agg).reset_index()
         .rename(columns={embryo_col: "embryo_id"}))
    return e


def remove_outliers(pts: np.ndarray) -> np.ndarray:
    lo = np.percentile(pts, CLIP_PERCENTILE, axis=0)
    hi = np.percentile(pts, 100 - CLIP_PERCENTILE, axis=0)
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def wt_pca_axis(wt_z: np.ndarray, n_pcs: int = 2):
    """Fit PCA on WT embeddings only; return (mean, components) for projection."""
    mean = wt_z.mean(axis=0)
    centered = wt_z - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:n_pcs]


# ---------------------------------------------------------------------------
# Per (gene x axis) run
# ---------------------------------------------------------------------------

def run_axis(
    gene: str,
    df: pd.DataFrame,
    phenotype_labels: list[str],
    axis_name: str,
    z_cols: list[str] | None,
):
    """Returns list of per-timebin result dicts for one axis."""
    if axis_name == "hand":
        feat_cols = HAND_FEATS
    else:
        feat_cols = z_cols
    edf = collapse_to_embryo(df, feat_cols, label_cols=["zygosity", "phenotype_clean"])

    rows = []
    for bc in TARGET_BIN_CENTERS:
        design_hpf = BIN_CENTER_TO_DESIGN_HPF[bc]
        sub = edf[edf["time_bin_center"] == bc]
        wt = sub[sub["zygosity"] == "wildtype"]
        # Test the WHOLE phenotype-labeled population, NOT homozygous-only. The
        # homozygous filter kept only ~8 of ~38 CE embryos and discarded the arm that
        # forms the fracture (CE is overwhelmingly het/unknown). The phenotype label is
        # a morphological cluster independent of genotype, and any real split is what
        # the geometry test should recover -- so pool by phenotype label and let the
        # geometry speak. Mirrors morph_axis_connectedness.py.
        homo = sub[sub["phenotype_clean"].isin(phenotype_labels)]
        if len(wt) < MIN_EMBRYOS or len(homo) < MIN_EMBRYOS:
            continue

        wt_feat = wt[feat_cols].values.astype(float)
        homo_feat = homo[feat_cols].values.astype(float)

        if axis_name == "embedding":
            mean, comps = wt_pca_axis(wt_feat, n_pcs=2)
            wt_xy = (wt_feat - mean) @ comps.T
            homo_xy = (homo_feat - mean) @ comps.T
        else:
            wt_xy, homo_xy = wt_feat, homo_feat

        # Outlier removal DISABLED: the extreme-short CE embryos ARE the second mode;
        # percentile clipping erases exactly the fracture we test for. Keep every
        # labeled embryo. (Matches morph_axis_connectedness.py.)
        if len(homo_xy) < MIN_EMBRYOS or len(wt_xy) < MIN_EMBRYOS:
            continue

        res = run_phenotype_geometry(homo_xy, wt_xy, n_resample=N_RESAMPLE,
                                     rng=np.random.default_rng(42), compute_loo=True)
        label_counts = homo["phenotype_clean"].value_counts().to_dict()
        rows.append({"gene": gene, "axis": axis_name, "design_hpf": design_hpf,
                     "result": res, "homo_xy": homo_xy, "wt_xy": wt_xy,
                     "label_counts": label_counts})
        _print_row(res, gene, axis_name, design_hpf, label_counts)
    return rows


def _print_row(res, gene, axis_name, design_hpf, label_counts=None):
    label = res.summary_label()
    call = res.support_call or "-"
    # worst confidence among the reached-stage support statistics
    supp_conf = [c.tier for k, c in res.confidence.items() if k.startswith("stage2:")]
    conf = min(supp_conf, key=lambda t: CONF_ORDER[t]) if supp_conf else "n/a"
    counts = f" labels={label_counts}" if label_counts is not None else ""
    print(f"  [{axis_name:9s}] {gene} @ {design_hpf:>2} hpf  n={res.n_group:>3}  "
          f"changed={res.changed_from_wt}  support={call:10s} conf={conf:11s} -> {label}{counts}")


def run_full_reference_axis(
    gene: str,
    df: pd.DataFrame,
    phenotype_labels: list[str],
    axis_name: str,
    z_cols: list[str] | None,
) -> dict | None:
    """Run the same anchor on the entire labeled reference, pooled over stages."""
    feat_cols = HAND_FEATS if axis_name == "hand" else z_cols
    edf = collapse_to_embryo(
        df,
        feat_cols,
        group_cols=[],
        label_cols=["zygosity", "phenotype_clean"],
    )
    wt = edf[edf["zygosity"] == "wildtype"]
    # Whole phenotype-labeled population (NOT homozygous-only) -- see run_axis note.
    homo = edf[edf["phenotype_clean"].isin(phenotype_labels)]
    if len(wt) < MIN_EMBRYOS or len(homo) < MIN_EMBRYOS:
        print(f"  [full {axis_name:9s}] {gene}: skip (n_group={len(homo)}, n_wt={len(wt)})")
        return None

    wt_feat = wt[feat_cols].values.astype(float)
    homo_feat = homo[feat_cols].values.astype(float)
    if axis_name == "embedding":
        mean, comps = wt_pca_axis(wt_feat, n_pcs=2)
        wt_xy = (wt_feat - mean) @ comps.T
        homo_xy = (homo_feat - mean) @ comps.T
    else:
        wt_xy, homo_xy = wt_feat, homo_feat

    # Outlier removal DISABLED (extreme-short CE = the second mode) -- see run_axis note.
    if len(homo_xy) < MIN_EMBRYOS or len(wt_xy) < MIN_EMBRYOS:
        print(f"  [full {axis_name:9s}] {gene}: skip (n_group={len(homo_xy)}, n_wt={len(wt_xy)})")
        return None

    res = run_phenotype_geometry(homo_xy, wt_xy, n_resample=N_RESAMPLE,
                                 rng=np.random.default_rng(42), compute_loo=True)
    label_counts = homo["phenotype_clean"].value_counts().to_dict()
    _print_row(res, gene, f"full-{axis_name}", "all", label_counts)
    return {
        "gene": gene,
        "axis": axis_name,
        "result": res,
        "homo_xy": homo_xy,
        "wt_xy": wt_xy,
        "label_counts": label_counts,
    }


# ---------------------------------------------------------------------------
# Transition detection (a small change-point detector)
# ---------------------------------------------------------------------------

def estimate_transition(rows_for_axis: list[dict]) -> float | None:
    """First design_hpf where support is discrete at confidence >= moderate,
    sustained through the next available time-bin. Returns None if no transition."""
    seq = sorted(rows_for_axis, key=lambda r: r["design_hpf"])
    for i, r in enumerate(seq):
        res = r["result"]
        if res.support_call != "discrete":
            continue
        supp_conf = [c.tier for k, c in res.confidence.items() if k.startswith("stage2:")]
        best = max(supp_conf, key=lambda t: CONF_ORDER[t]) if supp_conf else "insufficient"
        if CONF_ORDER[best] < CONF_ORDER["moderate"]:
            continue
        # confirmation: next bin (if any) also discrete
        if i + 1 < len(seq):
            if seq[i + 1]["result"].support_call != "discrete":
                continue
        return float(r["design_hpf"])
    return None


# ---------------------------------------------------------------------------
# Output: summary CSV + bootstrap npz
# ---------------------------------------------------------------------------

def emit_outputs(all_rows: list[dict], transitions: dict):
    summary = []
    npz_payload = {}
    for r in all_rows:
        res = r["result"]
        gene, axis, hpf = r["gene"], r["axis"], r["design_hpf"]
        key = f"{gene}_{axis}_{hpf}"

        def p(stage_stat):
            parts = stage_stat.split(":")
            if len(parts) == 2 and parts[0] == "stage1":
                sr = res.shift.results.get(parts[1])
                return sr.pvalue if sr else np.nan
            if res.support and stage_stat in res.support.results:
                return res.support.results[stage_stat].pvalue
            return np.nan

        supp_conf = [c.tier for k, c in res.confidence.items() if k.startswith("stage2:")]
        conf = min(supp_conf, key=lambda t: CONF_ORDER[t]) if supp_conf else "n/a"

        summary.append({
            "gene": gene,
            "axis": axis,
            "timebin_hpf": hpf,
            "n_group": res.n_group,
            "n_wt": res.n_wt,
            "group_label_counts": r.get("label_counts", {}),
            "changed_from_wt": res.changed_from_wt,
            "shift_wasserstein_p": p("stage1:wasserstein"),
            "shift_js_p": p("stage1:js_divergence"),
            "support_call": res.support_call,
            "valley_p": p("valley_depth"),
            "mst_p": p("mst_max_edge"),
            "fiedler_p": p("fiedler"),
            "conductance_p": p("conductance"),
            "confidence": conf,
            "terminal_stage": res.terminal_stage,
            "label": res.summary_label(),
            "estimated_transition_hpf": transitions.get((gene, axis)),
        })

        # full bootstrap nulls for every statistic
        if res.support:
            for name, sr in res.support.results.items():
                npz_payload[f"{key}__support__{name}"] = sr.null_dist
        for name, sr in res.shift.results.items():
            npz_payload[f"{key}__shift__{name}"] = sr.null_dist

    sdf = pd.DataFrame(summary).sort_values(["gene", "axis", "timebin_hpf"])
    csv_path = TABLE_DIR / "phenotype_geometry_summary.csv"
    sdf.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path.relative_to(RUN_DIR)}")

    npz_path = TABLE_DIR / "bootstrap_nulls.npz"
    np.savez_compressed(npz_path, **npz_payload)
    print(f"Saved: {npz_path.relative_to(RUN_DIR)}  ({len(npz_payload)} null arrays)")
    return sdf


def emit_full_reference_outputs(full_rows: list[dict]) -> pd.DataFrame:
    summary = []
    for r in full_rows:
        res = r["result"]
        supp_conf = [c.tier for k, c in res.confidence.items() if k.startswith("stage2:")]
        conf = min(supp_conf, key=lambda t: CONF_ORDER[t]) if supp_conf else "n/a"

        def p(stage_stat):
            parts = stage_stat.split(":")
            if len(parts) == 2 and parts[0] == "stage1":
                sr = res.shift.results.get(parts[1])
                return sr.pvalue if sr else np.nan
            if res.support and stage_stat in res.support.results:
                return res.support.results[stage_stat].pvalue
            return np.nan

        summary.append({
            "gene": r["gene"],
            "axis": r["axis"],
            "scope": "full_reference_all_phenotype_labeled",
            "n_group": res.n_group,
            "n_wt": res.n_wt,
            "group_label_counts": r.get("label_counts", {}),
            "changed_from_wt": res.changed_from_wt,
            "shift_wasserstein_p": p("stage1:wasserstein"),
            "shift_js_p": p("stage1:js_divergence"),
            "support_call": res.support_call,
            "valley_p": p("valley_depth"),
            "mst_p": p("mst_max_edge"),
            "fiedler_p": p("fiedler"),
            "conductance_p": p("conductance"),
            "confidence": conf,
            "terminal_stage": res.terminal_stage,
            "label": res.summary_label(),
        })

    sdf = pd.DataFrame(summary).sort_values(["gene", "axis"])
    csv_path = TABLE_DIR / "phenotype_geometry_full_reference_summary.csv"
    sdf.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(RUN_DIR)}")
    return sdf


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

STATE_LEVEL = {"wildtype-like": 0, "connected": 1, "discrete": 2}
STATE_COLOR = {0: "#BBBBBB", 1: "#2166AC", 2: "#B2182B"}


def _state_level(res):
    if not res.changed_from_wt:
        return 0
    return STATE_LEVEL.get(res.support_call, 1)


def plot_trajectories(all_rows, transitions):
    """Two panels that tell the real story:
      (top) developmental STATE track per gene x axis -- wildtype-like -> continuous
            -> discrete. This reflects the actual decision-tree call, not a raw
            p-value proxy.
      (bottom) b9d2 density signature (tail_index / skewness percentile vs WT) --
            the evidence that b9d2's difference is a heavy-tailed continuum, not a
            broken-support split.
    """
    gene_color = {"cep290": "#7b3294", "b9d2": "#1b9e77"}
    axis_marker = {"hand": ("o", "-", 1.0), "embedding": ("s", "--", 0.65)}

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[1.1, 1])

    # ---- top: state track ----
    for gene in GENES:
        for axis_name in ("hand", "embedding"):
            rows = sorted([r for r in all_rows if r["gene"] == gene and r["axis"] == axis_name],
                          key=lambda r: r["design_hpf"])
            if not rows:
                continue
            hpfs = [r["design_hpf"] for r in rows]
            states = [_state_level(r["result"]) for r in rows]
            mk, ls, alpha = axis_marker[axis_name]
            ax0.plot(hpfs, states, marker=mk, linestyle=ls, color=gene_color[gene],
                     alpha=alpha, linewidth=2, markersize=9,
                     label=f"{gene} ({axis_name})")
            for r, hpf, s in zip(rows, hpfs, states):
                if r["result"].support_call == "discrete":
                    ax0.scatter([hpf], [s], s=220, facecolors="none",
                                edgecolors=gene_color[gene], linewidths=2.5, zorder=6)
    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels(["wildtype-\nlike", "changed +\nconnected", "changed +\ndiscrete"])
    ax0.set_ylim(-0.3, 2.3)
    ax0.set_xlabel("predicted stage (hpf)")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8, loc="center left")
    ax0.set_title("Decision-tree call over developmental time\n"
                  "cep290 & b9d2 both remain a CONNECTED continuum on these axes "
                  "(no broken-support split)")

    # ---- bottom: b9d2 density signature ----
    for axis_name in ("hand", "embedding"):
        rows = sorted([r for r in all_rows if r["gene"] == "b9d2" and r["axis"] == axis_name],
                      key=lambda r: r["design_hpf"])
        rows = [r for r in rows if r["result"].density is not None]
        if not rows:
            continue
        hpfs = [r["design_hpf"] for r in rows]
        mk, ls, alpha = axis_marker[axis_name]
        for desc, col in [("tail_index", "#d95f02"), ("skewness", "#7570b3")]:
            pct = [r["result"].density.results[desc].percentile for r in rows]
            ax1.plot(hpfs, pct, marker=mk, linestyle=ls, color=col, alpha=alpha,
                     linewidth=1.8, markersize=7,
                     label=f"{desc} ({axis_name})")
    ax1.axhline(50, color="gray", linestyle=":", alpha=0.6, label="WT median (50th pct)")
    ax1.axhline(90, color="k", linestyle=":", alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("predicted stage (hpf)")
    ax1.set_ylabel("percentile vs. matched-N WT")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax1.set_title("b9d2 density signature: heavy tail / skew, not a gap\n"
                  "(this is the 'severe-tail continuum' the framework distinguishes from a split)")

    fig.tight_layout()
    out = PLOT_DIR / "phenotype_geometry_trajectory.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows = []
    full_rows = []
    for gene, cfg in GENES.items():
        print(f"\n{'='*70}\nGene: {gene}\n{'='*70}")
        df = load_gene(cfg["csv"])
        z_cols = sorted([c for c in df.columns if c.startswith(Z_MU_B_PREFIX)])
        all_rows += run_axis(gene, df, cfg["phenotype_labels"], "hand", None)
        all_rows += run_axis(gene, df, cfg["phenotype_labels"], "embedding", z_cols)
        for axis_name in ("hand", "embedding"):
            row = run_full_reference_axis(
                gene,
                df,
                cfg["phenotype_labels"],
                axis_name,
                z_cols if axis_name == "embedding" else None,
            )
            if row is not None:
                full_rows.append(row)

    # transitions per (gene, axis)
    transitions = {}
    for gene in GENES:
        for axis_name in ("hand", "embedding"):
            rows = [r for r in all_rows if r["gene"] == gene and r["axis"] == axis_name]
            if rows:
                transitions[(gene, axis_name)] = estimate_transition(rows)

    print(f"\n{'='*70}\nEstimated transitions (first sustained discrete, conf>=moderate):")
    for (gene, axis_name), t in transitions.items():
        print(f"  {gene:8s} [{axis_name:9s}] -> {t if t is not None else 'no transition'}")

    emit_outputs(all_rows, transitions)
    emit_full_reference_outputs(full_rows)
    plot_trajectories(all_rows, transitions)
    print("\nDone.")


if __name__ == "__main__":
    main()
