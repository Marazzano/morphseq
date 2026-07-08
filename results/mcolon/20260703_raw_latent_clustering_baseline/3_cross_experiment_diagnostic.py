"""
3_cross_experiment_diagnostic.py
----------------------------------
Diagnose whether the "columns" fragmentation in the raw z_mu_b condensation is
a pure experiment (batch) effect (already present at UMAP init, condensation
just preserves it) or something the condensation solver itself introduces/
amplifies -- and compare that against the classifier-margin arm (init'd from
margin-space UMAP, not raw z_mu_b) to see whether margin space starts/ends
with the same experiment contamination or is genuinely experiment-free.

Method: for each time bin, look only at embryos that co-occur in that bin
(overlapping time). Split all pairwise distances into same-experiment vs.
cross-experiment. Compute the ratio (cross-experiment dist / same-experiment
dist) at two stages, for BOTH arms:
  (1) x0        — UMAP init, pre-condensation
  (2) positions — final condensed output

Arms:
  raw    -- this folder's condensed_raw_zmub/condensed_positions.npz
  margin -- 20260407_pbx_analysis_cont pinned baseline
            (combined_raw_condensation_5class_bin4_perm500/condensed_positions.npz)

If the cross/same ratio is already >1 at init and grows (or stays roughly
equal) after condensation -> experiment separation is baked into the input
space; condensation is not the cause, at best it sharpens a pre-existing
split. If the ratio is ~1 at init but blows up after condensation -> the
solver itself is inducing/amplifying the experiment split (e.g. via
attract_k / outlier terms latching onto experiment-correlated local
neighborhoods).

Comparing raw vs. margin at the SAME stage tells us whether the classifier
margin projection is what suppresses experiment contamination (margin ratio
~1 throughout) or whether margin space carries some of it too, just less.

CAVEAT -- experiment overlap is NOT uniform across time bins. Only 3 PBX
experiments exist (20251207_pbx, 20260304, 20260306), and they were imaged
in a staggered, mostly non-overlapping window: 20260304 covers early bins,
20251207_pbx + 20260304 co-occur mid-range, 20260306 takes over late. A
cross-experiment ratio can only be computed where >=2 experiments are
present in the SAME bin -- bins with a single experiment are silently
skipped. In practice this means the ratio is only measured across a narrow
~48-76hpf overlap window, not the full developmental range. Any apparent
trend in the ratio over time bins should be read as a trend WITHIN that
overlap window only, not as "experiment effect strengthens/weakens with
development" -- there is no cross-experiment data outside it to check that
claim against. See _print_experiment_support() output for exact per-bin
counts.

Also included: a per-arm biology-preservation check (Luecken et al. 2022's
second axis, alongside the batch-mixing ratio above). Genotype labels are
used HERE ONLY to validate whether Harmony correction accidentally removed
real phenotype signal -- they are never used to fit the correction itself
(4_harmony_correction.py is label-free). We measure, per experiment
(so genotype separation is not itself contaminated by cross-experiment
distance), the mean pairwise distance between crispant genotypes and
inj_ctrl vs. the mean pairwise distance within inj_ctrl -- a simple
distance-based stand-in for the ASW/cLISI metrics used in the scRNA-seq
integration-benchmarking literature. If this separation shrinks materially
after Harmony correction relative to raw_zmub, that is the overcorrection
signal to watch for.

Outputs:
  tables/cross_experiment_diagnostic.csv              -- raw arm only (kept for continuity)
  tables/cross_experiment_diagnostic_compare.csv      -- all arms, long format
  tables/cross_experiment_diagnostic_by_genotype.csv  -- all arms, per genotype
  tables/biology_preservation_by_experiment.csv       -- all arms, genotype-separation check
  figures/cross_experiment_ratio_over_time.png            -- raw arm only
  figures/cross_experiment_ratio_raw_vs_margin.png         -- all arms, side by side
  figures/cross_experiment_ratio_by_genotype.png           -- all arms, per genotype panel
  figures/biology_preservation_vs_batch_mixing.png         -- overcorrection tradeoff scatter
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

_HERE = Path(__file__).resolve().parent
IN_DIR = _HERE / "figures" / "condensed_raw_zmub"
TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"

MARGIN_NPZ = (
    _HERE.parent
    / "20260407_pbx_analysis_cont"
    / "results" / "positioning" / "trajectory"
    / "combined_raw_condensation_5class_bin4_perm500"
    / "condensed_positions.npz"
)

HARMONY_THETAS = [1.0, 2.0, 4.0]

# arm_name -> npz path. Populated in main() once harmony paths are checked to exist.


def _experiment_of(embryo_id: str) -> str:
    # embryo_id like "20251207_pbx_A01_e01" or "20260304_A01_e01"
    parts = embryo_id.split("_")
    if parts[1].isalpha():  # e.g. "20251207_pbx_..."
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _same_vs_cross(coords: np.ndarray, experiment_labels: np.ndarray) -> tuple[float, float]:
    """coords: (n, d) positions present in this bin. Returns (mean_same, mean_cross)."""
    n = len(coords)
    if n < 3:
        return np.nan, np.nan
    d = squareform(pdist(coords))
    same_mask = experiment_labels[:, None] == experiment_labels[None, :]
    iu = np.triu_indices(n, k=1)
    same = same_mask[iu]
    dvals = d[iu]
    mean_same = dvals[same].mean() if same.any() else np.nan
    mean_cross = dvals[~same].mean() if (~same).any() else np.nan
    return mean_same, mean_cross


def _same_vs_cross_by_genotype(
    coords: np.ndarray, experiment_labels: np.ndarray, genotype_labels: np.ndarray
) -> list[dict]:
    """Restrict same/cross-experiment pairs to embryos sharing a genotype, so
    experiment separation is measured with phenotype held fixed. Returns one
    dict per genotype present with >=3 embryos and >=2 experiments."""
    rows = []
    for gt in sorted(set(genotype_labels)):
        gt_mask = genotype_labels == gt
        n_gt = gt_mask.sum()
        if n_gt < 3:
            continue
        gt_experiments = experiment_labels[gt_mask]
        if len(set(gt_experiments)) < 2:
            continue
        mean_same, mean_cross = _same_vs_cross(coords[gt_mask], gt_experiments)
        rows.append({
            "genotype": gt,
            "n_embryos": int(n_gt),
            "same_experiment_dist": mean_same,
            "cross_experiment_dist": mean_cross,
            "ratio": mean_cross / mean_same if mean_same else np.nan,
        })
    return rows


def _run_arm(npz_path: Path, arm_name: str) -> pd.DataFrame:
    print(f"\nLoading {arm_name} arm: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    x0 = data["x0"]                # (N_e, T, d)
    positions = data["positions"]  # (N_e, T, d)
    mask = data["mask"]            # (N_e, T)
    embryo_ids = data["embryo_ids"]
    time_values = data["time_values"]

    experiment_labels = np.array([_experiment_of(e) for e in embryo_ids])
    print(f"  Experiments found: {sorted(set(experiment_labels))}")

    rows = []
    for ti, t in enumerate(time_values):
        present = mask[:, ti]
        n_present = present.sum()
        if n_present < 3:
            continue
        labels_t = experiment_labels[present]
        if len(set(labels_t)) < 2:
            continue  # no cross-experiment pairs possible in this bin

        same_init, cross_init = _same_vs_cross(x0[present, ti, :], labels_t)
        same_final, cross_final = _same_vs_cross(positions[present, ti, :], labels_t)

        rows.append({
            "arm": arm_name,
            "time_bin": t,
            "n_embryos": int(n_present),
            "n_experiments": len(set(labels_t)),
            "same_experiment_dist_init": same_init,
            "cross_experiment_dist_init": cross_init,
            "ratio_init": cross_init / same_init if same_init else np.nan,
            "same_experiment_dist_final": same_final,
            "cross_experiment_dist_final": cross_final,
            "ratio_final": cross_final / same_final if same_final else np.nan,
        })

    return pd.DataFrame(rows)


def _run_arm_by_genotype(npz_path: Path, arm_name: str) -> pd.DataFrame:
    """Same/cross-experiment ratio per genotype per time bin per stage, with
    genotype held fixed within each comparison (isolates pure experiment
    effect from any real phenotype-driven divergence)."""
    data = np.load(npz_path, allow_pickle=True)

    x0 = data["x0"]
    positions = data["positions"]
    mask = data["mask"]
    embryo_ids = data["embryo_ids"]
    time_values = data["time_values"]
    genotypes = np.array([str(g) for g in data["labels"]])

    experiment_labels = np.array([_experiment_of(e) for e in embryo_ids])

    rows = []
    for ti, t in enumerate(time_values):
        present = mask[:, ti]
        if present.sum() < 3:
            continue
        e_t = experiment_labels[present]
        g_t = genotypes[present]

        init_rows = _same_vs_cross_by_genotype(x0[present, ti, :], e_t, g_t)
        final_rows = _same_vs_cross_by_genotype(positions[present, ti, :], e_t, g_t)
        final_by_gt = {r["genotype"]: r for r in final_rows}

        for r in init_rows:
            gt = r["genotype"]
            fr = final_by_gt.get(gt, {})
            rows.append({
                "arm": arm_name,
                "time_bin": t,
                "genotype": gt,
                "n_embryos": r["n_embryos"],
                "same_experiment_dist_init": r["same_experiment_dist"],
                "cross_experiment_dist_init": r["cross_experiment_dist"],
                "ratio_init": r["ratio"],
                "same_experiment_dist_final": fr.get("same_experiment_dist", np.nan),
                "cross_experiment_dist_final": fr.get("cross_experiment_dist", np.nan),
                "ratio_final": fr.get("ratio", np.nan),
            })

    return pd.DataFrame(rows)


def _biology_preservation(npz_path: Path, arm_name: str) -> pd.DataFrame:
    """Per-experiment, per-time-bin: mean pairwise distance between each
    crispant genotype and inj_ctrl, vs. mean pairwise distance within
    inj_ctrl. Computed WITHIN a single experiment (not cross-experiment) so
    this is a pure biology-separation measure, uncontaminated by the
    experiment-effect this whole diagnostic is about. Uses the FINAL
    condensed positions (the stage that matters for downstream clustering).

    Genotype labels used for validation only -- see module docstring.
    """
    data = np.load(npz_path, allow_pickle=True)
    positions = data["positions"]
    mask = data["mask"]
    embryo_ids = data["embryo_ids"]
    time_values = data["time_values"]
    genotypes = np.array([str(g) for g in data["labels"]])
    experiment_labels = np.array([_experiment_of(e) for e in embryo_ids])

    rows = []
    for ti, t in enumerate(time_values):
        present = mask[:, ti]
        if present.sum() < 6:
            continue
        exp_t = experiment_labels[present]
        gt_t = genotypes[present]
        pos_t = positions[present, ti, :]

        for exp in sorted(set(exp_t)):
            exp_mask = exp_t == exp
            if exp_mask.sum() < 6:
                continue
            gt_exp = gt_t[exp_mask]
            pos_exp = pos_t[exp_mask]

            ctrl_mask = gt_exp == "inj_ctrl"
            if ctrl_mask.sum() < 3:
                continue
            d = squareform(pdist(pos_exp))
            iu_ctrl = np.triu_indices(ctrl_mask.sum(), k=1)
            ctrl_idx = np.where(ctrl_mask)[0]
            within_ctrl = d[np.ix_(ctrl_idx, ctrl_idx)][iu_ctrl].mean() if ctrl_mask.sum() >= 2 else np.nan

            for gt in sorted(set(gt_exp)):
                if gt in ("inj_ctrl", "", "wik_ab"):
                    continue
                gt_mask = gt_exp == gt
                if gt_mask.sum() < 3:
                    continue
                gt_idx = np.where(gt_mask)[0]
                between = d[np.ix_(ctrl_idx, gt_idx)].mean()
                rows.append({
                    "arm": arm_name,
                    "time_bin": t,
                    "experiment": exp,
                    "genotype": gt,
                    "n_ctrl": int(ctrl_mask.sum()),
                    "n_genotype": int(gt_mask.sum()),
                    "within_ctrl_dist": within_ctrl,
                    "ctrl_vs_genotype_dist": between,
                    "separation_ratio": between / within_ctrl if within_ctrl else np.nan,
                })

    return pd.DataFrame(rows)


def _print_experiment_support(npz_path: Path, arm_name: str) -> None:
    """Show n_embryos per experiment per time bin, so it's clear which bins
    actually have cross-experiment overlap (only those bins can contribute
    a same/cross-experiment ratio -- bins with a single experiment present
    are silently skipped upstream)."""
    data = np.load(npz_path, allow_pickle=True)
    mask = data["mask"]
    embryo_ids = data["embryo_ids"]
    time_values = data["time_values"]
    experiment_labels = np.array([_experiment_of(e) for e in embryo_ids])

    rows = []
    for ti, t in enumerate(time_values):
        present = mask[:, ti]
        counts = pd.Series(experiment_labels[present]).value_counts().to_dict()
        rows.append({"time_bin": t, "n_experiments_present": len(counts), **counts})
    df = pd.DataFrame(rows).fillna(0)
    overlap_bins = (df["n_experiments_present"] >= 2).sum()
    print(f"\n[{arm_name}] experiment support over time "
          f"({overlap_bins}/{len(df)} bins have >=2 experiments -> only these contribute to the ratio):")
    print(df.to_string(index=False))


def _build_arm_registry() -> dict[str, Path]:
    arms = {
        "raw_zmub": IN_DIR / "condensed_positions.npz",
        "margin": MARGIN_NPZ,
    }
    for theta in HARMONY_THETAS:
        p = _HERE / "figures" / f"condensed_harmony_theta{theta:g}" / "condensed_positions.npz"
        if p.exists():
            arms[f"harmony_theta{theta:g}"] = p
        else:
            print(f"  (skipping harmony_theta{theta:g}: {p} not found yet)")
    return arms


ARM_COLORS = {
    "raw_zmub": "#B2182B",
    "margin": "#2166AC",
    "harmony_theta1": "#F7B267",
    "harmony_theta2": "#9467bd",
    "harmony_theta4": "#2CA02C",
}


def main() -> None:
    arms = _build_arm_registry()
    print(f"\nArms in this run: {list(arms.keys())}")

    for arm_name, npz_path in arms.items():
        _print_experiment_support(npz_path, arm_name)

    # ── Batch-mixing ratio, all arms ────────────────────────────────────────
    per_arm_dfs = {}
    for arm_name, npz_path in arms.items():
        df = _run_arm(npz_path, arm_name)
        per_arm_dfs[arm_name] = df
        print(df.drop(columns="arm").to_string(index=False))

    per_arm_dfs["raw_zmub"].drop(columns="arm").to_csv(TABLES / "cross_experiment_diagnostic.csv", index=False)
    print(f"Saved -> {TABLES / 'cross_experiment_diagnostic.csv'}")

    combined = pd.concat(per_arm_dfs.values(), ignore_index=True)
    out_csv = TABLES / "cross_experiment_diagnostic_compare.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    # ── Summary verdict per arm ────────────────────────────────────────────
    verdict_rows = []
    for arm_name, df in per_arm_dfs.items():
        mean_ratio_init = df["ratio_init"].mean()
        mean_ratio_final = df["ratio_final"].mean()
        verdict_rows.append({"arm": arm_name, "mean_ratio_init": mean_ratio_init, "mean_ratio_final": mean_ratio_final})
        print(f"\n[{arm_name}] mean cross/same ratio -- init: {mean_ratio_init:.3f}  final: {mean_ratio_final:.3f}")
        if mean_ratio_init > 1.05 and mean_ratio_final >= mean_ratio_init * 0.9:
            print(f"  VERDICT [{arm_name}]: experiment separation present at init and preserved/sharpened by condensation.")
        elif mean_ratio_final > mean_ratio_init * 1.3:
            print(f"  VERDICT [{arm_name}]: experiment separation grows substantially during condensation.")
        elif mean_ratio_init <= 1.05 and mean_ratio_final <= 1.05:
            print(f"  VERDICT [{arm_name}]: no meaningful experiment separation at either stage.")
        else:
            print(f"  VERDICT [{arm_name}]: ambiguous — inspect per-bin table and plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Plot: raw arm alone (kept for continuity with prior run) ──────────
    raw_df = per_arm_dfs["raw_zmub"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(raw_df["time_bin"], raw_df["ratio_init"], "o-", label="init (UMAP)", color="#2166AC")
    ax.plot(raw_df["time_bin"], raw_df["ratio_final"], "o-", label="final (condensed)", color="#B2182B")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="no experiment separation (ratio=1)")
    ax.set_xlabel("time bin (hpf)")
    ax.set_ylabel("cross-experiment / same-experiment mean distance")
    ax.set_title("Experiment separation over time: raw z_mu_b, init vs. final")
    ax.legend()
    fig.tight_layout()
    out_png = FIGURES / "cross_experiment_ratio_over_time.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved -> {out_png}")

    # ── Plot: all arms, both stages, side by side ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for arm_name, df in per_arm_dfs.items():
        color = ARM_COLORS.get(arm_name, None)
        axes[0].plot(df["time_bin"], df["ratio_init"], "o-", label=arm_name, color=color)
        axes[1].plot(df["time_bin"], df["ratio_final"], "o-", label=arm_name, color=color)
    for ax, title in zip(axes, ("UMAP init (pre-condensation)", "Final condensed positions")):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("time bin (hpf)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cross-experiment / same-experiment mean distance")
    fig.suptitle("Experiment separation: all arms")
    fig.tight_layout()
    out_png2 = FIGURES / "cross_experiment_ratio_raw_vs_margin.png"
    fig.savefig(out_png2, dpi=150)
    print(f"Saved -> {out_png2}")

    # ── Per-genotype breakdown (genotype held fixed within each comparison) ─
    print("\n" + "=" * 70)
    print("Per-genotype experiment-effect breakdown (same genotype, cross-experiment)")
    print("=" * 70)

    per_arm_gt_dfs = {arm_name: _run_arm_by_genotype(npz_path, arm_name) for arm_name, npz_path in arms.items()}
    combined_gt = pd.concat(per_arm_gt_dfs.values(), ignore_index=True)

    out_csv_gt = TABLES / "cross_experiment_diagnostic_by_genotype.csv"
    combined_gt.to_csv(out_csv_gt, index=False)
    print(f"Saved -> {out_csv_gt}")

    summary = (
        combined_gt.groupby(["arm", "genotype"])[["ratio_init", "ratio_final"]]
        .mean()
        .reset_index()
        .sort_values(["arm", "genotype"])
    )
    print(summary.to_string(index=False))

    genotypes = sorted(combined_gt["genotype"].unique())
    n_gt = len(genotypes)
    fig, axes = plt.subplots(1, n_gt, figsize=(5 * n_gt, 5), sharey=True)
    if n_gt == 1:
        axes = [axes]

    for ax, gt in zip(axes, genotypes):
        for arm_name, gt_df in per_arm_gt_dfs.items():
            sub = gt_df[gt_df["genotype"] == gt]
            color = ARM_COLORS.get(arm_name, None)
            ax.plot(sub["time_bin"], sub["ratio_final"], "o-", color=color, label=arm_name)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(gt)
        ax.set_xlabel("time bin (hpf)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("cross-experiment / same-experiment mean distance\n(genotype held fixed, final)")

    fig.suptitle("Experiment separation per genotype: all arms")
    fig.tight_layout()
    out_png3 = FIGURES / "cross_experiment_ratio_by_genotype.png"
    fig.savefig(out_png3, dpi=150)
    print(f"Saved -> {out_png3}")

    # ── Biology preservation check (overcorrection sentinel) ───────────────
    print("\n" + "=" * 70)
    print("Biology-preservation check: inj_ctrl vs. crispant separation, WITHIN experiment")
    print("(genotype used for validation only, not for fitting the correction)")
    print("=" * 70)

    bio_dfs = {arm_name: _biology_preservation(npz_path, arm_name) for arm_name, npz_path in arms.items()}
    combined_bio = pd.concat(bio_dfs.values(), ignore_index=True)
    out_csv_bio = TABLES / "biology_preservation_by_experiment.csv"
    combined_bio.to_csv(out_csv_bio, index=False)
    print(f"Saved -> {out_csv_bio}")

    bio_summary = (
        combined_bio.groupby("arm")["separation_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"separation_ratio": "mean_genotype_separation_ratio"})
    )
    print(bio_summary.to_string(index=False))

    # ── Overcorrection tradeoff scatter: batch-mixing (lower=better, toward 1)
    #    vs. biology-preservation (higher=better, genotype separation ratio) ─
    fig, ax = plt.subplots(figsize=(7, 6))
    for arm_name in arms:
        mean_batch_ratio = per_arm_dfs[arm_name]["ratio_final"].mean()
        mean_bio_ratio = bio_dfs[arm_name]["separation_ratio"].mean() if len(bio_dfs[arm_name]) else np.nan
        color = ARM_COLORS.get(arm_name, None)
        ax.scatter(mean_batch_ratio, mean_bio_ratio, s=120, color=color, label=arm_name, zorder=3)
        ax.annotate(arm_name, (mean_batch_ratio, mean_bio_ratio), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("mean cross/same-experiment ratio, final (lower/closer-to-1 = better batch mixing)")
    ax.set_ylabel("mean inj_ctrl-vs-crispant separation ratio, final (higher = more biology preserved)")
    ax.set_title("Overcorrection tradeoff: batch mixing vs. biology preservation")
    fig.tight_layout()
    out_png4 = FIGURES / "biology_preservation_vs_batch_mixing.png"
    fig.savefig(out_png4, dpi=150)
    print(f"Saved -> {out_png4}")

    # ── Recommendation ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    harmony_arms = [a for a in arms if a.startswith("harmony_theta")]
    if harmony_arms:
        raw_batch_ratio = per_arm_dfs["raw_zmub"]["ratio_final"].mean()
        raw_bio_ratio = bio_dfs["raw_zmub"]["separation_ratio"].mean() if len(bio_dfs["raw_zmub"]) else np.nan
        candidates = []
        for arm_name in harmony_arms:
            batch_ratio = per_arm_dfs[arm_name]["ratio_final"].mean()
            bio_ratio = bio_dfs[arm_name]["separation_ratio"].mean() if len(bio_dfs[arm_name]) else np.nan
            improved_mixing = batch_ratio < raw_batch_ratio
            preserved_bio = (not np.isnan(bio_ratio)) and (bio_ratio >= 0.85 * raw_bio_ratio)
            candidates.append((arm_name, batch_ratio, bio_ratio, improved_mixing, preserved_bio))
            print(f"  {arm_name}: batch_ratio={batch_ratio:.3f} (raw={raw_batch_ratio:.3f}), "
                  f"bio_ratio={bio_ratio:.3f} (raw={raw_bio_ratio:.3f}), "
                  f"improved_mixing={improved_mixing}, preserved_bio(>=85% of raw)={preserved_bio}")
        valid = [c for c in candidates if c[3] and c[4]]
        if valid:
            best = min(valid, key=lambda c: c[1])
            print(f"\n  -> Recommended: {best[0]} (best batch-mixing among candidates that preserved biology)")
        else:
            print("\n  -> No theta both improved batch-mixing AND preserved biology within tolerance. "
                  "Inspect biology_preservation_vs_batch_mixing.png before choosing.")


if __name__ == "__main__":
    main()
