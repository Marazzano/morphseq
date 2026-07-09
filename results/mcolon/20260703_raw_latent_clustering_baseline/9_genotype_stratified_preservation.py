"""
9_genotype_stratified_preservation.py
-------------------------------------
Script 8 showed the classifier-margin space PARTIALLY reorganizes the raw z_mu_b
manifold (worse than a fair per-bin 10-dim PCA control), scale-dependent, worst
late. That pooled all genotypes. This script asks the sharper question:

  Is the excess (beyond-compression) margin reorganization SELECTIVE for
  crispants, or GENERIC (also hits equivalent controls), or BATCH-LINKED
  (drawn along experiment identity)?

Core estimand, per genotype g, neighborhood size k, time bin t:
  D_g(k,t)  = O_g^PC10(k,t) - O_g^margin(k,t)     classifier-specific loss
                                                  beyond dimensional compression
  DD_g(k,t) = D_g(k,t) - D_controls(k,t)          excess vs equivalent controls
              (D_controls = embryo-row-weighted pool of WT + inj_ctrl)

Controls WT (wik_ab) and inj_ctrl are nominally EQUIVALENT and kept SEPARATE.
The primary control property is D_WT ~= D_inj (agreement), NOT D~=0: controls may
be poorly preserved (compressed margin region) yet still "correct" if the two
equivalent cohorts agree. Crispant selectivity is judged vs that baseline.

Spaces (identical to script 8, per-bin z-scored, per-bin PCA):
  raw80    80-dim z_mu_b     (tables/pbx_binned_zmub_with_wt.csv, built here w/ WT)
  margin10 pairwise margins  (combined_pairwise_5class_bin4_perm500/pairwise_raw_vectors.csv)
  rawPC10  first 10 PCs of raw80 (fair-compression control)

Neighbor-composition analyses (corrections 4-6) use availability baselines and
matched nulls so genotype abundance and experiment structure cannot masquerade
as biology.

Replication gate runs WT-EXCLUDED and must reproduce script 8's pooled
0.27 @ k=10 / 0.66 @ k=50. WT-inclusive "analysis mode" is expected to differ
(WT changes every candidate pool) and is never compared to the old numbers.

Outputs: see bottom of main().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── reuse script 8's self-contained primitives ───────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from importlib import import_module
_s8 = import_module("8_neighbor_preservation")
impute_margin = _s8.impute_margin
zscore = _s8.zscore
pca_project = _s8.pca_project
knn_indices = _s8.knn_indices
mean_overlap = _s8.mean_overlap

TABLES = _HERE / "tables"
FIGURES = _HERE / "figures"

MARGIN_CSV = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/"
    "20260407_pbx_analysis_cont/results/positioning/pairwise/"
    "combined_pairwise_5class_bin4_perm500/pairwise_raw_vectors.csv"
)
RAW_WT_CSV = TABLES / "pbx_binned_zmub_with_wt.csv"

CONTROLS = ("wik_ab", "inj_ctrl")
CRISPANTS = ("pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant")
GROUPS = (*CONTROLS, *CRISPANTS)

K_VALUES = (5, 10, 15, 30, 50)
WITHIN_K = (3, 8)   # local k=3 + broader k=8 (ordering robustness check)
COMPOSITION_K = 10
PC_D = 10
MIN_BIN_N = max(K_VALUES) + 1        # bin needs enough embryos for full-pop kNN
N_BOOT = 500
RANDOM_STATE = 0


# ── WT-inclusive raw-latent table (correction: don't mutate 0_load) ───────────
def build_raw_with_wt():
    """Load PBX latents INCLUDING wik_ab (WT), bin 4hpf, persist. Mirrors
    0_load_pbx_latents.py exactly except the genotype list."""
    if RAW_WT_CSV.exists():
        return pd.read_csv(RAW_WT_CSV, low_memory=False)
    sys.path.insert(0, str(_HERE.parent / "20260329_pbx_crispant_analysis_cont"))
    from common import load_bridge_ready_dataframe, BRIDGE_PLUS_WIK_AB_GENOTYPES
    from analyze.utils.binning import bin_embryos_by_time

    df = load_bridge_ready_dataframe(genotypes=BRIDGE_PLUS_WIK_AB_GENOTYPES)
    df = df[df["stage_hpf_bridge"].notna()].copy()
    binned = bin_embryos_by_time(df, time_col="stage_hpf_bridge", bin_width=4.0)
    binned.to_csv(RAW_WT_CSV, index=False)
    print(f"Built {RAW_WT_CSV.name}: {len(binned)} rows, "
          f"{binned.embryo_id.nunique()} embryos")
    return binned


def load_joined(include_wt: bool):
    """Inner-join raw80 + margin10 + experiment_id on (embryo_id, time_bin).
    include_wt=False drops wik_ab (replication mode)."""
    raw = build_raw_with_wt()
    mar = pd.read_csv(MARGIN_CSV, low_memory=False)
    z_cols = [c for c in raw.columns if "z_mu_b" in c]
    m_cols = [c for c in mar.columns if "__vs__" in c]
    raw_keep = raw[["embryo_id", "time_bin", "genotype", *z_cols]]
    mar_keep = mar[["embryo_id", "time_bin", "experiment_id", *m_cols]]
    j = raw_keep.merge(mar_keep, on=["embryo_id", "time_bin"], how="inner")
    if not include_wt:
        j = j[j.genotype != "wik_ab"].copy()
    return j, z_cols, m_cols


# ── per-bin spaces + neighbor sets ────────────────────────────────────────────
def bin_spaces(sub, z_cols, m_cols):
    """Per-bin z-scored raw80, margin10, rawPC10 (all fit within this bin)."""
    raw80 = zscore(sub[z_cols].values.astype(float))
    margin = zscore(impute_margin(sub[m_cols].values.astype(float)))
    rawPCd = zscore(pca_project(sub[z_cols].values.astype(float), PC_D))
    return raw80, margin, rawPCd


def per_focal_overlap(nn_a, nn_b, k):
    """Per-row overlap fraction |A_i∩B_i|/k (not averaged)."""
    n = nn_a.shape[0]
    out = np.empty(n)
    for i in range(n):
        a = set(nn_a[i, :k].tolist())
        b = set(nn_b[i, :k].tolist())
        out[i] = len(a & b) / k
    return out


# ── (1)+(2) genotype-stratified preservation & diff-in-diff ───────────────────
def preservation_by_bin(joined, z_cols, m_cols):
    """Per (genotype, k, bin): O_margin, O_PC10, D, n  (full-population kNN)."""
    recs = []
    kmax = max(K_VALUES)
    for b, sub in joined.groupby("time_bin"):
        sub = sub.reset_index(drop=True)
        if len(sub) < MIN_BIN_N:
            continue
        raw80, margin, rawPCd = bin_spaces(sub, z_cols, m_cols)
        nn_raw = knn_indices(raw80, kmax)
        nn_mar = knn_indices(margin, kmax)
        nn_pcd = knn_indices(rawPCd, kmax)
        geno = sub.genotype.values
        for k in K_VALUES:
            ov_m = per_focal_overlap(nn_raw, nn_mar, k)   # raw vs margin
            ov_p = per_focal_overlap(nn_raw, nn_pcd, k)   # raw vs PC10 control
            for g in GROUPS:
                mask = geno == g
                if not mask.any():
                    continue
                Om = ov_m[mask].mean()
                Op = ov_p[mask].mean()
                recs.append(dict(time_bin=b, genotype=g, k=k, n=int(mask.sum()),
                                 O_margin=Om, O_PC10=Op, D=Op - Om))
    return pd.DataFrame(recs)


def pool_estimands(pb):
    """Embryo-row-weighted (primary) + bin-weighted (sensitivity) pools per (g,k)."""
    rows = []
    for (g, k), grp in pb.groupby(["genotype", "k"]):
        w = grp.n.values
        rows.append(dict(
            genotype=g, k=k,
            O_margin_rowwt=np.average(grp.O_margin, weights=w),
            O_PC10_rowwt=np.average(grp.O_PC10, weights=w),
            D_rowwt=np.average(grp.D, weights=w),
            O_margin_binwt=grp.O_margin.mean(),
            D_binwt=grp.D.mean(),
            n_total=int(w.sum()), n_bins=len(grp),
        ))
    return pd.DataFrame(rows)


def diff_in_diff(pb):
    """DD_g(k,t) = D_g - D_controls, with D_controls = row-weighted WT+inj pool.
    Returns per-(g,k,bin) DD plus row-weighted and bin-weighted pooled DD."""
    ctrl = pb[pb.genotype.isin(CONTROLS)]
    # D_controls per (k,bin): embryo-row weighted over the two control groups
    dc = (ctrl.groupby(["k", "time_bin"])
              .apply(lambda gp: np.average(gp.D, weights=gp.n), include_groups=False)
              .rename("D_controls").reset_index())
    m = pb.merge(dc, on=["k", "time_bin"], how="left")
    m["DD"] = m.D - m.D_controls
    pooled = []
    for (g, k), grp in m.groupby(["genotype", "k"]):
        pooled.append(dict(genotype=g, k=k,
                           DD_rowwt=np.average(grp.DD, weights=grp.n),
                           DD_binwt=grp.DD.mean(), n_total=int(grp.n.sum())))
    return m, pd.DataFrame(pooled)


# ── (3) control-vs-control consistency w/ bootstrap ───────────────────────────
def control_consistency(pb, joined, z_cols, m_cols):
    """|O_WT - O_inj|(k,t) with bootstrap CI on each group's O_margin.
    Bootstrap resamples focal embryos within (group,bin); neighbor sets fixed."""
    rng = np.random.default_rng(RANDOM_STATE)
    kmax = max(K_VALUES)
    rows = []
    for b, sub in joined.groupby("time_bin"):
        sub = sub.reset_index(drop=True)
        if len(sub) < MIN_BIN_N:
            continue
        raw80, margin, _ = bin_spaces(sub, z_cols, m_cols)
        nn_raw = knn_indices(raw80, kmax)
        nn_mar = knn_indices(margin, kmax)
        geno = sub.genotype.values
        for k in K_VALUES:
            ov = per_focal_overlap(nn_raw, nn_mar, k)
            stats = {}
            for g in CONTROLS:
                idx = np.where(geno == g)[0]
                if len(idx) == 0:
                    stats[g] = (np.nan, np.nan, np.nan, 0)
                    continue
                vals = ov[idx]
                boot = [rng.choice(vals, len(vals), replace=True).mean()
                        for _ in range(N_BOOT)]
                stats[g] = (vals.mean(), np.percentile(boot, 2.5),
                            np.percentile(boot, 97.5), len(idx))
            (o_wt, wt_lo, wt_hi, n_wt) = stats["wik_ab"]
            (o_inj, inj_lo, inj_hi, n_inj) = stats["inj_ctrl"]
            rows.append(dict(time_bin=b, k=k, O_WT=o_wt, O_inj=o_inj,
                             gap=abs(o_wt - o_inj) if np.isfinite(o_wt) and np.isfinite(o_inj) else np.nan,
                             WT_lo=wt_lo, WT_hi=wt_hi, inj_lo=inj_lo, inj_hi=inj_hi,
                             n_WT=n_wt, n_inj=n_inj))
    return pd.DataFrame(rows)


# ── (4)-(6) availability-adjusted / matched-null neighbor composition ─────────
def neighbor_composition(joined, z_cols, m_cols, k=COMPOSITION_K):
    """Lost AND gained neighbor composition at fixed k.

    The headline claim is that margin-space GAINS same-genotype neighbors after
    adjusting for class balance. That claim is only robust if the null is the set
    of embryos ACTUALLY ELIGIBLE to become a gained neighbor. A gained neighbor of
    focal i cannot be i itself, and cannot be an embryo already in N_i^raw (it would
    then be retained, not gained). So the correct per-focal replacement pool is

        C_i = bin \\ ({i} u N_i^raw)

    and the same-genotype/same-experiment baselines are the shares WITHIN C_i, not
    within the whole bin. Using whole-bin shares (the old code) is wrong in both
    directions: if raw neighbors were already same-genotype they are removed from
    C_i (baseline should drop, old null over-states it, R_gen deflated); if raw
    neighbors were cross-genotype the pool is same-genotype-rich (old null
    under-states it, R_gen inflated). Conditioning on C_i removes that confound.

    Lost neighbors L_i = N_i^raw \\ N_i^margin are scored symmetrically against the
    pool of embryos eligible to be RETAINED-then-dropped, C^raw_i = N_i^raw \\ {i}
    (the raw neighborhood is the only place a lost neighbor can come from), so the
    natural null there is the raw-neighborhood composition itself.

    Returns (per_focal_df, availability_df)."""
    focal_rows = []
    avail_rows = []
    for b, sub in joined.groupby("time_bin"):
        sub = sub.reset_index(drop=True)
        n_t = len(sub)
        if n_t < MIN_BIN_N:
            continue
        raw80, margin, _ = bin_spaces(sub, z_cols, m_cols)
        nn_raw = knn_indices(raw80, k)
        nn_mar = knn_indices(margin, k)
        geno = sub.genotype.values
        expt = sub.experiment_id.astype(str).values
        all_idx = np.arange(n_t)

        # availability-adjusted GAINED genotype shares, per focal genotype, using
        # the per-focal eligible pool C_i averaged over focal embryos of genotype g
        for g in GROUPS:
            fmask = np.where(geno == g)[0]
            if len(fmask) == 0:
                continue
            for h in GROUPS:
                adj_vals = []  # P_gained(h) - avail_C_i(h), per focal
                for i in fmask:
                    G_i = set(nn_mar[i].tolist()) - set(nn_raw[i].tolist())
                    if not G_i:
                        continue
                    # eligible replacement pool C_i = bin \ ({i} u N_i^raw)
                    excl = set(nn_raw[i].tolist()) | {i}
                    C_i = all_idx[~np.isin(all_idx, list(excl))]
                    if len(C_i) == 0:
                        continue
                    avail_h = np.mean(geno[C_i] == h)
                    gi = list(G_i)
                    adj_vals.append(np.mean(geno[gi] == h) - avail_h)
                if adj_vals:
                    avail_rows.append(dict(time_bin=b, focal_genotype=g, kind="genotype",
                                           target=h, adj=np.mean(adj_vals),
                                           n_focal=len(adj_vals)))

        # per-focal matched-null enrichment against C_i, for gained AND lost
        for i in range(n_t):
            e_i, g_i = expt[i], geno[i]
            raw_set = set(nn_raw[i].tolist())
            G_i = np.array(sorted(set(nn_mar[i].tolist()) - raw_set), dtype=int)
            L_i = np.array(sorted(raw_set - set(nn_mar[i].tolist())), dtype=int)

            # eligible replacement pool for a gained neighbor
            excl = raw_set | {i}
            C_i = all_idx[~np.isin(all_idx, list(excl))]
            if len(C_i) == 0:
                continue
            E_exp = np.mean(expt[C_i] == e_i)   # conditioned baseline (correction 7/10)
            E_gen = np.mean(geno[C_i] == g_i)

            rec = dict(time_bin=b, genotype=g_i, experiment=e_i)

            if len(G_i):
                obs_gen = np.mean(geno[G_i] == g_i)
                obs_exp = np.mean(expt[G_i] == e_i)
                rec.update(n_gained=len(G_i),
                           obs_gen=obs_gen, null_gen=E_gen,   # observed vs chance (genotype)
                           obs_exp=obs_exp, null_exp=E_exp,   # observed vs chance (experiment)
                           R_gen=obs_gen - E_gen,
                           R_exp=obs_exp - E_exp)
                # correction 6: among same-genotype GAINED, same-experiment?
                same_geno = G_i[geno[G_i] == g_i]
                if len(same_geno):
                    # baseline = same-experiment share among same-genotype eligibles
                    C_gen = C_i[geno[C_i] == g_i]
                    base = np.mean(expt[C_gen] == e_i) if len(C_gen) else np.nan
                    cond_exp = np.mean(expt[same_geno] == e_i)
                    rec.update(cond_exp=cond_exp, cond_exp_base=base,
                               cond_exp_enrich=(cond_exp - base)
                               if np.isfinite(base) else np.nan)
            else:
                rec.update(n_gained=0, R_gen=np.nan, R_exp=np.nan)

            # LOST-neighbor composition: null = the raw neighborhood itself
            if len(L_i):
                C_raw = np.array(sorted(raw_set), dtype=int)
                Lg = np.mean(geno[C_raw] == g_i)
                Le = np.mean(expt[C_raw] == e_i)
                rec.update(n_lost=len(L_i),
                           Lost_gen=np.mean(geno[L_i] == g_i) - Lg,
                           Lost_exp=np.mean(expt[L_i] == e_i) - Le)
            else:
                rec.update(n_lost=0, Lost_gen=np.nan, Lost_exp=np.nan)

            focal_rows.append(rec)
    return pd.DataFrame(focal_rows), pd.DataFrame(avail_rows)


# ── (SECONDARY) within-group preservation, adaptive local k ───────────────────
def within_group_preservation(joined, z_cols, m_cols):
    """Same-genotype candidates only. Local k=3 plus a broader k to show the
    genotype ordering isn't a k=3 artifact. Locality guard k <= floor((n-1)/2)."""
    recs = []
    for b, sub in joined.groupby("time_bin"):
        for g, gsub in sub.groupby("genotype"):
            gsub = gsub.reset_index(drop=True)
            n = len(gsub)
            k_cap = (n - 1) // 2
            for k in WITHIN_K:
                if k > k_cap or n < k + 1:
                    continue
                raw80, margin, _ = bin_spaces(gsub, z_cols, m_cols)
                nn_raw = knn_indices(raw80, k)
                nn_mar = knn_indices(margin, k)
                ov, _ = mean_overlap(nn_raw, nn_mar, k)
                recs.append(dict(time_bin=b, genotype=g, k=k, n=n, O_within=ov))
    return pd.DataFrame(recs)


SCATTER_K = 5   # fine-structure scale; churn = neighbors-changed/k is discrete,
                # so we show a violin per (churn level × genotype), not a scatter


# ── how much of the reorganization is genotype-aligned? (per-embryo) ──────────
def reorg_vs_alignment(joined, z_cols, m_cols, k=SCATTER_K):
    """Per focal embryo, at fixed k, decompose the raw→margin reorganization into
    'how much churned' vs 'how genotype-aligned was the churn':

        churn_i = 1 - |N_i^raw ∩ N_i^margin| / k        (0 = unchanged, 1 = all new)
        R_gen_i = same-genotype share of ADDED neighbors
                  - same-genotype share of the eligible pool C_i               (1a, per embryo)

    R_gen is null-subtracted (chance genotype share among embryos actually
    eligible to be gained), so a churn→R_gen slope is NOT the added-set-denominator
    tautology: it asks whether embryos that reorganize MORE do so preferentially
    TOWARD their own genotype. Slope≈0 with high churn = reorganization driven by
    something other than genotype (the residual we can't get a clean null for)."""
    rows = []
    for b, sub in joined.groupby("time_bin"):
        sub = sub.reset_index(drop=True)
        n_t = len(sub)
        if n_t < MIN_BIN_N:
            continue
        raw80, margin, _ = bin_spaces(sub, z_cols, m_cols)
        nn_raw = knn_indices(raw80, k)
        nn_mar = knn_indices(margin, k)
        geno = sub.genotype.values
        eid = sub.embryo_id.values
        all_idx = np.arange(n_t)
        for i in range(n_t):
            raw_set = set(nn_raw[i].tolist())
            mar_set = set(nn_mar[i].tolist())
            churn = 1.0 - len(raw_set & mar_set) / k
            G_i = np.array(sorted(mar_set - raw_set), dtype=int)
            excl = raw_set | {i}
            C_i = all_idx[~np.isin(all_idx, list(excl))]
            if len(G_i) == 0 or len(C_i) == 0:
                r_gen = np.nan
            else:
                obs = np.mean(geno[G_i] == geno[i])
                null = np.mean(geno[C_i] == geno[i])
                r_gen = obs - null
            rows.append(dict(embryo_id=eid[i], time_bin=b, genotype=geno[i],
                             k=k, churn=churn, n_gained=len(G_i), R_gen=r_gen))
    return pd.DataFrame(rows)


# ── artifact paths (single source of truth for the compute/plot split) ────────
PB_CSV       = TABLES / "preservation_by_genotype_bin.csv"
POOLED_CSV   = TABLES / "preservation_pooled.csv"
DD_BIN_CSV   = TABLES / "preservation_diff_in_diff_by_bin.csv"
DD_POOL_CSV  = TABLES / "preservation_diff_in_diff.csv"
CONS_CSV     = TABLES / "control_consistency.csv"
COMP_CSV     = TABLES / "neighbor_composition.csv"
COMP_AV_CSV  = TABLES / "neighbor_composition_availability.csv"
WG_CSV       = TABLES / "within_group_preservation.csv"
REORG_CSV    = TABLES / "reorg_vs_alignment.csv"


def compute_artifacts():
    """Heavy compute (bootstraps, per-focal composition). Writes every table the
    figure needs to disk, then returns them. Run once; re-plot from disk after."""
    # ── replication gate: WT-EXCLUDED, reproduce script 8 pooled numbers ──────
    jx, zc, mc = load_joined(include_wt=False)
    pbx_rep = preservation_by_bin(jx, zc, mc)
    # pooled over ALL non-WT genotypes, n-weighted (matches script 8's pooling)
    rep = (pbx_rep.groupby("k")
                  .apply(lambda gp: np.average(gp.O_margin, weights=gp.n), include_groups=False)
                  .rename("O_margin_pooled"))
    r10, r50 = rep.get(10, np.nan), rep.get(50, np.nan)
    gate = "PASS" if (abs(r10 - 0.27) < 0.02 and abs(r50 - 0.66) < 0.02) else "WARN"
    print(f"[Replication gate | WT-EXCLUDED] pooled O_margin k=10={r10:.3f} "
          f"(expect 0.27), k=50={r50:.3f} (expect 0.66) -> {gate}\n")

    # ── analysis mode: WT INCLUDED ────────────────────────────────────────────
    j, zc, mc = load_joined(include_wt=True)
    print(f"[Analysis | WT-INCLUDED] {len(j)} embryo×bin rows, "
          f"{j.embryo_id.nunique()} embryos, {j.time_bin.nunique()} bins")
    print(f"  genotype counts: {j.genotype.value_counts().to_dict()}\n")

    pb = preservation_by_bin(j, zc, mc)
    pb.to_csv(PB_CSV, index=False)
    pooled = pool_estimands(pb)
    pooled.to_csv(POOLED_CSV, index=False)

    dd_bin, dd_pool = diff_in_diff(pb)
    dd_bin.to_csv(DD_BIN_CSV, index=False)
    dd_pool.to_csv(DD_POOL_CSV, index=False)

    cons = control_consistency(pb, j, zc, mc)
    cons.to_csv(CONS_CSV, index=False)

    comp_focal, comp_avail = neighbor_composition(j, zc, mc)
    comp_focal.to_csv(COMP_CSV, index=False)
    comp_avail.to_csv(COMP_AV_CSV, index=False)

    wg = within_group_preservation(j, zc, mc)
    wg.to_csv(WG_CSV, index=False)

    reorg = reorg_vs_alignment(j, zc, mc)
    reorg.to_csv(REORG_CSV, index=False)

    return dict(pooled=pooled, dd_bin=dd_bin, dd_pool=dd_pool, cons=cons,
                comp_focal=comp_focal, pb=pb, wg=wg, reorg=reorg)


def load_artifacts():
    """Read the tables compute_artifacts() wrote. Used by --plot-only."""
    missing = [p.name for p in (PB_CSV, POOLED_CSV, DD_BIN_CSV, DD_POOL_CSV,
                                CONS_CSV, COMP_CSV, WG_CSV, REORG_CSV) if not p.exists()]
    if missing:
        raise SystemExit(f"--plot-only needs precomputed tables; missing: {missing}\n"
                         f"Run without --plot-only once to build them.")
    return dict(
        pooled=pd.read_csv(POOLED_CSV), dd_bin=pd.read_csv(DD_BIN_CSV),
        dd_pool=pd.read_csv(DD_POOL_CSV), cons=pd.read_csv(CONS_CSV),
        comp_focal=pd.read_csv(COMP_CSV), pb=pd.read_csv(PB_CSV),
        wg=pd.read_csv(WG_CSV), reorg=pd.read_csv(REORG_CSV),
    )


def report_and_plot(A):
    """Console report + figure, from artifacts (dict from compute/load)."""
    pooled, dd_pool, cons = A["pooled"], A["dd_pool"], A["cons"]
    comp_focal, pb, wg, dd_bin = A["comp_focal"], A["pb"], A["wg"], A["dd_bin"]
    reorg = A["reorg"]

    # ── console report ────────────────────────────────────────────────────────
    print("=== O_margin(k) by genotype (embryo-row weighted) ===")
    piv = pooled.pivot(index="genotype", columns="k", values="O_margin_rowwt")
    print(piv.reindex(GROUPS).round(3).to_string())

    print("\n=== D_g (row-wt) and ΔD_g (row-wt, vs pooled controls) at k=10 ===")
    d10 = pooled[pooled.k == 10].set_index("genotype")["D_rowwt"]
    dd10 = dd_pool[dd_pool.k == 10].set_index("genotype")["DD_rowwt"]
    for g in GROUPS:
        tag = "control" if g in CONTROLS else "crispant"
        print(f"  {g:22s} ({tag:8s})  D={d10.get(g, np.nan):+.3f}  ΔD={dd10.get(g, np.nan):+.3f}")

    print("\n=== Control consistency |O_WT - O_inj| (mean over bins) ===")
    for k in K_VALUES:
        sub = cons[cons.k == k]
        print(f"  k={k:2d}: mean gap {sub.gap.mean():.3f}  "
              f"(WT n/bin {sub.n_WT.min()}–{sub.n_WT.max()})")

    print("\n=== Gained-neighbor enrichment vs eligible-pool null C_i (mean over focal) ===")
    ecols = ["R_gen", "R_exp", "cond_exp_enrich"]
    ef = comp_focal.groupby("genotype")[ecols].mean()
    print(ef.reindex(GROUPS).round(3).to_string())
    print("  R_gen>0 = margin GAINS same-genotype neighbors beyond class balance "
          "(the headline finding); R_exp>0 = batch; cond_exp_enrich>0 = same-genotype "
          "gains still prefer same experiment (batch within biology)")

    print("\n=== Lost-neighbor composition vs raw-neighborhood null (mean over focal) ===")
    lf = comp_focal.groupby("genotype")[["Lost_gen", "Lost_exp"]].mean()
    print(lf.reindex(GROUPS).round(3).to_string())
    print("  Lost_gen<0 = dropped neighbors are LESS same-genotype than the raw "
          "neighborhood (margin preferentially discards cross-genotype raw ties)")

    make_figure(pooled, dd_bin, cons, comp_focal, pb, wg)
    print(f"\nSaved figure -> figures/genotype_stratified_preservation.png")

    print("\n=== Reorganization vs genotype-alignment (per embryo, k=%d) ===" % SCATTER_K)
    rr = reorg.dropna(subset=["R_gen"])
    if len(rr) > 2:
        slope, r = np.polyfit(rr.churn, rr.R_gen, 1)[0], np.corrcoef(rr.churn, rr.R_gen)[0, 1]
        print(f"  n={len(rr)} embryo×bins; churn→R_gen slope={slope:+.3f}, r={r:+.3f}")
        print("  slope>0 = embryos that reorganize MORE reorganize toward their own genotype")
    make_scatter_figure(reorg)
    print(f"Saved figure -> figures/reorg_vs_alignment.png")


GENO_LABEL = {"wik_ab": "WT", "inj_ctrl": "inj_ctrl",
              "pbx1b_crispant": "pbx1b", "pbx4_crispant": "pbx4",
              "pbx1b_pbx4_crispant": "double"}
COLORS = {"wik_ab": "#2166AC", "inj_ctrl": "#4393C3",
          "pbx1b_crispant": "#F4A582", "pbx4_crispant": "#D6604D",
          "pbx1b_pbx4_crispant": "#B2182B"}


TITLE_FS, LABEL_FS, TICK_FS = 12, 9.5, 8.5   # 3-size type system


def _subtitle(ax, takeaway, define=None, good=True):
    """Under each title: an italic colour-coded TAKEAWAY, then a grey DEFINE line
    spelling out the metric in plain words (so no bare 'O' appears anywhere)."""
    ax.set_title(ax.get_title(), fontsize=TITLE_FS, fontweight="bold", pad=30)
    ax.text(0.5, 1.10, takeaway, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=LABEL_FS, style="italic", wrap=True,
            color=("#1b7837" if good else "#b2182b"))
    if define:
        ax.text(0.5, 1.045, define, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=TICK_FS - 0.5, color="#555")


def _boot_ci(values, n_boot=N_BOOT, seed=0):
    """Percentile 95% CI of the mean by resampling `values`."""
    values = np.asarray([v for v in values if np.isfinite(v)])
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _shared_genotype_legend(fig):
    """One legend for the whole figure: colour = genotype, marker = control/crispant."""
    from matplotlib.lines import Line2D
    handles = []
    for g in GROUPS:
        mk = "s" if g in CONTROLS else "o"
        handles.append(Line2D([0], [0], marker=mk, ls="none", ms=9,
                              mfc=COLORS[g], mec="k",
                              label=f"{GENO_LABEL[g]}" + ("  (control)" if g in CONTROLS else "")))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=LABEL_FS,
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def make_figure(pooled, dd_bin, cons, comp_focal, pb, wg):
    """Four panels, each built backwards from ONE thing to show. Rules:
      (1) x = the thing under examination; (2) y-label states what high/low MEAN;
      (3) the conclusion is an italic subtitle, and a grey line under it DEFINES
          the metric in plain words (never a bare 'O').

    (1a) added neighbors are same-GENOTYPE far above chance   -> the finding
    (1b) same plot for EXPERIMENT: added neighbors sit at chance -> no batch bias
    (2)  preservation over time: the late drop is controls' PCA rising, not crispant
    (3)  within-genotype preservation over time: double crispant lowest (k=3 & k=8)
    """
    plt.rcParams.update({"axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS,
                         "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
                         "legend.fontsize": TICK_FS})
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.32)

    def mk(g):
        return "s" if g in CONTROLS else "o"

    # per-genotype observed & null same-class shares of ADDED neighbors
    obs = comp_focal.groupby("genotype")[["obs_gen", "null_gen", "obs_exp", "null_exp"]].mean()

    def paired_obs_null(ax, obs_col, null_col, class_word):
        gg = [g for g in GROUPS if g in obs.index]
        x = np.arange(len(gg))
        o = [obs.loc[g, obs_col] for g in gg]
        nblo = [obs.loc[g, null_col] for g in gg]
        # bootstrap CI on the observed share
        cis = []
        for g in gg:
            v = comp_focal[comp_focal.genotype == g][obs_col].values
            cis.append(_boot_ci(v))
        elo = [max(0.0, o[i] - cis[i][0]) for i in range(len(gg))]
        ehi = [max(0.0, cis[i][1] - o[i]) for i in range(len(gg))]
        ax.bar(x - 0.2, o, width=0.38, color=[COLORS[g] for g in gg],
               label=f"observed same-{class_word} share")
        ax.errorbar(x - 0.2, o, yerr=[elo, ehi], fmt="none", ecolor="k",
                    elinewidth=0.8, capsize=2)
        ax.bar(x + 0.2, nblo, width=0.38, color="#cccccc", edgecolor="k",
               label=f"null: chance share in the eligible pool")
        ax.set_xticks(x); ax.set_xticklabels([GENO_LABEL[g] for g in gg])
        ax.grid(alpha=0.3, axis="y")

    # ══ (1a) added neighbors: observed vs null same-GENOTYPE share ══════════════
    ax = fig.add_subplot(gs[0, 0])
    paired_obs_null(ax, "obs_gen", "null_gen", "genotype")
    ax.set(ylabel="Fraction of ADDED neighbors\nsharing focal embryo's genotype", ylim=(0, 1))
    ax.set_title("(1a) Added neighbors: share GENOTYPE?")
    ax.legend(loc="upper left")
    _subtitle(ax, "Yes — far above chance everywhere.",
              define="added = a margin k-NN that wasn't a raw k-NN",
              good=True)

    # ══ (1b) same plot for EXPERIMENT ══════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 0])
    paired_obs_null(ax, "obs_exp", "null_exp", "experiment")
    ax.set(ylabel="Fraction of ADDED neighbors\nsharing focal embryo's experiment", ylim=(0, 1))
    ax.set_title("(1b) Added neighbors: share EXPERIMENT?")
    ax.legend(loc="upper left")
    _subtitle(ax, "No — observed sits on the null: no batch bias.",
              define="same axes as (1a), experiment substituted for genotype",
              good=True)

    # ══ Pod 2 — controls preservation over time, low k=5 vs high k=15 merged ════
    def ctl_pooled(k):
        c = pb[(pb.genotype.isin(CONTROLS)) & (pb.k == k)]
        return (c.groupby("time_bin")
                 .apply(lambda gp: pd.Series(
                     dict(O_margin=np.average(gp.O_margin, weights=gp.n),
                          O_PC10=np.average(gp.O_PC10, weights=gp.n))),
                        include_groups=False)
                 .reset_index().sort_values("time_bin"))

    ax = fig.add_subplot(gs[0, 1])
    # colour = space (margin blue / PC10 red); line style = scale (k=5 solid fine,
    # k=15 dashed broad)
    for k_w, ls, tag in [(5, "-", "k=5 fine"), (15, "--", "k=15 broad")]:
        d = ctl_pooled(k_w)
        ax.plot(d.time_bin, d.O_margin, ls=ls, marker="o", ms=3.5, lw=1.8,
                color="#2166AC", label=f"margin, {tag}")
        ax.plot(d.time_bin, d.O_PC10, ls=ls, marker="o", ms=3.5, lw=1.8,
                color="#B2182B", label=f"PC10 baseline, {tag}")
    ax.axvspan(80, 118, color="gray", alpha=0.06)
    ax.set(ylabel="Raw neighbors kept\n(↑ better preserved)", ylim=(0, 0.75))
    ax.legend(loc="upper left", fontsize=TICK_FS - 1.5, ncol=1); ax.grid(alpha=0.3)
    ax.set_xlabel("developmental time (hpf)")
    ax.set_title("(2a) CONTROLS, over time")
    _subtitle(ax, "Controls' PC10 baseline RISES late — this drives the ΔD effect.",
              define="blue = margin, red = PC10; solid = fine k=5, dashed = broad k=15",
              good=False)

    # gs[1, 1] intentionally left empty

    # ══ Pod 3 — within-genotype preservation, split: 3a k=3 / 3b k=8 ═══════════
    def _smooth(x, y):
        """Monotone-x cubic spline through the points for a smoother read; falls
        back to the raw line if too few points to interpolate."""
        if len(x) < 4:
            return x, y
        from scipy.interpolate import PchipInterpolator
        xs = np.linspace(x.min(), x.max(), 200)
        return xs, PchipInterpolator(x, y)(xs)

    def within_panel(ax, k_w):
        wk = wg[wg.k == k_w]
        for g in GROUPS:
            sub = wk[wk.genotype == g].sort_values("time_bin")
            if len(sub) < 2:
                continue
            xs, ys = _smooth(sub.time_bin.values.astype(float),
                             sub.O_within.values.astype(float))
            ax.plot(xs, ys, "-", color=COLORS[g], lw=1.5, alpha=0.9)
            ax.plot(sub.time_bin, sub.O_within, color=COLORS[g], lw=0,
                    marker=mk(g), ms=3)
        ax.set(ylabel="Same-genotype neighbors kept\n(↑ internal order kept)", ylim=(0, 0.9))
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    within_panel(ax, 3)
    ax.set_title("(3a) Within genotype, k=3 (local)")
    _subtitle(ax, "Double crispant sits lowest.",
              define="(2) restricted to the focal embryo's own genotype", good=False)

    ax = fig.add_subplot(gs[1, 2])
    within_panel(ax, 8)
    ax.set(xlabel="developmental time (hpf)")
    ax.set_title("(3b) Within genotype, k=8 (broader)")
    _subtitle(ax, "Same ordering at broader k — not a k=3 artifact.",
              define="same as (3a), wider neighborhood", good=False)

    _shared_genotype_legend(fig)
    fig.suptitle("Does the classifier-margin space reorganize the raw latent by BIOLOGY or by BATCH?",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.savefig(FIGURES / "genotype_stratified_preservation.png", dpi=130,
                bbox_inches="tight")


def make_scatter_figure(reorg):
    """Standalone: is the raw→margin reorganization genotype-driven?

    Churn (=neighbors-changed/k) is inherently discrete, so instead of a strip
    scatter we draw, at EACH churn level, one violin PER genotype showing the
    distribution of R_gen (same-genotype share of ADDED neighbors minus the
    eligible-pool chance share). High R_gen = the churn moved that embryo toward
    its own genotype. If genotype alignment held roughly constant across churn
    levels within a genotype, alignment is a floor property of the reorganization,
    not something proportional to how MUCH an embryo reorganized."""
    # larger type throughout — the top panel is condensed to afford it
    BIG_LABEL, BIG_TICK = LABEL_FS + 3, TICK_FS + 3
    plt.rcParams.update({"axes.titlesize": TITLE_FS + 2, "axes.labelsize": BIG_LABEL,
                         "xtick.labelsize": BIG_TICK, "ytick.labelsize": BIG_TICK})
    reorg = reorg.copy()
    reorg["churn"] = reorg["churn"].round(6)
    # ALL possible churn levels for k=5 (0/5 … 5/5), shown even when empty so an
    # absent level reads as "no embryos here", not "forgotten". Proportions below
    # are over EVERY embryo of the genotype (incl. churn=0 and null-R_gen rows);
    # violins above use only rows with a defined enrichment.
    levels = [i / SCATTER_K for i in range(SCATTER_K + 1)]
    rr = reorg.dropna(subset=["R_gen"])            # violin-eligible rows
    genos = [g for g in GROUPS if (reorg.genotype == g).any()]

    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(12, 8.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.08))
    step = 1.0                       # x-spacing between churn levels
    grp_w = 0.8                      # total width occupied by the genotype cluster
    vw = grp_w / max(len(genos), 1)  # per-violin / per-bar slot width
    offsets = (np.arange(len(genos)) - (len(genos) - 1) / 2) * vw

    # per-genotype totals for the proportion panel (each genotype sums to 1 over
    # ALL its embryos, including any at churn=0 / undefined enrichment)
    geno_tot = {g: max((reorg.genotype == g).sum(), 1) for g in genos}
    n_max = max((((rr.churn == lv) & (rr.genotype == g)).sum()
                 for lv in levels for g in genos), default=1) or 1
    rng = np.random.default_rng(RANDOM_STATE)

    for li, lv in enumerate(levels):
        base = li * step
        for gi, g in enumerate(genos):
            xpos = base + offsets[gi]
            v = rr[(rr.churn == lv) & (rr.genotype == g)]["R_gen"].values  # violin
            # width ∝ sqrt(n): a 326-embryo cell is visibly fatter than a 10-embryo
            # one, without the biggest dwarfing everything. Also sets jitter spread.
            w = vw * 0.9 * np.sqrt(len(v) / n_max) if len(v) else 0.0
            if len(v) >= 3 and np.ptp(v) > 0:
                parts = ax.violinplot([v], positions=[xpos], widths=max(w, vw * 0.1),
                                      showmeans=False, showextrema=False)
                for body in parts["bodies"]:      # thin unfilled outline over points
                    body.set_facecolor("none"); body.set_alpha(1.0)
                    body.set_edgecolor(COLORS[g]); body.set_linewidth(1.1)
            if len(v):
                # jittered raw points: the actual mass, so big n reads as big
                jit = rng.uniform(-1, 1, len(v)) * max(w, vw * 0.12) * 0.45
                ax.scatter(xpos + jit, v, s=5, color=COLORS[g], alpha=0.30,
                           edgecolors="none", zorder=3)
                ax.plot([xpos - vw * 0.4, xpos + vw * 0.4], [v.mean()] * 2,
                        color="k", lw=1.4, zorder=6)
            # ── proportion panel: fraction of THIS genotype at this churn level ──
            # counted over EVERY embryo (churn=0 included), so a 0-height bar means
            # genuinely no embryos at this level, not a dropped null-enrichment row
            n_here = ((reorg.churn == lv) & (reorg.genotype == g)).sum()
            frac = n_here / geno_tot[g]
            axp.bar(xpos, frac, width=vw * 0.9, color=COLORS[g],
                    alpha=0.85, edgecolor="none")
            # n above each bar (this is the panel about counts / mass)
            if n_here:
                axp.text(xpos, frac + 0.008, f"{n_here}", ha="left", va="bottom",
                         rotation=45, rotation_mode="anchor",
                         fontsize=TICK_FS + 1, fontweight="bold",
                         color="k", zorder=5)

    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.set_ylabel("Same-genotype enrichment of the churn\n(observed − null; ↑ toward own genotype)")
    ax.set_title("How much of the raw→margin reorganization is genotype-driven?",
                 fontsize=TITLE_FS + 2, fontweight="bold", pad=14)
    ax.grid(alpha=0.3, axis="y")

    axp.set_xticks([li * step for li in range(len(levels))])
    axp.set_xticklabels([f"{lv:.1f}\n({int(round(lv * SCATTER_K))}/{SCATTER_K} changed)"
                         for lv in levels])
    axp.set_xlabel(f"Neighborhood churn at k={SCATTER_K}  "
                   "(1 − raw∩margin overlap; → more reorganized)")
    axp.set_ylabel("Proportion of\ngenotype")
    axp.grid(alpha=0.3, axis="y")
    axp.margins(y=0.18)   # headroom for the rotated n labels above the bars

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLORS[g], label=GENO_LABEL[g]) for g in genos]
    handles.append(Line2D([0], [0], color="k", lw=1.2, label="mean"))
    ax.legend(handles=handles, loc="lower left", fontsize=TICK_FS + 2, ncol=3,
              frameon=True)
    fig.savefig(FIGURES / "reorg_vs_alignment.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip heavy compute; read cached tables from tables/ and "
                         "re-render the figure only.")
    args = ap.parse_args()
    A = load_artifacts() if args.plot_only else compute_artifacts()
    report_and_plot(A)
