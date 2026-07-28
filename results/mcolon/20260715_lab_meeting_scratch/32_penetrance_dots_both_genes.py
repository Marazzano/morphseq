"""Derived-label penetrance dots for BOTH genes, unknown zygosity dropped.

Trains a 3-class phenotype model per gene and plots one dot per embryo, stacked so the
non-penetrant class forms the baseline band, with the predicted-penetrant rule and its rate
printed on top.

    b9d2    CE / HTA / Not Penetrant
    cep290  High_to_Low / Low_to_High / Not Penetrant

Both genes use their own curated labels as the reference, out-of-fold via k-fold grouped by
embryo, then every remaining embryo is scored by the full model. The rate on each rule is the
CLASSIFIER'S call ("% pen."), not a measured biological penetrance.

The `unknown` zygosity column is dropped: it is a sequencing gap, not a genotype, and mixing it
into a penetrance panel invites reading it as a biological class.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/32_penetrance_dots_both_genes.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
OUTPUT_DIR = RUN_DIR / "figures" / "penetrance_composite"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
NP_LABEL = "Not Penetrant"
N_FOLDS = 5
RANDOM_STATE = 42

# Only true genotype calls; `unknown` is a sequencing gap, not a zygosity.
ZYGOSITY_ORDER = ["wildtype", "heterozygous", "homozygous"]
ZYG_LABEL = {"wildtype": "WT", "heterozygous": "het", "homozygous": "homo"}

B9D2_SOURCE = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
CEP290_SOURCE = PROJECT_ROOT / "results/mcolon/20260607_sci_cilia_gene14_imaging_qc/tables/reference_cep290_clean.csv"

GENES = {
    "b9d2": {
        "classes": ["CE", "HTA", NP_LABEL],
        "colors": {"CE": "#1B9E77", "HTA": "#D95F02", NP_LABEL: "#9AA0A6"},
    },
    "cep290": {
        "classes": ["High_to_Low", "Low_to_High", NP_LABEL],
        "colors": {"High_to_Low": "#E76FA2", "Low_to_High": "#2FB7B0", NP_LABEL: "#9AA0A6"},
    },
}


def _pipe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced",
                          random_state=RANDOM_STATE),
    )


def load_b9d2() -> tuple[pd.DataFrame, list[str]]:
    """Curated b9d2 labels -> CE / HTA / Not Penetrant, plus zygosity."""
    df = pd.read_csv(B9D2_SOURCE, low_memory=False)
    # Curated phenotype calls that contradict a wildtype genotype are conflicts, not training data.
    df = df[~df[ID_COL].isin({"20251125_B06_e01", "20251125_F12_e01"})]
    z = sorted([c for c in df.columns if c.startswith("z_mu_b_")],
               key=lambda c: int(c.split("_")[-1]))
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *z])
    df["zygosity"] = df["genotype"].str.replace("b9d2_", "", regex=False)

    lab = df["cluster_categories"]
    df["train_label"] = pd.NA
    df.loc[lab == "CE", "train_label"] = "CE"
    df.loc[(df["zygosity"] == "homozygous") & lab.isin(["HTA", "BA_rescue"]),
           "train_label"] = "HTA"
    df.loc[lab == "wildtype", "train_label"] = NP_LABEL
    return df, z


def load_cep290() -> tuple[pd.DataFrame, list[str]]:
    """cep290 already ships curated phenotype_clean labels across every zygosity."""
    probe = pd.read_csv(CEP290_SOURCE, nrows=0).columns
    z = sorted([c for c in probe if c.startswith("z_mu_b_")],
               key=lambda c: int(c.split("_")[-1]))
    keep = {ID_COL, TIME_COL, "zygosity", "phenotype_clean", "use_embryo_flag", "pair",
            "baseline_deviation_normalized", "total_length_um"}
    df = pd.read_csv(CEP290_SOURCE,
                     usecols=lambda c: c in keep or c.startswith("z_mu_b_"),
                     low_memory=False)
    if "use_embryo_flag" in df.columns:
        f = df["use_embryo_flag"]
        m = f if f.dtype == bool else f.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
        df = df[m.fillna(False)]
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *z])
    df["train_label"] = df["phenotype_clean"].where(
        df["phenotype_clean"].isin(GENES["cep290"]["classes"]))
    return df, z


def score(df: pd.DataFrame, z: list[str], classes: list[str]) -> pd.DataFrame:
    """Out-of-fold probabilities for labeled embryos; full-model scores for the rest."""
    tr_mask = df["train_label"].notna()
    tr = df[tr_mask]
    y = tr["train_label"].to_numpy()
    X = tr[z].to_numpy()
    groups = tr[ID_COL].to_numpy()

    oof = np.full((len(tr), len(classes)), np.nan)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for a, b in cv.split(X, y, groups=groups):
        m = _pipe().fit(X[a], y[a])
        pr = m.predict_proba(X[b])
        for j, c in enumerate(classes):
            oof[b, j] = pr[:, list(m.classes_).index(c)] if c in m.classes_ else 0.0

    full = _pipe().fit(X, y)
    df = df.copy()
    pcols = [f"p_{i}" for i in range(len(classes))]
    for c in pcols:
        df[c] = np.nan
    df.loc[tr.index, pcols] = oof
    rest = df.index[~tr_mask]
    if len(rest):
        pr = full.predict_proba(df.loc[rest, z].to_numpy())
        for j, c in enumerate(classes):
            df.loc[rest, pcols[j]] = (pr[:, list(full.classes_).index(c)]
                                      if c in full.classes_ else 0.0)

    per = df.groupby(ID_COL)[pcols].mean()
    emb = per.idxmax(axis=1).map({p: c for p, c in zip(pcols, classes)})
    out = df.drop_duplicates(ID_COL)[[ID_COL, "zygosity"]].copy()
    out["phenotype"] = out[ID_COL].map(emb)
    return out[out["zygosity"].isin(ZYGOSITY_ORDER)].dropna(subset=["phenotype"])


def panel(ax, emb: pd.DataFrame, gene: str, rng) -> None:
    classes = GENES[gene]["classes"]
    colors = GENES[gene]["colors"]
    counts = (emb.groupby(["zygosity", "phenotype"]).size()
                 .unstack("phenotype")
                 .reindex(index=ZYGOSITY_ORDER, columns=classes).fillna(0).astype(int))
    totals = counts.sum(axis=1)
    pen_pct = (counts[[c for c in classes if c != NP_LABEL]].sum(axis=1)
               .div(totals.replace(0, np.nan)).fillna(0.0) * 100)

    print(f"\n[{gene}]")
    print(counts.assign(total=totals).to_string())
    print("predicted penetrant %:", pen_pct.round(1).to_dict())

    for xi, zyg in enumerate(ZYGOSITY_ORDER):
        col = emb[emb["zygosity"] == zyg]
        total = max(int(totals[zyg]), 1)
        y0 = 0.0
        for pheno in reversed(classes):          # NP at the bottom as the baseline band
            n = int((col["phenotype"] == pheno).sum())
            if n == 0:
                continue
            frac = n / total
            ys = y0 + rng.uniform(0.02, 0.98, size=n) * frac
            xs = xi + rng.uniform(-0.085, 0.085, size=n)
            ax.scatter(xs, ys, s=34, color=colors[pheno], alpha=0.72,
                       edgecolors="white", linewidths=0.4, zorder=3)
            y0 += frac

        pen = pen_pct[zyg] / 100.0
        ax.plot([xi - 0.15, xi + 0.15], [1 - pen, 1 - pen],
                color="#222222", linewidth=2.0, zorder=5)
        ax.annotate(f"{pen_pct[zyg]:.0f}% pen.", (xi, 1 - pen),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#222222", zorder=6)

    ax.set_xticks(range(len(ZYGOSITY_ORDER)))
    ax.set_xticklabels([f"{ZYG_LABEL[z]}\nn={int(totals[z])}" for z in ZYGOSITY_ORDER],
                       fontsize=10)
    # Columns pulled in tight against the axes.
    ax.set_xlim(-0.38, len(ZYGOSITY_ORDER) - 0.62)
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(gene, fontsize=12, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor=colors[c], markeredgecolor="white", label=c)
               for c in classes]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=1 if gene == "cep290" else len(classes))
    ax.grid(True, axis="y", color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    rng = np.random.default_rng(0)
    b9, zb = load_b9d2()
    cep, zc = load_cep290()
    embs = {
        "b9d2": score(b9, zb, GENES["b9d2"]["classes"]),
        "cep290": score(cep, zc, GENES["cep290"]["classes"]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Per-gene figures.
    for gene, emb in embs.items():
        fig, ax = plt.subplots(figsize=(3.5, 5.0))
        panel(ax, emb, gene, rng)
        ax.set_ylabel("% of embryos within zygosity", fontsize=10)
        out = OUTPUT_DIR / f"C_embryo_dots_derived__{gene}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out.relative_to(RUN_DIR)}")

    # Combined side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 5.0), sharey=True)
    for ax, gene in zip(axes, ["cep290", "b9d2"]):
        panel(ax, embs[gene], gene, rng)
    axes[0].set_ylabel("% of embryos within zygosity", fontsize=10)
    fig.suptitle("Predicted penetrance by zygosity (derived 3-class labels)",
                 fontsize=11, y=1.01)
    out = OUTPUT_DIR / "C_embryo_dots_derived__both.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(RUN_DIR)}")


if __name__ == "__main__":
    main()
