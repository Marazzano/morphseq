"""
Trajectory plots for sci_ timelapse plates using plot_feature_over_time.

Features: baseline_deviation_normalized + total_length_um.
x-axis: predicted_stage_hpf restricted to the query window (overlapping hpf range).
color_by: predicted phenotype label (query) or cluster_categories (reference).
Colors: from sequenced_focus_config.PHENOTYPE_COLORS.

For each gene (b9d2, cep290) produces two figures:

  homozygous_focus/{gene}_homo_trajectory.png
    Columns: [Query homo] [Reference homo]  ×  Rows: [baseline_dev_norm] [total_length_um]

  all_phenotypes/{gene}_all_trajectory.png
    Columns: [homo] [wt_sibling] [AB] [reference]  ×  Rows: [baseline_dev_norm] [total_length_um]

Each column group is a separate call to plot_feature_over_time (it creates its own figure).
The final PNG is assembled by stacking the per-column PNGs side by side.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/make_trajectory_plots_sci.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

import build_reference_and_transfer as T  # noqa: E402
from src.analyze.viz.plotting import plot_feature_over_time  # noqa: E402
from sequenced_focus_config import PHENOTYPE_COLORS  # noqa: E402

SEQ_FOCUS = RUN_DIR / "time_series" / "sequenced_focus"
OUT_HOMO = SEQ_FOCUS / "homozygous_focus"
OUT_ALL = SEQ_FOCUS / "all_phenotypes"

FEATURES = ["baseline_deviation_normalized", "total_length_um"]
PHENO_LABEL_COL = "cluster_categories"

SCI_PLATES = {
    "b9d2": "20260414_sci_b9d2_48hpf_plate01",
    "cep290": "20260415_sci_cep290_48hpf_plate01",
}

# Canonical colors from sequenced_focus_config — add Intermediate alias + unlabeled
PHENO_COLORS: dict[str, dict[str, str]] = {
    gene: {**colors, "Intermediate": colors.get("Low_to_High", "#cccccc"), "unlabeled": "#cccccc"}
    for gene, colors in PHENOTYPE_COLORS.items()
}


def zygosity_group(genotype: str) -> str:
    g = str(genotype).lower()
    if g.endswith("_homozygous"):
        return "homo"
    if "ab_wildtype" in g or g == "ab":
        return "AB"
    if g.endswith("_heterozygous") or g.endswith("_wildtype"):
        return "wt_sibling"
    return "other"


def load_ref_frames(ref_path: Path, gene: str) -> pd.DataFrame:
    use = {T.GENO_COL, T.GROUP_COL, T.TIME_COL, PHENO_LABEL_COL,
           "baseline_deviation_normalized", "total_length_um"}
    df = pd.read_csv(ref_path, usecols=lambda c: c in use, low_memory=False)
    df = df.dropna(subset=[PHENO_LABEL_COL, T.TIME_COL])
    df = df[~df[PHENO_LABEL_COL].astype(str).isin(["unlabeled", "nan"])]
    if gene == "cep290":
        df.loc[df[PHENO_LABEL_COL] == "Intermediate", PHENO_LABEL_COL] = "Low_to_High"
    if gene == "b9d2":
        df[PHENO_LABEL_COL] = df[PHENO_LABEL_COL].replace("BA_rescue", "HTA")
    return df


def load_query_frames(exp: str, gene: str, predictions: pd.DataFrame) -> pd.DataFrame:
    use = {T.GROUP_COL, T.TIME_COL, T.GENO_COL, "well",
           "baseline_deviation_normalized", "total_length_um"}
    p = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
    df = pd.read_csv(p, usecols=lambda c: c in use, low_memory=False)
    df[T.GENO_COL] = df[T.GENO_COL].map(lambda v: T.GENOTYPE_RENAME.get(str(v), str(v)))
    pred_map = predictions.set_index("query_embryo_id")["predicted_label"]
    seq_map = predictions.set_index("query_embryo_id")["sequenced"]
    df["predicted_label"] = df[T.GROUP_COL].map(pred_map)
    df["sequenced"] = df[T.GROUP_COL].map(seq_map).fillna(0).astype(int)
    df["zygosity_group"] = df[T.GENO_COL].map(zygosity_group)
    return df.dropna(subset=[T.TIME_COL, "predicted_label"])


def fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def panel_fig(df: pd.DataFrame, color_by: str, colors: dict, title: str,
              hpf_min: float, hpf_max: float, label_col_override: str | None = None) -> plt.Figure:
    """Render a 2-row (feature × time) figure for one group. Returns matplotlib Figure."""
    cb = label_col_override or color_by
    figs = plot_feature_over_time(
        df,
        features=FEATURES,
        time_col=T.TIME_COL,
        id_col=T.GROUP_COL,
        color_by=cb,
        color_lookup=colors,
        backend="matplotlib",
        output_path=None,
        title=title,
        xlim=(hpf_min, hpf_max),
        show_individual=True,
        show_trend=True,
        show_error_band=False,
    )
    # plot_feature_over_time returns a Figure (matplotlib backend)
    if isinstance(figs, (list, tuple)):
        return figs[0]
    return figs


def hstack_images(images: list[Image.Image], out_path: Path) -> None:
    """Paste images side by side (all resized to same height) and save."""
    max_h = max(img.height for img in images)
    resized = [img.resize((int(img.width * max_h / img.height), max_h), Image.LANCZOS)
               for img in images]
    total_w = sum(img.width for img in resized)
    canvas = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width
    canvas.save(out_path, dpi=(150, 150))
    print(f"  → {out_path.relative_to(RUN_DIR)}")


def make_homo_figure(gene: str) -> None:
    pred_path = OUT_HOMO / f"{gene}_homo_predictions.csv"
    if not pred_path.exists():
        print(f"  MISSING {pred_path.name} — run make_label_transfer_sci_timelapse.py first")
        return

    preds = pd.read_csv(pred_path)
    ref_path = T.B9D2_REF if gene == "b9d2" else T.CEP290_REF
    exp = SCI_PLATES[gene]
    colors = PHENO_COLORS[gene]

    qry = load_query_frames(exp, gene, preds)
    qry_homo = qry[(qry["sequenced"] > 0) & (qry["zygosity_group"] == "homo")].copy()

    hpf_min = float(qry[T.TIME_COL].min()) - 1
    hpf_max = float(qry[T.TIME_COL].max()) + 1

    HOMO_KEEP = {"b9d2": {"CE", "HTA"}, "cep290": {"High_to_Low", "Low_to_High"}}
    ref_all = load_ref_frames(ref_path, gene)
    homo_mask = ref_all[T.GENO_COL].str.endswith("_homozygous", na=False)
    keep_labels = HOMO_KEEP.get(gene, set())
    ref_homo = ref_all[
        homo_mask
        & ref_all[PHENO_LABEL_COL].isin(keep_labels)
        & ref_all[T.TIME_COL].between(hpf_min, hpf_max)
    ].copy()

    n_q = qry_homo[T.GROUP_COL].nunique()
    n_r = ref_homo[T.GROUP_COL].nunique()
    print(f"  [{gene} homo] query n={n_q}  ref n={n_r}  window [{hpf_min:.0f}, {hpf_max:.0f}]")
    print(f"  query labels: {qry_homo['predicted_label'].value_counts().to_dict()}")
    print(f"  ref labels:   {ref_homo[PHENO_LABEL_COL].value_counts().to_dict()}")

    imgs = [
        fig_to_image(panel_fig(qry_homo, "predicted_label", colors,
                               f"{gene.upper()} Query — homo (n={n_q})", hpf_min, hpf_max)),
        fig_to_image(panel_fig(ref_homo, PHENO_LABEL_COL, colors,
                               f"{gene.upper()} Reference — homo (n={n_r})", hpf_min, hpf_max)),
    ]
    hstack_images(imgs, OUT_HOMO / f"{gene}_homo_trajectory.png")


def make_all_figure(gene: str) -> None:
    pred_path = OUT_ALL / f"{gene}_all_predictions.csv"
    if not pred_path.exists():
        print(f"  MISSING {pred_path.name} — run make_label_transfer_sci_timelapse.py first")
        return

    preds = pd.read_csv(pred_path)
    ref_path = T.B9D2_REF if gene == "b9d2" else T.CEP290_REF
    exp = SCI_PLATES[gene]
    colors = PHENO_COLORS[gene]

    qry = load_query_frames(exp, gene, preds)
    qry_seq = qry[qry["sequenced"] > 0].copy()

    hpf_min = float(qry[T.TIME_COL].min()) - 1
    hpf_max = float(qry[T.TIME_COL].max()) + 1

    ref_all = load_ref_frames(ref_path, gene)
    ref_win = ref_all[ref_all[T.TIME_COL].between(hpf_min, hpf_max)].copy()

    groups = [("homo", "Homozygous"), ("wt_sibling", "WT sibling"), ("AB", "AB control")]
    imgs = []
    for grp_key, grp_label in groups:
        sub = qry_seq[qry_seq["zygosity_group"] == grp_key].copy()
        n = sub[T.GROUP_COL].nunique()
        print(f"  [{gene} all] {grp_label}: n={n}  "
              f"labels={sub['predicted_label'].value_counts().to_dict()}")
        if sub.empty:
            # blank placeholder
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.text(0.5, 0.5, f"{grp_label}\nn=0", ha="center", va="center", fontsize=12)
            ax.axis("off")
            imgs.append(fig_to_image(fig))
        else:
            imgs.append(fig_to_image(panel_fig(sub, "predicted_label", colors,
                                                f"{gene.upper()} {grp_label} (n={n})",
                                                hpf_min, hpf_max)))

    n_ref = ref_win[T.GROUP_COL].nunique()
    print(f"  [{gene} all] Reference: n={n_ref}  labels={ref_win[PHENO_LABEL_COL].value_counts().to_dict()}")
    imgs.append(fig_to_image(panel_fig(ref_win, PHENO_LABEL_COL, colors,
                                        f"{gene.upper()} Reference (n={n_ref})",
                                        hpf_min, hpf_max)))

    hstack_images(imgs, OUT_ALL / f"{gene}_all_trajectory.png")


def main() -> None:
    OUT_HOMO.mkdir(parents=True, exist_ok=True)
    OUT_ALL.mkdir(parents=True, exist_ok=True)

    for gene in ("b9d2", "cep290"):
        print(f"\n{'='*60}")
        print(f"{gene.upper()}")
        make_homo_figure(gene)
        make_all_figure(gene)

    print("\nDone.")


if __name__ == "__main__":
    main()
