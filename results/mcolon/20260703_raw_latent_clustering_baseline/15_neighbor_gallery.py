"""
15_neighbor_gallery.py

Restricted to the 72hpf time slice (matched developmental stage), rank all
other embryos by raw 80-dim (z_mu_b) Euclidean distance to the target outlier
embryo (20260304_E03_e02), and build a tiered image gallery (5 closest /
5 medium / 5 far) so the distances can be eyeballed against actual morphology.

Images are the BF EMBRYO SNIPS (bf_embryo_snips) -- the exact preprocessed
crops that go into the embedding model -- NOT the raw full-frame ND2 jpgs.
Snip path = {SNIP_ROOT}/{experiment_date}/{snip_id}.jpg.

Run with:
    conda run -n segmentation_grounded_sam --no-capture-output python 15_neighbor_gallery.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "tables", "pbx_binned_zmub_with_wt.csv")
OUT_CSV = os.path.join(HERE, "tables", "gallery_neighbors_72hpf.csv")
OUT_PNG = os.path.join(HERE, "figures", "neighbor_gallery_72hpf.png")

TARGET_EMBRYO = "20260304_E03_e02"
TARGET_TIME_BIN = 72.0

N_PER_TIER = 5
TIERS = ["close", "medium", "far"]

# Root of the BF embryo snips that feed the embedding model.
SNIP_ROOT = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/training_data/bf_embryo_snips"


def snip_path(row):
    """Resolve the model-input snip for a row: {SNIP_ROOT}/{experiment_date}/{snip_id}.jpg."""
    return os.path.join(SNIP_ROOT, str(row["experiment_date"]), f"{row['snip_id']}.jpg")


def load_data():
    df = pd.read_csv(TABLE_PATH, low_memory=False)
    zcols = [c for c in df.columns if c.startswith("z_mu_b")]
    assert len(zcols) == 80, f"expected 80 z_mu_b cols, got {len(zcols)}"
    sub = df[df["time_bin"] == TARGET_TIME_BIN].reset_index(drop=True)
    print(f"[load] {len(df)} total rows, {len(sub)} rows at time_bin={TARGET_TIME_BIN}")
    return sub, zcols


def compute_distances(sub, zcols):
    target_rows = sub[sub["embryo_id"] == TARGET_EMBRYO]
    if len(target_rows) != 1:
        raise ValueError(
            f"expected exactly 1 target row at time_bin={TARGET_TIME_BIN}, "
            f"got {len(target_rows)}"
        )
    target_row = target_rows.iloc[0]
    target_vec = target_row[zcols].to_numpy(dtype=float)

    others = sub[sub["embryo_id"] != TARGET_EMBRYO].copy()
    other_mat = others[zcols].to_numpy(dtype=float)
    dists = np.linalg.norm(other_mat - target_vec[None, :], axis=1)
    others["distance"] = dists
    others = others.sort_values("distance").reset_index(drop=True)
    print(f"[dist] target={TARGET_EMBRYO}, {len(others)} candidate neighbors")
    return target_row, others


def build_tiers(others):
    n_needed = N_PER_TIER * 3
    if len(others) < n_needed:
        raise ValueError(f"need {n_needed} neighbors, only have {len(others)}")

    close = others.iloc[0:N_PER_TIER].copy()
    medium = others.iloc[N_PER_TIER:2 * N_PER_TIER].copy()
    far = others.iloc[2 * N_PER_TIER:3 * N_PER_TIER].copy()

    close["tier"] = "close"
    medium["tier"] = "medium"
    far["tier"] = "far"

    gallery = pd.concat([close, medium, far], ignore_index=True)
    gallery["rank"] = np.arange(1, len(gallery) + 1)
    return gallery


def load_snip(row):
    """Load the model-input BF embryo snip for a row (already a preprocessed crop)."""
    return Image.open(snip_path(row))


def make_tile_label(row, is_target=False):
    label = (
        f"{'TARGET  ' if is_target else ''}{row['embryo_id']}\n"
        f"{row['genotype']}  |  {row['experiment_id']}\n"
        f"t={row['time_bin']:.0f}hpf"
    )
    if not is_target:
        label = f"rank {int(row['rank'])}  d={row['distance']:.2f}\n" + label
    return label


def plot_gallery(target_row, gallery):
    n_cols = N_PER_TIER
    n_rows = 1 + len(TIERS)  # target row + 3 tiers
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows),
        squeeze=False,
    )

    skipped = []

    # Row 0: target, centered (use middle column, blank the rest)
    for c in range(n_cols):
        ax = axes[0][c]
        ax.axis("off")
    mid_col = n_cols // 2
    ax = axes[0][mid_col]
    try:
        crop = load_snip(target_row)
        ax.imshow(np.asarray(crop), cmap="gray")
    except Exception as e:
        ax.text(0.5, 0.5, "IMAGE\nUNAVAILABLE", ha="center", va="center", fontsize=10, color="red")
        skipped.append((target_row["embryo_id"], "target", str(e)))
        print(f"[warn] target image failed: {e}")
    ax.set_title(make_tile_label(target_row, is_target=True), fontsize=9)
    ax.axis("off")
    for spine_ax in [axes[0][mid_col]]:
        for spine in spine_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(2)

    tier_row_label = {"close": "5 CLOSEST", "medium": "NEXT 5 (MEDIUM)", "far": "NEXT 5 (FAR)"}

    for ti, tier in enumerate(TIERS):
        row_idx = ti + 1
        tier_df = gallery[gallery["tier"] == tier].reset_index(drop=True)
        for c in range(n_cols):
            ax = axes[row_idx][c]
            ax.axis("off")
            if c >= len(tier_df):
                continue
            row = tier_df.iloc[c]
            try:
                crop = load_snip(row)
                ax.imshow(np.asarray(crop), cmap="gray")
            except Exception as e:
                ax.text(0.5, 0.5, "IMAGE\nUNAVAILABLE", ha="center", va="center", fontsize=9, color="red")
                skipped.append((row["embryo_id"], tier, str(e)))
                print(f"[warn] skipping {row['embryo_id']} ({tier}): {e}")
            ax.set_title(make_tile_label(row), fontsize=8)

        # row label on the left margin
        axes[row_idx][0].text(
            -0.15, 0.5, tier_row_label[tier], transform=axes[row_idx][0].transAxes,
            rotation=90, va="center", ha="center", fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        f"Raw 80-dim neighbor gallery at 72hpf  |  target = {TARGET_EMBRYO}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {OUT_PNG}")
    return skipped


def main():
    sub, zcols = load_data()
    target_row, others = compute_distances(sub, zcols)
    gallery = build_tiers(others)

    gallery["snip_path"] = gallery.apply(snip_path, axis=1)
    out_cols = [
        "rank", "distance", "tier", "embryo_id", "genotype", "experiment_id",
        "time_bin", "snip_id", "snip_path",
    ]
    gallery[out_cols].to_csv(OUT_CSV, index=False)
    print(f"[table] saved {OUT_CSV}")

    skipped = plot_gallery(target_row, gallery)

    print("\n=== SUMMARY ===")
    print(f"Target: {TARGET_EMBRYO} @ {TARGET_TIME_BIN}hpf")
    for tier in TIERS:
        tdf = gallery[gallery["tier"] == tier]
        print(f"\n-- {tier.upper()} --")
        for _, r in tdf.iterrows():
            print(f"  rank {r['rank']:>2}  d={r['distance']:6.2f}  {r['embryo_id']:<28} "
                  f"{r['genotype']:<22} {r['experiment_id']}")
    if skipped:
        print(f"\n[skipped {len(skipped)} images]")
        for emb, tier, err in skipped:
            print(f"  {emb} ({tier}): {err}")
    else:
        print("\nAll 16 images (target + 15 neighbors) rendered successfully.")


if __name__ == "__main__":
    main()
