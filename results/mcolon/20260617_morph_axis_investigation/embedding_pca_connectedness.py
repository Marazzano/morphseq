"""
embedding_pca_connectedness.py
------------------------------
Same connectedness analysis as the hand-axis panel, but the 2-D axis is PC1/PC2 of
the raw z_mu_b embeddings, FIT ON THE MUTANT GROUP (WT projected into that frame).

Hypothesis under test: b9d2's CE/HTA split lives in embedding directions the
hand-picked morphometric axis (length x curvature) does not capture, so in
embedding-PCA space b9d2 should read MORE discrete than on the hand axis.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/embedding_pca_connectedness.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from connectedness_panel import GENES, build_panel, make_embedding_pca_provider


def main() -> None:
    provider = make_embedding_pca_provider()
    for gene, cfg in GENES.items():
        build_panel(
            gene, cfg, provider,
            out_name=f"{gene}_connectedness_panel_embedding_pca.png",
            title_extra="  (raw-embedding PCA)",
            extra_cols=None,  # z_mu_b_* auto-detected by the provider; loader carries them
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
