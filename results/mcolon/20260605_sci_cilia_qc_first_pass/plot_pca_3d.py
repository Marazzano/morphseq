"""
Generate reference-space 3D PCA batch-effect QC plots.

This is intentionally separate from model/prediction generation.
"""

from __future__ import annotations

from sci_cilia_qc_config import PLOTS_DIR, RUN_DIR
import make_3d_pca
import make_3d_pca_sci

make_3d_pca.OUT = PLOTS_DIR / "pca_3d"
make_3d_pca_sci.OUT = PLOTS_DIR / "pca_3d"


def main() -> None:
    print("Generating snapshot/query PCA plots...")
    make_3d_pca.main()
    print("\nGenerating sci timelapse PCA plots...")
    make_3d_pca_sci.main()
    print(f"\nPCA HTMLs under: {(PLOTS_DIR / 'pca_3d').relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
