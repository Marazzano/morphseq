"""Run the full processing chain and cache artifacts for the notebook.

Writes to ``data/``:
    reference_pca.csv   wildtype reference, projected
    hotfish_pca.csv     20240813, projected, with workbook temperatures
    gene7_pca.csv       GENE7, projected onto the same basis
    spline_weighted.csv / spline_unweighted.csv
    pca_variance.csv
    pca_basis.joblib    the fitted PCA (so the notebook never refits)
"""
from pathlib import Path
import sys
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import legacy_reference as lr
import morph_pca_spline as mps
from morphseq_integration import build_master_table

OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

print("[1/4] loading legacy companion + building reference sets ...")
sets = lr.build_reference_sets()
print(f"      reference {len(sets.reference)} snips / {sets.reference.embryo_id.nunique()} embryos")
print(f"      hotfish   {len(sets.hotfish)} snips, temps {sorted(sets.hotfish.temperature.unique())}")

print("[2/4] building GENE7 master table ...")
master = build_master_table()
g7 = mps.gene7_latents_from_master(master[master.has_morph].copy())
g7["predicted_stage_hpf"] = pd.to_numeric(g7["stage"], errors="coerce")
print(f"      GENE7 {len(g7)} wells with morphology")

print("[3/4] fitting PCA (reference + 20240813; GENE7 transformed only) ...")
fitted = mps.fit_morph_pca(sets, gene7=g7)
print(fitted.explained_variance.to_string(index=False))

print("[4/4] fitting reference splines (2 variants x 50 bootstraps) ...")
splines = mps.fit_reference_splines(fitted)

fitted.reference.to_csv(OUT / "reference_pca.csv", index=False)
fitted.hotfish.to_csv(OUT / "hotfish_pca.csv", index=False)
fitted.gene7.to_csv(OUT / "gene7_pca.csv", index=False)
fitted.explained_variance.to_csv(OUT / "pca_variance.csv", index=False)
for name, frame in splines.items():
    frame.to_csv(OUT / f"spline_{name}.csv", index=False)
joblib.dump(fitted.pca, OUT / "pca_basis.joblib")
print(f"\nwrote artifacts to {OUT}")
