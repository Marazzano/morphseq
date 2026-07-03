"""Ad-hoc: gallery of ALIVE-ONLY snips at surface_area_qc k_lower (raised from 0.7 -> 0.75 -> 0.8
-> 0.85 -> 0.9). See docs/refactors/streamline-snakemake/target/specs/tech_debt/surface_area_qc_pose_confound.md.

Band quartiles (worst_of_worst/borderline_fail/borderline_pass/clear_pass) are computed over the
FULL alive population, not a pre-filtered window nearest the threshold -- an earlier version of
this script pre-filtered to the 64 nearest snips, which made "clear_pass" mean only "clearest of
those 64" and mislabeled a snip passing by 2% as clear. Ranking against the full population keeps
the bucket labels honest.

0.8 still let 20250912_E07_e01_BF_t0008 (k_needed=0.802) and 20250912_H04_e02_BF_t0014
(k_needed=0.802) pass -- both visually confirmed yolk-only masks. Bumped to 0.85 to clear them.
0.85 still let round-blob masks like 20250912_C11_e02_BF_t0048/t0049 pass comfortably in
borderline_pass, while pulling in real elongated larvae as collateral in borderline_fail. Bumped
to 0.9 per explicit call: prefer losing thin/low-info real embryos over keeping bad masks.
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "results/mcolon/20260702_qc_reporting_v1")
from pathlib import Path
import pandas as pd

from report_scripts._loaders import merged, snip_image_paths
from data_pipeline.quality_control.surface_area_qc.reference import (
    interpolate_reference_band, load_packaged_surface_area_reference,
)
from data_pipeline.viz.reporting import render_quartile_gallery_vs_band

SA_K_LOWER, SA_K_UPPER = 0.9, 1.4  # k_lower raised 0.7 -> 0.8 -> 0.85 -> 0.9

mg = merged("feature_extraction", "mask_geometry", "mask_geometry")
sp = merged("feature_extraction", "stage_predictions", "stage_predictions")
sa = merged("quality_control", "surface_area_qc", "surface_area_qc")
dd = merged("quality_control", "death_detection", "death_detection_qc")
reference = load_packaged_surface_area_reference(version="v1")

alive = dd[(dd.viability_dead_flag == False) & (dd.persistence_dead_flag == False)][["snip_id"]]

df = sa[["snip_id"]].merge(mg[["snip_id", "area_um2"]], on="snip_id", how="left").merge(
    sp[["snip_id", "predicted_stage_hpf"]], on="snip_id", how="left"
).merge(alive, on="snip_id", how="inner")

band = [interpolate_reference_band(float(s), reference) for s in df["predicted_stage_hpf"]]
df["sa_reference_p5"] = [p5 for p5, _ in band]
df["sa_reference_p95"] = [p95 for _, p95 in band]
df["sa_lower_threshold"] = SA_K_LOWER * df["sa_reference_p5"]
df["sa_upper_threshold"] = SA_K_UPPER * df["sa_reference_p95"]

df = df.merge(snip_image_paths(), on="snip_id", how="left")
df.to_csv("results/mcolon/20260702_qc_reporting_v1/scratch_scripts/full_alive_k080.csv", index=False)

# Full-population band ranking (NOT pre-filtered) -- this is the honest version.
gallery = render_quartile_gallery_vs_band(
    df, "area_um2", "sa_lower_threshold", "sa_upper_threshold",
    image_path_col="resolved_image_path", label_col="snip_id",
    title="surface_area_qc (ALIVE ONLY, full population) -- k_lower=0.9",
    output_path=Path("results/mcolon/20260702_qc_reporting_v1/scratch_scripts/lower_border_gallery_k080.png"),
    n_per_band=16,
)
print("saved", gallery)

# sanity: confirm our flagged bad snips now fail
targets = [
    "20250912_H04_e01_BF_t0020", "20250912_H05_e02_BF_t0016", "20250912_H04_e02_BF_t0015",
    "20250912_E07_e01_BF_t0008", "20250912_H04_e02_BF_t0014",
    "20250912_C11_e02_BF_t0048", "20250912_C11_e02_BF_t0049",
]
sub = df[df.snip_id.isin(targets)].copy()
sub["status"] = sub.apply(
    lambda r: "PASS" if r.sa_lower_threshold <= r.area_um2 <= r.sa_upper_threshold else "FAIL", axis=1
)
print(sub[["snip_id", "area_um2", "sa_lower_threshold", "status"]].to_string(index=False))
