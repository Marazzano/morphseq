"""Ad-hoc: gallery of snips nearest the surface_area_qc LOWER band threshold (k_lower=0.7).

Debug script for investigating whether k_lower should move — see whether snips right at the
lower boundary are legitimately small embryos or bad (yolk-only) segmentations.
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
from data_pipeline.viz.reporting import render_quartile_gallery_vs_band, select_quartile_bands_vs_band, _render_band_page

SA_K_LOWER, SA_K_UPPER = 0.7, 1.4

mg = merged("feature_extraction", "mask_geometry", "mask_geometry")
sp = merged("feature_extraction", "stage_predictions", "stage_predictions")
sa = merged("quality_control", "surface_area_qc", "surface_area_qc")
reference = load_packaged_surface_area_reference(version="v1")

df = sa.merge(mg[["snip_id", "area_um2"]], on="snip_id", how="left").merge(
    sp[["snip_id", "predicted_stage_hpf"]], on="snip_id", how="left"
)
band = [interpolate_reference_band(float(s), reference) for s in df["predicted_stage_hpf"]]
df["sa_reference_p5"] = [p5 for p5, _ in band]
df["sa_reference_p95"] = [p95 for _, p95 in band]
df["sa_lower_threshold"] = SA_K_LOWER * df["sa_reference_p5"]
df["sa_upper_threshold"] = SA_K_UPPER * df["sa_reference_p95"]

df = df.merge(snip_image_paths(), on="snip_id", how="left")

df["dist_to_lower"] = (df["area_um2"] - df["sa_lower_threshold"]).abs()
border = df.nsmallest(64, "dist_to_lower").copy()
border["_signed"] = border["area_um2"] - border["sa_lower_threshold"]
border = border.sort_values("_signed")

fail_side = border[border["_signed"] < 0].tail(16).iloc[::-1]
pass_side = border[border["_signed"] >= 0].head(16)

out = pd.concat([fail_side.assign(_grp="fail_near_lower"), pass_side.assign(_grp="pass_near_lower")])
out.to_csv("results/mcolon/20260702_qc_reporting_v1/scratch_scripts/lower_border_snips.csv", index=False)

# mm^2 is much easier to read than um^2 e-notation (1 mm^2 = 1e6 um^2).
for col in ["area_um2", "sa_lower_threshold", "sa_upper_threshold"]:
    border[col.replace("_um2", "_mm2").replace("threshold", "threshold_mm2")] = border[col] / 1e6

def badge_text(row: pd.Series) -> str:
    return (
        f"area = {row['area_um2']/1e6:.3f} mm^2  "
        f"(band [{row['sa_lower_threshold']/1e6:.3f}, {row['sa_upper_threshold']/1e6:.3f}] mm^2)"
    )

def is_fail(row: pd.Series) -> bool:
    return not (row["sa_lower_threshold"] <= row["area_um2"] <= row["sa_upper_threshold"])

bands = select_quartile_bands_vs_band(
    border, "area_um2", "sa_lower_threshold", "sa_upper_threshold", n_per_band=16
)
gallery = _render_band_page(
    bands, badge_text, is_fail,
    metric_col="area_um2", image_path_col="resolved_image_path", label_col="snip_id",
    title="surface_area_qc — snips nearest the LOWER band threshold (k_lower=0.7), area in mm^2",
    output_path=Path("results/mcolon/20260702_qc_reporting_v1/scratch_scripts/lower_border_gallery.png"),
    n_per_band=16,
    band_cols=4,
)
print("saved", gallery)
