"""Ad-hoc: gallery of ALIVE-ONLY snips currently FAILING the LOWER surface_area_qc band.

Question: is k_lower=0.7 too aggressive, killing real small embryos? Restricts to snips with
area_um2 < sa_lower_threshold (the lower-band failures only, not upper-band/oversized failures),
and to alive embryos (death_detection viability_dead_flag==False AND persistence_dead_flag==False)
so dead-embryo losses don't cloud the read. Ranked from worst (furthest below threshold) to
mildest (right at the edge) so we can see both extremes.
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
from data_pipeline.viz.reporting import select_quartile_bands_vs_band, _render_band_page

SA_K_LOWER, SA_K_UPPER = 0.7, 1.4

mg = merged("feature_extraction", "mask_geometry", "mask_geometry")
sp = merged("feature_extraction", "stage_predictions", "stage_predictions")
sa = merged("quality_control", "surface_area_qc", "surface_area_qc")
dd = merged("quality_control", "death_detection", "death_detection_qc")
reference = load_packaged_surface_area_reference(version="v1")

alive = dd[(dd.viability_dead_flag == False) & (dd.persistence_dead_flag == False)][["snip_id"]]

df = sa.merge(mg[["snip_id", "area_um2"]], on="snip_id", how="left").merge(
    sp[["snip_id", "predicted_stage_hpf"]], on="snip_id", how="left"
).merge(alive, on="snip_id", how="inner")

band = [interpolate_reference_band(float(s), reference) for s in df["predicted_stage_hpf"]]
df["sa_reference_p5"] = [p5 for p5, _ in band]
df["sa_reference_p95"] = [p95 for _, p95 in band]
df["sa_lower_threshold"] = SA_K_LOWER * df["sa_reference_p5"]
df["sa_upper_threshold"] = SA_K_UPPER * df["sa_reference_p95"]

df = df.merge(snip_image_paths(), on="snip_id", how="left")

# only lower-band failures (area below threshold) -- not upper-band (oversized) failures.
lower_fails = df[df["area_um2"] < df["sa_lower_threshold"]].copy()
print(f"alive snips failing the LOWER band: {len(lower_fails)} / {len(df)} alive total")
lower_fails.to_csv(
    "results/mcolon/20260702_qc_reporting_v1/scratch_scripts/lower_band_fails_alive_only.csv", index=False
)

def badge_text(row: pd.Series) -> str:
    return (
        f"area = {row['area_um2']/1e6:.3f} mm^2  "
        f"(band [{row['sa_lower_threshold']/1e6:.3f}, {row['sa_upper_threshold']/1e6:.3f}] mm^2)"
    )

def is_fail(row: pd.Series) -> bool:
    return not (row["sa_lower_threshold"] <= row["area_um2"] <= row["sa_upper_threshold"])

bands = select_quartile_bands_vs_band(
    lower_fails, "area_um2", "sa_lower_threshold", "sa_upper_threshold", n_per_band=16
)
gallery = _render_band_page(
    bands, badge_text, is_fail,
    metric_col="area_um2", image_path_col="resolved_image_path", label_col="snip_id",
    title="surface_area_qc (ALIVE ONLY) -- LOWER-band FAILURES only, area in mm^2",
    output_path=Path("results/mcolon/20260702_qc_reporting_v1/scratch_scripts/lower_band_fails_alive_only.png"),
    n_per_band=16,
    band_cols=4,
)
print("saved", gallery)
