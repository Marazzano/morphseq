"""
3_dim_snip_galleries.py [dim_suffix]
------------------------------------
Interpret a raw z_mu_b latent dimension by looking at actual embryo snips.

For the target dimension (default 85), pick 4 evenly-distributed developmental
time points and, at each, render ONE quartile gallery of embryo snips ranked by
that dimension's value: lowest-expressing (worst_of_worst) → highest-expressing
(clear_pass), with a couple of mid-expressing embryos in the borderline bands.
Eyeballing these four galleries reveals what morphological feature the dimension
encodes.

Uses the data_pipeline reporting framework (render_quartile_gallery) — the same
gallery machinery exercised by tests/data_pipeline/viz/test_reporting.py.

Data is the per-frame tfap2 aggregate parquet (one row per snip, carrying
z_mu_b_* + predicted_stage_hpf + snip_id), so each gallery card maps to exactly
one snip image. The binned table is NOT used here (binning averages frames and
has no single snip to show).

Output: figures/dim_snips/z_mu_b_<suffix>/gallery_t<hpf>.png  (one per timepoint)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE.parent / "20260413_tfap2_followup" / "scripts"))

from data_pipeline.viz.reporting import render_quartile_gallery  # noqa: E402

import importlib.util  # noqa: E402
_apr_common_path = _HERE.parent / "20260413_tfap2_followup" / "scripts" / "common.py"
_spec = importlib.util.spec_from_file_location("_tfap2_apr_common", _apr_common_path)
_apr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apr)

# Snip image store: {SNIP_ROOT}/{experiment_date}/{snip_id}.jpg
SNIP_ROOT = _REPO / "morphseq_playground" / "training_data" / "bf_embryo_snips"

N_TIMEPOINTS = 4          # evenly-distributed developmental time points
STAGE_WINDOW = 3.0        # ± hpf collected around each target stage
N_PER_BAND = 6            # snips per band (low / mid-low / mid-high / high)


def _resolve_snip_path(row: pd.Series) -> str:
    return str(SNIP_ROOT / str(row["experiment_date"]) / f"{row['snip_id']}.jpg")


def main(dim_suffix: int = 85) -> None:
    dim_col = f"z_mu_b_{dim_suffix}"
    out_dir = _HERE / "figures" / "dim_snips" / f"z_mu_b_{dim_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _apr.load_aggregate_dataframe()
    df = df.dropna(subset=[dim_col, "predicted_stage_hpf", "snip_id", "experiment_date"]).copy()
    df["experiment_date"] = df["experiment_date"].astype(str)

    # Resolve snip image paths and keep only rows whose snip actually exists.
    df["image_path"] = df.apply(_resolve_snip_path, axis=1)
    exists = df["image_path"].map(lambda p: Path(p).exists())
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"Dropping {n_missing}/{len(df)} rows whose snip image is missing on disk")
    df = df[exists].copy()

    # 4 evenly-distributed target stages across the observed range.
    lo, hi = df["predicted_stage_hpf"].quantile([0.02, 0.98])
    targets = np.linspace(lo, hi, N_TIMEPOINTS)
    print(f"Interpreting {dim_col} at stages: {[round(t, 1) for t in targets]} hpf")

    for tgt in targets:
        sel = df[(df["predicted_stage_hpf"] - tgt).abs() <= STAGE_WINDOW].copy()
        if len(sel) < 2 * N_PER_BAND:
            print(f"  t={tgt:.0f}hpf: only {len(sel)} snips in window — skipping")
            continue

        # Cutoff = median dim value in this window, so bands split evenly into
        # low-expressing (worst_of_worst) vs high-expressing (clear_pass).
        cutoff = float(sel[dim_col].median())
        out_path = out_dir / f"gallery_t{tgt:.0f}hpf.png"
        render_quartile_gallery(
            sel,
            dim_col,
            cutoff,
            fail_direction="below",  # "fail" side = LOW expressers (bottom bands)
            image_path_col="image_path",
            label_col="snip_id",
            title=(
                f"{dim_col} morphology @ ~{tgt:.0f} hpf  "
                f"(bottom=lowest, top=highest expressers; median cutoff={cutoff:.2f})"
            ),
            output_path=out_path,
            n_per_band=N_PER_BAND,
        )
        print(f"  t={tgt:.0f}hpf: {len(sel)} snips -> {out_path}")

    print(f"\nDone. Galleries in {out_dir}")


if __name__ == "__main__":
    suffix = int(sys.argv[1]) if len(sys.argv) > 1 else 85
    main(suffix)
