"""
0_load_tfap2_latents.py
-----------------------
Load the raw tfap2 embryo latents, bin by predicted stage (4 hpf bins), and
persist to tables/tfap2_binned_zmub.csv.

Analogue of 20260703_raw_latent_clustering_baseline/0_load_pbx_latents.py, but
for all 16 tfap2 genotypes. Reuses the already-aggregated first-pass parquet
(597 embryos, 80 z_mu_b dims, predicted_stage_hpf, experiment_id) — no
re-aggregation needed. experiment_id is carried through so the viewer's
experiment view can color by batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
# Reuse the tfap2 first-pass aggregate loader (load_aggregate_dataframe).
sys.path.insert(0, str(_HERE.parent / "20260413_tfap2_followup" / "scripts"))

from analyze.utils.binning import bin_embryos_by_time  # noqa: E402
from common import BIN_WIDTH  # local common.py  # noqa: E402

# common.py from the April tfap2 followup provides load_aggregate_dataframe.
# Import under an alias to avoid colliding with our local common.py above.
import importlib.util  # noqa: E402

_apr_common_path = _HERE.parent / "20260413_tfap2_followup" / "scripts" / "common.py"
_spec = importlib.util.spec_from_file_location("_tfap2_apr_common", _apr_common_path)
_apr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apr)

OUT_DIR = _HERE / "tables"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Loading tfap2 aggregate parquet...")
    df = _apr.load_aggregate_dataframe()
    print(f"  Rows: {len(df):,}  embryos: {df['embryo_id'].nunique()}  "
          f"genotypes: {df['genotype'].nunique()}")

    # Drop rows lacking a genotype label or a stage estimate.
    n_before = len(df)
    df = df[df["genotype"].notna() & df["predicted_stage_hpf"].notna()].copy()
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with null genotype/stage")

    # experiment_id must be a plain column carried through the bin. It is constant
    # per embryo, so bin_embryos_by_time's take-first meta merge preserves it.
    if "experiment_id" not in df.columns:
        raise ValueError("expected 'experiment_id' column in aggregate parquet")
    df["experiment_id"] = df["experiment_id"].astype(str)

    # bin_embryos_by_time auto-detects z_mu_b* cols (biological block only) and
    # keeps non-aggregated metadata (genotype, experiment_id) per embryo.
    binned = bin_embryos_by_time(
        df, time_col="predicted_stage_hpf", bin_width=BIN_WIDTH
    )
    print(f"  Binned rows: {len(binned):,}  (embryo, bin) pairs")

    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    print(f"  Latent dimensions: {len(z_cols)}")
    print(f"  Bins: {sorted(binned['time_bin'].unique())}")
    print(f"  Genotypes: {sorted(binned['genotype'].unique())}")
    print(f"  Experiments: {sorted(binned['experiment_id'].unique())}")

    out = OUT_DIR / "tfap2_binned_zmub.csv"
    binned.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
