"""
0_load_pbx_latents.py
---------------------
Load PBX embryo latents, bin by time (4 hpf bins), and persist to tables/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE.parent / "20260329_pbx_crispant_analysis_cont"))

from analyze.utils.binning import bin_embryos_by_time
from common import load_bridge_ready_dataframe, SHARED_GENOTYPES

OUT_DIR = _HERE / "tables"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Loading PBX latents...")
    df = load_bridge_ready_dataframe(genotypes=SHARED_GENOTYPES)
    print(f"  Rows after load/filter: {len(df):,}  embryos: {df['embryo_id'].nunique()}")

    # Use stage_hpf_bridge (predicted_stage_hpf with reconstructed fallback); drop rows without it
    n_before = len(df)
    df = df[df["stage_hpf_bridge"].notna()].copy()
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with no stage estimate")

    # bin_embryos_by_time auto-detects z_mu_b* cols (biological block only)
    binned = bin_embryos_by_time(df, time_col="stage_hpf_bridge", bin_width=4.0)
    print(f"  Binned rows: {len(binned):,}  (embryo, bin) pairs")

    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    print(f"  Latent dimensions: {len(z_cols)}")
    print(f"  Bins: {sorted(binned['time_bin'].unique())}")

    out = OUT_DIR / "pbx_binned_zmub.csv"
    binned.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
