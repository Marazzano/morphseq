"""Fit one shared Harmony correction across all raw z_mu_b observations.

Unlike ``4_harmony_correction.py``, this never fits independent per-time-bin
maps.  One correction is estimated over all observed embryo-time rows using
experiment as the sole batch covariate; developmental time is retained as
biological structure rather than supplied to Harmony.
"""
from __future__ import annotations

from pathlib import Path

import harmonypy
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
OUT_DIR = HERE / "figures" / "harmony_corrected_global"
OUT_DIR.mkdir(parents=True, exist_ok=True)
THETA_GRID = [4.0]


def experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if parts[1].isalpha() else parts[0]


def pivot_to_tensor(binned: pd.DataFrame, z_cols: list[str]):
    embryo_ids = np.array(sorted(binned["embryo_id"].unique()))
    time_values = np.array(sorted(binned["time_bin"].unique()), dtype=float)
    features = np.full((len(embryo_ids), len(time_values), len(z_cols)), np.nan)
    mask = np.zeros(features.shape[:2], dtype=bool)
    labels = np.full(len(embryo_ids), "", dtype=object)
    embryo_index = {embryo_id: i for i, embryo_id in enumerate(embryo_ids)}
    time_index = {time_bin: i for i, time_bin in enumerate(time_values)}
    for _, row in binned.iterrows():
        i, t = embryo_index[row["embryo_id"]], time_index[row["time_bin"]]
        features[i, t] = row[z_cols].to_numpy(dtype=float)
        mask[i, t] = True
        labels[i] = str(row["genotype"])
    return features, mask, embryo_ids, time_values, labels


def main() -> None:
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [column for column in binned if column.startswith("z_mu_b")]
    features, mask, embryo_ids, time_values, labels = pivot_to_tensor(binned, z_cols)
    observations = np.argwhere(mask)
    matrix = features[mask]
    experiments = np.asarray([experiment_of(str(embryo_ids[i])) for i, _ in observations])
    metadata = pd.DataFrame({"experiment": experiments})
    nclust = min(50, max(2, len(matrix) // 30))

    print(f"Global Harmony: {len(matrix)} observations, {len(z_cols)} dimensions, "
          f"experiments={sorted(set(experiments))}, nclust={nclust}")
    for theta in THETA_GRID:
        harmony = harmonypy.run_harmony(
            matrix, metadata, vars_use=["experiment"], theta=theta,
            nclust=nclust, random_state=42, verbose=False,
        )
        corrected = features.copy()
        corrected[mask] = np.asarray(harmony.Z_corr)
        output = OUT_DIR / f"z_corrected_global_theta{theta:g}.npz"
        np.savez(
            output, features=corrected, mask=mask, embryo_ids=embryo_ids,
            time_values=time_values, labels=labels, theta=theta,
            correction_scope="global",
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
