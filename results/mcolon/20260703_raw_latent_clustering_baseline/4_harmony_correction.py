"""
4_harmony_correction.py
------------------------
Unsupervised experiment/batch correction of raw z_mu_b, using Harmony
(Korsunsky et al. 2019, Nat. Methods) -- an iterative soft k-means +
per-cluster/per-batch linear correction operating directly on a fixed
low-dimensional embedding. No genotype/cell-type labels are used to fit
the correction; genotype is only consulted downstream (in
3_cross_experiment_diagnostic.py) to check the correction did not remove
real phenotype signal (the "biology-preservation" axis of Luecken et al.
2022, Nat. Methods, "Benchmarking atlas-level data integration").

Design choices (see PLAN / STATUS.md for full reasoning):
- Correction is applied PER TIME BIN, independently. Each 4hpf bin's
  present embryos are corrected using only the experiments present in
  that bin. Bins with a single experiment present pass through unchanged
  (nothing to correct against -- Harmony would have no diversity signal
  to act on, and forcing it would risk manufacturing a correction).
- `theta` (per-cluster batch-diversity penalty) is swept over a small grid
  including 0 (no-op passthrough, used as a sanity baseline matching the
  uncorrected raw arm) so the effect of correction strength can be read
  off directly via 3_cross_experiment_diagnostic.py rather than guessed.
- `nclust` is capped relative to the smallest per-experiment group size in
  each bin (Harmony's own default of min(N/30, 100) is too coarse for our
  N ~60-190 per bin -- would yield 2-6 clusters, too few for meaningful
  per-cluster batch correction). We use nclust = max(2, min(20, N // 10)).

Outputs:
  figures/harmony_corrected/z_corrected_theta{THETA}.npz
    -- same (N_e, T, K) tensor shape as the input, corrected in-place
       per bin; embryo_ids/time_values/mask/labels carried through unchanged
       so downstream scripts can consume it exactly like the raw tensor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import harmonypy

_HERE = Path(__file__).resolve().parent
TABLES = _HERE / "tables"
OUT_DIR = _HERE / "figures" / "harmony_corrected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THETA_GRID = [0.0, 1.0, 2.0, 4.0]


def _experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    if parts[1].isalpha():
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _pivot_to_tensor(
    binned: pd.DataFrame, z_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot (embryo_id, time_bin, z_cols) -> (N_e, T, K), mask, embryo_ids, time_values, labels."""
    embryo_ids = np.array(sorted(binned["embryo_id"].unique()))
    time_values = np.array(sorted(binned["time_bin"].unique()), dtype=float)
    N_e, T, K = len(embryo_ids), len(time_values), len(z_cols)

    eid_idx = {e: i for i, e in enumerate(embryo_ids)}
    t_idx = {t: i for i, t in enumerate(time_values)}

    features = np.full((N_e, T, K), np.nan)
    mask = np.zeros((N_e, T), dtype=bool)
    labels_arr = np.full(N_e, "", dtype=object)

    for _, row in binned.iterrows():
        ei = eid_idx[row["embryo_id"]]
        ti = t_idx[row["time_bin"]]
        features[ei, ti, :] = row[z_cols].values.astype(float)
        mask[ei, ti] = True
        labels_arr[ei] = str(row.get("genotype", ""))

    return features, mask, embryo_ids, time_values, labels_arr


def _correct_bin(
    z_bin: np.ndarray, experiment_labels: np.ndarray, theta: float
) -> np.ndarray:
    """z_bin: (n_present, K). Returns corrected (n_present, K), or z_bin
    unchanged if <2 experiments present or theta == 0 (explicit no-op)."""
    n_present = z_bin.shape[0]
    if theta == 0.0 or len(set(experiment_labels)) < 2 or n_present < 6:
        return z_bin.copy()

    meta = pd.DataFrame({"experiment": experiment_labels})
    n_per_experiment = meta["experiment"].value_counts().min()
    nclust = max(2, min(20, n_present // 10, n_per_experiment))

    ho = harmonypy.run_harmony(
        z_bin,
        meta,
        vars_use=["experiment"],
        theta=theta,
        nclust=nclust,
        random_state=42,
        verbose=False,
    )
    return np.asarray(ho.Z_corr)


def main() -> None:
    binned = pd.read_csv(TABLES / "pbx_binned_zmub.csv", low_memory=False)
    z_cols = [c for c in binned.columns if "z_mu_b" in c]
    print(f"Loaded: {len(binned)} rows, {len(z_cols)} z_mu_b dims")

    features, mask, embryo_ids, time_values, labels_arr = _pivot_to_tensor(binned, z_cols)
    experiment_labels_full = np.array([_experiment_of(e) for e in embryo_ids])
    print(f"Tensor shape: {features.shape}")

    for theta in THETA_GRID:
        print(f"\n=== theta={theta} ===")
        corrected = features.copy()
        n_bins_corrected = 0

        for ti, t in enumerate(time_values):
            present = mask[:, ti]
            n_present = present.sum()
            if n_present < 3:
                continue

            z_bin = features[present, ti, :]
            exp_bin = experiment_labels_full[present]

            if theta > 0.0 and len(set(exp_bin)) >= 2 and n_present >= 6:
                corrected[present, ti, :] = _correct_bin(z_bin, exp_bin, theta)
                n_bins_corrected += 1
            # else: passthrough, already copied via corrected = features.copy()

        print(f"  Corrected {n_bins_corrected}/{len(time_values)} bins "
              f"(remaining bins passed through unchanged: <2 experiments present or theta=0)")

        out_path = OUT_DIR / f"z_corrected_theta{theta:g}.npz"
        np.savez(
            out_path,
            features=corrected,
            mask=mask,
            embryo_ids=embryo_ids,
            time_values=time_values,
            labels=labels_arr,
            theta=theta,
        )
        print(f"  Saved -> {out_path}")

    print("\nDone. Run 1_cluster_raw_latent.py with --harmony-theta <value> next.")


if __name__ == "__main__":
    main()
