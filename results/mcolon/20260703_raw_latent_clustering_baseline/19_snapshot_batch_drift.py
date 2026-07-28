"""Measure whether condensation creates batch fragmentation over saved snapshots.

This uses the already-saved raw z_mu_b run; it does not rerun optimization.
At each snapshot it measures, within stage-matched slices:
  * cross-experiment / same-experiment mean distance, and
  * the same ratio with genotype held fixed.

If fragmentation appears only late, the solver duration is implicated.  If it
is present at x0 and increases immediately, the attraction graph is operating
on a batch-structured local geometry instead.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
NPZ = HERE / "figures" / "condensed_raw_zmub" / "condensed_positions.npz"
TABLE_OUT = HERE / "tables" / "snapshot_batch_drift.csv"
FIG_OUT = HERE / "figures" / "snapshot_batch_drift.png"


def experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if parts[1].isalpha() else parts[0]


def mean_cross_same_ratio(coords, mask, experiments, labels, *, hold_genotype):
    ratios = []
    for ti in range(mask.shape[1]):
        present = np.flatnonzero(mask[:, ti])
        groups = np.unique(labels[present]) if hold_genotype else [None]
        for genotype in groups:
            idx = present if genotype is None else present[labels[present] == genotype]
            if len(idx) < 3 or len(np.unique(experiments[idx])) < 2:
                continue
            pos = coords[idx, ti, :]
            dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
            tri = np.triu_indices(len(idx), k=1)
            same = experiments[idx][tri[0]] == experiments[idx][tri[1]]
            if same.any() and (~same).any():
                ratios.append(dist[tri][~same].mean() / dist[tri][same].mean())
    return float(np.mean(ratios))


def main():
    d = np.load(NPZ, allow_pickle=True)
    mask = d["mask"]
    labels = np.asarray([str(x) for x in d["labels"]])
    experiments = np.asarray([experiment_of(str(x)) for x in d["embryo_ids"]])

    states = [("x0", -1, d["x0"])]
    states.extend(("snapshot", int(iteration), d["position_history"][i])
                  for i, iteration in enumerate(d["snapshot_iters"]))
    states.append(("final", 500, d["positions"]))

    rows = []
    for state, iteration, coords in states:
        displacement = np.sqrt(((coords - d["x0"]) ** 2).sum(axis=2))[mask].mean()
        rows.append({
            "state": state,
            "iteration": iteration,
            "cross_same_ratio": mean_cross_same_ratio(
                coords, mask, experiments, labels, hold_genotype=False),
            "cross_same_ratio_held_genotype": mean_cross_same_ratio(
                coords, mask, experiments, labels, hold_genotype=True),
            "mean_displacement_from_x0": displacement,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(out["iteration"], out["cross_same_ratio"], "o-", label="all pairs")
    ax.plot(out["iteration"], out["cross_same_ratio_held_genotype"], "o-",
            label="within genotype")
    ax.axhline(1, color="0.5", lw=1, ls="--")
    ax.set(xlabel="condensation iteration", ylabel="cross-experiment / same-experiment distance",
           title="Batch separation over saved condensation snapshots")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=180)
    print(f"Saved {TABLE_OUT}\nSaved {FIG_OUT}")


if __name__ == "__main__":
    main()
