"""Render the global raw-space Harmony initializer and final condensation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

import analyze.trajectory_condensation as tc

RUN_DIR = HERE / "figures" / "condensed_harmony_global_theta4"
COLORS = {"20251207_pbx": "#6A3D9A", "20260304": "#1B9E77", "20260306": "#D95F02"}
GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC", "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd", "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if parts[1].isalpha() else parts[0]


def main() -> None:
    run = tc.load_run(RUN_DIR / "condensed_positions.npz", color_map={})
    experiments = np.asarray([experiment_of(str(embryo_id)) for embryo_id in run.embryo_ids])
    if run.x0 is None:
        raise ValueError("Global-Harmony run has no aligned-UMAP initializer")
    tc.time_slice_html(
        run.x0, run.mask, run.time_values,
        labels=experiments, color_map=COLORS, embryo_ids=run.embryo_ids,
        title="global Harmony on raw 80-D z_mu_b → aligned UMAP initializer | experiment",
        output_path=RUN_DIR / "initializer_by_experiment.html",
    )
    tc.time_slice_html(
        run.x0, run.mask, run.time_values,
        labels=run.labels, color_map=GENOTYPE_COLORS, embryo_ids=run.embryo_ids,
        title="global Harmony on raw 80-D z_mu_b → aligned UMAP initializer | genotype",
        output_path=RUN_DIR / "initializer_by_genotype.html",
    )
    tc.time_slice_html(
        run.positions, run.mask, run.time_values,
        labels=experiments, color_map=COLORS, embryo_ids=run.embryo_ids,
        title="global Harmony θ=4 final condensation, colored by experiment",
        output_path=RUN_DIR / "time_slice_by_experiment.html",
    )
    print("Saved global raw-space Harmony initializer and final time slices")


if __name__ == "__main__":
    main()
