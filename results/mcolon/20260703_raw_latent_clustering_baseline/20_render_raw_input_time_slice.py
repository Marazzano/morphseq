"""Render the pre-condensation raw-z_mu_b UMAP initialization as a time-slice HTML.

The raw input is 80-dimensional and cannot be shown directly in a 2D slice.
This viewer therefore renders ``x0``: the aligned UMAP initialization produced
from those raw inputs, before any trajectory-condensation force is applied.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_CACHE = Path("/tmp") / "morphseq_20260703_condensation_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import analyze.trajectory_condensation as tc


RUN_DIR = HERE / "figures" / "condensed_raw_zmub_relative_api"
NPZ_PATH = RUN_DIR / "condensed_positions.npz"
OUT_PATH = RUN_DIR / "time_slice_raw_umap_init.html"
OUT_EXPERIMENT_PATH = RUN_DIR / "time_slice_raw_umap_init_by_experiment.html"
COLORS = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd",
    "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if parts[1].isalpha() else parts[0]


def main() -> None:
    run = tc.load_run(NPZ_PATH, title="raw z_mu_b UMAP initialization", color_map=COLORS)
    if run.x0 is None:
        raise ValueError(f"{NPZ_PATH} has no x0 initialization")
    tc.time_slice_html(
        run.x0,
        run.mask,
        run.time_values,
        labels=run.labels,
        color_map=run.color_map,
        embryo_ids=run.embryo_ids,
        title="raw z_mu_b input → aligned UMAP initialization (pre-condensation)",
        output_path=OUT_PATH,
    )
    experiments = np.asarray([experiment_of(str(embryo_id)) for embryo_id in run.embryo_ids])
    experiment_colors = {
        "20251207_pbx": "#6A3D9A",
        "20260304": "#1B9E77",
        "20260306": "#D95F02",
    }
    tc.time_slice_html(
        run.x0,
        run.mask,
        run.time_values,
        labels=experiments,
        color_map=experiment_colors,
        embryo_ids=run.embryo_ids,
        title="raw z_mu_b input → aligned UMAP initialization (pre-condensation), colored by experiment",
        output_path=OUT_EXPERIMENT_PATH,
    )
    print(f"Saved raw-input time-slice HTML: {OUT_PATH}\n"
          f"Saved experiment-colored time-slice HTML: {OUT_EXPERIMENT_PATH}")


if __name__ == "__main__":
    main()
