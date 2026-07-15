"""Render the existing Harmony theta=4 condensation, colored by experiment."""
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


RUN_DIR = HERE / "figures" / "condensed_harmony_theta4"
NPZ_PATH = RUN_DIR / "condensed_positions.npz"
EXPERIMENT_COLORS = {
    "20251207_pbx": "#6A3D9A",
    "20260304": "#1B9E77",
    "20260306": "#D95F02",
}


def experiment_of(embryo_id: str) -> str:
    parts = embryo_id.split("_")
    return "_".join(parts[:2]) if parts[1].isalpha() else parts[0]


def main() -> None:
    run = tc.load_run(NPZ_PATH, title="Harmony θ=4 condensation", color_map={})
    experiments = np.asarray([experiment_of(str(embryo_id)) for embryo_id in run.embryo_ids])
    snapshot_times = [float(t) for t in run.time_values[::5]]
    if float(run.time_values[-1]) not in snapshot_times:
        snapshot_times.append(float(run.time_values[-1]))

    fig, _ = tc.plotting.plot_panels(
        run.positions, run.mask, run.time_values,
        labels=experiments,
        color_map=EXPERIMENT_COLORS,
        snapshot_times=snapshot_times,
        title="Harmony θ=4 final condensation, colored by experiment",
    )
    png_path = RUN_DIR / "plot_panels_by_experiment.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.clf()

    html_path = RUN_DIR / "time_slice_by_experiment.html"
    tc.time_slice_html(
        run.positions, run.mask, run.time_values,
        labels=experiments,
        color_map=EXPERIMENT_COLORS,
        embryo_ids=run.embryo_ids,
        title="Harmony θ=4 final condensation, colored by experiment",
        output_path=html_path,
    )
    print(f"Saved {png_path}\nSaved {html_path}")


if __name__ == "__main__":
    main()
