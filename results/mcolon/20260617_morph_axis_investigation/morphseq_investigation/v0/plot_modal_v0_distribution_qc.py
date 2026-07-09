"""Render visual QA for the minimal V0 modal-organization distributions.

This is intentionally pre-metric. Its job is to verify that the generated point
clouds look like the intended density landscapes before running ordering tests.

Run:
    conda run -n morphseq-env --no-capture-output python \
        results/mcolon/20260617_morph_axis_investigation/morphseq_investigation/v0/plot_modal_v0_distribution_qc.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphseq_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/morphseq_xdg_cache")
sys.path.insert(0, str(RUN_DIR.parents[2] / "src"))
sys.path.insert(0, str(RUN_DIR))

from morphseq_investigation.plotting.modal_distribution_plotting import (  # noqa: E402
    DistributionVisualSpec,
    plot_v0_distribution_qc_grid,
)
from morphseq_investigation.v0.modal_v0_distributions import V0_DISTRIBUTIONS  # noqa: E402


DEFAULT_OUT = RUN_DIR / "plots" / "modal_v0_distribution_qc.png"


def build_visual_specs(n: int, seed: int) -> list[DistributionVisualSpec]:
    rng_master = np.random.default_rng(seed)
    out = []
    for spec in V0_DISTRIBUTIONS:
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        realization = spec.realize(n, rng)
        out.append(
            DistributionVisualSpec(
                distribution_id=spec.distribution_id,
                points=realization.points,
                component_labels=realization.component_labels,
                composed_grid=realization.truth.composed_grid,
                note=spec.note,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="Sample size per V0 distribution.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path.")
    args = parser.parse_args()

    specs = build_visual_specs(args.n, args.seed)
    out = plot_v0_distribution_qc_grid(
        specs,
        args.out,
        title=f"V0 modal density-composition visual QA (n={args.n}, seed={args.seed})",
        auto_scale=False,
        include_metric_row=True,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
