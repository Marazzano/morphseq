"""Facet Build06 B9D2 total length by predicted phenotype, pair, and experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SHARED_SCRIPT = RUN_DIR / "05_plot_b9d2_curvature_by_phenotype_and_experiment.py"
OUTPUT_PNG = (
    RUN_DIR
    / "figures/b9d2_curvature_by_phenotype_and_experiment"
    / "total_length_faceted_by_phenotype_colored_by_experiment.png"
)


def _load_shared_module():
    spec = importlib.util.spec_from_file_location("b9d2_build06_phenotype_facets", SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared plotting implementation: {SHARED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    shared = _load_shared_module()
    shared.main(
        metric_col="total_length_um",
        y_label="Total length (µm)",
        output_path=OUTPUT_PNG,
        title="Build06 B9D2 predicted-phenotype length by pair and experiment",
    )


if __name__ == "__main__":
    main()
