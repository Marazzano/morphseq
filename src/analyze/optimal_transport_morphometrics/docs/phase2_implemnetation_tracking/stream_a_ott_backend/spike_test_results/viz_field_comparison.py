#!/usr/bin/env python3
"""POT vs OTT field comparison using the PROVEN plotting code from debug_uot_params.py.

Uses the same plot_flow_field, plot_transport_cost_field, and
plot_creation_destruction_maps functions that produce known-good output.

Runs at eps=1e-4 (concordance sweet spot) and eps=1e-5 (production POT)
to directly compare both backends' velocity, cost, and mass fields.

USAGE:
    PYTHONPATH=src:$PYTHONPATH python src/analyze/optimal_transport_morphometrics/docs/phase2_implemnetation_tracking/stream_a_ott_backend/spike_test_results/viz_field_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

morphseq_root = Path(__file__).resolve().parents[7]  # spike_test_results -> ... -> src -> morphseq
sys.path.insert(0, str(morphseq_root / "src"))
# Also add root for debug_uot_params imports
sys.path.insert(0, str(morphseq_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyze.utils.optimal_transport import UOTConfig, UOTFramePair, POTBackend, MassMode
from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend
from analyze.optimal_transport_morphometrics.uot_masks.run_transport import run_uot_pair
from analyze.optimal_transport_morphometrics.uot_masks.frame_mask_io import load_mask_from_csv
from analyze.optimal_transport_morphometrics.uot_masks.preprocess import preprocess_pair_canonical
from analyze.optimal_transport_morphometrics.uot_masks.uot_grid import CanonicalGridConfig

# Import the proven plotting functions from debug_uot_params
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "debug_uot_params",
    str(morphseq_root / "results" / "mcolon" / "20260121_uot-mvp" / "debug_uot_params.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["debug_uot_params"] = _mod
_spec.loader.exec_module(_mod)
from debug_uot_params import (
    plot_flow_field,
    plot_flow_field_quiver,
    plot_transport_cost_field,
    plot_creation_destruction_maps,
    plot_mask_overlay_only,
    plot_overlay_transport_field,
    compute_mass_metrics,
    compute_velocity_metrics,
    VisualizationConfig,
    CANONICAL_GRID_SHAPE,
    COORD_SCALE,
    UM_PER_PX,
)

CSV_PATH = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
EMBRYO_A = "20251113_A05_e01"
EMBRYO_B = "20251113_E04_e01"
TARGET_HPF = 48.0
STAGE_TOL = 1.0

REG_M = 10.0
EPSILON_VALUES = [1e-4, 1e-5]

OUTPUT_DIR = Path(__file__).resolve().parent


def find_frame_at_stage(csv_path, embryo_id, target_hpf, tolerance_hpf):
    df = pd.read_csv(csv_path, usecols=["embryo_id", "frame_index", "predicted_stage_hpf"])
    subset = df[
        (df["embryo_id"] == embryo_id) &
        (df["predicted_stage_hpf"] >= target_hpf - tolerance_hpf) &
        (df["predicted_stage_hpf"] <= target_hpf + tolerance_hpf)
    ]
    if subset.empty:
        return None, None
    subset = subset.copy()
    subset["dist"] = (subset["predicted_stage_hpf"] - target_hpf).abs()
    closest = subset.loc[subset["dist"].idxmin()]
    return int(closest["frame_index"]), float(closest["predicted_stage_hpf"])


def make_config(epsilon):
    return UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=REG_M,
        downsample_factor=1,
        downsample_divisor=1,
        padding_px=16,
        mass_mode=MassMode.UNIFORM,
        align_mode="none",
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=COORD_SCALE,
        use_pair_frame=True,
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=UM_PER_PX,
        canonical_grid_shape_hw=CANONICAL_GRID_SHAPE,
        canonical_grid_align_mode="yolk",
        canonical_grid_center_mode="joint_centering",
    )


def run_and_plot(pair, epsilon, backend, backend_name, output_subdir, plot_src_mask, plot_tgt_mask):
    """Run transport and generate all proven-format plots."""
    config = make_config(epsilon)
    eps_str = f"{epsilon:.0e}".replace("-", "m")

    out_dir = output_subdir / f"{backend_name}_eps{eps_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running {backend_name} at eps={epsilon:.0e}...")
    t0 = time.time()
    result = run_uot_pair(pair, config=config, backend=backend)
    elapsed = time.time() - t0
    print(f"    cost={result.cost:.6f}  time={elapsed:.1f}s")

    # Compute metrics
    mass_diag = compute_mass_metrics(result, pair.src.embryo_mask, pair.tgt.embryo_mask)
    vel_diag = compute_velocity_metrics(result.velocity_px_per_frame_yx)

    created_pct = mass_diag.get("created_mass_pct", float("nan"))
    destroyed_pct = mass_diag.get("destroyed_mass_pct", float("nan"))
    proportion_transported = mass_diag.get("proportion_transported", float("nan"))

    print(f"    created={created_pct:.2f}%  destroyed={destroyed_pct:.2f}%  "
          f"proportion_transported={proportion_transported:.4f}")
    print(f"    mean_vel={vel_diag['mean_velocity_px']:.2f}  max_vel={vel_diag['max_velocity_px']:.2f}")

    # Generate all the proven-format plots
    viz_config = VisualizationConfig()

    # 1. Flow field (support mask + velocity magnitude + histogram)
    plot_flow_field(
        plot_src_mask, result, proportion_transported,
        out_dir / "flow_field.png", viz_config=viz_config,
    )

    # 2. Quiver plot
    plot_flow_field_quiver(
        plot_src_mask, result,
        out_dir / "flow_field_quiver.png",
        stride=6, viz_config=viz_config,
    )

    # 3. Transport cost per pixel
    plot_transport_cost_field(plot_src_mask, result, out_dir / "transport_cost_field.png")

    # 4. Mask overlay
    plot_mask_overlay_only(plot_src_mask, plot_tgt_mask, out_dir / "overlay_masks.png")

    # 5. Overlay + transport arrows
    plot_overlay_transport_field(
        plot_src_mask, plot_tgt_mask, result,
        out_dir / "overlay_transport.png",
        stride=6, viz_config=viz_config,
    )

    # 6. Creation/destruction maps
    plot_creation_destruction_maps(
        result.mass_created_px, result.mass_destroyed_px,
        created_pct, destroyed_pct,
        out_dir / "creation_destruction.png", viz_config=viz_config,
    )

    print(f"    Plots saved to {out_dir}")
    return result, mass_diag, vel_diag


def main():
    print("=" * 70)
    print("POT vs OTT FIELD COMPARISON (using proven debug_uot_params plotting)")
    print("=" * 70)

    # Load frames
    frame_a, stage_a = find_frame_at_stage(CSV_PATH, EMBRYO_A, TARGET_HPF, STAGE_TOL)
    frame_b, stage_b = find_frame_at_stage(CSV_PATH, EMBRYO_B, TARGET_HPF, STAGE_TOL)
    assert frame_a is not None and frame_b is not None
    print(f"  {EMBRYO_A} frame={frame_a} stage={stage_a:.1f} hpf")
    print(f"  {EMBRYO_B} frame={frame_b} stage={stage_b:.1f} hpf")

    data_root = morphseq_root / "morphseq_playground"
    src_frame = load_mask_from_csv(CSV_PATH, EMBRYO_A, frame_a, data_root=data_root)
    tgt_frame = load_mask_from_csv(CSV_PATH, EMBRYO_B, frame_b, data_root=data_root)
    pair = UOTFramePair(src=src_frame, tgt=tgt_frame)

    # Preprocess masks to canonical grid for plotting (same as debug_uot_params does)
    config_for_preprocess = make_config(1e-4)  # epsilon doesn't matter for preprocessing
    canonical_config = CanonicalGridConfig(
        reference_um_per_pixel=config_for_preprocess.canonical_grid_um_per_pixel,
        grid_shape_hw=config_for_preprocess.canonical_grid_shape_hw,
        align_mode=config_for_preprocess.canonical_grid_align_mode,
        downsample_factor=1,
    )
    plot_src_mask, plot_tgt_mask, preprocess_meta = preprocess_pair_canonical(
        pair.src, pair.tgt, config_for_preprocess, canonical_config
    )
    print(f"  Preprocessed to canonical grid: {plot_src_mask.shape}")

    comparison_dir = OUTPUT_DIR / "pot_vs_ott_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for eps in EPSILON_VALUES:
        print(f"\n{'='*60}")
        print(f"EPSILON = {eps:.0e}, REG_M = {REG_M}")
        print(f"{'='*60}")

        eps_dir = comparison_dir / f"eps_{eps:.0e}"
        eps_dir.mkdir(parents=True, exist_ok=True)

        # Run both backends
        pot_result, pot_mass, pot_vel = run_and_plot(
            pair, eps, POTBackend(), "POT", eps_dir, plot_src_mask, plot_tgt_mask
        )
        ott_result, ott_mass, ott_vel = run_and_plot(
            pair, eps, OTTBackend(), "OTT", eps_dir, plot_src_mask, plot_tgt_mask
        )

        # Concordance summary
        pct_diff = abs(ott_result.cost - pot_result.cost) / max(abs(pot_result.cost), 1e-12) * 100
        print(f"\n  CONCORDANCE at eps={eps:.0e}:")
        print(f"    Cost: POT={pot_result.cost:.6f}  OTT={ott_result.cost:.6f}  diff={pct_diff:.2f}%")
        print(f"    Created: POT={pot_mass.get('created_mass_pct', 0):.2f}%  OTT={ott_mass.get('created_mass_pct', 0):.2f}%")
        print(f"    Destroyed: POT={pot_mass.get('destroyed_mass_pct', 0):.2f}%  OTT={ott_mass.get('destroyed_mass_pct', 0):.2f}%")
        print(f"    Mean vel: POT={pot_vel['mean_velocity_px']:.2f}  OTT={ott_vel['mean_velocity_px']:.2f}")

    print(f"\nAll results in: {comparison_dir}")
    print("Compare POT/ vs OTT/ subdirectories for each epsilon value.")


if __name__ == "__main__":
    main()
