"""Summary metrics for UOT transport results."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from .config import Coupling

if TYPE_CHECKING:
    from .pair_frame import PairFrameGeometry


def summarize_metrics(
    cost: float,
    coupling: Coupling,
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    metric: str,
    m_src: float = None,
    m_tgt: float = None,
    pair_frame: "PairFrameGeometry" = None,
) -> Dict[str, float]:
    if coupling is None:
        transported_mass = float("nan")
    else:
        transported_mass = float(np.asarray(coupling).sum())

    created_mass = float(mass_created_hw.sum())
    destroyed_mass = float(mass_destroyed_hw.sum())

    mean_transport_cost = float("nan")
    mean_transport_distance = float("nan")
    if transported_mass > 0:
        mean_transport_cost = float(cost / transported_mass)
        if metric == "sqeuclidean":
            mean_transport_distance = float(np.sqrt(mean_transport_cost))
        else:
            mean_transport_distance = mean_transport_cost

    # Add percentage-based metrics if source and target masses are provided
    created_mass_pct = float("nan")
    destroyed_mass_pct = float("nan")
    proportion_transported = float("nan")

    if m_tgt is not None and m_tgt > 0:
        created_mass_pct = 100.0 * created_mass / m_tgt

    if m_src is not None and m_src > 0:
        destroyed_mass_pct = 100.0 * destroyed_mass / m_src

    if m_src is not None and m_tgt is not None and min(m_src, m_tgt) > 0:
        proportion_transported = transported_mass / min(m_src, m_tgt)

    # Physical area calculations using pair_frame
    created_area_um2 = float("nan")
    destroyed_area_um2 = float("nan")
    if pair_frame is not None:
        created_area_um2 = float(created_mass * pair_frame.px_area_um2)
        destroyed_area_um2 = float(destroyed_mass * pair_frame.px_area_um2)

    return {
        "transported_mass": transported_mass,
        "created_mass": created_mass,
        "destroyed_mass": destroyed_mass,
        "mean_transport_cost": mean_transport_cost,
        "mean_transport_distance": mean_transport_distance,
        "created_mass_pct": created_mass_pct,
        "destroyed_mass_pct": destroyed_mass_pct,
        "proportion_transported": proportion_transported,
        "created_area_um2": created_area_um2,
        "destroyed_area_um2": destroyed_area_um2,
    }
