"""Summary metrics for UOT transport results."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import Coupling


def summarize_metrics(
    cost: float,
    coupling: Coupling,
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    metric: str,
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

    return {
        "transported_mass": transported_mass,
        "created_mass": created_mass,
        "destroyed_mass": destroyed_mass,
        "mean_transport_cost": mean_transport_cost,
        "mean_transport_distance": mean_transport_distance,
    }
