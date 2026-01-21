"""Create mass maps and velocity fields from UOT coupling."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .config import Coupling


def _compute_marginals(coupling: Coupling) -> Tuple[np.ndarray, np.ndarray]:
    if sp is not None and sp.issparse(coupling):
        mu_hat = np.asarray(coupling.sum(axis=1)).ravel()
        nu_hat = np.asarray(coupling.sum(axis=0)).ravel()
        return mu_hat, nu_hat
    coupling = np.asarray(coupling)
    return coupling.sum(axis=1), coupling.sum(axis=0)


def compute_transport_maps(
    coupling: Coupling,
    support_src_yx: np.ndarray,
    support_tgt_yx: np.ndarray,
    weights_src: np.ndarray,
    weights_tgt: np.ndarray,
    work_shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coupling is None:
        raise ValueError("Coupling is required to compute transport maps.")

    mu_hat, nu_hat = _compute_marginals(coupling)
    mass_destroyed = np.maximum(0.0, weights_src - mu_hat)
    mass_created = np.maximum(0.0, weights_tgt - nu_hat)

    mass_destroyed_hw = np.zeros(work_shape_hw, dtype=np.float32)
    mass_created_hw = np.zeros(work_shape_hw, dtype=np.float32)

    src_y = support_src_yx[:, 0].astype(int)
    src_x = support_src_yx[:, 1].astype(int)
    tgt_y = support_tgt_yx[:, 0].astype(int)
    tgt_x = support_tgt_yx[:, 1].astype(int)

    mass_destroyed_hw[src_y, src_x] = mass_destroyed.astype(np.float32)
    mass_created_hw[tgt_y, tgt_x] = mass_created.astype(np.float32)

    velocity_field = np.zeros((*work_shape_hw, 2), dtype=np.float32)

    if sp is not None and sp.issparse(coupling):
        coupling = coupling.tocoo()
        n_src = len(weights_src)
        sum_y = np.zeros((n_src, 2), dtype=np.float64)
        np.add.at(sum_y, coupling.row, coupling.data[:, None] * support_tgt_yx[coupling.col])
        mu_hat_safe = np.maximum(mu_hat, 1e-12)
        T = sum_y / mu_hat_safe[:, None]
    else:
        coupling_dense = np.asarray(coupling, dtype=np.float64)
        mu_hat_safe = np.maximum(mu_hat, 1e-12)
        T = (coupling_dense @ support_tgt_yx) / mu_hat_safe[:, None]

    v = T - support_src_yx
    velocity_field[src_y, src_x, :] = v.astype(np.float32)

    return mass_created_hw, mass_destroyed_hw, velocity_field
