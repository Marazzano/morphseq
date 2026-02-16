"""
Phase 1 / Task 1.5 — JAX trainer (logistic + L1 + TV).

Checks:
- Training converges (objective decreases)
- Objective log contains all required terms
- Weight map concentrates in the tail (planted signal region)
- compute_logits shape and value sanity
- L1 and TV penalties are active (non-zero in log)

NOTE: Requires JAX. Tests are skipped if JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from conftest import GRID_H, GRID_W, N_CHANNELS
from roi_trainer import TrainResult, compute_logits, train


# ---- compute_logits ----

def test_compute_logits_shape(planted_data):
    """compute_logits returns (N,) array."""
    X = planted_data["X"]
    N = X.shape[0]
    w_full = np.random.randn(GRID_H, GRID_W, N_CHANNELS).astype(np.float32)
    b = 0.0

    logits = compute_logits(X, w_full, b)
    assert logits.shape == (N,)


def test_compute_logits_bias_shift(planted_data):
    """Adding bias b should shift all logits by b."""
    X = planted_data["X"]
    w_full = np.random.randn(GRID_H, GRID_W, N_CHANNELS).astype(np.float32)

    logits_0 = compute_logits(X, w_full, 0.0)
    logits_5 = compute_logits(X, w_full, 5.0)

    np.testing.assert_allclose(logits_5 - logits_0, 5.0, atol=1e-5)


def test_compute_logits_rejects_wrong_shapes():
    """3D X or 2D w_full should raise ValueError."""
    X_bad = np.zeros((10, 32, 32))  # missing channel dim
    w = np.zeros((32, 32, 1))

    with pytest.raises(ValueError):
        compute_logits(X_bad, w, 0.0)


# ---- Training ----

def test_trainer_converges(planted_data, class_weights, tiny_trainer_config):
    """Objective should decrease over training."""
    result = train(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        class_weights=class_weights,
        lam=1e-3,
        mu=1e-3,
        config=tiny_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    assert isinstance(result, TrainResult)
    assert len(result.objective_log) >= 2

    first_loss = result.objective_log[0]["total_objective"]
    last_loss = result.objective_log[-1]["total_objective"]
    assert last_loss < first_loss, (
        f"Training did not reduce objective: first={first_loss}, last={last_loss}"
    )


def test_objective_log_has_required_terms(planted_data, class_weights, tiny_trainer_config):
    """Each log entry must contain the terms specified in PLAN.md Section D."""
    result = train(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        class_weights=class_weights,
        lam=1e-3,
        mu=1e-3,
        config=tiny_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    required_keys = {
        "logistic_loss_raw", "l1_raw", "tv_raw",
        "l1_weighted", "tv_weighted", "total_objective",
    }
    for entry in result.objective_log:
        missing = required_keys - set(entry.keys())
        assert not missing, f"Missing objective log keys: {missing}"


def test_weight_map_tail_concentration(planted_data, class_weights, tiny_trainer_config):
    """
    BIOLOGICAL PRIOR CHECK (cep290):
    The weight map magnitude should be higher in the tail region (bottom rows)
    than in the head region (top rows), since that's where signal was planted.
    """
    from conftest import TAIL_START_ROW

    result = train(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        class_weights=class_weights,
        lam=1e-3,
        mu=1e-3,
        config=tiny_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    w_mag = np.sqrt(np.sum(result.w_full ** 2, axis=-1))
    mask = planted_data["mask_ref"]

    # Mean weight magnitude in tail vs head
    tail_mask = np.zeros_like(mask)
    tail_mask[TAIL_START_ROW:, :] = True
    tail_mask = tail_mask & mask
    head_mask = np.zeros_like(mask)
    head_mask[:TAIL_START_ROW, :] = True
    head_mask = head_mask & mask

    tail_mean = w_mag[tail_mask].mean() if tail_mask.any() else 0
    head_mean = w_mag[head_mask].mean() if head_mask.any() else 0

    assert tail_mean > head_mean, (
        f"Weight map should concentrate in tail: tail_mean={tail_mean:.4f}, head_mean={head_mean:.4f}"
    )


def test_train_result_channel_names(planted_data, class_weights, tiny_trainer_config):
    """TrainResult must carry channel_names (required for Phase 2)."""
    result = train(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        class_weights=class_weights,
        lam=1e-3,
        mu=1e-3,
        config=tiny_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    assert result.channel_names == ("total_cost",)
    assert "channel_names" in result.config


def test_l1_tv_penalties_nonzero(planted_data, class_weights, tiny_trainer_config):
    """With non-zero lam and mu, both L1 and TV weighted terms should be > 0."""
    result = train(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        class_weights=class_weights,
        lam=1e-2,
        mu=1e-2,
        config=tiny_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    last_entry = result.objective_log[-1]
    assert last_entry["l1_weighted"] > 0, "L1 weighted penalty should be > 0"
    assert last_entry["tv_weighted"] > 0, "TV weighted penalty should be > 0"
