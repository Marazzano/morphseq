from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.core.losses import loss_functions
from src.core.losses.loss_configs import BasicLoss, MetricLoss
from src.core.losses.loss_functions import NTXentLoss
from src.core.losses.metric_loss import (
    MetricLossConfigurationError,
    MetricTargetMasks,
    build_metric_target_masks,
    euclidean_supcon_lout,
    supcon_lout_from_logits,
    validate_metric_parameters,
)
from src.core.models.model_utils import ModelOutput


def _directed_cycle_targets() -> MetricTargetMasks:
    return MetricTargetMasks(
        positive=torch.tensor(
            [[False, True, False], [False, False, True], [True, False, False]]
        ),
        denominator=~torch.eye(3, dtype=torch.bool),
        context="policy=hand_computed_one_positive, loss_age_window=fixture",
    )


def test_lout_matches_hand_computed_one_positive_per_anchor() -> None:
    logits = torch.tensor(
        [[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [4.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    expected = torch.stack(
        (
            torch.logsumexp(torch.tensor([2.0, 0.0], dtype=torch.float64), dim=0) - 2.0,
            torch.logsumexp(torch.tensor([1.0, 3.0], dtype=torch.float64), dim=0) - 3.0,
            torch.logsumexp(torch.tensor([4.0, 2.0], dtype=torch.float64), dim=0) - 4.0,
        )
    ).mean()

    actual = supcon_lout_from_logits(logits, _directed_cycle_targets())
    assert actual.item() == pytest.approx(expected.item(), abs=1e-12)


def test_lout_matches_hand_computed_multi_positive_and_differs_from_lin() -> None:
    logits = torch.tensor(
        [[0.0, 3.0, -1.0], [2.0, 0.0, 0.0], [-2.0, 4.0, 0.0]],
        dtype=torch.float64,
    )
    off_diagonal = ~torch.eye(3, dtype=torch.bool)
    targets = MetricTargetMasks(
        positive=off_diagonal,
        denominator=off_diagonal,
        context="policy=hand_computed_multi_positive, loss_age_window=fixture",
    )
    expected_rows = []
    legacy_lin_rows = []
    for row_index in range(3):
        row = logits[row_index, off_diagonal[row_index]]
        expected_rows.append(torch.logsumexp(row, dim=0) - row.mean())
        legacy_lin_rows.append(-(torch.logsumexp(row, dim=0) - torch.logsumexp(row, dim=0)))
    expected_lout = torch.stack(expected_rows).mean()
    legacy_lin = torch.stack(legacy_lin_rows).mean()

    actual = supcon_lout_from_logits(logits, targets)
    assert actual.item() == pytest.approx(expected_lout.item(), abs=1e-12)
    assert legacy_lin.item() == pytest.approx(0.0)
    assert actual.item() != pytest.approx(legacy_lin.item())


def test_explicit_negatives_affect_only_the_denominator() -> None:
    targets = _directed_cycle_targets()
    baseline = torch.tensor(
        [[0.0, 1.0, -2.0], [-2.0, 0.0, 1.0], [1.0, -2.0, 0.0]],
        dtype=torch.float64,
    )
    harder_negatives = baseline.clone()
    harder_negatives[0, 2] = 3.0
    harder_negatives[1, 0] = 3.0
    harder_negatives[2, 1] = 3.0

    assert supcon_lout_from_logits(harder_negatives, targets) > supcon_lout_from_logits(
        baseline, targets
    )


def test_euclidean_similarity_preserves_legacy_dimension_scaling() -> None:
    features = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.5, 0.0]], dtype=torch.float64
    )
    targets = _directed_cycle_targets()
    temperature = 0.25
    legacy_logits = -(
        torch.cdist(features, features, p=2).pow(2) / (features.shape[1] / 2)
    ).pow(0.5) / temperature

    expected = supcon_lout_from_logits(legacy_logits, targets)
    actual = euclidean_supcon_lout(features, targets, temperature=temperature)
    assert actual.item() == pytest.approx(expected.item(), abs=1e-12)


def test_ntxent_selects_only_biological_latents_and_gradients_are_finite_nonzero() -> None:
    module = NTXentLoss.__new__(NTXentLoss)
    nn.Module.__init__(module)
    module.cfg = SimpleNamespace(
        biological_indices=torch.tensor([2, 3]),
        relation_policy=None,
        self_target_prob=1.0,
        temperature=0.5,
    )
    features = torch.tensor(
        [
            [100.0, -100.0, 0.0, 0.0],
            [-50.0, 75.0, 2.0, 0.0],
            [-200.0, 300.0, 0.2, 0.1],
            [400.0, -500.0, 2.3, -0.2],
        ],
        requires_grad=True,
    )

    loss = module._nt_xent_loss_euclidean(features)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()
    assert torch.equal(features.grad[:, :2], torch.zeros_like(features.grad[:, :2]))
    assert torch.count_nonzero(features.grad[:, 2:]) > 0


def test_euclidean_logits_are_stable_at_extreme_float32_distance_and_temperature() -> None:
    features = torch.tensor(
        [[1e20, 0.0], [5e19, 0.0], [-1e20, 0.0], [-5e19, 0.0]],
        dtype=torch.float32,
    )
    targets = build_metric_target_masks(
        num_features=4,
        explicit_positive_pairs=((0, 1), (2, 3)),
    )

    loss = euclidean_supcon_lout(features, targets, temperature=1e-12)
    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)


def test_sampler_and_loss_age_windows_are_distinct_configuration_concepts() -> None:
    cfg = MetricLoss(
        latent_dim=4,
        sampler_age_window=0.5,
        loss_age_window=2.25,
    )

    assert cfg.sampler_age_window == pytest.approx(0.5)
    assert cfg.loss_age_window == pytest.approx(2.25)
    assert cfg.time_window == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"temperature": 0.0}, "contrastive_temperature must be > 0"),
        ({"temperature": float("nan")}, "contrastive_temperature must be finite"),
        ({"metric_weight": -1.0}, "metric_weight must be >= 0"),
        ({"sampler_age_window": float("inf")}, "sampler_age_window must be finite"),
        ({"loss_age_window": -0.1}, "loss_age_window must be >= 0"),
    ],
)
def test_invalid_or_nonfinite_metric_parameters_fail(overrides, match) -> None:
    values = {
        "temperature": 0.1,
        "metric_weight": 1.0,
        "sampler_age_window": 1.5,
        "loss_age_window": 2.0,
    }
    values.update(overrides)
    with pytest.raises(MetricLossConfigurationError, match=match):
        validate_metric_parameters(**values)


class _DummyLPIPS(nn.Module):
    def forward(self, x, y):
        return torch.zeros((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)


def test_metric_forward_preserves_unrelated_vae_components(monkeypatch) -> None:
    monkeypatch.setattr(loss_functions.lpips, "LPIPS", lambda **_: _DummyLPIPS())
    shared = dict(
        max_epochs=1,
        input_dim=(1, 4, 3),
        reconstruction_loss="L2",
        kld_weight=0.7,
        pips_flag=False,
        pips_weight=0.0,
        use_gan=False,
        use_pips_eval=False,
    )
    basic = BasicLoss(**shared).create_module()
    metric = MetricLoss(
        **shared,
        latent_dim=4,
        frac_nuisance_latents=0.25,
        self_target_prob=1.0,
        metric_weight=0.3,
    ).create_module()
    x0 = torch.rand(2, 1, 4, 3)
    model_output = ModelOutput(
        recon_x=torch.rand(2, 1, 4, 3),
        mu=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0],
             [0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 2.3, -0.2]]
        ),
        logvar=torch.zeros(4, 4),
    )
    basic_output = basic({"data": x0}, model_output)
    metric_output = metric(
        {
            "data": torch.stack((x0, x0), dim=1),
            "self_stats": [None, None, None],
            "other_stats": [None, None, None],
        },
        model_output,
    )

    for name in ("pixel_loss", "kld_loss", "pips_loss", "gan_loss"):
        assert torch.as_tensor(metric_output[name]).item() == pytest.approx(
            torch.as_tensor(basic_output[name]).item()
        )
