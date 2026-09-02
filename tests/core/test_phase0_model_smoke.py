from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.core.losses import loss_functions
from src.core.losses.loss_configs import BasicLoss, MetricLoss
from src.core.models.legacy_models import VAE, metricVAE
from src.core.models.model_utils import ModelOutput


INPUT_DIM = (1, 288, 128)
LATENT_DIM = 4


class _DummyLPIPS(nn.Module):
    def forward(self, x, y):
        return torch.zeros((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)


class _TinyEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, logvar_value=20.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mu_head = nn.Linear(INPUT_DIM[0], latent_dim)
        self.logvar = nn.Parameter(torch.full((latent_dim,), logvar_value))

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        return ModelOutput(
            embedding=self.mu_head(pooled),
            log_covariance=self.logvar.unsqueeze(0).expand(x.shape[0], -1),
        )


class _TinyDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.output_head = nn.Linear(latent_dim, INPUT_DIM[0])

    def forward(self, z):
        pixels = torch.sigmoid(self.output_head(z)).view(z.shape[0], INPUT_DIM[0], 1, 1)
        return ModelOutput(reconstruction=pixels.expand(z.shape[0], *INPUT_DIM))


def _assert_finite_loss_terms(test_case, output, names):
    for name in names:
        value = torch.as_tensor(output[name])
        test_case.assertTrue(torch.isfinite(value).all(), f"{name} was not finite: {value}")


class Phase0ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        self.lpips_patch = patch.object(
            loss_functions.lpips,
            "LPIPS",
            side_effect=lambda **_: _DummyLPIPS(),
        )
        self.lpips_patch.start()

    def tearDown(self):
        self.lpips_patch.stop()

    def test_basic_vae_cpu_forward_backward(self):
        cfg = BasicLoss(
            max_epochs=1,
            input_dim=INPUT_DIM,
            pips_flag=False,
            pips_weight=0.0,
            use_gan=False,
            use_pips_eval=False,
        )
        model = VAE(
            config=SimpleNamespace(),
            encoder=_TinyEncoder(),
            decoder=_TinyDecoder(),
        )
        loss_fn = cfg.create_module()
        batch = {"data": torch.rand(2, *INPUT_DIM)}

        model_output = model(batch["data"])
        loss_output = loss_fn(batch, model_output)
        loss_output.loss.backward()

        self.assertEqual(model_output.recon_x.shape, batch["data"].shape)
        self.assertLessEqual(model_output.logvar.max().item(), 5.0)
        _assert_finite_loss_terms(
            self,
            loss_output,
            ("loss", "recon_loss", "pixel_loss", "kld_loss", "pips_loss", "gan_loss"),
        )
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_metric_vae_cpu_forward_backward_with_paired_views(self):
        cfg = MetricLoss(
            max_epochs=1,
            input_dim=INPUT_DIM,
            latent_dim=LATENT_DIM,
            frac_nuisance_latents=0.25,
            self_target_prob=1.0,
            pips_flag=False,
            pips_weight=0.0,
            use_gan=False,
            use_pips_eval=False,
        )
        model = metricVAE(
            config=SimpleNamespace(lossconfig=cfg),
            encoder=_TinyEncoder(),
            decoder=_TinyDecoder(),
        )
        loss_fn = cfg.create_module()
        paired_images = torch.rand(2, 2, *INPUT_DIM)
        self.assertEqual(tuple(paired_images.shape[1:]), (2, 1, 288, 128))

        batch = {
            "data": paired_images,
            "self_stats": [
                torch.tensor([0, 1]),
                torch.tensor([10.0, 20.0]),
                torch.tensor([0, 0]),
            ],
            "other_stats": [
                torch.tensor([0, 1]),
                torch.tensor([10.0, 20.0]),
                torch.tensor([0, 0]),
            ],
        }

        model_output = model(batch["data"])
        loss_output = loss_fn(batch, model_output)
        loss_output.loss.backward()

        self.assertEqual(model_output.recon_x.shape, paired_images[:, 0].shape)
        self.assertEqual(model_output.mu.shape, (4, LATENT_DIM))
        self.assertLessEqual(model_output.logvar.max().item(), 5.0)
        _assert_finite_loss_terms(
            self,
            loss_output,
            (
                "loss",
                "recon_loss",
                "metric_loss",
                "pixel_loss",
                "kld_loss",
                "pips_loss",
                "gan_loss",
            ),
        )
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_pixel_scale_uses_configured_image_geometry(self):
        default_loss = BasicLoss(
            input_dim=(1, 288, 128),
            pips_flag=False,
            use_gan=False,
            use_pips_eval=False,
        ).create_module()
        alternate_loss = BasicLoss(
            input_dim=(1, 64, 32),
            pips_flag=False,
            use_gan=False,
            use_pips_eval=False,
        ).create_module()

        self.assertAlmostEqual(default_loss._pixel_scale(), (288 * 128) / 100)
        self.assertAlmostEqual(alternate_loss._pixel_scale(), (64 * 32) / 100)


if __name__ == "__main__":
    unittest.main()
