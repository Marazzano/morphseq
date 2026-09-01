from pathlib import Path
import unittest

from omegaconf import OmegaConf

from src.core.lightning.train_config import LitTrainConfig
from src.core.models.model_components.arch_configs import LegacyArchitecture
from src.core.models.model_configs import resolve_arch


class Phase0ImportAndConfigTests(unittest.TestCase):
    def test_pipeline_vanilla_uses_canonical_convvae_architecture_name(self):
        config_path = (
            Path(__file__).resolve().parents[2]
            / "src/core/hydra_configs/model/vae_pipeline_vanilla.yaml"
        )
        model_config = OmegaConf.load(config_path)

        architecture = resolve_arch(
            OmegaConf.to_container(model_config.ddconfig, resolve=True)
        )

        self.assertIsInstance(architecture, LegacyArchitecture)
        self.assertEqual(architecture.name, "convVAE")

    def test_trainer_hardware_can_be_configured_for_cpu(self):
        cfg = LitTrainConfig(
            accelerator="cpu",
            devices=1,
            strategy="auto",
            precision=32,
            benchmark=False,
        )

        self.assertEqual(cfg.accelerator, "cpu")
        self.assertEqual(cfg.devices, 1)
        self.assertEqual(cfg.strategy, "auto")
        self.assertEqual(cfg.precision, 32)
        self.assertFalse(cfg.benchmark)


if __name__ == "__main__":
    unittest.main()
