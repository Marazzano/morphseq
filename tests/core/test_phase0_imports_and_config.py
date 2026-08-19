import unittest

from src.core.lightning.train_config import LitTrainConfig


class Phase0ImportAndConfigTests(unittest.TestCase):
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
