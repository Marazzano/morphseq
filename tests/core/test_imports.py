from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import src.core.models.model_configs as model_configs
from src.core.data.dataset_configs import NTXentDataConfig


CONFIG_DIR = Path(__file__).resolve().parents[2] / "src" / "core" / "hydra_configs"


class CoreImportTests(unittest.TestCase):
    def test_metric_hydra_config_builds_concrete_training_data_config(self):
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
            cfg = compose(config_name="base_cluster_metric")

        raw_cfg = OmegaConf.to_container(cfg, resolve=False)
        model_cfg = model_configs.metricVAEConfig.from_cfg(raw_cfg)

        self.assertIs(type(model_cfg.dataconfig), NTXentDataConfig)
        self.assertEqual(type(model_cfg.dataconfig).__module__, "src.core.data.dataset_configs")
        self.assertEqual(model_cfg.dataconfig.target_name, "NTXentDataset")
        self.assertEqual(tuple(model_cfg.lossconfig.input_dim), tuple(model_cfg.ddconfig.input_dim))


if __name__ == "__main__":
    unittest.main()
