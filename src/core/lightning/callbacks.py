import os
from pytorch_lightning import Callback
import pickle
from pathlib import Path

from src.core.run.provenance import RunProvenanceBundle, write_run_provenance


class SaveRunMetadata(Callback):
    def __init__(self, data_cfg):
        # build your payload once
        self.index_dict = {
                "train": data_cfg.train_indices,
                "eval":  data_cfg.eval_indices,
                "test":  data_cfg.test_indices,
            }
        # if hasattr(data_cfg, "metric_array"):
        #     self.metric_array = data_cfg.metric_array
        self._written = False

    def on_train_start(self, trainer, pl_module):
        if self._written:
            return
        run_dir = trainer.logger.save_dir        # tb_logs/run_name/version_x
        index_dir  = os.path.join(run_dir, "split_indices.pkl")
        with open(index_dir, "wb") as file:
            pickle.dump(self.index_dict, file)

        # if hasattr(self, "metric_array"):
        #     metric_dir = os.path.join(run_dir, "metric_array.npy")
        #     self.metric_array = self.metric_array

        self._written = True


class SaveRunProvenance(Callback):
    """Persist the manifest-backed run identity independently of online logging."""

    def __init__(
        self,
        *,
        data_cfg,
        resolved_config,
        run_artifacts_dir,
        adapter_git_revision,
        publish_to_wandb=False,
    ):
        manifest_result = getattr(data_cfg, "manifest_result", None)
        if manifest_result is None:
            raise ValueError(
                "SaveRunProvenance requires a built manifest_result; "
                "call PipelineDataConfig.make_metadata() first."
            )
        self.bundle = RunProvenanceBundle.from_manifest_result(
            manifest_result=manifest_result,
            resolved_config=resolved_config,
            adapter_git_revision=adapter_git_revision,
        )
        self.run_artifacts_dir = Path(run_artifacts_dir)
        self.publish_to_wandb = bool(publish_to_wandb)
        self._written = False

    def on_train_start(self, trainer, pl_module):
        if self._written:
            return

        wandb_run = None
        if self.publish_to_wandb:
            for logger in trainer.loggers:
                if logger.__class__.__name__ == "WandbLogger":
                    wandb_run = logger.experiment
                    break
            if wandb_run is None:
                raise RuntimeError(
                    "Manifest provenance was configured for W&B publication, "
                    "but the Trainer has no WandbLogger."
                )

        write_run_provenance(
            run_artifacts_dir=self.run_artifacts_dir,
            bundle=self.bundle,
            wandb_run=wandb_run,
        )
        self._written = True

class EpochListCheckpoint(Callback):
    def __init__(self, epochs, dirpath="checkpoints"):
        super().__init__()
        self.epochs = set(int(e) for e in epochs)
        self.dir = Path(dirpath)
        self.dir.mkdir(exist_ok=True, parents=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # +1 because current_epoch is 0-based
        epoch = trainer.current_epoch + 1
        if epoch in self.epochs:
            ckpt_path = self.dir / f"epoch{epoch:04d}.ckpt"
            trainer.save_checkpoint(ckpt_path)
