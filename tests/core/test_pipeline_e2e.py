from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
from torch import nn

from src.core.data.dataset_configs import PipelineDataConfig
from src.core.data.manifest_types import (
    CovariatePolicy,
    ManifestPolicy,
    QCPolicy,
    SplitPolicy,
    StagePolicy,
)
from src.core.lightning.callbacks import SaveRunProvenance
from src.core.lightning import pl_wrappers
from src.core.lightning.pl_wrappers import LitModel
from src.core.lightning.train_config import LitTrainConfig
from src.core.losses import loss_functions
from src.core.losses.loss_configs import BasicLoss
from src.core.models.legacy_models import VAE
from src.core.models.model_utils import ModelOutput
from src.core.run.provenance import read_selected_identity
from tests.core.fixtures.pipeline_source_tables import (
    BF_PRODUCT,
    RFP_PRODUCT,
    write_pipeline_source_fixture,
)


INPUT_DIM = (1, 288, 128)
LATENT_DIM = 4


class _DummyLPIPS(nn.Module):
    def forward(self, x, y):
        return torch.zeros((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(INPUT_DIM[0], LATENT_DIM)
        self.log_var = nn.Linear(INPUT_DIM[0], LATENT_DIM)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -10.0)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        pooled = self.pool(x).flatten(1)
        return ModelOutput(
            embedding=self.embedding(pooled),
            log_covariance=self.log_var(pooled),
        )


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_head = nn.Linear(LATENT_DIM, INPUT_DIM[0])

    def forward(self, z: torch.Tensor) -> ModelOutput:
        pixels = torch.sigmoid(self.output_head(z)).view(z.shape[0], INPUT_DIM[0], 1, 1)
        return ModelOutput(reconstruction=pixels.expand(z.shape[0], *INPUT_DIM))


class _ValidationCounter(pl.Callback):
    def __init__(self) -> None:
        self.batch_count = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.batch_count += 1


def _manifest_policy(experiment_id: str) -> ManifestPolicy:
    return ManifestPolicy(
        name="temporary_a3_generated_pipeline_smoke",
        version="1",
        experiment_ids=(experiment_id,),
        allowed_product_keys=(BF_PRODUCT, RFP_PRODUCT),
        selected_product_key=BF_PRODUCT,
        require_valid_snip=True,
        qc=QCPolicy(name="temporary_a3_fixture_qc", version="1"),
        stage=StagePolicy(enabled=False),
        covariates=CovariatePolicy(
            enabled=True,
            required_columns=("incubation_temperature_c", "elapsed_time_s"),
        ),
        splits=SplitPolicy(
            enabled=True,
            train_fraction=0.6,
            eval_fraction=0.2,
            test_fraction=0.2,
            tolerance=0.2,
            salt="morphseq-core-split-v1",
        ),
    )


def _write_pixels(source_paths) -> None:
    import pandas as pd

    inventory = pd.read_csv(source_paths.snip_inventory)
    for raw_path in inventory["processed_snip_path"]:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((73, 39), 96, dtype=np.uint8)).save(path)


def _make_loss(monkeypatch) -> nn.Module:
    monkeypatch.setattr(loss_functions.lpips, "LPIPS", lambda **_: _DummyLPIPS())
    config = BasicLoss(
        max_epochs=5,
        input_dim=INPUT_DIM,
        reconstruction_loss="L2",
        kld_weight=0.0,
        schedule_kld=False,
        pips_flag=False,
        pips_weight=0.0,
        schedule_pips=False,
        use_gan=False,
        gan_weight=0.0,
        schedule_gan=False,
        use_pips_eval=False,
    )
    return config.create_module()


def _make_lit_model(data_config: PipelineDataConfig, loss_fn: nn.Module) -> LitModel:
    train_config = LitTrainConfig(
        benchmark=False,
        max_epochs=5,
        lr_base=1e-3,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision=32,
        encoder_lr_scale=1.0,
    )
    return LitModel(
        model=VAE(
            config=SimpleNamespace(),
            encoder=_TinyEncoder(),
            decoder=_TinyDecoder(),
        ),
        loss_fn=loss_fn,
        data_cfg=data_config,
        train_cfg=train_config,
        batch_key="data",
        eval_gpu_flag=False,
        log_epoch=False,
    )


def _fixed_reconstruction_objective(lit: LitModel, data: torch.Tensor) -> float:
    torch.manual_seed(917)
    with torch.no_grad():
        reconstruction = lit(data).recon_x
        return float(torch.mean((reconstruction - data) ** 2))


def test_generated_pipeline_vanilla_a3_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Exercise the same manifest/dataset/training seam without host pipeline dependencies."""

    torch.manual_seed(20260827)
    np.random.seed(20260827)
    experiment_id = "opaque-a3-experiment"
    source_paths = write_pipeline_source_fixture(
        tmp_path / "pipeline-sources",
        experiment_id,
        row_count=12,
    )
    _write_pixels(source_paths)
    policy = _manifest_policy(experiment_id)
    data_config = PipelineDataConfig(
        pipeline_output_root=tmp_path,
        manifest_policy=policy,
        source_paths={experiment_id: source_paths},
        input_dim=INPUT_DIM,
        batch_size=2,
        num_workers=0,
        loader_seed=41,
        drop_last_train=False,
        pin_memory=False,
        persistent_workers=False,
        max_decode_failures=0,
    )
    manifest = data_config.make_metadata()

    assert len(manifest.observation_table) == 12
    assert len(manifest.asset_table) == 13
    assert len(manifest.resolved_sample_table) == 12
    for row in manifest.resolved_sample_table.itertuples(index=False):
        matches = manifest.asset_table.loc[
            manifest.asset_table["snip_id"].eq(row.snip_id)
            & manifest.asset_table["snip_product_key"].eq(row.snip_product_key)
            & manifest.asset_table["z_index"].isna()
        ]
        assert len(matches) == 1
        assert matches.iloc[0]["processed_snip_path"] == row.processed_snip_path
        assert row.snip_product_key == BF_PRODUCT
        assert Path(row.processed_snip_path).is_file()

    split_counts = manifest.split_assignments["split"].value_counts().to_dict()
    assert set(split_counts) == {"train", "eval", "test"}
    assert (
        manifest.resolved_sample_table.groupby("physical_embryo_id")["split"].nunique()
        == 1
    ).all()

    monkeypatch.setattr(
        pl_wrappers,
        "lpips_score",
        lambda **kwargs: torch.tensor(0.0, device=kwargs["model_output"].recon_x.device),
    )
    monkeypatch.setattr(
        pl_wrappers,
        "ssim_score",
        lambda **kwargs: torch.tensor(0.0, device=kwargs["model_output"].recon_x.device),
    )
    loss_fn = _make_loss(monkeypatch)
    loss_values: list[float] = []
    original_forward = loss_fn.forward

    def _recording_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        loss_values.append(float(output.loss.detach().cpu()))
        return output

    loss_fn.forward = _recording_forward
    lit = _make_lit_model(data_config, loss_fn)
    train_batch = next(iter(lit.train_dataloader()))
    assert train_batch.data.dtype == torch.float32
    assert train_batch.data.shape == (2, *INPUT_DIM)
    assert bool(torch.isfinite(train_batch.data).all())
    assert float(train_batch.data.min()) >= 0.0
    assert float(train_batch.data.max()) <= 1.0
    for field in (
        "incubation_temperature_c",
        "temperature_status",
        "elapsed_time_s",
        "elapsed_time_status",
        "time_index",
        "predicted_stage_hpf",
        "stage_status",
        "stage_model_version",
        "snip_id",
        "physical_embryo_id",
        "snip_product_key",
        "z_index",
        "asset_key",
        "processed_snip_path",
    ):
        assert field in train_batch

    initial_objective = _fixed_reconstruction_objective(lit, train_batch.data)
    before_parameters = {
        name: parameter.detach().clone() for name, parameter in lit.model.named_parameters()
    }
    validation_counter = _ValidationCounter()
    adapter_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()
    provenance_dir = tmp_path / "run-artifacts" / "provenance"
    provenance_callback = SaveRunProvenance(
        data_cfg=data_config,
        resolved_config={
            "run_name": "temporary_a3_generated_pipeline_smoke",
            "input_dim": list(INPUT_DIM),
            "wandb": {"enabled": False},
        },
        run_artifacts_dir=provenance_dir,
        adapter_git_revision=adapter_revision,
        publish_to_wandb=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision=32,
        max_epochs=5,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=3,
        limit_val_batches=2,
        callbacks=[validation_counter, provenance_callback],
        deterministic=True,
    )
    trainer.fit(lit)

    assert validation_counter.batch_count > 0
    assert loss_values and all(np.isfinite(value) for value in loss_values)
    assert any(
        not torch.equal(before_parameters[name], parameter.detach())
        for name, parameter in lit.model.named_parameters()
    )
    final_objective = _fixed_reconstruction_objective(lit, train_batch.data)
    assert final_objective < initial_objective

    checkpoint_path = tmp_path / "checkpoints" / "a3.ckpt"
    checkpoint_path.parent.mkdir()
    trainer.save_checkpoint(checkpoint_path)
    assert checkpoint_path.is_file()

    reloaded_loss = _make_loss(monkeypatch)
    fresh = _make_lit_model(data_config, reloaded_loss)
    reloaded = LitModel.load_from_checkpoint(
        checkpoint_path,
        model=fresh.model,
        loss_fn=reloaded_loss,
        data_cfg=data_config,
        train_cfg=fresh.train_cfg,
        batch_key="data",
        eval_gpu_flag=False,
        log_epoch=False,
    )
    held_out = next(iter(reloaded.test_dataloader()))
    with torch.no_grad():
        encoder_output = reloaded.model.encoder(held_out.data)
        reconstructed = reloaded(held_out.data).recon_x
    assert encoder_output.embedding.shape == (held_out.data.shape[0], LATENT_DIM)
    assert reconstructed.shape == held_out.data.shape
    assert bool(torch.isfinite(encoder_output.embedding).all())
    assert bool(torch.isfinite(reconstructed).all())

    observation_ids, asset_keys = read_selected_identity(provenance_dir)
    assert observation_ids == tuple(manifest.resolved_sample_table["snip_id"])
    assert asset_keys == tuple(
        (row.snip_id, row.snip_product_key, None)
        for row in manifest.resolved_sample_table.itertuples(index=False)
    )
