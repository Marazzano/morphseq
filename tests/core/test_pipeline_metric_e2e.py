from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
from torch import nn

from src.core.data.dataset_configs import PipelineMetricDataConfig
from src.core.data.manifest_types import (
    CovariatePolicy,
    ManifestPolicy,
    MetricMappingPolicy,
    QCPolicy,
    SplitPolicy,
    StagePolicy,
)
from src.core.lightning import pl_wrappers
from src.core.lightning.callbacks import SaveRunProvenance
from src.core.lightning.pl_wrappers import LitModel
from src.core.lightning.train_config import LitTrainConfig
from src.core.losses import loss_functions
from src.core.losses.loss_configs import MetricLoss
from src.core.metric import (
    DifferentEmbryoCandidatePolicy,
    MetricMappingArtifact,
    MetricMappingEntry,
    MetricPairingPolicy,
    SameEmbryoCandidatePolicy,
    build_test_only_policy,
)
from src.core.models.legacy_models import metricVAE
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


def _mapping() -> MetricMappingArtifact:
    return MetricMappingArtifact(
        name="test_only_fixture_genotype_to_single_metric_group",
        version="1",
        metadata_columns=("genotype",),
        entries=(
            MetricMappingEntry(("source/genotype/0",), "test_only_single_group"),
            MetricMappingEntry(("source/genotype/1",), "test_only_single_group"),
        ),
        scientific_policy=False,
    )


def _manifest_policy(experiment_id: str, mapping: MetricMappingArtifact) -> ManifestPolicy:
    return ManifestPolicy(
        name="test_only_a4_generated_metric_smoke",
        version="1",
        experiment_ids=(experiment_id,),
        allowed_product_keys=(BF_PRODUCT, RFP_PRODUCT),
        selected_product_key=BF_PRODUCT,
        require_valid_snip=True,
        qc=QCPolicy(name="test_only_a4_fixture_qc", version="1"),
        stage=StagePolicy(
            enabled=True,
            required=True,
            accepted_statuses=("predicted",),
        ),
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
        metric_mapping=MetricMappingPolicy(
            enabled=True,
            name=mapping.name,
            scientific_policy=False,
        ),
    )


def _pairing_policy() -> MetricPairingPolicy:
    return MetricPairingPolicy(
        name="test_only_same_observation_metric_pairs",
        version="1",
        stage_column="predicted_stage_hpf",
        stage_source="fixture stage_predictions predicted_stage_hpf",
        sampler_age_window=0.0,
        same_embryo=SameEmbryoCandidatePolicy(
            enabled=True,
            allow_same_observation=True,
        ),
        different_embryo=DifferentEmbryoCandidatePolicy(enabled=False),
        same_embryo_probability=1.0,
        base_seed=20260831,
        scientific_policy=False,
    )


def _write_pixels(source_paths) -> None:
    import pandas as pd

    inventory = pd.read_csv(source_paths.snip_inventory)
    for row_number, raw_path in enumerate(inventory["processed_snip_path"]):
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((59, 37), 64 + row_number, dtype=np.uint8)).save(path)


def _loss_config(relation_policy) -> MetricLoss:
    return MetricLoss(
        max_epochs=2,
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        frac_nuisance_latents=0.25,
        relation_policy=relation_policy,
        metric_group_names=("test_only_single_group",),
        sampler_age_window=0.0,
        loss_age_window=0.0,
        self_target_prob=1.0,
        temperature=0.2,
        metric_weight=0.5,
        schedule_metric=False,
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


def _lit_model(data_config, loss_config) -> LitModel:
    train_config = LitTrainConfig(
        benchmark=False,
        max_epochs=2,
        lr_base=1e-3,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        precision=32,
        encoder_lr_scale=1.0,
    )
    model = metricVAE(
        config=SimpleNamespace(lossconfig=loss_config),
        encoder=_TinyEncoder(),
        decoder=_TinyDecoder(),
    )
    return LitModel(
        model=model,
        loss_fn=loss_config.create_module(),
        data_cfg=data_config,
        train_cfg=train_config,
        batch_key="data",
        eval_gpu_flag=False,
        log_epoch=False,
    )


def test_generated_pipeline_a4_test_only_metric_end_to_end(tmp_path, monkeypatch) -> None:
    """Exercise C1+C2+C3 integration without asserting a scientific policy."""

    torch.manual_seed(20260831)
    np.random.seed(20260831)
    monkeypatch.setattr(loss_functions.lpips, "LPIPS", lambda **_: _DummyLPIPS())
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

    experiment_id = "opaque-test-only-a4-experiment"
    source_paths = write_pipeline_source_fixture(
        tmp_path / "pipeline-sources", experiment_id, row_count=12
    )
    _write_pixels(source_paths)
    mapping = _mapping()
    relation_policy = build_test_only_policy(("test_only_single_group",))
    pairing_policy = _pairing_policy()
    data_config = PipelineMetricDataConfig(
        pipeline_output_root=tmp_path,
        manifest_policy=_manifest_policy(experiment_id, mapping),
        source_paths={experiment_id: source_paths},
        metric_mapping_artifact=mapping,
        relation_policy=relation_policy,
        pairing_policy=pairing_policy,
        input_dim=INPUT_DIM,
        batch_size=2,
        num_workers=0,
        loader_seed=73,
        max_decode_failures=0,
    )
    manifest = data_config.make_metadata()
    assert len(manifest.resolved_sample_table) == 12
    assert set(manifest.resolved_sample_table["metric_group"]) == {
        "test_only_single_group"
    }
    assert len(data_config.pair_preflight_reports) == 3

    loss_config = _loss_config(relation_policy)
    lit = _lit_model(data_config, loss_config)
    train_batch = next(iter(lit.train_dataloader()))
    assert train_batch.data.shape == (2, 2, *INPUT_DIM)
    assert len(train_batch.self_stats) == 3
    assert len(train_batch.other_stats) == 3
    assert list(train_batch.self_stats[2]) == [
        "test_only_single_group",
        "test_only_single_group",
    ]
    assert train_batch.pair_candidate_kind == ["same_embryo", "same_embryo"]

    before_parameters = {
        name: parameter.detach().clone()
        for name, parameter in lit.model.named_parameters()
    }
    loss_values: list[float] = []
    original_forward = lit.loss_fn.forward

    def recording_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        loss_values.append(float(output.loss.detach().cpu()))
        return output

    lit.loss_fn.forward = recording_forward
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
            "run_name": "test_only_a4_generated_metric_smoke",
            "scientific_policy": False,
            "sampler_age_window": pairing_policy.sampler_age_window,
            "loss_age_window": loss_config.loss_age_window,
            "contrastive_temperature": loss_config.temperature,
            "metric_weight": loss_config.metric_weight,
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
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=2,
        limit_val_batches=1,
        callbacks=[validation_counter, provenance_callback],
        deterministic=True,
    )
    trainer.fit(lit)

    assert trainer.global_step > 0
    assert validation_counter.batch_count > 0
    assert loss_values and all(np.isfinite(value) for value in loss_values)
    assert any(
        not torch.equal(before_parameters[name], parameter.detach())
        for name, parameter in lit.model.named_parameters()
    )

    checkpoint_path = tmp_path / "checkpoints" / "a4-test-only.ckpt"
    checkpoint_path.parent.mkdir()
    trainer.save_checkpoint(checkpoint_path)
    fresh = _lit_model(data_config, _loss_config(relation_policy))
    reloaded = LitModel.load_from_checkpoint(
        checkpoint_path,
        model=fresh.model,
        loss_fn=fresh.loss_fn,
        data_cfg=data_config,
        train_cfg=fresh.train_cfg,
        batch_key="data",
        eval_gpu_flag=False,
        log_epoch=False,
    )
    held_out = next(iter(reloaded.test_dataloader()))
    with torch.no_grad():
        output = reloaded(held_out.data)
    assert output.mu.shape == (held_out.data.shape[0] * 2, LATENT_DIM)
    assert output.recon_x.shape == held_out.data[:, 0].shape
    assert bool(torch.isfinite(output.mu).all())
    assert bool(torch.isfinite(output.recon_x).all())

    observation_ids, asset_keys = read_selected_identity(provenance_dir)
    assert observation_ids == tuple(manifest.resolved_sample_table["snip_id"])
    assert len(asset_keys) == len(observation_ids)
    metric_provenance = json.loads(
        (provenance_dir / "mapping_policy.json").read_text(encoding="utf-8")
    )
    assert metric_provenance["mapping"]["scientific_policy"] is False
    assert metric_provenance["relation_policy"]["scope"] == "test_only"
    assert "test_only" in metric_provenance["pairing_policy"]["name"]
    assert metric_provenance["pairing_policy"]["sampler_age_window"] == 0.0
