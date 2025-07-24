import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Sequence, Dict, Any
from src.flux.flux_model import MLPVelocityField
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F


class ClockNVF(pl.LightningModule):
    """PyTorch‑Lightning module that trains a neural‑velocity field with
    hierarchical (experiment → embryo) clock‑scaling.

    Velocity model
        v_pred = γ_{e,k} * f_θ(z)
        γ_{e,k} = exp(log_s_e + δ_{e,k})
    where
        log_s_e  : learnable per‑experiment log‑scale
        δ_{e,k}  : learnable embryo‑specific deviation with Gaussian shrinkage.
    """

    def __init__(
        self,
        dim: int,
        num_exp: int,
        num_embryo: int,
        dataset: Any,
        train_indices: Sequence[int] = None,  # indices for training set
        val_indices: Sequence[int] = None,    # indices for validation set
        batch_size: int = 8192,
        num_workers: int = 2,
        lambda_emb: float = 1000,  # regularization for embryo-specific deviations
        lambda_curv: float = 0.1,  # regularization for curvature of the velocity field
        lambda_ang: float = 1,  # regularization for angular loss
        infer_embryo_clock: bool = False,
        hidden: Sequence[int] = (256, 128, 128),
        sigma: float = 0.2,  # shrinkage std‑dev for δ
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        # shared geometry network f_θ
        self.field = MLPVelocityField(dim, hidden)

        # hierarchical clock parameters
        self.log_s = nn.Parameter(torch.zeros(num_exp))   # experiment speed (log‑space)
        self.delta = nn.Parameter(torch.zeros(num_embryo))  # embryo deviation

        # dataset info
        self.ds = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers  # number of workers for DataLoader
        self.train_indices = train_indices  # indices for training set
        self.val_indices = val_indices      # indices for validation set

        # regularization hyper‑params
        self.sigma = sigma
        self.lambda_emb = lambda_emb  # regularization for embryo-specific deviations
        self.lambda_curv = lambda_curv  # regularization for curvature of the velocity field
        self.lambda_ang = lambda_ang  # regularization for angular loss

        # hyper‑params
        self.lr = lr
        self.weight_decay = weight_decay
        self.infer_embryo_clock = infer_embryo_clock

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, z: torch.Tensor, exp_idx: torch.Tensor, emb_idx: torch.Tensor) -> torch.Tensor:
        """Predict velocity for latent positions *z*.

        Args:
            z:        (B, d) latent embeddings
            exp_idx:  (B,) experiment indices (int)
            emb_idx:  (B,) embryo indices (int)
        Returns:
            (B, d) velocity predictions
        """
        if self.infer_embryo_clock:
            gamma = torch.exp(self.log_s[exp_idx] + self.delta[emb_idx])  # (B,)
        else:
            gamma = torch.exp(self.log_s[exp_idx])
            
        return self.field(z) * gamma.unsqueeze(-1)

    # --------------------------------------------------------
    # Loss helpers
    # --------------------------------------------------------
    def _loss(self, batch: Dict[str, torch.Tensor], scale_factor=1e4):
        z0 = batch["z0"]    # (B, d)
        dz_target = batch["dz"]            # (B, d)
        exp_idx = batch["exp"]      # (B,)
        emb_idx = batch["emb"]      # (B,)
        
        # finite‑difference velocity
        dz_pred = self.forward(z0, exp_idx, emb_idx)

        # compute loss

        # MSE
        loss_mse = scale_factor * torch.mean((dz_pred - dz_target) ** 2) # MSE

        # cosine similarity
        # Angular loss
        cos_sim  = F.cosine_similarity(dz_pred, dz_target, dim=-1)
        loss_ang = torch.mean(1 - cos_sim)

        # embryo term regularization
        loss_emb = scale_factor * torch.mean(self.delta ** 2) / (self.sigma ** 2) # penalty term for large embryo-specific deviations

        # compute curvature loss
        with torch.enable_grad():
            curve_loss = scale_factor * self._curvature_loss(batch)

        # get total
        loss = loss_mse + self.lambda_ang*loss_ang +  self.lambda_emb * loss_emb + self.lambda_curv * curve_loss

        # calculate other metrics
        logs = {
            "loss_mse": loss_mse,
            "loss_emb": loss_emb,
            "loss_ang": loss_ang,
            "curve_loss": curve_loss,
            "total": loss,
        }
        logs = self._calculate_metrics(dz_target=dz_target, dz_pred=dz_pred, logs=logs)

        return loss, logs
    

    def _curvature_loss(self, batch: Dict[str, torch.Tensor]):
        """Compute Jacobian-norm penalty via Hutchinson’s estimator."""

        # 1️ make z0 a grad-tracking leaf
        z0 = batch["z0"].detach().clone().requires_grad_(True)  # (B, d)
        exp_idx = batch["exp"]
        emb_idx = batch["emb"]

        # 2️ forward on that same z0
        dz_pred = self.forward(z0, exp_idx, emb_idx)            # (B, d)

        # 3️ Hutchinson trace estimator
        v = torch.randn_like(z0)                                # probe
        inner = (dz_pred * v).sum()                             # scalar
        Jv = torch.autograd.grad(
            outputs=inner,
            inputs=z0,
            create_graph=True,
        )[0]                                                     # (B, d)

        # 4️ mean squared norm = Frobenius norm^2 estimate
        return (Jv ** 2).mean()
    
    def _log_loss(self, logs: Dict[str, torch.Tensor], phase: str):
        """Log loss values for a given phase (train/val)."""
        for k, v in logs.items():
            self.log(f"{phase}/{k}", v, prog_bar=True, on_epoch=True, on_step=False)

        # log total loss
        self.log(f"{phase}/total", logs["total"], prog_bar=True, on_epoch=True, on_step=False)

    def _calculate_metrics(self, 
                           dz_target: torch.Tensor,
                           dz_pred: torch.Tensor,
                           logs: Dict[str, torch.Tensor]):
        
        """Calculate and log metrics for a given phase (train/val)."""
        # Log the metrics for the current phase

        # cosine (directional similarity)
        cos_sim = torch.nn.functional.cosine_similarity(dz_pred, dz_target, dim=-1)
        avg_cos = cos_sim.mean()
        logs["cosine_similarity"] = avg_cos

        # magnitude (Magnitude of the velocity field)
        mag_error = ((dz_pred.norm(dim=-1) - dz_target.norm(dim=-1)) ** 2).mean()
        logs["magnitude_error"] = mag_error

        # R2
        target_var = dz_target.var(dim=0).sum()
        resid_var = ((dz_pred - dz_target) ** 2).mean(dim=0).sum()
        r2 = 1 - (resid_var / target_var)
        logs["r2"] = r2

        return logs
        
    # --------------------------------------------------------
    # Lightning hooks
    # --------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, logs = self._loss(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, prog_bar=True)

        self._log_loss(logs, "train")

        return loss


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, logs = self._loss(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, prog_bar=False)

        self._log_loss(logs, "val")

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    
    def train_dataloader(self):
        ds = self.ds
        # get indices for images to use for training
        if self.train_indices is None:
            train_sampler = None
        else:
            train_sampler = SubsetRandomSampler(self.train_indices)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True
        )

    def val_dataloader(self):
        ds = self.ds
        # get indices for images to use for validation
        if self.val_indices is None:
            val_sampler = None
        else:
            val_sampler = SubsetRandomSampler(self.val_indices)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False
        )