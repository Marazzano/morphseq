"""Potential network: small MLP mapping R^d -> R (scalar potential field).

The baseline developmental potential phi_0(z) is the foundation of all model
variants. Gradients grad_phi_0 are computed via torch autograd and drive the
drift in the SDE.

Model spec references: §5.1 (potential networks), §5.3 (initialization).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PotentialNetwork(nn.Module):
    """Small MLP mapping R^d -> R (a scalar potential field).

    Gradients grad_phi(z) are computed via torch.autograd.grad with
    create_graph=True so that second-order gradients flow back into network
    weights during training.

    Args:
        input_dim: Dimension of latent space (d).
        hidden_dim: Width of hidden layers.
        n_hidden: Number of hidden layers.
        activation: Smooth nonlinearity — "softplus" or "elu".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        if activation == "softplus":
            act_fn = nn.Softplus
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unsupported activation: {activation!r}. Use 'softplus' or 'elu'.")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """Evaluate phi(z).

        Args:
            z: (*, d) input points.

        Returns:
            (*,) scalar potential values.
        """
        return self.net(z).squeeze(-1)

    def gradient(self, z: Tensor) -> Tensor:
        """Compute grad_z phi(z) via torch.autograd.grad.

        Handles requires_grad internally. Uses create_graph=True so that
        gradients through this operation propagate into network weights
        during backprop. Works correctly even inside a torch.no_grad()
        context (e.g. in predict()).

        Args:
            z: (*, d) input points.

        Returns:
            (*, d) gradient vectors. Has grad_fn when the network has
            trainable parameters (i.e. supports second-order gradients).
        """
        with torch.enable_grad():
            z_input = z.detach().requires_grad_(True)
            phi = self.forward(z_input)
            (grad,) = torch.autograd.grad(
                outputs=phi,
                inputs=z_input,
                grad_outputs=torch.ones_like(phi),
                create_graph=True,
                retain_graph=True,
            )
        return grad
