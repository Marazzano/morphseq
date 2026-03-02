"""
Legacy VAE package.

This repository vendors parts of the `pythae` API under `src.legacy.vae`. Some
callers import a small set of symbols directly from the package root (e.g.
`from src.legacy.vae import AutoModel`), so we re-export the relevant classes
here for compatibility.
"""

from .models.auto_model import AutoModel  # noqa: F401
from .models.base.base_utils import ModelOutput  # noqa: F401
from .models.nn import BaseDecoder, BaseEncoder  # noqa: F401

__all__ = [
    "AutoModel",
    "ModelOutput",
    "BaseEncoder",
    "BaseDecoder",
]
