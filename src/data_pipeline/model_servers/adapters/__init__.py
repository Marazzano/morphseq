"""Adapter registry side-effect module.

Importing this package registers every known adapter class into
`data_pipeline.model_servers.adapter_base`'s name->class table, so the harness's
`--adapter <name>` CLI flag can find them. Import order does not matter; each
adapter module registers itself via the `@register_adapter(...)` decorator at
import time.
"""

from data_pipeline.model_servers.adapters import fake  # noqa: F401

try:
    from data_pipeline.model_servers.adapters import sam2  # noqa: F401
except ImportError:
    # The SAM2 adapter imports torch + the sam2 package. Keep the harness importable
    # (and the fake-adapter tests runnable) even in environments without those deps.
    pass

try:
    from data_pipeline.model_servers.adapters import grounding_dino  # noqa: F401
except ImportError:
    # The GroundingDINO adapter imports torch + the groundingdino repo package. Keep the
    # harness importable (and the fake-adapter tests runnable) even without those deps.
    pass

try:
    from data_pipeline.model_servers.adapters import unet_aux_masks  # noqa: F401
except ImportError:
    # The UNet auxiliary-mask adapter imports torch + segmentation_models_pytorch.
    # Keep the harness importable (and the fake-adapter tests runnable) even in
    # environments without those deps.
    pass
