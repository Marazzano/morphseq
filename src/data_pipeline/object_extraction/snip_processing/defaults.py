"""Checkpoint-compatible defaults for canonical embryo snips.

These values are part of the input contract for the legacy morphology VAE. Keep
all command-line, orchestration, and direct-call fallbacks anchored here so a
missing config value cannot silently change the model's image distribution.
"""

DEFAULT_TARGET_PIXEL_SIZE_UM = 6.5
DEFAULT_BLEND_RADIUS_UM = 75.0

