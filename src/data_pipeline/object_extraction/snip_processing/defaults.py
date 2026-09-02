"""Checkpoint-compatible defaults for canonical embryo snips.

These values are part of the input contract for the legacy morphology VAE. Keep
all command-line, orchestration, and direct-call fallbacks anchored here so a
missing config value cannot silently change the model's image distribution.
"""

DEFAULT_TARGET_PIXEL_SIZE_UM = 6.5
DEFAULT_BLEND_RADIUS_UM = 75.0

# Synthetic snip background, pegged rather than measured.
#
# WHY PEGGED: these were derived at render time by sampling non-mask pixels from the source frames
# (``_estimate_background``, 50 frames x 5000 pixels, scaled by 0.1). That made the background a
# function of whatever the source images happened to look like, so an upstream change to
# materialization or intensity correction silently moved it. Measured 2026-08-27 on
# 20250612_30hpf_ctrl_atf6: the legacy corpus sits at mean 10.47 / std 3.29 while a fresh render of
# the same plate produced mean 15.56 / std 1.58 -- 61% of the total legacy-vs-regenerated pixel
# difference, from a parameter nobody set.
#
# The pegged values match the legacy corpus. The background is synthetic noise; there is nothing to
# be gained from re-deriving it per run, and a great deal to be lost.
DEFAULT_BACKGROUND_MEAN = 10.0
DEFAULT_BACKGROUND_STD = 3.0
