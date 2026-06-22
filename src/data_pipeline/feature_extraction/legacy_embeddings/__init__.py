"""Legacy representation-model (VAE) embeddings — the model world.

The legacy VAE encoder is Python-3.9-pinned. The embeddings stage runs its whole
command body in Python 3.9 while the main pipeline (3.10) only schedules the job
and validates the output file: **only files cross the env boundary, never model
objects** (see ``specs/model_input_handoff_contract.md`` §9).

Foundation files (already present):
- ``model_paths.resolve_legacy_model_dir`` — path-pure resolver for the on-disk
  legacy model layout (``models_root/legacy/<model_name>/``).
- ``snip_source.collect_snip_inputs`` — reads snip_inventory, resolves image paths,
  gates on ``is_valid_snip``.
- ``load_model_smoke`` — standalone 3.9 script: load the model in-process, print
  metadata, exit. Proves the env + weights + loader work.

Next pass (encode loop) adds:
- ``contract.py`` — latents schema + ``validate_latent_embeddings()``.
- ``transforms.py`` — ``snip_to_model_input_tensor(model_input_shape)``.
- ``encode.py`` — pure encode loop (loaded encoder + SnipInputs → DataFrame).
- ``entrypoint.py`` — 3.9 CLI: load model once, iterate wells, write shards.

Spec: ``docs/.../specs/features/targets/legacy_embeddings.md``.
*Rules come when the product exists.*

Do NOT implement ``load_model_subprocess.py``: it belongs to an abandoned
object-transfer design and is the wrong shape under the file-boundary doctrine.
"""
