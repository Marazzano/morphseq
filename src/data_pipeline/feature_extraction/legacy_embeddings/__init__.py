"""Legacy representation-model (VAE) embeddings — the model world.

The legacy VAE encoder is Python-3.9-pinned. The embeddings stage runs its whole
command body in Python 3.9 while the main pipeline (3.10) only schedules the job
and validates the output file: **only files cross the env boundary, never model
objects** (see ``specs/model_input_handoff_contract.md`` §9).

Files:

- ``legacy_vae_inference_loader`` — **the ONLY file allowed to import** ``legacy.vae.*``.
  Exposes ``load_legacy_vae_encoder(model_dir, *, device)`` → ``LegacyVaeEncoder``
  and ``LegacyVaeEncoder.encode_batch(x)`` → ``{"mu": tensor, "logvar": tensor_or_None}``.
- ``model_paths.resolve_legacy_model_dir`` — path-pure resolver for the on-disk
  legacy model layout (``models_root/legacy/<model_name>/``).
- ``snip_source.collect_snip_inputs`` — reads snip_inventory, resolves image paths,
  gates on ``is_valid_snip``.
- ``transforms.snip_to_model_input_tensor`` — PNG → ``[C, H, W]`` float32 tensor.
- ``encode.encode_snips`` — pure batch loop: ``EncoderProtocol`` + SnipInputs → DataFrame.
- ``contract.validate_latent_embeddings`` — latents schema + validator.
- ``load_model_smoke`` — standalone 3.9 script: loads via adapter, prints metadata.

Next: ``entrypoint.py`` — 3.9 CLI: load model once, iterate wells, write shards.

Spec: ``docs/.../specs/features/targets/legacy_embeddings.md``.

Do NOT implement ``load_model_subprocess.py``: it belongs to an abandoned
object-transfer design and is the wrong shape under the file-boundary doctrine.
"""
