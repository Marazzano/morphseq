"""Legacy representation-model (VAE) embeddings — the model world.

The legacy VAE encoder is Python-3.9-pinned. The embeddings stage runs its whole
command body in Python 3.9 while the main pipeline (3.10) only schedules the job
and validates the output file: **only files cross the env boundary, never model
objects** (see ``specs/model_input_handoff_contract.md`` §9).

This pass lays the loading *foundation* only:
- ``model_paths.resolve_legacy_model_dir`` — path-pure resolver for the on-disk
  legacy model layout (``models_root/legacy/<model_name>/``).
- ``load_model_smoke`` — a standalone 3.9 script that loads the model in-process
  and reports its metadata, proving the env + weights + loader work.

The encode loop, the ``compute-embeddings`` task verb, and the Snakemake rules
arrive in a later pass — *rules come when the product exists*.

Do NOT implement ``load_model_subprocess.py``: it belongs to an abandoned
object-transfer design and is the wrong shape under the file-boundary doctrine.
"""
