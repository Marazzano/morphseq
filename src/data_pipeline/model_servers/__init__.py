"""Resident model-server harness — keep a model loaded ONCE across a run's well jobs.

Snakemake gives every well job its own process, so a per-well rule reloads its model N times.
Where load dominates real work that is the pipeline's largest avoidable cost: the 4x-UNet A/B
measured 69.94s of load against 2.3-12.6s of work, i.e. 280.39s per-well vs 97.47s served over
3 wells (2.88x), with all 2,100 mask PNGs byte-identical. Across 576 wells ~11h of repeated
loading collapses to ~70s.

A service rule loads the model once and holds it; per-well rules become thin socket clients. The
DAG shape is unchanged -- still one job per well, same shards -- because Snakemake only checks
that `output` exists and the command exited 0, not which process wrote it.

WIRED (opt-in, default off), each behind a config toggle:
    frame_detections.use_model_server   -> service_grounding_dino
    unet_snip.use_model_server          -> service_unet_aux_masks

NOT SERVED, deliberately: `latent_embeddings` (legacy VAE). It is the third step that reloads a
model per well and it still carries EXECUTION_RUN_BATCH in the registry, but it runs on CPU, so
there is no GPU init to amortize and no card being held idle -- the payoff is a fraction of the
GPU cases above and does not justify a resident process. Its registry row is the stale one to fix,
not its execution.

Socket paths come from `socket_paths.service_socket_pattern` -- read that module before touching
one; a service/client path disagreement is a silent DAG failure and has bitten this code once.

See `README.md` for the adapter contract, and docs/MODEL_SERVER_WIRING.md for the rule pattern.
"""
