# Wiring model servers into the Snakemake DAG

Status: design, pre-implementation. The harness, SAM2 adapter, and GroundingDINO
adapter exist and are proven equivalent to their per-well paths; nothing is wired yet.

---

## What wiring must not change

The DAG. One node per well, same inputs, same per-well shard outputs, no batch
wildcard, no batch marker file. Snakemake decides a job succeeded by checking that
its declared `output` exists and the shell command exited 0 — it has no opinion
about which process wrote the file. That is the whole reason a resident server is
invisible to the workflow.

---

## The resource inversion (get this wrong and the pipeline silently deadlocks)

Today `frame_detections_per_well` declares `resources: gpu=1` because that process
loads GroundingDINO itself. Once the model lives in a server, **the client process
holds no GPU memory at all** — it opens a socket, sends two paths, and blocks.

So wiring inverts the declaration:

| Rule | Before | After |
|---|---|---|
| `frame_detections_per_well` (becomes a client) | `gpu=1` | **no gpu resource** |
| `service_grounding_dino` (new) | — | `gpu=1` |

If the client keeps `gpu=1` while the service also declares it, the service takes
the only unit and no client is ever schedulable. Snakemake does not error: it
correctly concludes nothing is runnable and blocks forever. There is no log line
that says why. This is the single most expensive mistake available here, which is
why the warning already sits in the rule files.

`snip_auxiliary_masks` gets the same treatment once its adapter lands.
`frame_masks` (SAM2) should NOT be converted — measured ratio ~0.005, so serving it
buys ~0.5% and adds a moving part. It keeps its plain per-well rule and its `gpu=1`.

---

## Service lifetime — the unsolved part

A service must start before its first client and be torn down after its last. The
spec's §5.2 states this; Snakemake has no native "sidecar process" concept, so it
has to be built. Three candidate mechanisms, none free:

1. **`onstart`/`onsuccess`/`onerror` handlers.** Start servers at workflow start,
   kill at end. Simple, but starts every server on every run regardless of whether
   the DAG contains any consumer — wasteful, and it loads GPU weights for runs that
   never touch them.

2. **A service rule whose output is wrapped in `service(...)`.** Fits the DAG
   naturally: clients declare the socket as an `input`, so ordering is automatic,
   the service starts only when a consumer needs it, and Snakemake tears it down
   after the last consumer finishes.

3. **A wrapper script around the whole invocation** that starts servers, runs
   snakemake, and kills servers in a `trap`. Keeps Snakemake ignorant entirely.
   Crude, but robust, and it composes with the existing SGE submit scripts.

**DECIDED: (2). `service()` IS supported in 7.32.4 — verified, no upgrade needed.**

It is an output *flag function* in `snakemake.io`, not a rule directive, so it does
not appear as `Rule.is_service` or as a parser keyword — checking for those is
misleading. The working form is:

```python
rule service_grounding_dino:
    output: service("...sock")
```

A dry-run confirms Snakemake marks it `output: sock.tmp (service)` and builds the
DAG. Do NOT adopt (1) — the unconditional load cost defeats the purpose. Do not
upgrade to Snakemake 8 for this; 7→8 restructured executor plugins and would put
145 jobs, ~25 rule files, and 22 submit scripts at risk to obtain a feature that
already works.

### Consequence: services are incompatible with `--cores 1`

A service job holds a core for the entire run, so the service plus any consumer
needs **at least 2 cores**. With `--cores 1` the run deadlocks on cores before the
GPU resource is ever consulted:

```
WorkflowError: Not enough resources ... Excess Resources: _cores: 2/1
```

Several existing runtime configs and the Tier-1 smoke path use `--cores 1`. Any
invocation that pulls in a served step must move to `--cores 2` or more. This is a
further argument for leaving `frame_masks` (SAM2) unserved: a `--cores 1` run stays
viable for everything that does not need a resident model.

---

## Client failure modes still unhandled

- **No client-side retry.** A dead server currently gives the client a transport
  error and exit 2. Spec §7 wants retry-with-backoff, failing loudly and
  distinguishably from a data error. Not yet implemented.
- **Client death mid-request.** If Snakemake kills a client (timeout, Ctrl-C), the
  server keeps working and writes the output anyway. Snakemake marks the job failed,
  but the output file now exists — a rerun may skip work it should redo. Not
  addressed by the spec either; worth an explicit decision before production use.
- **Server death mid-run.** Every remaining client fails with a connection error.
  Loud and diagnosable, but there is no supervision or restart.

---

## Verification

Config: `configs/runtime_configs/config_smoke_model_server_keyence_2well.yaml`
(A01/A02 of `20260702_hotchem_24hpf_plate01` — two of the three wells the GDINO
equivalence proof used, so their upstream shards are known good).

**Note that `configs/runtime_configs/` is gitignored**, so that file is on-disk
only. To recreate it, the load-bearing keys are:

```yaml
experiments: [20260702_hotchem_24hpf_plate01]
microscope: "Keyence"
target_wells: {20260702_hotchem_24hpf_plate01: [A01, A02]}
frame_detections: {use_model_server: true}
image_materialization: {smoke_max_time_indices: 1, products: [{channel_id: BF, image_product_type: projection, projection_method: focus_stack}]}
snip_qc:
  exclusion_flags:  # full set from snip_qc/contract.py MINUS the two z_stack-dependent ones
    [viability_dead_flag, persistence_dead_flag, sa_outlier_flag, edge_flag,
     discontinuous_mask_flag, overlapping_mask_flag]
```

The `snip_qc` block is required, not incidental: Keyence z_stack materialization is
unimplemented (PLANNED_REVISIONS §2), so `focus_flag` and `motion_blur_flag` cannot
resolve — both map to `BF__z_stack`. Without dropping them the DAG dies during input
resolution, before any model-server code runs.

### Status

- **DONE — DAG topology.** Dry-run yields `service_grounding_dino 1` +
  `frame_detections_per_well_served 2`. Toggling off yields the original per-well
  rule and no service. Both directions verified.
- **DONE — adapter equivalence in isolation.** Served vs per-well output is
  cell-for-cell identical on 3 real wells (commit 49652e05).
- **BLOCKED — real end-to-end run.** These two wells' shards are write-protected on
  the shared `nlammers` tree from an earlier run, so a rerun hits
  `ProtectedOutputException`. Needs either a well whose shards are writable, a
  scratch `output_root`, or the protection cleared. This is a storage-permissions
  condition, not a defect in the wiring.

Remaining pass criteria for that run:
1. Exactly ONE "adapter loaded in Xs" line in the service log — not one per well.
2. Shards identical to an unserved run of the same wells.
3. No silent hang. A hang means the client rule re-acquired `gpu=1` and is
   deadlocked against the service holding the same unit.
