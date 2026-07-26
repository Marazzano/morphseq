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

## Verification plan

Two wells of a Keyence experiment (lighter compute than YX1), one timepoint, from
raw through `analysis_ready`. Base config:
`configs/runtime_configs/config_smoke_centralize_cutover_keyence_hotchem_A01.yaml`
(already 1 well, `smoke_max_time_indices: 1`) extended to a second well.

Pass criteria:
1. The DAG builds and runs to `analysis_ready` with servers wired.
2. Per-well outputs are identical to a non-served run of the same two wells.
3. The service starts once, not once per well (check the log for a single load line).
4. No deadlock under `--cores > 1 --resources gpu=1`.

Criterion 2 is the one that matters; the adapters were proven equivalent in
isolation, but not yet through the real rule path.
