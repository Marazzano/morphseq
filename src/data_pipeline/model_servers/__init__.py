"""Resident model-server harness (PROTOTYPE).

This package is a prototype for keeping a model resident in a long-lived server
process instead of paying model-load cost once per Snakemake well-job. It is NOT
wired into the Snakemake DAG — every rule today still runs the existing per-well
CLI path (`data_pipeline.pipeline_orchestrator.tasks`) exactly as before.

See `README.md` in this directory for usage and the adapter contract.
"""
