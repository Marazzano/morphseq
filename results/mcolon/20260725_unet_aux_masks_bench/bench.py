"""Equivalence + throughput harness for the snip_auxiliary_masks (UNet) model-server prototype.

Runs N real wells (experiment 20260319, Keyence) two ways:
  1. PER-WELL: fresh `python -m data_pipeline.pipeline_orchestrator.tasks snip-auxiliary-masks`
     subprocess per well (mirrors today's Snakemake rule exactly, including process-level
     4x-UNet reload every time).
  2. SERVED: one ModelServer(SnipAuxiliaryMasksAdapter) process (loads the 4 UNets ONCE),
     N client calls -- one per well.

Compares outputs for row/content equivalence and reports wall-clock for both, run on CPU
(no GPU available on this node -- see report for why that still proves correctness, and
what it can/can't say about GPU throughput).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)

sys.path.insert(
    0,
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-afed7356581c76751/src",
)

import pandas as pd

DATA = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/pipeline")
MODELS_ROOT = DATA / "models"
OUTPUT_ROOT = DATA / "output"
CONFIG_YAML = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-afed7356581c76751"
    "/src/data_pipeline/pipeline_orchestrator/config.yaml"
)
DEVICE = "cuda"  # submitted to an SGE GPU node; see report for the CPU-node fallback path

WELLS = ["20260319_A01", "20260319_B01", "20260319_C01"]

SCRATCH = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-afed7356581c76751"
    "/results/mcolon/20260725_unet_aux_masks_bench/scratch"
)
PERWELL_OUT = SCRATCH / "perwell_out"
SERVED_OUT = SCRATCH / "served_out"
# AF_UNIX socket paths are limited to ~108 bytes on Linux; the deeply nested worktree
# path under SCRATCH exceeds that. /tmp is node-local anyway (appropriate for a Unix
# socket, which never needs to be visible off-node) and short.
import os as _os

SOCKET_PATH = Path(f"/tmp/unet_bench_{_os.getpid()}.sock")

PYTHON = "/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python"

UNET_SNIP_CONFIG = {
    "models_root": str(MODELS_ROOT),
    "device": DEVICE,
    "models": {
        "via": {"checkpoint": "segmentation/via_v1_0100", "model_input_shape": [576, 256]},
        "yolk": {"checkpoint": "segmentation/yolk_v1_0050", "model_input_shape": [576, 256]},
        "focus": {"checkpoint": "segmentation/focus_v0_0100", "model_input_shape": [576, 256]},
        "bubble": {"checkpoint": "segmentation/bubble_v0_0100", "model_input_shape": [576, 256]},
    },
}


def real_snip_inventory_csv(well_id: str) -> Path:
    experiment = well_id.split("_")[0]
    return (
        OUTPUT_ROOT / "object_extraction" / experiment / "snips" / "per_well" / well_id
        / f"{well_id}_snip_inventory.csv"
    )


def stage_well_inputs(well_id: str, staging_root: Path) -> Path:
    """Mirror one well's snip_inventory + every referenced snip PNG into an isolated
    local root, preserving processed_snip_path's relative layout exactly.

    `processed_snip_path` in the real snip_inventory is stored RELATIVE TO the real
    pipeline output_root (verified: e.g.
    "object_extraction/20260319/snips/per_well/20260319_A01/snips/.../t0000.png").
    `resolve_snip_inventory_image_paths` resolves it by joining onto whatever
    `output_root` is passed at call time (`resolve_from_root`), and
    `run_snip_auxiliary_masks`/the adapter's handle() both derive the MASK WRITE
    location from that same `output_root` (`masks_dir = output_root / "object_extraction"`).
    So per-well and served runs can't safely share the real output_root as their
    write target (they would collide/overwrite each other's masks and make the two
    runs impossible to diff) -- staging a local copy first gives each run (per-well,
    served) an independent, diffable output_root while keeping the on-disk relative
    path contract identical to production.
    """
    staging_root.mkdir(parents=True, exist_ok=True)
    src_csv = real_snip_inventory_csv(well_id)
    df = pd.read_csv(src_csv)

    for _, row in df.iterrows():
        rel = row["processed_snip_path"]
        if pd.isna(rel):
            continue
        src_path = OUTPUT_ROOT / rel
        dst_path = staging_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists():
            shutil.copyfile(src_path, dst_path)

    dst_csv_dir = staging_root / "object_extraction" / well_id.split("_")[0] / "snips" / "per_well" / well_id
    dst_csv_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_csv_dir / f"{well_id}_snip_inventory.csv"
    df.to_csv(dst_csv, index=False)
    return dst_csv


def run_per_well(well_id: str, staged_csv: Path, out_root: Path) -> float:
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{well_id}_snip_auxiliary_masks.csv"
    cmd = [
        PYTHON, "-m", "data_pipeline.pipeline_orchestrator.tasks", "snip-auxiliary-masks",
        "--snip-inventory-csv", str(staged_csv),
        "--output-root", str(out_root),
        "--output-csv", str(out_csv),
        "--config-yaml", str(CONFIG_YAML),
        "--models-root", str(MODELS_ROOT),
    ]
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.monotonic() - t0
    if result.returncode != 0:
        print(f"PER-WELL FAILED for {well_id}:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        raise RuntimeError(f"per-well subprocess failed for {well_id}")
    return dt


def start_server():
    from data_pipeline.model_servers.harness import ModelServer
    from data_pipeline.model_servers.adapters.unet_aux_masks import SnipAuxiliaryMasksAdapter

    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    adapter = SnipAuxiliaryMasksAdapter(unet_snip_config_json=json.dumps(UNET_SNIP_CONFIG))
    server = ModelServer(socket_path=SOCKET_PATH, adapter=adapter)

    thread_error: list[BaseException] = []

    def _run():
        try:
            server.serve_forever()
        except BaseException as e:  # noqa: BLE001 - surface any load()/bind() failure to the main thread
            thread_error.append(e)
            raise

    thread = threading.Thread(target=_run, daemon=True)
    t0 = time.monotonic()
    thread.start()
    deadline = time.monotonic() + 600.0
    while not SOCKET_PATH.exists():
        if thread_error:
            raise RuntimeError(f"server thread died during load(): {thread_error[0]!r}") from thread_error[0]
        if not thread.is_alive():
            raise RuntimeError("server thread exited before creating the socket file (see traceback above)")
        if time.monotonic() > deadline:
            raise TimeoutError("server did not become ready in time")
        time.sleep(0.05)
    load_time = time.monotonic() - t0
    return server, thread, load_time


def run_served(well_id: str, staged_csv: Path, out_root: Path) -> float:
    from data_pipeline.model_servers.client import call_server

    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{well_id}_snip_auxiliary_masks.csv"
    payload = {
        "snip_inventory_csv": str(staged_csv),
        "output_root": str(out_root),
        "output_csv": str(out_csv),
    }
    t0 = time.monotonic()
    response = call_server(SOCKET_PATH, payload, timeout=600.0)
    dt = time.monotonic() - t0
    if not response.ok:
        raise RuntimeError(f"SERVED failed for {well_id}: {response.error}")
    return dt


def compare_outputs(well_id: str) -> dict:
    perwell_csv = PERWELL_OUT / well_id / f"{well_id}_snip_auxiliary_masks.csv"
    served_csv = SERVED_OUT / well_id / f"{well_id}_snip_auxiliary_masks.csv"

    df_pw = pd.read_csv(perwell_csv)
    df_sv = pd.read_csv(served_csv)

    sort_cols = ["snip_id", "auxiliary_mask_type"]
    df_pw = df_pw.sort_values(sort_cols).reset_index(drop=True)
    df_sv = df_sv.sort_values(sort_cols).reset_index(drop=True)

    result = {
        "well_id": well_id,
        "rows_perwell": len(df_pw),
        "rows_served": len(df_sv),
        "row_count_match": len(df_pw) == len(df_sv),
    }

    # Compare all columns except checkpoint_path/auxiliary_mask_path, which legitimately differ
    # because the two runs write to different output_root scratch dirs (PERWELL_OUT vs SERVED_OUT).
    compare_cols = [c for c in df_pw.columns if c not in ("auxiliary_mask_path",)]
    non_path_equal = df_pw[compare_cols].equals(df_sv[compare_cols])
    result["non_path_columns_equal"] = non_path_equal
    if not non_path_equal:
        for c in compare_cols:
            if not df_pw[c].equals(df_sv[c]):
                result.setdefault("mismatched_columns", []).append(c)

    result["all_valid_perwell"] = bool(df_pw["is_valid_auxiliary_mask"].all())
    result["all_valid_served"] = bool(df_sv["is_valid_auxiliary_mask"].all())

    # Byte-compare the actual mask PNGs (not just the manifest rows).
    mismatched_pixels = []
    n_compared = 0
    for _, (row_pw, row_sv) in enumerate(zip(df_pw.itertuples(), df_sv.itertuples())):
        p_pw = Path(str(row_pw.auxiliary_mask_path))
        p_sv = Path(str(row_sv.auxiliary_mask_path))
        if not (p_pw.exists() and p_sv.exists()):
            continue
        n_compared += 1
        if p_pw.read_bytes() != p_sv.read_bytes():
            mismatched_pixels.append((row_pw.snip_id, row_pw.auxiliary_mask_type))
    result["masks_compared"] = n_compared
    result["masks_byte_identical"] = len(mismatched_pixels) == 0
    if mismatched_pixels:
        result["mismatched_masks"] = mismatched_pixels[:10]

    return result


def main():
    for d in (PERWELL_OUT, SERVED_OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    print("=" * 70)
    print("STAGING: mirroring real snip_inventory + PNGs into isolated per-well roots")
    print("=" * 70)
    staged = {}
    for well_id in WELLS:
        staging_root = SCRATCH / "staged_inputs" / well_id
        staged_csv = stage_well_inputs(well_id, staging_root)
        staged[well_id] = (staged_csv, staging_root)
        n_rows = len(pd.read_csv(staged_csv))
        print(f"  {well_id}: staged {n_rows} snip_inventory rows -> {staging_root}")

    print()
    print("=" * 70)
    print("PER-WELL (existing path): fresh subprocess per well, reloads 4 UNets each time")
    print("=" * 70)
    perwell_times = {}
    t_perwell_total_start = time.monotonic()
    for well_id in WELLS:
        staged_csv, staging_root = staged[well_id]
        # per-well run reads staged inputs (isolated copy of real data) but must write
        # into the SAME staging_root (its own processed_snip_path is relative to it),
        # so mirror the staged CSV once more into a dedicated write-root for THIS run.
        run_root = PERWELL_OUT / well_id
        run_root.mkdir(parents=True, exist_ok=True)
        # Copy staged snip images into this run's own root too (cheap symlinks) so
        # PERWELL_OUT and SERVED_OUT never share a write target.
        for rel_dir in ("object_extraction",):
            src = staging_root / rel_dir
            dst = run_root / rel_dir
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
        run_csv = run_root / "object_extraction" / well_id.split("_")[0] / "snips" / "per_well" / well_id / f"{well_id}_snip_inventory.csv"
        dt = run_per_well(well_id, run_csv, run_root)
        perwell_times[well_id] = dt
        print(f"  {well_id}: {dt:.2f}s")
    t_perwell_total = time.monotonic() - t_perwell_total_start

    print()
    print("=" * 70)
    print("SERVED (resident server): load once, N requests")
    print("=" * 70)
    server, thread, load_time = start_server()
    print(f"  server load() time: {load_time:.2f}s (loads all 4 UNet checkpoints once)")
    served_times = {}
    t_served_requests_start = time.monotonic()
    for well_id in WELLS:
        staged_csv, staging_root = staged[well_id]
        run_root = SERVED_OUT / well_id
        run_root.mkdir(parents=True, exist_ok=True)
        for rel_dir in ("object_extraction",):
            src = staging_root / rel_dir
            dst = run_root / rel_dir
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
        run_csv = run_root / "object_extraction" / well_id.split("_")[0] / "snips" / "per_well" / well_id / f"{well_id}_snip_inventory.csv"
        dt = run_served(well_id, run_csv, run_root)
        served_times[well_id] = dt
        print(f"  {well_id}: {dt:.2f}s (request time only, model already loaded)")
    t_served_requests_total = time.monotonic() - t_served_requests_start
    t_served_total = load_time + t_served_requests_total
    server._stop_event.set()
    thread.join(timeout=10.0)

    print()
    print("=" * 70)
    print("EQUIVALENCE CHECK")
    print("=" * 70)
    all_match = True
    for well_id in WELLS:
        result = compare_outputs(well_id)
        print(f"  {well_id}: {json.dumps(result, indent=2)}")
        if not (result["row_count_match"] and result["non_path_columns_equal"] and result["masks_byte_identical"]):
            all_match = False

    print()
    print("=" * 70)
    print("THROUGHPUT SUMMARY")
    print("=" * 70)
    print(f"PER-WELL total wall clock ({len(WELLS)} wells, {len(WELLS)}x model reload): {t_perwell_total:.2f}s")
    print(f"  per-well breakdown: {perwell_times}")
    print(f"SERVED total wall clock (1x model load + {len(WELLS)} requests): {t_served_total:.2f}s")
    print(f"  load time: {load_time:.2f}s, request-only total: {t_served_requests_total:.2f}s")
    print(f"  request breakdown: {served_times}")
    print(f"Speedup (this run, N={len(WELLS)}): {t_perwell_total / t_served_total:.2f}x")

    print()
    print(f"ALL OUTPUTS MATCH: {all_match}")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
