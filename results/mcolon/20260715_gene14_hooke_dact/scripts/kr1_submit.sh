#!/usr/bin/env bash
#$ -N gene14_kr1
#$ -q trapnell-login.q
#$ -l mfree=24G
#$ -l h_rt=24:00:00
#$ -j y
#$ -pe serial 8
#$ -cwd
#$ -V
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260715_gene14_hooke_dact/output
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260715_gene14_hooke_dact/output
# -----------------------------------------------------------------------------
# Submit the KR1 baseline hooke fits (kr1_baseline_fits.R) to the grid.
# The interactive login node has only ~8G RAM; loading the 7.3G BPCells CDS
# OOM-kills there (exit 137). mfree is PER-SLOT, so 24G x 8 = 192G headroom.
#
# Submit:
#   qsub results/mcolon/20260715_gene14_hooke_dact/scripts/kr1_submit.sh
# Watch:
#   qstat ; tail -f results/mcolon/20260715_gene14_hooke_dact/output/gene14_kr1.o<JOBID>
# Results land in output/: kr1_raw_contrasts.tsv/.rds, kr1_dacts.tsv
# -----------------------------------------------------------------------------
set -euo pipefail
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq

# BPCells.so needs HDF5 1.14.3 (libhdf5.so.310); -V doesn't reliably carry
# LD_LIBRARY_PATH to the exec node, so set it explicitly.
HDF5_LIB=/net/gs/vol3/software/modules-sw/hdf5/1.14.3-cxxenabled/Linux/Ubuntu22.04/x86_64/lib
export LD_LIBRARY_PATH="${HDF5_LIB}:${LD_LIBRARY_PATH:-}"
echo "[submit] libhdf5.so.310 -> $(ls ${HDF5_LIB}/libhdf5.so.310 2>&1)"

# BPCells writes multi-GB on-disk scratch into the cwd; point TMPDIR at the
# gitignored tmp/ tree and run FROM there so the repo root stays clean.
export TMPDIR="/net/trapnell/vol1/home/mdcolon/proj/morphseq/tmp/bpcells_scratch/${JOB_ID:-$$}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR"
cd "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "[submit] TMPDIR=$TMPDIR (bpcells scratch; auto-removed on exit)"

RSCRIPT=/net/gs/vol3/software/modules-sw/R/4.4.1/Linux/Ubuntu22.04/x86_64/bin/Rscript
RSRC=/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260715_gene14_hooke_dact/scripts/kr1_baseline_fits.R
echo "[submit] host=$(hostname) slots=${NSLOTS:-NA} mfree=24G/slot cwd=$(pwd) start=$(date)"
"$RSCRIPT" "$RSRC"
echo "[submit] done=$(date)"
