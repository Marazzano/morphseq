#!/bin/bash
#$ -N unet_bench
#$ -l gpgpu=1
#$ -l cuda=1
#$ -l h_rt=3600
#$ -l m_mem_free=16G
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-afed7356581c76751/results/mcolon/20260725_unet_aux_masks_bench/bench.o
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-afed7356581c76751/results/mcolon/20260725_unet_aux_masks_bench/bench.e
#$ -q trapnell-short.q
source /net/trapnell/vol1/home/mdcolon/software/miniconda3/etc/profile.d/conda.sh
conda activate segmentation_grounded_sam
export PYTHONPATH=/net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-afed7356581c76751/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
nvidia-smi -L
timeout 1800 python -u /net/trapnell/vol1/home/mdcolon/proj/morphseq/.claude/worktrees/agent-afed7356581c76751/results/mcolon/20260725_unet_aux_masks_bench/bench.py
echo "EXIT_CODE:$?"
