#!/bin/bash
#$ -N coll_yx1_e2e
#$ -l h_rt=14400
#$ -l m_mem_free=64G
#$ -o /net/trapnell/vol1/home/mdcolon/proj/morphseq/.coll_wt/.collection_test_scratch/yx1_e2e.o
#$ -e /net/trapnell/vol1/home/mdcolon/proj/morphseq/.coll_wt/.collection_test_scratch/yx1_e2e.e
#$ -q trapnell-long.q
# CPU-only: this is metadata ingest (ND2 header/stage reads), no GPU. The ND2s are 18G each and
# there are 3 of them, hence the generous walltime + memory.
source /net/trapnell/vol1/home/mdcolon/software/miniconda3/etc/profile.d/conda.sh
conda activate segmentation_grounded_sam
export PYTHONPATH=/net/trapnell/vol1/home/mdcolon/proj/morphseq/.coll_wt/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/.coll_wt
python -u .collection_test_scratch/yx1_e2e.py
echo "EXIT_CODE:$?"
