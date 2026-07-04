#!/bin/bash
# Morning well-accounting check for the Tier 2 (WIDTH) through-line run.
#   docs/refactors/streamline-snakemake/target/specs/data_flow_test_plan.md  (Tier 2 success criteria)
#
# Pass condition (the plan): missing_wells == none AND merged snip_qc covers exactly the
# discovered wells. A green qsub job is NOT enough — width bugs hide in a well that slips a grate.
#
#   bash tier2_well_accounting.sh
set -euo pipefail

EXP=20250912
ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output
DISCOVERED="${ROOT}/acquisition/${EXP}/discovered_wells.txt"
PERWELL_DIR="${ROOT}/quality_control/${EXP}/snip_qc/per_well"
MERGED="${ROOT}/quality_control/${EXP}/snip_qc/${EXP}_snip_qc.parquet"

source /net/trapnell/vol1/home/mdcolon/software/miniconda3/etc/profile.d/conda.sh
conda activate segmentation_grounded_sam

echo "=== Tier 2 well-accounting — ${EXP} — $(date) ==="
n_disc=$(wc -l < "${DISCOVERED}")
echo "discovered wells:        ${n_disc}"

# Per-well snip_qc shards that produced a validated verdict.
n_shards=$(find "${PERWELL_DIR}" -name "*_snip_qc.parquet.validated" 2>/dev/null | wc -l)
echo "validated per-well shards: ${n_shards}"

# Which discovered wells are MISSING a validated verdict (the wells that fell into a grate).
echo "--- wells missing a validated snip_qc verdict ---"
missing=0
while read -r w; do
  [ -z "$w" ] && continue
  if [ ! -f "${PERWELL_DIR}/${EXP}_${w}/${EXP}_${w}_snip_qc.parquet.validated" ]; then
    echo "  MISSING: ${w}"; missing=$((missing+1))
  fi
done < "${DISCOVERED}"
[ "$missing" -eq 0 ] && echo "  (none — all discovered wells have a verdict)"

# Merged table: exists? and does its well set == discovered set?
echo "--- merged snip_qc ---"
if [ -f "${MERGED}" ]; then
  echo "  merged exists: ${MERGED}"
  python - "$MERGED" "$DISCOVERED" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1])
merged_wells = set(df['well_id'].unique())
disc = set(f"{l.strip()}" for l in open(sys.argv[2]) if l.strip())
# discovered file holds bare well tokens (B01); merged well_id is exp_well (20250912_B01)
disc_full = {w if w.startswith('20') else None for w in disc}
print(f"  merged rows: {len(df)}, merged wells: {len(merged_wells)}")
# normalize: compare suffixes
mw_suf = {w.split('_')[-1] for w in merged_wells}
print(f"  wells in merged but NOT discovered (extra): {sorted(mw_suf - disc) or 'none'}")
print(f"  wells discovered but NOT in merged (missing): {sorted(disc - mw_suf) or 'none'}")
print(f"  duplicate snip_id in merged: {(~df['snip_id'].is_unique)}")
PY
else
  echo "  MERGED MISSING — merge_snip_qc did not run (a well failed, so --keep-going skipped the merge)."
  echo "  This is the expected signal of a width failure: investigate the missing wells above."
fi

echo "=== PASS iff missing==0 AND merged exists AND extra==none AND missing==none ==="
