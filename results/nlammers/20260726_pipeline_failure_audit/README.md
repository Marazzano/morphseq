# Keyence + YX1 pipeline audit dashboard

Refresh the Excel audit and re-execute the Jupyter dashboard from the repository root:

```bash
results/nlammers/20260726_pipeline_failure_audit/refresh_dashboard.sh
```

The refresh reads the archive manifest, canonical pipeline artifacts, validated sentinels, and
front/back-half logs. Approximate frame totals are estimated from at most five per-well canonical
frame inventories per dataset; raw TIFF and ND2 planes are not enumerated.

Outputs:

- `pipeline_audit.xlsx`: one row per dataset plus stage evidence, overlapping failure details,
  grouped taxonomy, recovery priorities, log provenance, and a data dictionary.
- `pipeline_failure_dashboard.ipynb`: executed executive dashboard and detailed failure reference.

Use `--no-execute-notebook` to rebuild the workbook and notebook source without running the
notebook cells.
