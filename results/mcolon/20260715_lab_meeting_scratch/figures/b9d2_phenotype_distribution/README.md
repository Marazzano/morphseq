# B9D2 phenotype distribution by pair

`phenotype_distribution_by_pair.png` recreates the historical faceted
distribution for experiments 20251121 and 20251125.

Differences from the historical four-color plot:

- `CE` stays `CE`, `BA_rescue` is pooled into `HTA`, and non-penetrant stays gray
- colors use the canonical package palette (`CE=#1b9e77`, `HTA=#d95f02`,
  `non_penetrant=#9E9E9E`)
- input comes from the cleaned SCI cilia QC reference table
- one QC-passing row per embryo is counted

Columns are B9D2 pairs 2, 4, 5, 6, 7, and 8. Rows are wildtype,
heterozygous, and homozygous. Each panel is normalized to 100%, with embryo counts
printed above its bars.
