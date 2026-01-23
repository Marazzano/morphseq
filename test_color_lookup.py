#!/usr/bin/env python
"""Test the ColorLookup object."""

from src.analyze.trajectory_analysis.config import GENOTYPE_COLORS

# Test as dict-like object
print('Testing GENOTYPE_COLORS object:')
print(f'Type: {type(GENOTYPE_COLORS)}')
print(f'\nDirect suffix matches:')
print(f'  wildtype: {GENOTYPE_COLORS["wildtype"]}')
print(f'  crispant: {GENOTYPE_COLORS["crispant"]}')
print(f'\nPrefix + suffix matches:')
print(f'  gene1_heterozygous: {GENOTYPE_COLORS["gene1_heterozygous"]}')
print(f'  sox10_homozygous: {GENOTYPE_COLORS["sox10_homozygous"]}')
print(f'\nFallback for unmatched:')
print(f'  unknown_value: {GENOTYPE_COLORS["unknown_value"]}')

# Test caching
print(f'\nCached items after lookups:')
for key, value in GENOTYPE_COLORS.items():
    print(f'  {key}: {value}')

# Test dict-like methods
print(f'\nDict-like interface:')
print(f'  "wildtype" in GENOTYPE_COLORS: {"wildtype" in GENOTYPE_COLORS}')
print(f'  get("missing", "default"): {GENOTYPE_COLORS.get("missing", "default")}')
