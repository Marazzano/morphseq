from pathlib import Path
import pandas as pd
from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import (
    union_collection_acquisition_inventories, SourceChild)
from data_pipeline.acquisition.metadata_ingest.scope.keyence.extract_scope_metadata import extract_keyence_scope_metadata

RAW = Path('.collection_test_scratch/raw/chem28c_coll')
OUT = Path('.collection_test_scratch/out'); OUT.mkdir(parents=True, exist_ok=True)

def read_source(source, experiment_id):
    return extract_keyence_scope_metadata(
        raw_data_dir=RAW, experiment_id=source.child_name,
        output_csv=OUT / f'scope_{source.child_name}.csv')

sources = [SourceChild(child_name='20250622_plate01_t28hpf', scope='Keyence'),
           SourceChild(child_name='20250623_plate01_t52hpf', scope='Keyence')]
unioned = union_collection_acquisition_inventories(
    collection_name='chem28c_coll', sources=sources, read_source=read_source)
print('ROWS', len(unioned))
print('EXPID', list(unioned['experiment_id'].unique()))
print('NSOURCES', list(unioned['n_sources'].unique()))
print('WELLS', sorted(unioned['well_id'].astype(str).unique())[:6])
print('TIMEIDX', sorted(unioned['time_index'].unique()))
print('AGE_by_tidx', unioned.groupby('time_index')['start_age_hpf'].first().to_dict())
per_well_tidx = unioned.groupby('well_id')['time_index'].nunique()
print('WELLS_WITH_2_TIMEPOINTS', int((per_well_tidx==2).sum()), 'of', len(per_well_tidx))
unioned.to_csv(OUT/'unioned_frame_inventory.csv', index=False)
print('OK')
