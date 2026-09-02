from pathlib import Path
import pandas as pd
from data_pipeline.acquisition.metadata_ingest.collection_acquisition_union import (
    union_collection_acquisition_inventories, SourceChild)
from data_pipeline.acquisition.metadata_ingest.scope.keyence.acquisition_inventory import (
    build_keyence_acquisition_inventory)
from data_pipeline.acquisition.metadata_ingest.scope.keyence.extract_scope_metadata import (
    make_keyence_plane_scraper)

RAW = Path('.collection_test_scratch/raw/chem28c_coll')
scrape = make_keyence_plane_scraper()

def read_source(source, experiment_id):
    # correct layer: the ACQUISITION INVENTORY builder (mints channel_id, elapsed_time_s),
    # not the raw scope extractor.
    return build_keyence_acquisition_inventory(
        experiment_id=source.child_name,
        raw_data_dir=RAW / source.child_name,   # this builder takes the experiment dir directly
        scrape_plane_metadata=scrape)

sources = [SourceChild(child_name='20250622_plate01_t28hpf', scope='Keyence'),
           SourceChild(child_name='20250623_plate01_t52hpf', scope='Keyence')]
unioned = union_collection_acquisition_inventories(
    collection_name='chem28c_coll', sources=sources, read_source=read_source)
print('ROWS', len(unioned))
print('EXPID', list(unioned['experiment_id'].unique()))
print('NSOURCES', list(unioned['n_sources'].unique()))
print('has channel_id/elapsed_time_s:', 'channel_id' in unioned.columns, 'elapsed_time_s' in unioned.columns)
print('TIMEIDX', sorted(unioned['time_index'].unique()))
print('AGE_by_tidx', unioned.groupby('time_index')['start_age_hpf'].first().to_dict())
from data_pipeline.acquisition.metadata_ingest.scope.keyence.acquisition_inventory import validate_keyence_acquisition_inventory
try:
    validate_keyence_acquisition_inventory(unioned)
    print('VALIDATOR: PASS')
except Exception as e:
    print('VALIDATOR:', str(e)[:250])
unioned.to_csv('.collection_test_scratch/out/unioned_acq_inventory.csv', index=False)
