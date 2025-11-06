"""
Process and normalize plate layout metadata from Excel/CSV files.

This module reads raw plate layout files, normalizes column names to match
the schema, and validates the output.
"""

from pathlib import Path
import pandas as pd

from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
from data_pipeline.io.validators import validate_dataframe_schema


def process_plate_layout(
    input_file: Path,
    experiment_id: str,
    output_csv: Path
) -> pd.DataFrame:
    """
    Normalize and validate plate metadata from Excel or CSV file.

    Args:
        input_file: Path to input Excel (.xlsx) or CSV file
        experiment_id: Experiment identifier to add to each row
        output_csv: Path where validated CSV will be written

    Returns:
        Validated DataFrame with normalized column names

    Raises:
        ValueError: If required columns are missing or contain null values
        FileNotFoundError: If input file doesn't exist
    """
    # Load file (handle both Excel and CSV)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    elif input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}. Use .xlsx, .xls, or .csv")

    # Normalize column names
    df = _normalize_column_names(df)

    # Add experiment_id if not present
    if 'experiment_id' not in df.columns:
        df['experiment_id'] = experiment_id

    # Generate well_id (format: experiment_id_well_index)
    if 'well_id' not in df.columns:
        df['well_id'] = df['experiment_id'] + '_' + df['well_index']

    # Validate against schema
    validate_dataframe_schema(df, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata")

    # Write validated CSV
    df.to_csv(output_csv, index=False)

    return df


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to match schema expectations.

    Handles common variations in plate layout files:
    - well → well_index
    - experiment_date → experiment_id
    - chem_perturbation → treatment
    - temperature → temperature_c

    Args:
        df: Raw DataFrame from input file

    Returns:
        DataFrame with normalized column names
    """
    # Define column mappings (input_name → schema_name)
    column_mappings = {
        # Well identifiers
        'well': 'well_index',
        'Well': 'well_index',
        'well_name': 'well_index',

        # Experiment ID
        'experiment_date': 'experiment_id',
        'experiment': 'experiment_id',
        'exp_id': 'experiment_id',

        # Treatment
        'chem_perturbation': 'treatment',
        'chemical_perturbation': 'treatment',
        'drug': 'treatment',

        # Temperature
        'temperature': 'temperature_c',
        'temp': 'temperature_c',
        'temp_c': 'temperature_c',

        # Age
        'start_age': 'start_age_hpf',
        'age_hpf': 'start_age_hpf',
        'age': 'start_age_hpf',

        # Embryos
        'embryos': 'embryos_per_well',
        'n_embryos': 'embryos_per_well',
    }

    # Apply mappings
    df = df.rename(columns=column_mappings)

    return df
