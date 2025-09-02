"""
Pipeline-first service for embedding ingestion/generation and df02 merge.

Centralizes embedding operations under clear, testable functions used by Build06.
Keeps existing assessment scripts intact while providing canonical pipeline outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, List, Union
import pandas as pd
import logging

# Import calculate_morph_embeddings lazily to avoid heavy dependency chain
# from src.analyze.analysis_utils import calculate_morph_embeddings


def resolve_model_dir(data_root: Union[str, Path], model_name: str) -> Path:
    """
    Resolves model directory path and validates it contains required files.
    
    Args:
        data_root: Path to data root directory
        model_name: Name of model (e.g., "20241107_ds_sweep01_optimum")
        
    Returns:
        Path to validated model directory
        
    Raises:
        FileNotFoundError: If model directory or config not found
    """
    data_root = Path(data_root)
    model_dir = data_root / "models" / "legacy" / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    # Check for model_config.json or other required files
    config_files = list(model_dir.glob("*config*.json"))
    if not config_files:
        # Check for alternative model structure (final_model subdirectory)
        final_model_dir = model_dir / "final_model"
        if final_model_dir.exists():
            config_files = list(final_model_dir.glob("*config*.json"))
            if config_files:
                return final_model_dir
        
        # If no config found, still return the directory - some models may not need it
        logging.warning(f"No config file found in {model_dir}, proceeding anyway")
    
    return model_dir


def ensure_latents_for_experiments(
    data_root: Union[str, Path],
    model_name: str,
    experiments: List[str],
    generate_missing: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Path]:
    """
    Ensures latent CSV files exist for all experiments.
    
    Args:
        data_root: Path to data root directory
        model_name: Name of model for embedding generation
        experiments: List of experiment names
        generate_missing: If True, generate missing latent files
        logger: Optional logger for output
        
    Returns:
        Dict mapping experiment names to their latent CSV paths
        
    Raises:
        FileNotFoundError: If latent files missing and generate_missing=False
    """
    data_root = Path(data_root)
    latent_dir = data_root / "analysis" / "latent_embeddings" / "legacy" / model_name
    
    latent_paths = {}
    missing_experiments = []
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    for exp in experiments:
        latent_path = latent_dir / f"morph_latents_{exp}.csv"
        
        if latent_path.exists():
            latent_paths[exp] = latent_path
            logger.info(f"Found existing latents for {exp}: {latent_path}")
        else:
            missing_experiments.append(exp)
            logger.warning(f"Missing latents for {exp}: {latent_path}")
    
    if missing_experiments:
        if generate_missing:
            logger.info(f"Generating missing latents for {len(missing_experiments)} experiments")
            
            # Lazy import to avoid heavy dependency chain
            try:
                from src.analyze.analysis_utils import calculate_morph_embeddings
            except ImportError as e:
                logger.error(f"Cannot import calculate_morph_embeddings: {e}")
                logger.error("This is required for --generate-missing-latents. Please check your environment.")
                raise ImportError(
                    "Missing dependencies for embedding generation. "
                    "Please ensure all required packages (einops, torch, etc.) are installed."
                ) from e
            
            # Use existing calculate_morph_embeddings function
            # Note: calculate_morph_embeddings writes to
            #   <data_root>/analysis/latent_embeddings/legacy/<model_name>/morph_latents_<exp>.csv
            # We pass through model_name as-is so callers can direct outputs to a custom tag.
            calculate_morph_embeddings(
                data_root=data_root,
                model_name=model_name,
                model_class="legacy",
                experiments=missing_experiments
            )
            
            # Verify generation succeeded
            for exp in missing_experiments:
                latent_path = latent_dir / f"morph_latents_{exp}.csv"
                if latent_path.exists():
                    latent_paths[exp] = latent_path
                    logger.info(f"Generated latents for {exp}: {latent_path}")
                else:
                    logger.error(f"Failed to generate latents for {exp}")
                    raise FileNotFoundError(f"Could not generate latents for {exp}")
        else:
            logger.error(f"Missing latent files for {len(missing_experiments)} experiments")
            raise FileNotFoundError(
                f"Missing latent files: {missing_experiments}. "
                f"Use --generate-missing-latents to create them."
            )
    
    return latent_paths


def generate_latents_with_repo_images(
    repo_root: Union[str, Path],
    data_root: Union[str, Path],
    model_name: str,
    latents_tag: Optional[str],
    experiments: List[str],
    logger: Optional[logging.Logger] = None,
):
    """
    Generate per-experiment latent CSVs using snips under the repo root
    (SAM2 naming) while loading the model from the central data root.

    Outputs are written under:
      <data_root>/analysis/latent_embeddings/legacy/<latents_dir>/morph_latents_<exp>.csv

    Args:
        repo_root: Project/repo root that contains training_data/bf_embryo_snips
        data_root: Central data root for models and analysis outputs
        model_name: Name of the model directory under <data_root>/models/legacy
        latents_tag: Optional alt directory name for outputs (defaults to model_name)
        experiments: Experiment names to process
        logger: Optional logger
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    repo_root = Path(repo_root)
    data_root = Path(data_root)
    latents_dir = data_root / "analysis" / "latent_embeddings" / "legacy" / (latents_tag or model_name)
    latents_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports to avoid heavy startup
    try:
        import torch
        import numpy as np
        from torch.utils.data import DataLoader
        from src.analyze.analysis_utils import extract_embeddings_legacy
        from src.vae.models.auto_model import AutoModel
        from src.data.data_transforms import basic_transform
        from src.data.dataset_configs import EvalDataConfig
    except Exception as e:
        logger.error(f"Failed to import dependencies for latent generation: {e}")
        raise

    # Load model from central data_root
    model_dir = resolve_model_dir(data_root, model_name)
    logger.info(f"Loading model from {model_dir}")
    lit_model = AutoModel.load_from_folder(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataloader using repo's snips
    input_size = (288, 128)  # matches legacy defaults
    transform = basic_transform(target_size=input_size)
    eval_cfg = EvalDataConfig(experiments=experiments, root=repo_root, return_sample_names=True, transforms=transform)
    dataset = eval_cfg.create_dataset()
    dl = DataLoader(dataset, batch_size=64, num_workers=eval_cfg.num_workers, shuffle=False)

    # Run encoder and collect embeddings
    logger.info(f"Generating embeddings for {len(experiments)} experiments from repo snips")
    lit_model.to(device).eval()
    latent_df = extract_embeddings_legacy(lit_model=lit_model, dataloader=dl, device=device)

    # Write per-experiment CSVs
    for exp in experiments:
        exp_df = latent_df.loc[latent_df["experiment_date"] == exp]
        out_path = latents_dir / f"morph_latents_{exp}.csv"
        exp_df.to_csv(out_path, index=False)
        logger.info(f"Wrote {len(exp_df)} rows to {out_path}")


def load_latents(latent_paths: Dict[str, Path], logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Loads and combines per-experiment latent CSV files.
    
    Args:
        latent_paths: Dict mapping experiment names to CSV paths
        logger: Optional logger for output
        
    Returns:
        Combined DataFrame with snip_id + z_mu_* columns
        
    Raises:
        ValueError: If DataFrames have inconsistent schemas or invalid data
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    dfs = []
    
    for exp, path in latent_paths.items():
        logger.info(f"Loading latents from {path}")
        df = pd.read_csv(path)
        
        # Validate required columns
        if 'snip_id' not in df.columns:
            raise ValueError(f"Missing 'snip_id' column in {path}")
        
        # Find z_mu columns
        z_cols = [col for col in df.columns if col.startswith('z_mu_')]
        if not z_cols:
            logger.warning(f"No z_mu_* columns found in {path}")
            continue
        
        # Select snip_id and all z_mu columns (including z_mu_b_*, z_mu_n_* variants)
        keep_cols = ['snip_id'] + z_cols
        df_subset = df[keep_cols].copy()
        
        # Validate finite values in z columns
        z_numeric_cols = [col for col in z_cols if df_subset[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        for col in z_numeric_cols:
            if not df_subset[col].isna().all():  # Skip if all NaN
                import numpy as np
                finite_mask = np.isfinite(df_subset[col])
                if not finite_mask.all():
                    invalid_count = (~finite_mask).sum()
                    logger.warning(f"Found {invalid_count} non-finite values in {col} from {path}")
                    # Replace non-finite with NaN
                    df_subset.loc[~finite_mask, col] = pd.NA
        
        dfs.append(df_subset)
        logger.info(f"Loaded {len(df_subset)} rows, {len(z_cols)} z_mu columns from {exp}")
    
    if not dfs:
        raise ValueError("No valid latent data found")
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates by snip_id, keeping first occurrence
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['snip_id'], keep='first')
    final_count = len(combined_df)
    
    if initial_count != final_count:
        logger.warning(f"Removed {initial_count - final_count} duplicate snip_ids")
    
    logger.info(f"Combined latents: {len(combined_df)} unique snip_ids")
    
    return combined_df


def normalize_snip_ids(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Normalizes snip_id formats to ensure consistent joining.
    
    Handles variations like '_s####' vs '_####' suffixes from different naming schemes.
    
    Args:
        df: DataFrame with snip_id column
        logger: Optional logger for output
        
    Returns:
        DataFrame with normalized snip_ids
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    df = df.copy()
    
    # Ensure consistent string format
    df['snip_id'] = df['snip_id'].astype(str)
    
    # Log some sample snip_ids for debugging
    sample_ids = df['snip_id'].head(3).tolist()
    logger.info(f"Sample snip_ids: {sample_ids}")
    
    logger.info(f"Normalized {len(df)} snip_ids")
    
    return df


def merge_df02_with_embeddings(
    root: Union[str, Path],
    latents_df: pd.DataFrame,
    overwrite: bool = False,
    out_name: str = "embryo_metadata_df03.csv",
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Merges df02 with embeddings to create df03.
    
    Args:
        root: Pipeline root directory
        latents_df: DataFrame with embeddings (snip_id + z_mu_* columns)
        overwrite: Allow overwriting existing df03
        out_name: Output filename
        logger: Optional logger for output
        
    Returns:
        Path to created df03 file
        
    Raises:
        FileNotFoundError: If df02 doesn't exist
        FileExistsError: If df03 exists and overwrite=False
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    root = Path(root)
    df02_path = root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
    df03_path = root / "metadata" / "combined_metadata_files" / out_name
    
    if not df02_path.exists():
        raise FileNotFoundError(f"df02 not found: {df02_path}")
    
    if df03_path.exists() and not overwrite:
        raise FileExistsError(f"df03 already exists: {df03_path}. Use --overwrite to replace.")
    
    # Load df02
    logger.info(f"Loading df02 from {df02_path}")
    df02 = pd.read_csv(df02_path)
    
    # Normalize snip_ids in both DataFrames
    df02_norm = normalize_snip_ids(df02, logger)
    latents_norm = normalize_snip_ids(latents_df, logger)
    
    # Merge on snip_id (left join to preserve all df02 rows)
    logger.info("Merging df02 with embeddings")
    df03 = df02_norm.merge(latents_norm, on='snip_id', how='left')
    
    # Calculate join coverage
    embedding_cols = [col for col in df03.columns if col.startswith('z_mu_')]
    if embedding_cols:
        # Check coverage using first z column as indicator
        first_z_col = embedding_cols[0]
        matched_count = df03[first_z_col].notna().sum()
        total_count = len(df03)
        coverage = matched_count / total_count if total_count > 0 else 0
        
        logger.info(f"Join coverage: {matched_count}/{total_count} ({coverage:.1%})")
        
        if coverage < 0.95:
            logger.warning(f"Low join coverage: {coverage:.1%} < 95%")
    else:
        logger.warning("No embedding columns found after merge")
    
    # Write df03
    logger.info(f"Writing df03 to {df03_path}")
    df03.to_csv(df03_path, index=False)
    
    logger.info(f"✔️  Created df03 with {len(df03)} rows, {len(embedding_cols)} embedding columns")
    
    return df03_path


def merge_train_with_embeddings(
    root: Union[str, Path],
    train_name: str,
    latents_df: pd.DataFrame,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Optionally merges training metadata with embeddings.
    
    Args:
        root: Pipeline root directory
        train_name: Training run name
        latents_df: DataFrame with embeddings
        overwrite: Allow overwriting existing file
        logger: Optional logger for output
        
    Returns:
        Path to created file, or None if training metadata doesn't exist
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    root = Path(root)
    train_meta_path = root / "training_data" / train_name / "embryo_metadata_df_train.csv"
    train_out_path = root / "training_data" / train_name / "embryo_metadata_with_embeddings.csv"
    
    if not train_meta_path.exists():
        logger.info(f"Training metadata not found: {train_meta_path}")
        return None
    
    if train_out_path.exists() and not overwrite:
        raise FileExistsError(f"Training output exists: {train_out_path}. Use --overwrite to replace.")
    
    logger.info(f"Loading training metadata from {train_meta_path}")
    train_df = pd.read_csv(train_meta_path)
    
    # Normalize snip_ids
    train_norm = normalize_snip_ids(train_df, logger)
    latents_norm = normalize_snip_ids(latents_df, logger)
    
    # Merge
    train_with_embeddings = train_norm.merge(latents_norm, on='snip_id', how='left')
    
    # Write output
    logger.info(f"Writing training output to {train_out_path}")
    train_with_embeddings.to_csv(train_out_path, index=False)
    
    return train_out_path


def export_df03_copies_by_experiment(
    df03: pd.DataFrame,
    data_root: Union[str, Path],
    model_name: str,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Exports per-experiment df03 copies for analysis users.
    
    Args:
        df03: Complete df03 DataFrame
        data_root: Data root directory
        model_name: Model name for output directory
        overwrite: Allow overwriting existing files
        logger: Optional logger for output
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    data_root = Path(data_root)
    export_dir = data_root / "metadata" / "metadata_n_embeddings" / model_name
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if 'experiment_date' not in df03.columns:
        logger.warning("No 'experiment_date' column found, cannot create per-experiment exports")
        return
    
    experiments = df03['experiment_date'].dropna().unique()
    
    for exp in experiments:
        exp_df = df03[df03['experiment_date'] == exp]
        exp_path = export_dir / f"df03_{exp}.csv"
        
        if exp_path.exists() and not overwrite:
            logger.warning(f"Skipping existing file: {exp_path} (use --overwrite)")
            continue
        
        exp_df.to_csv(exp_path, index=False)
        logger.info(f"Exported {len(exp_df)} rows for {exp} to {exp_path}")


def build_df03_with_embeddings(
    root: Union[str, Path],
    data_root: Union[str, Path],
    model_name: str = "20241107_ds_sweep01_optimum",
    latents_tag: Optional[str] = None,
    experiments: Optional[List[str]] = None,
    generate_missing: bool = False,
    use_repo_snips: bool = False,
    export_analysis: bool = False,
    train_name: Optional[str] = None,
    write_train_output: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    One-shot orchestrator for Build06 - the main entry point.
    
    Args:
        root: Pipeline root directory
        data_root: Data root directory (for models and latents)
        model_name: Model name for embeddings
        experiments: Optional explicit experiment list
        generate_missing: Generate missing latent files
        export_analysis: Export per-experiment df03 copies
        train_name: Training run name for optional join
        write_train_output: Write training metadata with embeddings
        overwrite: Allow overwriting existing files
        dry_run: Print planned actions without executing
        logger: Optional logger for output
        
    Returns:
        Path to created df03 file
    """
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    root = Path(root)
    data_root = Path(data_root)
    
    logger.info("=== Build06: Standardize Embeddings Generation ===")
    logger.info(f"Root: {root}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Model: {model_name}")
    if latents_tag:
        logger.info(f"Latents tag: {latents_tag}")
    
    # 1) Resolve df02 and experiment list
    df02_path = root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
    if not df02_path.exists():
        raise FileNotFoundError(f"df02 not found: {df02_path}")
    
    if experiments is None:
        # Infer experiments from df02
        df02 = pd.read_csv(df02_path)
        if 'experiment_date' in df02.columns:
            experiments = df02['experiment_date'].dropna().unique().tolist()
            logger.info(f"Inferred {len(experiments)} experiments from df02")
        else:
            raise ValueError("No experiments provided and cannot infer from df02")
    
    logger.info(f"Processing experiments: {experiments}")
    
    if dry_run:
        logger.info("=== DRY RUN - Planned Actions ===")
        logger.info(f"Would process {len(experiments)} experiments")
        logger.info(f"Generate missing latents: {generate_missing}")
        logger.info(f"Use repo snips: {use_repo_snips}")
        logger.info(f"Export analysis copies: {export_analysis}")
        logger.info(f"Write training output: {write_train_output}")
        return df02_path  # Return dummy path for dry run
    
    # 2) Resolve model directory
    model_dir = resolve_model_dir(data_root, model_name)
    logger.info(f"Using model: {model_dir}")
    
    # 3) Ensure latents exist (read/write under <data_root>/analysis/.../<latents_dir>)
    latents_dir_name = latents_tag or model_name
    try:
        latent_paths = ensure_latents_for_experiments(
            data_root, latents_dir_name, experiments, False, logger
        )
    except FileNotFoundError:
        # Missing are expected in first pass; we'll generate below if requested
        latent_paths = {}

    # Generate missing if requested
    missing = [exp for exp in experiments if exp not in latent_paths]
    if missing and generate_missing:
        if use_repo_snips:
            logger.info("Generating missing latents from repo snips (SAM2-aligned)")
            generate_latents_with_repo_images(root, data_root, model_name, latents_dir_name, missing, logger)
        else:
            # Use standard generator, which writes under the given model_name (latents_dir_name)
            _ = ensure_latents_for_experiments(data_root, latents_dir_name, missing, True, logger)
        # Reload paths after generation
        latent_paths = ensure_latents_for_experiments(
            data_root, latents_dir_name, experiments, False, logger
        )
    
    # 4) Load and combine latents
    latents_df = load_latents(latent_paths, logger)
    
    # 5) Merge with df02 to create df03
    df03_path = merge_df02_with_embeddings(root, latents_df, overwrite, logger=logger)
    
    # 6) Optional training output
    if write_train_output and train_name:
        train_output = merge_train_with_embeddings(root, train_name, latents_df, overwrite, logger)
        if train_output:
            logger.info(f"✔️  Created training output: {train_output}")
    
    # 7) Optional analysis exports
    if export_analysis:
        df03 = pd.read_csv(df03_path)
        export_df03_copies_by_experiment(df03, data_root, model_name, overwrite, logger)
    
    logger.info("✔️  Build06 complete")
    return df03_path
