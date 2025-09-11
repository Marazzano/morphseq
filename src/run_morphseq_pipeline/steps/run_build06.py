from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union
import logging
import os

from ..services.gen_embeddings import build_df03_with_embeddings


def run_build06(
    root: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
    model_name: str = "20241107_ds_sweep01_optimum",
    experiments: Optional[List[str]] = None,
    generate_missing: bool = False,
    overwrite_latents: bool = False,
    export_analysis: bool = False,
    train_name: Optional[str] = None,
    write_train_output: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    """Standardize morphological embedding generation (Build06).
    
    Centralizes embedding aggregation by:
    1. Ensuring latents exist for experiments in df02 (optionally generates missing via legacy batch path)
    2. Aggregating per-experiment latents and merging into df02 to produce canonical df03
    3. Optionally writing per-experiment df03 under central data root for analysis users
    4. Providing clear CLI, safety checks, and non-destructive defaults
    
    This is the new approach that replaces training-subset embedding generation with
    full df02 aggregation using the legacy latent store as source of truth.
    
    Args:
        root: Data root directory (contains metadata, models, latents, etc.)
        data_root: Optional external data root. If None, uses root for everything.
        model_name: Model name for embedding generation (default: "20241107_ds_sweep01_optimum")
        experiments: Optional explicit experiment list (else inferred from df02)
        generate_missing: Generate missing latent files using legacy batch path
        export_analysis: Export per-experiment df03 copies to data root
        train_name: Training run name for optional join
        write_train_output: Write training metadata with embeddings
        overwrite: Allow overwriting existing files
        dry_run: Print planned actions without executing
        
    Returns:
        Path to created df03 file
        
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If configuration is invalid
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Delegate to the service orchestrator
    return build_df03_with_embeddings(
        root=root,
        data_root=data_root,
        model_name=model_name,
        experiments=experiments,
        generate_missing=generate_missing,
        overwrite_latents=overwrite_latents,
        export_analysis=export_analysis,
        train_name=train_name,
        write_train_output=write_train_output,
        overwrite=overwrite,
        dry_run=dry_run,
        logger=logger,
    )
