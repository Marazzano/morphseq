#!/usr/bin/env python3
"""
Standalone Python 3.9 script for generating embeddings using legacy models.
This script is called as a subprocess from the main Build06 pipeline.
"""

import sys
import os
from pathlib import Path
import json

def main():
    # Parse command line arguments
    if len(sys.argv) != 6:
        print("Usage: python generate_embeddings_py39.py <data_root> <model_name> <model_class> <experiments_json> <batch_size>", file=sys.stderr)
        return 1
    
    data_root = Path(sys.argv[1])
    model_name = sys.argv[2] 
    model_class = sys.argv[3]
    experiments = json.loads(sys.argv[4])
    batch_size = int(sys.argv[5])
    
    # Verbosity via env var (default: quiet)
    verbose = os.environ.get("MORPHSEQ_EMBED_VERBOSE", "0") == "1"
    if verbose:
        print(f"Python 3.9 subprocess running...")
        print(f"Python version: {sys.version_info}")
        print(f"Data root: {data_root}")
        print(f"Model: {model_name}")
        print(f"Experiments: {experiments}")
    else:
        # Minimal header
        print(f"Generating embeddings for {', '.join(experiments)} using {model_name}...")
    
    # Validate Python version
    if sys.version_info[:2] != (3, 9):
        print(f"ERROR: Expected Python 3.9, got {sys.version_info[0]}.{sys.version_info[1]}", file=sys.stderr)
        return 1
    
    try:
        # Import the minimal dependencies needed
        import torch
        import numpy as np
        import pandas as pd
        from torch.utils.data import DataLoader
        
        # Add repo to Python path - assumes script is run from morphseq repo root
        # (subprocess caller sets cwd=repo_root)
        if verbose:
            print("🔍 Running from repo root, adding '.' to Python path")
        sys.path.insert(0, ".")
        
        # Now import the specific modules we need (without the problematic imports)
        from src.vae.models.auto_model import AutoModel
        from src.data.dataset_configs import EvalDataConfig
        from src.data.data_transforms import basic_transform
        
        if verbose:
            print("✅ Successfully imported dependencies in Python 3.9")
        
        # Core embedding extraction logic (simplified)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Using device: {device}")
        
        # Load model
        model_dir = data_root / "models" / model_class / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        if verbose:
            print(f"Loading model from: {model_dir}")
        lit_model = AutoModel.load_from_folder(str(model_dir))
        lit_model.to(device)
        lit_model.eval()
        
        input_size = (288, 128)
        transform = basic_transform(target_size=input_size)
        
        # Process each experiment
        all_embeddings = []
        
        for exp in experiments:
            if verbose:
                print(f"Processing experiment: {exp}")
            
            # Initialize data config
            eval_data_config = EvalDataConfig(
                experiments=[exp],
                root=data_root,
                return_sample_names=True, 
                transforms=transform
            )
            
            # Create dataset from config
            dataset = eval_data_config.create_dataset()
            
            # Create dataloader
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Try to obtain a deterministic list of file paths from the dataset for fallback naming
            dataset_paths = None
            for attr in ("paths", "file_paths", "files", "filenames", "samples"):
                if hasattr(dataset, attr):
                    val = getattr(dataset, attr)
                    # TorchVision datasets sometimes use list of tuples (path, label)
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (tuple, list)):
                        try:
                            val = [v[0] for v in val]
                        except Exception:
                            pass
                    if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (str, Path)):
                        dataset_paths = [str(p) for p in val]
                        break

            processed_count = 0  # used to slice dataset_paths when batch lacks names
            
            if verbose:
                print(f"Processing {len(dl)} batches...")
            
            exp_embeddings = []
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(dl):
                    if verbose and batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}/{len(dl)}")
                    
                    # Handle DatasetOutput from BasicEvalDataset
                    if hasattr(batch_data, 'data'):
                        # It's a DatasetOutput-like object
                        x = batch_data.data
                        sample_names = None
                        # Prefer explicit file path collections
                        if hasattr(batch_data, 'paths') and batch_data.paths is not None:
                            sample_names = [Path(p).stem for p in batch_data.paths]
                        elif hasattr(batch_data, 'sample_names') and batch_data.sample_names is not None:
                            sample_names = [Path(p).stem for p in batch_data.sample_names]
                        elif hasattr(batch_data, 'filenames') and batch_data.filenames is not None:
                            sample_names = [Path(p).stem for p in batch_data.filenames]
                        # As a last resort, accept string labels only (not tensors)
                        elif hasattr(batch_data, 'label') and isinstance(batch_data.label, (list, tuple)) and len(batch_data.label) > 0 and isinstance(batch_data.label[0], (str, bytes)):
                            sample_names = [Path(p).stem for p in batch_data.label]
                        else:
                            # Fallback: derive names by slicing dataset_paths in order
                            if dataset_paths is None:
                                raise TypeError(
                                    "Batch does not provide file paths and dataset has no path list. "
                                    "Ensure the dataset returns (image, path) or exposes .paths/.sample_names/.files"
                                )
                            start = processed_count
                            end = start + x.shape[0]
                            sample_names = [Path(p).stem for p in dataset_paths[start:end]]
                            processed_count = end
                    else:
                        # Fallback to tuple unpacking if it's not DatasetOutput
                        x, paths_or_names = batch_data
                        # paths_or_names should be list[str] of file paths or stems
                        sample_names = [Path(p).stem for p in paths_or_names]

                    # Sanity check alignment
                    if x.shape[0] != len(sample_names):
                        raise ValueError(f"Batch size ({x.shape[0]}) does not match sample_names ({len(sample_names)})")
                    
                    x = x.to(device)
                    
                    # Extract embeddings using encoder
                    encoder_output = lit_model.encoder(x)
                    
                    # Handle different encoder output formats
                    if hasattr(encoder_output, 'embedding'):
                        z_mu = encoder_output.embedding
                    elif hasattr(encoder_output, 'mu'):
                        z_mu = encoder_output.mu
                    elif isinstance(encoder_output, (tuple, list)) and len(encoder_output) >= 1:
                        z_mu = encoder_output[0]  # First element is usually mu
                    else:
                        z_mu = encoder_output
                    
                    # Move to CPU and convert to numpy
                    z_mu_np = z_mu.cpu().numpy()
                    
                    # Create dataframe for this batch
                    for i, sample_name in enumerate(sample_names):
                        row = {"snip_id": sample_name, "experiment_date": exp}
                        
                        # Add embedding dimensions
                        for j in range(z_mu_np.shape[1]):
                            row[f"z_mu_{j:02d}"] = z_mu_np[i, j]
                        
                        exp_embeddings.append(row)
            
            if verbose:
                print(f"✅ Extracted {len(exp_embeddings)} embeddings for {exp}")
            all_embeddings.extend(exp_embeddings)
        
        # Create final dataframe
        latent_df = pd.DataFrame(all_embeddings)
        if verbose:
            print(f"✅ Total embeddings extracted: {len(latent_df)}")
        
        # Save embeddings per experiment
        save_root = data_root / "analysis" / "latent_embeddings" / model_class / model_name
        save_root.mkdir(parents=True, exist_ok=True)
        
        for exp in experiments:
            exp_df = latent_df[latent_df["experiment_date"] == exp]
            output_path = save_root / f"morph_latents_{exp}.csv"
            exp_df.to_csv(output_path, index=False)
            # Always print saved path (concise summary)
            print(f"Saved embeddings: {output_path}")
        
        if verbose:
            print("✅ Embedding generation completed successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Error in Python 3.9 subprocess: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
