#!/usr/bin/env python3
"""
Subprocess script to load legacy models in Python 3.9 environment.

This script is called from a subprocess when the main process is not running Python 3.9.
It loads the model using AutoModel.load_from_folder() and saves the state dict to a temp file.
"""

import argparse
import sys
import tempfile
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Load legacy model in Python 3.9 subprocess")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--output-path", required=True, help="Path to save model state")
    parser.add_argument("--device", default="cpu", help="Device to load model on")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    
    # Validate Python version
    if sys.version_info[:2] != (3, 9):
        print(f"ERROR: Expected Python 3.9, got {sys.version_info[0]}.{sys.version_info[1]}", file=sys.stderr)
        return 1
    
    # Validate model path
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}", file=sys.stderr)
        return 1
    
    try:
        # Import here to avoid issues in main process
        sys.path.insert(0, str(Path(__file__).parent))
        from src.vae.models.auto_model import AutoModel
        
        print(f"Loading model from {model_path}")
        lit_model = AutoModel.load_from_folder(str(model_path))

        # Move to specified device
        device = args.device
        lit_model.to(device)
        lit_model.eval()

        # Capture lightweight metadata needed downstream (e.g. MetricVAE splits)
        metadata = {
            'model_name': getattr(lit_model, 'model_name', None),
            'latent_dim': getattr(lit_model, 'latent_dim', None),
            'nuisance_indices': None,
        }
        nuisance_indices = getattr(lit_model, 'nuisance_indices', None)
        if nuisance_indices is not None:
            try:
                if hasattr(nuisance_indices, 'detach'):
                    nuisance_indices = nuisance_indices.detach()
                if hasattr(nuisance_indices, 'cpu'):
                    nuisance_indices = nuisance_indices.cpu()
                if hasattr(nuisance_indices, 'tolist'):
                    nuisance_indices = nuisance_indices.tolist()
                metadata['nuisance_indices'] = nuisance_indices
            except Exception:
                # Best-effort capture; downstream code will handle None gracefully
                metadata['nuisance_indices'] = None

        # Save the full model using torch.jit.script for better compatibility
        try:
            # Try to script the model for cross-version compatibility
            scripted_model = torch.jit.script(lit_model.eval())
            model_data = {
                'scripted_model': scripted_model,
                'model_type': 'scripted',
                'python_version': f"{sys.version_info[0]}.{sys.version_info[1]}",
                'device': str(device),
                'metadata': metadata,
            }
        except Exception as script_error:
            print(f"Warning: Could not script model ({script_error}), falling back to state dict")
            # Fallback to state dict approach
            model_data = {
                'model_state_dict': lit_model.state_dict(),
                'model_config': lit_model.model_config.__dict__ if hasattr(lit_model, 'model_config') else {},
                'model_class': lit_model.__class__.__name__,
                'model_type': 'state_dict',
                'python_version': f"{sys.version_info[0]}.{sys.version_info[1]}",
                'device': str(device),
                'metadata': metadata,
            }
        
        print(f"Saving model to {output_path}")
        torch.save(model_data, output_path, pickle_protocol=2)  # Use older protocol for compatibility
        
        print("✅ Model loaded and saved successfully")
        return 0
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
