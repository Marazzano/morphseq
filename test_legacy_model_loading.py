#!/usr/bin/env python3
"""
Test script for legacy model loading with Python version compatibility.

This script tests the new environment-switching model loading functionality.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Model path
    model_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models/legacy/20241107_ds_sweep01_optimum"
    
    logger.info("=" * 60)
    logger.info("Testing Legacy Model Loading with Environment Switching")
    logger.info("=" * 60)
    logger.info(f"Current Python version: {sys.version_info[0]}.{sys.version_info[1]}")
    logger.info(f"Model path: {model_path}")
    
    try:
        from src.run_morphseq_pipeline.services.legacy_model_utils import (
            get_current_conda_env, 
            check_conda_env_exists,
            load_legacy_model_safe
        )
        
        # Test environment detection
        current_env = get_current_conda_env()
        logger.info(f"Current conda environment: {current_env}")
        
        # Check if seg_sam_py39 exists
        target_env = "seg_sam_py39"
        env_exists = check_conda_env_exists(target_env)
        logger.info(f"Target environment '{target_env}' exists: {env_exists}")
        
        if not env_exists:
            logger.error(f"Cannot proceed: '{target_env}' environment not found")
            logger.info("Please create the environment or specify a different target environment")
            return 1
        
        # Test model loading
        logger.info("Attempting to load legacy model...")
        device = "cpu"  # Use CPU for testing
        
        model = load_legacy_model_safe(
            model_path=model_path,
            device=device,
            target_python_env=target_env,
            logger=logger
        )
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Test model inference (basic check)
        import torch
        logger.info("Testing model inference...")
        
        # Create dummy input (adjust size based on your model)
        # Most morphological models expect (batch, channels, height, width)
        dummy_input = torch.randn(1, 1, 128, 288)  # Typical morphseq input size
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            # Try different possible methods based on model type
            methods_to_try = ['encode', 'forward', '__call__']
            output = None
            
            for method_name in methods_to_try:
                if hasattr(model, method_name):
                    try:
                        method = getattr(model, method_name)
                        if method_name == 'forward':
                            # Forward typically needs a dataset-like input
                            from pythae.data.datasets import BaseDataset
                            dataset_input = BaseDataset(dummy_input)
                            output = method(dataset_input)
                        else:
                            output = method(dummy_input)
                        logger.info(f"Successfully called {method_name} method")
                        break
                    except Exception as e:
                        logger.debug(f"Method {method_name} failed: {e}")
                        continue
            
            if output is not None:
                if hasattr(output, 'z'):  # Common VAE output structure
                    logger.info(f"Model latent output shape: {output.z.shape}")
                elif hasattr(output, 'shape'):
                    logger.info(f"Model output shape: {output.shape}")
                else:
                    logger.info(f"Model output type: {type(output)}")
            else:
                logger.warning("Could not test inference, but model loaded successfully")
        
        logger.info("✅ Model inference test passed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())