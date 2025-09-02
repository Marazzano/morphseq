#!/usr/bin/env python3
"""
Simple test script for the new Build06 implementation.

Usage:
    python test_build06_simple.py --root /path/to/your/morphseq/project --data-root /path/to/data/root --dry-run

This will test the new CLI interface without actually running embedding generation.
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test the new Build06 CLI")
    parser.add_argument("--root", required=True, help="Path to morphseq project root")
    parser.add_argument("--data-root", help="Path to data root (defaults to MORPHSEQ_DATA_ROOT)")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    
    args = parser.parse_args()
    
    # Set up environment
    if args.data_root:
        os.environ["MORPHSEQ_DATA_ROOT"] = args.data_root
    
    # Build the command
    cmd_args = [
        "build06",
        "--root", args.root,
        "--model-name", "20241107_ds_sweep01_optimum",
    ]
    
    if args.data_root:
        cmd_args.extend(["--data-root", args.data_root])
    
    if args.dry_run:
        cmd_args.append("--dry-run")
    
    print("Testing Build06 with command:")
    print(f"python -m src.run_morphseq_pipeline.cli {' '.join(cmd_args[1:])}")
    print()
    
    # Import and run the CLI
    try:
        from src.run_morphseq_pipeline.cli import main as cli_main
        result = cli_main(cmd_args)
        
        if result == 0:
            print("✅ Build06 test completed successfully!")
        else:
            print(f"❌ Build06 test failed with exit code: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running Build06: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())