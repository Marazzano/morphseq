#!/usr/bin/env python3
"""
Test script to regenerate metadata with fixed timestamp extraction
Usage: python test_metadata_regeneration.py <experiment_name>
"""
import sys
import pandas as pd
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from build.build01B_compile_yx1_images_torch import build_ff_from_yx1

def test_metadata_regeneration(exp_name: str):
    """Test metadata regeneration with fixed timestamps"""

    data_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground"
    repo_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/"

    metadata_file = Path(data_root) / "metadata" / "built_metadata_files" / f"{exp_name}_metadata.csv"
    backup_file = metadata_file.with_suffix(".csv.backup")

    print(f"🧪 Testing metadata regeneration for {exp_name}")
    print(f"📁 Metadata file: {metadata_file}")

    # Backup existing metadata if it exists
    if metadata_file.exists():
        print(f"💾 Backing up existing metadata to {backup_file}")
        shutil.copy2(metadata_file, backup_file)

        # Load original for comparison
        df_original = pd.read_csv(metadata_file)
        print(f"\n📊 Original metadata (first 5 rows):")
        print(f"  Time (s): {df_original['Time (s)'].head().tolist()}")
        print(f"  Time Rel (s): {df_original['Time Rel (s)'].head().tolist()}")
        print(f"  Intervals: {df_original['Time (s)'].diff().dropna().head().tolist()}")

        if 'predicted_stage_hpf' in df_original.columns:
            print(f"  Predicted stage (hpf): {df_original['predicted_stage_hpf'].head().tolist()}")
    else:
        print("ℹ️  No existing metadata found")
        df_original = None

    print(f"\n🔄 Regenerating metadata with fixed timestamp extraction...")

    # Regenerate metadata using our fixed function
    try:
        build_ff_from_yx1(
            data_root=data_root,
            repo_root=repo_root,
            exp_name=exp_name,
            metadata_only=True,
            overwrite=True
        )

        # Load new metadata for comparison
        df_new = pd.read_csv(metadata_file)
        print(f"\n✅ New metadata generated successfully!")
        print(f"📊 New metadata (first 5 rows):")
        print(f"  Time (s): {df_new['Time (s)'].head().tolist()}")
        print(f"  Time Rel (s): {df_new['Time Rel (s)'].head().tolist()}")
        print(f"  Intervals: {df_new['Time (s)'].diff().dropna().head().tolist()}")

        if 'predicted_stage_hpf' in df_new.columns:
            print(f"  Predicted stage (hpf): {df_new['predicted_stage_hpf'].head().tolist()}")

        # Compare if we have original
        if df_original is not None:
            print(f"\n🔍 Comparison:")
            orig_intervals = df_original['Time (s)'].diff().dropna()
            new_intervals = df_new['Time (s)'].diff().dropna()

            print(f"  Original median interval: {orig_intervals.median():.2f}s ({orig_intervals.median()/60:.2f}min)")
            print(f"  New median interval: {new_intervals.median():.2f}s ({new_intervals.median()/60:.2f}min)")

            orig_duration = df_original['Time (s)'].max() - df_original['Time (s)'].min()
            new_duration = df_new['Time (s)'].max() - df_new['Time (s)'].min()

            print(f"  Original total duration: {orig_duration:.1f}s ({orig_duration/3600:.2f}h)")
            print(f"  New total duration: {new_duration:.1f}s ({new_duration/3600:.2f}h)")

            if 'predicted_stage_hpf' in df_original.columns and 'predicted_stage_hpf' in df_new.columns:
                orig_stages = df_original['predicted_stage_hpf'].head()
                new_stages = df_new['predicted_stage_hpf'].head()
                print(f"  Original stages: {orig_stages.tolist()}")
                print(f"  New stages: {new_stages.tolist()}")

        print(f"\n💾 Metadata saved to: {metadata_file}")
        if backup_file.exists():
            print(f"💾 Original backup at: {backup_file}")

    except Exception as e:
        print(f"❌ Error during regeneration: {e}")
        import traceback
        traceback.print_exc()

        # Restore backup if regeneration failed
        if backup_file.exists():
            print(f"🔄 Restoring backup...")
            shutil.copy2(backup_file, metadata_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_metadata_regeneration.py <experiment_name>")
        print("Example: python test_metadata_regeneration.py 20250519")
        sys.exit(1)

    exp_name = sys.argv[1]
    test_metadata_regeneration(exp_name)