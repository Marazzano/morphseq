#!/usr/bin/env python3
"""
Join z_mu morphological embeddings with training metadata
Created: August 31, 2025
Purpose: Combine VAE embeddings with biological metadata on snip_id level
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def join_embeddings_metadata(training_data_path, output_path=None):
    """Join z_mu embeddings with training metadata on snip_id"""
    
    training_data_path = Path(training_data_path)
    
    # Load embeddings
    embeddings_path = training_data_path / "embeddings.csv"
    metadata_path = training_data_path / "embryo_metadata_df_train.csv"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print(f"📊 Loading embeddings from: {embeddings_path}")
    embeddings_df = pd.read_csv(embeddings_path)
    
    print(f"📊 Loading metadata from: {metadata_path}")  
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"   • Embeddings: {len(embeddings_df)} rows, {len(embeddings_df.columns)} columns")
    print(f"   • Metadata: {len(metadata_df)} rows, {len(metadata_df.columns)} columns")
    
    # Join on snip_id
    print("🔗 Joining on snip_id...")
    joined_df = pd.merge(metadata_df, embeddings_df, on='snip_id', how='inner')
    
    print(f"✅ Join complete: {len(joined_df)} rows, {len(joined_df.columns)} columns")
    
    # Verify z_mu columns are present
    z_mu_cols = [col for col in joined_df.columns if col.startswith('z_mu_')]
    print(f"🧬 Morphological embedding dimensions: {len(z_mu_cols)} (z_mu_00 to z_mu_{len(z_mu_cols)-1:02d})")
    
    # Key biological columns to verify
    key_bio_cols = ['snip_id', 'embryo_id', 'experiment_date', 'master_perturbation', 
                   'predicted_stage_hpf', 'surface_area_um', 'phenotype', 'well']
    present_bio_cols = [col for col in key_bio_cols if col in joined_df.columns]
    print(f"🔬 Key biological metadata: {len(present_bio_cols)}/{len(key_bio_cols)} columns present")
    
    # Save joined dataset
    if output_path is None:
        output_path = training_data_path / "embryo_metadata_with_embeddings.csv"
    else:
        output_path = Path(output_path)
    
    joined_df.to_csv(output_path, index=False)
    print(f"💾 Complete dataset saved: {output_path}")
    
    # Summary stats
    print("\n📋 Dataset Summary:")
    print(f"   • Total embryos: {len(joined_df)}")
    print(f"   • Experiments: {joined_df['experiment_date'].nunique() if 'experiment_date' in joined_df.columns else 'N/A'}")
    print(f"   • Perturbations: {joined_df['master_perturbation'].unique().tolist() if 'master_perturbation' in joined_df.columns else 'N/A'}")
    print(f"   • Wells: {joined_df['well'].unique().tolist() if 'well' in joined_df.columns else 'N/A'}")
    print(f"   • Stage range: {joined_df['predicted_stage_hpf'].min():.2f}-{joined_df['predicted_stage_hpf'].max():.2f} hpf" if 'predicted_stage_hpf' in joined_df.columns else "")
    
    # Sample of z_mu values
    if z_mu_cols:
        print(f"\n🧬 Sample z_mu values (first embryo):")
        sample_embeddings = joined_df[z_mu_cols].iloc[0]
        print(f"   {sample_embeddings.to_dict()}")
    
    return joined_df, output_path

def main():
    parser = argparse.ArgumentParser(description="Join z_mu embeddings with training metadata")
    parser.add_argument("--training-data", required=True, help="Path to training data directory")
    parser.add_argument("--output", help="Output CSV path (optional)")
    
    args = parser.parse_args()
    
    print("🔗 Joining morphological embeddings with metadata")
    print("=" * 60)
    
    joined_df, output_path = join_embeddings_metadata(args.training_data, args.output)
    
    print(f"\n✅ SUCCESS: Complete morphological embedding dataset ready!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Format: {len(joined_df)} embryos × {len(joined_df.columns)} features")

if __name__ == "__main__":
    main()