from __future__ import annotations
from pathlib import Path

from src.build.build02B_segment_bf_main import apply_unet


def run_build02(
    root: str | Path,
    mode: str = "legacy",
    model_name: str = "mask_v1_0050",  # Legacy parameter, ignored in new mode
    n_classes: int = 2,  # Legacy parameter, ignored in new mode
    num_workers: int = 1,  # Number of DataLoader workers (use 1 by default to avoid huge worker spawn)
    overwrite: bool = False,
) -> None:
    """Run complete QC segmentation suite or skip if using SAM2.
    
    When mode='legacy', runs all 5 UNet models for complete QC mask generation:
    - mask_v0_0100: embryo masks (2 classes)
    - yolk_v1_0050: yolk masks (1 class)  
    - focus_v0_0100: focus masks (1 class)
    - bubble_v0_0100: bubble masks (1 class)
    - via_v1_0100: viability masks (1 class)

    Writes masks under `segmentation/{model}_predictions/{exp}/`.
    """
    if mode == "skip":
        print("ℹ️  build02: skipping legacy segmentation (SAM2 path expected)")
        return

    # Normalize and resolve data root to an absolute path to avoid accidental double-prefixing
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    print(f"ℹ️  build02: using data root: {root}")

    # Ensure at least one worker; 0 or negative values are treated as 1 to avoid multiprocessing explosions
    if num_workers is None or num_workers < 1:
        num_workers = 1
    
    # Define all 5 UNet models for complete QC mask suite
    models = [
        ("mask_v0_0100", 1),      # embryo masks  
        ("yolk_v1_0050", 1),      # yolk masks
        ("focus_v0_0100", 1),     # focus masks
        ("bubble_v0_0100", 1),    # bubble masks
        ("via_v1_0100", 1)        # viability masks
    ]
    
    print("🚀 Running Build02 complete QC segmentation suite (5 UNet models):")
    
    failed_models = []
    
    for model_name, n_classes in models:
        try:
            print(f"🎭 Processing {model_name} ({n_classes} classes)...")
            apply_unet(
                root=str(root), 
                model_name=model_name, 
                n_classes=n_classes, 
                overwrite_flag=overwrite,
                n_workers=num_workers
            )
            print(f"✅ {model_name} completed successfully")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            failed_models.append((model_name, str(e)))
            continue
    
    # Summary
    successful_models = len(models) - len(failed_models)
    print(f"\n📊 Build02 Results: {successful_models}/{len(models)} models completed")
    
    if failed_models:
        print("❌ Failed models:")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error}")
        
        # Decide whether to fail completely or continue  
        if successful_models == 0:
            raise RuntimeError("All Build02 models failed")
        else:
            print("⚠️  Continuing with partial success - some QC functionality may be degraded")
    
    print("✔️  Build02 (complete QC segmentation suite) finished.")
