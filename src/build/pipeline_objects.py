"""
MorphSeq Pipeline Objects: Intelligent Experiment Management and Orchestration

This module provides the core classes for managing MorphSeq experiments with intelligent
tracking, state management, and pipeline orchestration. It enables seamless coordination
between per-experiment processing and global cohort-level operations.

Architecture Overview:
===================

Individual Experiments (Experiment class):
├── Raw data acquisition & FF image creation (Build01)
├── QC mask generation (Build02) OR SAM2 segmentation 
├── Embryo processing & df01 contribution (Build03)
└── Latent embedding generation (per-experiment)

Global Operations (ExperimentManager class):
├── df01 → QC & staging → df02 (Build04)
└── df02 + latents → final dataset → df03 (Build06)

Key Features:
============
✓ **Intelligent State Tracking**: JSON-based state files with timestamp comparison
✓ **Automatic State Sync**: Detects existing work from previous runs
✓ **Dependency Management**: Ensures correct execution order and prerequisites
✓ **Duplicate Prevention**: Avoids reprocessing data already in downstream files
✓ **Flexible Orchestration**: Supports individual steps or full end-to-end workflows

Classes:
========
- Experiment: Manages individual experiment lifecycle and per-experiment operations
- ExperimentManager: Orchestrates multiple experiments and global operations

Author: Claude Code with MorphSeq Team
Stage: 3 - Intelligent Pipeline Orchestration (Complete)
"""

from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust "2" if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01A_compile_keyence_torch import stitch_ff_from_keyence, build_ff_from_keyence
from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1
import pandas as pd
import multiprocessing
from typing import Literal, Optional, Dict, List, Sequence, Union
import torch
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import glob2 as glob
import functools
import os
import logging
import itertools
from src.build.export_utils import PATTERNS, _match_files, has_output, newest_mtime, _mod_time
from src.build.build03A_process_images import segment_wells, compile_embryo_stats, extract_embryo_snips
from src.build.build02B_segment_bf_main import apply_unet

# Dependency simplification notes (comments only; no behavior change):
# - glob2: used only to match files; replace with `Path.glob()` to drop glob2.
# - pandas in this file is used primarily for IO; could be minimized if needed.
# - Consider lazy imports for heavyweight modules in methods that need them (e.g. stitch/segment).


log = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)



def record(step: str):
    """
    Decorator that only marks `step` if the wrapped method
    completes without throwing.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            try:
                result = fn(self, *args, **kwargs)
            except Exception:
                # the exception propagates, and we do *not* record the step
                raise
            else:
                # only record on success
                self.record_step(step)
                return result
        return wrapped
    return decorator


@dataclass
class Experiment:
    """
    Represents a single MorphSeq experiment with intelligent pipeline tracking.
    
    This class manages the complete lifecycle of a MorphSeq experiment, tracking
    the state of each pipeline step and determining what work needs to be done.
    
    Pipeline Flow:
    1. Raw data → FF images (Build01)
    2. FF images → QC masks (Build02) 
    3. FF images → SAM2 segmentation (SAM2)
    4. SAM2/QC masks → embryo processing (Build03) → contributes to df01
    5. Individual processing → latent embeddings (per-experiment)
    6. df01 → df02 (Build04, global QC)
    7. df02 + latents → df03 (Build06, global merge)
    
    State Management:
    - Tracks completion via JSON state files in metadata/experiments/
    - Uses file timestamps to detect when inputs are newer than outputs
    - Automatically syncs with existing combined metadata files (df01/df02/df03)
    - Avoids duplicate processing by checking downstream file inclusion
    
    Attributes:
        date: Experiment date identifier (e.g., "20250529_30hpf_ctrl_atf6")
        data_root: Path to MorphSeq data directory
        n_workers: Number of CPU workers for processing (auto-calculated if not set)
        flags: Dict tracking which pipeline steps have completed
        timestamps: Dict tracking when each step was last run
        repo_root: Path to MorphSeq repository
    """
    date: str
    data_root: Union[Path, str]
    n_workers: int = 1
    flags:      Dict[str,bool] = field(init=False)
    timestamps: Dict[str,str]  = field(init=False)
    repo_root:  Path = field(init=False)

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.flags = {}
        self.timestamps = {}

        # Determine repo path relative to this script
        script_path = Path(__file__).resolve()
        self.repo_root = script_path.parents[2]

        self._load_state()
        self._sync_with_disk()

    # # ——— public API ——————————————————————————————————————————————————
    def record_step(self, step: str):
        """Mark `step` done right now and save state."""
        now = datetime.utcnow().isoformat()
        self.flags[step] = True
        self.timestamps[step] = now
        self._save_state()

    # ——— timestamp helpers ————————————————————————————————————————————
    def _ts(self, key: str, default: float = 0.0) -> float:
        """
        Safely return a float UNIX timestamp for a recorded step.
        Accepts either float seconds or ISO-8601 string in the state file.
        """
        val = self.timestamps.get(key, None)
        if val is None:
            return default
        try:
            # common case: already a float (mtime)
            if isinstance(val, (int, float)):
                return float(val)
            # maybe an ISO string
            return datetime.fromisoformat(str(val)).timestamp()
        except Exception:
            return default

    def _safe_mtime_compare(self, file_path: Path, timestamp_key: str) -> bool:
        """Return True if file is newer than recorded timestamp for key."""
        try:
            if not file_path or not file_path.exists():
                return False
            file_mtime = file_path.stat().st_mtime
            last_run = self._ts(timestamp_key)
            return file_mtime > last_run
        except Exception:
            return False

    # ——— path properties ———————————————————————————————————————————————
    @property
    def num_cpu_workers(self, prefactor: float = 0.25, min_workers: int = 1, max_workers: int = 24) -> int:
        """
        Returns a recommended number of CPU workers.
        By default uses half of all logical cores (but at least min_workers).
        You can tune `prefactor` between 0 and 1.
        """
        total = os.cpu_count() or 1
        n = min(max(min_workers, int(total * prefactor)), max_workers)
        return n
    
    @property
    def has_gpu(self) -> bool:
        """
        Returns True if PyTorch can see at least one CUDA device.
        Falls back to False if torch isn’t installed.
        """
        return torch.cuda.is_available()
    
    @property
    def gpu_names(self) -> List[str]:
        """
        Returns a list of device names, e.g. ['Tesla V100', ...].
        Empty if no GPUs or torch unavailable.
        """
        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    @property
    def microscope(self) -> Optional[Literal["Keyence","YX1"]]:
        key = self.data_root/"raw_image_data"/"Keyence"/self.date
        yx  = self.data_root/"raw_image_data"/"YX1"   /self.date
        if   key.exists() and not yx.exists(): return "Keyence"
        elif yx.exists() and not key.exists(): return "YX1"
        elif key.exists() and yx.exists():
            raise RuntimeError(f"Ambiguous raw data for {self.date}")
        else:
            return None

    @property
    def raw_path(self) -> Optional[Path]:
        m = self.microscope
        return (self.data_root/"raw_image_data"/m/self.date) if m else None
    
    @property
    def meta_path(self) -> Optional[Path]:
        p = self.repo_root/"metadata"/"plate_metadata"/ f"{self.date}_well_metadata.xlsx"
        return p if p.exists() else None
    
    @property
    def meta_path_built(self) -> Optional[Path]:
        p = self.data_root/"metadata"/"built_metadata_files"/ f"{self.date}_metadata.csv"
        return p if p.exists() else None
    
    @property
    def meta_path_embryo(self) -> Optional[Path]:
        p = self.data_root/"metadata"/"embryo_metadata_files"/ f"{self.date}_embryo_metadata.csv"
        return p if p.exists() else None
    
    @property
    def snip_path(self) -> Optional[Path]:
        p = self.data_root/"training_data"/"bf_embryo_snips"/ self.date
        return p if p.exists() else None

    @property
    def ff_path(self) -> Optional[Path]:
        if self.microscope=="Keyence":
            p = self.data_root/"built_image_data"/"Keyence"/"FF_images"/self.date
        else:
            p = self.data_root/"built_image_data"/"stitched_FF_images"/self.date
        return p if p.exists() else None

    @property
    def stitch_ff_path(self) -> Optional[Path]:
        if self.microscope=="Keyence":
            p = self.data_root/"built_image_data"/"stitched_FF_images"/self.date
        else:
            p = self.raw_path
        return p if p and p.exists() else None
    
    @property
    def stitch_z_path(self) -> Optional[Path]:
        if self.microscope=="Keyence":
            p = self.data_root/"built_image_data"/"Keyence_stitched_z"/self.date
        else:
            p = self.raw_path
        return p if p and p.exists() else None

    @property
    def mask_path(self) -> Optional[Path]:
        seg_root = self.data_root/"segmentation"
        masks = [d for d in seg_root.glob("mask*") if d.is_dir()]
        if not masks: return None
        candidate = masks[0]/self.date
        return candidate if candidate.exists() else None

    # ——— new per-experiment tracking properties ————————————————
    @property
    def sam2_csv_path(self) -> Path:
        """Expected per-experiment SAM2 metadata CSV path."""
        try:
            return self.data_root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{self.date}.csv"
        except Exception:
            # Fall back to simple join; avoids raising in status views
            return Path(str(self.data_root)) / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{self.date}.csv"

    def qc_mask_status(self) -> tuple[int, int]:
        """Return (present_count, total_count) across the 5 QC mask model outputs."""
        mask_types = [
            "mask_v0_0100",
            "yolk_v1_0050",
            "focus_v0_0100",
            "bubble_v0_0100",
            "via_v1_0100",
        ]
        present = 0
        try:
            seg_root = self.data_root / "segmentation"
            for mt in mask_types:
                if (seg_root / f"{mt}_predictions" / self.date).exists():
                    present += 1
        except Exception:
            # treat as zero present on errors
            present = 0
        return present, len(mask_types)

    @property
    def has_all_qc_masks(self) -> bool:
        p, t = self.qc_mask_status()
        return p == t and t > 0

    def get_latent_path(self, model_name: str) -> Path:
        try:
            return (
                self.data_root
                / "analysis"
                / "latent_embeddings"
                / "legacy"
                / model_name
                / f"morph_latents_{self.date}.csv"
            )
        except Exception:
            return Path(str(self.data_root)) / "analysis" / "latent_embeddings" / "legacy" / model_name / f"morph_latents_{self.date}.csv"

    def has_latents(self, model_name: str = "20241107_ds_sweep01_optimum") -> bool:
        try:
            return self.get_latent_path(model_name).exists()
        except Exception:
            return False

    # ——— Stage 3: Downstream file tracking ————————————————————————————
    
    def is_in_df01(self) -> bool:
        """
        Check if this experiment exists in the combined embryo metadata (df01).
        
        df01 is created by Build03 and contains processed embryo data from all experiments.
        If an experiment is in df01, it means Build03 has already been run for it.
        
        Returns:
            bool: True if experiment_date appears in df01.csv
        """
        try:
            import pandas as pd
            df01_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"
            if not df01_path.exists():
                return False
            df01 = pd.read_csv(df01_path)
            return self.date in df01['experiment_date'].values if 'experiment_date' in df01.columns else False
        except Exception:
            return False

    def is_in_df02(self) -> bool:
        """
        Check if this experiment exists in the QC'd embryo metadata (df02).
        
        df02 is created by Build04 and contains QC'd + staged embryo data.
        If an experiment is in df02, it has passed through global QC processing.
        
        Returns:
            bool: True if experiment_date appears in df02.csv
        """
        try:
            import pandas as pd
            df02_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"
            if not df02_path.exists():
                return False
            df02 = pd.read_csv(df02_path)
            return self.date in df02['experiment_date'].values if 'experiment_date' in df02.columns else False
        except Exception:
            return False

    def is_in_df03(self) -> bool:
        """
        Check if this experiment exists in the final dataset (df03).
        
        df03 is created by Build06 and contains df02 data merged with latent embeddings.
        This is the final, analysis-ready dataset.
        
        Returns:
            bool: True if experiment_date appears in df03.csv
        """
        try:
            import pandas as pd
            df03_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"
            if not df03_path.exists():
                return False
            df03 = pd.read_csv(df03_path)
            return self.date in df03['experiment_date'].values if 'experiment_date' in df03.columns else False
        except Exception:
            return False

    def needs_build06_merge(self, model_name: str = "20241107_ds_sweep01_optimum") -> bool:
        """
        Determine if THIS specific experiment needs to be merged in Build06.
        
        This is the most precise check for Build06 requirements. An experiment needs
        merging if its latent embeddings are newer than the current df03 file.
        
        Logic:
        - If no df03 exists → needs merge (if has latents)
        - If latent embeddings are newer than df03 → needs merge  
        - If latent embeddings are older than df03 → already merged
        
        This avoids unnecessary rebuilds when experiments are already current in df03.
        
        Args:
            model_name: Model name for latent embeddings (default: latest model)
            
        Returns:
            bool: True if this experiment's latents need merging into df03
        """
        try:
            df03_path = self.data_root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"
            
            # If no df03 exists, needs merge if we have latents
            if not df03_path.exists():
                return self.has_latents(model_name)
            
            # Key insight: Compare latent timestamp vs df03 timestamp
            if self.has_latents(model_name):
                latent_path = self.get_latent_path(model_name)
                df03_time = df03_path.stat().st_mtime
                latent_time = latent_path.stat().st_mtime
                return latent_time > df03_time  # Latents newer = needs merge
                
            return False
        except Exception:
            return False

    # ——— Stage 3: Pipeline step requirements ————————————————————————

    @property
    def needs_sam2(self) -> bool:
        """
        Determine if SAM2 segmentation needs to run for this experiment.
        
        SAM2 is the modern embryo segmentation approach that produces superior
        segmentation compared to legacy Build02 QC masks. 
        
        Logic: SAM2 is needed if the SAM2 metadata CSV doesn't exist yet.
        
        Returns:
            bool: True if SAM2 segmentation needs to run
        """
        try:
            return not self.sam2_csv_path.exists()
        except Exception:
            return False

    @property
    def state_file(self) -> Path:
        p = self.data_root/"metadata"/"experiments"/f"{self.date}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def needs_export(self) -> bool:
        # last_run = self.timestamps.get("export", 0)
        # newest   = newest_mtime(self.raw_path, PATTERNS["raw"])
        return not self.flags["ff"]

    # @property
    # def needs_build_metadata(self) -> bool:
    #     return (_mod_time(self.ff_path) >
    #             datetime.fromisoformat(self.timestamps.get("metadata", "1970-01-01T00:00:00")).timestamp())

    @property
    def needs_stitch(self) -> bool:
        last_run = self.timestamps.get("stitch", 0)
        newest   = newest_mtime(self.raw_path, PATTERNS["raw"])
        return newest >= last_run

    @property
    def needs_stitch_z(self) -> bool:
        last_run = self.timestamps.get("stitch_z", 0)
        newest   = newest_mtime(self.raw_path, PATTERNS["raw"])
        return newest >= last_run

    @property
    def needs_segment(self) -> bool:
        return (_mod_time(self.mask_path) >
                datetime.fromisoformat(self.timestamps.get("segment", "1970-01-01T00:00:00")).timestamp())
    
    @property
    def needs_stats(self) -> bool:
        last_run = newest_mtime(self.mask_path, PATTERNS["snips"])
        newest   = newest_mtime(self.snip_path, PATTERNS["segment"])
        return newest >= last_run

    @property
    def needs_build03(self) -> bool:
        """
        Determine if Build03 embryo processing needs to run for this experiment.
        
        Build03 processes embryo segmentation data (from SAM2 or QC masks) and 
        appends the results to the global df01.csv file. This is the bridge between
        per-experiment segmentation and global embryo datasets.
        
        Multi-layered Logic:
        1. If experiment already exists in df01 → auto-sync local state, return False
        2. If recorded locally as complete → check if inputs are newer than last run
        3. If never run locally → check if segmentation data is available
        
        Segmentation Sources (in order of preference):
        - SAM2 CSV (modern, superior segmentation)
        - Build02 QC masks (legacy, 5 UNet models)
        
        State Sync:
        Automatically updates local JSON state if experiment found in df01 but not
        tracked locally (handles legacy processing or external pipeline runs).
        
        Returns:
            bool: True if Build03 needs to run for this experiment
        """
        try:
            # PRIORITY 1: Check if already processed (exists in df01)
            if self.is_in_df01():
                # Auto-sync: Update local state if missing timestamp
                if "build03" not in self.timestamps:
                    self.flags['build03'] = True
                    self.flags['contributed_to_df01'] = True  
                    self.timestamps['build03'] = datetime.utcnow().isoformat()
                    self._save_state()
                return False  # Already processed globally
            
            # PRIORITY 2: If locally recorded as complete, check input freshness
            last_build03 = self._ts("build03", 0)
            if last_build03 > 0:
                # Check if SAM2 output is newer than last Build03 run
                if self.sam2_csv_path.exists():
                    sam2_time = self.sam2_csv_path.stat().st_mtime
                    if sam2_time > last_build03:
                        return True  # SAM2 data updated, needs reprocessing
                # Note: Could also check QC mask timestamps here if needed
                return False  # Local tracking shows complete and current
            
            # PRIORITY 3: Never run locally, check if segmentation data available
            has_sam2_data = self.sam2_csv_path.exists()
            has_qc_masks = self.has_all_qc_masks
            
            return has_sam2_data or has_qc_masks  # Can run if either source available
            
        except Exception:
            return False

    # ——— internal sync logic —————————————————————————————————————————————
    def _sync_with_disk(self) -> None:
        """Called once from __post_init__ to refresh flags + timestamps."""
        changed = False
        for step, path in {
            "raw"    : self.raw_path,
            "meta"   : self.meta_path,
            "meta_built" : self.meta_path_built,
            "meta_embryo": self.meta_path_embryo,
            "ff"     : self.ff_path,
            "stitch" : self.stitch_ff_path,
            "stitch_z" : self.stitch_z_path,
            "segment": self.mask_path,
            "snips": self.snip_path,
            }.items():
            
            if step not in ["meta", "meta_built", "meta_embryo"]:
                present  = has_output(path, PATTERNS[step])
            else:
                present = path is not None

            previous = self.flags.get(step, None)

            # flag housekeeping ---------------------------------------------------
            if present != previous:
                self.flags[step] = present
                changed = True

            # timestamp housekeeping ---------------------------------------------
            if present:
                if step not in ["meta", "meta_built", "meta_embryo"]:
                    mt = newest_mtime(path, PATTERNS[step])
                else:
                    mt = path.stat().st_mtime
                self.timestamps[step] = mt
            else:
                self.timestamps.pop(step, None)

        if changed:
            self._save_state()


    # ——— call pipeline functions —————————————————————————————————————————————
    @record("ff")
    def export_images(self):
        if self.microscope == "Keyence":
            build_ff_from_keyence(data_root=self.data_root, repo_root=self.repo_root, exp_name=self.date, overwrite=True)
        else:
            build_ff_from_yx1(data_root=self.data_root, repo_root=self.repo_root, exp_name=self.date, overwrite=True)

    @record("meta_built")
    def export_metadata(self):
        if self.microscope == "Keyence":
            build_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, metadata_only=True)
        else:
            build_ff_from_yx1(data_root=self.data_root, exp_name=self.date, overwrite=True, metadata_only=True)


    @record("stitch")
    def stitch_images(self):
        if self.microscope == "Keyence":
            stitch_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
            stitch_z_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
        else:
            pass

    # @record("stitch")
    def stitch_z_images(self):
        if self.microscope == "Keyence":
            # stitch_ff_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
            stitch_z_from_keyence(data_root=self.data_root, exp_name=self.date, overwrite=True, n_workers=self.num_cpu_workers)
        else:
            pass

    @record("segment")
    def segment_images(self, force_update: bool=False):
        # We need to pull the current models
        model_name_vec = ["mask_v0_0100", "via_v1_0100", "yolk_v1_0050", "focus_v0_0100", "bubble_v0_0100"] 
        # apply unet for each model
        for model_name in model_name_vec:
            apply_unet(
                root=self.data_root,
                model_name=model_name,
                n_classes=1,
                checkpoint_path=None,  # use the latest checkpoint
                n_workers=self.num_cpu_workers,
                overwrite_flag=force_update,
                make_sample_figures=True,
                n_sample_figures=100
            )

    # @record()
    def process_image_masks(self, force_update: bool=False):
        tracked_df = segment_wells(root=self.data_root, exp_name=self.date)
        stats_df = compile_embryo_stats(root=self.data_root, tracked_df=tracked_df)
        extract_embryo_snips(root=self.data_root, stats_df=stats_df, overwrite_flag=force_update)

    # ——— Stage 3: Orchestration execution methods ——————————————————————

    @record("sam2")
    def run_sam2(self, workers: int = 8, **kwargs):
        """Execute SAM2 segmentation for this experiment"""
        # Import here to avoid circular dependencies
        from ..run_morphseq_pipeline.steps.run_sam2 import run_sam2
        print(f"🎯 Running SAM2 for {self.date}")
        result = run_sam2(
            root=str(self.data_root), 
            exp=self.date, 
            workers=workers, 
            **kwargs
        )
        return result

    @record("build03")
    def run_build03(self, by_embryo: int = None, frames_per_embryo: int = None, **kwargs):
        """Execute Build03 for this experiment with SAM2/legacy detection"""
        print(f"🔬 Running Build03 for {self.date}")
        
        # Import here to avoid circular dependencies
        from ..run_morphseq_pipeline.steps.run_build03 import run_build03 as run_build03_step
        
        # Determine which path to use
        sam2_csv = None
        if self.sam2_csv_path.exists():
            print(f"  Using SAM2 masks from {self.sam2_csv_path}")
            sam2_csv = str(self.sam2_csv_path)
        else:
            print(f"  Using legacy Build02 masks")
            # Check if legacy masks exist
            if not self.has_all_qc_masks:
                raise RuntimeError(f"No SAM2 CSV and missing QC masks for {self.date}")
        
        # Call the actual Build03 function with proper parameters
        try:
            result = run_build03_step(
                root=str(self.data_root),
                exp=self.date,
                sam2_csv=sam2_csv,  # Will be None for legacy path
                by_embryo=by_embryo,
                frames_per_embryo=frames_per_embryo,
                n_workers=kwargs.get('n_workers', self.num_cpu_workers),
                df01_out=kwargs.get('df01_out', None)  # Use default if not specified
            )
            
            # Update df01 contribution tracking
            if result:
                self.flags['contributed_to_df01'] = True
                self.timestamps['last_df01_contribution'] = datetime.utcnow().isoformat()
                
            return result
            
        except Exception as e:
            print(f"  ❌ Build03 failed: {e}")
            raise

    @record("latents")
    def generate_latents(self, model_name: str = "20241107_ds_sweep01_optimum", **kwargs):
        """Generate latent embeddings for this experiment"""
        from ..analyze.gen_embeddings import ensure_embeddings_for_experiments
        print(f"🧬 Generating latents for {self.date}")
        success = ensure_embeddings_for_experiments(
            data_root=str(self.data_root),
            experiments=[self.date],
            model_name=model_name,
            **kwargs
        )
        return success

    # ——— load/save ——————————————————————————————————————————————————

    def _load_state(self):
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.flags      = data.get("flags", {})
            self.timestamps = data.get("timestamps", {})

    def _save_state(self):
        payload = {"flags": self.flags, "timestamps": self.timestamps}
        self.state_file.write_text(json.dumps(payload, indent=2))




class ExperimentManager:
    """
    Orchestrates multiple MorphSeq experiments and manages global pipeline operations.
    
    The ExperimentManager provides intelligent coordination between individual experiments
    and global processing steps. It handles both per-experiment operations and cohort-level
    data processing that spans multiple experiments.
    
    Key Responsibilities:
    1. **Experiment Discovery**: Auto-discovers experiments from raw_image_data structure
    2. **Global File Management**: Tracks combined metadata files (df01, df02, df03)
    3. **Intelligent Orchestration**: Determines what processing is needed across the cohort
    4. **Dependency Management**: Ensures correct order of per-experiment vs global steps
    
    Global Pipeline Flow:
    Per-experiment: [Raw → FF → QC/SAM2 → Build03 → Latents] 
                                     ↓
    Global: [df01] → Build04 → [df02] → Build06 → [df03]
    
    Combined Files:
    - df01: Raw embryo data from all Build03 runs (per-experiment contributions)
    - df02: QC'd + staged embryo data from Build04 (global processing)  
    - df03: Final dataset with embeddings from Build06 (global merge)
    
    Attributes:
        root: Path to MorphSeq data root directory
        exp_dir: Path to experiment state files (metadata/experiments/)
        experiments: Dict mapping experiment dates to Experiment objects
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.exp_dir = self.root / "metadata" / "experiments"
        self.experiments: dict[str, Experiment] = {}
        self.discover_experiments()
        self.update_experiment_status()

    # ——— Global File Management: Combined Metadata Files ————————————————

    @property
    def df01_path(self) -> Path:
        """
        Path to embryo_metadata_df01.csv - the raw combined embryo dataset.
        
        df01 contains embryo-level data from all experiments that have completed Build03.
        Each row represents one embryo at one timepoint, with morphological measurements
        and metadata. This file grows as more experiments complete Build03.
        
        Created by: Build03 (appends per-experiment data)
        Used by: Build04 input
        """
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"

    @property
    def df02_path(self) -> Path:
        """
        Path to embryo_metadata_df02.csv - the QC'd and staged embryo dataset.
        
        df02 is created by Build04 and contains df01 data after quality control,
        outlier removal, and developmental stage inference. This is a cleaned,
        analysis-ready version of df01.
        
        Created by: Build04 (global QC processing)
        Used by: Build06 input
        """
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"

    @property
    def df03_path(self) -> Path:
        """
        Path to embryo_metadata_df03.csv - the final analysis-ready dataset.
        
        df03 is created by Build06 and contains df02 data merged with latent
        morphological embeddings. This is the final dataset used for downstream
        analysis and machine learning applications.
        
        Created by: Build06 (global merge with embeddings)
        Used by: Analysis and ML workflows
        """
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"

    @property
    def needs_build04(self) -> bool:
        """
        Determine if Build04 global QC processing needs to run.
        
        Build04 processes the combined df01 embryo data through quality control,
        outlier detection, and developmental stage inference to produce df02.
        This is a global operation that processes all experiments together.
        
        Logic:
        - If df01 doesn't exist → False (no input data available)
        - If df02 doesn't exist → True (output needs to be created)
        - If df01 is newer than df02 → True (input has been updated)
        
        Returns:
            bool: True if Build04 global QC needs to run
        """
        # Can't run without input data
        if not self.df01_path.exists():
            return False
        # Output missing, needs to run
        if not self.df02_path.exists():
            return True
        # Check if input is newer than output
        try:
            return self.df01_path.stat().st_mtime > self.df02_path.stat().st_mtime
        except Exception:
            return False

    @property
    def needs_build06(self) -> bool:
        """
        Build06 is needed if:
        1. df02 exists but df03 doesn't, OR
        2. There are experiments in df02 that aren't in df03, OR
        3. There are latent files newer than df03 for experiments that should be included
        """
        if not self.df02_path.exists():
            return False
        if not self.df03_path.exists():
            return True
        
        try:
            # Get the set of experiments in df02 vs df03
            import pandas as pd
            
            df02 = pd.read_csv(self.df02_path)
            df03 = pd.read_csv(self.df03_path)
            
            # Get unique experiment dates from each file
            df02_experiments = set(df02['experiment_date'].unique()) if 'experiment_date' in df02.columns else set()
            df03_experiments = set(df03['experiment_date'].unique()) if 'experiment_date' in df03.columns else set()
            
            # Check if there are experiments in df02 that aren't in df03
            missing_from_df03 = df02_experiments - df03_experiments
            if missing_from_df03:
                return True
            
            # Check if any experiment's latent files are newer than df03
            # (but only for experiments that should be in the final dataset)
            df03_time = self.df03_path.stat().st_mtime
            for exp in self.experiments.values():
                if exp.date in df02_experiments and exp.has_latents():
                    latent_path = exp.get_latent_path()
                    if latent_path.exists() and latent_path.stat().st_mtime > df03_time:
                        return True
                            
            return False
        except Exception as e:
            # Fallback to simple timestamp comparison if DataFrame operations fail
            try:
                return self.df02_path.stat().st_mtime > self.df03_path.stat().st_mtime
            except Exception:
                return False

    # ——— Stage 3: Global orchestration methods ——————————————————————————

    def run_build04(self, **kwargs):
        """Execute global Build04 step (df01 -> df02)"""
        print("🔄 Running Build04 (global QC + staging)")
        from ..run_morphseq_pipeline.steps.run_build04 import run_build04
        return run_build04(root=str(self.root), **kwargs)

    def run_build06(self, model_name: str = "20241107_ds_sweep01_optimum", **kwargs):
        """Execute global Build06 step (df02 + latents -> df03)"""
        print("🔄 Running Build06 (global embeddings merge)")
        from ..run_morphseq_pipeline.steps.run_build06 import run_build06
        return run_build06(
            morphseq_repo_root=str(self.root.parent.parent if "data" in str(self.root) else self.root),
            data_root=str(self.root),
            model_name=model_name,
            **kwargs
        )


    def discover_experiments(self):
        # scan "raw_image_data" subfolders for dates
        raw = self.root / "raw_image_data"
        # collect all the dates first
        dates = []
        for mic in raw.iterdir():
            if (not mic.is_dir()) or (mic.name=="ignore"): 
                continue
            for d in mic.iterdir():
                if d.is_dir():
                    dates.append(d.name)
                    
        # dedupe & sort
        for date in sorted(set(dates)):
            self.experiments[date] = Experiment(date, self.root)

    # Helper to update all experiments to reflect presence/absence of files on disk
    def update_experiment_status(self):
        for exp in self.experiments.values():
            exp._sync_with_disk()

    def export_all(self):
        for exp in self.experiments.values():
            if exp.needs_export:
                try:
                    exp.export_images()
                except:
                    log.exception("Export & FF build failed for %s", exp.date)
    
    def export_all_metadata(self):
        for exp in self.experiments.values():
            try:
                exp.export_metadata()
            except:
                log.exception("Metadata build failed for %s", exp.date)

    def export_experiment_metadata(self, experiments=None):
        # Build and run a filtered list of experiments
        if experiments is None:
            experiments_list = list(self.experiments.values())
        else:
            experiments_list = [exp for exp in self.experiments.values() if exp.date in set(experiments)]

        for exp in experiments_list:
            try:
                exp.export_metadata()
            except Exception:
                log.exception("Metadata build failed for %s", exp.date)
            
    def stitch_all(self):
        for exp in self.experiments.values():
            if exp.needs_stitch:
                try:
                    exp.stitch_images()
                except:
                    log.exception("Stitching  failed for %s", exp.date) 


    def _run_step(
        self,
        step: str,                                   # name of Experiment method to call
        need_attr: str,                              # the corresponding “needs_*” flag
        *,
        experiments: list[str] | None = None,
        later_than: int | None = None,
        earlier_than: int = 99_999_999,
        force_update: bool = False,
        extra_filter: callable[[Experiment], bool] | None = None,
        friendly_name: str | None = None,            # text used in printouts
    ) -> None:
        """Find experiments that should run *step* and call it.

        extra_filter(exp) → bool lets you add step-specific constraints
        (e.g. microscope == "Keyence").  Leave it None if not needed.
        """
        # 0) sanity check ------------------------------------------------------
        if (experiments is None) == (later_than is None):
            raise ValueError("pass *either* experiments or later_than (not both)")

        # 1) pick the candidates ----------------------------------------------
        selected, dates = [], []
        for exp in self.experiments.values():
            # --- user-provided subset
            if experiments is not None and exp.date not in experiments:
                continue
            # --- date window
            if experiments is None:
                try:
                    di = int(exp.date[:8])
                except ValueError:
                    continue
                if not (later_than <= di < earlier_than):
                    continue
            # --- “needs_*” or forced
            if not getattr(exp, need_attr) and not force_update:
                continue
            # --- custom predicate
            if extra_filter and not extra_filter(exp):
                continue

            selected.append(exp)
            dates.append(exp.date)

        if not selected:
            print(f"No experiments to {friendly_name or step}.")
            return

        print(f"{friendly_name or step.capitalize()}:", ", ".join(sorted(dates)))

        # 2) run the step ------------------------------------------------------
        for exp in selected:
            try:
                getattr(exp, step)()          # e.g. exp.export_images()
            except Exception as e:
                log.exception("❌  %s failed for %s", step, exp.date)

    # ──────────────────────────────────────────────────────────────────────────
    # thin façade methods
    def export_experiments(self, **kwargs):
        self._run_step(
            "export_images", "needs_export",
            friendly_name="export",
            **kwargs
        )

    # thin façade methods
    def export_experiment_metadata(self, **kwargs):
        self._run_step(
            "export_images", "needs_export",
            friendly_name="export",
            **kwargs
        )

    def stitch_experiments(self, **kwargs):
        self._run_step(
            "stitch_images", "needs_stitch",
            extra_filter=lambda e: e.microscope == "Keyence",
            friendly_name="stitch",
            **kwargs
        )

    def stitch_z_experiments(self, **kwargs):
        self._run_step(
            "stitch_z_images", "needs_stitch_z",
            extra_filter=lambda e: e.microscope == "Keyence",
            friendly_name="stitch_z",
            **kwargs
        )

    def segment_experiments(self, **kwargs):
        self._run_step(
            "segment_images", "needs_segment",
            # extra_filter=lambda e: e.microscope == "Keyence",
            friendly_name="segment",
            **kwargs
        )

    def get_embryo_stats(self, **kwargs):
        self._run_step(
            "process_image_masks", "needs_stats",
            friendly_name="mask_stats",
            **kwargs
        )

    # def build_metadata_all(self):
    #     for exp in self.experiments.values():
    #         if exp.needs_build_metadata():
    #             exp.build_image_metadata()

    def report(self):
        for date, exp in self.experiments.items():
            print(f"{date}: raw={exp.flags['raw']}, meta={exp.flags['meta']}, ff={exp.flags['ff']}, stitch={exp.flags['stitch']}, stitch_z={exp.flags['stitch_z']}, segment={exp.flags['segment']}")



if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Call pipeline functions
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

    # load master experiment log
    manager = ExperimentManager(root=root)
    manager.report()

    exp = Experiment(date="20250703_chem3_34C_T01_1457", data_root=root)
    exp.process_image_masks()
    print("check")
