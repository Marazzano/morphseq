from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build.build01A_compile_keyence_torch import stitch_ff_from_keyence, build_ff_from_keyence
from src.build.build01AB_stitch_keyence_z_slices import stitch_z_from_keyence
from src.build.build01B_compile_yx1_images_torch import build_ff_from_yx1
import pandas as pd
import multiprocessing
from typing import Literal, Optional, Dict, List, Sequence
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
        Build03 appends to a global df01 but is driven per-experiment.
        We consider it needed when a SAM2 CSV exists and is newer than the
        last recorded build03 timestamp for this experiment.
        """
        try:
            return self._safe_mtime_compare(self.sam2_csv_path, "build03")
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
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.exp_dir = self.root / "metadata" / "experiments"
        self.experiments: dict[str, Experiment] = {}
        self.discover_experiments()
        self.update_experiment_status()

    # ——— global tracking (combined metadata) ————————————————————————
    @property
    def df01_path(self) -> Path:
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv"

    @property
    def df02_path(self) -> Path:
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df02.csv"

    @property
    def df03_path(self) -> Path:
        return self.root / "metadata" / "combined_metadata_files" / "embryo_metadata_df03.csv"

    @property
    def needs_build04(self) -> bool:
        # Can't run without input
        if not self.df01_path.exists():
            return False
        # Needs run if output missing
        if not self.df02_path.exists():
            return True
        try:
            return self.df01_path.stat().st_mtime > self.df02_path.stat().st_mtime
        except Exception:
            return False

    @property
    def needs_build06(self) -> bool:
        if not self.df02_path.exists():
            return False
        if not self.df03_path.exists():
            return True
        try:
            return self.df02_path.stat().st_mtime > self.df03_path.stat().st_mtime
        except Exception:
            return False


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
