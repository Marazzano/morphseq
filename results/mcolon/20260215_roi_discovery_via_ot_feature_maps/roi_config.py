"""
Configuration dataclasses for ROI discovery via OT feature maps.

Follows the same frozen-dataclass pattern as UOTConfig in
src/analyze/utils/optimal_transport/config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class FeatureSet(str, Enum):
    """Which OT-derived channels to use as input features."""
    COST = "cost"
    COST_DISP = "cost+disp"
    ALL_OT = "all_ot"


class ROISizePreset(str, Enum):
    """Maps to λ (L1 penalty) grid presets."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class SmoothnessPreset(str, Enum):
    """Maps to μ (TV penalty) grid presets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NullMode(str, Enum):
    PERMUTE = "permute"
    BOOTSTRAP = "bootstrap"
    BOTH = "both"
    NONE = "none"


class SelectionRule(str, Enum):
    """Deterministic selection rule for (λ,μ) from sweep."""
    PARETO_KNEE = "pareto_knee"       # Option A: knee on Pareto front
    EPSILON_BEST = "epsilon_best"     # Option B: smallest complexity within ε of best AUROC


# ---------------------------------------------------------------------------
# λ/μ presets — these are *starting points*, not universal truths.
# Biology-dependent tuning will refine them.
# ---------------------------------------------------------------------------

LAMBDA_PRESETS = {
    ROISizePreset.SMALL:  [1e-2, 3e-2, 1e-1, 3e-1],
    ROISizePreset.MEDIUM: [1e-3, 3e-3, 1e-2, 3e-2],
    ROISizePreset.LARGE:  [1e-4, 3e-4, 1e-3, 3e-3],
}

MU_PRESETS = {
    SmoothnessPreset.LOW:    [1e-4, 1e-3, 1e-2],
    SmoothnessPreset.MEDIUM: [1e-3, 1e-2, 1e-1],
    SmoothnessPreset.HIGH:   [1e-2, 1e-1, 1.0],
}


@dataclass(frozen=True)
class FeatureDatasetConfig:
    """Configuration for the on-disk FeatureDataset contract."""
    canonical_grid_hw: Tuple[int, int] = (512, 512)
    chunk_size_n: int = 8
    compression: str = "zstd"
    compression_level: int = 3
    iqr_multiplier: float = 1.5       # for QC outlier filter on total_cost_C
    group_key: str = "embryo_id"      # MANDATORY: prevents leakage in CV splits


@dataclass(frozen=True)
class ChannelSchema:
    """Describes a single feature channel in the dataset."""
    name: str
    definition: str
    units: str


# Standard channel schemas for OT-derived features
CHANNEL_SCHEMAS = {
    FeatureSet.COST: [
        ChannelSchema("total_cost", "Per-pixel OT transport cost", "cost_units"),
    ],
    FeatureSet.COST_DISP: [
        ChannelSchema("total_cost", "Per-pixel OT transport cost", "cost_units"),
        ChannelSchema("displacement_y", "Barycentric displacement (y)", "um"),
        ChannelSchema("displacement_x", "Barycentric displacement (x)", "um"),
    ],
    FeatureSet.ALL_OT: [
        ChannelSchema("total_cost", "Per-pixel OT transport cost", "cost_units"),
        ChannelSchema("displacement_y", "Barycentric displacement (y)", "um"),
        ChannelSchema("displacement_x", "Barycentric displacement (x)", "um"),
        ChannelSchema("mass_created", "Mass creation per pixel", "mass_units"),
        ChannelSchema("mass_destroyed", "Mass destruction per pixel", "mass_units"),
    ],
}


@dataclass(frozen=True)
class TrainerConfig:
    """JAX trainer configuration."""
    learn_res: int = 128
    output_res: int = 512
    learning_rate: float = 1e-2
    max_steps: int = 2000
    convergence_tol: float = 1e-6
    log_every: int = 100
    random_seed: int = 42


@dataclass(frozen=True)
class SweepConfig:
    """λ/μ sweep configuration."""
    lambda_values: Tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    mu_values: Tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    n_cv_folds: int = 5
    selection_rule: SelectionRule = SelectionRule.PARETO_KNEE
    # For PARETO_KNEE: beta controls knee sensitivity
    pareto_beta: float = 1.0
    # For EPSILON_BEST: ε tolerance on AUROC
    epsilon_auroc: float = 0.02
    # ROI extraction
    roi_quantile: float = 0.9    # threshold |w| at this quantile


@dataclass(frozen=True)
class NullConfig:
    """Null distribution + stability configuration."""
    null_mode: NullMode = NullMode.BOTH
    n_permute: int = 100
    n_boot: int = 200
    boot_roi_quantile: float = 0.9
    random_seed: int = 42


@dataclass
class ROIRunConfig:
    """
    Top-level run configuration combining all sub-configs.
    
    This dataclass is mutable (not frozen) to allow runtime modification
    of fields like out_dir and other configuration parameters.
    """
    genotype: str = "cep290"
    reference: str = "WT"
    features: FeatureSet = FeatureSet.COST
    roi_size: ROISizePreset = ROISizePreset.MEDIUM
    smoothness: SmoothnessPreset = SmoothnessPreset.MEDIUM

    dataset: FeatureDatasetConfig = field(default_factory=FeatureDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    nulls: NullConfig = field(default_factory=NullConfig)

    out_dir: Optional[str] = None

    def resolve_lambda_values(self) -> List[float]:
        """Get λ values from preset or sweep config."""
        return list(LAMBDA_PRESETS.get(self.roi_size, self.sweep.lambda_values))

    def resolve_mu_values(self) -> List[float]:
        """Get μ values from preset or sweep config."""
        return list(MU_PRESETS.get(self.smoothness, self.sweep.mu_values))


__all__ = [
    "FeatureSet",
    "ROISizePreset",
    "SmoothnessPreset",
    "NullMode",
    "SelectionRule",
    "FeatureDatasetConfig",
    "ChannelSchema",
    "CHANNEL_SCHEMAS",
    "TrainerConfig",
    "SweepConfig",
    "NullConfig",
    "ROIRunConfig",
    "LAMBDA_PRESETS",
    "MU_PRESETS",
]
