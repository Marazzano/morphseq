from __future__ import annotations

from pydantic.dataclasses import dataclass
from dataclasses import dataclass as std_dataclass, field, replace
from typing import Literal, List, Type, Callable, Any, Dict, Mapping, Optional, Union
from src.core.data.dataset_utils import make_seq_key, make_train_test_split
from src.core.data.data_transforms import basic_transform, contrastive_transform
from src.core.data.dataset_classes import BasicDataset, NTXentDataset, BasicEvalDataset
import os
import numpy as np
import pandas as pd
from src.core.data.dataset_utils import smart_read_csv
from pydantic   import ConfigDict
from pathlib import Path
from src.core.data.manifest_types import ManifestPolicy, PipelineManifestResult
from src.core.data.pipeline_manifest import (
    ArtifactPathFunction,
    AssetPathResolver,
    ExperimentSourcePaths,
    build_pipeline_manifest,
)
from src.core.metric import (
    CompiledMetricMapping,
    MetricMappingArtifact,
    MetricPairIndex,
    MetricPairSampler,
    MetricPairingPolicy,
    MetricRelationPolicy,
    PairPreflightReport,
    build_metric_provenance_payload,
    preflight_pair_indices,
    validate_metric_bundle_for_preset,
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class UrrDataConfig:

    seq_key: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    eval_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    batch_size: int = 64
    num_workers: int = 4
    wrap: bool = True
    root: str = "./data"

    @property
    def image_path(self) -> str:
        return os.path.join(self.root, "images")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.root, "metadata", "")

    @property
    def age_key_path(self) -> str:
        return os.path.join(self.root, "metadata", "age_key.csv")

    @property
    def pert_time_key_path(self) -> str:
        return os.path.join(self.root, "metadata", "perturbation_train_key.csv")

    def split_train_test(self):
        """
        Load the dataset from the specified file path using pandas.
        """
        # get seq key
        seq_key = make_seq_key(self.root)

        if os.path.isfile(self.age_key_path):
            age_key_df = pd.read_csv(self.age_key_path)
            age_key_df = age_key_df.loc[:, ["snip_id", "inferred_stage_hpf_reg"]]
            seq_key = seq_key.merge(age_key_df, how="left", on="snip_id")
        else:
            raise Exception("Stage key provided!")

        if os.path.isfile(self.pert_time_key_path):
            pert_time_key = pd.read_csv(self.pert_time_key_path)
        else:
            # raise Exception("No perturbation-time key provided!")
            pert_time_key = None

        seq_key, train_indices, eval_indices, test_indices = make_train_test_split(seq_key, pert_time_key=pert_time_key)

        self.seq_key = seq_key
        if (self.train_indices.size
                and self.eval_indices.size
                and self.test_indices.size
        ):
            pass
        else: #  overwrite if empty
            self.eval_indices = eval_indices
            self.test_indices = test_indices
            self.train_indices = train_indices


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class EvalDataConfig:

    experiments:   List[str]
    return_sample_names: bool = True

    transforms: Any = None
    batch_size: int = 64
    num_workers: int = 2
    wrap: bool = True
    root: Union[str, Path] = "./data"

    @property
    def data_path(self) -> Path:
        root = Path(self.root)
        return root / "training_data" / "bf_embryo_snips"

    def make_metadata(self):
        self.split_train_test()

    def create_dataset(self):

        # instantiate your dataset with both fixed and configurable args
        return BasicEvalDataset(
            root=self.data_path,
            experiments=self.experiments,
            transform=self.transforms,
            return_name=self.return_sample_names
        )

@dataclass# (config_wrapper=ConfigDict(arbitrary_types_allowed=True))
class BaseDataConfig(UrrDataConfig):

    # 1) pick by name, not by object
    target_name:   Literal["BasicDataset"] = "BasicDataset"

    # 2) a catch‐all for per‐dataset options
    target_kwargs: Dict[str,Any]        = field(default_factory=dict)

    # similarly for transform
    transform_name:   Literal["basic", "simclr"] = "simclr"
    transform_kwargs: Dict[str, Any]                = field(default_factory=dict)

    return_sample_names: bool = False

    def make_metadata(self):
        self.split_train_test()

    def create_dataset(self):
        # map names→classes/functions
        ds_map = {
            "BasicDataset": BasicDataset,
            # "OtherDataset": OtherDataset,
        }
        tf_map = {
            "basic": basic_transform,
            "simclr": contrastive_transform,
        }

        ds_cls = ds_map[self.target_name]
        tf_fn  = tf_map[self.transform_name]

        # build the actual transform
        transform = tf_fn(**self.transform_kwargs)

        # instantiate your dataset with both fixed and configurable args
        return ds_cls(
            root=self.image_path,
            return_name=self.return_sample_names,
            transform=transform,
            **self.target_kwargs
        )

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NTXentDataConfig(UrrDataConfig):

    seq_key_dict: Dict[str, np.ndarray] = field(
        default_factory=dict
    )
    metric_key: pd.DataFrame = field(default_factory=pd.DataFrame)

    metric_array: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    train_bool: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    eval_bool: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    test_bool: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    return_name: bool = True

    target_name:   Literal["NTXentDataset"] = "NTXentDataset"

    # 2) a catch‐all for per‐dataset options
    target_kwargs: Dict[str,Any]        = field(default_factory=dict)

    # these attributes will be pulled from loss config
    time_window: Optional[int] = None
    self_target_prob: Optional[int] = None

    # similarly for transform
    transform_name:   Literal["simclr"] = "simclr"
    transform_kwargs: Dict[str,Any]                = field(default_factory=dict)


    @property
    def metric_key_path(self) -> str:
        return os.path.join(self.root, "metadata", "metric_key.csv")


    def create_dataset(self):
        # map names→classes/functions
        ds_map = {
            "NTXentDataset": NTXentDataset,
            # "OtherDataset": OtherDataset,
        }
        tf_map = {
            "basic": basic_transform(),
            "simclr":     contrastive_transform,
        }

        ds_cls = ds_map[self.target_name]
        tf_fn  = tf_map[self.transform_name]

        # build the actual transform
        transform = tf_fn(**self.transform_kwargs)

        # instantiate your dataset with both fixed and configurable args
        return ds_cls(
            cfg=self,
            transform=transform,
        )

    def make_metadata(self):
        # produces train/test/eval indices and seq_key and pert_time_key
        self.split_train_test()

        # load metric key
        metric_key = smart_read_csv(self.metric_key_path)
        self.metric_key = metric_key

        # generate some indexing/metadata vectors
        seq_key = self.seq_key

        pert_id_vec = seq_key["perturbation_id"].to_numpy()
        e_id_vec = seq_key["embryo_id_num"].to_numpy()
        age_hpf_vec = seq_key["stage_hpf"].to_numpy()

        seq_key_dict = dict({"pert_id_vec": pert_id_vec, "e_id_vec": e_id_vec, "age_hpf_vec": age_hpf_vec})
        self.seq_key_dict = seq_key_dict

        # make array version of metric key
        pert_id_key = seq_key.loc[:, ["short_pert_name", "perturbation_id"]].drop_duplicates().reset_index(drop=True)
        metric_array = metric_key.to_numpy()
        pert_skel = pd.DataFrame(metric_key.index.tolist(), columns=["short_pert_name"])
        sort_skel = pert_skel.merge(pert_id_key, how="left", on="short_pert_name")
        id_sort_vec = np.argsort(sort_skel["perturbation_id"])

        metric_array = metric_array[id_sort_vec, :]
        self.metric_array = metric_array[:, id_sort_vec]

        # make boolean vactors for train, eval, and test groups
        self.train_bool = np.zeros(pert_id_vec.shape, dtype=np.bool_)
        self.train_bool[self.train_indices] = True
        self.eval_bool = np.zeros(pert_id_vec.shape, dtype=np.bool_)
        self.eval_bool[self.eval_indices] = True
        self.test_bool = np.zeros(pert_id_vec.shape, dtype=np.bool_)
        self.test_bool[self.test_indices] = True


@std_dataclass
class PipelineDataConfig:
    """Split-aware data configuration over one built manifest result.

    Manifest construction is explicit and happens once in ``make_metadata`` (or
    lazily on the first dataset request).  A2 owns the dataset implementation;
    this A1-owned configuration only passes the frozen resolved view and loader
    settings across that seam.
    """

    pipeline_output_root: Union[str, Path]
    manifest_policy: ManifestPolicy
    source_paths: Mapping[str, ExperimentSourcePaths] | None = None
    input_dim: tuple[int, int, int] = (1, 288, 128)
    transform: Any = None
    batch_size: int = 64
    num_workers: int = 4
    loader_seed: int = 0
    drop_last_train: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    max_decode_failures: int = 10
    artifact_path_authority: str = (
        "src/data_pipeline/pipeline_orchestrator/orchestration/paths.py:artifact_path"
    )
    asset_path_authority: str = (
        "src/data_pipeline/object_extraction/snip_processing/io.py:resolve_from_root"
    )
    artifact_path_fn: ArtifactPathFunction | None = field(default=None, repr=False)
    asset_path_resolver: AssetPathResolver | None = field(default=None, repr=False)
    manifest_result: PipelineManifestResult | None = field(default=None, init=False, repr=False)

    @property
    def resolved_sample_table(self) -> pd.DataFrame:
        if self.manifest_result is None:
            raise RuntimeError(
                "PipelineDataConfig manifest has not been built; call make_metadata() first."
            )
        return self.manifest_result.resolved_sample_table

    def make_metadata(self) -> PipelineManifestResult:
        self.manifest_result = build_pipeline_manifest(
            Path(self.pipeline_output_root),
            self.manifest_policy,
            source_paths=self.source_paths,
            artifact_path_fn=self.artifact_path_fn,
            asset_path_resolver=self.asset_path_resolver,
        )
        return self.manifest_result

    def create_dataset(self, *, split: str):
        if self.manifest_result is None:
            self.make_metadata()
        return BasicDataset(
            resolved_sample_table=self.resolved_sample_table,
            product_key=self.manifest_policy.selected_product_key,
            input_dim=self.input_dim,
            split=split,
            transform=self.transform,
            max_decode_failures=self.max_decode_failures,
        )


@std_dataclass
class PipelineMetricDataConfig(PipelineDataConfig):
    """Manifest-backed metric configuration with C1 mapping and C2 pair indexes."""

    metric_mapping_artifact: MetricMappingArtifact | None = None
    relation_policy: MetricRelationPolicy | None = None
    pairing_policy: MetricPairingPolicy | None = None
    distributed_rank: int = 0
    pair_indices: Mapping[str, MetricPairIndex] = field(
        default_factory=dict, init=False, repr=False
    )
    pair_preflight_reports: tuple[PairPreflightReport, ...] = field(
        default=(), init=False
    )

    @property
    def metric_provenance_payload(self) -> dict[str, Any]:
        mapping, relation = self._require_metric_policies()
        return build_metric_provenance_payload(
            mapping=mapping,
            relation_policy=relation,
        )

    def make_metadata(self) -> PipelineManifestResult:
        mapping, relation = self._require_metric_policies()
        pairing = self._require_pairing_policy()
        declared_mapping = self.manifest_policy.metric_mapping
        if not declared_mapping.enabled:
            raise ValueError(
                "PipelineMetricDataConfig requires manifest_policy.metric_mapping.enabled=True."
            )
        if declared_mapping.name != mapping.name:
            raise ValueError(
                f"manifest metric mapping name={declared_mapping.name!r} disagrees with "
                f"artifact name={mapping.name!r}."
            )
        if declared_mapping.scientific_policy != mapping.scientific_policy:
            raise ValueError(
                "manifest metric_mapping.scientific_policy disagrees with the mapping artifact."
            )
        if pairing.scientific_policy != mapping.scientific_policy:
            raise ValueError(
                "pairing_policy.scientific_policy disagrees with the mapping artifact."
            )
        validate_metric_bundle_for_preset(
            mapping=mapping,
            relation_policy=relation,
            scientific_preset=pairing.scientific_policy,
        )

        result = super().make_metadata()
        resolved = result.resolved_sample_table.copy()
        mapped = CompiledMetricMapping(mapping).map_records(
            resolved.to_dict(orient="records")
        )
        unassigned_indices = [
            index for index, group_name in enumerate(mapped.group_names)
            if group_name is None
        ]
        if unassigned_indices:
            snip_ids = resolved.iloc[unassigned_indices]["snip_id"].tolist()
            raise ValueError(
                f"metric mapping {mapping.name!r}@{mapping.version} left accepted rows "
                f"unassigned; snip_id values={snip_ids!r}."
            )
        resolved["metric_group"] = mapped.group_names
        resolved["metric_group_code"] = mapped.group_codes
        self.manifest_result = replace(result, resolved_sample_table=resolved)

        pair_indices: dict[str, MetricPairIndex] = {}
        for split in ("train", "eval", "test"):
            split_table = resolved.loc[resolved["split"].eq(split)].reset_index(drop=True)
            if split_table.empty:
                continue
            pair_indices[split] = MetricPairIndex(
                split_table,
                relation_policy=relation,
                pairing_policy=pairing,
                split=split,
            )
        self.pair_preflight_reports = preflight_pair_indices(pair_indices)
        self.pair_indices = pair_indices
        return self.manifest_result

    def create_dataset(self, *, split: str):
        if self.manifest_result is None or not self.pair_indices:
            self.make_metadata()
        try:
            pair_index = self.pair_indices[split]
        except KeyError as exc:
            raise ValueError(
                f"metric manifest has no preflighted pair index for split={split!r}."
            ) from exc
        pairing = self._require_pairing_policy()
        return NTXentDataset(
            resolved_sample_table=self.resolved_sample_table,
            product_key=self.manifest_policy.selected_product_key,
            pair_sampler=MetricPairSampler(pair_index, rank=self.distributed_rank),
            policy_name=pairing.name,
            scientific_policy=pairing.scientific_policy,
            input_dim=self.input_dim,
            split=split,
            transform=self.transform,
            max_decode_failures=self.max_decode_failures,
        )

    def _require_metric_policies(
        self,
    ) -> tuple[MetricMappingArtifact, MetricRelationPolicy]:
        if not isinstance(self.metric_mapping_artifact, MetricMappingArtifact):
            raise TypeError(
                "PipelineMetricDataConfig requires a MetricMappingArtifact."
            )
        if not isinstance(self.relation_policy, MetricRelationPolicy):
            raise TypeError(
                "PipelineMetricDataConfig requires a C1 MetricRelationPolicy."
            )
        return self.metric_mapping_artifact, self.relation_policy

    def _require_pairing_policy(self) -> MetricPairingPolicy:
        if not isinstance(self.pairing_policy, MetricPairingPolicy):
            raise TypeError("PipelineMetricDataConfig requires a MetricPairingPolicy.")
        return self.pairing_policy
