# SPDX-License-Identifier: Apache-2.0
"""Strict records for the prospective preprocessing adapter."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from .canonical import EvidenceError, require_sha256, safe_relative_path
from .records import (
    require_bool,
    require_int,
    require_list,
    require_number,
    require_object,
    require_string,
    strict_fields,
)

MONAI_COMMIT = "46a5272196a6c2590ca2589029eed8e4d56ff008"
MONAI_TAG_OBJECT = "416ede8a845eff9294c7656f110c763dec1410fa"
MODEL_ZOO_COMMIT = "5370ce6ea1dd132856b9c92e2fa125548594835d"
MILESTONE4_BUNDLE_SHA256 = "d3fdca9d8e5dccfbe5b1e755ca1a9fa782dd986e1fd00c5caa1997986a1c20df"
MODEL_INFERENCE_CONFIG_SHA256 = "22dead767f9bfcd6e3e7cbf5353ef1641e52f37acd9a20cc346361c1e603cf9e"
UV_LOCK_SHA256 = "5bbe48a51f4a649af5370a87ace2e87e2a51f564f2633412b2ff94f1a7b2a71f"
CHANNEL_ORDER = ("image-t1c", "image-t1", "image-t2", "image-flair")
COMPARISON_ATOL = 0.00002
COMPARISON_RTOL = 0.000002

RefusalStage = Literal[
    "attempt_record",
    "custody_verification",
    "channel_loading",
    "reference_normalization",
    "independent_normalization",
    "implementation_comparison",
    "snapshot_publication",
]
_REFUSAL_STAGES = frozenset(
    {
        "attempt_record",
        "custody_verification",
        "channel_loading",
        "reference_normalization",
        "independent_normalization",
        "implementation_comparison",
        "snapshot_publication",
    }
)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _ints(value: Any, name: str) -> tuple[int, ...]:
    return tuple(require_int(item, f"{name} item") for item in require_list(value, name))


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    git_blob_sha1: str
    sha256: str
    immutable_url: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        if len(self.git_blob_sha1) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_blob_sha1
        ):
            raise EvidenceError("invalid_git_blob", self.path)
        require_sha256(self.sha256)
        if MONAI_COMMIT not in self.immutable_url:
            raise EvidenceError("mutable_source_url", self.path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceIdentity:
        data = strict_fields(
            data,
            {"git_blob_sha1", "immutable_url", "path", "sha256"},
            "SourceIdentity",
        )
        return cls(
            require_string(data["path"], "path"),
            require_string(data["git_blob_sha1"], "git_blob_sha1"),
            require_string(data["sha256"], "sha256"),
            require_string(data["immutable_url"], "immutable_url"),
        )


@dataclass(frozen=True)
class ImplementationIdentity:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        require_sha256(self.sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationIdentity:
        data = strict_fields(data, {"path", "sha256"}, "ImplementationIdentity")
        return cls(
            require_string(data["path"], "path"),
            require_string(data["sha256"], "sha256"),
        )


@dataclass(frozen=True)
class RuntimeIdentity:
    python_version: Literal["3.13.15"]
    numpy_version: Literal["2.5.3"]
    nibabel_version: Literal["5.4.2"]
    byte_order: Literal["little"]
    uv_lock_sha256: str

    def __post_init__(self) -> None:
        if (
            self.python_version != "3.13.15"
            or self.numpy_version != "2.5.3"
            or self.nibabel_version != "5.4.2"
            or self.byte_order != "little"
            or self.uv_lock_sha256 != UV_LOCK_SHA256
        ):
            raise EvidenceError("untrusted_preprocessing_runtime", "runtime identity")
        require_sha256(self.uv_lock_sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeIdentity:
        data = strict_fields(
            data,
            {
                "byte_order",
                "nibabel_version",
                "numpy_version",
                "python_version",
                "uv_lock_sha256",
            },
            "RuntimeIdentity",
        )
        return cls(
            cast(Any, require_string(data["python_version"], "python_version")),
            cast(Any, require_string(data["numpy_version"], "numpy_version")),
            cast(Any, require_string(data["nibabel_version"], "nibabel_version")),
            cast(Any, require_string(data["byte_order"], "byte_order")),
            require_string(data["uv_lock_sha256"], "uv_lock_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterPlan:
    schema_version: Literal["voxelscope/preprocessing-adapter-plan/v1"]
    plan_id: str
    milestone4_bundle_sha256: str
    model_bundle_version: Literal["0.5.2"]
    model_zoo_commit: str
    model_inference_config_sha256: str
    monai_version: Literal["1.4.0"]
    monai_tag: Literal["1.4.0"]
    monai_tag_object: str
    monai_commit: str
    monai_sources: tuple[SourceIdentity, ...]
    observed_behavior: tuple[str, ...]
    assumptions: tuple[str, ...]
    channel_order: tuple[str, ...]
    label_role: Literal["label"]
    label_excluded: Literal[True]
    spatial_transforms: tuple[str, ...]
    loader: Literal["NibabelReader"]
    as_closest_canonical: Literal[False]
    scaled_data_before_cast: Literal[True]
    effective_dtype: Literal["<f4"]
    normalization_nonzero: Literal[True]
    normalization_channel_wise: Literal[True]
    normalization_population_std: Literal[True]
    background_zero_preserved: Literal[True]
    empty_channel_policy: Literal["refuse"]
    zero_variance_policy: Literal["refuse"]
    finite_policy: Literal["input-and-output-required"]
    output_layout: Literal["C,I,J,K"]
    output_order: Literal["C"]
    output_dtype: Literal["<f4"]
    canonical_implementation: Literal["direct-numpy-float32"]
    independent_implementation: Literal["chunked-welford-float64-statistics"]
    chunk_depth: int
    comparison_atol: float
    comparison_rtol: float
    exact_byte_comparison_recorded: Literal[True]
    dependency_oracle: Literal["official-source-derived"]
    monai_runtime_dependency_added: Literal[False]
    one_shot_execution: Literal[True]
    private_output: Literal[True]
    no_clobber: Literal[True]
    network_allowed: Literal[False]
    model_loading_allowed: Literal[False]
    inference_authorized: Literal[False]
    runtime_identity: RuntimeIdentity
    implementation_sources: tuple[ImplementationIdentity, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/preprocessing-adapter-plan/v1"
            or self.plan_id != "voxelscope-milestone-5-preprocessing-v1"
        ):
            raise EvidenceError("unsupported_schema", self.schema_version)
        fixed_hashes = (
            self.milestone4_bundle_sha256,
            self.model_inference_config_sha256,
        )
        for value in fixed_hashes:
            require_sha256(value)
        if (
            self.milestone4_bundle_sha256 != MILESTONE4_BUNDLE_SHA256
            or self.model_bundle_version != "0.5.2"
            or self.model_zoo_commit != MODEL_ZOO_COMMIT
            or self.model_inference_config_sha256 != MODEL_INFERENCE_CONFIG_SHA256
            or self.monai_version != "1.4.0"
            or self.monai_tag != "1.4.0"
            or self.monai_tag_object != MONAI_TAG_OBJECT
            or self.monai_commit != MONAI_COMMIT
        ):
            raise EvidenceError("untrusted_preprocessing_source", self.plan_id)
        if (
            self.channel_order != CHANNEL_ORDER
            or self.label_role != "label"
            or not self.label_excluded
            or self.spatial_transforms
            or self.loader != "NibabelReader"
            or self.as_closest_canonical
            or not self.scaled_data_before_cast
            or self.effective_dtype != "<f4"
            or not self.normalization_nonzero
            or not self.normalization_channel_wise
            or not self.normalization_population_std
            or not self.background_zero_preserved
            or self.empty_channel_policy != "refuse"
            or self.zero_variance_policy != "refuse"
            or self.finite_policy != "input-and-output-required"
            or self.output_layout != "C,I,J,K"
            or self.output_order != "C"
            or self.output_dtype != "<f4"
        ):
            raise EvidenceError("unsafe_preprocessing_contract", self.plan_id)
        if (
            self.canonical_implementation != "direct-numpy-float32"
            or self.independent_implementation != "chunked-welford-float64-statistics"
            or self.chunk_depth != 8
            or self.comparison_atol != COMPARISON_ATOL
            or self.comparison_rtol != COMPARISON_RTOL
            or not self.exact_byte_comparison_recorded
            or self.dependency_oracle != "official-source-derived"
            or self.monai_runtime_dependency_added
        ):
            raise EvidenceError("unsafe_comparison_policy", self.plan_id)
        if (
            not self.one_shot_execution
            or not self.private_output
            or not self.no_clobber
            or self.network_allowed
            or self.model_loading_allowed
            or self.inference_authorized
        ):
            raise EvidenceError("unsafe_execution_policy", self.plan_id)
        if self.runtime_identity.uv_lock_sha256 != UV_LOCK_SHA256:
            raise EvidenceError("untrusted_preprocessing_runtime", self.plan_id)
        if len(self.monai_sources) != 5 or len({item.path for item in self.monai_sources}) != 5:
            raise EvidenceError("incomplete_monai_sources", self.plan_id)
        if not self.observed_behavior or not self.assumptions:
            raise EvidenceError("incomplete_source_analysis", self.plan_id)
        expected_implementations = {
            "src/voxelscope/preprocessing.py",
            "src/voxelscope/preprocessing_records.py",
            "src/voxelscope/cli.py",
        }
        if {item.path for item in self.implementation_sources} != expected_implementations:
            raise EvidenceError("incomplete_implementation_identity", self.plan_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterPlan:
        expected = {
            "as_closest_canonical",
            "assumptions",
            "background_zero_preserved",
            "canonical_implementation",
            "channel_order",
            "chunk_depth",
            "comparison_atol",
            "comparison_rtol",
            "dependency_oracle",
            "effective_dtype",
            "empty_channel_policy",
            "exact_byte_comparison_recorded",
            "finite_policy",
            "implementation_sources",
            "independent_implementation",
            "inference_authorized",
            "label_excluded",
            "label_role",
            "loader",
            "milestone4_bundle_sha256",
            "model_bundle_version",
            "model_inference_config_sha256",
            "model_loading_allowed",
            "model_zoo_commit",
            "monai_commit",
            "monai_runtime_dependency_added",
            "monai_sources",
            "monai_tag",
            "monai_tag_object",
            "monai_version",
            "network_allowed",
            "no_clobber",
            "normalization_channel_wise",
            "normalization_nonzero",
            "normalization_population_std",
            "observed_behavior",
            "one_shot_execution",
            "output_dtype",
            "output_layout",
            "output_order",
            "plan_id",
            "private_output",
            "runtime_identity",
            "scaled_data_before_cast",
            "schema_version",
            "spatial_transforms",
            "zero_variance_policy",
        }
        data = strict_fields(data, expected, "AdapterPlan")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            require_string(data["plan_id"], "plan_id"),
            require_string(data["milestone4_bundle_sha256"], "milestone4_bundle_sha256"),
            cast(Any, require_string(data["model_bundle_version"], "model_bundle_version")),
            require_string(data["model_zoo_commit"], "model_zoo_commit"),
            require_string(
                data["model_inference_config_sha256"],
                "model_inference_config_sha256",
            ),
            cast(Any, require_string(data["monai_version"], "monai_version")),
            cast(Any, require_string(data["monai_tag"], "monai_tag")),
            require_string(data["monai_tag_object"], "monai_tag_object"),
            require_string(data["monai_commit"], "monai_commit"),
            tuple(
                SourceIdentity.from_dict(require_object(item, "MONAI source"))
                for item in require_list(data["monai_sources"], "monai_sources")
            ),
            _strings(data["observed_behavior"], "observed_behavior"),
            _strings(data["assumptions"], "assumptions"),
            _strings(data["channel_order"], "channel_order"),
            cast(Any, require_string(data["label_role"], "label_role")),
            cast(Any, require_bool(data["label_excluded"], "label_excluded")),
            _strings(data["spatial_transforms"], "spatial_transforms"),
            cast(Any, require_string(data["loader"], "loader")),
            cast(
                Any,
                require_bool(data["as_closest_canonical"], "as_closest_canonical"),
            ),
            cast(
                Any,
                require_bool(data["scaled_data_before_cast"], "scaled_data_before_cast"),
            ),
            cast(Any, require_string(data["effective_dtype"], "effective_dtype")),
            cast(
                Any,
                require_bool(data["normalization_nonzero"], "normalization_nonzero"),
            ),
            cast(
                Any,
                require_bool(
                    data["normalization_channel_wise"],
                    "normalization_channel_wise",
                ),
            ),
            cast(
                Any,
                require_bool(
                    data["normalization_population_std"],
                    "normalization_population_std",
                ),
            ),
            cast(
                Any,
                require_bool(
                    data["background_zero_preserved"],
                    "background_zero_preserved",
                ),
            ),
            cast(
                Any,
                require_string(data["empty_channel_policy"], "empty_channel_policy"),
            ),
            cast(
                Any,
                require_string(data["zero_variance_policy"], "zero_variance_policy"),
            ),
            cast(Any, require_string(data["finite_policy"], "finite_policy")),
            cast(Any, require_string(data["output_layout"], "output_layout")),
            cast(Any, require_string(data["output_order"], "output_order")),
            cast(Any, require_string(data["output_dtype"], "output_dtype")),
            cast(
                Any,
                require_string(data["canonical_implementation"], "canonical_implementation"),
            ),
            cast(
                Any,
                require_string(
                    data["independent_implementation"],
                    "independent_implementation",
                ),
            ),
            require_int(data["chunk_depth"], "chunk_depth"),
            require_number(data["comparison_atol"], "comparison_atol"),
            require_number(data["comparison_rtol"], "comparison_rtol"),
            cast(
                Any,
                require_bool(
                    data["exact_byte_comparison_recorded"],
                    "exact_byte_comparison_recorded",
                ),
            ),
            cast(Any, require_string(data["dependency_oracle"], "dependency_oracle")),
            cast(
                Any,
                require_bool(
                    data["monai_runtime_dependency_added"],
                    "monai_runtime_dependency_added",
                ),
            ),
            cast(
                Any,
                require_bool(data["one_shot_execution"], "one_shot_execution"),
            ),
            cast(Any, require_bool(data["private_output"], "private_output")),
            cast(Any, require_bool(data["no_clobber"], "no_clobber")),
            cast(Any, require_bool(data["network_allowed"], "network_allowed")),
            cast(
                Any,
                require_bool(data["model_loading_allowed"], "model_loading_allowed"),
            ),
            cast(
                Any,
                require_bool(data["inference_authorized"], "inference_authorized"),
            ),
            RuntimeIdentity.from_dict(require_object(data["runtime_identity"], "runtime_identity")),
            tuple(
                ImplementationIdentity.from_dict(require_object(item, "implementation source"))
                for item in require_list(
                    data["implementation_sources"],
                    "implementation_sources",
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelStatistics:
    role: str
    input_sha256: str
    raw_dtype: str
    effective_dtype: Literal["<f4"]
    nonzero_voxel_count: int
    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if self.role not in CHANNEL_ORDER:
            raise EvidenceError("invalid_channel_role", self.role)
        require_sha256(self.input_sha256)
        try:
            if self.effective_dtype != "<f4" or not self.raw_dtype:
                raise ValueError
        except ValueError as exc:
            raise EvidenceError("invalid_channel_dtype", self.role) from exc
        if (
            self.nonzero_voxel_count <= 0
            or not math.isfinite(self.mean)
            or not math.isfinite(self.standard_deviation)
            or self.standard_deviation <= 0
        ):
            raise EvidenceError("invalid_channel_statistics", self.role)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelStatistics:
        data = strict_fields(
            data,
            {
                "effective_dtype",
                "input_sha256",
                "mean",
                "nonzero_voxel_count",
                "raw_dtype",
                "role",
                "standard_deviation",
            },
            "ChannelStatistics",
        )
        return cls(
            require_string(data["role"], "role"),
            require_string(data["input_sha256"], "input_sha256"),
            require_string(data["raw_dtype"], "raw_dtype"),
            cast(Any, require_string(data["effective_dtype"], "effective_dtype")),
            require_int(data["nonzero_voxel_count"], "nonzero_voxel_count"),
            require_number(data["mean"], "mean"),
            require_number(data["standard_deviation"], "standard_deviation"),
        )


@dataclass(frozen=True)
class TensorArtifact:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        require_sha256(self.sha256)
        if self.size_bytes <= 0:
            raise EvidenceError("invalid_tensor_size", self.path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorArtifact:
        data = strict_fields(data, {"path", "sha256", "size_bytes"}, "TensorArtifact")
        return cls(
            require_string(data["path"], "path"),
            require_string(data["sha256"], "sha256"),
            require_int(data["size_bytes"], "size_bytes"),
        )


@dataclass(frozen=True)
class ImplementationComparison:
    exact_bytes_equal: bool
    within_tolerance: Literal[True]
    atol: float
    rtol: float
    maximum_absolute_difference: float
    maximum_relative_difference: float

    def __post_init__(self) -> None:
        if (
            not self.within_tolerance
            or self.atol != COMPARISON_ATOL
            or self.rtol != COMPARISON_RTOL
            or not math.isfinite(self.maximum_absolute_difference)
            or not math.isfinite(self.maximum_relative_difference)
            or self.maximum_absolute_difference < 0
            or self.maximum_relative_difference < 0
        ):
            raise EvidenceError("invalid_implementation_comparison", "comparison")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationComparison:
        data = strict_fields(
            data,
            {
                "atol",
                "exact_bytes_equal",
                "maximum_absolute_difference",
                "maximum_relative_difference",
                "rtol",
                "within_tolerance",
            },
            "ImplementationComparison",
        )
        return cls(
            require_bool(data["exact_bytes_equal"], "exact_bytes_equal"),
            cast(Any, require_bool(data["within_tolerance"], "within_tolerance")),
            require_number(data["atol"], "atol"),
            require_number(data["rtol"], "rtol"),
            require_number(
                data["maximum_absolute_difference"],
                "maximum_absolute_difference",
            ),
            require_number(
                data["maximum_relative_difference"],
                "maximum_relative_difference",
            ),
        )


@dataclass(frozen=True)
class PreprocessingReport:
    schema_version: Literal["voxelscope/preprocessing-report/v1"]
    status: Literal["go"]
    plan_sha256: str
    decision_sha256: str
    custody_plan_sha256: str
    custody_receipt_sha256: str
    custody_structural_report_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    channel_order: tuple[str, ...]
    label_excluded: Literal[True]
    geometry_identity_sha256: str
    geometry_preserved: Literal[True]
    input_spatial_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_layout: Literal["C,I,J,K"]
    output_dtype: Literal["<f4"]
    output_order: Literal["C"]
    reference_statistics: tuple[ChannelStatistics, ...]
    independent_statistics: tuple[ChannelStatistics, ...]
    reference_tensor: TensorArtifact
    independent_tensor: TensorArtifact
    implementation_comparison: ImplementationComparison
    finite_input: Literal[True]
    finite_output: Literal[True]
    background_zero_preserved: Literal[True]
    runtime_identity: RuntimeIdentity
    implementation_sources: tuple[ImplementationIdentity, ...]
    monai_sources: tuple[SourceIdentity, ...]
    model_loaded: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]
    completed_at: str

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/preprocessing-report/v1" or self.status != "go":
            raise EvidenceError("unsafe_preprocessing_status", self.status)
        for value in (
            self.plan_sha256,
            self.decision_sha256,
            self.custody_plan_sha256,
            self.custody_receipt_sha256,
            self.custody_structural_report_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
            self.geometry_identity_sha256,
        ):
            require_sha256(value)
        if (
            self.channel_order != CHANNEL_ORDER
            or not self.label_excluded
            or not self.geometry_preserved
            or len(self.input_spatial_shape) != 3
            or any(item <= 0 for item in self.input_spatial_shape)
            or self.output_shape != (4, *self.input_spatial_shape)
            or self.output_layout != "C,I,J,K"
            or self.output_dtype != "<f4"
            or self.output_order != "C"
            or tuple(item.role for item in self.reference_statistics) != CHANNEL_ORDER
            or tuple(item.role for item in self.independent_statistics) != CHANNEL_ORDER
            or not self.finite_input
            or not self.finite_output
            or not self.background_zero_preserved
            or self.model_loaded
            or self.inference_run
            or self.inference_authorized
            or not self.completed_at
        ):
            raise EvidenceError("invalid_preprocessing_report", self.status)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingReport:
        data = strict_fields(
            data,
            {
                "attempt_record_sha256",
                "authorization_sha256",
                "background_zero_preserved",
                "channel_order",
                "completed_at",
                "custody_plan_sha256",
                "custody_receipt_sha256",
                "custody_structural_report_sha256",
                "decision_sha256",
                "finite_input",
                "finite_output",
                "geometry_identity_sha256",
                "geometry_preserved",
                "implementation_comparison",
                "implementation_sources",
                "independent_statistics",
                "independent_tensor",
                "inference_authorized",
                "inference_run",
                "input_spatial_shape",
                "label_excluded",
                "model_loaded",
                "monai_sources",
                "output_dtype",
                "output_layout",
                "output_order",
                "output_shape",
                "plan_sha256",
                "reference_statistics",
                "reference_tensor",
                "runtime_identity",
                "schema_version",
                "status",
            },
            "PreprocessingReport",
        )
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["decision_sha256"], "decision_sha256"),
            require_string(data["custody_plan_sha256"], "custody_plan_sha256"),
            require_string(data["custody_receipt_sha256"], "custody_receipt_sha256"),
            require_string(
                data["custody_structural_report_sha256"],
                "custody_structural_report_sha256",
            ),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            _strings(data["channel_order"], "channel_order"),
            cast(Any, require_bool(data["label_excluded"], "label_excluded")),
            require_string(
                data["geometry_identity_sha256"],
                "geometry_identity_sha256",
            ),
            cast(
                Any,
                require_bool(data["geometry_preserved"], "geometry_preserved"),
            ),
            _ints(data["input_spatial_shape"], "input_spatial_shape"),
            _ints(data["output_shape"], "output_shape"),
            cast(Any, require_string(data["output_layout"], "output_layout")),
            cast(Any, require_string(data["output_dtype"], "output_dtype")),
            cast(Any, require_string(data["output_order"], "output_order")),
            tuple(
                ChannelStatistics.from_dict(require_object(item, "reference statistics"))
                for item in require_list(
                    data["reference_statistics"],
                    "reference_statistics",
                )
            ),
            tuple(
                ChannelStatistics.from_dict(require_object(item, "independent statistics"))
                for item in require_list(
                    data["independent_statistics"],
                    "independent_statistics",
                )
            ),
            TensorArtifact.from_dict(require_object(data["reference_tensor"], "reference_tensor")),
            TensorArtifact.from_dict(
                require_object(data["independent_tensor"], "independent_tensor")
            ),
            ImplementationComparison.from_dict(
                require_object(
                    data["implementation_comparison"],
                    "implementation_comparison",
                )
            ),
            cast(Any, require_bool(data["finite_input"], "finite_input")),
            cast(Any, require_bool(data["finite_output"], "finite_output")),
            cast(
                Any,
                require_bool(
                    data["background_zero_preserved"],
                    "background_zero_preserved",
                ),
            ),
            RuntimeIdentity.from_dict(require_object(data["runtime_identity"], "runtime_identity")),
            tuple(
                ImplementationIdentity.from_dict(require_object(item, "implementation source"))
                for item in require_list(
                    data["implementation_sources"],
                    "implementation_sources",
                )
            ),
            tuple(
                SourceIdentity.from_dict(require_object(item, "MONAI source"))
                for item in require_list(data["monai_sources"], "monai_sources")
            ),
            cast(Any, require_bool(data["model_loaded"], "model_loaded")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(
                Any,
                require_bool(data["inference_authorized"], "inference_authorized"),
            ),
            require_string(data["completed_at"], "completed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessingRefusal:
    schema_version: Literal["voxelscope/preprocessing-refusal/v1"]
    status: Literal["refused"]
    error_code: str
    last_completed_stage: RefusalStage
    plan_sha256: str
    custody_receipt_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    completed_channel_count: int
    snapshot_published: bool
    model_loaded: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]
    refused_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/preprocessing-refusal/v1"
            or self.status != "refused"
            or not self.error_code
            or self.last_completed_stage not in _REFUSAL_STAGES
        ):
            raise EvidenceError("invalid_preprocessing_refusal", self.status)
        for value in (
            self.plan_sha256,
            self.custody_receipt_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
        ):
            require_sha256(value)
        if (
            not 0 <= self.completed_channel_count <= 4
            or self.model_loaded
            or self.inference_run
            or self.inference_authorized
            or not self.refused_at
        ):
            raise EvidenceError("invalid_preprocessing_refusal", self.error_code)
        stage_progress = {
            "attempt_record": self.completed_channel_count == 0 and not self.snapshot_published,
            "custody_verification": self.completed_channel_count == 0
            and not self.snapshot_published,
            "channel_loading": 0 <= self.completed_channel_count <= 3
            and not self.snapshot_published,
            "reference_normalization": 0 <= self.completed_channel_count <= 3
            and not self.snapshot_published,
            "independent_normalization": 0 <= self.completed_channel_count <= 4
            and not self.snapshot_published,
            "implementation_comparison": self.completed_channel_count == 4
            and not self.snapshot_published,
            "snapshot_publication": self.completed_channel_count == 4,
        }
        if not stage_progress[self.last_completed_stage]:
            raise EvidenceError("invalid_refusal_progress", self.error_code)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingRefusal:
        data = strict_fields(
            data,
            {
                "attempt_record_sha256",
                "authorization_sha256",
                "completed_channel_count",
                "custody_receipt_sha256",
                "error_code",
                "inference_authorized",
                "inference_run",
                "last_completed_stage",
                "model_loaded",
                "plan_sha256",
                "refused_at",
                "schema_version",
                "snapshot_published",
                "status",
            },
            "PreprocessingRefusal",
        )
        stage = require_string(data["last_completed_stage"], "last_completed_stage")
        if stage not in _REFUSAL_STAGES:
            raise EvidenceError("invalid_refusal_stage", stage)
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["error_code"], "error_code"),
            cast(RefusalStage, stage),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["custody_receipt_sha256"], "custody_receipt_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            require_int(data["completed_channel_count"], "completed_channel_count"),
            cast(
                Any,
                require_bool(data["snapshot_published"], "snapshot_published"),
            ),
            cast(Any, require_bool(data["model_loaded"], "model_loaded")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(
                Any,
                require_bool(data["inference_authorized"], "inference_authorized"),
            ),
            require_string(data["refused_at"], "refused_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
