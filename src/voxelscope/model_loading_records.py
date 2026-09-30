# SPDX-License-Identifier: Apache-2.0
"""Strict records for prospective Milestone 8 model-loading qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from .canonical import EvidenceError, require_sha256, safe_relative_path
from .model_loading_contract import (
    ACQUISITION_PLAN_SHA256,
    LEGACY_SYNTHETIC_BUNDLE_SHA256,
    MILESTONE6_PLAN_SHA256,
    MILESTONE6_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_REPORT_SHA256,
    MILESTONE7_PLAN_SHA256,
    MILESTONE7_PUBLIC_BUNDLE_SHA256,
    MODEL_ARCHIVE_SHA256,
    MODEL_CHECKPOINT_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_LICENSE_SHA256,
    MODEL_ZOO_COMMIT,
    MONAI_COMMIT,
    PYTORCH_COMMIT,
    SOURCE_REGISTRY_SHA256,
)
from .records import (
    require_bool,
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

ARCHIVE_PATH = "artifacts/monai/brats_mri_segmentation_v0.5.2.zip"
RECEIPT_PATH = "receipts/monai-brats-bundle-v0.5.2.json"
SNAPSHOT_NAME: Literal["model-loading-qualification-v1"] = "model-loading-qualification-v1"
ATTEMPTS_NAME = "model-loading-attempts"
REPORT_PATH = "evidence/model-loading-report.json"
MANIFEST_PATH = "evidence/extraction-manifest.json"
CHECKPOINT_PATH = "extracted/brats_mri_segmentation/models/model.pt"
CONFIG_PATH = "extracted/brats_mri_segmentation/configs/inference.json"
LICENSE_PATH = "extracted/brats_mri_segmentation/LICENSE"
REQUIREMENTS_PATH = "model-loading-requirements-py312.txt"
EXTRACTION_PATHS = (LICENSE_PATH, CONFIG_PATH, CHECKPOINT_PATH)

AttemptStage = Literal[
    "attempt_marker",
    "receipt_verification",
    "archive_staging",
    "archive_verification",
    "member_extraction",
    "loader_worker",
    "report_publication",
    "snapshot_publication",
]
_ATTEMPT_STAGES = frozenset(
    {
        "attempt_marker",
        "receipt_verification",
        "archive_staging",
        "archive_verification",
        "member_extraction",
        "loader_worker",
        "report_publication",
        "snapshot_publication",
    }
)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _ints(value: Any, name: str) -> tuple[int, ...]:
    return tuple(require_int(item, f"{name} item") for item in require_list(value, name))


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
        return cls(require_string(data["path"], "path"), require_string(data["sha256"], "sha256"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionMember:
    path: str
    role: Literal["license", "inference-config", "pytorch-checkpoint"]
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        if self.role not in {"license", "inference-config", "pytorch-checkpoint"}:
            raise EvidenceError("invalid_extraction_role", self.role)
        if self.size_bytes <= 0:
            raise EvidenceError("invalid_size", self.path)
        require_sha256(self.sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionMember:
        data = strict_fields(data, {"path", "role", "sha256", "size_bytes"}, "ExtractionMember")
        role = require_string(data["role"], "role")
        if role not in {"license", "inference-config", "pytorch-checkpoint"}:
            raise EvidenceError("invalid_extraction_role", role)
        return cls(
            require_string(data["path"], "path"),
            cast(Any, role),
            require_int(data["size_bytes"], "size_bytes"),
            require_string(data["sha256"], "sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CpuRuntimeIdentity:
    operating_system: Literal["linux"]
    python_version: Literal["3.12"]
    torch_version: Literal["2.4.0"]
    torch_commit: str
    monai_version: Literal["1.4.0"]
    monai_commit: str
    device: Literal["cpu"]
    requirements_path: Literal["model-loading-requirements-py312.txt"]
    requirements_sha256: str

    def __post_init__(self) -> None:
        if (
            self.operating_system != "linux"
            or self.python_version != "3.12"
            or self.torch_version != "2.4.0"
            or self.torch_commit != PYTORCH_COMMIT
            or self.monai_version != "1.4.0"
            or self.monai_commit != MONAI_COMMIT
            or self.device != "cpu"
            or self.requirements_path != REQUIREMENTS_PATH
        ):
            raise EvidenceError("untrusted_model_runtime", "runtime identity differs")
        require_sha256(self.requirements_sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CpuRuntimeIdentity:
        data = strict_fields(
            data,
            {
                "device",
                "monai_commit",
                "monai_version",
                "operating_system",
                "python_version",
                "requirements_path",
                "requirements_sha256",
                "torch_commit",
                "torch_version",
            },
            "CpuRuntimeIdentity",
        )
        return cls(
            cast(Any, require_string(data["operating_system"], "operating_system")),
            cast(Any, require_string(data["python_version"], "python_version")),
            cast(Any, require_string(data["torch_version"], "torch_version")),
            require_string(data["torch_commit"], "torch_commit"),
            cast(Any, require_string(data["monai_version"], "monai_version")),
            require_string(data["monai_commit"], "monai_commit"),
            cast(Any, require_string(data["device"], "device")),
            cast(Any, require_string(data["requirements_path"], "requirements_path")),
            require_string(data["requirements_sha256"], "requirements_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureContract:
    constructor: Literal["monai.networks.nets.SegResNet"]
    spatial_dims: Literal[3]
    init_filters: Literal[16]
    in_channels: Literal[4]
    out_channels: Literal[3]
    dropout_prob: float
    blocks_down: tuple[int, ...]
    blocks_up: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.constructor != "monai.networks.nets.SegResNet"
            or self.spatial_dims != 3
            or self.init_filters != 16
            or self.in_channels != 4
            or self.out_channels != 3
            or self.dropout_prob != 0.2
            or self.blocks_down != (1, 2, 2, 4)
            or self.blocks_up != (1, 1, 1)
        ):
            raise EvidenceError("untrusted_architecture", self.constructor)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureContract:
        data = strict_fields(
            data,
            {
                "blocks_down",
                "blocks_up",
                "constructor",
                "dropout_prob",
                "in_channels",
                "init_filters",
                "out_channels",
                "spatial_dims",
            },
            "ArchitectureContract",
        )
        dropout = data["dropout_prob"]
        if type(dropout) not in {int, float}:
            raise EvidenceError("invalid_json_type", "dropout_prob must be a number")
        return cls(
            cast(Any, require_string(data["constructor"], "constructor")),
            cast(Any, require_int(data["spatial_dims"], "spatial_dims")),
            cast(Any, require_int(data["init_filters"], "init_filters")),
            cast(Any, require_int(data["in_channels"], "in_channels")),
            cast(Any, require_int(data["out_channels"], "out_channels")),
            float(dropout),
            _ints(data["blocks_down"], "blocks_down"),
            _ints(data["blocks_up"], "blocks_up"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelLoadingPlan:
    schema_version: Literal["voxelscope/model-loading-plan/v1"]
    plan_id: Literal["voxelscope-milestone-8-model-loading-v1"]
    source_registry_sha256: str
    acquisition_plan_sha256: str
    milestone7_plan_sha256: str
    milestone7_public_bundle_sha256: str
    milestone6_plan_sha256: str
    milestone6_report_sha256: str
    milestone6_public_bundle_sha256: str
    legacy_synthetic_bundle_sha256: str
    model_zoo_commit: str
    archive_path: Literal["artifacts/monai/brats_mri_segmentation_v0.5.2.zip"]
    archive_size_bytes: Literal[35082630]
    archive_sha256: str
    archive_max_members: Literal[32]
    archive_max_total_uncompressed_bytes: Literal[50000000]
    archive_max_member_bytes: Literal[20000000]
    archive_max_compression_ratio: Literal[20]
    extraction_members: tuple[ExtractionMember, ...]
    architecture: ArchitectureContract
    runtime: CpuRuntimeIdentity
    checkpoint_container: Literal["single-model-state-dict"]
    weights_only_required: Literal[True]
    strict_state_dict_required: Literal[True]
    finite_parameters_required: Literal[True]
    cpu_only: Literal[True]
    one_shot_execution: Literal[True]
    no_clobber: Literal[True]
    private_output: Literal[True]
    network_allowed: Literal[False]
    accelerator_allowed: Literal[False]
    forward_allowed: Literal[False]
    inference_authorized: Literal[False]
    loader_timeout_seconds: Literal[120]
    loader_memory_limit_bytes: Literal[2147483648]
    implementation_sources: tuple[ImplementationIdentity, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/model-loading-plan/v1"
            or self.plan_id != "voxelscope-milestone-8-model-loading-v1"
        ):
            raise EvidenceError("unsupported_schema", self.schema_version)
        expected = (
            (self.source_registry_sha256, SOURCE_REGISTRY_SHA256),
            (self.acquisition_plan_sha256, ACQUISITION_PLAN_SHA256),
            (self.milestone7_plan_sha256, MILESTONE7_PLAN_SHA256),
            (self.milestone7_public_bundle_sha256, MILESTONE7_PUBLIC_BUNDLE_SHA256),
            (self.milestone6_plan_sha256, MILESTONE6_PLAN_SHA256),
            (self.milestone6_report_sha256, MILESTONE6_REPORT_SHA256),
            (self.milestone6_public_bundle_sha256, MILESTONE6_PUBLIC_BUNDLE_SHA256),
            (self.legacy_synthetic_bundle_sha256, LEGACY_SYNTHETIC_BUNDLE_SHA256),
            (self.archive_sha256, MODEL_ARCHIVE_SHA256),
        )
        if any(observed != trusted for observed, trusted in expected):
            raise EvidenceError("untrusted_model_loading_identity", self.plan_id)
        if (
            self.model_zoo_commit != MODEL_ZOO_COMMIT
            or self.archive_path != ARCHIVE_PATH
            or self.archive_size_bytes != 35082630
            or self.archive_max_members != 32
            or self.archive_max_total_uncompressed_bytes != 50000000
            or self.archive_max_member_bytes != 20000000
            or self.archive_max_compression_ratio != 20
            or self.checkpoint_container != "single-model-state-dict"
            or not self.weights_only_required
            or not self.strict_state_dict_required
            or not self.finite_parameters_required
            or not self.cpu_only
            or not self.one_shot_execution
            or not self.no_clobber
            or not self.private_output
            or self.network_allowed
            or self.accelerator_allowed
            or self.forward_allowed
            or self.inference_authorized
            or self.loader_timeout_seconds != 120
            or self.loader_memory_limit_bytes != 2147483648
        ):
            raise EvidenceError("unsafe_model_loading_policy", self.plan_id)
        expected_members = {
            "brats_mri_segmentation/LICENSE": (
                "license",
                11357,
                MODEL_LICENSE_SHA256,
            ),
            "brats_mri_segmentation/configs/inference.json": (
                "inference-config",
                4257,
                MODEL_CONFIG_SHA256,
            ),
            "brats_mri_segmentation/models/model.pt": (
                "pytorch-checkpoint",
                18840620,
                MODEL_CHECKPOINT_SHA256,
            ),
        }
        observed_members = {
            member.path: (member.role, member.size_bytes, member.sha256)
            for member in self.extraction_members
        }
        if observed_members != expected_members:
            raise EvidenceError("unsafe_extraction_allowlist", self.plan_id)
        paths = [identity.path for identity in self.implementation_sources]
        if len(paths) != len(set(paths)) or not paths:
            raise EvidenceError("invalid_implementation_sources", self.plan_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelLoadingPlan:
        expected = {
            "accelerator_allowed",
            "acquisition_plan_sha256",
            "architecture",
            "archive_max_compression_ratio",
            "archive_max_member_bytes",
            "archive_max_members",
            "archive_max_total_uncompressed_bytes",
            "archive_path",
            "archive_sha256",
            "archive_size_bytes",
            "checkpoint_container",
            "cpu_only",
            "extraction_members",
            "finite_parameters_required",
            "forward_allowed",
            "implementation_sources",
            "inference_authorized",
            "legacy_synthetic_bundle_sha256",
            "loader_memory_limit_bytes",
            "loader_timeout_seconds",
            "milestone6_plan_sha256",
            "milestone6_public_bundle_sha256",
            "milestone6_report_sha256",
            "milestone7_plan_sha256",
            "milestone7_public_bundle_sha256",
            "model_zoo_commit",
            "network_allowed",
            "no_clobber",
            "one_shot_execution",
            "plan_id",
            "private_output",
            "runtime",
            "schema_version",
            "source_registry_sha256",
            "strict_state_dict_required",
            "weights_only_required",
        }
        data = strict_fields(data, expected, "ModelLoadingPlan")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["plan_id"], "plan_id")),
            require_string(data["source_registry_sha256"], "source_registry_sha256"),
            require_string(data["acquisition_plan_sha256"], "acquisition_plan_sha256"),
            require_string(data["milestone7_plan_sha256"], "milestone7_plan_sha256"),
            require_string(
                data["milestone7_public_bundle_sha256"], "milestone7_public_bundle_sha256"
            ),
            require_string(data["milestone6_plan_sha256"], "milestone6_plan_sha256"),
            require_string(data["milestone6_report_sha256"], "milestone6_report_sha256"),
            require_string(
                data["milestone6_public_bundle_sha256"], "milestone6_public_bundle_sha256"
            ),
            require_string(
                data["legacy_synthetic_bundle_sha256"], "legacy_synthetic_bundle_sha256"
            ),
            require_string(data["model_zoo_commit"], "model_zoo_commit"),
            cast(Any, require_string(data["archive_path"], "archive_path")),
            cast(Any, require_int(data["archive_size_bytes"], "archive_size_bytes")),
            require_string(data["archive_sha256"], "archive_sha256"),
            cast(Any, require_int(data["archive_max_members"], "archive_max_members")),
            cast(
                Any,
                require_int(
                    data["archive_max_total_uncompressed_bytes"],
                    "archive_max_total_uncompressed_bytes",
                ),
            ),
            cast(
                Any,
                require_int(data["archive_max_member_bytes"], "archive_max_member_bytes"),
            ),
            cast(
                Any,
                require_int(data["archive_max_compression_ratio"], "archive_max_compression_ratio"),
            ),
            tuple(
                ExtractionMember.from_dict(require_object(item, "extraction member"))
                for item in require_list(data["extraction_members"], "extraction_members")
            ),
            ArchitectureContract.from_dict(require_object(data["architecture"], "architecture")),
            CpuRuntimeIdentity.from_dict(require_object(data["runtime"], "runtime")),
            cast(Any, require_string(data["checkpoint_container"], "checkpoint_container")),
            cast(Any, require_bool(data["weights_only_required"], "weights_only_required")),
            cast(
                Any,
                require_bool(data["strict_state_dict_required"], "strict_state_dict_required"),
            ),
            cast(
                Any,
                require_bool(data["finite_parameters_required"], "finite_parameters_required"),
            ),
            cast(Any, require_bool(data["cpu_only"], "cpu_only")),
            cast(Any, require_bool(data["one_shot_execution"], "one_shot_execution")),
            cast(Any, require_bool(data["no_clobber"], "no_clobber")),
            cast(Any, require_bool(data["private_output"], "private_output")),
            cast(Any, require_bool(data["network_allowed"], "network_allowed")),
            cast(Any, require_bool(data["accelerator_allowed"], "accelerator_allowed")),
            cast(Any, require_bool(data["forward_allowed"], "forward_allowed")),
            cast(Any, require_bool(data["inference_authorized"], "inference_authorized")),
            cast(
                Any,
                require_int(data["loader_timeout_seconds"], "loader_timeout_seconds"),
            ),
            cast(
                Any,
                require_int(data["loader_memory_limit_bytes"], "loader_memory_limit_bytes"),
            ),
            tuple(
                ImplementationIdentity.from_dict(require_object(item, "implementation source"))
                for item in require_list(data["implementation_sources"], "implementation_sources")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoaderWorkerRequest:
    schema_version: Literal["voxelscope/loader-worker-request/v1"]
    operation: Literal["official-torch", "synthetic-json"]
    plan_sha256: str
    worker_source_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    config_path: str
    config_sha256: str
    architecture: ArchitectureContract
    runtime: CpuRuntimeIdentity
    runtime_manifest_sha256: str
    memory_limit_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/loader-worker-request/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.operation not in {"official-torch", "synthetic-json"}:
            raise EvidenceError("invalid_loader_operation", self.operation)
        for digest in (
            self.plan_sha256,
            self.worker_source_sha256,
            self.checkpoint_sha256,
            self.config_sha256,
            self.runtime_manifest_sha256,
        ):
            require_sha256(digest)
        if not self.checkpoint_path or not self.config_path or self.memory_limit_bytes <= 0:
            raise EvidenceError("invalid_loader_request", self.operation)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoaderWorkerRequest:
        data = strict_fields(
            data,
            {
                "architecture",
                "checkpoint_path",
                "checkpoint_sha256",
                "config_path",
                "config_sha256",
                "memory_limit_bytes",
                "operation",
                "plan_sha256",
                "runtime",
                "runtime_manifest_sha256",
                "schema_version",
                "worker_source_sha256",
            },
            "LoaderWorkerRequest",
        )
        operation = require_string(data["operation"], "operation")
        if operation not in {"official-torch", "synthetic-json"}:
            raise EvidenceError("invalid_loader_operation", operation)
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, operation),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["worker_source_sha256"], "worker_source_sha256"),
            require_string(data["checkpoint_path"], "checkpoint_path"),
            require_string(data["checkpoint_sha256"], "checkpoint_sha256"),
            require_string(data["config_path"], "config_path"),
            require_string(data["config_sha256"], "config_sha256"),
            ArchitectureContract.from_dict(require_object(data["architecture"], "architecture")),
            CpuRuntimeIdentity.from_dict(require_object(data["runtime"], "runtime")),
            require_string(data["runtime_manifest_sha256"], "runtime_manifest_sha256"),
            require_int(data["memory_limit_bytes"], "memory_limit_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoaderWorkerResult:
    schema_version: Literal["voxelscope/loader-worker-result/v1"]
    status: Literal["go", "refused"]
    error_code: str | None
    runtime_verified: bool
    weights_only_used: bool
    state_mapping_closed: bool
    strict_keys_verified: bool
    tensor_metadata_verified: bool
    finite_parameters_verified: bool
    cpu_only_verified: bool
    architecture_verified: bool
    output_channels_verified: bool
    parameter_tensor_count: int
    buffer_tensor_count: int
    parameter_element_count: int
    model_instantiated: bool
    state_loaded: bool
    forward_called: Literal[False]
    inference_run: Literal[False]
    accelerator_used: Literal[False]
    network_used: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/loader-worker-result/v1"
            or self.status not in {"go", "refused"}
            or (self.status == "go") == (self.error_code is not None)
            or min(
                self.parameter_tensor_count,
                self.buffer_tensor_count,
                self.parameter_element_count,
            )
            < 0
            or self.forward_called
            or self.inference_run
            or self.accelerator_used
            or self.network_used
        ):
            raise EvidenceError("invalid_loader_result", self.status)
        go_checks = (
            self.runtime_verified,
            self.weights_only_used,
            self.state_mapping_closed,
            self.strict_keys_verified,
            self.tensor_metadata_verified,
            self.finite_parameters_verified,
            self.cpu_only_verified,
            self.architecture_verified,
            self.output_channels_verified,
            self.model_instantiated,
            self.state_loaded,
        )
        if self.status == "go" and not all(go_checks):
            raise EvidenceError("success_shaped_loader_result", self.status)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoaderWorkerResult:
        expected = {
            "accelerator_used",
            "architecture_verified",
            "buffer_tensor_count",
            "cpu_only_verified",
            "error_code",
            "finite_parameters_verified",
            "forward_called",
            "inference_run",
            "model_instantiated",
            "network_used",
            "output_channels_verified",
            "parameter_element_count",
            "parameter_tensor_count",
            "runtime_verified",
            "schema_version",
            "state_loaded",
            "state_mapping_closed",
            "status",
            "strict_keys_verified",
            "tensor_metadata_verified",
            "weights_only_used",
        }
        data = strict_fields(data, expected, "LoaderWorkerResult")
        error = data["error_code"]
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(error, "error_code") if error is not None else None,
            require_bool(data["runtime_verified"], "runtime_verified"),
            require_bool(data["weights_only_used"], "weights_only_used"),
            require_bool(data["state_mapping_closed"], "state_mapping_closed"),
            require_bool(data["strict_keys_verified"], "strict_keys_verified"),
            require_bool(data["tensor_metadata_verified"], "tensor_metadata_verified"),
            require_bool(data["finite_parameters_verified"], "finite_parameters_verified"),
            require_bool(data["cpu_only_verified"], "cpu_only_verified"),
            require_bool(data["architecture_verified"], "architecture_verified"),
            require_bool(data["output_channels_verified"], "output_channels_verified"),
            require_int(data["parameter_tensor_count"], "parameter_tensor_count"),
            require_int(data["buffer_tensor_count"], "buffer_tensor_count"),
            require_int(data["parameter_element_count"], "parameter_element_count"),
            require_bool(data["model_instantiated"], "model_instantiated"),
            require_bool(data["state_loaded"], "state_loaded"),
            cast(Any, require_bool(data["forward_called"], "forward_called")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(Any, require_bool(data["accelerator_used"], "accelerator_used")),
            cast(Any, require_bool(data["network_used"], "network_used")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrivateArtifact:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        require_sha256(self.sha256)
        if self.size_bytes <= 0:
            raise EvidenceError("invalid_size", self.path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivateArtifact:
        data = strict_fields(data, {"path", "sha256", "size_bytes"}, "PrivateArtifact")
        return cls(
            require_string(data["path"], "path"),
            require_string(data["sha256"], "sha256"),
            require_int(data["size_bytes"], "size_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelLoadingAttempt:
    schema_version: Literal["voxelscope/model-loading-attempt/v1"]
    plan_sha256: str
    authorization_sha256: str
    receipt_sha256: str
    archive_sha256: str
    worker_python_sha256: str
    worker_runtime_sha256: str
    declared_private_reads: tuple[str, ...]
    declared_private_writes: tuple[str, ...]
    started_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/model-loading-attempt/v1"
            or not self.declared_private_reads
            or not self.declared_private_writes
            or not self.started_at
        ):
            raise EvidenceError("invalid_model_loading_attempt", self.schema_version)
        for digest in (
            self.plan_sha256,
            self.authorization_sha256,
            self.receipt_sha256,
            self.archive_sha256,
            self.worker_python_sha256,
            self.worker_runtime_sha256,
        ):
            require_sha256(digest)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelLoadingAttempt:
        data = strict_fields(
            data,
            {
                "archive_sha256",
                "authorization_sha256",
                "declared_private_reads",
                "declared_private_writes",
                "plan_sha256",
                "receipt_sha256",
                "schema_version",
                "started_at",
                "worker_python_sha256",
                "worker_runtime_sha256",
            },
            "ModelLoadingAttempt",
        )
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["receipt_sha256"], "receipt_sha256"),
            require_string(data["archive_sha256"], "archive_sha256"),
            require_string(data["worker_python_sha256"], "worker_python_sha256"),
            require_string(data["worker_runtime_sha256"], "worker_runtime_sha256"),
            _strings(data["declared_private_reads"], "declared_private_reads"),
            _strings(data["declared_private_writes"], "declared_private_writes"),
            require_string(data["started_at"], "started_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionManifest:
    schema_version: Literal["voxelscope/model-extraction-manifest/v1"]
    status: Literal["go"]
    plan_sha256: str
    archive_sha256: str
    receipt_sha256: str
    members: tuple[PrivateArtifact, ...]
    selected_member_count: Literal[3]
    arbitrary_members_extracted: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/model-extraction-manifest/v1"
            or self.status != "go"
            or self.selected_member_count != 3
            or self.arbitrary_members_extracted
            or tuple(member.path for member in self.members) != EXTRACTION_PATHS
        ):
            raise EvidenceError("invalid_extraction_manifest", self.status)
        for digest in (self.plan_sha256, self.archive_sha256, self.receipt_sha256):
            require_sha256(digest)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionManifest:
        data = strict_fields(
            data,
            {
                "arbitrary_members_extracted",
                "archive_sha256",
                "members",
                "plan_sha256",
                "receipt_sha256",
                "schema_version",
                "selected_member_count",
                "status",
            },
            "ExtractionManifest",
        )
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["archive_sha256"], "archive_sha256"),
            require_string(data["receipt_sha256"], "receipt_sha256"),
            tuple(
                PrivateArtifact.from_dict(require_object(item, "member"))
                for item in require_list(data["members"], "members")
            ),
            cast(
                Any,
                require_int(data["selected_member_count"], "selected_member_count"),
            ),
            cast(
                Any,
                require_bool(data["arbitrary_members_extracted"], "arbitrary_members_extracted"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelLoadingReport:
    schema_version: Literal["voxelscope/model-loading-report/v1"]
    status: Literal["go"]
    plan_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    receipt_sha256: str
    archive_sha256: str
    extraction_manifest: PrivateArtifact
    loader_result: LoaderWorkerResult
    runtime: CpuRuntimeIdentity
    model_instantiated: Literal[True]
    model_loaded: Literal[True]
    forward_called: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]
    accelerator_used: Literal[False]
    network_used: Literal[False]
    completed_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/model-loading-report/v1"
            or self.status != "go"
            or self.loader_result.status != "go"
            or not self.model_instantiated
            or not self.model_loaded
            or self.forward_called
            or self.inference_run
            or self.inference_authorized
            or self.accelerator_used
            or self.network_used
            or not self.completed_at
        ):
            raise EvidenceError("invalid_model_loading_report", self.status)
        for digest in (
            self.plan_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
            self.receipt_sha256,
            self.archive_sha256,
        ):
            require_sha256(digest)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelLoadingReport:
        expected = {
            "accelerator_used",
            "archive_sha256",
            "attempt_record_sha256",
            "authorization_sha256",
            "completed_at",
            "extraction_manifest",
            "forward_called",
            "inference_authorized",
            "inference_run",
            "loader_result",
            "model_instantiated",
            "model_loaded",
            "network_used",
            "plan_sha256",
            "receipt_sha256",
            "runtime",
            "schema_version",
            "status",
        }
        data = strict_fields(data, expected, "ModelLoadingReport")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            require_string(data["receipt_sha256"], "receipt_sha256"),
            require_string(data["archive_sha256"], "archive_sha256"),
            PrivateArtifact.from_dict(
                require_object(data["extraction_manifest"], "extraction_manifest")
            ),
            LoaderWorkerResult.from_dict(require_object(data["loader_result"], "loader_result")),
            CpuRuntimeIdentity.from_dict(require_object(data["runtime"], "runtime")),
            cast(Any, require_bool(data["model_instantiated"], "model_instantiated")),
            cast(Any, require_bool(data["model_loaded"], "model_loaded")),
            cast(Any, require_bool(data["forward_called"], "forward_called")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(Any, require_bool(data["inference_authorized"], "inference_authorized")),
            cast(Any, require_bool(data["accelerator_used"], "accelerator_used")),
            cast(Any, require_bool(data["network_used"], "network_used")),
            require_string(data["completed_at"], "completed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelLoadingCompletion:
    schema_version: Literal["voxelscope/model-loading-completion/v1"]
    status: Literal["completed"]
    plan_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    receipt_sha256: str
    archive_sha256: str
    snapshot_sha256: str
    report_sha256: str
    snapshot: Literal["model-loading-qualification-v1"]
    forward_called: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]
    completed_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/model-loading-completion/v1"
            or self.status != "completed"
            or self.snapshot != SNAPSHOT_NAME
            or self.forward_called
            or self.inference_run
            or self.inference_authorized
            or not self.completed_at
        ):
            raise EvidenceError("invalid_model_loading_completion", self.status)
        for digest in (
            self.plan_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
            self.receipt_sha256,
            self.archive_sha256,
            self.snapshot_sha256,
            self.report_sha256,
        ):
            require_sha256(digest)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelLoadingCompletion:
        expected = {
            "archive_sha256",
            "attempt_record_sha256",
            "authorization_sha256",
            "completed_at",
            "forward_called",
            "inference_authorized",
            "inference_run",
            "plan_sha256",
            "receipt_sha256",
            "report_sha256",
            "schema_version",
            "snapshot",
            "snapshot_sha256",
            "status",
        }
        data = strict_fields(data, expected, "ModelLoadingCompletion")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            require_string(data["receipt_sha256"], "receipt_sha256"),
            require_string(data["archive_sha256"], "archive_sha256"),
            require_string(data["snapshot_sha256"], "snapshot_sha256"),
            require_string(data["report_sha256"], "report_sha256"),
            cast(Any, require_string(data["snapshot"], "snapshot")),
            cast(Any, require_bool(data["forward_called"], "forward_called")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(Any, require_bool(data["inference_authorized"], "inference_authorized")),
            require_string(data["completed_at"], "completed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelLoadingRefusal:
    schema_version: Literal["voxelscope/model-loading-refusal/v1"]
    status: Literal["refused"]
    error_code: str
    last_completed_stage: AttemptStage
    plan_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    receipt_sha256: str
    archive_sha256: str
    extracted_member_count: int
    snapshot_published: bool
    model_instantiated: bool
    model_loaded: bool
    forward_called: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]
    refused_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/model-loading-refusal/v1"
            or self.status != "refused"
            or not self.error_code
            or self.last_completed_stage not in _ATTEMPT_STAGES
            or not 0 <= self.extracted_member_count <= 3
            or self.forward_called
            or self.inference_run
            or self.inference_authorized
            or not self.refused_at
        ):
            raise EvidenceError("invalid_model_loading_refusal", self.status)
        for digest in (
            self.plan_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
            self.receipt_sha256,
            self.archive_sha256,
        ):
            require_sha256(digest)
        if self.model_loaded and not self.model_instantiated:
            raise EvidenceError("invalid_refusal_progress", self.error_code)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelLoadingRefusal:
        expected = {
            "archive_sha256",
            "attempt_record_sha256",
            "authorization_sha256",
            "error_code",
            "extracted_member_count",
            "forward_called",
            "inference_authorized",
            "inference_run",
            "last_completed_stage",
            "model_instantiated",
            "model_loaded",
            "plan_sha256",
            "receipt_sha256",
            "refused_at",
            "schema_version",
            "snapshot_published",
            "status",
        }
        data = strict_fields(data, expected, "ModelLoadingRefusal")
        stage = require_string(data["last_completed_stage"], "last_completed_stage")
        if stage not in _ATTEMPT_STAGES:
            raise EvidenceError("invalid_refusal_stage", stage)
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["error_code"], "error_code"),
            cast(AttemptStage, stage),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            require_string(data["receipt_sha256"], "receipt_sha256"),
            require_string(data["archive_sha256"], "archive_sha256"),
            require_int(data["extracted_member_count"], "extracted_member_count"),
            require_bool(data["snapshot_published"], "snapshot_published"),
            require_bool(data["model_instantiated"], "model_instantiated"),
            require_bool(data["model_loaded"], "model_loaded"),
            cast(Any, require_bool(data["forward_called"], "forward_called")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(Any, require_bool(data["inference_authorized"], "inference_authorized")),
            require_string(data["refused_at"], "refused_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
