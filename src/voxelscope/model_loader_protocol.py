# SPDX-License-Identifier: Apache-2.0
"""Self-contained typed protocol used by the isolated model loader worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from .canonical import EvidenceError, require_sha256
from .records import (
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)


def _ints(value: Any, name: str) -> tuple[int, ...]:
    return tuple(require_int(item, f"{name} item") for item in require_list(value, name))


@dataclass(frozen=True)
class WorkerArchitecture:
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
    def from_dict(cls, data: dict[str, Any]) -> WorkerArchitecture:
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
            "WorkerArchitecture",
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


@dataclass(frozen=True)
class WorkerRuntime:
    operating_system: Literal["linux"]
    architecture: Literal["arm64"]
    python_version: Literal["3.12.14"]
    torch_version: Literal["2.4.0"]
    pytorch_tag: Literal["v2.4.0"]
    pytorch_tag_commit: Literal["d990dada86a8ad94882b5c23e859b88c0c255bda"]
    pytorch_wheel_source_commit: Literal["e4ee3be4063b7c430974252fdf7db42273388d86"]
    monai_version: Literal["1.4.0"]
    monai_commit: Literal["46a5272196a6c2590ca2589029eed8e4d56ff008"]
    device: Literal["cpu"]
    requirements_path: Literal["model-loading-requirements-py312.txt"]
    requirements_sha256: str
    runtime_build_path: Literal["research/model-loading-runtime-build-v1.json"]
    runtime_build_sha256: str

    def __post_init__(self) -> None:
        if (
            self.operating_system != "linux"
            or self.architecture != "arm64"
            or self.python_version != "3.12.14"
            or self.torch_version != "2.4.0"
            or self.pytorch_tag != "v2.4.0"
            or self.pytorch_tag_commit != "d990dada86a8ad94882b5c23e859b88c0c255bda"
            or self.pytorch_wheel_source_commit != "e4ee3be4063b7c430974252fdf7db42273388d86"
            or self.monai_version != "1.4.0"
            or self.monai_commit != "46a5272196a6c2590ca2589029eed8e4d56ff008"
            or self.device != "cpu"
            or self.requirements_path != "model-loading-requirements-py312.txt"
            or self.runtime_build_path != "research/model-loading-runtime-build-v1.json"
        ):
            raise EvidenceError("untrusted_model_runtime", "runtime identity differs")
        require_sha256(self.requirements_sha256)
        require_sha256(self.runtime_build_sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerRuntime:
        data = strict_fields(
            data,
            {
                "architecture",
                "device",
                "monai_commit",
                "monai_version",
                "operating_system",
                "python_version",
                "requirements_path",
                "requirements_sha256",
                "pytorch_tag",
                "pytorch_tag_commit",
                "pytorch_wheel_source_commit",
                "torch_version",
                "runtime_build_path",
                "runtime_build_sha256",
            },
            "WorkerRuntime",
        )
        return cls(
            cast(Any, require_string(data["operating_system"], "operating_system")),
            cast(Any, require_string(data["architecture"], "architecture")),
            cast(Any, require_string(data["python_version"], "python_version")),
            cast(Any, require_string(data["torch_version"], "torch_version")),
            cast(Any, require_string(data["pytorch_tag"], "pytorch_tag")),
            cast(Any, require_string(data["pytorch_tag_commit"], "pytorch_tag_commit")),
            cast(
                Any,
                require_string(
                    data["pytorch_wheel_source_commit"],
                    "pytorch_wheel_source_commit",
                ),
            ),
            cast(Any, require_string(data["monai_version"], "monai_version")),
            cast(Any, require_string(data["monai_commit"], "monai_commit")),
            cast(Any, require_string(data["device"], "device")),
            cast(Any, require_string(data["requirements_path"], "requirements_path")),
            require_string(data["requirements_sha256"], "requirements_sha256"),
            cast(Any, require_string(data["runtime_build_path"], "runtime_build_path")),
            require_string(data["runtime_build_sha256"], "runtime_build_sha256"),
        )


@dataclass(frozen=True)
class WorkerRequest:
    schema_version: Literal["voxelscope/loader-worker-request/v1"]
    operation: Literal["official-torch", "synthetic-json"]
    plan_sha256: str
    worker_source_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    config_path: str
    config_sha256: str
    architecture: WorkerArchitecture
    runtime: WorkerRuntime
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
    def from_dict(cls, data: dict[str, Any]) -> WorkerRequest:
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
            "WorkerRequest",
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
            WorkerArchitecture.from_dict(require_object(data["architecture"], "architecture")),
            WorkerRuntime.from_dict(require_object(data["runtime"], "runtime")),
            require_string(data["runtime_manifest_sha256"], "runtime_manifest_sha256"),
            require_int(data["memory_limit_bytes"], "memory_limit_bytes"),
        )


@dataclass(frozen=True)
class WorkerResult:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
