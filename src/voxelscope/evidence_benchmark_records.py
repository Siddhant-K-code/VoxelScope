# SPDX-License-Identifier: Apache-2.0
"""Strict records for auditable evidence communication benchmarks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .canonical import EvidenceError, require_sha256, safe_relative_path
from .evidence_communication_records import (
    ARTIFACT_TERMINAL_STATES,
    InvalidModelOutput,
    ModelIdentity,
)
from .records import (
    require_int,
    require_list,
    require_number,
    require_object,
    require_string,
    strict_fields,
)

BENCHMARK_FIXTURE_SCHEMA = "voxelscope/evidence-communication-benchmark-fixture/v3"
BENCHMARK_SCHEMA = "voxelscope/evidence-communication-benchmark/v3"
BENCHMARK_INDEX_SCHEMA = "voxelscope/evidence-communication-benchmark-index/v3"
BENCHMARK_RECEIPT_SCHEMA = "voxelscope/evidence-communication-benchmark-receipt/v3"
RUNNER_MEASUREMENT_SCHEMA = "voxelscope/evidence-communication-runner-measurement/v3"
INVALID_OUTPUT_RECORD_SCHEMA = "voxelscope/evidence-communication-invalid-output/v3"

PARSED_RUN_ARTIFACT_TYPES = (
    "communication_request",
    "model_draft",
    "verified_artifact",
    "communication_receipt",
    "runner_measurement",
)
INVALID_RUN_ARTIFACT_TYPES = (
    "communication_request",
    "invalid_model_output",
    "verified_artifact",
    "communication_receipt",
    "runner_measurement",
)
_CASE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_RUN_FILENAMES = {
    "benchmark_index": "index.json",
    "benchmark_report": "benchmark.json",
    "communication_request": "request.json",
    "model_draft": "draft.json",
    "invalid_model_output": "invalid-output.json",
    "verified_artifact": "artifact.json",
    "communication_receipt": "receipt.json",
    "runner_measurement": "measurement.json",
}


def benchmark_run_path(case_id: str, repeat_index: int) -> str:
    if _CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise EvidenceError("unsafe_benchmark_case_id", case_id)
    if repeat_index < 0:
        raise EvidenceError("invalid_repeat_index", str(repeat_index))
    return f"runs/{case_id}/repeat-{repeat_index:04d}"


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return require_number(value, name)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return require_int(value, name)


@dataclass(frozen=True)
class RunnerOption:
    name: str
    value: bool | int | float | str

    def __post_init__(self) -> None:
        if not self.name or _CASE_ID_PATTERN.fullmatch(self.name.replace("_", "-")) is None:
            raise EvidenceError("invalid_runner_option", self.name)
        if type(self.value) not in {bool, int, float, str}:
            raise EvidenceError("invalid_runner_option", self.name)
        if isinstance(self.value, float):
            require_number(self.value, self.name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunnerOption:
        value = strict_fields(data, {"name", "value"}, "RunnerOption")
        name = require_string(value["name"], "runner option name")
        option_value = value["value"]
        if type(option_value) not in {bool, int, float, str}:
            raise EvidenceError("invalid_runner_option", name)
        return cls(name, option_value)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class DeclaredEnvironmentSetting:
    name: str
    provenance: str
    value: str

    def __post_init__(self) -> None:
        if self.provenance != "declared_unverified":
            raise EvidenceError("invalid_setting_provenance", self.provenance)
        if not self.name or not self.value:
            raise EvidenceError("invalid_declared_environment", self.name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeclaredEnvironmentSetting:
        value = strict_fields(
            data,
            {"name", "provenance", "value"},
            "DeclaredEnvironmentSetting",
        )
        return cls(
            require_string(value["name"], "environment name"),
            require_string(value["provenance"], "environment provenance"),
            require_string(value["value"], "environment value"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "provenance": self.provenance,
            "value": self.value,
        }


@dataclass(frozen=True)
class RunnerConfiguration:
    context_window: int | None
    declared_environment: tuple[DeclaredEnvironmentSetting, ...]
    endpoint_locality: str
    json_mode: str
    options: tuple[RunnerOption, ...]
    temperature: float | None
    timeout_seconds: float | None

    def __post_init__(self) -> None:
        if self.context_window is not None and self.context_window <= 0:
            raise EvidenceError("invalid_context_window", str(self.context_window))
        if self.endpoint_locality not in {"localhost_only", "not_applicable"}:
            raise EvidenceError("invalid_endpoint_locality", self.endpoint_locality)
        if self.json_mode not in {"strict_json", "recorded_fixture"}:
            raise EvidenceError("invalid_json_mode", self.json_mode)
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise EvidenceError("invalid_timeout", str(self.timeout_seconds))
        names = tuple(item.name for item in self.options)
        if len(names) != len(set(names)) or names != tuple(sorted(names)):
            raise EvidenceError("invalid_runner_options", "options must be sorted and unique")
        settings = tuple(item.name for item in self.declared_environment)
        if len(settings) != len(set(settings)) or settings != tuple(sorted(settings)):
            raise EvidenceError(
                "invalid_declared_environment",
                "settings must be sorted and unique",
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunnerConfiguration:
        value = strict_fields(
            data,
            {
                "context_window",
                "declared_environment",
                "endpoint_locality",
                "json_mode",
                "options",
                "temperature",
                "timeout_seconds",
            },
            "RunnerConfiguration",
        )
        return cls(
            _optional_int(value["context_window"], "context_window"),
            tuple(
                DeclaredEnvironmentSetting.from_dict(
                    require_object(item, "declared environment setting")
                )
                for item in require_list(
                    value["declared_environment"],
                    "declared_environment",
                )
            ),
            require_string(value["endpoint_locality"], "endpoint_locality"),
            require_string(value["json_mode"], "json_mode"),
            tuple(
                RunnerOption.from_dict(require_object(item, "runner option"))
                for item in require_list(value["options"], "options")
            ),
            _optional_number(value["temperature"], "temperature"),
            _optional_number(value["timeout_seconds"], "timeout_seconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_window": self.context_window,
            "declared_environment": [item.to_dict() for item in self.declared_environment],
            "endpoint_locality": self.endpoint_locality,
            "json_mode": self.json_mode,
            "options": [item.to_dict() for item in self.options],
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class MeasurementValue:
    availability: str
    reason: str | None
    value: int | float | None

    def __post_init__(self) -> None:
        if self.availability == "available":
            if self.reason is not None or self.value is None:
                raise EvidenceError("invalid_measurement", "available value is incomplete")
            require_number(self.value, "measurement value")
        elif self.availability == "unavailable":
            if self.reason is None or self.value is not None:
                raise EvidenceError("invalid_measurement", "unavailable reason is incomplete")
        else:
            raise EvidenceError("invalid_measurement", self.availability)

    @classmethod
    def available(cls, value: int | float) -> MeasurementValue:
        return cls("available", None, value)

    @classmethod
    def unavailable(cls, reason: str) -> MeasurementValue:
        return cls("unavailable", reason, None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementValue:
        value = strict_fields(data, {"availability", "reason", "value"}, "MeasurementValue")
        raw_value = value["value"]
        if raw_value is not None and type(raw_value) not in {int, float}:
            raise EvidenceError("invalid_measurement", "value must be numeric or null")
        raw_reason = value["reason"]
        if raw_reason is not None and type(raw_reason) is not str:
            raise EvidenceError("invalid_measurement", "reason must be a string or null")
        return cls(
            require_string(value["availability"], "availability"),
            raw_reason,
            raw_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability,
            "reason": self.reason,
            "value": self.value,
        }


@dataclass(frozen=True)
class RunnerMeasurement:
    case_id: str
    input_tokens: MeasurementValue
    latency_ms: MeasurementValue
    model_identity: ModelIdentity
    output_tokens: MeasurementValue
    peak_memory_mb: MeasurementValue
    peak_metal_memory_mb: MeasurementValue
    repeat_index: int
    runner_configuration: RunnerConfiguration
    schema_version: str

    def __post_init__(self) -> None:
        benchmark_run_path(self.case_id, self.repeat_index)
        if self.schema_version != RUNNER_MEASUREMENT_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        for name, measurement in (
            ("input_tokens", self.input_tokens),
            ("latency_ms", self.latency_ms),
            ("output_tokens", self.output_tokens),
            ("peak_memory_mb", self.peak_memory_mb),
            ("peak_metal_memory_mb", self.peak_metal_memory_mb),
        ):
            if measurement.value is not None and measurement.value < 0:
                raise EvidenceError("invalid_measurement", name)
        for name, measurement in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if measurement.value is not None and type(measurement.value) is not int:
                raise EvidenceError("invalid_measurement", f"{name} must be an integer")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunnerMeasurement:
        value = strict_fields(
            data,
            {
                "case_id",
                "input_tokens",
                "latency_ms",
                "model_identity",
                "output_tokens",
                "peak_memory_mb",
                "peak_metal_memory_mb",
                "repeat_index",
                "runner_configuration",
                "schema_version",
            },
            "RunnerMeasurement",
        )
        return cls(
            require_string(value["case_id"], "case_id"),
            MeasurementValue.from_dict(require_object(value["input_tokens"], "input_tokens")),
            MeasurementValue.from_dict(require_object(value["latency_ms"], "latency_ms")),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            MeasurementValue.from_dict(require_object(value["output_tokens"], "output_tokens")),
            MeasurementValue.from_dict(require_object(value["peak_memory_mb"], "peak_memory_mb")),
            MeasurementValue.from_dict(
                require_object(value["peak_metal_memory_mb"], "peak_metal_memory_mb")
            ),
            require_int(value["repeat_index"], "repeat_index"),
            RunnerConfiguration.from_dict(
                require_object(value["runner_configuration"], "runner_configuration")
            ),
            require_string(value["schema_version"], "schema_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_tokens": self.input_tokens.to_dict(),
            "latency_ms": self.latency_ms.to_dict(),
            "model_identity": self.model_identity.to_dict(),
            "output_tokens": self.output_tokens.to_dict(),
            "peak_memory_mb": self.peak_memory_mb.to_dict(),
            "peak_metal_memory_mb": self.peak_metal_memory_mb.to_dict(),
            "repeat_index": self.repeat_index,
            "runner_configuration": self.runner_configuration.to_dict(),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class InvalidOutputRecord:
    case_id: str
    invalid_model_output: InvalidModelOutput
    model_identity: ModelIdentity
    repeat_index: int
    request_id: str
    schema_version: str
    terminal_state: str

    def __post_init__(self) -> None:
        benchmark_run_path(self.case_id, self.repeat_index)
        if self.schema_version != INVALID_OUTPUT_RECORD_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state != "refused":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvalidOutputRecord:
        value = strict_fields(
            data,
            {
                "case_id",
                "invalid_model_output",
                "model_identity",
                "repeat_index",
                "request_id",
                "schema_version",
                "terminal_state",
            },
            "InvalidOutputRecord",
        )
        return cls(
            require_string(value["case_id"], "case_id"),
            InvalidModelOutput.from_dict(
                require_object(value["invalid_model_output"], "invalid_model_output")
            ),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            require_int(value["repeat_index"], "repeat_index"),
            require_string(value["request_id"], "request_id"),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "invalid_model_output": self.invalid_model_output.to_dict(),
            "model_identity": self.model_identity.to_dict(),
            "repeat_index": self.repeat_index,
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class BenchmarkFile:
    artifact_type: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.artifact_type not in _RUN_FILENAMES:
            raise EvidenceError("invalid_benchmark_artifact_type", self.artifact_type)
        safe_relative_path(self.path)
        require_sha256(self.sha256, "benchmark file sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkFile:
        value = strict_fields(data, {"artifact_type", "path", "sha256"}, "BenchmarkFile")
        return cls(
            require_string(value["artifact_type"], "artifact_type"),
            require_string(value["path"], "path"),
            require_string(value["sha256"], "sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BenchmarkRunIndex:
    artifact_types: tuple[str, ...]
    case_id: str
    files: tuple[BenchmarkFile, ...]
    model_identity: ModelIdentity
    repeat_index: int
    run_path: str
    semantic_sha256: str
    terminal_state: str

    def __post_init__(self) -> None:
        expected_path = benchmark_run_path(self.case_id, self.repeat_index)
        if self.run_path != expected_path:
            raise EvidenceError("benchmark_run_path_mismatch", self.run_path)
        safe_relative_path(self.run_path)
        if self.terminal_state not in ARTIFACT_TERMINAL_STATES:
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.semantic_sha256, "semantic_sha256")
        expected_types = (
            INVALID_RUN_ARTIFACT_TYPES
            if "invalid_model_output" in self.artifact_types
            else PARSED_RUN_ARTIFACT_TYPES
        )
        if self.artifact_types != expected_types:
            raise EvidenceError("benchmark_artifact_set_mismatch", self.run_path)
        if tuple(item.artifact_type for item in self.files) != expected_types:
            raise EvidenceError("benchmark_file_order_mismatch", self.run_path)
        expected_paths = tuple(
            f"{self.run_path}/{_RUN_FILENAMES[artifact_type]}" for artifact_type in expected_types
        )
        if tuple(item.path for item in self.files) != expected_paths:
            raise EvidenceError("benchmark_file_path_mismatch", self.run_path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRunIndex:
        value = strict_fields(
            data,
            {
                "artifact_types",
                "case_id",
                "files",
                "model_identity",
                "repeat_index",
                "run_path",
                "semantic_sha256",
                "terminal_state",
            },
            "BenchmarkRunIndex",
        )
        return cls(
            tuple(
                require_string(item, "artifact type")
                for item in require_list(value["artifact_types"], "artifact_types")
            ),
            require_string(value["case_id"], "case_id"),
            tuple(
                BenchmarkFile.from_dict(require_object(item, "benchmark file"))
                for item in require_list(value["files"], "files")
            ),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            require_int(value["repeat_index"], "repeat_index"),
            require_string(value["run_path"], "run_path"),
            require_string(value["semantic_sha256"], "semantic_sha256"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_types": list(self.artifact_types),
            "case_id": self.case_id,
            "files": [item.to_dict() for item in self.files],
            "model_identity": self.model_identity.to_dict(),
            "repeat_index": self.repeat_index,
            "run_path": self.run_path,
            "semantic_sha256": self.semantic_sha256,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class BenchmarkIndex:
    atlas_sha256: str
    fixture_sha256: str
    model_identity: ModelIdentity
    runner_configuration: RunnerConfiguration
    runs: tuple[BenchmarkRunIndex, ...]
    schema_version: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_INDEX_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state != "completed":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.atlas_sha256, "atlas_sha256")
        require_sha256(self.fixture_sha256, "fixture_sha256")
        keys = tuple((item.case_id, item.repeat_index) for item in self.runs)
        if not self.runs or len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
            raise EvidenceError("invalid_benchmark_run_index", "runs must be sorted and unique")
        if any(item.model_identity != self.model_identity for item in self.runs):
            raise EvidenceError("benchmark_model_identity_mismatch", "run identity differs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkIndex:
        value = strict_fields(
            data,
            {
                "atlas_sha256",
                "fixture_sha256",
                "model_identity",
                "runner_configuration",
                "runs",
                "schema_version",
                "terminal_state",
            },
            "BenchmarkIndex",
        )
        return cls(
            require_string(value["atlas_sha256"], "atlas_sha256"),
            require_string(value["fixture_sha256"], "fixture_sha256"),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            RunnerConfiguration.from_dict(
                require_object(value["runner_configuration"], "runner_configuration")
            ),
            tuple(
                BenchmarkRunIndex.from_dict(require_object(item, "benchmark run index"))
                for item in require_list(value["runs"], "runs")
            ),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_sha256": self.atlas_sha256,
            "fixture_sha256": self.fixture_sha256,
            "model_identity": self.model_identity.to_dict(),
            "runner_configuration": self.runner_configuration.to_dict(),
            "runs": [item.to_dict() for item in self.runs],
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class BenchmarkReceipt:
    atlas_sha256: str
    benchmark_sha256: str
    closed_files: tuple[BenchmarkFile, ...]
    fixture_sha256: str
    index_sha256: str
    model_identity: ModelIdentity
    runner_configuration: RunnerConfiguration
    schema_version: str
    terminal_state: str
    transformation_id: str
    verifier_version: str

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_RECEIPT_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state != "closed":
            raise EvidenceError("receipt_not_closed", self.terminal_state)
        for name, digest in (
            ("atlas_sha256", self.atlas_sha256),
            ("benchmark_sha256", self.benchmark_sha256),
            ("fixture_sha256", self.fixture_sha256),
            ("index_sha256", self.index_sha256),
        ):
            require_sha256(digest, name)
        paths = tuple(item.path for item in self.closed_files)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise EvidenceError(
                "invalid_benchmark_receipt_files",
                "paths must be sorted and unique",
            )
        if "benchmark.json" not in paths or "index.json" not in paths:
            raise EvidenceError("invalid_benchmark_receipt_files", "top-level files are missing")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkReceipt:
        value = strict_fields(
            data,
            {
                "atlas_sha256",
                "benchmark_sha256",
                "closed_files",
                "fixture_sha256",
                "index_sha256",
                "model_identity",
                "runner_configuration",
                "schema_version",
                "terminal_state",
                "transformation_id",
                "verifier_version",
            },
            "BenchmarkReceipt",
        )
        return cls(
            require_string(value["atlas_sha256"], "atlas_sha256"),
            require_string(value["benchmark_sha256"], "benchmark_sha256"),
            tuple(
                BenchmarkFile.from_dict(require_object(item, "closed file"))
                for item in require_list(value["closed_files"], "closed_files")
            ),
            require_string(value["fixture_sha256"], "fixture_sha256"),
            require_string(value["index_sha256"], "index_sha256"),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            RunnerConfiguration.from_dict(
                require_object(value["runner_configuration"], "runner_configuration")
            ),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["terminal_state"], "terminal_state"),
            require_string(value["transformation_id"], "transformation_id"),
            require_string(value["verifier_version"], "verifier_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_sha256": self.atlas_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "closed_files": [item.to_dict() for item in self.closed_files],
            "fixture_sha256": self.fixture_sha256,
            "index_sha256": self.index_sha256,
            "model_identity": self.model_identity.to_dict(),
            "runner_configuration": self.runner_configuration.to_dict(),
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
            "transformation_id": self.transformation_id,
            "verifier_version": self.verifier_version,
        }
