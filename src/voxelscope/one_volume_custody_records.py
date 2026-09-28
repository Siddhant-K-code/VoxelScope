# SPDX-License-Identifier: Apache-2.0
"""Strict private evidence records for one-volume acquisition and validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np

from .canonical import EvidenceError, require_sha256, safe_relative_path
from .records import (
    require_bool,
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

RefusalStage = Literal[
    "attempt_record",
    "artifact_verification",
    "content_identity",
    "metadata_semantics",
    "structural_validation",
]
_REFUSAL_STAGES = frozenset(
    {
        "attempt_record",
        "artifact_verification",
        "content_identity",
        "metadata_semantics",
        "structural_validation",
    }
)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _ints(value: Any, name: str) -> tuple[int, ...]:
    return tuple(require_int(item, f"{name} item") for item in require_list(value, name))


def _floats(value: Any, name: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in require_list(value, name):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EvidenceError("invalid_number", name)
        converted = float(item)
        if not math.isfinite(converted):
            raise EvidenceError("invalid_number", name)
        values.append(converted)
    return tuple(values)


def _matrix(value: Any, name: str) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    rows = tuple(
        _floats(require_list(row, f"{name} row"), f"{name} row")
        for row in require_list(value, name)
    )
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise EvidenceError("invalid_affine", name)
    return rows


@dataclass(frozen=True)
class PrivateArtifactResult:
    artifact_id: str
    role: str
    relative_path: str
    size_bytes: int
    sha256: str
    source_hashes_verified: tuple[str, ...]
    immutable_version_verified: bool

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path)
        if self.size_bytes <= 0:
            raise EvidenceError("invalid_size", self.artifact_id)
        require_sha256(self.sha256)
        if not self.source_hashes_verified or not self.immutable_version_verified:
            raise EvidenceError("incomplete_artifact_verification", self.artifact_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivateArtifactResult:
        data = strict_fields(
            data,
            {
                "artifact_id",
                "immutable_version_verified",
                "relative_path",
                "role",
                "sha256",
                "size_bytes",
                "source_hashes_verified",
            },
            "PrivateArtifactResult",
        )
        return cls(
            require_string(data["artifact_id"], "artifact_id"),
            require_string(data["role"], "role"),
            require_string(data["relative_path"], "relative_path"),
            require_int(data["size_bytes"], "size_bytes"),
            require_string(data["sha256"], "sha256"),
            _strings(data["source_hashes_verified"], "source_hashes_verified"),
            require_bool(data["immutable_version_verified"], "immutable_version_verified"),
        )


@dataclass(frozen=True)
class NiftiStructure:
    artifact_id: str
    role: str
    nifti_version: Literal[1, 2]
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    voxel_count: int
    data_bytes: int
    spacing: tuple[float, ...]
    spatial_unit: str
    orientation: tuple[str, ...]
    affine: tuple[tuple[float, ...], ...]
    qform_code: int
    qform: tuple[tuple[float, ...], ...] | None
    sform_code: int
    sform: tuple[tuple[float, ...], ...] | None
    scaling_slope: float
    scaling_intercept: float
    finite_voxel_count: int
    nonzero_voxel_count: int | None
    mask_labels: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if self.nifti_version not in {1, 2}:
            raise EvidenceError("unsupported_nifti_version", str(self.nifti_version))
        if len(self.shape) != 3 or any(item <= 0 for item in self.shape):
            raise EvidenceError("invalid_nifti_shape", self.artifact_id)
        if (
            len(self.spacing) != 3
            or any(not math.isfinite(item) or item <= 0 for item in self.spacing)
            or len(self.orientation) != 3
        ):
            raise EvidenceError("invalid_nifti_geometry", self.artifact_id)
        orientation_axes = (
            {item for item in self.orientation if item in {"L", "R"}},
            {item for item in self.orientation if item in {"A", "P"}},
            {item for item in self.orientation if item in {"I", "S"}},
        )
        if any(len(axis) != 1 for axis in orientation_axes):
            raise EvidenceError("invalid_nifti_geometry", self.artifact_id)
        for matrix in (self.affine, self.qform, self.sform):
            if matrix is not None and (
                len(matrix) != 4
                or any(len(row) != 4 for row in matrix)
                or any(not math.isfinite(item) for row in matrix for item in row)
            ):
                raise EvidenceError("invalid_affine", self.artifact_id)
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise EvidenceError("unsupported_nifti_dtype", self.dtype) from exc
        if (
            dtype.kind not in {"i", "u", "f"}
            or dtype.itemsize > 8
            or self.byte_order not in {"<", ">"}
            or self.voxel_count != math.prod(self.shape)
            or self.data_bytes != self.voxel_count * dtype.itemsize
        ):
            raise EvidenceError("invalid_nifti_extent", self.artifact_id)
        if (
            not math.isfinite(self.scaling_slope)
            or self.scaling_slope == 0
            or not math.isfinite(self.scaling_intercept)
        ):
            raise EvidenceError("invalid_nifti_scaling", self.artifact_id)
        if not 0 <= self.qform_code <= 5 or not 0 <= self.sform_code <= 5:
            raise EvidenceError("invalid_nifti_geometry", self.artifact_id)
        if (self.qform_code == 0) != (self.qform is None) or (self.sform_code == 0) != (
            self.sform is None
        ):
            raise EvidenceError("invalid_nifti_geometry", self.artifact_id)
        if self.qform is None and self.sform is None:
            raise EvidenceError("invalid_nifti_geometry", self.artifact_id)
        if self.finite_voxel_count != self.voxel_count:
            raise EvidenceError("nonfinite_voxel", self.artifact_id)
        if self.role == "label":
            if (
                self.nonzero_voxel_count is not None
                or self.mask_labels is None
                or not set(self.mask_labels).issubset({0, 1, 2, 3})
                or 0 not in self.mask_labels
                or not set(self.mask_labels).intersection({1, 2, 3})
            ):
                raise EvidenceError("invalid_mask_record", self.artifact_id)
        elif (
            self.nonzero_voxel_count is None
            or not 0 < self.nonzero_voxel_count <= self.voxel_count
            or self.mask_labels is not None
        ):
            raise EvidenceError("invalid_image_record", self.artifact_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NiftiStructure:
        data = strict_fields(
            data,
            {
                "affine",
                "artifact_id",
                "byte_order",
                "data_bytes",
                "dtype",
                "finite_voxel_count",
                "mask_labels",
                "nifti_version",
                "nonzero_voxel_count",
                "orientation",
                "qform",
                "qform_code",
                "role",
                "scaling_intercept",
                "scaling_slope",
                "shape",
                "sform",
                "sform_code",
                "spacing",
                "spatial_unit",
                "voxel_count",
            },
            "NiftiStructure",
        )
        version = require_int(data["nifti_version"], "nifti_version")
        if version not in {1, 2}:
            raise EvidenceError("unsupported_nifti_version", str(version))
        nonzero_value = data["nonzero_voxel_count"]
        mask_value = data["mask_labels"]
        affine = _matrix(data["affine"], "affine")
        if affine is None:
            raise EvidenceError("invalid_affine", "affine")
        return cls(
            require_string(data["artifact_id"], "artifact_id"),
            require_string(data["role"], "role"),
            cast(Literal[1, 2], version),
            _ints(data["shape"], "shape"),
            require_string(data["dtype"], "dtype"),
            require_string(data["byte_order"], "byte_order"),
            require_int(data["voxel_count"], "voxel_count"),
            require_int(data["data_bytes"], "data_bytes"),
            _floats(data["spacing"], "spacing"),
            require_string(data["spatial_unit"], "spatial_unit"),
            _strings(data["orientation"], "orientation"),
            affine,
            require_int(data["qform_code"], "qform_code"),
            _matrix(data["qform"], "qform"),
            require_int(data["sform_code"], "sform_code"),
            _matrix(data["sform"], "sform"),
            float(data["scaling_slope"]),
            float(data["scaling_intercept"]),
            require_int(data["finite_voxel_count"], "finite_voxel_count"),
            (
                require_int(nonzero_value, "nonzero_voxel_count")
                if nonzero_value is not None
                else None
            ),
            _ints(mask_value, "mask_labels") if mask_value is not None else None,
        )


@dataclass(frozen=True)
class StructuralReport:
    schema_version: Literal["voxelscope/one-volume-structural-report/v1"]
    status: Literal["go"]
    decision_sha256: str
    plan_sha256: str
    authorization_sha256: str
    geometry_comparison: Literal["exact"]
    channel_order: tuple[str, ...]
    allowed_source_labels: tuple[int, ...]
    required_background_label: int
    minimum_tumor_label_count: int
    artifacts: tuple[NiftiStructure, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/one-volume-structural-report/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.status != "go" or self.geometry_comparison != "exact":
            raise EvidenceError("unsafe_structural_status", self.status)
        for value in (self.decision_sha256, self.plan_sha256, self.authorization_sha256):
            require_sha256(value)
        if self.channel_order != ("image-t1c", "image-t1", "image-t2", "image-flair"):
            raise EvidenceError("unsafe_channel_order", repr(self.channel_order))
        if self.allowed_source_labels != (0, 1, 2, 3):
            raise EvidenceError("unsafe_source_label_set", repr(self.allowed_source_labels))
        if self.required_background_label != 0 or self.minimum_tumor_label_count != 1:
            raise EvidenceError("unsafe_required_label_contract", "mask presence policy")
        if tuple(item.role for item in self.artifacts) != (*self.channel_order, "label"):
            raise EvidenceError("incomplete_structural_report", "artifact order")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuralReport:
        data = strict_fields(
            data,
            {
                "allowed_source_labels",
                "artifacts",
                "authorization_sha256",
                "channel_order",
                "decision_sha256",
                "geometry_comparison",
                "plan_sha256",
                "required_background_label",
                "minimum_tumor_label_count",
                "schema_version",
                "status",
            },
            "StructuralReport",
        )
        schema = require_string(data["schema_version"], "schema_version")
        status = require_string(data["status"], "status")
        comparison = require_string(data["geometry_comparison"], "geometry_comparison")
        if schema != "voxelscope/one-volume-structural-report/v1":
            raise EvidenceError("unsupported_schema", schema)
        if status != "go" or comparison != "exact":
            raise EvidenceError("unsafe_structural_status", status)
        return cls(
            "voxelscope/one-volume-structural-report/v1",
            "go",
            require_string(data["decision_sha256"], "decision_sha256"),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            "exact",
            _strings(data["channel_order"], "channel_order"),
            _ints(data["allowed_source_labels"], "allowed_source_labels"),
            require_int(data["required_background_label"], "required_background_label"),
            require_int(data["minimum_tumor_label_count"], "minimum_tumor_label_count"),
            tuple(
                NiftiStructure.from_dict(require_object(item, "NIfTI structure"))
                for item in require_list(data["artifacts"], "artifacts")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustodyReceipt:
    schema_version: Literal["voxelscope/one-volume-custody-receipt/v1"]
    status: Literal["verified"]
    decision_sha256: str
    plan_sha256: str
    approval_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    artifact_order: tuple[str, ...]
    artifacts: tuple[PrivateArtifactResult, ...]
    structural_report_path: str
    structural_report_sha256: str
    model_loaded: Literal[False]
    inference_run: Literal[False]
    verified_at: str

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/one-volume-custody-receipt/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.status != "verified" or self.model_loaded or self.inference_run:
            raise EvidenceError("unsafe_custody_status", self.status)
        for value in (
            self.decision_sha256,
            self.plan_sha256,
            self.approval_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
            self.structural_report_sha256,
        ):
            require_sha256(value)
        safe_relative_path(self.structural_report_path)
        if self.artifact_order != tuple(item.artifact_id for item in self.artifacts):
            raise EvidenceError("invalid_artifact_order", "receipt artifact order")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustodyReceipt:
        data = strict_fields(
            data,
            {
                "approval_sha256",
                "artifact_order",
                "artifacts",
                "attempt_record_sha256",
                "authorization_sha256",
                "decision_sha256",
                "inference_run",
                "model_loaded",
                "plan_sha256",
                "schema_version",
                "status",
                "structural_report_path",
                "structural_report_sha256",
                "verified_at",
            },
            "CustodyReceipt",
        )
        schema = require_string(data["schema_version"], "schema_version")
        status = require_string(data["status"], "status")
        if schema != "voxelscope/one-volume-custody-receipt/v1":
            raise EvidenceError("unsupported_schema", schema)
        if status != "verified":
            raise EvidenceError("unsafe_custody_status", status)
        if require_bool(data["model_loaded"], "model_loaded"):
            raise EvidenceError("unsafe_custody_status", "model_loaded")
        if require_bool(data["inference_run"], "inference_run"):
            raise EvidenceError("unsafe_custody_status", "inference_run")
        return cls(
            "voxelscope/one-volume-custody-receipt/v1",
            "verified",
            require_string(data["decision_sha256"], "decision_sha256"),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["approval_sha256"], "approval_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            _strings(data["artifact_order"], "artifact_order"),
            tuple(
                PrivateArtifactResult.from_dict(require_object(item, "artifact result"))
                for item in require_list(data["artifacts"], "artifacts")
            ),
            require_string(data["structural_report_path"], "structural_report_path"),
            require_string(data["structural_report_sha256"], "structural_report_sha256"),
            False,
            False,
            require_string(data["verified_at"], "verified_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcquisitionRefusal:
    schema_version: Literal["voxelscope/one-volume-acquisition-refusal/v1"]
    status: Literal["refused"]
    error_code: str
    last_completed_stage: RefusalStage
    decision_sha256: str
    plan_sha256: str
    approval_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    verified_artifact_count: int
    validated_medical_file_count: int
    geometry_comparison_count: int
    model_loaded: Literal[False]
    inference_authorized: Literal[False]
    refused_at: str

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/one-volume-acquisition-refusal/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.status != "refused" or not self.error_code or not self.last_completed_stage:
            raise EvidenceError("invalid_refusal", self.status)
        for value in (
            self.decision_sha256,
            self.plan_sha256,
            self.approval_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
        ):
            require_sha256(value)
        if (
            not 0 <= self.verified_artifact_count <= 8
            or not 0 <= self.validated_medical_file_count <= 5
            or not 0 <= self.geometry_comparison_count <= 4
            or self.model_loaded
            or self.inference_authorized
        ):
            raise EvidenceError("invalid_refusal_progress", self.error_code)
        if self.verified_artifact_count < 8 and (
            self.validated_medical_file_count != 0 or self.geometry_comparison_count != 0
        ):
            raise EvidenceError("invalid_refusal_progress", self.error_code)
        if self.validated_medical_file_count < 5 and self.geometry_comparison_count != 0:
            raise EvidenceError("invalid_refusal_progress", self.error_code)
        expected_ranges = {
            "attempt_record": (
                self.verified_artifact_count == 0
                and self.validated_medical_file_count == 0
                and self.geometry_comparison_count == 0
            ),
            "artifact_verification": (
                1 <= self.verified_artifact_count <= 7
                and self.validated_medical_file_count == 0
                and self.geometry_comparison_count == 0
            ),
            "content_identity": (
                self.verified_artifact_count == 8
                and self.validated_medical_file_count == 0
                and self.geometry_comparison_count == 0
            ),
            "metadata_semantics": (
                self.verified_artifact_count == 8
                and 0 <= self.validated_medical_file_count <= 5
                and 0 <= self.geometry_comparison_count <= 4
            ),
            "structural_validation": (
                self.verified_artifact_count == 8
                and self.validated_medical_file_count == 5
                and self.geometry_comparison_count == 4
            ),
        }
        if (
            self.last_completed_stage not in expected_ranges
            or not expected_ranges[self.last_completed_stage]
        ):
            raise EvidenceError("invalid_refusal_progress", self.error_code)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcquisitionRefusal:
        data = strict_fields(
            data,
            {
                "approval_sha256",
                "attempt_record_sha256",
                "authorization_sha256",
                "decision_sha256",
                "error_code",
                "geometry_comparison_count",
                "inference_authorized",
                "last_completed_stage",
                "model_loaded",
                "plan_sha256",
                "refused_at",
                "schema_version",
                "status",
                "validated_medical_file_count",
                "verified_artifact_count",
            },
            "AcquisitionRefusal",
        )
        schema = require_string(data["schema_version"], "schema_version")
        status = require_string(data["status"], "status")
        if schema != "voxelscope/one-volume-acquisition-refusal/v1":
            raise EvidenceError("unsupported_schema", schema)
        if status != "refused":
            raise EvidenceError("invalid_refusal", status)
        stage = require_string(data["last_completed_stage"], "last_completed_stage")
        if stage not in _REFUSAL_STAGES:
            raise EvidenceError("invalid_refusal_stage", stage)
        if require_bool(data["model_loaded"], "model_loaded"):
            raise EvidenceError("invalid_refusal", "model_loaded")
        if require_bool(data["inference_authorized"], "inference_authorized"):
            raise EvidenceError("invalid_refusal", "inference_authorized")
        return cls(
            "voxelscope/one-volume-acquisition-refusal/v1",
            "refused",
            require_string(data["error_code"], "error_code"),
            cast(RefusalStage, stage),
            require_string(data["decision_sha256"], "decision_sha256"),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(data["approval_sha256"], "approval_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            require_int(data["verified_artifact_count"], "verified_artifact_count"),
            require_int(
                data["validated_medical_file_count"],
                "validated_medical_file_count",
            ),
            require_int(data["geometry_comparison_count"], "geometry_comparison_count"),
            False,
            False,
            require_string(data["refused_at"], "refused_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
