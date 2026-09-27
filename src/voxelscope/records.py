# SPDX-License-Identifier: Apache-2.0
"""Versioned typed evidence records."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .canonical import EvidenceError, require_sha256, safe_relative_path

SCHEMA_VERSION = "voxelscope/v1"
REGIONS = ("TC", "WT", "ET")
MODALITIES = ("T1", "T1c", "T2", "FLAIR")
STAGES = (
    "load",
    "normalize",
    "window_enumeration",
    "h2d",
    "inference",
    "blending",
    "postprocess",
    "write",
)


def require_object(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise EvidenceError("invalid_json_type", f"{name} must be an object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise EvidenceError("invalid_json_type", f"{name} must be an array")
    return value


def require_string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise EvidenceError("invalid_json_type", f"{name} must be a string")
    return value


def require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise EvidenceError("invalid_json_type", f"{name} must be a boolean")
    return value


def require_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise EvidenceError("invalid_json_type", f"{name} must be an integer")
    return value


def require_number(value: Any, name: str) -> float:
    if type(value) not in {int, float}:
        raise EvidenceError("invalid_json_type", f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError("invalid_json_type", f"{name} must be finite")
    return result


def strict_fields(data: Any, expected: set[str], name: str) -> dict[str, Any]:
    value = require_object(data, name)
    if set(value) != expected:
        raise EvidenceError(
            "invalid_record",
            f"{name} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}",
        )
    return value


def triple_int(value: Any, name: str, *, positive: bool = True) -> tuple[int, int, int]:
    items = require_list(value, name) if type(value) is list else value
    if type(items) is not tuple or len(items) != 3:
        if type(value) is not list or len(value) != 3:
            raise EvidenceError("invalid_record", f"{name} must contain three integers")
        items = value
    result = (
        require_int(items[0], f"{name}[0]"),
        require_int(items[1], f"{name}[1]"),
        require_int(items[2], f"{name}[2]"),
    )
    if positive and any(item <= 0 for item in result):
        raise EvidenceError("invalid_record", f"{name} must be positive")
    return result


@dataclass(frozen=True)
class OverlapRatio:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0 or not 0 <= self.numerator < self.denominator:
            raise EvidenceError("invalid_overlap", "overlap requires 0 <= p < q")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OverlapRatio:
        data = strict_fields(data, {"denominator", "numerator"}, "OverlapRatio")
        return cls(
            require_int(data["numerator"], "numerator"),
            require_int(data["denominator"], "denominator"),
        )


@dataclass(frozen=True)
class WindowConfig:
    volume_shape: tuple[int, int, int]
    roi: tuple[int, int, int]
    overlap: tuple[OverlapRatio, OverlapRatio, OverlapRatio]
    blend_mode: Literal["constant", "gaussian"]
    sigma_scale: OverlapRatio | None
    padding_mode: Literal["right_zero"] = "right_zero"
    traversal_order: Literal["lexicographic_zyx"] = "lexicographic_zyx"

    def __post_init__(self) -> None:
        triple_int(self.volume_shape, "volume_shape")
        triple_int(self.roi, "roi")
        if len(self.overlap) != 3:
            raise EvidenceError("invalid_overlap", "three overlap ratios are required")
        if self.blend_mode not in {"constant", "gaussian"}:
            raise EvidenceError("unsupported_blend_mode", self.blend_mode)
        if (self.blend_mode == "gaussian") != (self.sigma_scale is not None):
            raise EvidenceError("ambiguous_blend", "sigma is required only for Gaussian blend")
        if self.padding_mode != "right_zero":
            raise EvidenceError("unsupported_padding", self.padding_mode)
        if self.traversal_order != "lexicographic_zyx":
            raise EvidenceError("unsupported_traversal", self.traversal_order)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowConfig:
        data = strict_fields(
            data,
            {
                "blend_mode",
                "overlap",
                "padding_mode",
                "roi",
                "sigma_scale",
                "traversal_order",
                "volume_shape",
            },
            "WindowConfig",
        )
        overlap_items = require_list(data["overlap"], "overlap")
        overlap = tuple(
            OverlapRatio.from_dict(require_object(item, "overlap item")) for item in overlap_items
        )
        if len(overlap) != 3:
            raise EvidenceError("invalid_overlap", "three overlap ratios are required")
        sigma_value = data["sigma_scale"]
        sigma = (
            OverlapRatio.from_dict(require_object(sigma_value, "sigma_scale"))
            if sigma_value is not None
            else None
        )
        return cls(
            triple_int(data["volume_shape"], "volume_shape"),
            triple_int(data["roi"], "roi"),
            overlap,
            require_string(data["blend_mode"], "blend_mode"),  # type: ignore[arg-type]
            sigma,
            require_string(data["padding_mode"], "padding_mode"),  # type: ignore[arg-type]
            require_string(data["traversal_order"], "traversal_order"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ArrayArtifact:
    path: str
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    file_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        if self.dtype not in {"<f4", "<f8", "|u1"}:
            raise EvidenceError("unsupported_array_dtype", self.dtype)
        if not self.shape or any(value <= 0 for value in self.shape) or self.size_bytes <= 0:
            raise EvidenceError("invalid_array_identity", self.path)
        require_sha256(self.file_sha256, "file_sha256")
        require_sha256(self.content_sha256, "content_sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrayArtifact:
        data = strict_fields(
            data,
            {"content_sha256", "dtype", "file_sha256", "path", "shape", "size_bytes"},
            "ArrayArtifact",
        )
        shape = require_list(data["shape"], "shape")
        return cls(
            require_string(data["path"], "path"),
            require_string(data["dtype"], "dtype"),
            tuple(require_int(value, "shape item") for value in shape),
            require_int(data["size_bytes"], "size_bytes"),
            require_string(data["file_sha256"], "file_sha256"),
            require_string(data["content_sha256"], "content_sha256"),
        )


@dataclass(frozen=True)
class WindowLedgerEntry:
    index: int
    start: tuple[int, int, int]
    end: tuple[int, int, int]
    source_start: tuple[int, int, int]
    source_end: tuple[int, int, int]
    pad_before: tuple[int, int, int]
    pad_after: tuple[int, int, int]
    roi: tuple[int, int, int]
    overlap: tuple[OverlapRatio, OverlapRatio, OverlapRatio]
    traversal_order: str
    blend_mode: str
    weight_identity_sha256: str
    input_region_sha256: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise EvidenceError("invalid_window_index", str(self.index))
        for field in ("start", "end", "source_start", "source_end", "pad_before", "pad_after"):
            triple_int(getattr(self, field), field, positive=False)
        triple_int(self.roi, "roi")
        require_sha256(self.weight_identity_sha256)
        require_sha256(self.input_region_sha256)


@dataclass(frozen=True)
class ThresholdConfig:
    comparator: Literal[">="]
    values: dict[str, float]
    version: Literal["fixed-0.5-v1"] = "fixed-0.5-v1"

    def __post_init__(self) -> None:
        if self.comparator != ">=" or set(self.values) != set(REGIONS):
            raise EvidenceError("invalid_threshold", "threshold fields differ")
        if self.version != "fixed-0.5-v1":
            raise EvidenceError("invalid_threshold", "unsupported threshold version")
        if any(value != 0.5 for value in self.values.values()):
            raise EvidenceError("invalid_threshold", "threshold must be inclusive 0.5")

    @classmethod
    def fixed(cls) -> ThresholdConfig:
        return cls(">=", {region: 0.5 for region in REGIONS})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdConfig:
        data = strict_fields(data, {"comparator", "values", "version"}, "ThresholdConfig")
        values = require_object(data["values"], "threshold values")
        return cls(
            require_string(data["comparator"], "comparator"),  # type: ignore[arg-type]
            {key: require_number(value, f"threshold {key}") for key, value in values.items()},
            require_string(data["version"], "version"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ComponentSummary:
    count: int
    voxel_sizes_descending: tuple[int, ...]
    connectivity: Literal[26] = 26

    def __post_init__(self) -> None:
        if self.connectivity != 26:
            raise EvidenceError("invalid_components", "connectivity must equal 26")
        if self.count != len(self.voxel_sizes_descending) or self.count < 0:
            raise EvidenceError("invalid_components", "component count differs")
        if tuple(sorted(self.voxel_sizes_descending, reverse=True)) != self.voxel_sizes_descending:
            raise EvidenceError("invalid_components", "sizes must be descending")
        if any(value <= 0 for value in self.voxel_sizes_descending):
            raise EvidenceError("invalid_components", "sizes must be positive")


@dataclass(frozen=True)
class VolumeIdentity:
    schema_version: str
    volume_id: str
    modality_order: tuple[str, str, str, str]
    spatial_shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    affine: ArrayArtifact
    modalities: dict[str, ArrayArtifact]
    label_sha256: str | None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.volume_id:
            raise EvidenceError("invalid_volume_identity", self.volume_id)
        if tuple(self.modality_order) != MODALITIES or set(self.modalities) != set(MODALITIES):
            raise EvidenceError("invalid_modalities", "required order is T1,T1c,T2,FLAIR")
        triple_int(self.spatial_shape, "spatial_shape")
        if len(self.spacing_mm) != 3 or any(value <= 0 for value in self.spacing_mm):
            raise EvidenceError("invalid_spacing", "spacing must be positive")
        if self.affine.shape != (4, 4) or self.affine.dtype != "<f8":
            raise EvidenceError("invalid_affine", "affine must be 4x4 float64")
        for item in self.modalities.values():
            if item.shape != self.spatial_shape or item.dtype != "<f4":
                raise EvidenceError("invalid_modality", item.path)
        if self.label_sha256 is not None:
            require_sha256(self.label_sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VolumeIdentity:
        data = strict_fields(
            data,
            {
                "affine",
                "label_sha256",
                "modalities",
                "modality_order",
                "schema_version",
                "spacing_mm",
                "spatial_shape",
                "volume_id",
            },
            "VolumeIdentity",
        )
        modality_order = require_list(data["modality_order"], "modality_order")
        spacing = require_list(data["spacing_mm"], "spacing_mm")
        if len(modality_order) != 4 or len(spacing) != 3:
            raise EvidenceError(
                "invalid_volume_identity", "modality order or spacing length differs"
            )
        modalities = require_object(data["modalities"], "modalities")
        label = data["label_sha256"]
        if label is not None:
            label = require_string(label, "label_sha256")
        return cls(
            require_string(data["schema_version"], "schema_version"),
            require_string(data["volume_id"], "volume_id"),
            tuple(require_string(item, "modality") for item in modality_order),  # type: ignore[arg-type]
            triple_int(data["spatial_shape"], "spatial_shape"),
            tuple(require_number(item, "spacing item") for item in spacing),  # type: ignore[arg-type]
            ArrayArtifact.from_dict(require_object(data["affine"], "affine")),
            {
                key: ArrayArtifact.from_dict(require_object(value, f"modality {key}"))
                for key, value in modalities.items()
            },
            label,
        )


@dataclass(frozen=True)
class ModelBundleIdentity:
    schema_version: str
    name: str
    version: str
    model_kind: Literal["synthetic_oracle", "pretrained"]
    source: str
    license: str
    config_path: str
    config_sha256: str
    weights_path: str | None
    weights_sha256: str | None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not all(
            (self.name, self.version, self.source, self.license)
        ):
            raise EvidenceError("invalid_model_identity", self.name)
        safe_relative_path(self.config_path)
        require_sha256(self.config_sha256)
        if self.model_kind not in {"synthetic_oracle", "pretrained"}:
            raise EvidenceError("invalid_model_kind", str(self.model_kind))
        if self.model_kind == "pretrained":
            if self.weights_path is None or self.weights_sha256 is None:
                raise EvidenceError("missing_model_weights", self.name)
            safe_relative_path(self.weights_path)
        elif self.weights_path is not None or self.weights_sha256 is not None:
            raise EvidenceError("unexpected_model_weights", self.name)
        if self.weights_sha256 is not None:
            require_sha256(self.weights_sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelBundleIdentity:
        data = strict_fields(
            data,
            {
                "config_path",
                "config_sha256",
                "license",
                "model_kind",
                "name",
                "schema_version",
                "source",
                "version",
                "weights_path",
                "weights_sha256",
            },
            "ModelBundleIdentity",
        )
        weights_path = data["weights_path"]
        weights_sha256 = data["weights_sha256"]
        return cls(
            schema_version=require_string(data["schema_version"], "schema_version"),
            name=require_string(data["name"], "name"),
            version=require_string(data["version"], "version"),
            model_kind=require_string(data["model_kind"], "model_kind"),  # type: ignore[arg-type]
            source=require_string(data["source"], "source"),
            license=require_string(data["license"], "license"),
            config_path=require_string(data["config_path"], "config_path"),
            config_sha256=require_string(data["config_sha256"], "config_sha256"),
            weights_path=(
                None if weights_path is None else require_string(weights_path, "weights_path")
            ),
            weights_sha256=(
                None if weights_sha256 is None else require_string(weights_sha256, "weights_sha256")
            ),
        )


@dataclass(frozen=True)
class StageTimingRecord:
    stage: str
    available: bool
    value: int | None
    unit: Literal["ns"]
    clock: str
    provenance_path: str
    provenance_sha256: str
    reason: str | None

    def __post_init__(self) -> None:
        if (
            self.stage not in STAGES
            or self.unit != "ns"
            or not self.clock
            or self.available != (self.value is not None)
        ):
            raise EvidenceError("invalid_timing", self.stage)
        safe_relative_path(self.provenance_path)
        if self.value is not None and (type(self.value) is not int or self.value < 0):
            raise EvidenceError("invalid_timing", self.stage)
        if type(self.available) is not bool:
            raise EvidenceError("invalid_timing", self.stage)
        if self.available == (self.reason is not None):
            raise EvidenceError("invalid_timing", self.stage)
        require_sha256(self.provenance_sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageTimingRecord:
        data = strict_fields(
            data,
            {
                "available",
                "clock",
                "provenance_path",
                "provenance_sha256",
                "reason",
                "stage",
                "unit",
                "value",
            },
            "StageTimingRecord",
        )
        value = data["value"]
        reason = data["reason"]
        return cls(
            stage=require_string(data["stage"], "stage"),
            available=require_bool(data["available"], "available"),
            value=None if value is None else require_int(value, "value"),
            unit=require_string(data["unit"], "unit"),  # type: ignore[arg-type]
            clock=require_string(data["clock"], "clock"),
            provenance_path=require_string(data["provenance_path"], "provenance_path"),
            provenance_sha256=require_string(data["provenance_sha256"], "provenance_sha256"),
            reason=None if reason is None else require_string(reason, "reason"),
        )


@dataclass(frozen=True)
class OutputIdentity:
    schema_version: str
    output_id: str
    channel_order: tuple[str, str, str]
    spacing_mm: tuple[float, float, float]
    probabilities: ArrayArtifact
    masks: dict[str, ArrayArtifact]
    threshold: ThresholdConfig
    components: dict[str, ComponentSummary]
    volume_identity_sha256: str
    model_identity_sha256: str
    window_ledger_sha256: str
    run_id: str
    arm_id: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or not self.output_id
            or tuple(self.channel_order) != REGIONS
        ):
            raise EvidenceError("invalid_output_identity", self.output_id)
        if set(self.masks) != set(REGIONS) or set(self.components) != set(REGIONS):
            raise EvidenceError("invalid_output_channels", self.output_id)
        if len(self.spacing_mm) != 3 or any(value <= 0 for value in self.spacing_mm):
            raise EvidenceError("invalid_spacing", "output spacing must be positive")
        if self.probabilities.dtype != "<f4" or len(self.probabilities.shape) != 4:
            raise EvidenceError("invalid_probability_artifact", self.output_id)
        if self.probabilities.shape[0] != 3:
            raise EvidenceError("invalid_probability_artifact", self.output_id)
        for item in self.masks.values():
            if item.dtype != "|u1" or item.shape != self.probabilities.shape[1:]:
                raise EvidenceError("invalid_mask_artifact", item.path)
        require_sha256(self.volume_identity_sha256, "volume_identity_sha256")
        require_sha256(self.model_identity_sha256, "model_identity_sha256")
        require_sha256(self.window_ledger_sha256, "window_ledger_sha256")
        if not self.run_id or not self.arm_id:
            raise EvidenceError("invalid_output_provenance", self.output_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputIdentity:
        data = strict_fields(
            data,
            {
                "arm_id",
                "channel_order",
                "components",
                "masks",
                "model_identity_sha256",
                "output_id",
                "probabilities",
                "run_id",
                "schema_version",
                "spacing_mm",
                "threshold",
                "volume_identity_sha256",
                "window_ledger_sha256",
            },
            "OutputIdentity",
        )
        channels = require_list(data["channel_order"], "channel_order")
        spacing = require_list(data["spacing_mm"], "spacing_mm")
        if len(channels) != 3 or len(spacing) != 3:
            raise EvidenceError(
                "invalid_output_identity", "channel order or spacing length differs"
            )
        components_data = require_object(data["components"], "components")
        components: dict[str, ComponentSummary] = {}
        for key, raw_value in components_data.items():
            value = strict_fields(
                raw_value,
                {"connectivity", "count", "voxel_sizes_descending"},
                f"ComponentSummary {key}",
            )
            sizes = require_list(value["voxel_sizes_descending"], "voxel_sizes_descending")
            components[key] = ComponentSummary(
                require_int(value["count"], "component count"),
                tuple(require_int(item, "component size") for item in sizes),
                require_int(value["connectivity"], "connectivity"),  # type: ignore[arg-type]
            )
        masks_data = require_object(data["masks"], "masks")
        return cls(
            require_string(data["schema_version"], "schema_version"),
            require_string(data["output_id"], "output_id"),
            tuple(require_string(item, "channel") for item in channels),  # type: ignore[arg-type]
            tuple(require_number(item, "spacing item") for item in spacing),  # type: ignore[arg-type]
            ArrayArtifact.from_dict(require_object(data["probabilities"], "probabilities")),
            {
                key: ArrayArtifact.from_dict(require_object(value, f"mask {key}"))
                for key, value in masks_data.items()
            },
            ThresholdConfig.from_dict(require_object(data["threshold"], "threshold")),
            components,
            require_string(data["volume_identity_sha256"], "volume_identity_sha256"),
            require_string(data["model_identity_sha256"], "model_identity_sha256"),
            require_string(data["window_ledger_sha256"], "window_ledger_sha256"),
            require_string(data["run_id"], "run_id"),
            require_string(data["arm_id"], "arm_id"),
        )


@dataclass(frozen=True)
class PlannedComparison:
    report_id: str
    reference_output_id: str
    candidate_output_id: str

    def __post_init__(self) -> None:
        if not all((self.report_id, self.reference_output_id, self.candidate_output_id)):
            raise EvidenceError("invalid_planned_comparison", "comparison IDs cannot be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedComparison:
        data = strict_fields(
            data,
            {"candidate_output_id", "reference_output_id", "report_id"},
            "PlannedComparison",
        )
        return cls(
            require_string(data["report_id"], "report_id"),
            require_string(data["reference_output_id"], "reference_output_id"),
            require_string(data["candidate_output_id"], "candidate_output_id"),
        )


@dataclass(frozen=True)
class ExpectedRefusal:
    refusal_id: str
    code: str
    status: Literal["refused"]
    stage: str
    evidence_path: str

    def __post_init__(self) -> None:
        if not all((self.refusal_id, self.code, self.stage)) or self.status != "refused":
            raise EvidenceError("invalid_expected_refusal", "invalid expected refusal")
        safe_relative_path(self.evidence_path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedRefusal:
        data = strict_fields(
            data,
            {"code", "evidence_path", "refusal_id", "stage", "status"},
            "ExpectedRefusal",
        )
        return cls(
            require_string(data["refusal_id"], "refusal_id"),
            require_string(data["code"], "code"),
            require_string(data["status"], "status"),  # type: ignore[arg-type]
            require_string(data["stage"], "stage"),
            require_string(data["evidence_path"], "evidence_path"),
        )


@dataclass(frozen=True)
class StudyManifest:
    schema_version: str
    study_id: str
    research_only: bool
    claim_scope: Literal["output_preservation", "diagnostic_accuracy"]
    lineage_status: Literal["unresolved", "proven_subject_nonoverlap"]
    lineage_evidence_sha256: tuple[str, ...]
    diagnostic_accuracy_allowed: bool
    volume_identity_path: str
    model_identity_path: str
    window_config: WindowConfig
    expected_output_ids: tuple[str, ...]
    planned_comparisons: tuple[PlannedComparison, ...]
    expected_refusals: tuple[ExpectedRefusal, ...]
    drift_acceptance_thresholds: None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.study_id:
            raise EvidenceError("invalid_study_manifest", self.study_id)
        if not self.research_only:
            raise EvidenceError("research_only_required", self.study_id)
        safe_relative_path(self.volume_identity_path)
        safe_relative_path(self.model_identity_path)
        if self.claim_scope not in {"output_preservation", "diagnostic_accuracy"}:
            raise EvidenceError("invalid_claim_scope", str(self.claim_scope))
        if self.lineage_status not in {"unresolved", "proven_subject_nonoverlap"}:
            raise EvidenceError("invalid_lineage_status", str(self.lineage_status))
        for digest in self.lineage_evidence_sha256:
            require_sha256(digest)
        proven = self.lineage_status == "proven_subject_nonoverlap" and bool(
            self.lineage_evidence_sha256
        )
        if self.diagnostic_accuracy_allowed != proven:
            raise EvidenceError("lineage_gate_failed", "accuracy requires non-overlap proof")
        if self.claim_scope == "diagnostic_accuracy" and not proven:
            raise EvidenceError("lineage_gate_failed", "diagnostic claim refused")
        if tuple(sorted(self.expected_output_ids)) != self.expected_output_ids or len(
            self.expected_output_ids
        ) != len(set(self.expected_output_ids)):
            raise EvidenceError("invalid_expected_outputs", "output IDs must be unique and sorted")
        report_ids = tuple(item.report_id for item in self.planned_comparisons)
        if tuple(sorted(report_ids)) != report_ids or len(report_ids) != len(set(report_ids)):
            raise EvidenceError(
                "invalid_planned_comparisons", "report IDs must be unique and sorted"
            )
        known_outputs = set(self.expected_output_ids)
        if any(
            item.reference_output_id not in known_outputs
            or item.candidate_output_id not in known_outputs
            for item in self.planned_comparisons
        ):
            raise EvidenceError(
                "invalid_planned_comparisons", "comparison references unknown output"
            )
        refusal_ids = tuple(item.refusal_id for item in self.expected_refusals)
        if tuple(sorted(refusal_ids)) != refusal_ids or len(refusal_ids) != len(set(refusal_ids)):
            raise EvidenceError(
                "invalid_expected_refusals", "refusal IDs must be unique and sorted"
            )
        if self.drift_acceptance_thresholds is not None:
            raise EvidenceError("unset_drift_thresholds_required", "thresholds must be null")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudyManifest:
        data = strict_fields(
            data,
            {
                "claim_scope",
                "diagnostic_accuracy_allowed",
                "drift_acceptance_thresholds",
                "expected_output_ids",
                "expected_refusals",
                "lineage_evidence_sha256",
                "lineage_status",
                "model_identity_path",
                "planned_comparisons",
                "research_only",
                "schema_version",
                "study_id",
                "volume_identity_path",
                "window_config",
            },
            "StudyManifest",
        )
        lineage = require_list(data["lineage_evidence_sha256"], "lineage_evidence_sha256")
        output_ids = require_list(data["expected_output_ids"], "expected_output_ids")
        comparisons = require_list(data["planned_comparisons"], "planned_comparisons")
        refusals = require_list(data["expected_refusals"], "expected_refusals")
        return cls(
            require_string(data["schema_version"], "schema_version"),
            require_string(data["study_id"], "study_id"),
            require_bool(data["research_only"], "research_only"),
            require_string(data["claim_scope"], "claim_scope"),  # type: ignore[arg-type]
            require_string(data["lineage_status"], "lineage_status"),  # type: ignore[arg-type]
            tuple(require_string(item, "lineage digest") for item in lineage),
            require_bool(data["diagnostic_accuracy_allowed"], "diagnostic_accuracy_allowed"),
            require_string(data["volume_identity_path"], "volume_identity_path"),
            require_string(data["model_identity_path"], "model_identity_path"),
            WindowConfig.from_dict(require_object(data["window_config"], "window_config")),
            tuple(require_string(item, "output ID") for item in output_ids),
            tuple(
                PlannedComparison.from_dict(require_object(item, "planned comparison"))
                for item in comparisons
            ),
            tuple(
                ExpectedRefusal.from_dict(require_object(item, "expected refusal"))
                for item in refusals
            ),
            data["drift_acceptance_thresholds"],
        )


@dataclass(frozen=True)
class MetricAvailability:
    available: bool
    value: float | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.available != (self.value is not None):
            raise EvidenceError("invalid_metric_availability", "value differs")
        if self.available == (self.reason is not None):
            raise EvidenceError("invalid_metric_availability", "reason differs")


@dataclass(frozen=True)
class BoundaryDriftReport:
    schema_version: str
    report_id: str
    reference_output_sha256: str
    candidate_output_sha256: str
    threshold: ThresholdConfig
    surface_dice_tolerance_mm: float
    probability_bytes_equal: bool
    probability_changed_element_count: int
    probability_changed_element_rate: float
    probability_mean_absolute_error: float
    probability_max_absolute_error: float
    changed_voxel_count: int
    changed_voxel_rate: float
    regions: dict[str, dict[str, Any]]
    acceptance_thresholds: None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.report_id:
            raise EvidenceError("invalid_drift_report", self.report_id)
        require_sha256(self.reference_output_sha256)
        require_sha256(self.candidate_output_sha256)
        if self.surface_dice_tolerance_mm < 0 or set(self.regions) != set(REGIONS):
            raise EvidenceError("invalid_drift_report", self.report_id)
        if self.acceptance_thresholds is not None:
            raise EvidenceError("unset_drift_thresholds_required", self.report_id)


@dataclass(frozen=True)
class FailureState:
    schema_version: str
    refusal_id: str
    status: Literal["failed", "refused"]
    code: str
    stage: str
    message: str
    evidence_path: str
    evidence_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not all(
            (self.refusal_id, self.code, self.stage, self.message)
        ):
            raise EvidenceError("invalid_failure", self.code)
        if self.status not in {"failed", "refused"}:
            raise EvidenceError("invalid_failure", "unknown failure status")
        safe_relative_path(self.evidence_path)
        for digest in self.evidence_sha256:
            require_sha256(digest)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureState:
        data = strict_fields(
            data,
            {
                "code",
                "evidence_path",
                "evidence_sha256",
                "message",
                "refusal_id",
                "schema_version",
                "stage",
                "status",
            },
            "FailureState",
        )
        digests = require_list(data["evidence_sha256"], "evidence_sha256")
        return cls(
            require_string(data["schema_version"], "schema_version"),
            require_string(data["refusal_id"], "refusal_id"),
            require_string(data["status"], "status"),  # type: ignore[arg-type]
            require_string(data["code"], "code"),
            require_string(data["stage"], "stage"),
            require_string(data["message"], "message"),
            require_string(data["evidence_path"], "evidence_path"),
            tuple(require_string(item, "evidence digest") for item in digests),
        )


@dataclass(frozen=True)
class RunReceipt:
    schema_version: str
    run_id: str
    status: Literal["succeeded", "failed", "refused"]
    command: str
    offline: bool
    synthetic_only: bool
    gpu_used: bool
    medical_data_used: bool
    model_weights_used: bool
    timings_path: str
    failure_path: str | None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.run_id or not self.command:
            raise EvidenceError("invalid_receipt", self.run_id)
        if not self.offline or not self.synthetic_only:
            raise EvidenceError("invalid_receipt", "PR 1 must be offline and synthetic")
        safe_relative_path(self.timings_path)
        if self.failure_path is not None:
            safe_relative_path(self.failure_path)
        if self.gpu_used or self.medical_data_used or self.model_weights_used:
            raise EvidenceError("invalid_receipt", "forbidden execution occurred")
        if self.status not in {"succeeded", "failed", "refused"}:
            raise EvidenceError("invalid_receipt", "unknown status")
        if self.status == "succeeded" and self.failure_path is not None:
            raise EvidenceError("invalid_receipt", "success cannot have failure path")
        if self.status != "succeeded" and self.failure_path is None:
            raise EvidenceError("invalid_receipt", "failure or refusal requires a failure path")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunReceipt:
        data = strict_fields(
            data,
            {
                "command",
                "failure_path",
                "gpu_used",
                "medical_data_used",
                "model_weights_used",
                "offline",
                "run_id",
                "schema_version",
                "status",
                "synthetic_only",
                "timings_path",
            },
            "RunReceipt",
        )
        failure_path = data["failure_path"]
        return cls(
            require_string(data["schema_version"], "schema_version"),
            require_string(data["run_id"], "run_id"),
            require_string(data["status"], "status"),  # type: ignore[arg-type]
            require_string(data["command"], "command"),
            require_bool(data["offline"], "offline"),
            require_bool(data["synthetic_only"], "synthetic_only"),
            require_bool(data["gpu_used"], "gpu_used"),
            require_bool(data["medical_data_used"], "medical_data_used"),
            require_bool(data["model_weights_used"], "model_weights_used"),
            require_string(data["timings_path"], "timings_path"),
            None if failure_path is None else require_string(failure_path, "failure_path"),
        )


@dataclass(frozen=True)
class BundleArtifact:
    path: str
    size_bytes: int
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        if self.size_bytes < 0 or not self.media_type:
            raise EvidenceError("invalid_bundle_artifact", self.path)
        require_sha256(self.sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleArtifact:
        data = strict_fields(data, {"media_type", "path", "sha256", "size_bytes"}, "BundleArtifact")
        return cls(
            require_string(data["path"], "path"),
            require_int(data["size_bytes"], "size_bytes"),
            require_string(data["sha256"], "sha256"),
            require_string(data["media_type"], "media_type"),
        )


@dataclass(frozen=True)
class BundleIndex:
    schema_version: str
    bundle_format: Literal["voxelscope-closed-bundle-v1"]
    artifacts: tuple[BundleArtifact, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.bundle_format != "voxelscope-closed-bundle-v1":
            raise EvidenceError("invalid_bundle_format", str(self.bundle_format))
        paths = [item.path for item in self.artifacts]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise EvidenceError("invalid_bundle_index", "paths must be sorted and unique")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleIndex:
        data = strict_fields(data, {"artifacts", "bundle_format", "schema_version"}, "BundleIndex")
        artifacts = require_list(data["artifacts"], "artifacts")
        return cls(
            require_string(data["schema_version"], "schema_version"),
            require_string(data["bundle_format"], "bundle_format"),  # type: ignore[arg-type]
            tuple(
                BundleArtifact.from_dict(require_object(item, "bundle artifact"))
                for item in artifacts
            ),
        )


@dataclass(frozen=True)
class InvalidOutputEvidence:
    schema_version: str
    evidence_id: str
    probabilities: ArrayArtifact
    threshold: ThresholdConfig

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.evidence_id:
            raise EvidenceError("invalid_refusal_evidence", self.evidence_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvalidOutputEvidence:
        data = strict_fields(
            data,
            {"evidence_id", "probabilities", "schema_version", "threshold"},
            "InvalidOutputEvidence",
        )
        return cls(
            require_string(data["schema_version"], "schema_version"),
            require_string(data["evidence_id"], "evidence_id"),
            ArrayArtifact.from_dict(require_object(data["probabilities"], "probabilities")),
            ThresholdConfig.from_dict(require_object(data["threshold"], "threshold")),
        )


def record_dict(record: Any) -> dict[str, Any]:
    result = asdict(record)
    if not isinstance(result, dict):
        raise TypeError("record must serialize to an object")
    return result
