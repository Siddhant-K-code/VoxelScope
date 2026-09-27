# SPDX-License-Identifier: Apache-2.0
"""Versioned typed evidence records."""

from __future__ import annotations

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


def strict_fields(data: dict[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise EvidenceError(
            "invalid_record",
            f"{name} fields differ: missing={sorted(expected - set(data))}, "
            f"extra={sorted(set(data) - expected)}",
        )


def triple_int(value: Any, name: str, *, positive: bool = True) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise EvidenceError("invalid_record", f"{name} must contain three integers")
    result = tuple(int(item) for item in value)
    if positive and any(item <= 0 for item in result):
        raise EvidenceError("invalid_record", f"{name} must be positive")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class OverlapRatio:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0 or not 0 <= self.numerator < self.denominator:
            raise EvidenceError("invalid_overlap", "overlap requires 0 <= p < q")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OverlapRatio:
        strict_fields(data, {"denominator", "numerator"}, "OverlapRatio")
        return cls(int(data["numerator"]), int(data["denominator"]))


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
        strict_fields(
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
        overlap = tuple(OverlapRatio.from_dict(item) for item in data["overlap"])
        if len(overlap) != 3:
            raise EvidenceError("invalid_overlap", "three overlap ratios are required")
        sigma = data["sigma_scale"]
        return cls(
            triple_int(data["volume_shape"], "volume_shape"),
            triple_int(data["roi"], "roi"),
            overlap,
            str(data["blend_mode"]),  # type: ignore[arg-type]
            OverlapRatio.from_dict(sigma) if sigma is not None else None,
            str(data["padding_mode"]),  # type: ignore[arg-type]
            str(data["traversal_order"]),  # type: ignore[arg-type]
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
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            raise EvidenceError("unsafe_path", self.path)
        if self.dtype not in {"<f4", "<f8", "|u1"}:
            raise EvidenceError("unsupported_array_dtype", self.dtype)
        if not self.shape or any(value <= 0 for value in self.shape) or self.size_bytes <= 0:
            raise EvidenceError("invalid_array_identity", self.path)
        require_sha256(self.file_sha256, "file_sha256")
        require_sha256(self.content_sha256, "content_sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrayArtifact:
        strict_fields(
            data,
            {"content_sha256", "dtype", "file_sha256", "path", "shape", "size_bytes"},
            "ArrayArtifact",
        )
        return cls(
            str(data["path"]),
            str(data["dtype"]),
            tuple(int(value) for value in data["shape"]),
            int(data["size_bytes"]),
            str(data["file_sha256"]),
            str(data["content_sha256"]),
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
        if any(value != 0.5 for value in self.values.values()):
            raise EvidenceError("invalid_threshold", "threshold must be inclusive 0.5")

    @classmethod
    def fixed(cls) -> ThresholdConfig:
        return cls(">=", {region: 0.5 for region in REGIONS})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdConfig:
        strict_fields(data, {"comparator", "values", "version"}, "ThresholdConfig")
        return cls(str(data["comparator"]), dict(data["values"]), str(data["version"]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ComponentSummary:
    count: int
    voxel_sizes_descending: tuple[int, ...]
    connectivity: Literal[26] = 26

    def __post_init__(self) -> None:
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
        strict_fields(
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
        return cls(
            str(data["schema_version"]),
            str(data["volume_id"]),
            tuple(data["modality_order"]),
            triple_int(data["spatial_shape"], "spatial_shape"),
            tuple(float(v) for v in data["spacing_mm"]),  # type: ignore[arg-type]
            ArrayArtifact.from_dict(data["affine"]),
            {key: ArrayArtifact.from_dict(value) for key, value in data["modalities"].items()},
            data["label_sha256"],
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
        if self.schema_version != SCHEMA_VERSION or not all((self.name, self.version, self.source)):
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
        strict_fields(
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
        return cls(
            schema_version=str(data["schema_version"]),
            name=str(data["name"]),
            version=str(data["version"]),
            model_kind=str(data["model_kind"]),  # type: ignore[arg-type]
            source=str(data["source"]),
            license=str(data["license"]),
            config_path=str(data["config_path"]),
            config_sha256=str(data["config_sha256"]),
            weights_path=None if data["weights_path"] is None else str(data["weights_path"]),
            weights_sha256=(
                None if data["weights_sha256"] is None else str(data["weights_sha256"])
            ),
        )


@dataclass(frozen=True)
class StageTimingRecord:
    stage: str
    available: bool
    value: int | None
    unit: Literal["ns"]
    clock: str
    provenance_sha256: str
    reason: str | None

    def __post_init__(self) -> None:
        if self.stage not in STAGES or self.available != (self.value is not None):
            raise EvidenceError("invalid_timing", self.stage)
        if self.value is not None and self.value < 0:
            raise EvidenceError("invalid_timing", self.stage)
        if self.available == (self.reason is not None):
            raise EvidenceError("invalid_timing", self.stage)
        require_sha256(self.provenance_sha256)


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

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or tuple(self.channel_order) != REGIONS:
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputIdentity:
        strict_fields(
            data,
            {
                "channel_order",
                "components",
                "masks",
                "output_id",
                "probabilities",
                "schema_version",
                "spacing_mm",
                "threshold",
            },
            "OutputIdentity",
        )
        components = {
            key: ComponentSummary(
                int(value["count"]),
                tuple(int(item) for item in value["voxel_sizes_descending"]),
                int(value["connectivity"]),  # type: ignore[arg-type]
            )
            for key, value in data["components"].items()
        }
        return cls(
            str(data["schema_version"]),
            str(data["output_id"]),
            tuple(data["channel_order"]),
            tuple(float(v) for v in data["spacing_mm"]),  # type: ignore[arg-type]
            ArrayArtifact.from_dict(data["probabilities"]),
            {key: ArrayArtifact.from_dict(value) for key, value in data["masks"].items()},
            ThresholdConfig.from_dict(data["threshold"]),
            components,
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
        strict_fields(
            data,
            {"candidate_output_id", "reference_output_id", "report_id"},
            "PlannedComparison",
        )
        return cls(
            str(data["report_id"]),
            str(data["reference_output_id"]),
            str(data["candidate_output_id"]),
        )


@dataclass(frozen=True)
class ExpectedRefusal:
    refusal_id: str
    code: str

    def __post_init__(self) -> None:
        if not self.refusal_id or not self.code:
            raise EvidenceError("invalid_expected_refusal", "refusal ID and code cannot be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedRefusal:
        strict_fields(data, {"code", "refusal_id"}, "ExpectedRefusal")
        return cls(str(data["refusal_id"]), str(data["code"]))


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
        strict_fields(
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
        return cls(
            str(data["schema_version"]),
            str(data["study_id"]),
            bool(data["research_only"]),
            str(data["claim_scope"]),  # type: ignore[arg-type]
            str(data["lineage_status"]),  # type: ignore[arg-type]
            tuple(data["lineage_evidence_sha256"]),
            bool(data["diagnostic_accuracy_allowed"]),
            str(data["volume_identity_path"]),
            str(data["model_identity_path"]),
            WindowConfig.from_dict(data["window_config"]),
            tuple(str(item) for item in data["expected_output_ids"]),
            tuple(PlannedComparison.from_dict(item) for item in data["planned_comparisons"]),
            tuple(ExpectedRefusal.from_dict(item) for item in data["expected_refusals"]),
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
    evidence_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not all(
            (self.refusal_id, self.code, self.stage, self.message)
        ):
            raise EvidenceError("invalid_failure", self.code)
        if self.status not in {"failed", "refused"}:
            raise EvidenceError("invalid_failure", "unknown failure status")
        for digest in self.evidence_sha256:
            require_sha256(digest)


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


@dataclass(frozen=True)
class BundleArtifact:
    path: str
    size_bytes: int
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            raise EvidenceError("unsafe_path", self.path)
        if self.size_bytes < 0 or not self.media_type:
            raise EvidenceError("invalid_bundle_artifact", self.path)
        require_sha256(self.sha256)


@dataclass(frozen=True)
class BundleIndex:
    schema_version: str
    bundle_format: Literal["voxelscope-closed-bundle-v1"]
    artifacts: tuple[BundleArtifact, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise EvidenceError("unsupported_schema", self.schema_version)
        paths = [item.path for item in self.artifacts]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise EvidenceError("invalid_bundle_index", "paths must be sorted and unique")


def record_dict(record: Any) -> dict[str, Any]:
    result = asdict(record)
    if not isinstance(result, dict):
        raise TypeError("record must serialize to an object")
    return result
