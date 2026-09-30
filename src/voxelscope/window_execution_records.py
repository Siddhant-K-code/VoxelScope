# SPDX-License-Identifier: Apache-2.0
"""Strict records for private milestone 7 window qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from .canonical import EvidenceError, require_sha256, safe_relative_path
from .records import (
    require_bool,
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
    triple_int,
)
from .window_execution_contract import (
    LEGACY_SYNTHETIC_BUNDLE_SHA256,
    MILESTONE5_ADAPTER_PLAN_SHA256,
    MILESTONE5_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_PLAN_SHA256,
    MILESTONE6_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_REPORT_SHA256,
)

CHANNEL_ORDER = ("image-t1c", "image-t1", "image-t2", "image-flair")
ROI_IJK = (240, 240, 160)
STRIDE_IJK = (120, 120, 80)
UV_LOCK_SHA256 = "5bbe48a51f4a649af5370a87ace2e87e2a51f564f2633412b2ff94f1a7b2a71f"
IMPLEMENTATION_PATHS = (
    "src/voxelscope/window_execution.py",
    "src/voxelscope/window_execution_records.py",
    "src/voxelscope/milestone7_cli.py",
)

ExecutionStage = Literal[
    "attempt_marker",
    "preprocessing_verification",
    "input_staging",
    "input_validation",
    "window_materialization",
    "ledger_publication",
    "snapshot_publication",
]
_EXECUTION_STAGES = frozenset(
    {
        "attempt_marker",
        "preprocessing_verification",
        "input_staging",
        "input_validation",
        "window_materialization",
        "ledger_publication",
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
        return cls(
            require_string(data["path"], "path"),
            require_string(data["sha256"], "sha256"),
        )


@dataclass(frozen=True)
class RuntimeIdentity:
    python_version: Literal["3.13.15"]
    numpy_version: Literal["2.5.3"]
    byte_order: Literal["little"]
    uv_lock_sha256: str

    def __post_init__(self) -> None:
        if (
            self.python_version != "3.13.15"
            or self.numpy_version != "2.5.3"
            or self.byte_order != "little"
            or self.uv_lock_sha256 != UV_LOCK_SHA256
        ):
            raise EvidenceError("untrusted_window_runtime", "runtime identity")
        require_sha256(self.uv_lock_sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeIdentity:
        data = strict_fields(
            data,
            {"byte_order", "numpy_version", "python_version", "uv_lock_sha256"},
            "RuntimeIdentity",
        )
        return cls(
            cast(Any, require_string(data["python_version"], "python_version")),
            cast(Any, require_string(data["numpy_version"], "numpy_version")),
            cast(Any, require_string(data["byte_order"], "byte_order")),
            require_string(data["uv_lock_sha256"], "uv_lock_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowExecutionPlan:
    schema_version: Literal["voxelscope/window-execution-plan/v1"]
    plan_id: Literal["voxelscope-milestone-7-window-execution-v1"]
    milestone5_adapter_plan_sha256: str
    milestone5_public_bundle_sha256: str
    milestone6_plan_sha256: str
    milestone6_report_sha256: str
    milestone6_public_bundle_sha256: str
    legacy_synthetic_bundle_sha256: str
    input_layout: Literal["C,I,J,K"]
    model_layout: Literal["N,C,D,H,W"]
    axis_mapping: tuple[str, ...]
    channel_order: tuple[str, ...]
    source_label_excluded: Literal[True]
    roi_ijk: tuple[int, int, int]
    overlap_numerator: Literal[1]
    overlap_denominator: Literal[2]
    stride_ijk: tuple[int, int, int]
    padding_policy: Literal["symmetric-floor-before-ceil-after-constant-zero"]
    traversal_order: Literal["lexicographic_ijk_k_fastest"]
    final_anchor_policy: Literal["final-window-ends-at-padded-extent"]
    crop_policy: Literal["crop-to-original-ijk-extents"]
    blend_mode: Literal["constant"]
    materialization_paths: tuple[str, ...]
    exact_window_bytes_required: Literal[True]
    stream_windows: Literal[True]
    persist_windows: Literal[False]
    finite_values_required: Literal[True]
    empty_tensor_policy: Literal["refuse"]
    zero_channel_policy: Literal["refuse"]
    complete_source_coverage_required: Literal[True]
    immutable_staged_input_required: Literal[True]
    one_shot_execution: Literal[True]
    private_output: Literal[True]
    no_clobber: Literal[True]
    network_allowed: Literal[False]
    model_loading_allowed: Literal[False]
    inference_authorized: Literal[False]
    synthetic_fixture_count: int
    runtime_identity: RuntimeIdentity
    implementation_sources: tuple[ImplementationIdentity, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-execution-plan/v1"
            or self.plan_id != "voxelscope-milestone-7-window-execution-v1"
        ):
            raise EvidenceError("unsupported_schema", self.schema_version)
        expected_hashes = (
            (self.milestone5_adapter_plan_sha256, MILESTONE5_ADAPTER_PLAN_SHA256),
            (self.milestone5_public_bundle_sha256, MILESTONE5_PUBLIC_BUNDLE_SHA256),
            (self.milestone6_plan_sha256, MILESTONE6_PLAN_SHA256),
            (self.milestone6_report_sha256, MILESTONE6_REPORT_SHA256),
            (self.milestone6_public_bundle_sha256, MILESTONE6_PUBLIC_BUNDLE_SHA256),
            (self.legacy_synthetic_bundle_sha256, LEGACY_SYNTHETIC_BUNDLE_SHA256),
        )
        for observed, expected in expected_hashes:
            require_sha256(observed)
            if observed != expected:
                raise EvidenceError("untrusted_window_identity", self.plan_id)
        if (
            self.input_layout != "C,I,J,K"
            or self.model_layout != "N,C,D,H,W"
            or self.axis_mapping != ("I->D", "J->H", "K->W")
            or self.channel_order != CHANNEL_ORDER
            or not self.source_label_excluded
            or self.roi_ijk != ROI_IJK
            or self.overlap_numerator != 1
            or self.overlap_denominator != 2
            or self.stride_ijk != STRIDE_IJK
            or self.padding_policy != "symmetric-floor-before-ceil-after-constant-zero"
            or self.traversal_order != "lexicographic_ijk_k_fastest"
            or self.final_anchor_policy != "final-window-ends-at-padded-extent"
            or self.crop_policy != "crop-to-original-ijk-extents"
            or self.blend_mode != "constant"
        ):
            raise EvidenceError("unsafe_window_execution_contract", self.plan_id)
        if (
            self.materialization_paths
            != ("full-symmetric-pad-then-slice", "direct-source-intersection-local-pad")
            or not self.exact_window_bytes_required
            or not self.stream_windows
            or self.persist_windows
            or not self.finite_values_required
            or self.empty_tensor_policy != "refuse"
            or self.zero_channel_policy != "refuse"
            or not self.complete_source_coverage_required
            or not self.immutable_staged_input_required
            or not self.one_shot_execution
            or not self.private_output
            or not self.no_clobber
            or self.network_allowed
            or self.model_loading_allowed
            or self.inference_authorized
            or self.synthetic_fixture_count != 8
        ):
            raise EvidenceError("unsafe_window_execution_policy", self.plan_id)
        if (
            len(self.implementation_sources) != len(IMPLEMENTATION_PATHS)
            or tuple(item.path for item in self.implementation_sources) != IMPLEMENTATION_PATHS
        ):
            raise EvidenceError("incomplete_implementation_identity", self.plan_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowExecutionPlan:
        expected = {
            "axis_mapping",
            "blend_mode",
            "channel_order",
            "complete_source_coverage_required",
            "crop_policy",
            "empty_tensor_policy",
            "exact_window_bytes_required",
            "final_anchor_policy",
            "finite_values_required",
            "implementation_sources",
            "immutable_staged_input_required",
            "inference_authorized",
            "input_layout",
            "legacy_synthetic_bundle_sha256",
            "materialization_paths",
            "milestone5_adapter_plan_sha256",
            "milestone5_public_bundle_sha256",
            "milestone6_plan_sha256",
            "milestone6_public_bundle_sha256",
            "milestone6_report_sha256",
            "model_layout",
            "model_loading_allowed",
            "network_allowed",
            "no_clobber",
            "one_shot_execution",
            "overlap_denominator",
            "overlap_numerator",
            "padding_policy",
            "persist_windows",
            "plan_id",
            "private_output",
            "roi_ijk",
            "runtime_identity",
            "schema_version",
            "source_label_excluded",
            "stream_windows",
            "stride_ijk",
            "synthetic_fixture_count",
            "traversal_order",
            "zero_channel_policy",
        }
        data = strict_fields(data, expected, "WindowExecutionPlan")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["plan_id"], "plan_id")),
            require_string(
                data["milestone5_adapter_plan_sha256"],
                "milestone5_adapter_plan_sha256",
            ),
            require_string(
                data["milestone5_public_bundle_sha256"],
                "milestone5_public_bundle_sha256",
            ),
            require_string(data["milestone6_plan_sha256"], "milestone6_plan_sha256"),
            require_string(data["milestone6_report_sha256"], "milestone6_report_sha256"),
            require_string(
                data["milestone6_public_bundle_sha256"],
                "milestone6_public_bundle_sha256",
            ),
            require_string(
                data["legacy_synthetic_bundle_sha256"],
                "legacy_synthetic_bundle_sha256",
            ),
            cast(Any, require_string(data["input_layout"], "input_layout")),
            cast(Any, require_string(data["model_layout"], "model_layout")),
            _strings(data["axis_mapping"], "axis_mapping"),
            _strings(data["channel_order"], "channel_order"),
            cast(
                Any,
                require_bool(data["source_label_excluded"], "source_label_excluded"),
            ),
            triple_int(data["roi_ijk"], "roi_ijk"),
            cast(Any, require_int(data["overlap_numerator"], "overlap_numerator")),
            cast(Any, require_int(data["overlap_denominator"], "overlap_denominator")),
            triple_int(data["stride_ijk"], "stride_ijk"),
            cast(Any, require_string(data["padding_policy"], "padding_policy")),
            cast(Any, require_string(data["traversal_order"], "traversal_order")),
            cast(Any, require_string(data["final_anchor_policy"], "final_anchor_policy")),
            cast(Any, require_string(data["crop_policy"], "crop_policy")),
            cast(Any, require_string(data["blend_mode"], "blend_mode")),
            _strings(data["materialization_paths"], "materialization_paths"),
            cast(
                Any,
                require_bool(
                    data["exact_window_bytes_required"],
                    "exact_window_bytes_required",
                ),
            ),
            cast(Any, require_bool(data["stream_windows"], "stream_windows")),
            cast(Any, require_bool(data["persist_windows"], "persist_windows")),
            cast(
                Any,
                require_bool(data["finite_values_required"], "finite_values_required"),
            ),
            cast(Any, require_string(data["empty_tensor_policy"], "empty_tensor_policy")),
            cast(Any, require_string(data["zero_channel_policy"], "zero_channel_policy")),
            cast(
                Any,
                require_bool(
                    data["complete_source_coverage_required"],
                    "complete_source_coverage_required",
                ),
            ),
            cast(
                Any,
                require_bool(
                    data["immutable_staged_input_required"],
                    "immutable_staged_input_required",
                ),
            ),
            cast(Any, require_bool(data["one_shot_execution"], "one_shot_execution")),
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
            require_int(data["synthetic_fixture_count"], "synthetic_fixture_count"),
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
class PrivateArtifact:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        require_sha256(self.sha256)
        if self.size_bytes <= 0:
            raise EvidenceError("invalid_private_artifact", self.path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivateArtifact:
        data = strict_fields(data, {"path", "sha256", "size_bytes"}, "PrivateArtifact")
        return cls(
            require_string(data["path"], "path"),
            require_string(data["sha256"], "sha256"),
            require_int(data["size_bytes"], "size_bytes"),
        )


@dataclass(frozen=True)
class WindowExecutionLedger:
    schema_version: Literal["voxelscope/window-execution-ledger/v1"]
    status: Literal["go"]
    input_shape: tuple[int, ...]
    padded_shape_ijk: tuple[int, int, int]
    roi_ijk: tuple[int, int, int]
    pad_before_ijk: tuple[int, int, int]
    pad_after_ijk: tuple[int, int, int]
    axis_start_counts: tuple[int, int, int]
    axis_starts_sha256: str
    window_count: int
    coordinate_sha256: str
    window_stream_sha256: str
    coverage_min: int
    coverage_max: int
    final_anchors_verified: Literal[True]
    local_padding_verified: Literal[True]
    source_intersections_verified: Literal[True]
    complete_source_coverage: Literal[True]
    crop_geometry_verified: Literal[True]
    exact_window_bytes_equal: Literal[True]
    finite_values: Literal[True]
    channel_order_preserved: Literal[True]
    label_excluded: Literal[True]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-execution-ledger/v1"
            or self.status != "go"
            or len(self.input_shape) != 4
            or self.input_shape[0] != len(CHANNEL_ORDER)
            or any(item <= 0 for item in self.input_shape)
            or any(item <= 0 for item in self.roi_ijk)
            or any(item <= 0 for item in self.padded_shape_ijk)
            or any(item < 0 for item in (*self.pad_before_ijk, *self.pad_after_ijk))
            or any(item <= 0 for item in self.axis_start_counts)
            or self.window_count <= 0
            or self.window_count
            != self.axis_start_counts[0] * self.axis_start_counts[1] * self.axis_start_counts[2]
            or self.coverage_min <= 0
            or self.coverage_max < self.coverage_min
            or not self.final_anchors_verified
            or not self.local_padding_verified
            or not self.source_intersections_verified
            or not self.complete_source_coverage
            or not self.crop_geometry_verified
            or not self.exact_window_bytes_equal
            or not self.finite_values
            or not self.channel_order_preserved
            or not self.label_excluded
        ):
            raise EvidenceError("invalid_window_execution_ledger", self.status)
        for digest in (
            self.axis_starts_sha256,
            self.coordinate_sha256,
            self.window_stream_sha256,
        ):
            require_sha256(digest)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowExecutionLedger:
        expected = {
            "axis_start_counts",
            "axis_starts_sha256",
            "channel_order_preserved",
            "complete_source_coverage",
            "coordinate_sha256",
            "coverage_max",
            "coverage_min",
            "crop_geometry_verified",
            "exact_window_bytes_equal",
            "final_anchors_verified",
            "finite_values",
            "input_shape",
            "label_excluded",
            "local_padding_verified",
            "pad_after_ijk",
            "pad_before_ijk",
            "padded_shape_ijk",
            "roi_ijk",
            "schema_version",
            "source_intersections_verified",
            "status",
            "window_count",
            "window_stream_sha256",
        }
        data = strict_fields(data, expected, "WindowExecutionLedger")
        input_shape = _ints(data["input_shape"], "input_shape")
        if len(input_shape) != 4:
            raise EvidenceError("invalid_window_execution_ledger", "input_shape")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            input_shape,
            triple_int(data["padded_shape_ijk"], "padded_shape_ijk"),
            triple_int(data["roi_ijk"], "roi_ijk"),
            triple_int(data["pad_before_ijk"], "pad_before_ijk", positive=False),
            triple_int(data["pad_after_ijk"], "pad_after_ijk", positive=False),
            triple_int(data["axis_start_counts"], "axis_start_counts"),
            require_string(data["axis_starts_sha256"], "axis_starts_sha256"),
            require_int(data["window_count"], "window_count"),
            require_string(data["coordinate_sha256"], "coordinate_sha256"),
            require_string(data["window_stream_sha256"], "window_stream_sha256"),
            require_int(data["coverage_min"], "coverage_min"),
            require_int(data["coverage_max"], "coverage_max"),
            cast(
                Any,
                require_bool(data["final_anchors_verified"], "final_anchors_verified"),
            ),
            cast(
                Any,
                require_bool(data["local_padding_verified"], "local_padding_verified"),
            ),
            cast(
                Any,
                require_bool(
                    data["source_intersections_verified"],
                    "source_intersections_verified",
                ),
            ),
            cast(
                Any,
                require_bool(
                    data["complete_source_coverage"],
                    "complete_source_coverage",
                ),
            ),
            cast(
                Any,
                require_bool(
                    data["crop_geometry_verified"],
                    "crop_geometry_verified",
                ),
            ),
            cast(
                Any,
                require_bool(
                    data["exact_window_bytes_equal"],
                    "exact_window_bytes_equal",
                ),
            ),
            cast(Any, require_bool(data["finite_values"], "finite_values")),
            cast(
                Any,
                require_bool(
                    data["channel_order_preserved"],
                    "channel_order_preserved",
                ),
            ),
            cast(Any, require_bool(data["label_excluded"], "label_excluded")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowExecutionReport:
    schema_version: Literal["voxelscope/window-execution-report/v1"]
    status: Literal["go"]
    plan_sha256: str
    preprocessing_plan_sha256: str
    preprocessing_report_sha256: str
    preprocessing_snapshot_sha256: str
    preprocessing_tensor_sha256: str
    staged_input_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    input_shape: tuple[int, ...]
    input_layout: Literal["C,I,J,K"]
    model_layout: Literal["N,C,D,H,W"]
    channel_order: tuple[str, ...]
    label_excluded: Literal[True]
    ledger: PrivateArtifact
    runtime_identity: RuntimeIdentity
    implementation_sources: tuple[ImplementationIdentity, ...]
    model_loaded: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]
    windows_persisted: Literal[False]
    completed_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-execution-report/v1"
            or self.status != "go"
            or self.preprocessing_plan_sha256 != MILESTONE5_ADAPTER_PLAN_SHA256
            or len(self.input_shape) != 4
            or self.input_shape[0] != len(CHANNEL_ORDER)
            or any(item <= 0 for item in self.input_shape)
            or self.input_layout != "C,I,J,K"
            or self.model_layout != "N,C,D,H,W"
            or self.channel_order != CHANNEL_ORDER
            or not self.label_excluded
            or self.runtime_identity.uv_lock_sha256 != UV_LOCK_SHA256
            or tuple(item.path for item in self.implementation_sources) != IMPLEMENTATION_PATHS
            or self.model_loaded
            or self.inference_run
            or self.inference_authorized
            or self.windows_persisted
            or not self.completed_at
        ):
            raise EvidenceError("invalid_window_execution_report", self.status)
        for digest in (
            self.plan_sha256,
            self.preprocessing_plan_sha256,
            self.preprocessing_report_sha256,
            self.preprocessing_snapshot_sha256,
            self.preprocessing_tensor_sha256,
            self.staged_input_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
        ):
            require_sha256(digest)
        if self.preprocessing_tensor_sha256 != self.staged_input_sha256:
            raise EvidenceError("staged_input_identity_mismatch", self.status)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowExecutionReport:
        expected = {
            "attempt_record_sha256",
            "authorization_sha256",
            "channel_order",
            "completed_at",
            "implementation_sources",
            "inference_authorized",
            "inference_run",
            "input_layout",
            "input_shape",
            "label_excluded",
            "ledger",
            "model_layout",
            "model_loaded",
            "plan_sha256",
            "preprocessing_plan_sha256",
            "preprocessing_report_sha256",
            "preprocessing_snapshot_sha256",
            "preprocessing_tensor_sha256",
            "runtime_identity",
            "schema_version",
            "staged_input_sha256",
            "status",
            "windows_persisted",
        }
        data = strict_fields(data, expected, "WindowExecutionReport")
        input_shape = _ints(data["input_shape"], "input_shape")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(
                data["preprocessing_plan_sha256"],
                "preprocessing_plan_sha256",
            ),
            require_string(
                data["preprocessing_report_sha256"],
                "preprocessing_report_sha256",
            ),
            require_string(
                data["preprocessing_snapshot_sha256"],
                "preprocessing_snapshot_sha256",
            ),
            require_string(
                data["preprocessing_tensor_sha256"],
                "preprocessing_tensor_sha256",
            ),
            require_string(data["staged_input_sha256"], "staged_input_sha256"),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            input_shape,
            cast(Any, require_string(data["input_layout"], "input_layout")),
            cast(Any, require_string(data["model_layout"], "model_layout")),
            _strings(data["channel_order"], "channel_order"),
            cast(Any, require_bool(data["label_excluded"], "label_excluded")),
            PrivateArtifact.from_dict(require_object(data["ledger"], "ledger")),
            RuntimeIdentity.from_dict(require_object(data["runtime_identity"], "runtime_identity")),
            tuple(
                ImplementationIdentity.from_dict(require_object(item, "implementation source"))
                for item in require_list(
                    data["implementation_sources"],
                    "implementation_sources",
                )
            ),
            cast(Any, require_bool(data["model_loaded"], "model_loaded")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(
                Any,
                require_bool(data["inference_authorized"], "inference_authorized"),
            ),
            cast(Any, require_bool(data["windows_persisted"], "windows_persisted")),
            require_string(data["completed_at"], "completed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowExecutionCompletion:
    schema_version: Literal["voxelscope/window-execution-completion/v1"]
    status: Literal["completed"]
    plan_sha256: str
    preprocessing_report_sha256: str
    preprocessing_snapshot_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    report_sha256: str
    ledger_sha256: str
    snapshot: Literal["window-qualification-v1"]
    model_loaded: Literal[False]
    inference_run: Literal[False]
    completed_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-execution-completion/v1"
            or self.status != "completed"
            or self.snapshot != "window-qualification-v1"
            or self.model_loaded
            or self.inference_run
            or not self.completed_at
        ):
            raise EvidenceError("invalid_window_execution_completion", self.status)
        for digest in (
            self.plan_sha256,
            self.preprocessing_report_sha256,
            self.preprocessing_snapshot_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
            self.report_sha256,
            self.ledger_sha256,
        ):
            require_sha256(digest)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowExecutionCompletion:
        data = strict_fields(
            data,
            {
                "attempt_record_sha256",
                "authorization_sha256",
                "completed_at",
                "inference_run",
                "ledger_sha256",
                "model_loaded",
                "plan_sha256",
                "preprocessing_report_sha256",
                "preprocessing_snapshot_sha256",
                "report_sha256",
                "schema_version",
                "snapshot",
                "status",
            },
            "WindowExecutionCompletion",
        )
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(
                data["preprocessing_report_sha256"],
                "preprocessing_report_sha256",
            ),
            require_string(
                data["preprocessing_snapshot_sha256"],
                "preprocessing_snapshot_sha256",
            ),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            require_string(data["report_sha256"], "report_sha256"),
            require_string(data["ledger_sha256"], "ledger_sha256"),
            cast(Any, require_string(data["snapshot"], "snapshot")),
            cast(Any, require_bool(data["model_loaded"], "model_loaded")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            require_string(data["completed_at"], "completed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowExecutionRefusal:
    schema_version: Literal["voxelscope/window-execution-refusal/v1"]
    status: Literal["refused"]
    error_code: str
    last_completed_stage: ExecutionStage
    plan_sha256: str
    preprocessing_report_sha256: str
    preprocessing_snapshot_sha256: str
    authorization_sha256: str
    attempt_record_sha256: str
    completed_window_count: int
    snapshot_published: bool
    model_loaded: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]
    refused_at: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-execution-refusal/v1"
            or self.status != "refused"
            or not self.error_code
            or self.last_completed_stage not in _EXECUTION_STAGES
            or self.completed_window_count < 0
            or self.model_loaded
            or self.inference_run
            or self.inference_authorized
            or not self.refused_at
        ):
            raise EvidenceError("invalid_window_execution_refusal", self.status)
        for digest in (
            self.plan_sha256,
            self.preprocessing_report_sha256,
            self.preprocessing_snapshot_sha256,
            self.authorization_sha256,
            self.attempt_record_sha256,
        ):
            require_sha256(digest)
        before_windows = {
            "attempt_marker",
            "preprocessing_verification",
            "input_staging",
            "input_validation",
        }
        if (
            self.last_completed_stage in before_windows
            and (self.completed_window_count != 0 or self.snapshot_published)
        ) or (
            self.last_completed_stage in {"window_materialization", "ledger_publication"}
            and self.snapshot_published
        ):
            raise EvidenceError("invalid_refusal_progress", self.error_code)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowExecutionRefusal:
        data = strict_fields(
            data,
            {
                "attempt_record_sha256",
                "authorization_sha256",
                "completed_window_count",
                "error_code",
                "inference_authorized",
                "inference_run",
                "last_completed_stage",
                "model_loaded",
                "plan_sha256",
                "preprocessing_report_sha256",
                "preprocessing_snapshot_sha256",
                "refused_at",
                "schema_version",
                "snapshot_published",
                "status",
            },
            "WindowExecutionRefusal",
        )
        stage = require_string(data["last_completed_stage"], "last_completed_stage")
        if stage not in _EXECUTION_STAGES:
            raise EvidenceError("invalid_refusal_stage", stage)
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["error_code"], "error_code"),
            cast(ExecutionStage, stage),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_string(
                data["preprocessing_report_sha256"],
                "preprocessing_report_sha256",
            ),
            require_string(
                data["preprocessing_snapshot_sha256"],
                "preprocessing_snapshot_sha256",
            ),
            require_string(data["authorization_sha256"], "authorization_sha256"),
            require_string(data["attempt_record_sha256"], "attempt_record_sha256"),
            require_int(data["completed_window_count"], "completed_window_count"),
            require_bool(data["snapshot_published"], "snapshot_published"),
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
