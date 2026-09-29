# SPDX-License-Identifier: Apache-2.0
"""Strict records for the milestone 6 positional window bridge."""

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

MONAI_COMMIT = "46a5272196a6c2590ca2589029eed8e4d56ff008"
MONAI_TAG_OBJECT = "416ede8a845eff9294c7656f110c763dec1410fa"
PYTORCH_COMMIT = "d990dada86a8ad94882b5c23e859b88c0c255bda"
MODEL_ZOO_COMMIT = "5370ce6ea1dd132856b9c92e2fa125548594835d"
MODEL_INFERENCE_CONFIG_SHA256 = "22dead767f9bfcd6e3e7cbf5353ef1641e52f37acd9a20cc346361c1e603cf9e"
MILESTONE5_BUNDLE_SHA256 = "08c0e0fe5871582b19fbad0887625e121c8371187449a4a7abbe7b1d97a524cf"
LEGACY_SYNTHETIC_BUNDLE_SHA256 = "c7f42adb2ec3d3f6ea51a7574e7dfc03e5198954ef6e25b6e41fe1d2517bd15c"
LEGACY_SYNTHETIC_LEDGER_SHA256 = "56fcdb4be17a7ebbc23a869fb4c74fb92af152d801ac96789062f580fb731999"
LEGACY_SYNTHETIC_COORDINATE_SHA256 = (
    "417808f2e12aee82275aa2def107e65008d000631fb2719941430dc827250d8e"
)
MODEL_ROI_IJK = (240, 240, 160)
MODEL_STRIDE_IJK = (120, 120, 80)
PROOF_CASE_IDS = (
    "asymmetric-components",
    "final-anchor",
    "small-volume-padding",
    "legacy-synthetic-v1",
)

BridgeStage = Literal[
    "plan_loading",
    "source_verification",
    "oracle_comparison",
    "report_publication",
]
_BRIDGE_STAGES = frozenset(
    {
        "plan_loading",
        "source_verification",
        "oracle_comparison",
        "report_publication",
    }
)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _ints(value: Any, name: str) -> tuple[int, ...]:
    return tuple(require_int(item, f"{name} item") for item in require_list(value, name))


@dataclass(frozen=True)
class SourceIdentity:
    repository: str
    commit: str
    release: str
    path: str
    git_blob_sha1: str
    sha256: str
    size_bytes: int
    immutable_url: str

    def __post_init__(self) -> None:
        if self.repository not in {
            "Project-MONAI/MONAI",
            "Project-MONAI/model-zoo",
            "pytorch/pytorch",
        }:
            raise EvidenceError("untrusted_source_repository", self.repository)
        if len(self.commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.commit
        ):
            raise EvidenceError("invalid_git_commit", self.repository)
        safe_relative_path(self.path)
        if len(self.git_blob_sha1) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_blob_sha1
        ):
            raise EvidenceError("invalid_git_blob", self.path)
        require_sha256(self.sha256)
        if self.size_bytes <= 0:
            raise EvidenceError("invalid_source_size", self.path)
        expected = f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{self.path}"
        if self.immutable_url != expected:
            raise EvidenceError("mutable_source_url", self.path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceIdentity:
        data = strict_fields(
            data,
            {
                "commit",
                "git_blob_sha1",
                "immutable_url",
                "path",
                "release",
                "repository",
                "sha256",
                "size_bytes",
            },
            "SourceIdentity",
        )
        return cls(
            require_string(data["repository"], "repository"),
            require_string(data["commit"], "commit"),
            require_string(data["release"], "release"),
            require_string(data["path"], "path"),
            require_string(data["git_blob_sha1"], "git_blob_sha1"),
            require_string(data["sha256"], "sha256"),
            require_int(data["size_bytes"], "size_bytes"),
            require_string(data["immutable_url"], "immutable_url"),
        )


@dataclass(frozen=True)
class ImplementationIdentity:
    path: str
    git_blob_sha1: str
    sha256: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        if len(self.git_blob_sha1) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_blob_sha1
        ):
            raise EvidenceError("invalid_git_blob", self.path)
        require_sha256(self.sha256)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationIdentity:
        data = strict_fields(
            data,
            {"git_blob_sha1", "path", "sha256"},
            "ImplementationIdentity",
        )
        return cls(
            require_string(data["path"], "path"),
            require_string(data["git_blob_sha1"], "git_blob_sha1"),
            require_string(data["sha256"], "sha256"),
        )


@dataclass(frozen=True)
class AxisBinding:
    preprocessing_axis: Literal["I", "J", "K"]
    preprocessing_spatial_position: int
    window_axis: Literal["S0", "S1", "S2"]
    window_position: int
    model_axis: Literal["D", "H", "W"]
    model_tensor_position: int
    roi_component_index: int
    roi_extent: int
    legacy_synthetic_axis: Literal["Z", "Y", "X"]
    legacy_label_scope: Literal["synthetic-coordinate-name-only"]
    anatomical_direction_assigned: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.preprocessing_spatial_position not in {0, 1, 2}
            or self.window_position not in {0, 1, 2}
            or self.model_tensor_position not in {2, 3, 4}
            or self.roi_component_index not in {0, 1, 2}
            or self.roi_extent <= 0
            or self.legacy_label_scope != "synthetic-coordinate-name-only"
            or self.anatomical_direction_assigned
        ):
            raise EvidenceError("invalid_axis_binding", self.preprocessing_axis)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AxisBinding:
        data = strict_fields(
            data,
            {
                "anatomical_direction_assigned",
                "legacy_label_scope",
                "legacy_synthetic_axis",
                "model_axis",
                "model_tensor_position",
                "preprocessing_axis",
                "preprocessing_spatial_position",
                "roi_component_index",
                "roi_extent",
                "window_axis",
                "window_position",
            },
            "AxisBinding",
        )
        return cls(
            cast(Any, require_string(data["preprocessing_axis"], "preprocessing_axis")),
            require_int(
                data["preprocessing_spatial_position"],
                "preprocessing_spatial_position",
            ),
            cast(Any, require_string(data["window_axis"], "window_axis")),
            require_int(data["window_position"], "window_position"),
            cast(Any, require_string(data["model_axis"], "model_axis")),
            require_int(data["model_tensor_position"], "model_tensor_position"),
            require_int(data["roi_component_index"], "roi_component_index"),
            require_int(data["roi_extent"], "roi_extent"),
            cast(
                Any,
                require_string(data["legacy_synthetic_axis"], "legacy_synthetic_axis"),
            ),
            cast(Any, require_string(data["legacy_label_scope"], "legacy_label_scope")),
            cast(
                Any,
                require_bool(
                    data["anatomical_direction_assigned"],
                    "anatomical_direction_assigned",
                ),
            ),
        )


@dataclass(frozen=True)
class WindowBridgePlan:
    schema_version: Literal["voxelscope/window-bridge-plan/v1"]
    plan_id: Literal["voxelscope-milestone-6-window-bridge-v1"]
    milestone5_bundle_sha256: str
    model_bundle_version: Literal["0.5.2"]
    model_zoo_commit: str
    model_inference_config_sha256: str
    monai_version: Literal["1.4.0"]
    monai_tag: Literal["1.4.0"]
    monai_tag_object: str
    monai_commit: str
    pytorch_version: Literal["2.4.0"]
    pytorch_tag: Literal["v2.4.0"]
    pytorch_commit: str
    upstream_sources: tuple[SourceIdentity, ...]
    implementation_sources: tuple[ImplementationIdentity, ...]
    observed_behavior: tuple[str, ...]
    input_layout: Literal["C,I,J,K"]
    window_layout: Literal["S0,S1,S2"]
    model_layout: Literal["N,C,D,H,W"]
    axis_bindings: tuple[AxisBinding, ...]
    roi_ijk: tuple[int, int, int]
    overlap_numerator: Literal[1]
    overlap_denominator: Literal[2]
    stride_formula: Literal["max(1,floor(roi*(1-overlap)))"]
    stride_ijk: tuple[int, int, int]
    coordinate_convention: Literal["zero-based-half-open"]
    traversal_order: Literal["lexicographic_ijk_k_fastest"]
    final_anchor_policy: Literal["min(index*stride,max(dimension-roi,0))"]
    padding_policy: Literal["high-side-constant-zero"]
    monai_padding_policy: Literal["symmetric-floor-before-ceil-after"]
    padding_policies_equivalent: Literal[False]
    crop_policy: Literal["crop-to-original-ijk-extents"]
    inversion_behavior: Literal["metadata-copy-no-value-spatial-inversion"]
    blend_mode: Literal["constant"]
    sw_batch_size: Literal[1]
    transpose_allowed: Literal[False]
    reorientation_allowed: Literal[False]
    resampling_allowed: Literal[False]
    affine_axis_relabeling_allowed: Literal[False]
    source_label_values: tuple[int, ...]
    model_output_values: tuple[int, ...]
    source_label_excluded: Literal[True]
    legacy_contract: Literal["synthetic-only-zyx-v1"]
    legacy_labels_anatomical: Literal[False]
    legacy_synthetic_bundle_sha256: str
    legacy_synthetic_ledger_sha256: str
    legacy_synthetic_coordinate_sha256: str
    proof_case_ids: tuple[str, ...]
    private_data_required: Literal[False]
    private_reads_allowed: Literal[False]
    model_loading_allowed: Literal[False]
    inference_authorized: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-bridge-plan/v1"
            or self.plan_id != "voxelscope-milestone-6-window-bridge-v1"
        ):
            raise EvidenceError("unsupported_schema", self.schema_version)
        for digest in (
            self.milestone5_bundle_sha256,
            self.model_inference_config_sha256,
            self.legacy_synthetic_bundle_sha256,
            self.legacy_synthetic_ledger_sha256,
            self.legacy_synthetic_coordinate_sha256,
        ):
            require_sha256(digest)
        if (
            self.milestone5_bundle_sha256 != MILESTONE5_BUNDLE_SHA256
            or self.model_bundle_version != "0.5.2"
            or self.model_zoo_commit != MODEL_ZOO_COMMIT
            or self.model_inference_config_sha256 != MODEL_INFERENCE_CONFIG_SHA256
            or self.monai_version != "1.4.0"
            or self.monai_tag != "1.4.0"
            or self.monai_tag_object != MONAI_TAG_OBJECT
            or self.monai_commit != MONAI_COMMIT
            or self.pytorch_version != "2.4.0"
            or self.pytorch_tag != "v2.4.0"
            or self.pytorch_commit != PYTORCH_COMMIT
        ):
            raise EvidenceError("untrusted_window_source", self.plan_id)
        expected_bindings = (
            ("I", 0, "S0", 0, "D", 2, 0, 240, "Z"),
            ("J", 1, "S1", 1, "H", 3, 1, 240, "Y"),
            ("K", 2, "S2", 2, "W", 4, 2, 160, "X"),
        )
        observed_bindings = tuple(
            (
                item.preprocessing_axis,
                item.preprocessing_spatial_position,
                item.window_axis,
                item.window_position,
                item.model_axis,
                item.model_tensor_position,
                item.roi_component_index,
                item.roi_extent,
                item.legacy_synthetic_axis,
            )
            for item in self.axis_bindings
        )
        if observed_bindings != expected_bindings:
            raise EvidenceError("unsafe_axis_mapping", self.plan_id)
        if (
            self.input_layout != "C,I,J,K"
            or self.window_layout != "S0,S1,S2"
            or self.model_layout != "N,C,D,H,W"
            or self.roi_ijk != MODEL_ROI_IJK
            or self.overlap_numerator != 1
            or self.overlap_denominator != 2
            or self.stride_formula != "max(1,floor(roi*(1-overlap)))"
            or self.stride_ijk != MODEL_STRIDE_IJK
            or self.coordinate_convention != "zero-based-half-open"
            or self.traversal_order != "lexicographic_ijk_k_fastest"
            or self.final_anchor_policy != "min(index*stride,max(dimension-roi,0))"
            or self.padding_policy != "high-side-constant-zero"
            or self.monai_padding_policy != "symmetric-floor-before-ceil-after"
            or self.padding_policies_equivalent
            or self.crop_policy != "crop-to-original-ijk-extents"
            or self.inversion_behavior != "metadata-copy-no-value-spatial-inversion"
            or self.blend_mode != "constant"
            or self.sw_batch_size != 1
        ):
            raise EvidenceError("unsafe_window_bridge_contract", self.plan_id)
        if (
            self.transpose_allowed
            or self.reorientation_allowed
            or self.resampling_allowed
            or self.affine_axis_relabeling_allowed
            or self.source_label_values != (0, 1, 2, 3)
            or self.model_output_values != (0, 1, 2, 4)
            or not self.source_label_excluded
            or self.legacy_contract != "synthetic-only-zyx-v1"
            or self.legacy_labels_anatomical
            or self.legacy_synthetic_bundle_sha256 != LEGACY_SYNTHETIC_BUNDLE_SHA256
            or self.legacy_synthetic_ledger_sha256 != LEGACY_SYNTHETIC_LEDGER_SHA256
            or self.legacy_synthetic_coordinate_sha256 != LEGACY_SYNTHETIC_COORDINATE_SHA256
            or self.proof_case_ids != PROOF_CASE_IDS
            or self.private_data_required
            or self.private_reads_allowed
            or self.model_loading_allowed
            or self.inference_authorized
        ):
            raise EvidenceError("unsafe_window_bridge_policy", self.plan_id)
        if {item.repository for item in self.upstream_sources} != {
            "Project-MONAI/MONAI",
            "Project-MONAI/model-zoo",
            "pytorch/pytorch",
        }:
            raise EvidenceError("incomplete_upstream_sources", self.plan_id)
        if {item.path for item in self.implementation_sources} != {
            "src/voxelscope/window_bridge.py",
            "src/voxelscope/window_bridge_records.py",
            "src/voxelscope/windows.py",
        }:
            raise EvidenceError("incomplete_implementation_identity", self.plan_id)
        if not self.observed_behavior:
            raise EvidenceError("incomplete_source_analysis", self.plan_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowBridgePlan:
        expected = {
            "affine_axis_relabeling_allowed",
            "axis_bindings",
            "blend_mode",
            "coordinate_convention",
            "crop_policy",
            "final_anchor_policy",
            "implementation_sources",
            "inference_authorized",
            "input_layout",
            "inversion_behavior",
            "legacy_contract",
            "legacy_labels_anatomical",
            "legacy_synthetic_bundle_sha256",
            "legacy_synthetic_coordinate_sha256",
            "legacy_synthetic_ledger_sha256",
            "milestone5_bundle_sha256",
            "model_bundle_version",
            "model_inference_config_sha256",
            "model_layout",
            "model_loading_allowed",
            "model_output_values",
            "model_zoo_commit",
            "monai_commit",
            "monai_padding_policy",
            "monai_tag",
            "monai_tag_object",
            "monai_version",
            "observed_behavior",
            "overlap_denominator",
            "overlap_numerator",
            "padding_policies_equivalent",
            "padding_policy",
            "plan_id",
            "private_data_required",
            "private_reads_allowed",
            "proof_case_ids",
            "pytorch_commit",
            "pytorch_tag",
            "pytorch_version",
            "reorientation_allowed",
            "resampling_allowed",
            "roi_ijk",
            "schema_version",
            "source_label_excluded",
            "source_label_values",
            "stride_formula",
            "stride_ijk",
            "sw_batch_size",
            "transpose_allowed",
            "traversal_order",
            "upstream_sources",
            "window_layout",
        }
        data = strict_fields(data, expected, "WindowBridgePlan")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["plan_id"], "plan_id")),
            require_string(data["milestone5_bundle_sha256"], "milestone5_bundle_sha256"),
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
            cast(Any, require_string(data["pytorch_version"], "pytorch_version")),
            cast(Any, require_string(data["pytorch_tag"], "pytorch_tag")),
            require_string(data["pytorch_commit"], "pytorch_commit"),
            tuple(
                SourceIdentity.from_dict(require_object(item, "upstream source"))
                for item in require_list(data["upstream_sources"], "upstream_sources")
            ),
            tuple(
                ImplementationIdentity.from_dict(require_object(item, "implementation source"))
                for item in require_list(
                    data["implementation_sources"],
                    "implementation_sources",
                )
            ),
            _strings(data["observed_behavior"], "observed_behavior"),
            cast(Any, require_string(data["input_layout"], "input_layout")),
            cast(Any, require_string(data["window_layout"], "window_layout")),
            cast(Any, require_string(data["model_layout"], "model_layout")),
            tuple(
                AxisBinding.from_dict(require_object(item, "axis binding"))
                for item in require_list(data["axis_bindings"], "axis_bindings")
            ),
            triple_int(data["roi_ijk"], "roi_ijk"),
            cast(Any, require_int(data["overlap_numerator"], "overlap_numerator")),
            cast(Any, require_int(data["overlap_denominator"], "overlap_denominator")),
            cast(Any, require_string(data["stride_formula"], "stride_formula")),
            triple_int(data["stride_ijk"], "stride_ijk"),
            cast(
                Any,
                require_string(data["coordinate_convention"], "coordinate_convention"),
            ),
            cast(Any, require_string(data["traversal_order"], "traversal_order")),
            cast(Any, require_string(data["final_anchor_policy"], "final_anchor_policy")),
            cast(Any, require_string(data["padding_policy"], "padding_policy")),
            cast(
                Any,
                require_string(data["monai_padding_policy"], "monai_padding_policy"),
            ),
            cast(
                Any,
                require_bool(
                    data["padding_policies_equivalent"],
                    "padding_policies_equivalent",
                ),
            ),
            cast(Any, require_string(data["crop_policy"], "crop_policy")),
            cast(Any, require_string(data["inversion_behavior"], "inversion_behavior")),
            cast(Any, require_string(data["blend_mode"], "blend_mode")),
            cast(Any, require_int(data["sw_batch_size"], "sw_batch_size")),
            cast(Any, require_bool(data["transpose_allowed"], "transpose_allowed")),
            cast(
                Any,
                require_bool(data["reorientation_allowed"], "reorientation_allowed"),
            ),
            cast(Any, require_bool(data["resampling_allowed"], "resampling_allowed")),
            cast(
                Any,
                require_bool(
                    data["affine_axis_relabeling_allowed"],
                    "affine_axis_relabeling_allowed",
                ),
            ),
            _ints(data["source_label_values"], "source_label_values"),
            _ints(data["model_output_values"], "model_output_values"),
            cast(
                Any,
                require_bool(data["source_label_excluded"], "source_label_excluded"),
            ),
            cast(Any, require_string(data["legacy_contract"], "legacy_contract")),
            cast(
                Any,
                require_bool(data["legacy_labels_anatomical"], "legacy_labels_anatomical"),
            ),
            require_string(
                data["legacy_synthetic_bundle_sha256"],
                "legacy_synthetic_bundle_sha256",
            ),
            require_string(
                data["legacy_synthetic_ledger_sha256"],
                "legacy_synthetic_ledger_sha256",
            ),
            require_string(
                data["legacy_synthetic_coordinate_sha256"],
                "legacy_synthetic_coordinate_sha256",
            ),
            _strings(data["proof_case_ids"], "proof_case_ids"),
            cast(
                Any,
                require_bool(data["private_data_required"], "private_data_required"),
            ),
            cast(
                Any,
                require_bool(data["private_reads_allowed"], "private_reads_allowed"),
            ),
            cast(
                Any,
                require_bool(data["model_loading_allowed"], "model_loading_allowed"),
            ),
            cast(
                Any,
                require_bool(data["inference_authorized"], "inference_authorized"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowBridgeReport:
    schema_version: Literal["voxelscope/window-bridge-report/v1"]
    status: Literal["go"]
    plan_sha256: str
    proof_case_count: int
    proof_sha256: str
    axis_mapping_status: Literal["go"]
    roi_mapping_status: Literal["go"]
    engine_oracle_status: Literal["go"]
    monai_scan_order_status: Literal["go"]
    voxelscope_padding_status: Literal["go"]
    monai_padding_source_status: Literal["go"]
    padding_policies_equivalent: Literal[False]
    crop_status: Literal["go"]
    blend_axis_status: Literal["go"]
    traversal_status: Literal["go"]
    final_anchor_status: Literal["go"]
    coverage_status: Literal["go"]
    legacy_bundle_status: Literal["go"]
    legacy_unpadded_enumeration_status: Literal["go"]
    legacy_general_padded_reuse_status: Literal["no-go"]
    private_data_accessed: Literal[False]
    private_shape_recorded: Literal[False]
    private_window_count_recorded: Literal[False]
    model_loaded: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-bridge-report/v1"
            or self.status != "go"
            or self.proof_case_count != len(PROOF_CASE_IDS)
        ):
            raise EvidenceError("unsafe_window_bridge_status", self.status)
        require_sha256(self.plan_sha256)
        require_sha256(self.proof_sha256)
        if (
            self.axis_mapping_status != "go"
            or self.roi_mapping_status != "go"
            or self.engine_oracle_status != "go"
            or self.monai_scan_order_status != "go"
            or self.voxelscope_padding_status != "go"
            or self.monai_padding_source_status != "go"
            or self.padding_policies_equivalent
            or self.crop_status != "go"
            or self.blend_axis_status != "go"
            or self.traversal_status != "go"
            or self.final_anchor_status != "go"
            or self.coverage_status != "go"
            or self.legacy_bundle_status != "go"
            or self.legacy_unpadded_enumeration_status != "go"
            or self.legacy_general_padded_reuse_status != "no-go"
            or self.private_data_accessed
            or self.private_shape_recorded
            or self.private_window_count_recorded
            or self.model_loaded
            or self.inference_run
            or self.inference_authorized
        ):
            raise EvidenceError("unsafe_window_bridge_report", self.status)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowBridgeReport:
        expected = {
            "axis_mapping_status",
            "blend_axis_status",
            "coverage_status",
            "crop_status",
            "engine_oracle_status",
            "final_anchor_status",
            "inference_authorized",
            "inference_run",
            "legacy_bundle_status",
            "legacy_general_padded_reuse_status",
            "legacy_unpadded_enumeration_status",
            "model_loaded",
            "monai_padding_source_status",
            "monai_scan_order_status",
            "padding_policies_equivalent",
            "plan_sha256",
            "private_data_accessed",
            "private_shape_recorded",
            "private_window_count_recorded",
            "proof_case_count",
            "proof_sha256",
            "roi_mapping_status",
            "schema_version",
            "status",
            "traversal_status",
            "voxelscope_padding_status",
        }
        data = strict_fields(data, expected, "WindowBridgeReport")
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["plan_sha256"], "plan_sha256"),
            require_int(data["proof_case_count"], "proof_case_count"),
            require_string(data["proof_sha256"], "proof_sha256"),
            cast(Any, require_string(data["axis_mapping_status"], "axis_mapping_status")),
            cast(Any, require_string(data["roi_mapping_status"], "roi_mapping_status")),
            cast(Any, require_string(data["engine_oracle_status"], "engine_oracle_status")),
            cast(
                Any,
                require_string(data["monai_scan_order_status"], "monai_scan_order_status"),
            ),
            cast(
                Any,
                require_string(data["voxelscope_padding_status"], "voxelscope_padding_status"),
            ),
            cast(
                Any,
                require_string(
                    data["monai_padding_source_status"],
                    "monai_padding_source_status",
                ),
            ),
            cast(
                Any,
                require_bool(
                    data["padding_policies_equivalent"],
                    "padding_policies_equivalent",
                ),
            ),
            cast(Any, require_string(data["crop_status"], "crop_status")),
            cast(Any, require_string(data["blend_axis_status"], "blend_axis_status")),
            cast(Any, require_string(data["traversal_status"], "traversal_status")),
            cast(Any, require_string(data["final_anchor_status"], "final_anchor_status")),
            cast(Any, require_string(data["coverage_status"], "coverage_status")),
            cast(Any, require_string(data["legacy_bundle_status"], "legacy_bundle_status")),
            cast(
                Any,
                require_string(
                    data["legacy_unpadded_enumeration_status"],
                    "legacy_unpadded_enumeration_status",
                ),
            ),
            cast(
                Any,
                require_string(
                    data["legacy_general_padded_reuse_status"],
                    "legacy_general_padded_reuse_status",
                ),
            ),
            cast(
                Any,
                require_bool(data["private_data_accessed"], "private_data_accessed"),
            ),
            cast(
                Any,
                require_bool(data["private_shape_recorded"], "private_shape_recorded"),
            ),
            cast(
                Any,
                require_bool(
                    data["private_window_count_recorded"],
                    "private_window_count_recorded",
                ),
            ),
            cast(Any, require_bool(data["model_loaded"], "model_loaded")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(
                Any,
                require_bool(data["inference_authorized"], "inference_authorized"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowBridgeRefusal:
    schema_version: Literal["voxelscope/window-bridge-refusal/v1"]
    status: Literal["refused"]
    refusal_id: str
    error_code: str
    stage: BridgeStage
    plan_sha256: str
    report_published: Literal[False]
    private_data_accessed: Literal[False]
    model_loaded: Literal[False]
    inference_run: Literal[False]
    inference_authorized: Literal[False]

    def __post_init__(self) -> None:
        if (
            self.schema_version != "voxelscope/window-bridge-refusal/v1"
            or self.status != "refused"
            or not self.refusal_id
            or not self.error_code
            or self.stage not in _BRIDGE_STAGES
        ):
            raise EvidenceError("invalid_window_bridge_refusal", self.status)
        require_sha256(self.plan_sha256)
        if (
            self.report_published
            or self.private_data_accessed
            or self.model_loaded
            or self.inference_run
            or self.inference_authorized
        ):
            raise EvidenceError("invalid_window_bridge_refusal", self.error_code)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowBridgeRefusal:
        data = strict_fields(
            data,
            {
                "error_code",
                "inference_authorized",
                "inference_run",
                "model_loaded",
                "plan_sha256",
                "private_data_accessed",
                "refusal_id",
                "report_published",
                "schema_version",
                "stage",
                "status",
            },
            "WindowBridgeRefusal",
        )
        stage = require_string(data["stage"], "stage")
        if stage not in _BRIDGE_STAGES:
            raise EvidenceError("invalid_window_bridge_stage", stage)
        return cls(
            cast(Any, require_string(data["schema_version"], "schema_version")),
            cast(Any, require_string(data["status"], "status")),
            require_string(data["refusal_id"], "refusal_id"),
            require_string(data["error_code"], "error_code"),
            cast(BridgeStage, stage),
            require_string(data["plan_sha256"], "plan_sha256"),
            cast(Any, require_bool(data["report_published"], "report_published")),
            cast(
                Any,
                require_bool(data["private_data_accessed"], "private_data_accessed"),
            ),
            cast(Any, require_bool(data["model_loaded"], "model_loaded")),
            cast(Any, require_bool(data["inference_run"], "inference_run")),
            cast(
                Any,
                require_bool(data["inference_authorized"], "inference_authorized"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
