# SPDX-License-Identifier: Apache-2.0
"""Closed public evidence for the milestone 6 positional window bridge."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .atomic import path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    is_link_like,
    load_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .records import require_list, require_object, require_string, strict_fields
from .window_bridge import load_window_bridge_plan, verify_window_bridge_report
from .window_bridge_records import WindowBridgePlan

PUBLIC_EVIDENCE_SCHEMA = "voxelscope/milestone-6-public-evidence/v1"
PUBLIC_BUNDLE_FORMAT = "voxelscope-closed-public-bundle-v1"
EXPECTED_PUBLIC_BUNDLE_SHA256 = "76895b2af591364156ac021ad482d2ed72fd96ad62f33042a4545238c587caa4"
PAYLOAD_PATHS = (
    "README.txt",
    "claim-matrix.json",
    "privacy-report.json",
    "public-summary-v1.json",
    "source-manifest.json",
)
INDEXED_PATHS = (*PAYLOAD_PATHS, "SHA256SUMS")
README_BYTES = (
    b"VoxelScope milestone 6 public evidence\n"
    b"\n"
    b"This closed bundle proves positional I/J/K mapping, unpadded enumeration,\n"
    b"MONAI symmetric padding, and crop behavior using public source identities\n"
    b"and asymmetric synthetic fixtures. Generic reuse of VoxelScope's legacy\n"
    b"high-side padded path is NO-GO. Legacy Z/Y/X labels remain synthetic\n"
    b"coordinate names, not anatomical directions. No private data was read.\n"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "affine",
        "custody_receipt_sha256",
        "dtype",
        "input_spatial_shape",
        "orientation",
        "private_path",
        "real_window_count",
        "spacing",
        "tensor_sha256",
        "voxel_count",
        "zooms",
    }
)
_FORBIDDEN_BYTES = (
    b"/Users/",
    b"/home/",
    b"one-volume-custody",
    b"input-reference.f32le",
    b"input-independent.f32le",
    b"preprocessing-report",
    b"4a73ef8f",
    b"0584b51b",
    b"ee268fb3",
)


def _summary() -> dict[str, Any]:
    return {
        "anatomical_axis_claimed": False,
        "axis_roi_mapping_status": "go",
        "blend_mode": "constant",
        "crop_status": "go",
        "inference_authorized": False,
        "inference_run": False,
        "input_layout": "C,I,J,K",
        "legacy_general_padded_reuse_status": "no-go",
        "legacy_synthetic_bundle_status": "go",
        "legacy_unpadded_enumeration_status": "go",
        "model_layout": "N,C,D,H,W",
        "model_loaded": False,
        "monai_padding_source_status": "go",
        "overall_status": "go",
        "padding_policies_equivalent": False,
        "private_data_accessed": False,
        "private_shape_recorded": False,
        "private_window_count_recorded": False,
        "roi_ijk": [240, 240, 160],
        "scan_interval_status": "go",
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "traversal_order": "lexicographic_ijk_k_fastest",
        "traversal_status": "go",
        "legacy_high_side_padding_oracle_status": "go",
    }


def _claim_matrix() -> dict[str, Any]:
    return {
        "not_claimed": [
            "anatomical axis identity",
            "clinical validity",
            "diagnostic accuracy",
            "model output spatial contract",
            "real inference correctness",
            "segmentation quality",
            "speedup or GPU performance",
        ],
        "proven": [
            "preprocessing I/J/K positions map unchanged to model spatial positions",
            "ROI components map positionally as I=240, J=240, K=160",
            "half-overlap scan intervals and final anchors match an independent oracle",
            "window traversal is lexicographic I/J/K with K changing fastest",
            "VoxelScope high-side constant-zero padding matches an independent oracle",
            "MONAI symmetric padding order is pinned as a distinct source fact",
            "output cropping restores original positional spatial extents",
            "the pinned inference configuration uses constant blending",
            "legacy synthetic bundle identity remains unchanged",
        ],
        "refused": [
            "claim that VoxelScope and MONAI padding placement are equivalent",
            "generic reuse of the legacy high-side padded path for real inference",
            "anatomical relabeling from the source affine",
            "private tensor access",
            "model loading",
            "inference",
        ],
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _privacy_report() -> dict[str, Any]:
    return {
        "medical_values_included": False,
        "private_evidence_identities_included": False,
        "private_paths_included": False,
        "private_tensor_hashes_included": False,
        "real_geometry_included": False,
        "real_input_shape_included": False,
        "real_window_count_included": False,
        "source_subject_locator_included": False,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _source_manifest(
    plan: WindowBridgePlan,
    plan_sha256: str,
    report_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "implementation_sources": [
            {
                "git_blob_sha1": item.git_blob_sha1,
                "path": item.path,
                "sha256": item.sha256,
            }
            for item in plan.implementation_sources
        ],
        "legacy_synthetic_bundle_sha256": plan.legacy_synthetic_bundle_sha256,
        "milestone5_bundle_sha256": plan.milestone5_bundle_sha256,
        "model_inference_config_sha256": plan.model_inference_config_sha256,
        "monai_commit": plan.monai_commit,
        "pytorch_commit": plan.pytorch_commit,
        "upstream_sources": [
            {
                "commit": item.commit,
                "git_blob_sha1": item.git_blob_sha1,
                "immutable_url": item.immutable_url,
                "path": item.path,
                "release": item.release,
                "repository": item.repository,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in plan.upstream_sources
        ],
        "plan_sha256": plan_sha256,
        "report_sha256": report_sha256,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _write_bytes(root: Path, relative: str, value: bytes) -> None:
    path = root / safe_relative_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def build_milestone6_public_bundle(
    plan_path: Path,
    report_path: Path,
    output: Path,
    *,
    repository_root: Path,
) -> str:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    report = verify_window_bridge_report(
        plan_path,
        report_path,
        repository_root=repository_root,
    )
    plan = load_window_bridge_plan(plan_path, repository_root=repository_root)
    if report.plan_sha256 != sha256_file(plan_path):
        raise EvidenceError("window_bridge_plan_mismatch", report.plan_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _write_bytes(stage, "README.txt", README_BYTES)
        records = {
            "claim-matrix.json": _claim_matrix(),
            "privacy-report.json": _privacy_report(),
            "public-summary-v1.json": _summary(),
            "source-manifest.json": _source_manifest(
                plan,
                report.plan_sha256,
                sha256_file(report_path),
                repository_root,
            ),
        }
        for relative, value in records.items():
            write_json(stage / relative, value)
        sums = "".join(
            f"{sha256_file(stage / relative)}  {relative}\n" for relative in sorted(PAYLOAD_PATHS)
        )
        _write_bytes(stage, "SHA256SUMS", sums.encode("ascii"))
        write_json(
            stage / "bundle.json",
            {
                "artifacts": [
                    {
                        "path": relative,
                        "sha256": sha256_file(stage / relative),
                        "size_bytes": (stage / relative).stat().st_size,
                    }
                    for relative in sorted(INDEXED_PATHS)
                ],
                "bundle_format": PUBLIC_BUNDLE_FORMAT,
                "schema_version": PUBLIC_EVIDENCE_SCHEMA,
            },
        )
        root_hash = sha256_file(stage / "bundle.json")
        _write_bytes(stage, "bundle.sha256", f"{root_hash}  bundle.json\n".encode("ascii"))
        verify_milestone6_public_bundle(
            stage,
            repository_root=repository_root,
            require_trusted_digest=False,
        )
        rename_no_replace(stage, output)
        return root_hash
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _scan_keys(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = _FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if forbidden:
            raise EvidenceError("private_field_in_public_evidence", sorted(forbidden)[0])
        for item in value.values():
            _scan_keys(item)
    elif isinstance(value, list):
        for item in value:
            _scan_keys(item)


def _read_regular_bounded(path: Path, *, maximum_bytes: int) -> bytes:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("special_file_forbidden", path.name)
    if path.stat().st_size > maximum_bytes:
        raise EvidenceError("public_file_limit", path.name)
    return path.read_bytes()


def verify_milestone6_public_bundle(
    root: Path,
    *,
    repository_root: Path,
    require_trusted_digest: bool = True,
) -> str:
    if is_link_like(root) or not root.is_dir():
        raise EvidenceError("missing_bundle", str(root))
    expected_files = {*INDEXED_PATHS, "bundle.json", "bundle.sha256"}
    actual_files: set[str] = set()
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        if is_link_like(current):
            raise EvidenceError("symlink_forbidden", "public bundle directory")
        for name in subdirectories:
            if is_link_like(current / name):
                raise EvidenceError("symlink_forbidden", "public bundle directory")
        for name in filenames:
            path = current / name
            if is_link_like(path) or not path.is_file():
                raise EvidenceError("special_file_forbidden", name)
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise EvidenceError("bundle_file_set_mismatch", "public evidence set differs")

    contents = {
        relative: _read_regular_bounded(root / relative, maximum_bytes=1024 * 1024)
        for relative in expected_files
    }
    root_hash = sha256_bytes(contents["bundle.json"])
    if (
        require_trusted_digest
        and EXPECTED_PUBLIC_BUNDLE_SHA256 != "TO_BE_FINALIZED"
        and root_hash != EXPECTED_PUBLIC_BUNDLE_SHA256
    ):
        raise EvidenceError("public_bundle_contract_mismatch", root_hash)
    if contents["bundle.sha256"] != f"{root_hash}  bundle.json\n".encode("ascii"):
        raise EvidenceError("invalid_bundle_digest", "public bundle digest differs")

    index = strict_fields(
        require_object(load_json_bytes(contents["bundle.json"]), "bundle index"),
        {"artifacts", "bundle_format", "schema_version"},
        "bundle index",
    )
    if (
        index["schema_version"] != PUBLIC_EVIDENCE_SCHEMA
        or index["bundle_format"] != PUBLIC_BUNDLE_FORMAT
    ):
        raise EvidenceError("invalid_bundle_index", "public bundle identity differs")
    indexed: set[str] = set()
    for raw in require_list(index["artifacts"], "artifacts"):
        item = strict_fields(
            require_object(raw, "artifact"),
            {"path", "sha256", "size_bytes"},
            "artifact",
        )
        relative = require_string(item["path"], "path")
        safe_relative_path(relative)
        if (
            relative in indexed
            or relative not in contents
            or item["size_bytes"] != len(contents[relative])
            or item["sha256"] != sha256_bytes(contents[relative])
        ):
            raise EvidenceError("bundle_artifact_mismatch", relative)
        indexed.add(relative)
    if indexed != set(INDEXED_PATHS):
        raise EvidenceError("bundle_index_set_mismatch", "public index differs")
    expected_sums = "".join(
        f"{sha256_bytes(contents[relative])}  {relative}\n" for relative in sorted(PAYLOAD_PATHS)
    ).encode("ascii")
    if contents["SHA256SUMS"] != expected_sums:
        raise EvidenceError("checksum_manifest_mismatch", "SHA256SUMS differs")
    if contents["README.txt"] != README_BYTES:
        raise EvidenceError("public_readme_mismatch", "README differs")

    records = {
        relative: load_json_bytes(contents[relative])
        for relative in PAYLOAD_PATHS
        if relative.endswith(".json")
    }
    for value in records.values():
        _scan_keys(value)
    joined = b"\n".join(contents.values())
    if any(pattern in joined for pattern in _FORBIDDEN_BYTES):
        raise EvidenceError("private_value_in_public_evidence", "forbidden value")
    if records["claim-matrix.json"] != _claim_matrix():
        raise EvidenceError("public_claim_matrix_mismatch", "claim matrix differs")
    if records["privacy-report.json"] != _privacy_report():
        raise EvidenceError("public_privacy_report_mismatch", "privacy report differs")
    if records["public-summary-v1.json"] != _summary():
        raise EvidenceError("public_summary_mismatch", "summary differs")

    manifest = require_object(records["source-manifest.json"], "source manifest")
    plan_path = repository_root / "research/window-bridge-plan-v1.json"
    report_path = repository_root / "research/window-bridge-report-v1.json"
    report = verify_window_bridge_report(
        plan_path,
        report_path,
        repository_root=repository_root,
    )
    plan = load_window_bridge_plan(plan_path, repository_root=repository_root)
    expected_manifest = _source_manifest(
        plan,
        report.plan_sha256,
        sha256_file(report_path),
        repository_root,
    )
    if manifest != expected_manifest:
        raise EvidenceError("source_manifest_mismatch", "source identities differ")
    return root_hash
