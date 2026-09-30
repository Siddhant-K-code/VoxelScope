# SPDX-License-Identifier: Apache-2.0
"""Synthetic-only closed public evidence for milestone 7."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

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
from .window_execution import (
    axis_starts,
    load_window_execution_plan,
    qualify_windows,
    symmetric_padding,
)
from .window_execution_contract import (
    LEGACY_SYNTHETIC_BUNDLE_SHA256,
    MILESTONE5_ADAPTER_PLAN_SHA256,
    MILESTONE5_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_PLAN_SHA256,
    MILESTONE6_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_REPORT_SHA256,
    TRUSTED_PUBLIC_BUNDLE_SHA256,
)

PUBLIC_EVIDENCE_SCHEMA = "voxelscope/milestone-7-public-evidence/v1"
PUBLIC_BUNDLE_FORMAT = "voxelscope-closed-public-bundle-v1"
PAYLOAD_PATHS = (
    "README.txt",
    "claim-matrix.json",
    "privacy-report.json",
    "public-summary-v1.json",
    "source-manifest.json",
)
INDEXED_PATHS = (*PAYLOAD_PATHS, "SHA256SUMS")
README_BYTES = (
    b"VoxelScope milestone 7 public evidence\n"
    b"\n"
    b"This closed bundle records implementation-only qualification of two\n"
    b"independent symmetric window materialization paths over synthetic fixtures.\n"
    b"No private preprocessing report, tensor, geometry, path, identity, or real\n"
    b"window count is included. No model was loaded and no inference was run.\n"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "affine",
        "approval_sha256",
        "authorization_sha256",
        "custody_receipt_sha256",
        "dtype",
        "input_shape",
        "orientation",
        "preprocessing_report_sha256",
        "preprocessing_snapshot_sha256",
        "private_path",
        "real_window_count",
        "receipt_sha256",
        "spacing",
        "tensor_sha256",
        "voxel_count",
        "window_stream_sha256",
        "zooms",
    }
)
_FORBIDDEN_BYTES = (
    b"/Users/",
    b"/home/",
    b"one-volume-custody",
    b"preprocessing-v1",
    b"input-reference.f32le",
    b"input-independent.f32le",
    b"preprocessing-report",
    b"window-execution-attempts",
)


def _fixture_values(shape: tuple[int, int, int]) -> np.ndarray[Any, np.dtype[np.float32]]:
    i, j, k = np.indices(shape, dtype=np.float32)
    base = i * np.float32(10000) + j * np.float32(100) + k + np.float32(1)
    return np.stack([base + np.float32(channel * 1000000) for channel in range(4)]).astype("<f4")


def qualify_synthetic_fixtures() -> tuple[int, int]:
    materialization_cases = (
        ((7, 8, 9), (4, 5, 3), (2, 2, 1)),
        ((3, 4, 5), (6, 7, 8), (3, 3, 4)),
        ((4, 5, 6), (8, 9, 10), (4, 4, 5)),
        ((5, 6, 7), (5, 6, 7), (2, 3, 3)),
        ((4, 11, 13), (7, 6, 5), (3, 3, 2)),
        ((9, 10, 11), (4, 5, 6), (2, 2, 3)),
        ((6, 7, 12), (5, 4, 5), (2, 2, 2)),
    )
    for shape, roi, stride in materialization_cases:
        ledger = qualify_windows(
            _fixture_values(shape),
            roi_ijk=roi,
            stride_ijk=stride,
        )
        if (
            ledger.window_count <= 0
            or not ledger.exact_window_bytes_equal
            or not ledger.complete_source_coverage
        ):
            raise EvidenceError("synthetic_fixture_failed", str(shape))
    if (
        symmetric_padding((240, 240, 160), (240, 240, 160)) != ((0, 0, 0), (0, 0, 0))
        or axis_starts(240, 240, 120) != (0,)
        or axis_starts(160, 160, 80) != (0,)
    ):
        raise EvidenceError("synthetic_fixture_failed", "exact ROI")
    refusal_cases = (
        (np.zeros((4, 1, 1, 1), dtype="<f4"), "zero_channel"),
        (
            np.full((4, 1, 1, 1), np.float32(np.nan), dtype="<f4"),
            "nonfinite_input",
        ),
        (np.empty((4, 0, 1, 1), dtype="<f4"), "invalid_tensor_layout"),
    )
    for values, expected in refusal_cases:
        try:
            qualify_windows(values, roi_ijk=(1, 1, 1), stride_ijk=(1, 1, 1))
        except EvidenceError as exc:
            if exc.code != expected:
                raise EvidenceError("synthetic_refusal_mismatch", expected) from exc
        else:
            raise EvidenceError("synthetic_refusal_missing", expected)
    return 8, len(refusal_cases)


def _summary() -> dict[str, Any]:
    success_count, refusal_count = qualify_synthetic_fixtures()
    return {
        "bounded_private_output_status": "go",
        "dual_materialization_status": "go",
        "exact_window_bytes_status": "go",
        "inference_authorized": False,
        "inference_run": False,
        "label_excluded_status": "go",
        "legacy_general_padded_reuse_status": "no-go",
        "model_loaded": False,
        "overall_status": "implementation-ready",
        "private_data_accessed": False,
        "real_execution_status": "not-run",
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "synthetic_fixture_count": success_count,
        "synthetic_fixture_status": "go",
        "synthetic_refusal_fixture_count": refusal_count,
        "synthetic_refusal_status": "go",
    }


def _claim_matrix() -> dict[str, Any]:
    return {
        "not_claimed": [
            "anatomical axis identity",
            "clinical validity",
            "diagnostic accuracy",
            "model output semantics",
            "private preprocessing qualification",
            "real window materialization",
            "segmentation quality",
        ],
        "proven": [
            "two independent symmetric window materialization paths agree byte-for-byte",
            "synthetic final anchors, padding placement, intersections, coverage, and crop agree",
            "synthetic traversal is lexicographic I/J/K with K changing fastest",
            "the private execution design is one-shot, no-clobber, bounded, and terminal",
            "historical milestone identities remain unchanged",
        ],
        "refused": [
            "generic reuse of the legacy high-side padded engine",
            "private data access during implementation",
            "window persistence",
            "model loading",
            "inference",
        ],
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _privacy_report() -> dict[str, Any]:
    return {
        "affine_included": False,
        "approval_identity_included": False,
        "dtype_observation_included": False,
        "medical_values_included": False,
        "orientation_included": False,
        "private_paths_included": False,
        "private_report_identity_included": False,
        "private_tensor_hashes_included": False,
        "real_input_shape_included": False,
        "real_window_count_included": False,
        "source_subject_locator_included": False,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "window_hashes_included": False,
    }


def _source_manifest(plan_path: Path, repository_root: Path) -> dict[str, Any]:
    plan, plan_sha256 = load_window_execution_plan(
        plan_path,
        repository_root=repository_root,
        verify_runtime=False,
    )
    return {
        "implementation_sources": [
            {"path": item.path, "sha256": item.sha256} for item in plan.implementation_sources
        ],
        "legacy_synthetic_bundle_sha256": LEGACY_SYNTHETIC_BUNDLE_SHA256,
        "milestone5_adapter_plan_sha256": MILESTONE5_ADAPTER_PLAN_SHA256,
        "milestone5_public_bundle_sha256": MILESTONE5_PUBLIC_BUNDLE_SHA256,
        "milestone6_plan_sha256": MILESTONE6_PLAN_SHA256,
        "milestone6_public_bundle_sha256": MILESTONE6_PUBLIC_BUNDLE_SHA256,
        "milestone6_report_sha256": MILESTONE6_REPORT_SHA256,
        "plan_sha256": plan_sha256,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _write_bytes(root: Path, relative: str, value: bytes) -> None:
    path = root / safe_relative_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def build_milestone7_public_bundle(
    plan_path: Path,
    output: Path,
    *,
    repository_root: Path,
) -> str:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _write_bytes(stage, "README.txt", README_BYTES)
        records = {
            "claim-matrix.json": _claim_matrix(),
            "privacy-report.json": _privacy_report(),
            "public-summary-v1.json": _summary(),
            "source-manifest.json": _source_manifest(plan_path, repository_root),
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
        verify_milestone7_public_bundle(
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


def verify_milestone7_public_bundle(
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
        and TRUSTED_PUBLIC_BUNDLE_SHA256 != "TO_BE_FINALIZED"
        and root_hash != TRUSTED_PUBLIC_BUNDLE_SHA256
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
    plan_path = repository_root / "research/window-execution-plan-v1.json"
    if records["source-manifest.json"] != _source_manifest(plan_path, repository_root):
        raise EvidenceError("source_manifest_mismatch", "source identities differ")
    return root_hash
