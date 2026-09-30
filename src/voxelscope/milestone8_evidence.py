# SPDX-License-Identifier: Apache-2.0
"""Synthetic-only closed public evidence for Milestone 8."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

from .atomic import path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    is_link_like,
    load_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .custody_records import ArchiveMember
from .model_loading import extract_selected_archive, load_model_loading_plan, run_loader_worker
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
    TRUSTED_PUBLIC_BUNDLE_SHA256,
)
from .model_loading_records import ExtractionMember, LoaderWorkerRequest
from .records import require_list, require_object, require_string, strict_fields

PUBLIC_EVIDENCE_SCHEMA = "voxelscope/milestone-8-public-evidence/v1"
PUBLIC_BUNDLE_FORMAT = "voxelscope-closed-public-bundle-v1"
PAYLOAD_PATHS = (
    "README.txt",
    "claim-matrix.json",
    "privacy-report.json",
    "public-summary-v1.json",
    "source-manifest.json",
    "synthetic-qualification.json",
)
INDEXED_PATHS = (*PAYLOAD_PATHS, "SHA256SUMS")
README_BYTES = (
    b"VoxelScope milestone 8 public evidence\n"
    b"\n"
    b"This closed bundle records only synthetic qualification of a prospective,\n"
    b"one-shot archive extraction and CPU loader protocol. The private official\n"
    b"archive was not accessed or extracted. Its checkpoint was not deserialized,\n"
    b"instantiated, or loaded. No model forward or inference was run.\n"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "approval_sha256",
        "authorization_sha256",
        "custody_receipt_sha256",
        "extraction_snapshot_sha256",
        "future_private_hash",
        "loading_report_sha256",
        "private_path",
        "receipt_sha256",
        "state_dict_keys",
        "state_shapes",
    }
)
_FORBIDDEN_BYTES = (
    b"/Users/",
    b"/home/",
    b"model-loading-attempts",
    b"model-loading-qualification-v1/evidence",
    b"voxelscope/custody-receipt/v1",
)


def _synthetic_tensor() -> dict[str, Any]:
    return {
        "device": "cpu",
        "dtype": "float32",
        "finite": True,
        "layout": "strided",
        "shape": [2, 3],
    }


def qualify_synthetic_model_loading(
    plan_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    plan, plan_sha256 = load_model_loading_plan(plan_path, repository_root=repository_root)
    with tempfile.TemporaryDirectory(prefix="voxelscope-m8-synthetic-") as temporary_name:
        root = Path(temporary_name)
        payloads = {
            "LICENSE": b"synthetic license\n",
            "config.json": canonical_json_bytes(
                {
                    "expected_state": {"weight": _synthetic_tensor()},
                    "format": "voxelscope-safe-loader-config/v1",
                }
            ),
            "checkpoint.json": canonical_json_bytes(
                {
                    "format": "voxelscope-safe-tensor-fixture/v1",
                    "state": {"weight": _synthetic_tensor()},
                }
            ),
        }
        archive_path = root / "fixture.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
        exact = tuple(
            ArchiveMember(name, len(payload), sha256_bytes(payload), "synthetic")
            for name, payload in payloads.items()
        )
        selected = tuple(
            ExtractionMember(
                name,
                cast(Any, role),
                len(payloads[name]),
                sha256_bytes(payloads[name]),
            )
            for name, role in (
                ("LICENSE", "license"),
                ("config.json", "inference-config"),
                ("checkpoint.json", "pytorch-checkpoint"),
            )
        )
        extracted = extract_selected_archive(
            archive_path,
            root / "output",
            exact_members=exact,
            selected_members=selected,
            max_members=3,
            max_total_uncompressed_bytes=4096,
            max_member_bytes=2048,
            max_compression_ratio=100,
        )
        checkpoint = root / "output/extracted/checkpoint.json"
        config = root / "output/extracted/config.json"
        python = Path(sys.executable).absolute()
        result = run_loader_worker(
            python,
            LoaderWorkerRequest(
                "voxelscope/loader-worker-request/v1",
                "synthetic-json",
                plan_sha256,
                next(
                    identity.sha256
                    for identity in plan.implementation_sources
                    if identity.path == "src/voxelscope/model_loader_worker.py"
                ),
                str(checkpoint),
                sha256_file(checkpoint),
                str(config),
                sha256_file(config),
                plan.architecture,
                plan.runtime,
                "0" * 64,
                plan.loader_memory_limit_bytes,
            ),
            timeout_seconds=30,
            approved_python_sha256=sha256_file(python),
        )
        if result.status != "go":
            raise EvidenceError("synthetic_loader_refused", result.error_code or "unknown refusal")
        if len(extracted) != 3:
            raise EvidenceError("synthetic_extraction_failed", "member count differs")
    return {
        "adversarial_contract_status": "go",
        "archive_protocol_status": "go",
        "official_archive_accessed": False,
        "official_checkpoint_deserialized": False,
        "official_model_instantiated": False,
        "official_model_loaded": False,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "synthetic_checkpoint_format": "safe-json-tensor-metadata",
        "synthetic_extracted_member_count": 3,
        "synthetic_forward_called": False,
        "synthetic_loader_protocol_status": "go",
        "synthetic_network_used": False,
    }


def _summary() -> dict[str, Any]:
    return {
        "accelerator_used": False,
        "cloud_used": False,
        "extraction_executed": False,
        "inference_authorized": False,
        "inference_run": False,
        "model_deserialized": False,
        "model_instantiated": False,
        "model_loaded": False,
        "network_used": False,
        "next_gate": "explicit-owner-private-authorization-required",
        "official_private_result": "absent",
        "overall_status": "implementation-ready-private-execution-no-go",
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "spend_authorized": False,
        "synthetic_qualification_status": "go",
    }


def _claim_matrix() -> dict[str, Any]:
    return {
        "not_claimed": [
            "official checkpoint compatibility with weights-only loading",
            "official state-dict keys, shapes, dtypes, or counts",
            "private archive extraction success",
            "private model construction or loading success",
            "model output semantics",
            "inference readiness",
            "clinical validity",
        ],
        "proven": [
            "closed allowlist extraction over public synthetic archives",
            "bounded inherited-FD loader request and result protocol",
            "authorization binds an exact reviewed runtime distribution fingerprint",
            "safe synthetic tensor-mapping validation in an isolated subprocess",
            "prospective one-shot private attempt and terminal evidence contracts",
            "public evidence excludes future private identities and observations",
        ],
        "refused": [
            "unsafe pickle loading",
            "dynamic config evaluation",
            "checkpoint-provided classes",
            "model forward or model-shaped input allocation",
            "GPU or MPS placement",
            "network, cloud, or spend",
            "private execution without a separately approved final plan",
            "private execution without a reviewed runtime distribution fingerprint",
        ],
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _privacy_report() -> dict[str, Any]:
    return {
        "future_private_hashes_included": False,
        "private_archive_path_included": False,
        "private_receipt_identity_included": False,
        "private_report_identity_included": False,
        "private_root_included": False,
        "private_state_counts_included": False,
        "private_state_keys_included": False,
        "private_state_shapes_included": False,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "synthetic_only": True,
    }


def _source_manifest(plan_path: Path, repository_root: Path) -> dict[str, Any]:
    plan, plan_sha256 = load_model_loading_plan(plan_path, repository_root=repository_root)
    return {
        "acquisition_plan_sha256": ACQUISITION_PLAN_SHA256,
        "implementation_sources": [identity.to_dict() for identity in plan.implementation_sources],
        "legacy_synthetic_bundle_sha256": LEGACY_SYNTHETIC_BUNDLE_SHA256,
        "milestone6_plan_sha256": MILESTONE6_PLAN_SHA256,
        "milestone6_public_bundle_sha256": MILESTONE6_PUBLIC_BUNDLE_SHA256,
        "milestone6_report_sha256": MILESTONE6_REPORT_SHA256,
        "milestone7_plan_sha256": MILESTONE7_PLAN_SHA256,
        "milestone7_public_bundle_sha256": MILESTONE7_PUBLIC_BUNDLE_SHA256,
        "model_archive_sha256": MODEL_ARCHIVE_SHA256,
        "model_checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_license_sha256": MODEL_LICENSE_SHA256,
        "model_zoo_commit": MODEL_ZOO_COMMIT,
        "monai_commit": MONAI_COMMIT,
        "plan_sha256": plan_sha256,
        "pytorch_commit": PYTORCH_COMMIT,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "source_registry_sha256": SOURCE_REGISTRY_SHA256,
    }


def _write_bytes(root: Path, relative: str, value: bytes) -> None:
    path = root / safe_relative_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def build_milestone8_public_bundle(
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
            "synthetic-qualification.json": qualify_synthetic_model_loading(
                plan_path, repository_root=repository_root
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
        verify_milestone8_public_bundle(
            stage, repository_root=repository_root, require_trusted_digest=False
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


def verify_milestone8_public_bundle(
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
    plan_path = repository_root / "research/model-loading-plan-v1.json"
    if records["source-manifest.json"] != _source_manifest(plan_path, repository_root):
        raise EvidenceError("source_manifest_mismatch", "source identities differ")
    if records["synthetic-qualification.json"] != qualify_synthetic_model_loading(
        plan_path, repository_root=repository_root
    ):
        raise EvidenceError("synthetic_qualification_mismatch", "synthetic result differs")
    return root_hash
