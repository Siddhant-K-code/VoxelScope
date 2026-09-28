# SPDX-License-Identifier: Apache-2.0
"""Deterministic public evidence for the milestone 4 admission gate."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import stat
import struct
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

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
from .custody_records import DigestIdentity
from .one_volume_contract import DECISION_SHA256, PLAN_SHA256
from .one_volume_custody import (
    _inspect_nifti,
    verify_trusted_one_volume,
    verify_trusted_one_volume_refusal,
)
from .one_volume_custody_records import AcquisitionRefusal
from .one_volume_records import ArtifactRole, PlannedArtifact
from .records import require_int, require_list, require_object, require_string, strict_fields

PUBLIC_EVIDENCE_SCHEMA = "voxelscope/milestone-4-public-evidence/v1"
PUBLIC_BUNDLE_FORMAT = "voxelscope-closed-public-bundle-v1"
EXPECTED_PUBLIC_BUNDLE_SHA256 = "d3fdca9d8e5dccfbe5b1e755ca1a9fa782dd986e1fd00c5caa1997986a1c20df"
GATE_ORDER = (
    "plan_approval",
    "acquisition",
    "content_identity",
    "nifti_structure",
    "finite_values",
    "mask_domain",
    "geometry",
    "preprocessing_adapter",
    "label_semantics",
    "inference_authorization",
)
GATE_STATES = frozenset({"passed", "failed", "not_reached", "not_authorized"})
PAYLOAD_PATHS = (
    "README.txt",
    "claim-matrix.json",
    "privacy-report.json",
    "protocol-config.json",
    "source-manifest.json",
    "stage-ledger.json",
    "synthetic-fixture-manifest.json",
    "synthetic/invalid-mask.synthetic-nifti-gzip",
    "synthetic/malformed.synthetic-nifti-gzip",
    "synthetic/positive-image.synthetic-nifti-gzip",
    "synthetic/positive-mask.synthetic-nifti-gzip",
    "public-summary-v1.json",
)
INDEXED_PATHS = (*PAYLOAD_PATHS, "SHA256SUMS")
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "affine",
        "dtype",
        "label_set",
        "mask_labels",
        "nonzero_voxel_count",
        "orientation",
        "private_path",
        "receipt",
        "receipt_sha256",
        "shape",
        "spacing",
        "structural_report_sha256",
        "zooms",
    }
)
_STRUCTURAL_REFUSAL_CODES = frozenset(
    {
        "gzip_trailing_member",
        "invalid_affine",
        "invalid_gzip",
        "invalid_nifti",
        "invalid_nifti_header",
        "invalid_nifti_magic",
        "invalid_nifti_offset",
        "invalid_nifti_orientation",
        "invalid_nifti_qfac",
        "invalid_nifti_scaling",
        "invalid_nifti_shape",
        "invalid_nifti_spacing",
        "invalid_nifti_units",
        "missing_nifti_transform",
        "nifti_data_limit",
        "nifti_decompression_limit",
        "nifti_extent_mismatch",
        "nifti_header_diagnostic",
        "nifti_shape_mismatch",
        "nifti_voxel_count_mismatch",
        "nifti_voxel_limit",
        "nifti_extensions_forbidden",
        "truncated_nifti",
        "unsupported_nifti_dtype",
        "unsupported_nifti_version",
    }
)
_ACQUISITION_REFUSAL_CODES = frozenset(
    {
        "acquisition_interrupted",
        "artifact_etag_mismatch",
        "artifact_hash_mismatch",
        "artifact_size_mismatch",
        "artifact_version_mismatch",
        "download_failed",
        "invalid_content_length",
        "output_exists",
        "redirect_forbidden",
        "symlink_forbidden",
        "unexpected_http_response",
        "unsafe_origin_policy",
        "unsafe_path",
        "unsafe_version_pin",
        "unsupported_content_encoding",
    }
)
README_BYTES = (
    b"VoxelScope milestone 4 public evidence\n"
    b"\n"
    b"This closed bundle reports sanitized gate states and deterministic synthetic\n"
    b"validator fixtures. It cannot reconstruct or independently reproduce the\n"
    b"private medical-data result. No model load or inference is authorized.\n"
)


def _raw_nifti(data: np.ndarray[Any, Any]) -> bytes:
    if data.shape != (2, 2, 2) or data.dtype not in {np.dtype(np.float32), np.dtype(np.uint8)}:
        raise ValueError("public fixtures use only fixed 2x2x2 float32 or uint8 arrays")
    datatype, bitpix = (16, 32) if data.dtype == np.dtype(np.float32) else (2, 8)
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, 2, 2, 2, 1, 1, 1, 1)
    struct.pack_into("<2h", header, 70, datatype, bitpix)
    struct.pack_into("<8f", header, 76, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    header[123] = 2
    struct.pack_into("<2h", header, 252, 1, 1)
    struct.pack_into("<4f", header, 280, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, 1.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, 1.0, 0.0)
    header[344:348] = b"n+1\0"
    return bytes(header) + data.astype(data.dtype.newbyteorder("<")).tobytes(order="F")


def _synthetic_payloads() -> dict[str, bytes]:
    image = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    mask = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=np.uint8)
    invalid_mask = np.array([[[0, 0], [7, 7]], [[0, 0], [7, 7]]], dtype=np.uint8)
    return {
        "synthetic/invalid-mask.synthetic-nifti-gzip": gzip.compress(
            _raw_nifti(invalid_mask), mtime=0
        ),
        "synthetic/malformed.synthetic-nifti-gzip": b"synthetic malformed gzip",
        "synthetic/positive-image.synthetic-nifti-gzip": gzip.compress(_raw_nifti(image), mtime=0),
        "synthetic/positive-mask.synthetic-nifti-gzip": gzip.compress(_raw_nifti(mask), mtime=0),
    }


def _synthetic_fixture_manifest(fixtures: dict[str, bytes]) -> dict[str, Any]:
    specifications = (
        (
            "synthetic/positive-image.synthetic-nifti-gzip",
            "image-t1c",
            "go",
            None,
        ),
        ("synthetic/positive-mask.synthetic-nifti-gzip", "label", "go", None),
        (
            "synthetic/invalid-mask.synthetic-nifti-gzip",
            "label",
            "no-go",
            "unexpected_mask_label",
        ),
        (
            "synthetic/malformed.synthetic-nifti-gzip",
            "image-t1c",
            "no-go",
            "invalid_gzip",
        ),
    )
    return {
        "fixtures": [
            {
                "expected_error": error,
                "expected_status": status,
                "path": path,
                "role": role,
                "sha256": hashlib.sha256(fixtures[path]).hexdigest(),
                "size_bytes": len(fixtures[path]),
                "synthetic": True,
            }
            for path, role, status, error in specifications
        ],
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _gate_states(
    outcome: Literal["go", "no-go"], first_failed_gate: str | None
) -> list[dict[str, str]]:
    if outcome == "go":
        return [
            {
                "gate": gate,
                "state": (
                    "not_authorized"
                    if gate
                    in {
                        "preprocessing_adapter",
                        "label_semantics",
                        "inference_authorization",
                    }
                    else "passed"
                ),
            }
            for gate in GATE_ORDER
        ]
    if first_failed_gate not in GATE_ORDER[:-3]:
        raise EvidenceError("invalid_public_gate", str(first_failed_gate))
    failed_index = GATE_ORDER.index(first_failed_gate)
    result: list[dict[str, str]] = []
    for index, gate in enumerate(GATE_ORDER):
        if gate in {
            "preprocessing_adapter",
            "label_semantics",
            "inference_authorization",
        }:
            state = "not_authorized"
        elif index < failed_index:
            state = "passed"
        elif index == failed_index:
            state = "failed"
        else:
            state = "not_reached"
        result.append({"gate": gate, "state": state})
    return result


def _write_payload(root: Path, relative: str, data: bytes) -> None:
    path = root / safe_relative_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _source_manifest() -> dict[str, Any]:
    sources = (
        Path(__file__).with_name("one_volume_custody.py"),
        Path(__file__).with_name("one_volume_custody_records.py"),
    )
    return {
        "decision_sha256": DECISION_SHA256,
        "plan_sha256": PLAN_SHA256,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "validator_sources": [
            {
                "path": f"src/voxelscope/{source.name}",
                "sha256": sha256_file(source),
            }
            for source in sources
        ],
    }


def _protocol_config() -> dict[str, Any]:
    return {
        "allowed_source_labels": [0, 1, 2, 3],
        "decision_sha256": DECISION_SHA256,
        "exact_geometry_required": True,
        "extensions_allowed": False,
        "maximum_data_bytes": 268435456,
        "maximum_dimension": 512,
        "maximum_voxels": 64000000,
        "minimum_tumor_label_count": 1,
        "plan_sha256": PLAN_SHA256,
        "required_background_label": 0,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _claim_matrix() -> dict[str, Any]:
    return {
        "diagnostic_accuracy_claimed": False,
        "inference_authorized": False,
        "label_semantic_equivalence_claimed": False,
        "patient_specific_claimed": False,
        "preprocessing_adapter_approved": False,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _privacy_report() -> dict[str, Any]:
    return {
        "forbidden_private_fields_present": False,
        "private_medical_hashes_published": False,
        "private_result_independently_reconstructable": False,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "synthetic_fixtures_only": True,
    }


def _gate_status(gates: list[dict[str, str]], name: str) -> str:
    state = next(item["state"] for item in gates if item["gate"] == name)
    if state == "passed":
        return "go"
    if state == "failed":
        return "no-go"
    return state


def _progress_counts(
    outcome: Literal["go", "no-go"],
    first_failed_gate: str | None,
    *,
    verified_artifact_count: int | None,
    validated_medical_file_count: int | None,
    geometry_comparison_count: int | None,
) -> dict[str, int]:
    if outcome == "go":
        counts = (8, 5, 4)
    elif None not in (
        verified_artifact_count,
        validated_medical_file_count,
        geometry_comparison_count,
    ):
        counts = cast(
            tuple[int, int, int],
            (
                verified_artifact_count,
                validated_medical_file_count,
                geometry_comparison_count,
            ),
        )
    else:
        defaults = {
            "acquisition": (0, 0, 0),
            "content_identity": (8, 0, 0),
            "nifti_structure": (8, 0, 0),
            "geometry": (8, 5, 0),
            "finite_values": (8, 0, 0),
            "mask_domain": (8, 0, 0),
        }
        counts = defaults.get(str(first_failed_gate), (-1, -1, -1))
    verified, validated, compared = counts
    if not 0 <= verified <= 8 or not 0 <= validated <= 5 or not 0 <= compared <= 4:
        raise EvidenceError("invalid_public_progress", "progress count is outside the fixed bound")
    if outcome == "go" and counts != (8, 5, 4):
        raise EvidenceError("invalid_public_progress", "GO requires complete progress")
    if outcome == "no-go":
        if first_failed_gate == "acquisition" and (verified >= 8 or validated or compared):
            raise EvidenceError("invalid_public_progress", "acquisition refusal progress differs")
        if first_failed_gate == "content_identity" and counts != (8, 0, 0):
            raise EvidenceError("invalid_public_progress", "content refusal progress differs")
        if first_failed_gate in {"nifti_structure", "finite_values", "mask_domain"} and (
            verified != 8 or validated >= 5 or compared
        ):
            raise EvidenceError("invalid_public_progress", "NIfTI refusal progress differs")
        if first_failed_gate == "geometry" and (verified != 8 or validated != 5 or compared >= 4):
            raise EvidenceError("invalid_public_progress", "geometry refusal progress differs")
    return {
        "failed_structural_gate_count": 0 if outcome == "go" else 1,
        "geometry_comparison_count": compared,
        "nifti_validated_medical_file_count": validated,
        "verified_acquired_artifact_count": verified,
    }


def _structural_status(gates: list[dict[str, str]]) -> str:
    states = {
        item["state"]
        for item in gates
        if item["gate"] in {"nifti_structure", "geometry", "finite_values", "mask_domain"}
    }
    if states == {"passed"}:
        return "go"
    if "failed" in states:
        return "no-go"
    return "not_reached"


def _public_refusal_gate(error_code: str) -> str:
    if error_code in _ACQUISITION_REFUSAL_CODES:
        return "acquisition"
    if error_code in {"dataset_metadata_mismatch", "invalid_dataset_metadata"}:
        return "content_identity"
    if error_code in _STRUCTURAL_REFUSAL_CODES or error_code == "nifti_dtype_mismatch":
        return "nifti_structure"
    if error_code == "nifti_geometry_mismatch":
        return "geometry"
    if error_code in {"empty_image", "nonfinite_voxel", "nifti_voxel_read_failed"}:
        return "finite_values"
    if error_code in {
        "missing_expected_mask_label",
        "noninteger_mask",
        "unexpected_mask_label",
    }:
        return "mask_domain"
    raise EvidenceError("unpublishable_refusal", "unknown refusal code")


def _validated_public_refusal_gate(refusal: AcquisitionRefusal) -> str:
    first_failed_gate = _public_refusal_gate(refusal.error_code)
    progress = (
        refusal.verified_artifact_count,
        refusal.validated_medical_file_count,
        refusal.geometry_comparison_count,
    )
    valid = {
        "acquisition": (
            refusal.last_completed_stage in {"attempt_record", "artifact_verification"}
            and 0 <= progress[0] <= 7
            and progress[1:] == (0, 0)
        ),
        "content_identity": (
            refusal.last_completed_stage == "content_identity" and progress == (8, 0, 0)
        ),
        "nifti_structure": (
            refusal.last_completed_stage == "metadata_semantics"
            and progress[0] == 8
            and 0 <= progress[1] <= 5
            and progress[2] == 0
        ),
        "finite_values": (
            refusal.last_completed_stage == "metadata_semantics"
            and progress[0] == 8
            and 0 <= progress[1] <= 4
            and progress[2] == 0
        ),
        "mask_domain": (
            refusal.last_completed_stage == "metadata_semantics"
            and progress[0] == 8
            and progress[1] == 4
            and progress[2] == 0
        ),
        "geometry": (
            refusal.last_completed_stage == "metadata_semantics"
            and progress[0] == 8
            and progress[1] == 5
            and 0 <= progress[2] <= 3
        ),
    }
    if not valid[first_failed_gate]:
        raise EvidenceError("refusal_progress_mismatch", f"{first_failed_gate} progress differs")
    return first_failed_gate


def _build_milestone4_public_bundle(
    output: Path,
    *,
    outcome: Literal["go", "no-go"],
    first_failed_gate: str | None = None,
    verified_artifact_count: int | None = None,
    validated_medical_file_count: int | None = None,
    geometry_comparison_count: int | None = None,
) -> str:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        gates = _gate_states(outcome, first_failed_gate)
        progress_counts = _progress_counts(
            outcome,
            first_failed_gate,
            verified_artifact_count=verified_artifact_count,
            validated_medical_file_count=validated_medical_file_count,
            geometry_comparison_count=geometry_comparison_count,
        )
        fixtures = _synthetic_payloads()
        for relative, payload in fixtures.items():
            _write_payload(stage, relative, payload)
        fixture_manifest = _synthetic_fixture_manifest(fixtures)
        records = {
            "protocol-config.json": _protocol_config(),
            "public-summary-v1.json": {
                "artifact_count": 8,
                "gates": gates,
                "medical_artifact_count": 5,
                "metadata_artifact_count": 3,
                "overall_status": outcome,
                **progress_counts,
                "schema_version": PUBLIC_EVIDENCE_SCHEMA,
            },
            "stage-ledger.json": {
                "acquisition_status": _gate_status(gates, "acquisition"),
                "attempt_terminal_status": "completed" if outcome == "go" else "refused",
                "content_identity_status": _gate_status(gates, "content_identity"),
                "model_loaded": False,
                "network_scope": "eight-pinned-objects-only",
                "schema_version": PUBLIC_EVIDENCE_SCHEMA,
                "structural_validation_status": _structural_status(gates),
            },
            "claim-matrix.json": _claim_matrix(),
            "privacy-report.json": _privacy_report(),
            "synthetic-fixture-manifest.json": fixture_manifest,
            "source-manifest.json": _source_manifest(),
        }
        _write_payload(stage, "README.txt", README_BYTES)
        for relative, value in records.items():
            write_json(stage / relative, value)
        checksum_lines = [
            f"{sha256_file(stage / relative)}  {relative}\n" for relative in sorted(PAYLOAD_PATHS)
        ]
        _write_payload(stage, "SHA256SUMS", "".join(checksum_lines).encode("ascii"))
        index = {
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
        }
        write_json(stage / "bundle.json", index)
        root_hash = sha256_file(stage / "bundle.json")
        _write_payload(stage, "bundle.sha256", f"{root_hash}  bundle.json\n".encode("ascii"))
        rename_no_replace(stage, output)
        return root_hash
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def build_milestone4_public_bundle(
    decision_path: Path,
    plan_path: Path,
    root: Path,
    output: Path,
    *,
    approve_plan_sha256: str,
    repository_root: Path | None = None,
) -> str:
    verify_trusted_one_volume(
        decision_path,
        plan_path,
        root,
        approve_plan_sha256=approve_plan_sha256,
        repository_root=repository_root,
    )
    return _build_milestone4_public_bundle(output, outcome="go")


def build_milestone4_refusal_public_bundle(
    decision_path: Path,
    plan_path: Path,
    root: Path,
    output: Path,
    *,
    approve_plan_sha256: str,
    repository_root: Path | None = None,
) -> str:
    refusal = verify_trusted_one_volume_refusal(
        decision_path,
        plan_path,
        root,
        approve_plan_sha256=approve_plan_sha256,
        repository_root=repository_root,
    )
    first_failed_gate = _validated_public_refusal_gate(refusal)
    return _build_milestone4_public_bundle(
        output,
        outcome="no-go",
        first_failed_gate=first_failed_gate,
        verified_artifact_count=refusal.verified_artifact_count,
        validated_medical_file_count=refusal.validated_medical_file_count,
        geometry_comparison_count=refusal.geometry_comparison_count,
    )


def _scan_keys(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if forbidden:
            raise EvidenceError("private_field_in_public_evidence", sorted(forbidden)[0])
        for item in value.values():
            _scan_keys(item)
    elif isinstance(value, list):
        for item in value:
            _scan_keys(item)


def _fixture_artifact(path: str, role: ArtifactRole, payload: bytes) -> PlannedArtifact:
    digest = hashlib.sha256(payload).hexdigest()
    return PlannedArtifact(
        f"public-synthetic-{role}",
        role,
        f"https://raw.githubusercontent.com/OpenNeuroDatasets/ds007045/{'0' * 40}/fixture",
        f"git-blob-sha1:{'0' * 40}",
        ("https://raw.githubusercontent.com",),
        path,
        len(payload),
        (DigestIdentity("sha256", digest),),
        role in {"image-t1c", "image-t1", "image-t2", "image-flair", "label"},
        role in {"image-t1c", "image-t1", "image-t2", "image-flair", "label"},
    )


def _read_regular_bounded(path: Path, *, maximum_bytes: int) -> bytes:
    if is_link_like(path):
        raise EvidenceError("symlink_forbidden", "public bundle entry")
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise EvidenceError("unsafe_path", "public bundle entry") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_size > maximum_bytes
        ):
            raise EvidenceError("public_file_limit", "public bundle entry")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, min(65536, maximum_bytes - size + 1)):
            size += len(chunk)
            if size > maximum_bytes:
                raise EvidenceError("public_file_limit", "public bundle entry")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if size != opened.st_size or after.st_size != opened.st_size:
            raise EvidenceError("public_file_changed", "public bundle entry")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify_milestone4_public_bundle(root: Path, *, require_trusted_digest: bool = True) -> str:
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
        relative: _read_regular_bounded(root / relative, maximum_bytes=2 * 1024 * 1024)
        for relative in expected_files
    }
    root_hash = sha256_bytes(contents["bundle.json"])
    if require_trusted_digest:
        if EXPECTED_PUBLIC_BUNDLE_SHA256 is None:
            raise EvidenceError("public_evidence_not_pinned", "trusted bundle digest is not set")
        if root_hash != EXPECTED_PUBLIC_BUNDLE_SHA256:
            raise EvidenceError("public_bundle_contract_mismatch", "trusted digest differs")
    if contents["bundle.sha256"] != f"{root_hash}  bundle.json\n".encode("ascii"):
        raise EvidenceError("invalid_bundle_digest", "public bundle digest differs")
    index = require_object(load_json_bytes(contents["bundle.json"]), "public bundle index")
    index = strict_fields(
        index,
        {"artifacts", "bundle_format", "schema_version"},
        "public bundle index",
    )
    if (
        require_string(index["schema_version"], "schema_version") != PUBLIC_EVIDENCE_SCHEMA
        or require_string(index["bundle_format"], "bundle_format") != PUBLIC_BUNDLE_FORMAT
    ):
        raise EvidenceError("invalid_bundle_index", "public bundle identity differs")
    artifacts = require_list(index["artifacts"], "artifacts")
    indexed_paths: set[str] = set()
    for raw in artifacts:
        item = strict_fields(
            require_object(raw, "artifact"),
            {"path", "sha256", "size_bytes"},
            "artifact",
        )
        relative = require_string(item["path"], "path")
        safe_relative_path(relative)
        if (
            relative in indexed_paths
            or relative not in contents
            or item["size_bytes"] != len(contents[relative])
            or item["sha256"] != sha256_bytes(contents[relative])
        ):
            raise EvidenceError("bundle_artifact_mismatch", relative)
        indexed_paths.add(relative)
    if indexed_paths != set(INDEXED_PATHS):
        raise EvidenceError("bundle_index_set_mismatch", "public index differs")
    expected_sums = "".join(
        f"{sha256_bytes(contents[relative])}  {relative}\n" for relative in sorted(PAYLOAD_PATHS)
    ).encode("ascii")
    if contents["SHA256SUMS"] != expected_sums:
        raise EvidenceError("checksum_manifest_mismatch", "SHA256SUMS differs")
    json_records = {
        relative: load_json_bytes(contents[relative])
        for relative in PAYLOAD_PATHS
        if relative.endswith(".json")
    }
    for value in json_records.values():
        _scan_keys(value)
    if contents["README.txt"] != README_BYTES:
        raise EvidenceError("public_readme_mismatch", "README differs")
    if json_records["protocol-config.json"] != _protocol_config():
        raise EvidenceError("public_protocol_mismatch", "protocol configuration differs")
    if json_records["source-manifest.json"] != _source_manifest():
        raise EvidenceError("public_source_manifest_mismatch", "source manifest differs")
    if json_records["claim-matrix.json"] != _claim_matrix():
        raise EvidenceError("public_claim_matrix_mismatch", "claim matrix differs")
    if json_records["privacy-report.json"] != _privacy_report():
        raise EvidenceError("public_privacy_report_mismatch", "privacy report differs")
    summary = require_object(json_records["public-summary-v1.json"], "summary")
    summary = strict_fields(
        summary,
        {
            "artifact_count",
            "failed_structural_gate_count",
            "gates",
            "geometry_comparison_count",
            "medical_artifact_count",
            "metadata_artifact_count",
            "nifti_validated_medical_file_count",
            "overall_status",
            "schema_version",
            "verified_acquired_artifact_count",
        },
        "summary",
    )
    outcome = require_string(summary["overall_status"], "overall_status")
    if (
        outcome not in {"go", "no-go"}
        or summary["artifact_count"] != 8
        or summary["medical_artifact_count"] != 5
        or summary["metadata_artifact_count"] != 3
        or summary["schema_version"] != PUBLIC_EVIDENCE_SCHEMA
    ):
        raise EvidenceError("invalid_public_summary", "summary differs")
    gates = require_list(summary["gates"], "gates")
    if [require_object(item, "gate").get("gate") for item in gates] != list(GATE_ORDER):
        raise EvidenceError("invalid_public_gate_order", "gate order differs")
    if any(require_object(item, "gate").get("state") not in GATE_STATES for item in gates):
        raise EvidenceError("invalid_public_gate_state", "gate state differs")
    failed = [
        require_object(item, "gate").get("gate")
        for item in gates
        if require_object(item, "gate").get("state") == "failed"
    ]
    first_failed = require_string(failed[0], "failed gate") if len(failed) == 1 else None
    if gates != _gate_states(cast(Literal["go", "no-go"], outcome), first_failed):
        raise EvidenceError("invalid_public_gate_state", "gate sequence differs")
    expected_progress = _progress_counts(
        cast(Literal["go", "no-go"], outcome),
        first_failed,
        verified_artifact_count=require_int(
            summary["verified_acquired_artifact_count"],
            "verified_acquired_artifact_count",
        ),
        validated_medical_file_count=require_int(
            summary["nifti_validated_medical_file_count"],
            "nifti_validated_medical_file_count",
        ),
        geometry_comparison_count=require_int(
            summary["geometry_comparison_count"],
            "geometry_comparison_count",
        ),
    )
    observed_progress = {key: require_int(summary[key], key) for key in expected_progress}
    if observed_progress != expected_progress:
        raise EvidenceError("invalid_public_progress", "summary progress differs")
    ledger = require_object(json_records["stage-ledger.json"], "stage ledger")
    expected_gates = cast(list[dict[str, str]], gates)
    expected_ledger = {
        "acquisition_status": _gate_status(expected_gates, "acquisition"),
        "attempt_terminal_status": "completed" if outcome == "go" else "refused",
        "content_identity_status": _gate_status(expected_gates, "content_identity"),
        "model_loaded": False,
        "network_scope": "eight-pinned-objects-only",
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "structural_validation_status": _structural_status(expected_gates),
    }
    if ledger != expected_ledger:
        raise EvidenceError("public_stage_ledger_mismatch", "stage ledger differs")
    fixture_manifest = require_object(
        json_records["synthetic-fixture-manifest.json"], "fixture manifest"
    )
    fixture_manifest = strict_fields(
        fixture_manifest,
        {"fixtures", "schema_version"},
        "fixture manifest",
    )
    if fixture_manifest["schema_version"] != PUBLIC_EVIDENCE_SCHEMA:
        raise EvidenceError("invalid_synthetic_fixture", "schema version")
    trusted_fixtures = _synthetic_payloads()
    if fixture_manifest != _synthetic_fixture_manifest(trusted_fixtures):
        raise EvidenceError("invalid_synthetic_fixture", "fixture manifest differs")
    fixtures = require_list(fixture_manifest["fixtures"], "fixtures")
    expected_fixture_paths = {
        relative for relative in PAYLOAD_PATHS if relative.startswith("synthetic/")
    }
    observed_fixture_paths: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="voxelscope-public-fixtures-") as temporary:
        inspection = Path(temporary)
        for raw in fixtures:
            fixture = strict_fields(
                require_object(raw, "fixture"),
                {
                    "expected_error",
                    "expected_status",
                    "path",
                    "role",
                    "sha256",
                    "size_bytes",
                    "synthetic",
                },
                "fixture",
            )
            relative = require_string(fixture["path"], "fixture path")
            role = require_string(fixture["role"], "fixture role")
            if role not in {"image-t1c", "label"}:
                raise EvidenceError("invalid_synthetic_fixture", role)
            safe_relative_path(relative)
            payload = contents.get(relative, b"")
            if (
                relative in observed_fixture_paths
                or fixture["synthetic"] is not True
                or fixture["size_bytes"] != len(payload)
                or fixture["sha256"] != hashlib.sha256(payload).hexdigest()
                or payload != trusted_fixtures.get(relative)
            ):
                raise EvidenceError("invalid_synthetic_fixture", relative)
            observed_fixture_paths.add(relative)
            artifact = _fixture_artifact(relative, cast(ArtifactRole, role), payload)
            expected_status = require_string(fixture["expected_status"], "expected status")
            fixture_path = inspection / Path(relative).name
            fixture_path.write_bytes(payload)
            try:
                _inspect_nifti(artifact, fixture_path, inspection)
            except EvidenceError as exc:
                if expected_status != "no-go" or fixture.get("expected_error") != exc.code:
                    raise EvidenceError("synthetic_fixture_mismatch", relative) from exc
            else:
                if expected_status != "go" or fixture.get("expected_error") is not None:
                    raise EvidenceError("synthetic_fixture_mismatch", relative)
            finally:
                fixture_path.unlink(missing_ok=True)
    if observed_fixture_paths != expected_fixture_paths:
        raise EvidenceError("invalid_synthetic_fixture", "fixture set")
    return root_hash
