# SPDX-License-Identifier: Apache-2.0
"""Sanitized closed public evidence for milestone 5."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

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
from .preprocessing import verify_preprocessing, verify_preprocessing_refusal
from .preprocessing_records import (
    MILESTONE4_BUNDLE_SHA256,
    MODEL_INFERENCE_CONFIG_SHA256,
    MODEL_ZOO_COMMIT,
    MONAI_COMMIT,
    PreprocessingRefusal,
)
from .records import (
    require_bool,
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

PUBLIC_EVIDENCE_SCHEMA = "voxelscope/milestone-5-public-evidence/v1"
PUBLIC_BUNDLE_FORMAT = "voxelscope-closed-public-bundle-v1"
EXPECTED_PUBLIC_BUNDLE_SHA256S = frozenset(
    {
        "183ff1b55aa1fea501acfaf71d4bc90432741fe1f40e1932576a7b04f96c9e02",
        "1acb7df00ad5489ef9001a8ccfaa5482ba1bfdeb6681831cec301cdb373adaff",
        "3fe384ecadd4933d9d858df69fdc8b915fd438adc5907acb44943a12a9dc5532",
        "6d0285a87299c5cc51b5e3ace51f3a4aba6b4b64613cc0bc9d569b051426de60",
        "71c5551d84bf4d783d5ef5cb0ef3a80da2779666944e6c8f73929fa0eca266b5",
        "8cb05ed962478516817d0c14e1f1ff67ee5557805a57749b1a14cf05b4492442",
        "95afe5721c9bd1981c504e902c854f02c0aa2022d4a585aa3b82935cea391a14",
        "e1ee2658d3a73bbd8d539e34b73127607a3b3c2df2fb9ba0ee10a7833c835975",
        "fc39350bfcb70f5caf4deae9f4789bc5f4c2f9ff41b344e96c84cc9a5a462c7e",
    }
)
PAYLOAD_PATHS = (
    "README.txt",
    "claim-matrix.json",
    "privacy-report.json",
    "public-summary-v1.json",
    "source-manifest.json",
)
INDEXED_PATHS = (*PAYLOAD_PATHS, "SHA256SUMS")
README_BYTES = (
    b"VoxelScope milestone 5 public evidence\n"
    b"\n"
    b"This closed bundle reports only sanitized preprocessing gate states and\n"
    b"public implementation identities. It contains no medical values, private\n"
    b"paths, tensors, geometry, or private evidence identities. Inference remains\n"
    b"unauthorized.\n"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "affine",
        "background_zero_preserved",
        "custody_receipt_sha256",
        "custody_structural_report_sha256",
        "dtype",
        "geometry_identity_sha256",
        "input_sha256",
        "input_spatial_shape",
        "mean",
        "nonzero_voxel_count",
        "orientation",
        "output_shape",
        "private_path",
        "receipt",
        "receipt_sha256",
        "shape",
        "spacing",
        "standard_deviation",
        "tensor",
        "tensor_sha256",
        "zooms",
    }
)
_FORBIDDEN_BYTES = (
    b"sub-",
    b"artifacts/openneuro",
    b"/Users/",
    b"/home/",
    b"input-reference.f32le",
    b"input-independent.f32le",
    b"preprocessing-report",
)


_RefusalRule = tuple[str, int, int, frozenset[bool]]
_ATTEMPT_RULE: _RefusalRule = ("attempt_record", 0, 0, frozenset({False}))
_CUSTODY_RULE: _RefusalRule = ("custody_verification", 0, 0, frozenset({False}))
_CHANNEL_RULE: _RefusalRule = ("channel_loading", 0, 3, frozenset({False}))
_REFERENCE_RULE: _RefusalRule = ("reference_normalization", 0, 3, frozenset({False}))
_INDEPENDENT_RULE: _RefusalRule = ("independent_normalization", 0, 3, frozenset({False}))
_INDEPENDENT_CLEANUP_RULE: _RefusalRule = (
    "independent_normalization",
    0,
    4,
    frozenset({False}),
)
_COMPARISON_RULE: _RefusalRule = (
    "implementation_comparison",
    4,
    4,
    frozenset({False}),
)
_PUBLICATION_BEFORE_RENAME_RULE: _RefusalRule = (
    "snapshot_publication",
    4,
    4,
    frozenset({False}),
)
_PUBLICATION_INTERRUPTION_RULE: _RefusalRule = (
    "snapshot_publication",
    4,
    4,
    frozenset({False, True}),
)
_PUBLISHABLE_REFUSAL_RULES: dict[str, tuple[_RefusalRule, ...]] = {
    "acquisition_refused": (_ATTEMPT_RULE,),
    "atomic_publish_unsupported": (_PUBLICATION_BEFORE_RENAME_RULE,),
    "attempt_record_mismatch": (_ATTEMPT_RULE,),
    "background_changed": (_REFERENCE_RULE, _INDEPENDENT_RULE),
    "custody_artifact_mismatch": (_ATTEMPT_RULE, _CHANNEL_RULE),
    "custody_identity_drift": (_CUSTODY_RULE,),
    "custody_plan_mismatch": (_CUSTODY_RULE,),
    "custody_receipt_mismatch": (_ATTEMPT_RULE,),
    "custody_snapshot_file_set_mismatch": (_ATTEMPT_RULE,),
    "effective_dtype_overflow": (_REFERENCE_RULE, _INDEPENDENT_RULE),
    "empty_channel": (_REFERENCE_RULE, _INDEPENDENT_RULE),
    "implementation_disagreement": (_COMPARISON_RULE,),
    "implementation_shape_mismatch": (_COMPARISON_RULE,),
    "incomplete_channel_set": (_CUSTODY_RULE,),
    "invalid_channel_shape": (_INDEPENDENT_RULE,),
    "invalid_implementation_comparison": (_COMPARISON_RULE,),
    "invalid_nifti": (_ATTEMPT_RULE, _CHANNEL_RULE),
    "missing_acquisition_completion": (_ATTEMPT_RULE,),
    "missing_custody_snapshot": (_ATTEMPT_RULE,),
    "nifti_voxel_read_failed": (_CHANNEL_RULE,),
    "nonfinite_input": (_REFERENCE_RULE, _INDEPENDENT_RULE),
    "nonfinite_output": (_REFERENCE_RULE, _INDEPENDENT_RULE, _COMPARISON_RULE),
    "nonfinite_statistics": (_REFERENCE_RULE, _INDEPENDENT_RULE),
    "normalization_overflow": (_REFERENCE_RULE, _INDEPENDENT_RULE),
    "output_exists": (_PUBLICATION_BEFORE_RENAME_RULE,),
    "preprocessing_attempt_mismatch": (_ATTEMPT_RULE,),
    "preprocessing_interrupted": (
        _ATTEMPT_RULE,
        _CUSTODY_RULE,
        _CHANNEL_RULE,
        _REFERENCE_RULE,
        _INDEPENDENT_CLEANUP_RULE,
        _COMPARISON_RULE,
        _PUBLICATION_INTERRUPTION_RULE,
    ),
    "private_directory_permissions": (_ATTEMPT_RULE,),
    "private_file_permissions": (_ATTEMPT_RULE,),
    "source_geometry_drift": (_CHANNEL_RULE,),
    "structural_report_hash_mismatch": (_ATTEMPT_RULE,),
    "structural_report_mismatch": (_ATTEMPT_RULE,),
    "symlink_forbidden": (_ATTEMPT_RULE, _CUSTODY_RULE, _CHANNEL_RULE),
    "unsafe_path": (_ATTEMPT_RULE, _CUSTODY_RULE, _CHANNEL_RULE),
    "zero_variance_channel": (_REFERENCE_RULE, _INDEPENDENT_RULE),
}
_REFUSAL_PHASE_BY_STAGE = {
    "attempt_record": "custody",
    "custody_verification": "custody",
    "channel_loading": "channel-processing",
    "reference_normalization": "channel-processing",
    "independent_normalization": "channel-processing",
    "implementation_comparison": "comparison",
    "snapshot_publication": "publication",
}


def _go_summary() -> dict[str, Any]:
    return {
        "attempt_terminal_status": "completed",
        "completed_channel_count": 4,
        "finite_output_status": "go",
        "geometry_preserved_status": "go",
        "independent_implementation_count": 2,
        "inference_authorized": False,
        "inference_run": False,
        "input_channel_count": 4,
        "label_excluded_status": "go",
        "model_loaded": False,
        "overall_status": "go",
        "preprocessing_adapter_status": "go",
        "private_output_status": "go",
        "refusal_phase": "none",
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "terminal_evidence_status": "go",
    }


def _refusal_summary(refusal_phase: str, completed_channel_count: int) -> dict[str, Any]:
    if refusal_phase == "custody":
        if completed_channel_count != 0:
            raise EvidenceError("invalid_public_progress", refusal_phase)
        adapter, reached, private = "no-go", "not-reached", "not-reached"
    elif refusal_phase == "channel-processing":
        if not 0 <= completed_channel_count <= 4:
            raise EvidenceError("invalid_public_progress", refusal_phase)
        adapter, reached, private = "no-go", "not-reached", "not-reached"
    elif refusal_phase == "comparison":
        if completed_channel_count != 4:
            raise EvidenceError("invalid_public_progress", refusal_phase)
        adapter, reached, private = "no-go", "go", "not-reached"
    elif refusal_phase == "publication":
        if completed_channel_count != 4:
            raise EvidenceError("invalid_public_progress", refusal_phase)
        adapter, reached, private = "go", "go", "no-go"
    else:
        raise EvidenceError("invalid_public_refusal_phase", refusal_phase)
    return {
        "attempt_terminal_status": "refused",
        "completed_channel_count": completed_channel_count,
        "finite_output_status": reached,
        "geometry_preserved_status": reached,
        "independent_implementation_count": 2,
        "inference_authorized": False,
        "inference_run": False,
        "input_channel_count": 4,
        "label_excluded_status": reached,
        "model_loaded": False,
        "overall_status": "no-go",
        "preprocessing_adapter_status": adapter,
        "private_output_status": private,
        "refusal_phase": refusal_phase,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "terminal_evidence_status": "go",
    }


def _validated_public_refusal(refusal: PreprocessingRefusal) -> tuple[str, int]:
    rules = _PUBLISHABLE_REFUSAL_RULES.get(refusal.error_code)
    if rules is None:
        raise EvidenceError("unpublishable_refusal", refusal.error_code)
    if not any(
        refusal.last_completed_stage == stage
        and minimum <= refusal.completed_channel_count <= maximum
        and refusal.snapshot_published in snapshot_states
        for stage, minimum, maximum, snapshot_states in rules
    ):
        raise EvidenceError("refusal_progress_mismatch", refusal.error_code)
    phase = _REFUSAL_PHASE_BY_STAGE[refusal.last_completed_stage]
    _refusal_summary(phase, refusal.completed_channel_count)
    return phase, refusal.completed_channel_count


def _source_manifest(
    adapter_plan_sha256: str,
    implementation_sources: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    return {
        "adapter_plan_sha256": adapter_plan_sha256,
        "implementation_sources": [
            {"repository_path": path, "sha256": sha256} for path, sha256 in implementation_sources
        ],
        "milestone4_bundle_sha256": MILESTONE4_BUNDLE_SHA256,
        "model_inference_config_sha256": MODEL_INFERENCE_CONFIG_SHA256,
        "model_zoo_commit": MODEL_ZOO_COMMIT,
        "monai_commit": MONAI_COMMIT,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _claim_matrix() -> dict[str, Any]:
    return {
        "claims": [
            {"claim": "preprocessing_adapter_execution", "scope": "mechanical-only"},
            {"claim": "two_implementation_agreement", "scope": "prospective-tolerance"},
            {"claim": "geometry_preservation", "scope": "no-spatial-transform"},
            {"claim": "label_exclusion", "scope": "input-tensor-only"},
            {"claim": "private_output_publication", "scope": "sanitized-status-only"},
            {"claim": "terminal_evidence", "scope": "sanitized-status-only"},
        ],
        "excluded_claims": [
            "accuracy",
            "clinical",
            "diagnostic",
            "inference",
            "patient-specific",
            "performance",
        ],
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _privacy_report() -> dict[str, Any]:
    return {
        "medical_values_included": False,
        "private_evidence_identities_included": False,
        "private_paths_included": False,
        "real_geometry_included": False,
        "real_statistics_included": False,
        "source_subject_locator_included": False,
        "tensor_bytes_or_hashes_included": False,
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
    }


def _write_bytes(root: Path, relative: str, value: bytes) -> None:
    path = root / safe_relative_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _build_public_bundle(
    output: Path,
    *,
    outcome: Literal["go", "no-go"],
    adapter_plan_sha256: str,
    implementation_sources: tuple[tuple[str, str], ...],
    refusal_phase: str | None = None,
    completed_channel_count: int | None = None,
) -> str:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        if outcome == "go":
            if refusal_phase is not None or completed_channel_count is not None:
                raise EvidenceError("invalid_public_outcome", outcome)
            summary = _go_summary()
        else:
            if refusal_phase is None or completed_channel_count is None:
                raise EvidenceError("invalid_public_outcome", outcome)
            summary = _refusal_summary(refusal_phase, completed_channel_count)
        _write_bytes(stage, "README.txt", README_BYTES)
        records = {
            "claim-matrix.json": _claim_matrix(),
            "privacy-report.json": _privacy_report(),
            "public-summary-v1.json": summary,
            "source-manifest.json": _source_manifest(
                adapter_plan_sha256,
                implementation_sources,
            ),
        }
        for relative, value in records.items():
            write_json(stage / relative, value)
        sums = "".join(
            f"{sha256_file(stage / relative)}  {relative}\n" for relative in sorted(PAYLOAD_PATHS)
        )
        _write_bytes(stage, "SHA256SUMS", sums.encode("ascii"))
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
        _write_bytes(
            stage,
            "bundle.sha256",
            f"{root_hash}  bundle.json\n".encode("ascii"),
        )
        verify_milestone5_public_bundle(stage, require_trusted_digest=False)
        rename_no_replace(stage, output)
        return root_hash
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def build_milestone5_public_bundle(
    decision_path: Path,
    custody_plan_path: Path,
    adapter_plan_path: Path,
    root: Path,
    output: Path,
    *,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    repository_root: Path,
) -> str:
    report = verify_preprocessing(
        decision_path,
        custody_plan_path,
        adapter_plan_path,
        root,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        repository_root=repository_root,
    )
    return _build_public_bundle(
        output,
        outcome="go",
        adapter_plan_sha256=report.plan_sha256,
        implementation_sources=tuple(
            (item.path, item.sha256) for item in report.implementation_sources
        ),
    )


def build_milestone5_refusal_public_bundle(
    adapter_plan_path: Path,
    root: Path,
    output: Path,
    *,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    repository_root: Path,
) -> str:
    refusal = verify_preprocessing_refusal(
        adapter_plan_path,
        root,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        repository_root=repository_root,
    )
    from .preprocessing import load_adapter_plan

    plan, plan_sha256 = load_adapter_plan(
        adapter_plan_path,
        repository_root=repository_root,
    )
    if refusal.plan_sha256 != plan_sha256:
        raise EvidenceError("preprocessing_refusal_mismatch", "plan identity differs")
    refusal_phase, completed_channel_count = _validated_public_refusal(refusal)
    return _build_public_bundle(
        output,
        outcome="no-go",
        adapter_plan_sha256=plan_sha256,
        implementation_sources=tuple(
            (item.path, item.sha256) for item in plan.implementation_sources
        ),
        refusal_phase=refusal_phase,
        completed_channel_count=completed_channel_count,
    )


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
    size = path.stat().st_size
    if size > maximum_bytes:
        raise EvidenceError("public_file_limit", path.name)
    return path.read_bytes()


def verify_milestone5_public_bundle(root: Path, *, require_trusted_digest: bool = True) -> str:
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
    if require_trusted_digest:
        if root_hash not in EXPECTED_PUBLIC_BUNDLE_SHA256S:
            raise EvidenceError("public_bundle_contract_mismatch", "trusted digest differs")
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
    summary = strict_fields(
        require_object(records["public-summary-v1.json"], "summary"),
        set(_go_summary()),
        "summary",
    )
    outcome = require_string(summary["overall_status"], "overall_status")
    if outcome == "go":
        expected_summary = _go_summary()
    elif outcome == "no-go":
        expected_summary = _refusal_summary(
            require_string(summary["refusal_phase"], "refusal_phase"),
            require_int(summary["completed_channel_count"], "completed_channel_count"),
        )
    else:
        raise EvidenceError("invalid_public_summary", "outcome differs")
    if summary != expected_summary:
        raise EvidenceError("invalid_public_summary", "summary differs")
    if (
        require_int(summary["input_channel_count"], "input_channel_count") != 4
        or require_int(
            summary["independent_implementation_count"],
            "independent_implementation_count",
        )
        != 2
        or require_bool(summary["inference_authorized"], "inference_authorized")
        or require_bool(summary["model_loaded"], "model_loaded")
        or require_bool(summary["inference_run"], "inference_run")
    ):
        raise EvidenceError("invalid_public_summary", "unsafe summary")
    source = require_object(records["source-manifest.json"], "source manifest")
    if (
        source.get("schema_version") != PUBLIC_EVIDENCE_SCHEMA
        or source.get("milestone4_bundle_sha256") != MILESTONE4_BUNDLE_SHA256
        or source.get("model_inference_config_sha256") != MODEL_INFERENCE_CONFIG_SHA256
        or source.get("model_zoo_commit") != MODEL_ZOO_COMMIT
        or source.get("monai_commit") != MONAI_COMMIT
    ):
        raise EvidenceError("public_source_manifest_mismatch", "source manifest differs")
    return root_hash
