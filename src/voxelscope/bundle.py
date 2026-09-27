# SPDX-License-Identifier: Apache-2.0
"""Closed evidence bundles and semantic verification."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .arrays import read_array
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    load_json,
    safe_relative_path,
    sha256_file,
    write_json,
)
from .drift import compare, component_summary, threshold_masks, validate_nested_masks
from .records import (
    REGIONS,
    STAGES,
    BundleArtifact,
    BundleIndex,
    FailureState,
    ModelBundleIdentity,
    OutputIdentity,
    RunReceipt,
    StageTimingRecord,
    StudyManifest,
    VolumeIdentity,
    strict_fields,
)
from .windows import build_window_evidence


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def finalize_bundle(root: Path) -> str:
    payload = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name not in {"bundle.json", "bundle.sha256"}
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    artifacts = tuple(
        BundleArtifact(
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            sha256_file(path),
            _media_type(path),
        )
        for path in payload
    )
    index = BundleIndex("voxelscope/v1", "voxelscope-closed-bundle-v1", artifacts)
    write_json(root / "bundle.json", index)
    root_hash = sha256_file(root / "bundle.json")
    (root / "bundle.sha256").write_text(f"{root_hash}  bundle.json\n", encoding="ascii")
    return root_hash


def _parse_index(data: dict[str, Any]) -> BundleIndex:
    if set(data) != {"artifacts", "bundle_format", "schema_version"}:
        raise EvidenceError("invalid_bundle_index", "bundle index fields differ")
    artifacts_list = []
    for item in data["artifacts"]:
        if not isinstance(item, dict):
            raise EvidenceError("invalid_bundle_artifact", "artifact must be an object")
        strict_fields(item, {"media_type", "path", "sha256", "size_bytes"}, "BundleArtifact")
        artifacts_list.append(
            BundleArtifact(
                str(item["path"]),
                int(item["size_bytes"]),
                str(item["sha256"]),
                str(item["media_type"]),
            )
        )
    artifacts = tuple(artifacts_list)
    return BundleIndex(str(data["schema_version"]), str(data["bundle_format"]), artifacts)  # type: ignore[arg-type]


def load_output(path: Path) -> tuple[OutputIdentity, np.ndarray[Any, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError("invalid_output_identity", "output identity must be an object")
    identity = OutputIdentity.from_dict(data)
    probabilities = read_array(path.parent, asdict(identity.probabilities))
    masks = threshold_masks(probabilities, identity.threshold)
    validate_nested_masks(masks)
    for index, region in enumerate(REGIONS):
        stored = read_array(path.parent, asdict(identity.masks[region])).astype(bool)
        if not np.array_equal(stored, masks[index]):
            raise EvidenceError("threshold_mask_mismatch", f"stored {region} mask differs")
        if component_summary(stored) != identity.components[region]:
            raise EvidenceError("component_summary_mismatch", region)
    return identity, probabilities


def _load_volume(path: Path) -> tuple[VolumeIdentity, np.ndarray[Any, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError("invalid_volume_identity", "volume identity must be an object")
    identity = VolumeIdentity.from_dict(data)
    read_array(path.parent, asdict(identity.affine))
    arrays = [
        read_array(path.parent, asdict(identity.modalities[name]))
        for name in identity.modality_order
    ]
    return identity, np.stack(arrays)


def verify_bundle(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError("missing_bundle", str(root))
    digest_file = root / "bundle.sha256"
    index_file = root / "bundle.json"
    if digest_file.is_symlink() or index_file.is_symlink():
        raise EvidenceError("symlink_forbidden", "bundle root files cannot be symlinks")
    if not digest_file.is_file() or not index_file.is_file():
        raise EvidenceError("missing_bundle_index", str(root))
    root_hash = sha256_file(index_file)
    expected_digest = f"{root_hash}  bundle.json\n".encode("ascii")
    if digest_file.read_bytes() != expected_digest:
        raise EvidenceError(
            "invalid_bundle_digest", "bundle.sha256 is not canonical or does not match"
        )

    index_data = load_json(index_file)
    if not isinstance(index_data, dict):
        raise EvidenceError("invalid_bundle_index", "bundle index must be an object")
    index = _parse_index(index_data)
    expected_files = {"bundle.json", "bundle.sha256", *(item.path for item in index.artifacts)}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise EvidenceError(
            "bundle_file_set_mismatch",
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}",
        )
    for item in index.artifacts:
        relative = safe_relative_path(item.path)
        path = ensure_no_symlink(root, relative)
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise EvidenceError("bundle_artifact_size_mismatch", item.path)
        if sha256_file(path) != item.sha256:
            raise EvidenceError("bundle_artifact_hash_mismatch", item.path)
        if item.media_type != _media_type(path):
            raise EvidenceError("bundle_artifact_media_type_mismatch", item.path)
        if item.media_type == "application/json":
            load_json(path)
    indexed_artifact_digests = {item.sha256 for item in index.artifacts}

    manifest_data = load_json(root / "study-manifest.json")
    if not isinstance(manifest_data, dict):
        raise EvidenceError("invalid_study_manifest", "manifest must be an object")
    manifest = StudyManifest.from_dict(manifest_data)
    if (
        manifest.claim_scope != "output_preservation"
        or manifest.lineage_status != "unresolved"
        or manifest.lineage_evidence_sha256
        or manifest.diagnostic_accuracy_allowed
    ):
        raise EvidenceError("synthetic_manifest_scope_mismatch", "PR 1 scope was escalated")
    volume_path = ensure_no_symlink(root, safe_relative_path(manifest.volume_identity_path))
    volume, modalities = _load_volume(volume_path)
    if volume.spatial_shape != manifest.window_config.volume_shape:
        raise EvidenceError("manifest_volume_mismatch", "volume shape differs")

    model_path = ensure_no_symlink(root, safe_relative_path(manifest.model_identity_path))
    model_data = load_json(model_path)
    if not isinstance(model_data, dict):
        raise EvidenceError("invalid_model_identity", "model identity must be an object")
    model = ModelBundleIdentity.from_dict(model_data)
    config_path = ensure_no_symlink(root, safe_relative_path(model.config_path))
    if not config_path.is_file() or model.config_sha256 != sha256_file(config_path):
        raise EvidenceError("model_config_hash_mismatch", "model config differs")
    if model.weights_path is not None:
        weights_path = ensure_no_symlink(root, safe_relative_path(model.weights_path))
        if not weights_path.is_file() or model.weights_sha256 != sha256_file(weights_path):
            raise EvidenceError("model_weights_hash_mismatch", "model weights differ")

    ledger_data = load_json(root / "windows" / "window-ledger.json")
    if not isinstance(ledger_data, dict):
        raise EvidenceError("invalid_window_ledger", "ledger must be an object")
    expected_evidence, expected_weights = build_window_evidence(modalities, manifest.window_config)
    weight_artifact = ledger_data.pop("weight_artifact", None)
    if canonical_json_bytes(ledger_data) != canonical_json_bytes(
        expected_evidence.to_dict()
    ) or not isinstance(weight_artifact, dict):
        raise EvidenceError("window_ledger_mismatch", "ledger differs from deterministic rebuild")
    stored_weights = read_array(root / "windows", weight_artifact)
    if not np.array_equal(stored_weights, expected_weights):
        raise EvidenceError("weight_map_mismatch", "weight map differs")

    output_paths = sorted(root.glob("outputs/*/output.json"))
    outputs_by_id: dict[str, tuple[str, OutputIdentity, np.ndarray[Any, Any]]] = {}
    for output_path in output_paths:
        identity, probabilities = load_output(output_path)
        if output_path.parent.name != identity.output_id:
            raise EvidenceError("output_path_id_mismatch", output_path.as_posix())
        if identity.output_id in outputs_by_id:
            raise EvidenceError("duplicate_output_id", identity.output_id)
        outputs_by_id[identity.output_id] = (sha256_file(output_path), identity, probabilities)
    actual_output_ids = set(outputs_by_id)
    expected_output_ids = set(manifest.expected_output_ids)
    if actual_output_ids != expected_output_ids:
        raise EvidenceError(
            "planned_output_set_mismatch",
            f"missing={sorted(expected_output_ids - actual_output_ids)}, "
            f"extra={sorted(actual_output_ids - expected_output_ids)}",
        )

    report_fields = {
        "acceptance_thresholds",
        "candidate_output_sha256",
        "changed_voxel_count",
        "changed_voxel_rate",
        "probability_bytes_equal",
        "probability_changed_element_count",
        "probability_changed_element_rate",
        "probability_max_absolute_error",
        "probability_mean_absolute_error",
        "reference_output_sha256",
        "regions",
        "report_id",
        "schema_version",
        "surface_dice_tolerance_mm",
        "threshold",
    }
    reports_by_id = {path.stem: path for path in (root / "reports").glob("*.json")}
    planned_by_id = {item.report_id: item for item in manifest.planned_comparisons}
    if set(reports_by_id) != set(planned_by_id):
        raise EvidenceError(
            "planned_report_set_mismatch",
            f"missing={sorted(set(planned_by_id) - set(reports_by_id))}, "
            f"extra={sorted(set(reports_by_id) - set(planned_by_id))}",
        )
    for report_id, plan in planned_by_id.items():
        report_path = reports_by_id[report_id]
        report = load_json(report_path)
        if not isinstance(report, dict):
            raise EvidenceError("invalid_drift_report", report_path.name)
        strict_fields(report, report_fields, "BoundaryDriftReport")
        if report_path.stem != report["report_id"] or report["report_id"] != plan.report_id:
            raise EvidenceError("report_path_id_mismatch", report_path.name)
        reference_hash, reference_identity, reference_probabilities = outputs_by_id[
            plan.reference_output_id
        ]
        candidate_hash, candidate_identity, candidate_probabilities = outputs_by_id[
            plan.candidate_output_id
        ]
        if (
            report["reference_output_sha256"] != reference_hash
            or report["candidate_output_sha256"] != candidate_hash
        ):
            raise EvidenceError("unbound_drift_report", report_path.name)
        if (
            reference_identity.spacing_mm != candidate_identity.spacing_mm
            or reference_identity.threshold != candidate_identity.threshold
        ):
            raise EvidenceError("incompatible_drift_outputs", report_path.name)
        expected_report = compare(
            reference_probabilities,
            candidate_probabilities,
            spacing_mm=reference_identity.spacing_mm,
            reference_output_sha256=reference_hash,
            candidate_output_sha256=candidate_hash,
            report_id=report_id,
            threshold=reference_identity.threshold,
            surface_dice_tolerance_mm=float(report["surface_dice_tolerance_mm"]),
        )
        if canonical_json_bytes(report) != canonical_json_bytes(expected_report):
            raise EvidenceError("drift_report_mismatch", report_path.name)

    refusal_paths = {
        path.stem: path
        for path in (root / "refusals").glob("*.json")
        if not path.name.endswith("-request.json")
    }
    expected_refusals = {item.refusal_id: item.code for item in manifest.expected_refusals}
    if set(refusal_paths) != set(expected_refusals):
        raise EvidenceError(
            "planned_refusal_set_mismatch",
            f"missing={sorted(set(expected_refusals) - set(refusal_paths))}, "
            f"extra={sorted(set(refusal_paths) - set(expected_refusals))}",
        )
    for refusal_id, refusal_path in refusal_paths.items():
        refusal_data = load_json(refusal_path)
        if not isinstance(refusal_data, dict):
            raise EvidenceError("invalid_failure", refusal_path.name)
        strict_fields(
            refusal_data,
            {
                "code",
                "evidence_sha256",
                "message",
                "refusal_id",
                "schema_version",
                "stage",
                "status",
            },
            "FailureState",
        )
        refusal = FailureState(**refusal_data)
        if refusal.refusal_id != refusal_id or refusal.code != expected_refusals[refusal_id]:
            raise EvidenceError("refusal_path_id_mismatch", refusal_path.name)
        if (
            not refusal.evidence_sha256
            or not set(refusal.evidence_sha256) <= indexed_artifact_digests
        ):
            raise EvidenceError("unbound_refusal_evidence", refusal_path.name)

    receipt_data = load_json(root / "run-receipt.json")
    if not isinstance(receipt_data, dict):
        raise EvidenceError("invalid_receipt", "receipt must be an object")
    strict_fields(
        receipt_data,
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
    receipt = RunReceipt(**receipt_data)
    if (
        receipt.status != "succeeded"
        or receipt.run_id != "synthetic-fixture-v1"
        or receipt.command != "voxelscope fixture build"
        or receipt.failure_path is not None
    ):
        raise EvidenceError("invalid_receipt", "synthetic fixture receipt differs")
    timing_path = ensure_no_symlink(root, safe_relative_path(receipt.timings_path))
    if not timing_path.is_file():
        raise EvidenceError("invalid_receipt", "timing path does not resolve")
    timings_data = load_json(timing_path)
    if not isinstance(timings_data, dict):
        raise EvidenceError("invalid_timing", "timings must be an object")
    strict_fields(timings_data, {"records", "schema_version"}, "StageTimings")
    if timings_data["schema_version"] != "voxelscope/v1":
        raise EvidenceError("unsupported_schema", "stage timings")
    timing_records = []
    for item in timings_data["records"]:
        strict_fields(
            item,
            {"available", "clock", "provenance_sha256", "reason", "stage", "unit", "value"},
            "StageTimingRecord",
        )
        timing_records.append(StageTimingRecord(**item))
    if tuple(record.stage for record in timing_records) != STAGES:
        raise EvidenceError("invalid_timing", "timing stages or order differ")
    if receipt.failure_path is not None:
        failure_path = ensure_no_symlink(root, safe_relative_path(receipt.failure_path))
        if not failure_path.is_file() or failure_path not in set(refusal_paths.values()):
            raise EvidenceError("invalid_receipt", "failure path is not a verified refusal")
    return root_hash
