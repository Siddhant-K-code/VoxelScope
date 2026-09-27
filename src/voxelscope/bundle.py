# SPDX-License-Identifier: Apache-2.0
"""Closed evidence bundles and semantic verification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .arrays import read_array
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    is_link_like,
    load_json,
    safe_relative_path,
    sha256_file,
    write_json,
)
from .drift import compare, component_summary, threshold_masks, validate_nested_masks
from .records import (
    REGIONS,
    STAGES,
    ArrayArtifact,
    BundleArtifact,
    BundleIndex,
    FailureState,
    InvalidOutputEvidence,
    ModelBundleIdentity,
    OutputIdentity,
    RunReceipt,
    StageTimingRecord,
    StudyManifest,
    VolumeIdentity,
    WindowConfig,
    require_list,
    require_object,
    require_string,
    strict_fields,
)
from .synthetic_contract import (
    SYNTHETIC_ARM_ID,
    SYNTHETIC_COMPARISONS,
    SYNTHETIC_MODEL_CONFIG,
    SYNTHETIC_MODEL_CONFIG_SHA256,
    SYNTHETIC_OUTPUT_IDS,
    SYNTHETIC_REFUSALS,
    SYNTHETIC_REQUIRED_ARTIFACT_PATHS,
    SYNTHETIC_REQUIRED_DIRECTORIES,
    SYNTHETIC_RUN_ID,
    SYNTHETIC_SPACING_MM,
    SYNTHETIC_STUDY_ID,
    SYNTHETIC_SUMMARY_BYTES,
    SYNTHETIC_SURFACE_DICE_TOLERANCE_MM,
    SYNTHETIC_TIMING_PROVENANCE,
    SYNTHETIC_VOLUME_ID,
    synthetic_affine,
    synthetic_modalities,
    synthetic_outputs,
    synthetic_window_config,
)
from .windows import build_window_evidence


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def _scan_tree(root: Path) -> tuple[dict[str, Path], set[str]]:
    if is_link_like(root):
        raise EvidenceError("symlink_forbidden", str(root))
    files: dict[str, Path] = {}
    directories: set[str] = set()
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        directory, prefix = stack.pop()
        if is_link_like(directory):
            raise EvidenceError("symlink_forbidden", prefix or str(root))
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                entry_path = Path(entry.path)
                if entry.is_symlink() or is_link_like(entry_path):
                    raise EvidenceError("symlink_forbidden", relative)
                if entry.is_dir(follow_symlinks=False):
                    directories.add(relative)
                    stack.append((entry_path, relative))
                elif entry.is_file(follow_symlinks=False):
                    files[relative] = entry_path
                else:
                    raise EvidenceError("special_file_forbidden", relative)
    return files, directories


def finalize_bundle(root: Path) -> str:
    files, _ = _scan_tree(root)
    payload = [
        (relative, path)
        for relative, path in sorted(files.items())
        if relative not in {"bundle.json", "bundle.sha256"}
    ]
    artifacts = tuple(
        BundleArtifact(relative, path.stat().st_size, sha256_file(path), _media_type(path))
        for relative, path in payload
    )
    index = BundleIndex("voxelscope/v1", "voxelscope-closed-bundle-v1", artifacts)
    write_json(root / "bundle.json", index)
    root_hash = sha256_file(root / "bundle.json")
    (root / "bundle.sha256").write_bytes(f"{root_hash}  bundle.json\n".encode("ascii"))
    return root_hash


def _parse_index(data: dict[str, Any]) -> BundleIndex:
    return BundleIndex.from_dict(data)


def _require_indexed_reference(
    root: Path,
    indexed_paths: set[str],
    reference: str,
    *,
    parent: str = "",
) -> Path:
    relative = safe_relative_path(reference)
    lexical = safe_relative_path(
        f"{parent}/{relative.as_posix()}" if parent else relative.as_posix()
    )
    if lexical.as_posix() not in indexed_paths:
        raise EvidenceError("unindexed_reference", lexical.as_posix())
    return ensure_no_symlink(root, lexical)


def _require_indexed_array_references(
    root: Path,
    indexed_paths: set[str],
    parent: str,
    artifacts: list[Any],
) -> None:
    for artifact in artifacts:
        _require_indexed_reference(root, indexed_paths, artifact.path, parent=parent)


def load_output(path: Path) -> tuple[OutputIdentity, np.ndarray[Any, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError("invalid_output_identity", "output identity must be an object")
    identity = OutputIdentity.from_dict(data)
    probabilities = read_array(path.parent, identity.probabilities)
    masks = threshold_masks(probabilities, identity.threshold)
    validate_nested_masks(masks)
    for index, region in enumerate(REGIONS):
        stored = read_array(path.parent, identity.masks[region]).astype(bool)
        if not np.array_equal(stored, masks[index]):
            raise EvidenceError("threshold_mask_mismatch", f"stored {region} mask differs")
        if component_summary(stored) != identity.components[region]:
            raise EvidenceError("component_summary_mismatch", region)
    return identity, probabilities


def _load_volume(
    path: Path,
) -> tuple[VolumeIdentity, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError("invalid_volume_identity", "volume identity must be an object")
    identity = VolumeIdentity.from_dict(data)
    affine = read_array(path.parent, identity.affine)
    arrays = [
        read_array(path.parent, identity.modalities[name]) for name in identity.modality_order
    ]
    return identity, np.stack(arrays), affine


def verify_bundle(root: Path) -> str:
    if is_link_like(root) or not root.is_dir():
        raise EvidenceError("missing_bundle", str(root))
    digest_file = root / "bundle.sha256"
    index_file = root / "bundle.json"
    if is_link_like(digest_file) or is_link_like(index_file):
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
    indexed_paths = {item.path for item in index.artifacts}
    if indexed_paths != set(SYNTHETIC_REQUIRED_ARTIFACT_PATHS):
        raise EvidenceError(
            "synthetic_artifact_set_mismatch",
            f"missing={sorted(set(SYNTHETIC_REQUIRED_ARTIFACT_PATHS) - indexed_paths)}, "
            f"extra={sorted(indexed_paths - set(SYNTHETIC_REQUIRED_ARTIFACT_PATHS))}",
        )
    expected_files = {"bundle.json", "bundle.sha256", *indexed_paths}
    scanned_files, scanned_directories = _scan_tree(root)
    actual_files = set(scanned_files)
    if scanned_directories != set(SYNTHETIC_REQUIRED_DIRECTORIES):
        raise EvidenceError(
            "synthetic_directory_set_mismatch",
            f"missing={sorted(set(SYNTHETIC_REQUIRED_DIRECTORIES) - scanned_directories)}, "
            f"extra={sorted(scanned_directories - set(SYNTHETIC_REQUIRED_DIRECTORIES))}",
        )
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
    manifest_comparisons = tuple(
        (item.report_id, item.reference_output_id, item.candidate_output_id)
        for item in manifest.planned_comparisons
    )
    manifest_refusals = tuple(
        (item.refusal_id, item.code, item.status, item.stage, item.message, item.evidence_path)
        for item in manifest.expected_refusals
    )
    trusted_refusals = tuple(
        (item.refusal_id, item.code, item.status, item.stage, item.message, item.evidence_path)
        for item in SYNTHETIC_REFUSALS
    )
    if (
        manifest.study_id != SYNTHETIC_STUDY_ID
        or manifest.volume_identity_path != "volume-identity.json"
        or manifest.model_identity_path != "model-identity.json"
        or manifest.window_config != synthetic_window_config()
        or manifest.expected_output_ids != SYNTHETIC_OUTPUT_IDS
        or manifest_comparisons != SYNTHETIC_COMPARISONS
        or manifest_refusals != trusted_refusals
        or manifest.claim_scope != "output_preservation"
        or manifest.lineage_status != "unresolved"
        or manifest.lineage_evidence_sha256
        or manifest.diagnostic_accuracy_allowed
    ):
        raise EvidenceError("synthetic_manifest_scope_mismatch", "PR 1 scope was escalated")
    if (root / "SUMMARY.txt").read_bytes() != SYNTHETIC_SUMMARY_BYTES:
        raise EvidenceError("synthetic_summary_mismatch", "summary text differs")
    volume_path = _require_indexed_reference(root, indexed_paths, manifest.volume_identity_path)
    volume_data = load_json(volume_path)
    volume_identity = VolumeIdentity.from_dict(require_object(volume_data, "VolumeIdentity"))
    _require_indexed_array_references(
        root,
        indexed_paths,
        "",
        [volume_identity.affine, *volume_identity.modalities.values()],
    )
    volume, modalities, affine = _load_volume(volume_path)
    if (
        volume.volume_id != SYNTHETIC_VOLUME_ID
        or volume.spatial_shape != synthetic_window_config().volume_shape
        or volume.spacing_mm != SYNTHETIC_SPACING_MM
        or volume.label_sha256 is not None
        or not np.array_equal(modalities, synthetic_modalities())
        or not np.array_equal(affine, synthetic_affine())
    ):
        raise EvidenceError("synthetic_volume_mismatch", "volume differs from trusted fixture")

    model_path = _require_indexed_reference(root, indexed_paths, manifest.model_identity_path)
    model_data = load_json(model_path)
    if not isinstance(model_data, dict):
        raise EvidenceError("invalid_model_identity", "model identity must be an object")
    model = ModelBundleIdentity.from_dict(model_data)
    if (
        model.name != "VoxelScope synthetic identity oracle"
        or model.version != "1"
        or model.model_kind != "synthetic_oracle"
        or model.source != "generated-locally"
        or model.license != "Apache-2.0"
        or model.config_path != "synthetic-model-config.json"
        or model.weights_path is not None
        or model.weights_sha256 is not None
    ):
        raise EvidenceError("synthetic_model_identity_mismatch", "model identity differs")
    config_path = _require_indexed_reference(root, indexed_paths, model.config_path)
    if (
        not config_path.is_file()
        or model.config_sha256 != SYNTHETIC_MODEL_CONFIG_SHA256
        or model.config_sha256 != sha256_file(config_path)
    ):
        raise EvidenceError("model_config_hash_mismatch", "model config differs")
    model_config = load_json(config_path)
    if canonical_json_bytes(model_config) != canonical_json_bytes(SYNTHETIC_MODEL_CONFIG):
        raise EvidenceError("synthetic_model_config_mismatch", "model config semantics differ")
    if model.weights_path is not None:
        weights_path = _require_indexed_reference(root, indexed_paths, model.weights_path)
        if not weights_path.is_file() or model.weights_sha256 != sha256_file(weights_path):
            raise EvidenceError("model_weights_hash_mismatch", "model weights differ")

    ledger_path = _require_indexed_reference(root, indexed_paths, "windows/window-ledger.json")
    ledger_data = load_json(ledger_path)
    if not isinstance(ledger_data, dict):
        raise EvidenceError("invalid_window_ledger", "ledger must be an object")
    expected_evidence, expected_weights = build_window_evidence(modalities, manifest.window_config)
    weight_artifact = ledger_data.pop("weight_artifact", None)
    if canonical_json_bytes(ledger_data) != canonical_json_bytes(
        expected_evidence.to_dict()
    ) or not isinstance(weight_artifact, dict):
        raise EvidenceError("window_ledger_mismatch", "ledger differs from deterministic rebuild")
    weight_identity = ArrayArtifact.from_dict(weight_artifact)
    _require_indexed_array_references(root, indexed_paths, "windows", [weight_identity])
    stored_weights = read_array(root / "windows", weight_identity)
    if not np.array_equal(stored_weights, expected_weights):
        raise EvidenceError("weight_map_mismatch", "weight map differs")

    provenance_hashes = {
        "volume_identity_sha256": sha256_file(volume_path),
        "model_identity_sha256": sha256_file(model_path),
        "window_ledger_sha256": sha256_file(ledger_path),
    }
    trusted_outputs = synthetic_outputs()
    output_paths = sorted(root.glob("outputs/*/output.json"))
    outputs_by_id: dict[str, tuple[str, OutputIdentity, np.ndarray[Any, Any]]] = {}
    for output_path in output_paths:
        output_data = load_json(output_path)
        identity = OutputIdentity.from_dict(require_object(output_data, "OutputIdentity"))
        _require_indexed_array_references(
            root,
            indexed_paths,
            f"outputs/{output_path.parent.name}",
            [identity.probabilities, *identity.masks.values()],
        )
        _, probabilities = load_output(output_path)
        if output_path.parent.name != identity.output_id:
            raise EvidenceError("output_path_id_mismatch", output_path.as_posix())
        if identity.output_id in outputs_by_id:
            raise EvidenceError("duplicate_output_id", identity.output_id)
        if (
            probabilities.shape[1:] != volume.spatial_shape
            or identity.spacing_mm != volume.spacing_mm
        ):
            raise EvidenceError("output_volume_mismatch", identity.output_id)
        if (
            identity.volume_identity_sha256 != provenance_hashes["volume_identity_sha256"]
            or identity.model_identity_sha256 != provenance_hashes["model_identity_sha256"]
            or identity.window_ledger_sha256 != provenance_hashes["window_ledger_sha256"]
            or identity.run_id != SYNTHETIC_RUN_ID
            or identity.arm_id != SYNTHETIC_ARM_ID
        ):
            raise EvidenceError("output_provenance_mismatch", identity.output_id)
        if identity.output_id not in trusted_outputs or not np.array_equal(
            probabilities, trusted_outputs[identity.output_id]
        ):
            raise EvidenceError("synthetic_output_mismatch", identity.output_id)
        outputs_by_id[identity.output_id] = (sha256_file(output_path), identity, probabilities)
    actual_output_ids = set(outputs_by_id)
    expected_output_ids = set(SYNTHETIC_OUTPUT_IDS)
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
    planned_by_id = {
        report_id: (reference_id, candidate_id)
        for report_id, reference_id, candidate_id in SYNTHETIC_COMPARISONS
    }
    if set(reports_by_id) != set(planned_by_id):
        raise EvidenceError(
            "planned_report_set_mismatch",
            f"missing={sorted(set(planned_by_id) - set(reports_by_id))}, "
            f"extra={sorted(set(reports_by_id) - set(planned_by_id))}",
        )
    for report_id, (reference_output_id, candidate_output_id) in planned_by_id.items():
        report_path = reports_by_id[report_id]
        report = load_json(report_path)
        if not isinstance(report, dict):
            raise EvidenceError("invalid_drift_report", report_path.name)
        strict_fields(report, report_fields, "BoundaryDriftReport")
        if report_path.stem != report["report_id"] or report["report_id"] != report_id:
            raise EvidenceError("report_path_id_mismatch", report_path.name)
        reference_hash, reference_identity, reference_probabilities = outputs_by_id[
            reference_output_id
        ]
        candidate_hash, candidate_identity, candidate_probabilities = outputs_by_id[
            candidate_output_id
        ]
        if report["surface_dice_tolerance_mm"] != SYNTHETIC_SURFACE_DICE_TOLERANCE_MM:
            raise EvidenceError("surface_tolerance_mismatch", report_path.name)
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
            surface_dice_tolerance_mm=SYNTHETIC_SURFACE_DICE_TOLERANCE_MM,
        )
        if canonical_json_bytes(report) != canonical_json_bytes(expected_report):
            raise EvidenceError("drift_report_mismatch", report_path.name)

    trusted_refusals_by_id = {item.refusal_id: item for item in SYNTHETIC_REFUSALS}
    refusal_paths = {
        refusal_id: root / "refusals" / f"{refusal_id}.json"
        for refusal_id in trusted_refusals_by_id
    }
    for refusal_id, refusal_path in refusal_paths.items():
        refusal_data = load_json(refusal_path)
        if not isinstance(refusal_data, dict):
            raise EvidenceError("invalid_failure", refusal_path.name)
        refusal = FailureState.from_dict(refusal_data)
        expected_refusal = trusted_refusals_by_id[refusal_id]
        if (
            refusal.refusal_id != refusal_id
            or refusal.code != expected_refusal.code
            or refusal.status != expected_refusal.status
            or refusal.stage != expected_refusal.stage
            or refusal.message != expected_refusal.message
            or refusal.evidence_path != expected_refusal.evidence_path
        ):
            raise EvidenceError("refusal_contract_mismatch", refusal_path.name)
        evidence_path = _require_indexed_reference(root, indexed_paths, refusal.evidence_path)
        evidence_digest = sha256_file(evidence_path)
        if (
            not evidence_path.is_file()
            or refusal.evidence_sha256 != (evidence_digest,)
            or evidence_digest not in indexed_artifact_digests
        ):
            raise EvidenceError("unbound_refusal_evidence", refusal_path.name)
        try:
            if refusal_id == "invalid-nesting":
                evidence_data = load_json(evidence_path)
                evidence = InvalidOutputEvidence.from_dict(
                    require_object(evidence_data, "invalid nesting evidence")
                )
                _require_indexed_array_references(
                    root, indexed_paths, "refusals", [evidence.probabilities]
                )
                probabilities = read_array(evidence_path.parent, evidence.probabilities)
                masks = threshold_masks(probabilities, evidence.threshold)
                tc, wt, et = masks
                if not bool(np.any(et & ~tc)) or bool(np.any(tc & ~wt)):
                    raise EvidenceError(
                        "invalid_refusal_evidence",
                        "trusted evidence must violate only ET subset TC",
                    )
                validate_nested_masks(masks)
            elif refusal_id == "invalid-padding":
                request_data = load_json(evidence_path)
                WindowConfig.from_dict(require_object(request_data, "invalid padding request"))
            else:
                raise EvidenceError("unexpected_refusal", refusal_id)
        except EvidenceError as reproduced:
            if reproduced.code != expected_refusal.code:
                raise EvidenceError(
                    "refusal_reproduction_mismatch",
                    f"{refusal_id} produced {reproduced.code}",
                ) from reproduced
        else:
            raise EvidenceError("refusal_not_reproduced", refusal_id)

    receipt_data = load_json(root / "run-receipt.json")
    receipt = RunReceipt.from_dict(require_object(receipt_data, "RunReceipt"))
    if (
        receipt.status != "succeeded"
        or receipt.run_id != SYNTHETIC_RUN_ID
        or receipt.command != "voxelscope fixture build"
        or receipt.failure_path is not None
    ):
        raise EvidenceError("invalid_receipt", "synthetic fixture receipt differs")
    if receipt.timings_path != "stage-timings.json":
        raise EvidenceError("invalid_receipt", "timing path spelling differs")
    timing_path = _require_indexed_reference(root, indexed_paths, receipt.timings_path)
    if not timing_path.is_file():
        raise EvidenceError("invalid_receipt", "timing path does not resolve exactly")
    provenance_path = _require_indexed_reference(root, indexed_paths, "timing-provenance.json")
    provenance_data = load_json(provenance_path)
    if canonical_json_bytes(provenance_data) != canonical_json_bytes(SYNTHETIC_TIMING_PROVENANCE):
        raise EvidenceError("timing_provenance_mismatch", "timing provenance differs")
    provenance_digest = sha256_file(provenance_path)
    timings_data = load_json(timing_path)
    timings = strict_fields(timings_data, {"records", "schema_version"}, "StageTimings")
    if require_string(timings["schema_version"], "schema_version") != "voxelscope/v1":
        raise EvidenceError("unsupported_schema", "stage timings")
    timing_items = require_list(timings["records"], "timing records")
    timing_records = [
        StageTimingRecord.from_dict(require_object(item, "timing record")) for item in timing_items
    ]
    if tuple(record.stage for record in timing_records) != STAGES:
        raise EvidenceError("invalid_timing", "timing stages or order differ")
    for record in timing_records:
        if (
            record.available
            or record.value is not None
            or record.reason != "not_executed_pr1"
            or record.clock != "not-measured"
            or record.unit != "ns"
            or record.provenance_path != "timing-provenance.json"
            or record.provenance_sha256 != provenance_digest
        ):
            raise EvidenceError("invalid_timing", record.stage)
    return root_hash
