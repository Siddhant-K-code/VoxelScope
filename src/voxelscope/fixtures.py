# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic evidence bundle construction."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .arrays import write_array
from .atomic import path_occupied, rename_no_replace
from .bundle import finalize_bundle
from .canonical import EvidenceError, sha256_file, write_json
from .drift import compare, component_summary, threshold_masks, validate_nested_masks
from .records import (
    MODALITIES,
    REGIONS,
    SCHEMA_VERSION,
    STAGES,
    ArrayArtifact,
    ExpectedRefusal,
    FailureState,
    InvalidOutputEvidence,
    ModelBundleIdentity,
    OutputIdentity,
    PlannedComparison,
    RunReceipt,
    StageTimingRecord,
    StudyManifest,
    ThresholdConfig,
    VolumeIdentity,
)
from .synthetic_contract import (
    SYNTHETIC_ARM_ID,
    SYNTHETIC_COMPARISONS,
    SYNTHETIC_MODEL_CONFIG,
    SYNTHETIC_MODEL_CONFIG_SHA256,
    SYNTHETIC_OUTPUT_IDS,
    SYNTHETIC_REFUSALS,
    SYNTHETIC_RUN_ID,
    SYNTHETIC_SPACING_MM,
    SYNTHETIC_STUDY_ID,
    SYNTHETIC_SUMMARY_BYTES,
    SYNTHETIC_SURFACE_DICE_TOLERANCE_MM,
    SYNTHETIC_TIMING_PROVENANCE,
    SYNTHETIC_VOLUME_ID,
    reference_probabilities,
    synthetic_affine,
    synthetic_modalities,
    synthetic_outputs,
    synthetic_window_config,
)
from .windows import build_window_evidence

SPACING_MM = SYNTHETIC_SPACING_MM


def _artifact(data: dict[str, Any], path: str | None = None) -> ArrayArtifact:
    if path is not None:
        data = {**data, "path": path}
    return ArrayArtifact.from_dict(data)


def _write_output(
    directory: Path,
    output_id: str,
    probabilities: np.ndarray[Any, Any],
    *,
    volume_identity_sha256: str,
    model_identity_sha256: str,
    window_ledger_sha256: str,
) -> Path:
    validate_nested_masks(threshold_masks(probabilities))
    directory.mkdir(parents=True, exist_ok=True)
    probability = _artifact(write_array(directory / "probabilities.f32le", probabilities, "<f4"))
    masks_array = threshold_masks(probabilities)
    masks: dict[str, ArrayArtifact] = {}
    components = {}
    for index, region in enumerate(REGIONS):
        masks[region] = _artifact(
            write_array(
                directory / f"{region.lower()}.u8", masks_array[index].astype(np.uint8), "|u1"
            )
        )
        components[region] = component_summary(masks_array[index])
    identity = OutputIdentity(
        SCHEMA_VERSION,
        output_id,
        REGIONS,
        SPACING_MM,
        probability,
        masks,
        ThresholdConfig.fixed(),
        components,
        volume_identity_sha256,
        model_identity_sha256,
        window_ledger_sha256,
        SYNTHETIC_RUN_ID,
        SYNTHETIC_ARM_ID,
    )
    path = directory / "output.json"
    write_json(path, identity)
    return path


def _write_bundle(root: Path) -> str:
    arrays_dir = root / "arrays"
    modalities = synthetic_modalities()
    modality_records: dict[str, ArrayArtifact] = {}
    for index, modality in enumerate(MODALITIES):
        data = write_array(arrays_dir / f"{modality.lower()}.f32le", modalities[index], "<f4")
        modality_records[modality] = _artifact(data, f"arrays/{data['path']}")
    affine_data = write_array(arrays_dir / "affine.f64le", synthetic_affine(), "<f8")
    volume = VolumeIdentity(
        SCHEMA_VERSION,
        SYNTHETIC_VOLUME_ID,
        MODALITIES,
        modalities.shape[1:],
        SPACING_MM,
        _artifact(affine_data, "arrays/affine.f64le"),
        modality_records,
        None,
    )
    volume_path = root / "volume-identity.json"
    write_json(volume_path, volume)

    config_path = root / "synthetic-model-config.json"
    write_json(config_path, SYNTHETIC_MODEL_CONFIG)
    model = ModelBundleIdentity(
        SCHEMA_VERSION,
        "VoxelScope synthetic identity oracle",
        "1",
        "synthetic_oracle",
        "generated-locally",
        "Apache-2.0",
        config_path.name,
        SYNTHETIC_MODEL_CONFIG_SHA256,
        None,
        None,
    )
    model_path = root / "model-identity.json"
    write_json(model_path, model)

    config = synthetic_window_config()
    evidence, weights = build_window_evidence(modalities, config)
    weight_data = write_array(root / "windows" / "weight-map.f64le", weights, "<f8")
    ledger = evidence.to_dict()
    ledger["weight_artifact"] = weight_data
    ledger_path = root / "windows" / "window-ledger.json"
    write_json(ledger_path, ledger)

    manifest = StudyManifest(
        SCHEMA_VERSION,
        SYNTHETIC_STUDY_ID,
        True,
        "output_preservation",
        "unresolved",
        (),
        False,
        volume_path.name,
        model_path.name,
        config,
        SYNTHETIC_OUTPUT_IDS,
        tuple(PlannedComparison(*item) for item in SYNTHETIC_COMPARISONS),
        tuple(
            ExpectedRefusal(
                item.refusal_id,
                item.code,
                item.status,
                item.stage,
                item.message,
                item.evidence_path,
            )
            for item in SYNTHETIC_REFUSALS
        ),
        None,
    )
    write_json(root / "study-manifest.json", manifest)

    provenance = {
        "volume_identity_sha256": sha256_file(volume_path),
        "model_identity_sha256": sha256_file(model_path),
        "window_ledger_sha256": sha256_file(ledger_path),
    }
    outputs: dict[str, Path] = {}
    expected_outputs = synthetic_outputs()
    for output_id in SYNTHETIC_OUTPUT_IDS:
        outputs[output_id] = _write_output(
            root / "outputs" / output_id,
            output_id,
            expected_outputs[output_id],
            **provenance,
        )
    for report_id, reference_id, candidate_id in SYNTHETIC_COMPARISONS:
        report = compare(
            expected_outputs[reference_id],
            expected_outputs[candidate_id],
            spacing_mm=SPACING_MM,
            reference_output_sha256=sha256_file(outputs[reference_id]),
            candidate_output_sha256=sha256_file(outputs[candidate_id]),
            report_id=report_id,
            surface_dice_tolerance_mm=SYNTHETIC_SURFACE_DICE_TOLERANCE_MM,
        )
        write_json(root / "reports" / f"{report_id}.json", report)

    invalid = reference_probabilities().copy()
    invalid[2, 0, 0, 0] = 0.9
    invalid_data = write_array(root / "refusals" / "invalid-nesting.f32le", invalid, "<f4")
    invalid_evidence_path = root / "refusals" / "invalid-nesting-evidence.json"
    write_json(
        invalid_evidence_path,
        InvalidOutputEvidence(
            SCHEMA_VERSION,
            "invalid-nesting",
            _artifact(invalid_data),
            ThresholdConfig.fixed(),
        ),
    )
    nesting_contract = SYNTHETIC_REFUSALS[0]
    write_json(
        root / "refusals" / "invalid-nesting.json",
        FailureState(
            SCHEMA_VERSION,
            nesting_contract.refusal_id,
            "refused",
            nesting_contract.code,
            nesting_contract.stage,
            nesting_contract.message,
            nesting_contract.evidence_path,
            (sha256_file(invalid_evidence_path),),
        ),
    )

    invalid_padding = as_invalid_padding_request()
    invalid_padding_path = root / "refusals" / "invalid-padding-request.json"
    write_json(invalid_padding_path, invalid_padding)
    padding_contract = SYNTHETIC_REFUSALS[1]
    write_json(
        root / "refusals" / "invalid-padding.json",
        FailureState(
            SCHEMA_VERSION,
            padding_contract.refusal_id,
            "refused",
            padding_contract.code,
            padding_contract.stage,
            padding_contract.message,
            padding_contract.evidence_path,
            (sha256_file(invalid_padding_path),),
        ),
    )

    timing_provenance_path = root / "timing-provenance.json"
    write_json(timing_provenance_path, SYNTHETIC_TIMING_PROVENANCE)
    timing_digest = sha256_file(timing_provenance_path)
    timings = [
        StageTimingRecord(
            stage,
            False,
            None,
            "ns",
            "not-measured",
            timing_provenance_path.name,
            timing_digest,
            "not_executed_pr1",
        )
        for stage in STAGES
    ]
    write_json(root / "stage-timings.json", {"records": timings, "schema_version": SCHEMA_VERSION})
    write_json(
        root / "run-receipt.json",
        RunReceipt(
            SCHEMA_VERSION,
            SYNTHETIC_RUN_ID,
            "succeeded",
            "voxelscope fixture build",
            True,
            True,
            False,
            False,
            False,
            "stage-timings.json",
            None,
        ),
    )
    (root / "SUMMARY.txt").write_bytes(SYNTHETIC_SUMMARY_BYTES)

    return finalize_bundle(root)


def as_invalid_padding_request() -> dict[str, Any]:
    config = synthetic_window_config()
    return {
        "blend_mode": config.blend_mode,
        "overlap": [
            {"denominator": item.denominator, "numerator": item.numerator}
            for item in config.overlap
        ],
        "padding_mode": "symmetric_reflect",
        "roi": list(config.roi),
        "sigma_scale": None,
        "traversal_order": config.traversal_order,
        "volume_shape": list(config.volume_shape),
    }


def build_fixture_bundle(output: Path) -> str:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        root_hash = _write_bundle(temporary)
        rename_no_replace(temporary, output)
        return root_hash
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
