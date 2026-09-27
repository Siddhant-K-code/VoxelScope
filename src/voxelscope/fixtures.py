# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic evidence bundle construction."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .arrays import write_array
from .bundle import finalize_bundle
from .canonical import EvidenceError, canonical_json_bytes, sha256_bytes, sha256_file, write_json
from .drift import compare, component_summary, threshold_masks, validate_nested_masks
from .records import (
    MODALITIES,
    REGIONS,
    SCHEMA_VERSION,
    STAGES,
    ArrayArtifact,
    ExpectedRefusal,
    FailureState,
    ModelBundleIdentity,
    OutputIdentity,
    OverlapRatio,
    PlannedComparison,
    RunReceipt,
    StageTimingRecord,
    StudyManifest,
    ThresholdConfig,
    VolumeIdentity,
    WindowConfig,
)
from .windows import build_window_evidence

SPACING_MM = (1.0, 1.5, 2.0)


def _artifact(data: dict[str, Any], path: str | None = None) -> ArrayArtifact:
    if path is not None:
        data = {**data, "path": path}
    return ArrayArtifact.from_dict(data)


def synthetic_modalities(shape: tuple[int, int, int] = (7, 8, 9)) -> np.ndarray[Any, Any]:
    z, y, x = np.indices(shape, dtype=np.float32)
    return np.stack(
        [
            (z + 2 * y + 3 * x) / 50,
            (2 * z + y + x) / 30,
            (z * z + y + x) / 60,
            (z + y * y + x) / 80,
        ]
    ).astype("<f4")


def reference_probabilities(shape: tuple[int, int, int] = (7, 8, 9)) -> np.ndarray[Any, Any]:
    masks = np.zeros((3, *shape), dtype=bool)
    masks[1, 1:6, 2:6, 2:7] = True
    masks[0, 2:5, 2:6, 3:6] = True
    masks[2, 3:5, 4:6, 3:5] = True
    masks[2, 2, 2, 5] = True
    probabilities = np.where(masks, 0.9, 0.1).astype("<f4")
    validate_nested_masks(threshold_masks(probabilities))
    return probabilities


def scenario_probabilities() -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    reference = reference_probabilities()
    identical = reference.copy()
    boundary = reference.copy()
    boundary[1, 1, 2, 7] = 0.9
    lost = reference.copy()
    lost[2, 2, 2, 5] = 0.1
    extra = reference.copy()
    extra[1, 0, 7, 8] = 0.9
    probability_only = reference.copy()
    probability_only[0, 0, 0, 0] = 0.2
    empty_reference = reference.copy()
    empty_reference[2] = 0.1
    empty_candidate = empty_reference.copy()
    empty_candidate[1, 2, 2, 2] = 0.8
    return {
        "identical": (reference, identical),
        "one-voxel-boundary": (reference, boundary),
        "small-et-lost": (reference, lost),
        "extra-false-positive": (reference, extra),
        "probability-only": (reference, probability_only),
        "empty-surface": (empty_reference, empty_candidate),
    }


def _write_output(directory: Path, output_id: str, probabilities: np.ndarray[Any, Any]) -> Path:
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
    affine = np.eye(4, dtype="<f8")
    affine[0, 0], affine[1, 1], affine[2, 2] = SPACING_MM
    affine_data = write_array(arrays_dir / "affine.f64le", affine, "<f8")
    volume = VolumeIdentity(
        SCHEMA_VERSION,
        "synthetic-volume-v1",
        MODALITIES,
        modalities.shape[1:],
        SPACING_MM,
        _artifact(affine_data, "arrays/affine.f64le"),
        modality_records,
        None,
    )
    write_json(root / "volume-identity.json", volume)
    model_config = {
        "executor": "identity-window-oracle",
        "input_channels": 4,
        "output_channels": ["TC", "WT", "ET"],
        "schema_version": SCHEMA_VERSION,
    }
    write_json(root / "synthetic-model-config.json", model_config)
    model = ModelBundleIdentity(
        SCHEMA_VERSION,
        "VoxelScope synthetic identity oracle",
        "1",
        "synthetic_oracle",
        "generated-locally",
        "Apache-2.0",
        "synthetic-model-config.json",
        sha256_file(root / "synthetic-model-config.json"),
        None,
        None,
    )
    write_json(root / "model-identity.json", model)
    config = WindowConfig(
        volume_shape=modalities.shape[1:],
        roi=(8, 5, 6),
        overlap=(OverlapRatio(1, 2), OverlapRatio(1, 2), OverlapRatio(1, 2)),
        blend_mode="constant",
        sigma_scale=None,
    )
    evidence, weights = build_window_evidence(modalities, config)
    weight_data = write_array(root / "windows" / "weight-map.f64le", weights, "<f8")
    ledger = evidence.to_dict()
    ledger["weight_artifact"] = weight_data
    write_json(root / "windows" / "window-ledger.json", ledger)
    manifest = StudyManifest(
        SCHEMA_VERSION,
        "synthetic-feasibility-v0",
        True,
        "output_preservation",
        "unresolved",
        (),
        False,
        "volume-identity.json",
        "model-identity.json",
        config,
        (
            "empty-reference",
            "empty-surface",
            "extra-false-positive",
            "identical",
            "one-voxel-boundary",
            "probability-only",
            "reference",
            "small-et-lost",
        ),
        (
            PlannedComparison("empty-surface", "empty-reference", "empty-surface"),
            PlannedComparison("extra-false-positive", "reference", "extra-false-positive"),
            PlannedComparison("identical", "reference", "identical"),
            PlannedComparison("one-voxel-boundary", "reference", "one-voxel-boundary"),
            PlannedComparison("probability-only", "reference", "probability-only"),
            PlannedComparison("small-et-lost", "reference", "small-et-lost"),
        ),
        (
            ExpectedRefusal("invalid-nesting", "invalid_nested_regions"),
            ExpectedRefusal("invalid-padding", "unsupported_padding"),
        ),
        None,
    )
    write_json(root / "study-manifest.json", manifest)

    outputs: dict[str, Path] = {}
    reference = reference_probabilities()
    outputs["reference"] = _write_output(root / "outputs" / "reference", "reference", reference)
    pairs = scenario_probabilities()
    for name, (ref, candidate) in pairs.items():
        if name == "empty-surface":
            outputs["empty-reference"] = _write_output(
                root / "outputs" / "empty-reference", "empty-reference", ref
            )
        outputs[name] = _write_output(root / "outputs" / name, name, candidate)
    for name, (ref, candidate) in pairs.items():
        reference_path = (
            outputs["empty-reference"] if name == "empty-surface" else outputs["reference"]
        )
        report = compare(
            ref,
            candidate,
            spacing_mm=SPACING_MM,
            reference_output_sha256=sha256_file(reference_path),
            candidate_output_sha256=sha256_file(outputs[name]),
            report_id=name,
        )
        write_json(root / "reports" / f"{name}.json", report)

    invalid = reference.copy()
    invalid[2, 0, 0, 0] = 0.9
    invalid_data = write_array(root / "refusals" / "invalid-nesting.f32le", invalid, "<f4")
    nesting_failure = FailureState(
        SCHEMA_VERSION,
        "invalid-nesting",
        "refused",
        "invalid_nested_regions",
        "postprocess",
        "ET is not a subset of TC",
        (str(invalid_data["file_sha256"]),),
    )
    write_json(root / "refusals" / "invalid-nesting.json", nesting_failure)
    invalid_padding = {
        "mode": "symmetric_reflect",
        "requested_roi": [8, 5, 6],
        "schema_version": SCHEMA_VERSION,
    }
    write_json(root / "refusals" / "invalid-padding-request.json", invalid_padding)
    padding_failure = FailureState(
        SCHEMA_VERSION,
        "invalid-padding",
        "refused",
        "unsupported_padding",
        "window_enumeration",
        "Only explicit high-side zero padding is supported",
        (sha256_file(root / "refusals" / "invalid-padding-request.json"),),
    )
    write_json(root / "refusals" / "invalid-padding.json", padding_failure)

    provenance = sha256_bytes(canonical_json_bytes({"runtime": "not-executed-pr1"}))
    timings = [
        StageTimingRecord(stage, False, None, "ns", "not-measured", provenance, "not_executed_pr1")
        for stage in STAGES
    ]
    write_json(root / "stage-timings.json", {"records": timings, "schema_version": SCHEMA_VERSION})
    receipt = RunReceipt(
        SCHEMA_VERSION,
        "synthetic-fixture-v1",
        "succeeded",
        "voxelscope fixture build",
        True,
        True,
        False,
        False,
        False,
        "stage-timings.json",
        None,
    )
    write_json(root / "run-receipt.json", receipt)
    (root / "SUMMARY.txt").write_bytes(
        b"VoxelScope synthetic evidence bundle\n"
        b"Research use only. No medical data, model weights, GPU, or network were used.\n"
        b"Claim scope: output preservation only. Diagnostic accuracy is not claimed.\n"
        b"Scenarios: identical, boundary shift, ET loss, false positive, "
        b"probability-only, empty surface.\n"
        b"Root digest: see bundle.sha256.\n"
    )
    return finalize_bundle(root)


def build_fixture_bundle(output: Path) -> str:
    if output.exists():
        raise EvidenceError("output_exists", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        root_hash = _write_bundle(temporary)
        os.replace(temporary, output)
        return root_hash
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
