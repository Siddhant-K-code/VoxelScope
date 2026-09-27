# SPDX-License-Identifier: Apache-2.0
"""Trusted, immutable contract for the PR 1 synthetic study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .records import OverlapRatio, WindowConfig

SYNTHETIC_RUN_ID = "synthetic-fixture-v1"
SYNTHETIC_ARM_ID = "synthetic-oracle"
SYNTHETIC_STUDY_ID = "synthetic-feasibility-v0"
SYNTHETIC_VOLUME_ID = "synthetic-volume-v1"
SYNTHETIC_SHAPE = (7, 8, 9)
SYNTHETIC_SPACING_MM = (1.0, 1.5, 2.0)
SYNTHETIC_SURFACE_DICE_TOLERANCE_MM = 1.0
SYNTHETIC_SUMMARY_BYTES = (
    b"VoxelScope synthetic evidence bundle\n"
    b"Research use only. No medical data, model weights, GPU, or network were used.\n"
    b"Claim scope: output preservation only. Diagnostic accuracy is not claimed.\n"
    b"Scenarios: identical, boundary shift, ET loss, false positive, "
    b"probability-only, empty surface.\n"
    b"Root digest: see bundle.sha256.\n"
)
SYNTHETIC_MODEL_CONFIG_SHA256 = "3a4829cb2733bc6f0811f441231073b2589243b6ad61ce3fdf4b04a14413d97d"
SYNTHETIC_MODEL_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "executor": "identity-window-oracle",
        "input_channels": 4,
        "output_channels": ("TC", "WT", "ET"),
        "schema_version": "voxelscope/v1",
    }
)
SYNTHETIC_OUTPUT_IDS = (
    "empty-reference",
    "empty-surface",
    "extra-false-positive",
    "identical",
    "one-voxel-boundary",
    "probability-only",
    "reference",
    "small-et-lost",
)
SYNTHETIC_COMPARISONS = (
    ("empty-surface", "empty-reference", "empty-surface"),
    ("extra-false-positive", "reference", "extra-false-positive"),
    ("identical", "reference", "identical"),
    ("one-voxel-boundary", "reference", "one-voxel-boundary"),
    ("probability-only", "reference", "probability-only"),
    ("small-et-lost", "reference", "small-et-lost"),
)


@dataclass(frozen=True)
class SyntheticRefusalContract:
    refusal_id: str
    code: str
    status: Literal["refused"]
    stage: str
    message: str
    evidence_path: str


SYNTHETIC_REFUSALS = (
    SyntheticRefusalContract(
        "invalid-nesting",
        "et_not_subset_tc",
        "refused",
        "postprocess",
        "ET is not a subset of TC",
        "refusals/invalid-nesting-evidence.json",
    ),
    SyntheticRefusalContract(
        "invalid-padding",
        "unsupported_padding",
        "refused",
        "window_enumeration",
        "Only explicit high-side zero padding is supported",
        "refusals/invalid-padding-request.json",
    ),
)
SYNTHETIC_TIMING_PROVENANCE: Mapping[str, str] = MappingProxyType(
    {"schema_version": "voxelscope/v1", "runtime": "not-executed-pr1"}
)


def synthetic_window_config() -> WindowConfig:
    return WindowConfig(
        volume_shape=SYNTHETIC_SHAPE,
        roi=(8, 5, 6),
        overlap=(OverlapRatio(1, 2), OverlapRatio(1, 2), OverlapRatio(1, 2)),
        blend_mode="constant",
        sigma_scale=None,
    )


def synthetic_modalities() -> np.ndarray[Any, Any]:
    z, y, x = np.indices(SYNTHETIC_SHAPE, dtype=np.float32)
    return np.stack(
        [
            (z + 2 * y + 3 * x) / 50,
            (2 * z + y + x) / 30,
            (z * z + y + x) / 60,
            (z + y * y + x) / 80,
        ]
    ).astype("<f4")


def synthetic_affine() -> np.ndarray[Any, Any]:
    affine = np.eye(4, dtype="<f8")
    affine[0, 0], affine[1, 1], affine[2, 2] = SYNTHETIC_SPACING_MM
    return affine


def reference_probabilities() -> np.ndarray[Any, Any]:
    masks = np.zeros((3, *SYNTHETIC_SHAPE), dtype=bool)
    masks[1, 1:6, 2:6, 2:7] = True
    masks[0, 2:5, 2:6, 3:6] = True
    masks[2, 3:5, 4:6, 3:5] = True
    masks[2, 2, 2, 5] = True
    return np.where(masks, 0.9, 0.1).astype("<f4")


def synthetic_outputs() -> dict[str, np.ndarray[Any, Any]]:
    reference = reference_probabilities()
    outputs = {"reference": reference}
    outputs["identical"] = reference.copy()
    boundary = reference.copy()
    boundary[1, 1, 2, 7] = 0.9
    outputs["one-voxel-boundary"] = boundary
    lost = reference.copy()
    lost[2, 2, 2, 5] = 0.1
    outputs["small-et-lost"] = lost
    extra = reference.copy()
    extra[1, 0, 7, 8] = 0.9
    outputs["extra-false-positive"] = extra
    probability_only = reference.copy()
    probability_only[0, 0, 0, 0] = 0.2
    outputs["probability-only"] = probability_only
    empty_reference = reference.copy()
    empty_reference[2] = 0.1
    outputs["empty-reference"] = empty_reference
    empty_candidate = empty_reference.copy()
    empty_candidate[1, 2, 2, 2] = 0.8
    outputs["empty-surface"] = empty_candidate
    return outputs


def scenario_probabilities() -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    outputs = synthetic_outputs()
    return {
        report_id: (outputs[reference_id], outputs[candidate_id])
        for report_id, reference_id, candidate_id in SYNTHETIC_COMPARISONS
    }


def required_artifact_paths() -> frozenset[str]:
    paths = {
        "SUMMARY.txt",
        "arrays/affine.f64le",
        "arrays/flair.f32le",
        "arrays/t1.f32le",
        "arrays/t1c.f32le",
        "arrays/t2.f32le",
        "model-identity.json",
        "run-receipt.json",
        "timing-provenance.json",
        "stage-timings.json",
        "study-manifest.json",
        "synthetic-model-config.json",
        "volume-identity.json",
        "windows/weight-map.f64le",
        "windows/window-ledger.json",
        "refusals/invalid-nesting-evidence.json",
        "refusals/invalid-nesting.f32le",
        "refusals/invalid-nesting.json",
        "refusals/invalid-padding-request.json",
        "refusals/invalid-padding.json",
    }
    for output_id in SYNTHETIC_OUTPUT_IDS:
        paths.update(
            {
                f"outputs/{output_id}/output.json",
                f"outputs/{output_id}/probabilities.f32le",
                f"outputs/{output_id}/tc.u8",
                f"outputs/{output_id}/wt.u8",
                f"outputs/{output_id}/et.u8",
            }
        )
    paths.update(f"reports/{report_id}.json" for report_id, _, _ in SYNTHETIC_COMPARISONS)
    return frozenset(paths)


SYNTHETIC_REQUIRED_ARTIFACT_PATHS = required_artifact_paths()
SYNTHETIC_REQUIRED_DIRECTORIES = frozenset(
    parent.as_posix()
    for path in SYNTHETIC_REQUIRED_ARTIFACT_PATHS
    for parent in Path(path).parents
    if parent.as_posix() != "."
)
