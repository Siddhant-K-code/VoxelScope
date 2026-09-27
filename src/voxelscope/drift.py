# SPDX-License-Identifier: Apache-2.0
"""Boundary and probability drift auditing."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from .arrays import canonical_array
from .canonical import EvidenceError
from .records import (
    REGIONS,
    BoundaryDriftReport,
    ComponentSummary,
    MetricAvailability,
    ThresholdConfig,
)

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)
_STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)


def validate_probabilities(value: npt.ArrayLike) -> np.ndarray[Any, Any]:
    result = canonical_array(value, "<f4")
    if result.ndim != 4 or result.shape[0] != 3:
        raise EvidenceError("invalid_probability_shape", "expected (3,Z,Y,X)")
    if bool((result < 0).any()) or bool((result > 1).any()):
        raise EvidenceError("invalid_probability_range", "probabilities must be in [0,1]")
    return result


def threshold_masks(
    probabilities: npt.ArrayLike, threshold: ThresholdConfig | None = None
) -> np.ndarray[Any, Any]:
    values = validate_probabilities(probabilities)
    policy = threshold or ThresholdConfig.fixed()
    return np.stack(
        [values[index] >= policy.values[region] for index, region in enumerate(REGIONS)]
    )


def validate_nested_masks(value: npt.ArrayLike) -> np.ndarray[Any, Any]:
    masks = np.asarray(value, dtype=bool)
    if masks.ndim != 4 or masks.shape[0] != 3:
        raise EvidenceError("invalid_mask_shape", "expected (3,Z,Y,X)")
    tc, wt, et = masks
    if bool(np.any(et & ~tc)):
        raise EvidenceError("invalid_nested_regions", "ET must be a subset of TC")
    if bool(np.any(tc & ~wt)):
        raise EvidenceError("invalid_nested_regions", "TC must be a subset of WT")
    return masks


def component_summary(value: npt.ArrayLike) -> ComponentSummary:
    labels, count = ndimage.label(np.asarray(value, dtype=bool), structure=_STRUCTURE_26)
    if count:
        sizes = tuple(sorted((int(v) for v in np.bincount(labels.ravel())[1:]), reverse=True))
    else:
        sizes = ()
    return ComponentSummary(int(count), sizes)


def _component_changes(
    reference: np.ndarray[Any, Any], candidate: np.ndarray[Any, Any]
) -> tuple[int, int]:
    ref_labels, ref_count = ndimage.label(reference, structure=_STRUCTURE_26)
    cand_labels, cand_count = ndimage.label(candidate, structure=_STRUCTURE_26)
    removed = sum(
        not bool(candidate[ref_labels == label].any()) for label in range(1, int(ref_count) + 1)
    )
    added = sum(
        not bool(reference[cand_labels == label].any()) for label in range(1, int(cand_count) + 1)
    )
    return int(added), int(removed)


def _unavailable(reason: str) -> dict[str, Any]:
    return asdict(MetricAvailability(False, None, reason))


def _surface_metrics(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
    spacing: tuple[float, float, float],
    tolerance: float,
) -> dict[str, Any]:
    ref_surface = reference & ~ndimage.binary_erosion(
        reference, structure=_STRUCTURE_6, border_value=0
    )
    cand_surface = candidate & ~ndimage.binary_erosion(
        candidate, structure=_STRUCTURE_6, border_value=0
    )
    ref_empty = not bool(ref_surface.any())
    cand_empty = not bool(cand_surface.any())
    if ref_empty or cand_empty:
        if ref_empty and cand_empty:
            reason = "both_surfaces_empty"
        elif ref_empty:
            reason = "reference_surface_empty"
        else:
            reason = "candidate_surface_empty"
        unavailable = _unavailable(reason)
        return {
            "hd95_mm": unavailable,
            "maximum_boundary_displacement_mm": unavailable,
            "surface_dice": unavailable,
        }
    to_candidate = ndimage.distance_transform_edt(~cand_surface, sampling=spacing)[ref_surface]
    to_reference = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)[cand_surface]
    distances = np.concatenate((to_candidate, to_reference)).astype(np.float64, copy=False)
    within = int(np.count_nonzero(to_candidate <= tolerance)) + int(
        np.count_nonzero(to_reference <= tolerance)
    )
    denominator = int(to_candidate.size + to_reference.size)
    return {
        "hd95_mm": asdict(MetricAvailability(True, float(np.percentile(distances, 95)), None)),
        "maximum_boundary_displacement_mm": asdict(
            MetricAvailability(True, float(distances.max()), None)
        ),
        "surface_dice": asdict(MetricAvailability(True, within / denominator, None)),
    }


def compare(
    reference_probabilities: npt.ArrayLike,
    candidate_probabilities: npt.ArrayLike,
    *,
    spacing_mm: tuple[float, float, float],
    reference_output_sha256: str,
    candidate_output_sha256: str,
    report_id: str,
    threshold: ThresholdConfig | None = None,
    surface_dice_tolerance_mm: float = 1.0,
) -> BoundaryDriftReport:
    reference = validate_probabilities(reference_probabilities)
    candidate = validate_probabilities(candidate_probabilities)
    if reference.shape != candidate.shape:
        raise EvidenceError("incompatible_output_shape", "shapes differ")
    if len(spacing_mm) != 3 or any(value <= 0 for value in spacing_mm):
        raise EvidenceError("invalid_spacing", "spacing must be positive")
    if surface_dice_tolerance_mm < 0:
        raise EvidenceError("invalid_surface_tolerance", "tolerance must be nonnegative")
    policy = threshold or ThresholdConfig.fixed()
    ref_masks = validate_nested_masks(threshold_masks(reference, policy))
    cand_masks = validate_nested_masks(threshold_masks(candidate, policy))
    probability_changed = reference != candidate
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    spatial_changed = np.any(ref_masks != cand_masks, axis=0)
    regions: dict[str, dict[str, Any]] = {}
    voxel_volume = float(np.prod(spacing_mm))
    for index, region in enumerate(REGIONS):
        ref = ref_masks[index]
        cand = cand_masks[index]
        changed = ref != cand
        ref_count = int(ref.sum())
        cand_count = int(cand.sum())
        denominator = ref_count + cand_count
        dice = 1.0 if denominator == 0 else 2 * int(np.count_nonzero(ref & cand)) / denominator
        added, removed = _component_changes(ref, cand)
        delta = cand_count - ref_count
        regions[region] = {
            "candidate_components": asdict(component_summary(cand)),
            "candidate_voxel_count": cand_count,
            "changed_voxel_count": int(changed.sum()),
            "changed_voxel_rate": float(changed.mean()),
            "components_added": added,
            "components_removed": removed,
            "dice": float(dice),
            "reference_components": asdict(component_summary(ref)),
            "reference_voxel_count": ref_count,
            "relative_volume_delta": None if ref_count == 0 else float(delta / ref_count),
            "surface": _surface_metrics(ref, cand, spacing_mm, surface_dice_tolerance_mm),
            "threshold_flips": {
                "negative_to_positive": int(np.count_nonzero(~ref & cand)),
                "positive_to_negative": int(np.count_nonzero(ref & ~cand)),
                "total": int(changed.sum()),
            },
            "volume_delta_mm3": float(delta * voxel_volume),
            "volume_delta_voxels": delta,
        }
    return BoundaryDriftReport(
        "voxelscope/v1",
        report_id,
        reference_output_sha256,
        candidate_output_sha256,
        policy,
        surface_dice_tolerance_mm,
        reference.tobytes(order="C") == candidate.tobytes(order="C"),
        int(probability_changed.sum()),
        float(probability_changed.mean()),
        float(error.mean()),
        float(error.max()),
        int(spatial_changed.sum()),
        float(spatial_changed.mean()),
        regions,
        None,
    )
