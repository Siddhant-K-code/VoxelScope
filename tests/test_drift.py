# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from voxelscope.canonical import EvidenceError
from voxelscope.drift import compare, threshold_masks, validate_nested_masks
from voxelscope.fixtures import SPACING_MM, reference_probabilities, scenario_probabilities

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def report(name: str):  # type: ignore[no-untyped-def]
    reference, candidate = scenario_probabilities()[name]
    return compare(
        reference,
        candidate,
        spacing_mm=SPACING_MM,
        reference_output_sha256=_HASH_A,
        candidate_output_sha256=_HASH_B,
        report_id=name,
    )


def test_identical_outputs_are_exact() -> None:
    result = report("identical")
    assert result.probability_bytes_equal
    assert result.probability_changed_element_count == 0
    assert result.changed_voxel_count == 0
    assert all(region["dice"] == 1.0 for region in result.regions.values())


def test_one_voxel_boundary_change_is_measured_in_mm() -> None:
    result = report("one-voxel-boundary")
    assert result.changed_voxel_count == 1
    wt = result.regions["WT"]
    assert wt["threshold_flips"]["negative_to_positive"] == 1
    assert wt["volume_delta_voxels"] == 1
    assert wt["volume_delta_mm3"] == 3.0
    assert wt["surface"]["maximum_boundary_displacement_mm"]["value"] == 2.0


def test_small_et_component_loss_is_not_hidden() -> None:
    et = report("small-et-lost").regions["ET"]
    assert et["components_removed"] == 1
    assert et["threshold_flips"]["positive_to_negative"] == 1
    assert et["volume_delta_voxels"] == -1


def test_extra_false_positive_component_is_added() -> None:
    wt = report("extra-false-positive").regions["WT"]
    assert wt["components_added"] == 1
    assert wt["volume_delta_voxels"] == 1


def test_probability_change_without_mask_change_is_visible() -> None:
    result = report("probability-only")
    assert not result.probability_bytes_equal
    assert result.probability_changed_element_count == 1
    assert result.probability_max_absolute_error == pytest.approx(0.1)
    assert result.changed_voxel_count == 0
    assert all(region["threshold_flips"]["total"] == 0 for region in result.regions.values())


def test_empty_surface_metrics_are_null_with_reason() -> None:
    et_surface = report("empty-surface").regions["ET"]["surface"]
    for metric in et_surface.values():
        assert metric == {"available": False, "reason": "both_surfaces_empty", "value": None}


def test_one_empty_surface_is_unavailable_not_zero() -> None:
    reference = reference_probabilities()
    candidate = reference.copy()
    candidate[2] = 0.1
    result = compare(
        reference,
        candidate,
        spacing_mm=SPACING_MM,
        reference_output_sha256=_HASH_A,
        candidate_output_sha256=_HASH_B,
        report_id="one-empty",
    )
    metric = result.regions["ET"]["surface"]["hd95_mm"]
    assert metric["value"] is None
    assert metric["reason"] == "candidate_surface_empty"


def test_threshold_is_inclusive_at_half() -> None:
    probabilities = np.zeros((3, 1, 1, 1), dtype=np.float32)
    probabilities[:, 0, 0, 0] = 0.5
    assert bool(threshold_masks(probabilities).all())


def test_invalid_nested_regions_are_refused() -> None:
    masks = np.zeros((3, 2, 2, 2), dtype=bool)
    masks[2, 0, 0, 0] = True
    with pytest.raises(EvidenceError) as caught:
        validate_nested_masks(masks)
    assert caught.value.code == "invalid_nested_regions"


def test_report_serializes_null_acceptance_thresholds() -> None:
    assert asdict(report("identical"))["acceptance_thresholds"] is None
