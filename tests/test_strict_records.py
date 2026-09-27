# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from voxelscope.canonical import EvidenceError
from voxelscope.records import ArrayArtifact, OverlapRatio, StudyManifest
from voxelscope.synthetic_contract import synthetic_window_config


@pytest.mark.parametrize("numerator", [1.9, True])
def test_overlap_rejects_non_integer_json_types(numerator: object) -> None:
    with pytest.raises(EvidenceError) as caught:
        OverlapRatio.from_dict({"numerator": numerator, "denominator": 2})
    assert caught.value.code == "invalid_json_type"


def test_array_shape_rejects_string_numbers() -> None:
    with pytest.raises(EvidenceError) as caught:
        ArrayArtifact.from_dict(
            {
                "content_sha256": "a" * 64,
                "dtype": "<f4",
                "file_sha256": "b" * 64,
                "path": "array.f32le",
                "shape": ["3", 4, 5],
                "size_bytes": 240,
            }
        )
    assert caught.value.code == "invalid_json_type"


def test_study_manifest_rejects_string_boolean() -> None:
    with pytest.raises(EvidenceError) as caught:
        StudyManifest.from_dict(
            {
                "claim_scope": "output_preservation",
                "diagnostic_accuracy_allowed": False,
                "drift_acceptance_thresholds": None,
                "expected_output_ids": [],
                "expected_refusals": [],
                "lineage_evidence_sha256": [],
                "lineage_status": "unresolved",
                "model_identity_path": "model.json",
                "planned_comparisons": [],
                "research_only": "false",
                "schema_version": "voxelscope/v1",
                "study_id": "strict-test",
                "volume_identity_path": "volume.json",
                "window_config": {
                    "blend_mode": "constant",
                    "overlap": [
                        {"denominator": 2, "numerator": 1},
                        {"denominator": 2, "numerator": 1},
                        {"denominator": 2, "numerator": 1},
                    ],
                    "padding_mode": "right_zero",
                    "roi": list(synthetic_window_config().roi),
                    "sigma_scale": None,
                    "traversal_order": "lexicographic_zyx",
                    "volume_shape": list(synthetic_window_config().volume_shape),
                },
            }
        )
    assert caught.value.code == "invalid_json_type"


def test_array_artifact_rejects_windows_separator() -> None:
    with pytest.raises(EvidenceError) as caught:
        ArrayArtifact.from_dict(
            {
                "content_sha256": "a" * 64,
                "dtype": "<f4",
                "file_sha256": "b" * 64,
                "path": "arrays\\escape.f32le",
                "shape": [1],
                "size_bytes": 4,
            }
        )
    assert caught.value.code == "unsafe_path"
