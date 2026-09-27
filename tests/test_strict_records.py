# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

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


def test_huge_json_integer_becomes_evidence_error() -> None:
    from voxelscope.records import require_number

    with pytest.raises(EvidenceError) as caught:
        require_number(10**400, "huge")
    assert caught.value.code == "invalid_json_type"


@pytest.mark.parametrize(
    "value",
    [
        "folder/name:stream",
        "folder/trailing.",
        "folder/trailing ",
        "folder/control\x00name",
        "folder/control\x1fname",
        "folder/control\x7fname",
        "CON",
        "con.txt",
        "PRN.json",
        "AUX",
        "NUL.bin",
        "COM1",
        "com9.log",
        "LPT1",
        "lpt9.txt",
    ],
)
def test_safe_relative_path_rejects_windows_ambiguous_names(value: str) -> None:
    from voxelscope.canonical import safe_relative_path

    with pytest.raises(EvidenceError) as caught:
        safe_relative_path(value)
    assert caught.value.code == "unsafe_path"


def test_link_like_helper_detects_mocked_junction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voxelscope.canonical import is_link_like

    target = tmp_path / "junction"
    target.mkdir()
    monkeypatch.setattr(type(target), "is_junction", lambda self: self == target, raising=False)
    assert is_link_like(target)


_WIN32_RESERVED_ALIASES = (
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
    *(f"COM{index}" for index in "¹²³"),
    *(f"LPT{index}" for index in "¹²³"),
)


@pytest.mark.parametrize("alias", _WIN32_RESERVED_ALIASES)
@pytest.mark.parametrize("variant", ("plain", "extension", "lowercase", "space-before-extension"))
def test_safe_relative_path_rejects_all_win32_reserved_aliases(alias: str, variant: str) -> None:
    from voxelscope.canonical import safe_relative_path

    if variant == "plain":
        value = alias
    elif variant == "extension":
        value = f"folder/{alias}.txt"
    elif variant == "lowercase":
        value = f"folder/{alias.lower()}.json"
    else:
        value = f"folder/{alias} .bin"
    with pytest.raises(EvidenceError) as caught:
        safe_relative_path(value)
    assert caught.value.code == "unsafe_path"
