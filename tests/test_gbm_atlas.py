# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from voxelscope.canonical import EvidenceError, load_json, sha256_file, write_json
from voxelscope.gbm_atlas import (
    assemble_gbm_evidence_atlas,
    build_gbm_evidence_atlas,
)
from voxelscope.gbm_atlas_cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research/gbm-evidence-atlas-v1"
MANIFEST = FIXTURE / "source-manifest.json"


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "fixture"
    shutil.copytree(FIXTURE, destination)
    return destination


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def _refresh_source_digest(root: Path, artifact_path: str) -> None:
    manifest_path = root / "source-manifest.json"
    manifest = _load_object(manifest_path)
    sources = manifest["sources"]
    assert isinstance(sources, list)
    for source in sources:
        assert isinstance(source, dict)
        if source["artifact_path"] == artifact_path:
            source["sha256"] = sha256_file(root / artifact_path)
            break
    else:
        raise AssertionError(f"source not found: {artifact_path}")
    write_json(manifest_path, manifest)


def _card(atlas: dict[str, Any], protein_id: str) -> dict[str, Any]:
    cards = atlas["cards"]
    assert isinstance(cards, list)
    return next(card for card in cards if card["canonical_protein_id"] == protein_id)


def test_replay_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = build_gbm_evidence_atlas(MANIFEST, first)
    second_result = build_gbm_evidence_atlas(MANIFEST, second)

    assert first_result == second_result
    assert (first / "atlas.json").read_bytes() == (second / "atlas.json").read_bytes()
    assert (first / "receipt.json").read_bytes() == (second / "receipt.json").read_bytes()


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    manifest_path = fixture / "source-manifest.json"
    manifest = _load_object(manifest_path)
    manifest["unexpected"] = True
    write_json(manifest_path, manifest)

    with pytest.raises(EvidenceError) as caught:
        build_gbm_evidence_atlas(manifest_path, tmp_path / "output")
    assert caught.value.code == "invalid_record"


def test_ambiguous_identifier_mapping_fails_closed(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    identifier_path = fixture / "identifier-map.json"
    identifier_map = _load_object(identifier_path)
    mappings = identifier_map["mappings"]
    assert isinstance(mappings, list)
    pten_mapping = mappings[1]
    assert isinstance(pten_mapping, dict)
    identifiers = pten_mapping["identifiers"]
    assert isinstance(identifiers, list)
    identifiers.append({"namespace": "hgnc-symbol", "value": "EGFR"})
    write_json(identifier_path, identifier_map)
    _refresh_source_digest(fixture, "identifier-map.json")

    with pytest.raises(EvidenceError) as caught:
        build_gbm_evidence_atlas(
            fixture / "source-manifest.json",
            tmp_path / "output",
        )
    assert caught.value.code == "ambiguous_identifier_mapping"


def test_cards_surface_missing_modalities_and_unsupported_joins() -> None:
    atlas, receipt = assemble_gbm_evidence_atlas(MANIFEST)
    pten = _card(atlas, "P60484")

    assert pten["missing_modalities"] == [
        "phosphorylation",
        "spatial_expression",
        "dependency",
        "normal_brain_expression",
    ]
    assert atlas["unsupported_joins"] == receipt["exclusions"]
    assert atlas["unsupported_joins"][0]["evidence_id"] == ("synthetic-unsupported-spatial-1")
    assert {item["status"] for item in atlas["source_freshness"]} == {
        "current",
        "stale",
    }


def test_cards_surface_agreement_and_contradictory_evidence() -> None:
    atlas, receipt = assemble_gbm_evidence_atlas(MANIFEST)

    assert _card(atlas, "P00533")["directional_assessment"]["state"] == "disagreement"
    assert _card(atlas, "P60484")["directional_assessment"]["state"] == "agreement"
    assert "contradictory_evidence" in {warning["code"] for warning in receipt["warnings"]}


def test_source_and_output_digests_change_with_evidence(tmp_path: Path) -> None:
    original_output = tmp_path / "original"
    original = build_gbm_evidence_atlas(MANIFEST, original_output)
    fixture = _copy_fixture(tmp_path)
    evidence_path = fixture / "evidence-items.json"
    evidence = _load_object(evidence_path)
    items = evidence["items"]
    assert isinstance(items, list)
    first_item = items[0]
    assert isinstance(first_item, dict)
    measurement = first_item["measurement"]
    assert isinstance(measurement, dict)
    measurement["value"] = 0.43
    write_json(evidence_path, evidence)
    _refresh_source_digest(fixture, "evidence-items.json")

    changed_output = tmp_path / "changed"
    changed = build_gbm_evidence_atlas(
        fixture / "source-manifest.json",
        changed_output,
    )
    original_receipt = load_json(original_output / "receipt.json")
    changed_receipt = load_json(changed_output / "receipt.json")

    assert original.atlas_sha256 != changed.atlas_sha256
    assert original_receipt["source_digests"] != changed_receipt["source_digests"]


def test_output_uses_no_clinical_language(tmp_path: Path) -> None:
    output = tmp_path / "atlas"
    build_gbm_evidence_atlas(MANIFEST, output)
    joined = b"\n".join(path.read_bytes().lower() for path in output.iterdir())

    for forbidden in (
        b"clinical",
        b"diagnos",
        b"drug",
        b"patient",
        b"prognos",
        b"recommend",
        b"survival",
        b"therapy",
        b"treatment",
    ):
        assert forbidden not in joined
    assert b"\xe2\x80\x94" not in joined


def test_cli_builds_synthetic_atlas(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "atlas"
    assert (
        main(
            [
                "build",
                "--manifest",
                str(MANIFEST),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    assert "gbm_atlas_status=built" in captured.out
    assert "synthetic_only=true ranking_performed=false" in captured.out
    assert captured.err == ""
