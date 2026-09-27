# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from voxelscope.canonical import EvidenceError, canonical_json_bytes, load_json, write_json
from voxelscope.cli import main
from voxelscope.one_volume import (
    load_one_volume_decision,
    verify_one_volume_decision,
    verify_one_volume_plan,
)
from voxelscope.one_volume_contract import EXPECTED_CANDIDATES, SELECTED_CANDIDATE_ID
from voxelscope.one_volume_records import OneVolumeAcquisitionPlan, OneVolumeDecision

ROOT = Path(__file__).parents[1]
DECISION = ROOT / "research" / "one-volume-source-decision-v1.json"
PLAN = ROOT / "research" / "one-volume-acquisition-plan-v1.json"


def test_one_volume_decision_is_canonical_trusted_source_go() -> None:
    decision, digest = verify_one_volume_decision(DECISION)
    assert decision.decision == "go"
    assert decision.inference_status == "no-go"
    assert decision.selected_candidate_id == SELECTED_CANDIDATE_ID
    assert decision.medical_data_acquired is False
    assert {candidate.candidate_id for candidate in decision.candidates} == EXPECTED_CANDIDATES
    assert [
        candidate.candidate_id for candidate in decision.candidates if candidate.disposition == "go"
    ] == [SELECTED_CANDIDATE_ID]
    assert DECISION.read_bytes() == canonical_json_bytes(decision)
    assert digest == hashlib.sha256(DECISION.read_bytes()).hexdigest()


def test_one_volume_plan_is_canonical_blocked_and_ordered() -> None:
    decision = load_one_volume_decision(DECISION)
    plan, digest = verify_one_volume_plan(PLAN, decision)
    assert plan.execution_status == "blocked"
    assert plan.network_default == "disabled"
    assert [artifact.role for artifact in plan.artifacts[:4]] == [
        "image-t1c",
        "image-t1",
        "image-t2",
        "image-flair",
    ]
    assert all(artifact.operator_approval_required for artifact in plan.artifacts[:5])
    assert PLAN.read_bytes() == canonical_json_bytes(plan)
    assert digest == hashlib.sha256(PLAN.read_bytes()).hexdigest()


def test_one_volume_decision_records_both_search_providers() -> None:
    decision = load_one_volume_decision(DECISION)
    assert {item.provider for item in decision.search_evidence} == {"exa.ai", "parallel.ai"}
    assert all(item.independent_query_count >= 2 for item in decision.search_evidence)
    assert all(item.verified_urls for item in decision.search_evidence)


def test_one_volume_decision_go_requires_exact_artifact_set() -> None:
    data = load_json(DECISION)
    selected = next(
        item for item in data["candidates"] if item["candidate_id"] == SELECTED_CANDIDATE_ID
    )
    selected["artifacts"] = selected["artifacts"][:-1]
    with pytest.raises(EvidenceError) as caught:
        OneVolumeDecision.from_dict(data)
    assert caught.value.code == "unsafe_candidate_go"


def test_one_volume_decision_refuses_medical_data_acquisition_flag() -> None:
    data = load_json(DECISION)
    data["medical_data_acquired"] = True
    with pytest.raises(EvidenceError) as caught:
        OneVolumeDecision.from_dict(data)
    assert caught.value.code == "medical_data_acquisition_forbidden"


def test_one_volume_decision_requires_remaining_inference_blockers() -> None:
    data = load_json(DECISION)
    data["controlling_blockers"] = []
    with pytest.raises(EvidenceError) as caught:
        OneVolumeDecision.from_dict(data)
    assert caught.value.code == "incomplete_decision"


def test_one_volume_decision_rejects_missing_search_provider() -> None:
    data = load_json(DECISION)
    data["search_evidence"] = data["search_evidence"][:1]
    with pytest.raises(EvidenceError) as caught:
        OneVolumeDecision.from_dict(data)
    assert caught.value.code == "missing_search_provider"


def test_one_volume_trusted_contract_rejects_tampering(tmp_path: Path) -> None:
    data = load_json(DECISION)
    data["next_gate"] = "tampered"
    path = tmp_path / "decision.json"
    write_json(path, data)
    with pytest.raises(EvidenceError) as caught:
        verify_one_volume_decision(path)
    assert caught.value.code == "one_volume_contract_mismatch"


def test_one_volume_plan_requires_approval_for_medical_data() -> None:
    data = load_json(PLAN)
    data["artifacts"][0]["operator_approval_required"] = False
    with pytest.raises(EvidenceError) as caught:
        OneVolumeAcquisitionPlan.from_dict(data)
    assert caught.value.code == "medical_data_approval_required"


def test_one_volume_plan_rejects_wrong_channel_order() -> None:
    data = load_json(PLAN)
    data["artifacts"][0], data["artifacts"][1] = data["artifacts"][1], data["artifacts"][0]
    with pytest.raises(EvidenceError) as caught:
        OneVolumeAcquisitionPlan.from_dict(data)
    assert caught.value.code == "incomplete_acquisition_plan"


def test_one_volume_plan_rejects_medical_data_misclassification() -> None:
    data = load_json(PLAN)
    data["artifacts"][0]["medical_data"] = False
    with pytest.raises(EvidenceError) as caught:
        OneVolumeAcquisitionPlan.from_dict(data)
    assert caught.value.code == "invalid_medical_data_classification"


def test_one_volume_go_requires_sha256_for_every_artifact() -> None:
    data = load_json(DECISION)
    selected = next(
        item for item in data["candidates"] if item["candidate_id"] == SELECTED_CANDIDATE_ID
    )
    selected["artifacts"][0]["hashes"] = [
        item for item in selected["artifacts"][0]["hashes"] if item["algorithm"] != "sha256"
    ]
    with pytest.raises(EvidenceError) as caught:
        OneVolumeDecision.from_dict(data)
    assert caught.value.code == "unsafe_candidate_go"


def test_one_volume_plan_rejects_duplicate_digest_algorithm() -> None:
    data = load_json(PLAN)
    data["artifacts"][0]["expected_hashes"].append(data["artifacts"][0]["expected_hashes"][0])
    with pytest.raises(EvidenceError) as caught:
        OneVolumeAcquisitionPlan.from_dict(data)
    assert caught.value.code == "duplicate_digest"


def test_one_volume_plan_rejects_tampering(tmp_path: Path) -> None:
    decision = load_one_volume_decision(DECISION)
    data = load_json(PLAN)
    data["blocked_reason"] = "tampered"
    path = tmp_path / "plan.json"
    write_json(path, data)
    with pytest.raises(EvidenceError) as caught:
        verify_one_volume_plan(path, decision)
    assert caught.value.code == "one_volume_plan_contract_mismatch"


def test_one_volume_cli_renders_no_go(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "source",
                "one-volume",
                "--record",
                str(DECISION),
                "--plan",
                str(PLAN),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Source path status: GO" in output
    assert "Inference status: NO-GO" in output
    assert "Acquisition plan: BLOCKED" in output
    assert "Medical data acquired: false" in output
    assert f"Candidates assessed: {len(EXPECTED_CANDIDATES)}" in output


@pytest.mark.parametrize("path", [DECISION, PLAN])
def test_one_volume_contract_uses_lf_only(path: Path) -> None:
    data = path.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"\n")
