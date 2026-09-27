# SPDX-License-Identifier: Apache-2.0
"""Offline verification and rendering for the one-volume source decision."""

from __future__ import annotations

from pathlib import Path

from .canonical import EvidenceError, is_link_like, load_json, sha256_file
from .one_volume_contract import verify_one_volume_contract, verify_one_volume_plan_contract
from .one_volume_records import OneVolumeAcquisitionPlan, OneVolumeDecision


def load_one_volume_decision(path: Path) -> OneVolumeDecision:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "decision record must be a regular file")
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError("invalid_one_volume_decision", "decision record must be an object")
    return OneVolumeDecision.from_dict(data)


def verify_one_volume_decision(path: Path) -> tuple[OneVolumeDecision, str]:
    decision = load_one_volume_decision(path)
    verify_one_volume_contract(decision)
    return decision, sha256_file(path)


def load_one_volume_plan(path: Path) -> OneVolumeAcquisitionPlan:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "acquisition plan must be a regular file")
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError(
            "invalid_one_volume_acquisition_plan", "acquisition plan must be an object"
        )
    return OneVolumeAcquisitionPlan.from_dict(data)


def verify_one_volume_plan(
    path: Path, decision: OneVolumeDecision
) -> tuple[OneVolumeAcquisitionPlan, str]:
    plan = load_one_volume_plan(path)
    verify_one_volume_plan_contract(plan, decision)
    return plan, sha256_file(path)


def render_one_volume_decision(decision: OneVolumeDecision, plan: OneVolumeAcquisitionPlan) -> str:
    lines = [
        f"Decision: {decision.decision_id}",
        f"Source path status: {decision.decision.upper()}",
        f"Inference status: {decision.inference_status.upper()}",
        "Network default: disabled",
        "Medical data acquired: false",
        f"Candidates assessed: {len(decision.candidates)}",
    ]
    for candidate in decision.candidates:
        lines.append(f"- {candidate.candidate_id}: {candidate.disposition.upper()}")
    lines.append(f"Acquisition plan: {plan.execution_status.upper()}")
    lines.append(f"Next gate: {decision.next_gate}")
    return "\n".join(lines) + "\n"
