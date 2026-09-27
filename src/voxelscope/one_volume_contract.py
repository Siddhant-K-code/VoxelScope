# SPDX-License-Identifier: Apache-2.0
"""Trusted pins for the milestone 3 one-volume source decision."""

from __future__ import annotations

from .canonical import EvidenceError, canonical_json_bytes, sha256_bytes
from .one_volume_records import OneVolumeAcquisitionPlan, OneVolumeDecision

DECISION_ID = "voxelscope-one-volume-source-decision-v1"
DECISION_SHA256 = "d5e69bd1fb57be4c5f8d5b674c8eb04dcd91d4ab62627da5b566ccf9c76ff065"
PLAN_ID = "voxelscope-openneuro-one-volume-acquisition-v1"
PLAN_SHA256 = "ec14dfbccea6638606c681cac412dfa2be8c980390280798e59f4e849fff237c"
MODEL_BUNDLE_ID = "monai-brats-mri-segmentation-ngc-v0.5.2"
SELECTED_CANDIDATE_ID = "openneuro-ds007045-v2.0.1"
EXPECTED_CANDIDATES = frozenset(
    {
        "cbica-brats-2020",
        "fets-2021",
        "monai-extra-test-data-0.8.1",
        "monai-tutorials-brats",
        "msd-task01-official-s3",
        SELECTED_CANDIDATE_ID,
        "synapse-brats-2023",
        "tcia-brats-2021",
        "tcia-brats-tcga-gbm",
        "tcia-brats-tcga-lgg",
        "tcia-dicom-glioma-seg",
        "tcia-upenn-gbm",
        "zenodo-19541844",
        "zenodo-239084",
    }
)


def verify_one_volume_contract(decision: OneVolumeDecision) -> None:
    if sha256_bytes(canonical_json_bytes(decision)) != DECISION_SHA256:
        raise EvidenceError("one_volume_contract_mismatch", "canonical decision hash")
    if (
        decision.decision_id != DECISION_ID
        or decision.decision != "go"
        or decision.inference_status != "no-go"
        or decision.selected_candidate_id != SELECTED_CANDIDATE_ID
    ):
        raise EvidenceError("one_volume_contract_mismatch", "decision identity or status")
    if decision.model_contract.bundle_id != MODEL_BUNDLE_ID:
        raise EvidenceError("one_volume_contract_mismatch", "model bundle identity")
    if {candidate.candidate_id for candidate in decision.candidates} != EXPECTED_CANDIDATES:
        raise EvidenceError("one_volume_contract_mismatch", "candidate set")
    dispositions = {
        candidate.candidate_id: candidate.disposition for candidate in decision.candidates
    }
    if dispositions.get(SELECTED_CANDIDATE_ID) != "go" or any(
        disposition != "no-go"
        for candidate_id, disposition in dispositions.items()
        if candidate_id != SELECTED_CANDIDATE_ID
    ):
        raise EvidenceError("one_volume_contract_mismatch", "candidate disposition")


def verify_one_volume_plan_contract(
    plan: OneVolumeAcquisitionPlan, decision: OneVolumeDecision
) -> None:
    verify_one_volume_contract(decision)
    if sha256_bytes(canonical_json_bytes(plan)) != PLAN_SHA256:
        raise EvidenceError("one_volume_plan_contract_mismatch", "canonical plan hash")
    if (
        plan.plan_id != PLAN_ID
        or plan.decision_id != decision.decision_id
        or plan.candidate_id != SELECTED_CANDIDATE_ID
    ):
        raise EvidenceError("one_volume_plan_contract_mismatch", "plan identity")
    candidate = next(
        item for item in decision.candidates if item.candidate_id == SELECTED_CANDIDATE_ID
    )
    decision_artifacts = {
        (
            artifact.role,
            artifact.source_url,
            artifact.immutable_id,
            artifact.size_bytes,
            artifact.hashes,
        )
        for artifact in candidate.artifacts
    }
    plan_artifacts = {
        (
            artifact.role,
            artifact.source_url,
            artifact.immutable_id,
            artifact.expected_size_bytes,
            artifact.expected_hashes,
        )
        for artifact in plan.artifacts
    }
    if plan_artifacts != decision_artifacts:
        raise EvidenceError("one_volume_plan_contract_mismatch", "artifact identity set")
