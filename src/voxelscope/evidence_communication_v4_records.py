# SPDX-License-Identifier: Apache-2.0
"""Strict v4 records for bounded evidence-communication drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import EvidenceError, canonical_json_bytes, require_sha256, sha256_bytes
from .evidence_benchmark_records import BenchmarkFile, MeasurementValue, RunnerConfiguration
from .evidence_communication_records import (
    ARTIFACT_TERMINAL_STATES,
    CLAIM_TYPES,
    Coverage,
    ModelIdentity,
    PlanRequirement,
    PromptIdentity,
)
from .records import (
    require_bool,
    require_int,
    require_list,
    require_number,
    require_object,
    require_string,
    strict_fields,
)

REQUEST_SCHEMA_V4 = "voxelscope/evidence-communication-request/v4"
DRAFT_SCHEMA_V4 = "voxelscope/evidence-communication-draft-plan/v4"
ARTIFACT_SCHEMA_V4 = "voxelscope/verified-evidence-communication/v4"
RECEIPT_SCHEMA_V4 = "voxelscope/evidence-communication-receipt/v4"
TRANSFORMATION_ID_V4 = "voxelscope/evidence-communication-compiler/v4"
VERIFIER_VERSION_V4 = "voxelscope/evidence-communication-verifier/v4"
BENCHMARK_FIXTURE_SCHEMA_V4 = "voxelscope/evidence-communication-benchmark-fixture/v4"
BENCHMARK_SCHEMA_V4 = "voxelscope/evidence-communication-benchmark/v4"
BENCHMARK_INDEX_SCHEMA_V4 = "voxelscope/evidence-communication-benchmark-index/v4"
BENCHMARK_RECEIPT_SCHEMA_V4 = "voxelscope/evidence-communication-benchmark-receipt/v4"
RUNNER_MEASUREMENT_SCHEMA_V4 = "voxelscope/evidence-communication-runner-measurement/v4"
INVALID_OUTPUT_RECORD_SCHEMA_V4 = "voxelscope/evidence-communication-invalid-output/v4"


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return require_string(value, name)


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return require_number(value, name)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return require_int(value, name)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise EvidenceError("duplicate_record_value", f"{name} must be unique")


@dataclass(frozen=True)
class ClaimSkeleton:
    canonical_gene_id: str
    canonical_protein_id: str
    claim_type: str
    comparison_key: str | None
    context: str | None
    direction: str | None
    modality: str | None
    requirement_ids: tuple[str, ...]
    skeleton_id: str
    source_ids: tuple[str, ...]
    state: str
    unit: str | None
    value: float | None

    def __post_init__(self) -> None:
        if self.claim_type not in CLAIM_TYPES:
            raise EvidenceError("unsupported_claim_type", self.claim_type)
        if not self.requirement_ids or not self.source_ids:
            raise EvidenceError("incomplete_claim_skeleton", self.skeleton_id)
        _unique(self.requirement_ids, "skeleton requirement_ids")
        _unique(self.source_ids, "skeleton source_ids")
        if not self.skeleton_id.startswith("skeleton:"):
            raise EvidenceError("invalid_skeleton_id", self.skeleton_id)
        require_sha256(self.skeleton_id.removeprefix("skeleton:"), "skeleton digest")
        expected = "skeleton:" + sha256_bytes(canonical_json_bytes(self.semantic_dict()))
        if self.skeleton_id != expected:
            raise EvidenceError("skeleton_digest_mismatch", self.skeleton_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimSkeleton:
        value = strict_fields(
            data,
            {
                "canonical_gene_id",
                "canonical_protein_id",
                "claim_type",
                "comparison_key",
                "context",
                "direction",
                "modality",
                "requirement_ids",
                "skeleton_id",
                "source_ids",
                "state",
                "unit",
                "value",
            },
            "ClaimSkeleton",
        )
        return cls(
            require_string(value["canonical_gene_id"], "canonical_gene_id"),
            require_string(value["canonical_protein_id"], "canonical_protein_id"),
            require_string(value["claim_type"], "claim_type"),
            _optional_string(value["comparison_key"], "comparison_key"),
            _optional_string(value["context"], "context"),
            _optional_string(value["direction"], "direction"),
            _optional_string(value["modality"], "modality"),
            _strings(value["requirement_ids"], "requirement_ids"),
            require_string(value["skeleton_id"], "skeleton_id"),
            _strings(value["source_ids"], "source_ids"),
            require_string(value["state"], "state"),
            _optional_string(value["unit"], "unit"),
            _optional_number(value["value"], "value"),
        )

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "canonical_gene_id": self.canonical_gene_id,
            "canonical_protein_id": self.canonical_protein_id,
            "claim_type": self.claim_type,
            "comparison_key": self.comparison_key,
            "context": self.context,
            "direction": self.direction,
            "modality": self.modality,
            "requirement_ids": list(self.requirement_ids),
            "source_ids": list(self.source_ids),
            "state": self.state,
            "unit": self.unit,
            "value": self.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "skeleton_id": self.skeleton_id}


@dataclass(frozen=True)
class CommunicationRequestV4:
    atlas_id: str
    canonical_gene_id: str
    canonical_protein_id: str
    excluded_claim_types: tuple[str, ...]
    plan: tuple[PlanRequirement, ...]
    prompt_identity: PromptIdentity
    request_id: str
    required_evidence_ids: tuple[str, ...]
    schema_version: str
    skeletons: tuple[ClaimSkeleton, ...]
    source_atlas_sha256: str
    source_card_id: str
    source_card_sha256: str
    synthetic_only: bool
    terminal_state: str
    transformation_id: str
    verifier_version: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.transformation_id != TRANSFORMATION_ID_V4:
            raise EvidenceError("unsupported_transformation", self.transformation_id)
        if self.verifier_version != VERIFIER_VERSION_V4:
            raise EvidenceError("unsupported_verifier", self.verifier_version)
        if not self.synthetic_only:
            raise EvidenceError("nonsynthetic_communication_forbidden", self.request_id)
        if self.terminal_state != "planned":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.source_atlas_sha256, "source_atlas_sha256")
        require_sha256(self.source_card_sha256, "source_card_sha256")
        _unique(self.required_evidence_ids, "required_evidence_ids")
        _unique(self.excluded_claim_types, "excluded_claim_types")
        requirement_ids = tuple(item.requirement_id for item in self.plan)
        skeleton_ids = tuple(item.skeleton_id for item in self.skeletons)
        _unique(requirement_ids, "plan requirement IDs")
        _unique(skeleton_ids, "skeleton IDs")
        covered = tuple(
            requirement_id
            for skeleton in self.skeletons
            for requirement_id in skeleton.requirement_ids
        )
        if len(covered) != len(set(covered)):
            raise EvidenceError("overlapping_skeleton_coverage", self.request_id)
        if set(covered) != set(requirement_ids):
            raise EvidenceError(
                "incomplete_skeleton_coverage",
                f"missing={sorted(set(requirement_ids) - set(covered))} "
                f"unknown={sorted(set(covered) - set(requirement_ids))}",
            )
        if any(
            skeleton.canonical_gene_id != self.canonical_gene_id
            or skeleton.canonical_protein_id != self.canonical_protein_id
            for skeleton in self.skeletons
        ):
            raise EvidenceError("skeleton_entity_mismatch", self.request_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationRequestV4:
        value = strict_fields(
            data,
            {
                "atlas_id",
                "canonical_gene_id",
                "canonical_protein_id",
                "excluded_claim_types",
                "plan",
                "prompt_identity",
                "request_id",
                "required_evidence_ids",
                "schema_version",
                "skeletons",
                "source_atlas_sha256",
                "source_card_id",
                "source_card_sha256",
                "synthetic_only",
                "terminal_state",
                "transformation_id",
                "verifier_version",
                "warnings",
            },
            "CommunicationRequestV4",
        )
        return cls(
            require_string(value["atlas_id"], "atlas_id"),
            require_string(value["canonical_gene_id"], "canonical_gene_id"),
            require_string(value["canonical_protein_id"], "canonical_protein_id"),
            _strings(value["excluded_claim_types"], "excluded_claim_types"),
            tuple(
                PlanRequirement.from_dict(require_object(item, "plan requirement"))
                for item in require_list(value["plan"], "plan")
            ),
            PromptIdentity.from_dict(require_object(value["prompt_identity"], "prompt_identity")),
            require_string(value["request_id"], "request_id"),
            _strings(value["required_evidence_ids"], "required_evidence_ids"),
            require_string(value["schema_version"], "schema_version"),
            tuple(
                ClaimSkeleton.from_dict(require_object(item, "claim skeleton"))
                for item in require_list(value["skeletons"], "skeletons")
            ),
            require_string(value["source_atlas_sha256"], "source_atlas_sha256"),
            require_string(value["source_card_id"], "source_card_id"),
            require_string(value["source_card_sha256"], "source_card_sha256"),
            require_bool(value["synthetic_only"], "synthetic_only"),
            require_string(value["terminal_state"], "terminal_state"),
            require_string(value["transformation_id"], "transformation_id"),
            require_string(value["verifier_version"], "verifier_version"),
            _strings(value["warnings"], "warnings"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_id": self.atlas_id,
            "canonical_gene_id": self.canonical_gene_id,
            "canonical_protein_id": self.canonical_protein_id,
            "excluded_claim_types": list(self.excluded_claim_types),
            "plan": [item.to_dict() for item in self.plan],
            "prompt_identity": self.prompt_identity.to_dict(),
            "request_id": self.request_id,
            "required_evidence_ids": list(self.required_evidence_ids),
            "schema_version": self.schema_version,
            "skeletons": [item.to_dict() for item in self.skeletons],
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_card_id": self.source_card_id,
            "source_card_sha256": self.source_card_sha256,
            "synthetic_only": self.synthetic_only,
            "terminal_state": self.terminal_state,
            "transformation_id": self.transformation_id,
            "verifier_version": self.verifier_version,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DraftTask:
    draft_text: str
    skeleton_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DraftTask:
        value = strict_fields(data, {"draft_text", "skeleton_id"}, "DraftTask")
        return cls(
            require_string(value["draft_text"], "draft_text"),
            require_string(value["skeleton_id"], "skeleton_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"draft_text": self.draft_text, "skeleton_id": self.skeleton_id}


@dataclass(frozen=True)
class DraftPlanEnvelopeV4:
    entries: tuple[DraftTask, ...]
    model_identity: ModelIdentity
    prompt_identity: PromptIdentity
    request_id: str
    schema_version: str
    source_atlas_sha256: str
    source_card_sha256: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != DRAFT_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state != "drafted":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.source_atlas_sha256, "source_atlas_sha256")
        require_sha256(self.source_card_sha256, "source_card_sha256")
        ids = tuple(item.skeleton_id for item in self.entries)
        _unique(ids, "draft skeleton IDs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DraftPlanEnvelopeV4:
        value = strict_fields(
            data,
            {
                "entries",
                "model_identity",
                "prompt_identity",
                "request_id",
                "schema_version",
                "source_atlas_sha256",
                "source_card_sha256",
                "terminal_state",
            },
            "DraftPlanEnvelopeV4",
        )
        return cls(
            tuple(
                DraftTask.from_dict(require_object(item, "draft task"))
                for item in require_list(value["entries"], "entries")
            ),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            PromptIdentity.from_dict(require_object(value["prompt_identity"], "prompt_identity")),
            require_string(value["request_id"], "request_id"),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["source_atlas_sha256"], "source_atlas_sha256"),
            require_string(value["source_card_sha256"], "source_card_sha256"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [item.to_dict() for item in self.entries],
            "model_identity": self.model_identity.to_dict(),
            "prompt_identity": self.prompt_identity.to_dict(),
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_card_sha256": self.source_card_sha256,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class TaskOutcome:
    duplicate_count: int
    expected_count: int
    matched_count: int
    omitted_count: int
    returned_count: int
    unknown_count: int

    def __post_init__(self) -> None:
        values = (
            self.duplicate_count,
            self.expected_count,
            self.matched_count,
            self.omitted_count,
            self.returned_count,
            self.unknown_count,
        )
        if any(value < 0 for value in values):
            raise EvidenceError("invalid_task_outcome", "counts must be nonnegative")
        if self.matched_count + self.omitted_count != self.expected_count:
            raise EvidenceError("invalid_task_outcome", "expected task counts do not close")
        if self.matched_count + self.duplicate_count + self.unknown_count != self.returned_count:
            raise EvidenceError("invalid_task_outcome", "returned task counts do not close")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskOutcome:
        value = strict_fields(
            data,
            {
                "duplicate_count",
                "expected_count",
                "matched_count",
                "omitted_count",
                "returned_count",
                "unknown_count",
            },
            "TaskOutcome",
        )
        return cls(
            require_int(value["duplicate_count"], "duplicate_count"),
            require_int(value["expected_count"], "expected_count"),
            require_int(value["matched_count"], "matched_count"),
            require_int(value["omitted_count"], "omitted_count"),
            require_int(value["returned_count"], "returned_count"),
            require_int(value["unknown_count"], "unknown_count"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "duplicate_count": self.duplicate_count,
            "expected_count": self.expected_count,
            "matched_count": self.matched_count,
            "omitted_count": self.omitted_count,
            "returned_count": self.returned_count,
            "unknown_count": self.unknown_count,
        }


@dataclass(frozen=True)
class InvalidModelOutputV4:
    error_code: str
    output_sha256: str
    output_size_bytes: int
    task_outcome: TaskOutcome

    def __post_init__(self) -> None:
        require_sha256(self.output_sha256, "model output sha256")
        if self.output_size_bytes < 0:
            raise EvidenceError("invalid_model_output_size", str(self.output_size_bytes))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvalidModelOutputV4:
        value = strict_fields(
            data,
            {"error_code", "output_sha256", "output_size_bytes", "task_outcome"},
            "InvalidModelOutputV4",
        )
        return cls(
            require_string(value["error_code"], "error_code"),
            require_string(value["output_sha256"], "output_sha256"),
            require_int(value["output_size_bytes"], "output_size_bytes"),
            TaskOutcome.from_dict(require_object(value["task_outcome"], "task_outcome")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "output_sha256": self.output_sha256,
            "output_size_bytes": self.output_size_bytes,
            "task_outcome": self.task_outcome.to_dict(),
        }


@dataclass(frozen=True)
class ClaimExclusionV4:
    reason: str
    skeleton_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimExclusionV4:
        value = strict_fields(data, {"reason", "skeleton_id"}, "ClaimExclusionV4")
        return cls(
            require_string(value["reason"], "reason"),
            require_string(value["skeleton_id"], "skeleton_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "skeleton_id": self.skeleton_id}


@dataclass(frozen=True)
class VerifiedClaimV4:
    canonical_gene_id: str
    canonical_protein_id: str
    claim_type: str
    comparison_key: str | None
    context: str | None
    direction: str | None
    modality: str | None
    requirement_ids: tuple[str, ...]
    skeleton_id: str
    source_ids: tuple[str, ...]
    state: str
    unit: str | None
    value: float | None
    verified_claim_id: str

    def __post_init__(self) -> None:
        if self.claim_type not in CLAIM_TYPES:
            raise EvidenceError("unsupported_claim_type", self.claim_type)
        if not self.requirement_ids or not self.source_ids:
            raise EvidenceError("incomplete_verified_claim", self.verified_claim_id)
        _unique(self.requirement_ids, "verified requirement_ids")
        _unique(self.source_ids, "verified source_ids")
        if not self.verified_claim_id.startswith("verified-v4:"):
            raise EvidenceError("invalid_verified_claim_id", self.verified_claim_id)
        require_sha256(
            self.verified_claim_id.removeprefix("verified-v4:"),
            "verified claim digest",
        )
        expected = "verified-v4:" + sha256_bytes(canonical_json_bytes(self.semantic_dict()))
        if self.verified_claim_id != expected:
            raise EvidenceError("verified_claim_digest_mismatch", self.verified_claim_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedClaimV4:
        value = strict_fields(
            data,
            {
                "canonical_gene_id",
                "canonical_protein_id",
                "claim_type",
                "comparison_key",
                "context",
                "direction",
                "modality",
                "requirement_ids",
                "skeleton_id",
                "source_ids",
                "state",
                "unit",
                "value",
                "verified_claim_id",
            },
            "VerifiedClaimV4",
        )
        return cls(
            require_string(value["canonical_gene_id"], "canonical_gene_id"),
            require_string(value["canonical_protein_id"], "canonical_protein_id"),
            require_string(value["claim_type"], "claim_type"),
            _optional_string(value["comparison_key"], "comparison_key"),
            _optional_string(value["context"], "context"),
            _optional_string(value["direction"], "direction"),
            _optional_string(value["modality"], "modality"),
            _strings(value["requirement_ids"], "requirement_ids"),
            require_string(value["skeleton_id"], "skeleton_id"),
            _strings(value["source_ids"], "source_ids"),
            require_string(value["state"], "state"),
            _optional_string(value["unit"], "unit"),
            _optional_number(value["value"], "value"),
            require_string(value["verified_claim_id"], "verified_claim_id"),
        )

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "canonical_gene_id": self.canonical_gene_id,
            "canonical_protein_id": self.canonical_protein_id,
            "claim_type": self.claim_type,
            "comparison_key": self.comparison_key,
            "context": self.context,
            "direction": self.direction,
            "modality": self.modality,
            "requirement_ids": list(self.requirement_ids),
            "skeleton_id": self.skeleton_id,
            "source_ids": list(self.source_ids),
            "state": self.state,
            "unit": self.unit,
            "value": self.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "verified_claim_id": self.verified_claim_id}


@dataclass(frozen=True)
class VerifiedSentenceV4:
    source_ids: tuple[str, ...]
    text: str
    verified_claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_ids or not self.verified_claim_ids:
            raise EvidenceError("unmapped_verified_sentence", self.text)
        _unique(self.source_ids, "sentence source_ids")
        _unique(self.verified_claim_ids, "sentence verified_claim_ids")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedSentenceV4:
        value = strict_fields(
            data,
            {"source_ids", "text", "verified_claim_ids"},
            "VerifiedSentenceV4",
        )
        return cls(
            _strings(value["source_ids"], "source_ids"),
            require_string(value["text"], "text"),
            _strings(value["verified_claim_ids"], "verified_claim_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ids": list(self.source_ids),
            "text": self.text,
            "verified_claim_ids": list(self.verified_claim_ids),
        }


@dataclass(frozen=True)
class VerifiedCommunicationArtifactV4:
    emitted_caveat_coverage: Coverage
    emitted_fact_coverage: Coverage
    exclusions: tuple[ClaimExclusionV4, ...]
    invalid_model_output: InvalidModelOutputV4 | None
    prose: str
    refusal_reasons: tuple[str, ...]
    request_id: str
    schema_version: str
    sentences: tuple[VerifiedSentenceV4, ...]
    source_atlas_sha256: str
    source_card_sha256: str
    task_outcome: TaskOutcome
    terminal_state: str
    verified_claims: tuple[VerifiedClaimV4, ...]
    verified_draft_caveat_coverage: Coverage
    verified_draft_fact_coverage: Coverage
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state not in ARTIFACT_TERMINAL_STATES:
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.source_atlas_sha256, "source_atlas_sha256")
        require_sha256(self.source_card_sha256, "source_card_sha256")
        claims = {item.verified_claim_id: item for item in self.verified_claims}
        if len(claims) != len(self.verified_claims):
            raise EvidenceError("duplicate_verified_claim_id", self.request_id)
        if self.terminal_state == "refused" and (self.prose or self.sentences):
            raise EvidenceError("refused_artifact_has_prose", self.request_id)
        if self.terminal_state != "refused" and self.prose != " ".join(
            item.text for item in self.sentences
        ):
            raise EvidenceError("prose_sentence_mismatch", self.request_id)
        if self.invalid_model_output is not None and (
            self.terminal_state != "refused" or self.verified_claims or self.exclusions
        ):
            raise EvidenceError("invalid_model_output_artifact_mismatch", self.request_id)
        mapped: set[str] = set()
        for sentence in self.sentences:
            sentence_claims = []
            for claim_id in sentence.verified_claim_ids:
                claim = claims.get(claim_id)
                if claim is None:
                    raise EvidenceError("unknown_sentence_verified_claim", claim_id)
                sentence_claims.append(claim)
                mapped.add(claim_id)
            expected_sources = {
                source_id for claim in sentence_claims for source_id in claim.source_ids
            }
            if set(sentence.source_ids) != expected_sources:
                raise EvidenceError("sentence_source_mapping_mismatch", sentence.text)
        if self.terminal_state != "refused" and mapped != set(claims):
            raise EvidenceError("unmapped_verified_claim", self.request_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedCommunicationArtifactV4:
        value = strict_fields(
            data,
            {
                "emitted_caveat_coverage",
                "emitted_fact_coverage",
                "exclusions",
                "invalid_model_output",
                "prose",
                "refusal_reasons",
                "request_id",
                "schema_version",
                "sentences",
                "source_atlas_sha256",
                "source_card_sha256",
                "task_outcome",
                "terminal_state",
                "verified_claims",
                "verified_draft_caveat_coverage",
                "verified_draft_fact_coverage",
                "warnings",
            },
            "VerifiedCommunicationArtifactV4",
        )
        invalid = value["invalid_model_output"]
        return cls(
            Coverage.from_dict(
                require_object(value["emitted_caveat_coverage"], "emitted_caveat_coverage")
            ),
            Coverage.from_dict(
                require_object(value["emitted_fact_coverage"], "emitted_fact_coverage")
            ),
            tuple(
                ClaimExclusionV4.from_dict(require_object(item, "claim exclusion"))
                for item in require_list(value["exclusions"], "exclusions")
            ),
            (
                InvalidModelOutputV4.from_dict(require_object(invalid, "invalid_model_output"))
                if invalid is not None
                else None
            ),
            require_string(value["prose"], "prose", nonempty=False),
            _strings(value["refusal_reasons"], "refusal_reasons"),
            require_string(value["request_id"], "request_id"),
            require_string(value["schema_version"], "schema_version"),
            tuple(
                VerifiedSentenceV4.from_dict(require_object(item, "verified sentence"))
                for item in require_list(value["sentences"], "sentences")
            ),
            require_string(value["source_atlas_sha256"], "source_atlas_sha256"),
            require_string(value["source_card_sha256"], "source_card_sha256"),
            TaskOutcome.from_dict(require_object(value["task_outcome"], "task_outcome")),
            require_string(value["terminal_state"], "terminal_state"),
            tuple(
                VerifiedClaimV4.from_dict(require_object(item, "verified claim"))
                for item in require_list(value["verified_claims"], "verified_claims")
            ),
            Coverage.from_dict(
                require_object(
                    value["verified_draft_caveat_coverage"],
                    "verified_draft_caveat_coverage",
                )
            ),
            Coverage.from_dict(
                require_object(
                    value["verified_draft_fact_coverage"],
                    "verified_draft_fact_coverage",
                )
            ),
            _strings(value["warnings"], "warnings"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitted_caveat_coverage": self.emitted_caveat_coverage.to_dict(),
            "emitted_fact_coverage": self.emitted_fact_coverage.to_dict(),
            "exclusions": [item.to_dict() for item in self.exclusions],
            "invalid_model_output": (
                self.invalid_model_output.to_dict()
                if self.invalid_model_output is not None
                else None
            ),
            "prose": self.prose,
            "refusal_reasons": list(self.refusal_reasons),
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "sentences": [item.to_dict() for item in self.sentences],
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_card_sha256": self.source_card_sha256,
            "task_outcome": self.task_outcome.to_dict(),
            "terminal_state": self.terminal_state,
            "verified_claims": [item.to_dict() for item in self.verified_claims],
            "verified_draft_caveat_coverage": self.verified_draft_caveat_coverage.to_dict(),
            "verified_draft_fact_coverage": self.verified_draft_fact_coverage.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CommunicationReceiptV4:
    artifact_sha256: str
    communication_terminal_state: str
    draft_sha256: str | None
    invalid_model_output: InvalidModelOutputV4 | None
    model_identity: ModelIdentity
    prompt_identity: PromptIdentity
    request_id: str
    request_sha256: str
    schema_version: str
    skeleton_ids: tuple[str, ...]
    source_atlas_sha256: str
    source_card_sha256: str
    terminal_state: str
    transformation_id: str
    verifier_version: str

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.communication_terminal_state not in ARTIFACT_TERMINAL_STATES:
            raise EvidenceError("invalid_terminal_state", self.communication_terminal_state)
        if self.terminal_state != "closed":
            raise EvidenceError("receipt_not_closed", self.terminal_state)
        if self.transformation_id != TRANSFORMATION_ID_V4:
            raise EvidenceError("unsupported_transformation", self.transformation_id)
        if self.verifier_version != VERIFIER_VERSION_V4:
            raise EvidenceError("unsupported_verifier", self.verifier_version)
        for name, digest in (
            ("artifact_sha256", self.artifact_sha256),
            ("request_sha256", self.request_sha256),
            ("source_atlas_sha256", self.source_atlas_sha256),
            ("source_card_sha256", self.source_card_sha256),
        ):
            require_sha256(digest, name)
        if self.draft_sha256 is not None:
            require_sha256(self.draft_sha256, "draft_sha256")
        if (self.draft_sha256 is None) == (self.invalid_model_output is None):
            raise EvidenceError("receipt_model_output_identity_mismatch", self.request_id)
        _unique(self.skeleton_ids, "receipt skeleton_ids")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationReceiptV4:
        value = strict_fields(
            data,
            {
                "artifact_sha256",
                "communication_terminal_state",
                "draft_sha256",
                "invalid_model_output",
                "model_identity",
                "prompt_identity",
                "request_id",
                "request_sha256",
                "schema_version",
                "skeleton_ids",
                "source_atlas_sha256",
                "source_card_sha256",
                "terminal_state",
                "transformation_id",
                "verifier_version",
            },
            "CommunicationReceiptV4",
        )
        invalid = value["invalid_model_output"]
        return cls(
            require_string(value["artifact_sha256"], "artifact_sha256"),
            require_string(
                value["communication_terminal_state"],
                "communication_terminal_state",
            ),
            _optional_string(value["draft_sha256"], "draft_sha256"),
            (
                InvalidModelOutputV4.from_dict(require_object(invalid, "invalid_model_output"))
                if invalid is not None
                else None
            ),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            PromptIdentity.from_dict(require_object(value["prompt_identity"], "prompt_identity")),
            require_string(value["request_id"], "request_id"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["schema_version"], "schema_version"),
            _strings(value["skeleton_ids"], "skeleton_ids"),
            require_string(value["source_atlas_sha256"], "source_atlas_sha256"),
            require_string(value["source_card_sha256"], "source_card_sha256"),
            require_string(value["terminal_state"], "terminal_state"),
            require_string(value["transformation_id"], "transformation_id"),
            require_string(value["verifier_version"], "verifier_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "communication_terminal_state": self.communication_terminal_state,
            "draft_sha256": self.draft_sha256,
            "invalid_model_output": (
                self.invalid_model_output.to_dict()
                if self.invalid_model_output is not None
                else None
            ),
            "model_identity": self.model_identity.to_dict(),
            "prompt_identity": self.prompt_identity.to_dict(),
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "skeleton_ids": list(self.skeleton_ids),
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_card_sha256": self.source_card_sha256,
            "terminal_state": self.terminal_state,
            "transformation_id": self.transformation_id,
            "verifier_version": self.verifier_version,
        }


@dataclass(frozen=True)
class RunnerMeasurementV4:
    case_id: str
    input_tokens: MeasurementValue
    latency_ms: MeasurementValue
    model_identity: ModelIdentity
    output_tokens: MeasurementValue
    peak_memory_mb: MeasurementValue
    peak_metal_memory_mb: MeasurementValue
    repeat_index: int
    runner_configuration: RunnerConfiguration
    schema_version: str

    def __post_init__(self) -> None:
        if self.schema_version != RUNNER_MEASUREMENT_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.repeat_index < 0:
            raise EvidenceError("invalid_repeat_index", str(self.repeat_index))
        for name, measurement in (
            ("input_tokens", self.input_tokens),
            ("latency_ms", self.latency_ms),
            ("output_tokens", self.output_tokens),
            ("peak_memory_mb", self.peak_memory_mb),
            ("peak_metal_memory_mb", self.peak_metal_memory_mb),
        ):
            if measurement.value is not None and measurement.value < 0:
                raise EvidenceError("invalid_measurement", name)
        for name, measurement in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if measurement.value is not None and type(measurement.value) is not int:
                raise EvidenceError("invalid_measurement", f"{name} must be an integer")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunnerMeasurementV4:
        value = strict_fields(
            data,
            {
                "case_id",
                "input_tokens",
                "latency_ms",
                "model_identity",
                "output_tokens",
                "peak_memory_mb",
                "peak_metal_memory_mb",
                "repeat_index",
                "runner_configuration",
                "schema_version",
            },
            "RunnerMeasurementV4",
        )
        return cls(
            require_string(value["case_id"], "case_id"),
            MeasurementValue.from_dict(require_object(value["input_tokens"], "input_tokens")),
            MeasurementValue.from_dict(require_object(value["latency_ms"], "latency_ms")),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            MeasurementValue.from_dict(require_object(value["output_tokens"], "output_tokens")),
            MeasurementValue.from_dict(require_object(value["peak_memory_mb"], "peak_memory_mb")),
            MeasurementValue.from_dict(
                require_object(value["peak_metal_memory_mb"], "peak_metal_memory_mb")
            ),
            require_int(value["repeat_index"], "repeat_index"),
            RunnerConfiguration.from_dict(
                require_object(value["runner_configuration"], "runner_configuration")
            ),
            require_string(value["schema_version"], "schema_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_tokens": self.input_tokens.to_dict(),
            "latency_ms": self.latency_ms.to_dict(),
            "model_identity": self.model_identity.to_dict(),
            "output_tokens": self.output_tokens.to_dict(),
            "peak_memory_mb": self.peak_memory_mb.to_dict(),
            "peak_metal_memory_mb": self.peak_metal_memory_mb.to_dict(),
            "repeat_index": self.repeat_index,
            "runner_configuration": self.runner_configuration.to_dict(),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class InvalidOutputRecordV4:
    case_id: str
    invalid_model_output: InvalidModelOutputV4
    model_identity: ModelIdentity
    repeat_index: int
    request_id: str
    schema_version: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != INVALID_OUTPUT_RECORD_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.repeat_index < 0:
            raise EvidenceError("invalid_repeat_index", str(self.repeat_index))
        if self.terminal_state != "refused":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvalidOutputRecordV4:
        value = strict_fields(
            data,
            {
                "case_id",
                "invalid_model_output",
                "model_identity",
                "repeat_index",
                "request_id",
                "schema_version",
                "terminal_state",
            },
            "InvalidOutputRecordV4",
        )
        return cls(
            require_string(value["case_id"], "case_id"),
            InvalidModelOutputV4.from_dict(
                require_object(value["invalid_model_output"], "invalid_model_output")
            ),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            require_int(value["repeat_index"], "repeat_index"),
            require_string(value["request_id"], "request_id"),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "invalid_model_output": self.invalid_model_output.to_dict(),
            "model_identity": self.model_identity.to_dict(),
            "repeat_index": self.repeat_index,
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class BenchmarkRunIndexV4:
    artifact_types: tuple[str, ...]
    case_id: str
    files: tuple[BenchmarkFile, ...]
    model_identity: ModelIdentity
    repeat_index: int
    run_path: str
    semantic_sha256: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.terminal_state not in ARTIFACT_TERMINAL_STATES:
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.semantic_sha256, "semantic_sha256")
        paths = tuple(item.path for item in self.files)
        if not self.files or len(paths) != len(set(paths)):
            raise EvidenceError("benchmark_file_set_mismatch", self.run_path)
        if self.artifact_types != tuple(item.artifact_type for item in self.files):
            raise EvidenceError("benchmark_artifact_set_mismatch", self.run_path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRunIndexV4:
        value = strict_fields(
            data,
            {
                "artifact_types",
                "case_id",
                "files",
                "model_identity",
                "repeat_index",
                "run_path",
                "semantic_sha256",
                "terminal_state",
            },
            "BenchmarkRunIndexV4",
        )
        return cls(
            _strings(value["artifact_types"], "artifact_types"),
            require_string(value["case_id"], "case_id"),
            tuple(
                BenchmarkFile.from_dict(require_object(item, "benchmark file"))
                for item in require_list(value["files"], "files")
            ),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            require_int(value["repeat_index"], "repeat_index"),
            require_string(value["run_path"], "run_path"),
            require_string(value["semantic_sha256"], "semantic_sha256"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_types": list(self.artifact_types),
            "case_id": self.case_id,
            "files": [item.to_dict() for item in self.files],
            "model_identity": self.model_identity.to_dict(),
            "repeat_index": self.repeat_index,
            "run_path": self.run_path,
            "semantic_sha256": self.semantic_sha256,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class BenchmarkIndexV4:
    atlas_sha256: str
    fixture_sha256: str
    model_identity: ModelIdentity
    runner_configuration: RunnerConfiguration
    runs: tuple[BenchmarkRunIndexV4, ...]
    schema_version: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_INDEX_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state != "completed":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.atlas_sha256, "atlas_sha256")
        require_sha256(self.fixture_sha256, "fixture_sha256")
        keys = tuple((item.case_id, item.repeat_index) for item in self.runs)
        if not self.runs or len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
            raise EvidenceError("invalid_benchmark_run_index", "runs must be sorted and unique")
        if any(item.model_identity != self.model_identity for item in self.runs):
            raise EvidenceError("benchmark_model_identity_mismatch", "run identity differs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkIndexV4:
        value = strict_fields(
            data,
            {
                "atlas_sha256",
                "fixture_sha256",
                "model_identity",
                "runner_configuration",
                "runs",
                "schema_version",
                "terminal_state",
            },
            "BenchmarkIndexV4",
        )
        return cls(
            require_string(value["atlas_sha256"], "atlas_sha256"),
            require_string(value["fixture_sha256"], "fixture_sha256"),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            RunnerConfiguration.from_dict(
                require_object(value["runner_configuration"], "runner_configuration")
            ),
            tuple(
                BenchmarkRunIndexV4.from_dict(require_object(item, "benchmark run"))
                for item in require_list(value["runs"], "runs")
            ),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_sha256": self.atlas_sha256,
            "fixture_sha256": self.fixture_sha256,
            "model_identity": self.model_identity.to_dict(),
            "runner_configuration": self.runner_configuration.to_dict(),
            "runs": [item.to_dict() for item in self.runs],
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class BenchmarkReceiptV4:
    atlas_sha256: str
    benchmark_sha256: str
    closed_files: tuple[BenchmarkFile, ...]
    fixture_sha256: str
    index_sha256: str
    model_identity: ModelIdentity
    runner_configuration: RunnerConfiguration
    schema_version: str
    terminal_state: str
    transformation_id: str
    verifier_version: str

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_RECEIPT_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state != "closed":
            raise EvidenceError("receipt_not_closed", self.terminal_state)
        if self.transformation_id != TRANSFORMATION_ID_V4:
            raise EvidenceError("unsupported_transformation", self.transformation_id)
        if self.verifier_version != VERIFIER_VERSION_V4:
            raise EvidenceError("unsupported_verifier", self.verifier_version)
        for name, digest in (
            ("atlas_sha256", self.atlas_sha256),
            ("benchmark_sha256", self.benchmark_sha256),
            ("fixture_sha256", self.fixture_sha256),
            ("index_sha256", self.index_sha256),
        ):
            require_sha256(digest, name)
        paths = tuple(item.path for item in self.closed_files)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise EvidenceError("invalid_benchmark_receipt_files", "paths must be sorted")
        if "benchmark.json" not in paths or "index.json" not in paths:
            raise EvidenceError("invalid_benchmark_receipt_files", "top-level files are missing")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkReceiptV4:
        value = strict_fields(
            data,
            {
                "atlas_sha256",
                "benchmark_sha256",
                "closed_files",
                "fixture_sha256",
                "index_sha256",
                "model_identity",
                "runner_configuration",
                "schema_version",
                "terminal_state",
                "transformation_id",
                "verifier_version",
            },
            "BenchmarkReceiptV4",
        )
        return cls(
            require_string(value["atlas_sha256"], "atlas_sha256"),
            require_string(value["benchmark_sha256"], "benchmark_sha256"),
            tuple(
                BenchmarkFile.from_dict(require_object(item, "closed file"))
                for item in require_list(value["closed_files"], "closed_files")
            ),
            require_string(value["fixture_sha256"], "fixture_sha256"),
            require_string(value["index_sha256"], "index_sha256"),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            RunnerConfiguration.from_dict(
                require_object(value["runner_configuration"], "runner_configuration")
            ),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["terminal_state"], "terminal_state"),
            require_string(value["transformation_id"], "transformation_id"),
            require_string(value["verifier_version"], "verifier_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_sha256": self.atlas_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "closed_files": [item.to_dict() for item in self.closed_files],
            "fixture_sha256": self.fixture_sha256,
            "index_sha256": self.index_sha256,
            "model_identity": self.model_identity.to_dict(),
            "runner_configuration": self.runner_configuration.to_dict(),
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
            "transformation_id": self.transformation_id,
            "verifier_version": self.verifier_version,
        }
