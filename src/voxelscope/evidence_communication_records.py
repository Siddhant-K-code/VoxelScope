# SPDX-License-Identifier: Apache-2.0
"""Strict records for the synthetic evidence communication compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import EvidenceError, require_sha256
from .records import (
    require_bool,
    require_int,
    require_list,
    require_number,
    require_object,
    require_string,
    strict_fields,
)

REQUEST_SCHEMA = "voxelscope/evidence-communication-request/v1"
DRAFT_SCHEMA = "voxelscope/evidence-communication-model-draft/v1"
ARTIFACT_SCHEMA = "voxelscope/verified-evidence-communication/v1"
RECEIPT_SCHEMA = "voxelscope/evidence-communication-receipt/v1"
TRANSFORMATION_ID = "voxelscope/evidence-communication-compiler/v1"
VERIFIER_VERSION = "voxelscope/evidence-communication-verifier/v1"

CLAIM_TYPES = frozenset(
    {
        "availability",
        "directional_assessment",
        "entity_identity",
        "evidence_observation",
        "non_clinical_boundary",
        "source_target_evidence",
    }
)
REQUIREMENT_KINDS = frozenset({"fact", "caveat"})
ARTIFACT_TERMINAL_STATES = frozenset({"accepted", "accepted_with_exclusions", "refused"})


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return require_string(value, name)


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return require_number(value, name)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise EvidenceError("duplicate_record_value", f"{name} must be unique")


@dataclass(frozen=True)
class PromptIdentity:
    prompt_id: str
    sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.sha256, "prompt sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptIdentity:
        value = strict_fields(data, {"prompt_id", "sha256"}, "PromptIdentity")
        return cls(
            require_string(value["prompt_id"], "prompt_id"),
            require_string(value["sha256"], "prompt sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"prompt_id": self.prompt_id, "sha256": self.sha256}


@dataclass(frozen=True)
class ModelIdentity:
    adapter: str
    endpoint: str | None
    model: str
    runtime: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelIdentity:
        value = strict_fields(
            data,
            {"adapter", "endpoint", "model", "runtime"},
            "ModelIdentity",
        )
        return cls(
            require_string(value["adapter"], "adapter"),
            _optional_string(value["endpoint"], "endpoint"),
            require_string(value["model"], "model"),
            require_string(value["runtime"], "runtime"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "endpoint": self.endpoint,
            "model": self.model,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class PlanRequirement:
    claim_type: str
    comparison_key: str | None
    context: str | None
    description: str
    direction: str | None
    kind: str
    modality: str | None
    requirement_id: str
    source_ids: tuple[str, ...]
    state: str
    unit: str | None
    value: float | None

    def __post_init__(self) -> None:
        if self.claim_type not in CLAIM_TYPES:
            raise EvidenceError("unsupported_claim_type", self.claim_type)
        if self.kind not in REQUIREMENT_KINDS:
            raise EvidenceError("invalid_requirement_kind", self.kind)
        if not self.source_ids:
            raise EvidenceError("missing_source_citation", self.requirement_id)
        _require_unique(self.source_ids, "requirement source_ids")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanRequirement:
        value = strict_fields(
            data,
            {
                "claim_type",
                "comparison_key",
                "context",
                "description",
                "direction",
                "kind",
                "modality",
                "requirement_id",
                "source_ids",
                "state",
                "unit",
                "value",
            },
            "PlanRequirement",
        )
        return cls(
            require_string(value["claim_type"], "claim_type"),
            _optional_string(value["comparison_key"], "comparison_key"),
            _optional_string(value["context"], "context"),
            require_string(value["description"], "description"),
            _optional_string(value["direction"], "direction"),
            require_string(value["kind"], "kind"),
            _optional_string(value["modality"], "modality"),
            require_string(value["requirement_id"], "requirement_id"),
            _strings(value["source_ids"], "source_ids"),
            require_string(value["state"], "state"),
            _optional_string(value["unit"], "unit"),
            _optional_number(value["value"], "value"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "comparison_key": self.comparison_key,
            "context": self.context,
            "description": self.description,
            "direction": self.direction,
            "kind": self.kind,
            "modality": self.modality,
            "requirement_id": self.requirement_id,
            "source_ids": list(self.source_ids),
            "state": self.state,
            "unit": self.unit,
            "value": self.value,
        }


@dataclass(frozen=True)
class CommunicationRequest:
    atlas_id: str
    canonical_gene_id: str
    canonical_protein_id: str
    excluded_claim_types: tuple[str, ...]
    plan: tuple[PlanRequirement, ...]
    prompt_identity: PromptIdentity
    request_id: str
    required_evidence_ids: tuple[str, ...]
    schema_version: str
    source_atlas_sha256: str
    source_card_id: str
    source_card_sha256: str
    synthetic_only: bool
    terminal_state: str
    transformation_id: str
    verifier_version: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.transformation_id != TRANSFORMATION_ID:
            raise EvidenceError("unsupported_transformation", self.transformation_id)
        if self.verifier_version != VERIFIER_VERSION:
            raise EvidenceError("unsupported_verifier", self.verifier_version)
        if not self.synthetic_only:
            raise EvidenceError("nonsynthetic_communication_forbidden", self.request_id)
        if self.terminal_state != "planned":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.source_atlas_sha256, "source_atlas_sha256")
        require_sha256(self.source_card_sha256, "source_card_sha256")
        _require_unique(self.required_evidence_ids, "required_evidence_ids")
        _require_unique(self.excluded_claim_types, "excluded_claim_types")
        requirement_ids = tuple(item.requirement_id for item in self.plan)
        _require_unique(requirement_ids, "plan requirement IDs")
        if not any(item.kind == "fact" for item in self.plan):
            raise EvidenceError("invalid_communication_plan", "facts are required")
        if not any(item.kind == "caveat" for item in self.plan):
            raise EvidenceError("invalid_communication_plan", "caveats are required")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationRequest:
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
                "source_atlas_sha256",
                "source_card_id",
                "source_card_sha256",
                "synthetic_only",
                "terminal_state",
                "transformation_id",
                "verifier_version",
                "warnings",
            },
            "CommunicationRequest",
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
class ProposedClaim:
    canonical_gene_id: str
    canonical_protein_id: str
    claim_id: str
    claim_type: str
    comparison_key: str | None
    context: str | None
    direction: str | None
    draft_text: str
    modality: str | None
    requirement_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    state: str | None
    unit: str | None
    value: float | None

    def __post_init__(self) -> None:
        if self.claim_type not in CLAIM_TYPES:
            raise EvidenceError("unsupported_claim_type", self.claim_type)
        if not self.source_ids:
            raise EvidenceError("missing_source_citation", self.claim_id)
        _require_unique(self.source_ids, "claim source_ids")
        _require_unique(self.requirement_ids, "claim requirement_ids")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProposedClaim:
        value = strict_fields(
            data,
            {
                "canonical_gene_id",
                "canonical_protein_id",
                "claim_id",
                "claim_type",
                "comparison_key",
                "context",
                "direction",
                "draft_text",
                "modality",
                "requirement_ids",
                "source_ids",
                "state",
                "unit",
                "value",
            },
            "ProposedClaim",
        )
        return cls(
            require_string(value["canonical_gene_id"], "canonical_gene_id"),
            require_string(value["canonical_protein_id"], "canonical_protein_id"),
            require_string(value["claim_id"], "claim_id"),
            require_string(value["claim_type"], "claim_type"),
            _optional_string(value["comparison_key"], "comparison_key"),
            _optional_string(value["context"], "context"),
            _optional_string(value["direction"], "direction"),
            require_string(value["draft_text"], "draft_text"),
            _optional_string(value["modality"], "modality"),
            _strings(value["requirement_ids"], "requirement_ids"),
            _strings(value["source_ids"], "source_ids"),
            _optional_string(value["state"], "state"),
            _optional_string(value["unit"], "unit"),
            _optional_number(value["value"], "value"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_gene_id": self.canonical_gene_id,
            "canonical_protein_id": self.canonical_protein_id,
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "comparison_key": self.comparison_key,
            "context": self.context,
            "direction": self.direction,
            "draft_text": self.draft_text,
            "modality": self.modality,
            "requirement_ids": list(self.requirement_ids),
            "source_ids": list(self.source_ids),
            "state": self.state,
            "unit": self.unit,
            "value": self.value,
        }


@dataclass(frozen=True)
class ModelDraftEnvelope:
    claims: tuple[ProposedClaim, ...]
    model_identity: ModelIdentity
    prompt_identity: PromptIdentity
    request_id: str
    schema_version: str
    source_atlas_sha256: str
    source_card_sha256: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != DRAFT_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state != "drafted":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.source_atlas_sha256, "source_atlas_sha256")
        require_sha256(self.source_card_sha256, "source_card_sha256")
        claim_ids = tuple(item.claim_id for item in self.claims)
        _require_unique(claim_ids, "claim IDs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelDraftEnvelope:
        value = strict_fields(
            data,
            {
                "claims",
                "model_identity",
                "prompt_identity",
                "request_id",
                "schema_version",
                "source_atlas_sha256",
                "source_card_sha256",
                "terminal_state",
            },
            "ModelDraftEnvelope",
        )
        return cls(
            tuple(
                ProposedClaim.from_dict(require_object(item, "proposed claim"))
                for item in require_list(value["claims"], "claims")
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
            "claims": [item.to_dict() for item in self.claims],
            "model_identity": self.model_identity.to_dict(),
            "prompt_identity": self.prompt_identity.to_dict(),
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_card_sha256": self.source_card_sha256,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class ClaimExclusion:
    claim_id: str
    draft_text: str
    reason: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimExclusion:
        value = strict_fields(data, {"claim_id", "draft_text", "reason"}, "ClaimExclusion")
        return cls(
            require_string(value["claim_id"], "claim_id"),
            require_string(value["draft_text"], "draft_text"),
            require_string(value["reason"], "reason"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "claim_id": self.claim_id,
            "draft_text": self.draft_text,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VerifiedSentence:
    claim_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    text: str

    def __post_init__(self) -> None:
        if not self.claim_ids or not self.source_ids:
            raise EvidenceError("unmapped_verified_sentence", self.text)
        _require_unique(self.claim_ids, "sentence claim_ids")
        _require_unique(self.source_ids, "sentence source_ids")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedSentence:
        value = strict_fields(data, {"claim_ids", "source_ids", "text"}, "VerifiedSentence")
        return cls(
            _strings(value["claim_ids"], "claim_ids"),
            _strings(value["source_ids"], "source_ids"),
            require_string(value["text"], "text"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_ids": list(self.claim_ids),
            "source_ids": list(self.source_ids),
            "text": self.text,
        }


@dataclass(frozen=True)
class Coverage:
    denominator: int
    numerator: int
    value: float

    def __post_init__(self) -> None:
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise EvidenceError("invalid_coverage", "coverage counts are invalid")
        if self.value != self.numerator / self.denominator:
            raise EvidenceError("invalid_coverage", "coverage value differs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Coverage:
        value = strict_fields(data, {"denominator", "numerator", "value"}, "Coverage")
        return cls(
            require_int(value["denominator"], "denominator"),
            require_int(value["numerator"], "numerator"),
            require_number(value["value"], "value"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "denominator": self.denominator,
            "numerator": self.numerator,
            "value": self.value,
        }


@dataclass(frozen=True)
class VerifiedCommunicationArtifact:
    accepted_claims: tuple[ProposedClaim, ...]
    caveat_coverage: Coverage
    exclusions: tuple[ClaimExclusion, ...]
    fact_coverage: Coverage
    prose: str
    refusal_reasons: tuple[str, ...]
    request_id: str
    schema_version: str
    sentences: tuple[VerifiedSentence, ...]
    source_atlas_sha256: str
    source_card_sha256: str
    terminal_state: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state not in ARTIFACT_TERMINAL_STATES:
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.source_atlas_sha256, "source_atlas_sha256")
        require_sha256(self.source_card_sha256, "source_card_sha256")
        if self.terminal_state == "refused" and (self.prose or self.sentences):
            raise EvidenceError("refused_artifact_has_prose", self.request_id)
        sentence_text = " ".join(item.text for item in self.sentences)
        if self.terminal_state != "refused" and self.prose != sentence_text:
            raise EvidenceError("prose_sentence_mismatch", self.request_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedCommunicationArtifact:
        value = strict_fields(
            data,
            {
                "accepted_claims",
                "caveat_coverage",
                "exclusions",
                "fact_coverage",
                "prose",
                "refusal_reasons",
                "request_id",
                "schema_version",
                "sentences",
                "source_atlas_sha256",
                "source_card_sha256",
                "terminal_state",
                "warnings",
            },
            "VerifiedCommunicationArtifact",
        )
        return cls(
            tuple(
                ProposedClaim.from_dict(require_object(item, "accepted claim"))
                for item in require_list(value["accepted_claims"], "accepted_claims")
            ),
            Coverage.from_dict(require_object(value["caveat_coverage"], "caveat_coverage")),
            tuple(
                ClaimExclusion.from_dict(require_object(item, "claim exclusion"))
                for item in require_list(value["exclusions"], "exclusions")
            ),
            Coverage.from_dict(require_object(value["fact_coverage"], "fact_coverage")),
            require_string(value["prose"], "prose", nonempty=False),
            _strings(value["refusal_reasons"], "refusal_reasons"),
            require_string(value["request_id"], "request_id"),
            require_string(value["schema_version"], "schema_version"),
            tuple(
                VerifiedSentence.from_dict(require_object(item, "verified sentence"))
                for item in require_list(value["sentences"], "sentences")
            ),
            require_string(value["source_atlas_sha256"], "source_atlas_sha256"),
            require_string(value["source_card_sha256"], "source_card_sha256"),
            require_string(value["terminal_state"], "terminal_state"),
            _strings(value["warnings"], "warnings"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_claims": [item.to_dict() for item in self.accepted_claims],
            "caveat_coverage": self.caveat_coverage.to_dict(),
            "exclusions": [item.to_dict() for item in self.exclusions],
            "fact_coverage": self.fact_coverage.to_dict(),
            "prose": self.prose,
            "refusal_reasons": list(self.refusal_reasons),
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "sentences": [item.to_dict() for item in self.sentences],
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_card_sha256": self.source_card_sha256,
            "terminal_state": self.terminal_state,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CommunicationReceipt:
    artifact_sha256: str
    communication_terminal_state: str
    draft_sha256: str
    exclusions: tuple[ClaimExclusion, ...]
    model_identity: ModelIdentity
    prompt_identity: PromptIdentity
    request_id: str
    request_sha256: str
    required_evidence_ids: tuple[str, ...]
    schema_version: str
    source_atlas_sha256: str
    source_card_sha256: str
    terminal_state: str
    transformation_id: str
    verifier_version: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.communication_terminal_state not in ARTIFACT_TERMINAL_STATES:
            raise EvidenceError("invalid_terminal_state", self.communication_terminal_state)
        if self.terminal_state != "closed":
            raise EvidenceError("receipt_not_closed", self.terminal_state)
        if self.transformation_id != TRANSFORMATION_ID:
            raise EvidenceError("unsupported_transformation", self.transformation_id)
        if self.verifier_version != VERIFIER_VERSION:
            raise EvidenceError("unsupported_verifier", self.verifier_version)
        for field_name, value in (
            ("artifact_sha256", self.artifact_sha256),
            ("draft_sha256", self.draft_sha256),
            ("request_sha256", self.request_sha256),
            ("source_atlas_sha256", self.source_atlas_sha256),
            ("source_card_sha256", self.source_card_sha256),
        ):
            require_sha256(value, field_name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationReceipt:
        value = strict_fields(
            data,
            {
                "artifact_sha256",
                "communication_terminal_state",
                "draft_sha256",
                "exclusions",
                "model_identity",
                "prompt_identity",
                "request_id",
                "request_sha256",
                "required_evidence_ids",
                "schema_version",
                "source_atlas_sha256",
                "source_card_sha256",
                "terminal_state",
                "transformation_id",
                "verifier_version",
                "warnings",
            },
            "CommunicationReceipt",
        )
        return cls(
            require_string(value["artifact_sha256"], "artifact_sha256"),
            require_string(
                value["communication_terminal_state"],
                "communication_terminal_state",
            ),
            require_string(value["draft_sha256"], "draft_sha256"),
            tuple(
                ClaimExclusion.from_dict(require_object(item, "claim exclusion"))
                for item in require_list(value["exclusions"], "exclusions")
            ),
            ModelIdentity.from_dict(require_object(value["model_identity"], "model_identity")),
            PromptIdentity.from_dict(require_object(value["prompt_identity"], "prompt_identity")),
            require_string(value["request_id"], "request_id"),
            require_string(value["request_sha256"], "request_sha256"),
            _strings(value["required_evidence_ids"], "required_evidence_ids"),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["source_atlas_sha256"], "source_atlas_sha256"),
            require_string(value["source_card_sha256"], "source_card_sha256"),
            require_string(value["terminal_state"], "terminal_state"),
            require_string(value["transformation_id"], "transformation_id"),
            require_string(value["verifier_version"], "verifier_version"),
            _strings(value["warnings"], "warnings"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "communication_terminal_state": self.communication_terminal_state,
            "draft_sha256": self.draft_sha256,
            "exclusions": [item.to_dict() for item in self.exclusions],
            "model_identity": self.model_identity.to_dict(),
            "prompt_identity": self.prompt_identity.to_dict(),
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "required_evidence_ids": list(self.required_evidence_ids),
            "schema_version": self.schema_version,
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_card_sha256": self.source_card_sha256,
            "terminal_state": self.terminal_state,
            "transformation_id": self.transformation_id,
            "verifier_version": self.verifier_version,
            "warnings": list(self.warnings),
        }
