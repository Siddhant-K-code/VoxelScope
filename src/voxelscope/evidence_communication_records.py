# SPDX-License-Identifier: Apache-2.0
"""Strict records for the synthetic evidence communication compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    require_sha256,
    sha256_bytes,
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

REQUEST_SCHEMA = "voxelscope/evidence-communication-request/v2"
DRAFT_SCHEMA = "voxelscope/evidence-communication-model-draft/v2"
ARTIFACT_SCHEMA = "voxelscope/verified-evidence-communication/v2"
RECEIPT_SCHEMA = "voxelscope/evidence-communication-receipt/v2"
TRANSFORMATION_ID = "voxelscope/evidence-communication-compiler/v2"
VERIFIER_VERSION = "voxelscope/evidence-communication-verifier/v2"

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
class VerifiedClaim:
    canonical_gene_id: str
    canonical_protein_id: str
    claim_type: str
    comparison_key: str | None
    context: str | None
    direction: str | None
    modality: str | None
    requirement_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    state: str | None
    unit: str | None
    value: float | None
    verified_claim_id: str

    def __post_init__(self) -> None:
        if self.claim_type not in CLAIM_TYPES:
            raise EvidenceError("unsupported_claim_type", self.claim_type)
        if not self.source_ids:
            raise EvidenceError("missing_source_citation", self.verified_claim_id)
        _require_unique(self.source_ids, "verified claim source_ids")
        _require_unique(self.requirement_ids, "verified claim requirement_ids")
        prefix = "verified:"
        if not self.verified_claim_id.startswith(prefix):
            raise EvidenceError("invalid_verified_claim_id", self.verified_claim_id)
        require_sha256(
            self.verified_claim_id.removeprefix(prefix),
            "verified claim digest",
        )
        expected_id = prefix + sha256_bytes(canonical_json_bytes(self.semantic_dict()))
        if self.verified_claim_id != expected_id:
            raise EvidenceError("verified_claim_digest_mismatch", self.verified_claim_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedClaim:
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
                "source_ids",
                "state",
                "unit",
                "value",
                "verified_claim_id",
            },
            "VerifiedClaim",
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
            _strings(value["source_ids"], "source_ids"),
            _optional_string(value["state"], "state"),
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
            "source_ids": list(self.source_ids),
            "state": self.state,
            "unit": self.unit,
            "value": self.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_dict(),
            "verified_claim_id": self.verified_claim_id,
        }


@dataclass(frozen=True)
class InvalidModelOutput:
    error_code: str
    output_sha256: str
    output_size_bytes: int

    def __post_init__(self) -> None:
        require_sha256(self.output_sha256, "model output sha256")
        if self.output_size_bytes < 0:
            raise EvidenceError("invalid_model_output_size", str(self.output_size_bytes))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvalidModelOutput:
        value = strict_fields(
            data,
            {"error_code", "output_sha256", "output_size_bytes"},
            "InvalidModelOutput",
        )
        return cls(
            require_string(value["error_code"], "error_code"),
            require_string(value["output_sha256"], "output_sha256"),
            require_int(value["output_size_bytes"], "output_size_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "output_sha256": self.output_sha256,
            "output_size_bytes": self.output_size_bytes,
        }


@dataclass(frozen=True)
class VerifiedSentence:
    source_ids: tuple[str, ...]
    text: str
    verified_claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.verified_claim_ids or not self.source_ids:
            raise EvidenceError("unmapped_verified_sentence", self.text)
        _require_unique(self.verified_claim_ids, "sentence verified_claim_ids")
        _require_unique(self.source_ids, "sentence source_ids")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedSentence:
        value = strict_fields(
            data,
            {"source_ids", "text", "verified_claim_ids"},
            "VerifiedSentence",
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
    emitted_caveat_coverage: Coverage
    emitted_fact_coverage: Coverage
    exclusions: tuple[ClaimExclusion, ...]
    invalid_model_output: InvalidModelOutput | None
    prose: str
    refusal_reasons: tuple[str, ...]
    request_id: str
    schema_version: str
    sentences: tuple[VerifiedSentence, ...]
    source_atlas_sha256: str
    source_card_sha256: str
    terminal_state: str
    verified_claims: tuple[VerifiedClaim, ...]
    verified_draft_caveat_coverage: Coverage
    verified_draft_fact_coverage: Coverage
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.terminal_state not in ARTIFACT_TERMINAL_STATES:
            raise EvidenceError("invalid_terminal_state", self.terminal_state)
        require_sha256(self.source_atlas_sha256, "source_atlas_sha256")
        require_sha256(self.source_card_sha256, "source_card_sha256")
        verified_claims = {item.verified_claim_id: item for item in self.verified_claims}
        if len(verified_claims) != len(self.verified_claims):
            raise EvidenceError("duplicate_verified_claim_id", self.request_id)
        if (
            self.emitted_fact_coverage.denominator != self.verified_draft_fact_coverage.denominator
            or self.emitted_caveat_coverage.denominator
            != self.verified_draft_caveat_coverage.denominator
        ):
            raise EvidenceError("coverage_denominator_mismatch", self.request_id)
        if self.terminal_state == "refused" and (self.prose or self.sentences):
            raise EvidenceError("refused_artifact_has_prose", self.request_id)
        if self.terminal_state == "refused" and (
            self.emitted_fact_coverage.numerator != 0 or self.emitted_caveat_coverage.numerator != 0
        ):
            raise EvidenceError("refused_artifact_has_emitted_coverage", self.request_id)
        if self.terminal_state != "refused" and (
            self.emitted_fact_coverage != self.verified_draft_fact_coverage
            or self.emitted_caveat_coverage != self.verified_draft_caveat_coverage
        ):
            raise EvidenceError("emitted_coverage_mismatch", self.request_id)
        if self.invalid_model_output is not None and (
            self.terminal_state != "refused" or self.verified_claims or self.exclusions
        ):
            raise EvidenceError("invalid_model_output_artifact_mismatch", self.request_id)
        sentence_text = " ".join(item.text for item in self.sentences)
        if self.terminal_state != "refused" and self.prose != sentence_text:
            raise EvidenceError("prose_sentence_mismatch", self.request_id)
        mapped_claim_ids: set[str] = set()
        for sentence in self.sentences:
            sentence_claims: list[VerifiedClaim] = []
            for claim_id in sentence.verified_claim_ids:
                claim = verified_claims.get(claim_id)
                if claim is None:
                    raise EvidenceError("unknown_sentence_verified_claim", claim_id)
                sentence_claims.append(claim)
                mapped_claim_ids.add(claim_id)
            expected_sources = {
                source_id for claim in sentence_claims for source_id in claim.source_ids
            }
            if set(sentence.source_ids) != expected_sources:
                raise EvidenceError("sentence_source_mapping_mismatch", sentence.text)
        if self.terminal_state != "refused" and mapped_claim_ids != set(verified_claims):
            raise EvidenceError("unmapped_verified_claim", self.request_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedCommunicationArtifact:
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
                "terminal_state",
                "verified_claims",
                "verified_draft_caveat_coverage",
                "verified_draft_fact_coverage",
                "warnings",
            },
            "VerifiedCommunicationArtifact",
        )
        invalid_value = value["invalid_model_output"]
        return cls(
            Coverage.from_dict(
                require_object(
                    value["emitted_caveat_coverage"],
                    "emitted_caveat_coverage",
                )
            ),
            Coverage.from_dict(
                require_object(value["emitted_fact_coverage"], "emitted_fact_coverage")
            ),
            tuple(
                ClaimExclusion.from_dict(require_object(item, "claim exclusion"))
                for item in require_list(value["exclusions"], "exclusions")
            ),
            (
                InvalidModelOutput.from_dict(require_object(invalid_value, "invalid_model_output"))
                if invalid_value is not None
                else None
            ),
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
            tuple(
                VerifiedClaim.from_dict(require_object(item, "verified claim"))
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
            "terminal_state": self.terminal_state,
            "verified_claims": [item.to_dict() for item in self.verified_claims],
            "verified_draft_caveat_coverage": (self.verified_draft_caveat_coverage.to_dict()),
            "verified_draft_fact_coverage": (self.verified_draft_fact_coverage.to_dict()),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CommunicationReceipt:
    artifact_sha256: str
    communication_terminal_state: str
    draft_sha256: str | None
    exclusions: tuple[ClaimExclusion, ...]
    invalid_model_output: InvalidModelOutput | None
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
            ("request_sha256", self.request_sha256),
            ("source_atlas_sha256", self.source_atlas_sha256),
            ("source_card_sha256", self.source_card_sha256),
        ):
            require_sha256(value, field_name)
        if self.draft_sha256 is not None:
            require_sha256(self.draft_sha256, "draft_sha256")
        if (self.draft_sha256 is None) == (self.invalid_model_output is None):
            raise EvidenceError("receipt_model_output_identity_mismatch", self.request_id)
        if self.invalid_model_output is not None and self.exclusions:
            raise EvidenceError("invalid_output_receipt_has_exclusions", self.request_id)
        if self.invalid_model_output is not None and self.communication_terminal_state != "refused":
            raise EvidenceError("invalid_output_receipt_not_refused", self.request_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunicationReceipt:
        value = strict_fields(
            data,
            {
                "artifact_sha256",
                "communication_terminal_state",
                "draft_sha256",
                "exclusions",
                "invalid_model_output",
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
            _optional_string(value["draft_sha256"], "draft_sha256"),
            tuple(
                ClaimExclusion.from_dict(require_object(item, "claim exclusion"))
                for item in require_list(value["exclusions"], "exclusions")
            ),
            (
                InvalidModelOutput.from_dict(
                    require_object(value["invalid_model_output"], "invalid_model_output")
                )
                if value["invalid_model_output"] is not None
                else None
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
            "invalid_model_output": (
                self.invalid_model_output.to_dict()
                if self.invalid_model_output is not None
                else None
            ),
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
