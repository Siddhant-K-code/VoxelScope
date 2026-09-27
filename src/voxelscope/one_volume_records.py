# SPDX-License-Identifier: Apache-2.0
"""Typed evidence for the milestone 3 one-volume source decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from .canonical import EvidenceError, safe_relative_path
from .custody_records import DigestIdentity, require_https_url
from .records import (
    require_bool,
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

GateStatus = Literal["pass", "fail", "unresolved"]
CandidateDisposition = Literal["go", "no-go"]
ArtifactRole = Literal[
    "image-t1c",
    "image-t1",
    "image-t2",
    "image-flair",
    "label",
    "metadata-dataset-description",
    "metadata-readme",
    "metadata-changes",
]
SearchProvider = Literal["exa.ai", "parallel.ai"]
PlanExecutionStatus = Literal["blocked"]

_GATE_STATUSES = frozenset({"pass", "fail", "unresolved"})
_REQUIRED_ARTIFACT_ORDER: tuple[ArtifactRole, ...] = (
    "image-t1c",
    "image-t1",
    "image-t2",
    "image-flair",
    "label",
    "metadata-dataset-description",
    "metadata-readme",
    "metadata-changes",
)
_REQUIRED_ARTIFACT_ROLES = frozenset(_REQUIRED_ARTIFACT_ORDER)
_ARTIFACT_ROLES = _REQUIRED_ARTIFACT_ROLES
_MEDICAL_ARTIFACT_ROLES = frozenset({"image-t1c", "image-t1", "image-t2", "image-flair", "label"})
_SOURCE_GO_GATES = (
    "official_publisher",
    "direct_nonbulk_image",
    "direct_nonbulk_label",
    "image_label_pairing",
    "metadata_lineage",
    "license",
)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _digests(value: Any, name: str) -> tuple[DigestIdentity, ...]:
    digests = tuple(
        DigestIdentity.from_dict(require_object(item, f"{name} item"))
        for item in require_list(value, name)
    )
    algorithms = [digest.algorithm for digest in digests]
    if len(algorithms) != len(set(algorithms)):
        raise EvidenceError("duplicate_digest", name)
    return digests


def _date(value: Any, name: str) -> str:
    text = require_string(value, name)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceError("invalid_date", name) from exc
    if parsed.isoformat() != text:
        raise EvidenceError("invalid_date", name)
    return text


def _gate_status(value: Any, name: str) -> GateStatus:
    status = require_string(value, name)
    if status not in _GATE_STATUSES:
        raise EvidenceError("invalid_gate_status", f"{name}: {status}")
    return cast(GateStatus, status)


@dataclass(frozen=True)
class CandidateArtifact:
    role: ArtifactRole
    source_url: str
    immutable_id: str
    size_bytes: int
    hashes: tuple[DigestIdentity, ...]

    def __post_init__(self) -> None:
        if self.role not in _ARTIFACT_ROLES:
            raise EvidenceError("invalid_artifact_role", self.role)
        require_https_url(self.source_url, "source_url")
        if not self.immutable_id:
            raise EvidenceError("missing_immutable_id", self.role)
        if self.size_bytes <= 0:
            raise EvidenceError("invalid_size", self.role)
        if not self.hashes:
            raise EvidenceError("missing_expected_hash", self.role)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateArtifact:
        data = strict_fields(
            data,
            {"hashes", "immutable_id", "role", "size_bytes", "source_url"},
            "CandidateArtifact",
        )
        role = require_string(data["role"], "role")
        if role not in _ARTIFACT_ROLES:
            raise EvidenceError("invalid_artifact_role", role)
        return cls(
            role,
            require_https_url(data["source_url"], "source_url"),
            require_string(data["immutable_id"], "immutable_id"),
            require_int(data["size_bytes"], "size_bytes"),
            _digests(data["hashes"], "hashes"),
        )


@dataclass(frozen=True)
class GateAssessment:
    official_publisher: GateStatus
    direct_nonbulk_image: GateStatus
    direct_nonbulk_label: GateStatus
    image_label_pairing: GateStatus
    metadata_lineage: GateStatus
    license: GateStatus
    model_input: GateStatus
    subject_overlap: GateStatus

    def statuses(self) -> tuple[GateStatus, ...]:
        return (
            self.official_publisher,
            self.direct_nonbulk_image,
            self.direct_nonbulk_label,
            self.image_label_pairing,
            self.metadata_lineage,
            self.license,
            self.model_input,
            self.subject_overlap,
        )

    def source_statuses(self) -> tuple[GateStatus, ...]:
        return tuple(cast(GateStatus, getattr(self, name)) for name in _SOURCE_GO_GATES)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateAssessment:
        fields = {
            "direct_nonbulk_image",
            "direct_nonbulk_label",
            "image_label_pairing",
            "license",
            "metadata_lineage",
            "model_input",
            "official_publisher",
            "subject_overlap",
        }
        data = strict_fields(data, fields, "GateAssessment")
        return cls(
            official_publisher=_gate_status(data["official_publisher"], "official_publisher"),
            direct_nonbulk_image=_gate_status(data["direct_nonbulk_image"], "direct_nonbulk_image"),
            direct_nonbulk_label=_gate_status(data["direct_nonbulk_label"], "direct_nonbulk_label"),
            image_label_pairing=_gate_status(data["image_label_pairing"], "image_label_pairing"),
            metadata_lineage=_gate_status(data["metadata_lineage"], "metadata_lineage"),
            license=_gate_status(data["license"], "license"),
            model_input=_gate_status(data["model_input"], "model_input"),
            subject_overlap=_gate_status(data["subject_overlap"], "subject_overlap"),
        )


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    name: str
    source_urls: tuple[str, ...]
    immutable_ids: tuple[str, ...]
    license_id: str | None
    access_summary: str
    compatibility_summary: str
    gates: GateAssessment
    artifacts: tuple[CandidateArtifact, ...]
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    blockers: tuple[str, ...]
    disposition: CandidateDisposition

    def __post_init__(self) -> None:
        if not self.source_urls:
            raise EvidenceError("missing_source_url", self.candidate_id)
        for url in self.source_urls:
            require_https_url(url, "source_url")
        if not self.evidence:
            raise EvidenceError("missing_evidence", self.candidate_id)
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise EvidenceError("duplicate_artifact_role", self.candidate_id)
        if self.disposition not in {"go", "no-go"}:
            raise EvidenceError("invalid_candidate_disposition", self.candidate_id)
        if self.disposition == "go":
            if any(status != "pass" for status in self.gates.source_statuses()):
                raise EvidenceError("unsafe_candidate_go", self.candidate_id)
            if not self.immutable_ids or self.license_id is None:
                raise EvidenceError("unsafe_candidate_go", "immutable source and license required")
            if set(roles) != _REQUIRED_ARTIFACT_ROLES:
                raise EvidenceError(
                    "unsafe_candidate_go",
                    "four images, label, and three metadata artifacts required",
                )
            if any(
                "sha256" not in {digest.algorithm for digest in artifact.hashes}
                for artifact in self.artifacts
            ):
                raise EvidenceError("unsafe_candidate_go", "every artifact requires SHA-256")
            if self.blockers:
                raise EvidenceError("unsafe_candidate_go", "GO candidate has blockers")
        elif not self.blockers:
            raise EvidenceError("missing_blocker", self.candidate_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateAssessment:
        data = strict_fields(
            data,
            {
                "access_summary",
                "artifacts",
                "assumptions",
                "blockers",
                "candidate_id",
                "compatibility_summary",
                "disposition",
                "evidence",
                "gates",
                "immutable_ids",
                "license_id",
                "name",
                "source_urls",
            },
            "CandidateAssessment",
        )
        license_value = data["license_id"]
        license_id = (
            require_string(license_value, "license_id") if license_value is not None else None
        )
        disposition = require_string(data["disposition"], "disposition")
        if disposition not in {"go", "no-go"}:
            raise EvidenceError("invalid_candidate_disposition", disposition)
        return cls(
            candidate_id=require_string(data["candidate_id"], "candidate_id"),
            name=require_string(data["name"], "name"),
            source_urls=tuple(
                require_https_url(item, "source_url")
                for item in require_list(data["source_urls"], "source_urls")
            ),
            immutable_ids=_strings(data["immutable_ids"], "immutable_ids"),
            license_id=license_id,
            access_summary=require_string(data["access_summary"], "access_summary"),
            compatibility_summary=require_string(
                data["compatibility_summary"], "compatibility_summary"
            ),
            gates=GateAssessment.from_dict(require_object(data["gates"], "gates")),
            artifacts=tuple(
                CandidateArtifact.from_dict(require_object(item, "artifact"))
                for item in require_list(data["artifacts"], "artifacts")
            ),
            evidence=_strings(data["evidence"], "evidence"),
            assumptions=_strings(data["assumptions"], "assumptions"),
            blockers=_strings(data["blockers"], "blockers"),
            disposition=cast(CandidateDisposition, disposition),
        )


@dataclass(frozen=True)
class SearchEvidence:
    provider: SearchProvider
    independent_query_count: int
    query_topics: tuple[str, ...]
    verified_urls: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.provider not in {"exa.ai", "parallel.ai"}:
            raise EvidenceError("unsupported_search_provider", self.provider)
        if self.independent_query_count < 2 or len(self.query_topics) < 2:
            raise EvidenceError("insufficient_independent_search", self.provider)
        if not self.verified_urls:
            raise EvidenceError("missing_direct_verification", self.provider)
        for url in self.verified_urls:
            require_https_url(url, "verified_url")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchEvidence:
        data = strict_fields(
            data,
            {"independent_query_count", "provider", "query_topics", "verified_urls"},
            "SearchEvidence",
        )
        provider = require_string(data["provider"], "provider")
        if provider not in {"exa.ai", "parallel.ai"}:
            raise EvidenceError("unsupported_search_provider", provider)
        return cls(
            cast(SearchProvider, provider),
            require_int(data["independent_query_count"], "independent_query_count"),
            _strings(data["query_topics"], "query_topics"),
            tuple(
                require_https_url(item, "verified_url")
                for item in require_list(data["verified_urls"], "verified_urls")
            ),
        )


@dataclass(frozen=True)
class LabelSemantic:
    value: int
    meaning: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelSemantic:
        data = strict_fields(data, {"meaning", "value"}, "LabelSemantic")
        return cls(
            require_int(data["value"], "value"),
            require_string(data["meaning"], "meaning"),
        )


@dataclass(frozen=True)
class ModelCompatibilityContract:
    bundle_id: str
    input_channel_order: tuple[str, ...]
    required_source_preprocessing: tuple[str, ...]
    inference_normalization: str
    source_label_semantics: tuple[LabelSemantic, ...]
    output_region_order: tuple[str, ...]
    allowed_measurement_scope: Literal["output-preservation-only"]

    def __post_init__(self) -> None:
        if self.input_channel_order != ("T1c", "T1", "T2", "FLAIR"):
            raise EvidenceError("unsafe_channel_order", repr(self.input_channel_order))
        if tuple(item.value for item in self.source_label_semantics) != (0, 1, 2, 4):
            raise EvidenceError("unsafe_label_semantics", "expected labels 0, 1, 2, and 4")
        if self.output_region_order != ("TC", "WT", "ET"):
            raise EvidenceError("unsafe_output_order", repr(self.output_region_order))
        if self.allowed_measurement_scope != "output-preservation-only":
            raise EvidenceError("unsafe_measurement_scope", self.allowed_measurement_scope)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCompatibilityContract:
        data = strict_fields(
            data,
            {
                "allowed_measurement_scope",
                "bundle_id",
                "inference_normalization",
                "input_channel_order",
                "output_region_order",
                "required_source_preprocessing",
                "source_label_semantics",
            },
            "ModelCompatibilityContract",
        )
        scope = require_string(data["allowed_measurement_scope"], "allowed_measurement_scope")
        if scope != "output-preservation-only":
            raise EvidenceError("unsafe_measurement_scope", scope)
        return cls(
            require_string(data["bundle_id"], "bundle_id"),
            _strings(data["input_channel_order"], "input_channel_order"),
            _strings(data["required_source_preprocessing"], "required_source_preprocessing"),
            require_string(data["inference_normalization"], "inference_normalization"),
            tuple(
                LabelSemantic.from_dict(require_object(item, "label semantic"))
                for item in require_list(data["source_label_semantics"], "source_label_semantics")
            ),
            _strings(data["output_region_order"], "output_region_order"),
            "output-preservation-only",
        )


@dataclass(frozen=True)
class OneVolumeDecision:
    schema_version: Literal["voxelscope/one-volume-source-decision/v1"]
    decision_id: str
    as_of: str
    decision: CandidateDisposition
    inference_status: Literal["no-go"]
    selected_candidate_id: str | None
    network_default: Literal["disabled"]
    medical_data_acquired: Literal[False]
    model_contract: ModelCompatibilityContract
    search_evidence: tuple[SearchEvidence, ...]
    candidates: tuple[CandidateAssessment, ...]
    controlling_blockers: tuple[str, ...]
    next_gate: str
    non_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/one-volume-source-decision/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        _date(self.as_of, "as_of")
        if self.network_default != "disabled":
            raise EvidenceError("unsafe_network_default", self.network_default)
        if self.inference_status != "no-go":
            raise EvidenceError("unsafe_inference_status", self.inference_status)
        if self.medical_data_acquired is not False:
            raise EvidenceError("medical_data_acquisition_forbidden", self.decision_id)
        if not self.controlling_blockers or not self.next_gate or not self.non_claims:
            raise EvidenceError("incomplete_decision", self.decision_id)
        providers = [item.provider for item in self.search_evidence]
        if set(providers) != {"exa.ai", "parallel.ai"} or len(providers) != 2:
            raise EvidenceError("missing_search_provider", self.decision_id)
        identifiers = [candidate.candidate_id for candidate in self.candidates]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise EvidenceError("invalid_candidate_set", self.decision_id)
        go_candidates = [
            candidate for candidate in self.candidates if candidate.disposition == "go"
        ]
        if self.decision == "go":
            if (
                len(go_candidates) != 1
                or self.selected_candidate_id != go_candidates[0].candidate_id
            ):
                raise EvidenceError("unsafe_decision_go", self.decision_id)
        elif self.decision == "no-go":
            if self.selected_candidate_id is not None or go_candidates:
                raise EvidenceError("unsafe_decision_no_go", self.decision_id)
        else:
            raise EvidenceError("invalid_decision", self.decision)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OneVolumeDecision:
        data = strict_fields(
            data,
            {
                "as_of",
                "candidates",
                "controlling_blockers",
                "decision",
                "decision_id",
                "inference_status",
                "medical_data_acquired",
                "model_contract",
                "network_default",
                "next_gate",
                "non_claims",
                "schema_version",
                "search_evidence",
                "selected_candidate_id",
            },
            "OneVolumeDecision",
        )
        schema = require_string(data["schema_version"], "schema_version")
        if schema != "voxelscope/one-volume-source-decision/v1":
            raise EvidenceError("unsupported_schema", schema)
        decision = require_string(data["decision"], "decision")
        if decision not in {"go", "no-go"}:
            raise EvidenceError("invalid_decision", decision)
        selected_value = data["selected_candidate_id"]
        selected = (
            require_string(selected_value, "selected_candidate_id")
            if selected_value is not None
            else None
        )
        network_default = require_string(data["network_default"], "network_default")
        if network_default != "disabled":
            raise EvidenceError("unsafe_network_default", network_default)
        acquired = require_bool(data["medical_data_acquired"], "medical_data_acquired")
        if acquired:
            raise EvidenceError("medical_data_acquisition_forbidden", "medical_data_acquired")
        return cls(
            "voxelscope/one-volume-source-decision/v1",
            require_string(data["decision_id"], "decision_id"),
            _date(data["as_of"], "as_of"),
            cast(CandidateDisposition, decision),
            cast(Literal["no-go"], require_string(data["inference_status"], "inference_status")),
            selected,
            "disabled",
            False,
            ModelCompatibilityContract.from_dict(
                require_object(data["model_contract"], "model_contract")
            ),
            tuple(
                SearchEvidence.from_dict(require_object(item, "search evidence"))
                for item in require_list(data["search_evidence"], "search_evidence")
            ),
            tuple(
                CandidateAssessment.from_dict(require_object(item, "candidate"))
                for item in require_list(data["candidates"], "candidates")
            ),
            _strings(data["controlling_blockers"], "controlling_blockers"),
            require_string(data["next_gate"], "next_gate"),
            _strings(data["non_claims"], "non_claims"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedArtifact:
    artifact_id: str
    role: ArtifactRole
    source_url: str
    immutable_id: str
    allowed_origins: tuple[str, ...]
    destination: str
    expected_size_bytes: int
    expected_hashes: tuple[DigestIdentity, ...]
    medical_data: bool
    operator_approval_required: bool

    def __post_init__(self) -> None:
        if self.role not in _ARTIFACT_ROLES:
            raise EvidenceError("invalid_artifact_role", self.role)
        source_url = require_https_url(self.source_url, "source_url")
        source_origin = f"{urlsplit(source_url).scheme}://{urlsplit(source_url).netloc}"
        if not self.allowed_origins or source_origin not in self.allowed_origins:
            raise EvidenceError("source_not_allowlisted", source_url)
        for origin in self.allowed_origins:
            require_https_url(origin, "allowed_origin", origin_only=True)
        safe_relative_path(self.destination)
        if not self.immutable_id:
            raise EvidenceError("missing_immutable_id", self.artifact_id)
        if self.expected_size_bytes <= 0:
            raise EvidenceError("invalid_size", self.artifact_id)
        if not self.expected_hashes:
            raise EvidenceError("missing_expected_hash", self.artifact_id)
        expected_medical = self.role in _MEDICAL_ARTIFACT_ROLES
        if self.medical_data != expected_medical:
            raise EvidenceError("invalid_medical_data_classification", self.artifact_id)
        if self.medical_data and not self.operator_approval_required:
            raise EvidenceError("medical_data_approval_required", self.artifact_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedArtifact:
        data = strict_fields(
            data,
            {
                "allowed_origins",
                "artifact_id",
                "destination",
                "expected_hashes",
                "expected_size_bytes",
                "immutable_id",
                "medical_data",
                "operator_approval_required",
                "role",
                "source_url",
            },
            "PlannedArtifact",
        )
        role = require_string(data["role"], "role")
        if role not in _ARTIFACT_ROLES:
            raise EvidenceError("invalid_artifact_role", role)
        return cls(
            require_string(data["artifact_id"], "artifact_id"),
            role,
            require_https_url(data["source_url"], "source_url"),
            require_string(data["immutable_id"], "immutable_id"),
            tuple(
                require_https_url(item, "allowed_origin", origin_only=True)
                for item in require_list(data["allowed_origins"], "allowed_origins")
            ),
            require_string(data["destination"], "destination"),
            require_int(data["expected_size_bytes"], "expected_size_bytes"),
            _digests(data["expected_hashes"], "expected_hashes"),
            require_bool(data["medical_data"], "medical_data"),
            require_bool(data["operator_approval_required"], "operator_approval_required"),
        )


@dataclass(frozen=True)
class OneVolumeAcquisitionPlan:
    schema_version: Literal["voxelscope/one-volume-acquisition-plan/v1"]
    plan_id: str
    decision_id: str
    candidate_id: str
    network_default: Literal["disabled"]
    execution_status: PlanExecutionStatus
    blocked_reason: str
    artifacts: tuple[PlannedArtifact, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/one-volume-acquisition-plan/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.network_default != "disabled":
            raise EvidenceError("unsafe_network_default", self.network_default)
        if self.execution_status != "blocked" or not self.blocked_reason:
            raise EvidenceError("unsafe_plan_status", self.plan_id)
        identifiers = [artifact.artifact_id for artifact in self.artifacts]
        destinations = [artifact.destination for artifact in self.artifacts]
        roles = [artifact.role for artifact in self.artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise EvidenceError("duplicate_acquisition_artifact", self.plan_id)
        if len(destinations) != len(set(destinations)):
            raise EvidenceError("duplicate_destination", self.plan_id)
        if tuple(roles) != _REQUIRED_ARTIFACT_ORDER:
            raise EvidenceError("incomplete_acquisition_plan", self.plan_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OneVolumeAcquisitionPlan:
        data = strict_fields(
            data,
            {
                "artifacts",
                "blocked_reason",
                "candidate_id",
                "decision_id",
                "execution_status",
                "network_default",
                "plan_id",
                "schema_version",
            },
            "OneVolumeAcquisitionPlan",
        )
        schema = require_string(data["schema_version"], "schema_version")
        if schema != "voxelscope/one-volume-acquisition-plan/v1":
            raise EvidenceError("unsupported_schema", schema)
        network = require_string(data["network_default"], "network_default")
        if network != "disabled":
            raise EvidenceError("unsafe_network_default", network)
        status = require_string(data["execution_status"], "execution_status")
        if status != "blocked":
            raise EvidenceError("unsafe_plan_status", status)
        return cls(
            "voxelscope/one-volume-acquisition-plan/v1",
            require_string(data["plan_id"], "plan_id"),
            require_string(data["decision_id"], "decision_id"),
            require_string(data["candidate_id"], "candidate_id"),
            "disabled",
            "blocked",
            require_string(data["blocked_reason"], "blocked_reason"),
            tuple(
                PlannedArtifact.from_dict(require_object(item, "planned artifact"))
                for item in require_list(data["artifacts"], "artifacts")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
