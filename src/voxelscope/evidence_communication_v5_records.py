# SPDX-License-Identifier: Apache-2.0
"""Strict records for the deterministic v5 discourse planner."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .canonical import EvidenceError, require_sha256, safe_relative_path
from .records import (
    require_bool,
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

REQUEST_SCHEMA_V5 = "voxelscope/evidence-communication-v5/discourse-request/v1"
PLAN_SCHEMA_V5 = "voxelscope/evidence-communication-v5/discourse-plan/v1"
CATALOG_SCHEMA_V5 = "voxelscope/evidence-communication-v5/sentence-catalog/v1"
IMPLEMENTATION_MANIFEST_SCHEMA_V5 = (
    "voxelscope/evidence-communication-v5/implementation-manifest/v1"
)
ELIGIBILITY_SCHEMA_V5 = "voxelscope/evidence-communication-v5/model-eligibility/v1"
OPTIMALITY_SCHEMA_V5 = "voxelscope/evidence-communication-v5/optimality-certificate/v1"
RENDERED_ARTIFACT_SCHEMA_V5 = "voxelscope/evidence-communication-v5/rendered-artifact/v1"
PUBLICATION_SCHEMA_V5 = "voxelscope/evidence-communication-v5/publication-record/v1"
MEASUREMENT_SCHEMA_V5 = "voxelscope/evidence-communication-v5/measurement-status/v1"
VERIFICATION_SCHEMA_V5 = "voxelscope/evidence-communication-v5/verification/v1"
INDEX_SCHEMA_V5 = "voxelscope/evidence-communication-v5/publication-index/v1"
RECEIPT_SCHEMA_V5 = "voxelscope/evidence-communication-v5/publication-receipt/v1"

AUDIENCE_PROFILE_V5 = "research_technical"
PLAN_MODE_V5 = "deterministic_exact_baseline"
MODEL_GATE_RESULT_V5 = "model_inference_forbidden"
MODEL_ELIGIBILITY_STATUS_V5 = "ineligible_deterministic_dominance"
NOT_APPLICABLE_NO_MODEL_V5 = "not_applicable_no_model"

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_PLACEMENTS = ("before", "after")
_UNIT_KINDS = frozenset({"fact", "caveat", "safety", "optional_context"})
_FILE_TYPES = frozenset(
    {
        "discourse_request",
        "sentence_catalog",
        "implementation_manifest",
        "eligibility_record",
        "optimality_certificate",
        "discourse_plan",
        "rendered_artifact",
        "publication_record",
        "measurement_status",
        "verification_record",
        "publication_index",
    }
)


def _identifier(value: str, name: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise EvidenceError("invalid_identifier", f"{name}: {value!r}")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise EvidenceError("duplicate_record_value", f"{name} must be unique")


def _sorted_unique(values: tuple[str, ...], name: str) -> None:
    _unique(values, name)
    if values != tuple(sorted(values)):
        raise EvidenceError("noncanonical_list_order", f"{name} must be sorted")


def _cycle_exists(nodes: set[str], edges: tuple[tuple[str, str], ...]) -> bool:
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for before, after in edges:
        if after not in outgoing[before]:
            outgoing[before].add(after)
            indegree[after] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.pop(0)
        visited += 1
        for successor in sorted(outgoing[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    return visited != len(nodes)


@dataclass(frozen=True)
class OrderingConstraintV5:
    before_unit_id: str
    after_unit_id: str

    def __post_init__(self) -> None:
        _identifier(self.before_unit_id, "before_unit_id")
        _identifier(self.after_unit_id, "after_unit_id")
        if self.before_unit_id == self.after_unit_id:
            raise EvidenceError("self_ordering_constraint", self.before_unit_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderingConstraintV5:
        value = strict_fields(
            data,
            {"after_unit_id", "before_unit_id"},
            "OrderingConstraintV5",
        )
        return cls(
            require_string(value["before_unit_id"], "before_unit_id"),
            require_string(value["after_unit_id"], "after_unit_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "after_unit_id": self.after_unit_id,
            "before_unit_id": self.before_unit_id,
        }


@dataclass(frozen=True)
class CaveatConstraintV5:
    caveat_unit_id: str
    allowed_anchor_ids: tuple[str, ...]
    allowed_placements: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.caveat_unit_id, "caveat_unit_id")
        if not self.allowed_anchor_ids or not self.allowed_placements:
            raise EvidenceError("empty_caveat_constraint", self.caveat_unit_id)
        for anchor_id in self.allowed_anchor_ids:
            _identifier(anchor_id, "allowed_anchor_id")
        _sorted_unique(self.allowed_anchor_ids, "caveat allowed_anchor_ids")
        _unique(self.allowed_placements, "caveat allowed_placements")
        if any(item not in _PLACEMENTS for item in self.allowed_placements):
            raise EvidenceError("invalid_caveat_placement", self.caveat_unit_id)
        expected = tuple(item for item in _PLACEMENTS if item in self.allowed_placements)
        if self.allowed_placements != expected:
            raise EvidenceError(
                "noncanonical_list_order",
                "allowed_placements must use before, then after",
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaveatConstraintV5:
        value = strict_fields(
            data,
            {"allowed_anchor_ids", "allowed_placements", "caveat_unit_id"},
            "CaveatConstraintV5",
        )
        return cls(
            require_string(value["caveat_unit_id"], "caveat_unit_id"),
            _strings(value["allowed_anchor_ids"], "allowed_anchor_ids"),
            _strings(value["allowed_placements"], "allowed_placements"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_anchor_ids": list(self.allowed_anchor_ids),
            "allowed_placements": list(self.allowed_placements),
            "caveat_unit_id": self.caveat_unit_id,
        }


@dataclass(frozen=True)
class OptionalUnitV5:
    unit_id: str
    utility: int
    estimated_max_bytes: int
    allowed_anchor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "optional unit_id")
        if not 0 <= self.utility <= 1000:
            raise EvidenceError("invalid_optional_utility", self.unit_id)
        if self.estimated_max_bytes <= 0:
            raise EvidenceError("invalid_optional_byte_bound", self.unit_id)
        if not self.allowed_anchor_ids:
            raise EvidenceError("empty_optional_anchor_set", self.unit_id)
        for anchor_id in self.allowed_anchor_ids:
            _identifier(anchor_id, "optional allowed_anchor_id")
        _sorted_unique(self.allowed_anchor_ids, "optional allowed_anchor_ids")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptionalUnitV5:
        value = strict_fields(
            data,
            {"allowed_anchor_ids", "estimated_max_bytes", "unit_id", "utility"},
            "OptionalUnitV5",
        )
        return cls(
            require_string(value["unit_id"], "unit_id"),
            require_int(value["utility"], "utility"),
            require_int(value["estimated_max_bytes"], "estimated_max_bytes"),
            _strings(value["allowed_anchor_ids"], "allowed_anchor_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_anchor_ids": list(self.allowed_anchor_ids),
            "estimated_max_bytes": self.estimated_max_bytes,
            "unit_id": self.unit_id,
            "utility": self.utility,
        }


@dataclass(frozen=True)
class DiscourseBudgetV5:
    max_output_bytes: int
    max_optional_units: int

    def __post_init__(self) -> None:
        if not 1 <= self.max_output_bytes <= 12000:
            raise EvidenceError("invalid_output_budget", str(self.max_output_bytes))
        if not 0 <= self.max_optional_units <= 64:
            raise EvidenceError("invalid_optional_budget", str(self.max_optional_units))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscourseBudgetV5:
        value = strict_fields(
            data,
            {"max_optional_units", "max_output_bytes"},
            "DiscourseBudgetV5",
        )
        return cls(
            require_int(value["max_output_bytes"], "max_output_bytes"),
            require_int(value["max_optional_units"], "max_optional_units"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_optional_units": self.max_optional_units,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True)
class DiscourseRequestV5:
    schema_version: str
    case_id: str
    sentence_catalog_sha256: str
    study_declaration_sha256: str
    renderer_id: str
    audience_profile: str
    mandatory_unit_ids: tuple[str, ...]
    mandatory_fact_ids: tuple[str, ...]
    mandatory_caveat_ids: tuple[str, ...]
    ordering_constraints: tuple[OrderingConstraintV5, ...]
    caveat_constraints: tuple[CaveatConstraintV5, ...]
    optional_units: tuple[OptionalUnitV5, ...]
    budget: DiscourseBudgetV5

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        _identifier(self.case_id, "case_id")
        require_sha256(self.sentence_catalog_sha256, "sentence_catalog_sha256")
        require_sha256(self.study_declaration_sha256, "study_declaration_sha256")
        _identifier(self.renderer_id, "renderer_id")
        if self.audience_profile != AUDIENCE_PROFILE_V5:
            raise EvidenceError("audience_profile_drift", self.audience_profile)
        for name, values in (
            ("mandatory_unit_ids", self.mandatory_unit_ids),
            ("mandatory_fact_ids", self.mandatory_fact_ids),
            ("mandatory_caveat_ids", self.mandatory_caveat_ids),
        ):
            if not values:
                raise EvidenceError("empty_mandatory_set", name)
            for value in values:
                _identifier(value, name)
            _sorted_unique(values, name)
        optional_ids = tuple(item.unit_id for item in self.optional_units)
        _sorted_unique(optional_ids, "optional unit IDs")
        if len(optional_ids) > 64:
            raise EvidenceError("too_many_optional_units", str(len(optional_ids)))
        if not self.caveat_constraints:
            raise EvidenceError("empty_caveat_constraints", self.case_id)
        mandatory = set(self.mandatory_unit_ids)
        if mandatory.intersection(optional_ids):
            raise EvidenceError("optional_mandatory_confusion", self.case_id)
        known = mandatory | set(optional_ids)
        ordering_keys = tuple(
            (item.before_unit_id, item.after_unit_id) for item in self.ordering_constraints
        )
        if len(ordering_keys) != len(set(ordering_keys)):
            raise EvidenceError("duplicate_ordering_constraint", self.case_id)
        if ordering_keys != tuple(sorted(ordering_keys)):
            raise EvidenceError("noncanonical_list_order", "ordering_constraints must be sorted")
        if len(ordering_keys) > 4096:
            raise EvidenceError("too_many_ordering_constraints", str(len(ordering_keys)))
        if any(before not in known or after not in known for before, after in ordering_keys):
            raise EvidenceError("unknown_ordering_unit", self.case_id)
        if _cycle_exists(known, ordering_keys):
            raise EvidenceError("ordering_cycle", self.case_id)
        caveat_ids = tuple(item.caveat_unit_id for item in self.caveat_constraints)
        _sorted_unique(caveat_ids, "caveat constraint IDs")
        if set(caveat_ids) - mandatory:
            raise EvidenceError("nonmandatory_caveat_constraint", self.case_id)
        for constraint in self.caveat_constraints:
            if constraint.caveat_unit_id in constraint.allowed_anchor_ids:
                raise EvidenceError("self_caveat_anchor", constraint.caveat_unit_id)
            if any(anchor not in mandatory for anchor in constraint.allowed_anchor_ids):
                raise EvidenceError("invalid_caveat_anchor", constraint.caveat_unit_id)
        for optional in self.optional_units:
            if any(anchor not in mandatory for anchor in optional.allowed_anchor_ids):
                raise EvidenceError("invalid_optional_anchor", optional.unit_id)
        if self.budget.max_optional_units > len(self.optional_units):
            raise EvidenceError(
                "optional_budget_exceeds_catalog",
                str(self.budget.max_optional_units),
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscourseRequestV5:
        value = strict_fields(
            data,
            {
                "audience_profile",
                "budget",
                "case_id",
                "caveat_constraints",
                "mandatory_caveat_ids",
                "mandatory_fact_ids",
                "mandatory_unit_ids",
                "optional_units",
                "ordering_constraints",
                "renderer_id",
                "schema_version",
                "sentence_catalog_sha256",
                "study_declaration_sha256",
            },
            "DiscourseRequestV5",
        )
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["case_id"], "case_id"),
            require_string(value["sentence_catalog_sha256"], "sentence_catalog_sha256"),
            require_string(value["study_declaration_sha256"], "study_declaration_sha256"),
            require_string(value["renderer_id"], "renderer_id"),
            require_string(value["audience_profile"], "audience_profile"),
            _strings(value["mandatory_unit_ids"], "mandatory_unit_ids"),
            _strings(value["mandatory_fact_ids"], "mandatory_fact_ids"),
            _strings(value["mandatory_caveat_ids"], "mandatory_caveat_ids"),
            tuple(
                OrderingConstraintV5.from_dict(require_object(item, "ordering constraint"))
                for item in require_list(value["ordering_constraints"], "ordering_constraints")
            ),
            tuple(
                CaveatConstraintV5.from_dict(require_object(item, "caveat constraint"))
                for item in require_list(value["caveat_constraints"], "caveat_constraints")
            ),
            tuple(
                OptionalUnitV5.from_dict(require_object(item, "optional unit"))
                for item in require_list(value["optional_units"], "optional_units")
            ),
            DiscourseBudgetV5.from_dict(require_object(value["budget"], "budget")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audience_profile": self.audience_profile,
            "budget": self.budget.to_dict(),
            "case_id": self.case_id,
            "caveat_constraints": [item.to_dict() for item in self.caveat_constraints],
            "mandatory_caveat_ids": list(self.mandatory_caveat_ids),
            "mandatory_fact_ids": list(self.mandatory_fact_ids),
            "mandatory_unit_ids": list(self.mandatory_unit_ids),
            "optional_units": [item.to_dict() for item in self.optional_units],
            "ordering_constraints": [item.to_dict() for item in self.ordering_constraints],
            "renderer_id": self.renderer_id,
            "schema_version": self.schema_version,
            "sentence_catalog_sha256": self.sentence_catalog_sha256,
            "study_declaration_sha256": self.study_declaration_sha256,
        }


@dataclass(frozen=True)
class CaveatAnchorDecisionV5:
    caveat_unit_id: str
    anchor_id: str
    placement: str

    def __post_init__(self) -> None:
        _identifier(self.caveat_unit_id, "caveat_unit_id")
        _identifier(self.anchor_id, "anchor_id")
        if self.placement not in _PLACEMENTS:
            raise EvidenceError("invalid_caveat_placement", self.placement)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaveatAnchorDecisionV5:
        value = strict_fields(
            data,
            {"anchor_id", "caveat_unit_id", "placement"},
            "CaveatAnchorDecisionV5",
        )
        return cls(
            require_string(value["caveat_unit_id"], "caveat_unit_id"),
            require_string(value["anchor_id"], "anchor_id"),
            require_string(value["placement"], "placement"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "anchor_id": self.anchor_id,
            "caveat_unit_id": self.caveat_unit_id,
            "placement": self.placement,
        }


@dataclass(frozen=True)
class DiscoursePlanV5:
    schema_version: str
    request_sha256: str
    plan_mode: str
    ordered_unit_ids: tuple[str, ...]
    selected_optional_unit_ids: tuple[str, ...]
    caveat_anchor_decisions: tuple[CaveatAnchorDecisionV5, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        require_sha256(self.request_sha256, "request_sha256")
        if self.plan_mode != PLAN_MODE_V5:
            raise EvidenceError("invalid_plan_mode", self.plan_mode)
        if not self.ordered_unit_ids:
            raise EvidenceError("empty_plan", self.request_sha256)
        for unit_id in self.ordered_unit_ids:
            _identifier(unit_id, "ordered_unit_id")
        _unique(self.ordered_unit_ids, "ordered_unit_ids")
        for unit_id in self.selected_optional_unit_ids:
            _identifier(unit_id, "selected_optional_unit_id")
        _sorted_unique(self.selected_optional_unit_ids, "selected_optional_unit_ids")
        if len(self.selected_optional_unit_ids) > 64:
            raise EvidenceError(
                "too_many_selected_optional_units",
                str(len(self.selected_optional_unit_ids)),
            )
        decision_keys = tuple(
            (item.caveat_unit_id, item.anchor_id, item.placement)
            for item in self.caveat_anchor_decisions
        )
        if len(decision_keys) != len(set(decision_keys)):
            raise EvidenceError("duplicate_caveat_decision", self.request_sha256)
        if decision_keys != tuple(sorted(decision_keys)):
            raise EvidenceError(
                "noncanonical_list_order",
                "caveat_anchor_decisions must be sorted",
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoursePlanV5:
        value = strict_fields(
            data,
            {
                "caveat_anchor_decisions",
                "ordered_unit_ids",
                "plan_mode",
                "request_sha256",
                "schema_version",
                "selected_optional_unit_ids",
            },
            "DiscoursePlanV5",
        )
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["plan_mode"], "plan_mode"),
            _strings(value["ordered_unit_ids"], "ordered_unit_ids"),
            _strings(value["selected_optional_unit_ids"], "selected_optional_unit_ids"),
            tuple(
                CaveatAnchorDecisionV5.from_dict(require_object(item, "caveat decision"))
                for item in require_list(
                    value["caveat_anchor_decisions"],
                    "caveat_anchor_decisions",
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "caveat_anchor_decisions": [item.to_dict() for item in self.caveat_anchor_decisions],
            "ordered_unit_ids": list(self.ordered_unit_ids),
            "plan_mode": self.plan_mode,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "selected_optional_unit_ids": list(self.selected_optional_unit_ids),
        }


@dataclass(frozen=True)
class SentenceUnitV5:
    unit_id: str
    unit_kind: str
    canonical_sentences: tuple[str, ...]
    citation_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    caveat_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "sentence unit_id")
        if self.unit_kind not in _UNIT_KINDS:
            raise EvidenceError("invalid_sentence_unit_kind", self.unit_kind)
        if not self.canonical_sentences:
            raise EvidenceError("empty_canonical_sentences", self.unit_id)
        for sentence in self.canonical_sentences:
            if not sentence or sentence != sentence.strip() or "\n" in sentence:
                raise EvidenceError("invalid_canonical_sentence", self.unit_id)
            try:
                sentence.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise EvidenceError("invalid_canonical_sentence", self.unit_id) from exc
        for name, identifiers in (
            ("citation_ids", self.citation_ids),
            ("fact_ids", self.fact_ids),
            ("caveat_ids", self.caveat_ids),
        ):
            for identifier in identifiers:
                _identifier(identifier, name)
            _sorted_unique(identifiers, f"{self.unit_id} {name}")
        if not self.citation_ids:
            raise EvidenceError("uncited_sentence_unit", self.unit_id)
        if self.unit_kind == "fact" and (not self.fact_ids or self.caveat_ids):
            raise EvidenceError("invalid_fact_unit", self.unit_id)
        if self.unit_kind in {"caveat", "safety"} and (not self.caveat_ids or self.fact_ids):
            raise EvidenceError("invalid_caveat_unit", self.unit_id)
        if self.unit_kind == "optional_context" and (self.fact_ids or self.caveat_ids):
            raise EvidenceError("invalid_optional_context_unit", self.unit_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SentenceUnitV5:
        value = strict_fields(
            data,
            {
                "canonical_sentences",
                "caveat_ids",
                "citation_ids",
                "fact_ids",
                "unit_id",
                "unit_kind",
            },
            "SentenceUnitV5",
        )
        return cls(
            require_string(value["unit_id"], "unit_id"),
            require_string(value["unit_kind"], "unit_kind"),
            _strings(value["canonical_sentences"], "canonical_sentences"),
            _strings(value["citation_ids"], "citation_ids"),
            _strings(value["fact_ids"], "fact_ids"),
            _strings(value["caveat_ids"], "caveat_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_sentences": list(self.canonical_sentences),
            "caveat_ids": list(self.caveat_ids),
            "citation_ids": list(self.citation_ids),
            "fact_ids": list(self.fact_ids),
            "unit_id": self.unit_id,
            "unit_kind": self.unit_kind,
        }


@dataclass(frozen=True)
class SentenceCatalogV5:
    schema_version: str
    catalog_id: str
    citation_ids: tuple[str, ...]
    units: tuple[SentenceUnitV5, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CATALOG_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        _identifier(self.catalog_id, "catalog_id")
        if not self.citation_ids or not self.units:
            raise EvidenceError("empty_sentence_catalog", self.catalog_id)
        for citation_id in self.citation_ids:
            _identifier(citation_id, "catalog citation_id")
        _sorted_unique(self.citation_ids, "catalog citation_ids")
        unit_ids = tuple(item.unit_id for item in self.units)
        _sorted_unique(unit_ids, "catalog unit IDs")
        known_citations = set(self.citation_ids)
        if any(
            citation_id not in known_citations
            for unit in self.units
            for citation_id in unit.citation_ids
        ):
            raise EvidenceError("unknown_catalog_citation", self.catalog_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SentenceCatalogV5:
        value = strict_fields(
            data,
            {"catalog_id", "citation_ids", "schema_version", "units"},
            "SentenceCatalogV5",
        )
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["catalog_id"], "catalog_id"),
            _strings(value["citation_ids"], "citation_ids"),
            tuple(
                SentenceUnitV5.from_dict(require_object(item, "sentence unit"))
                for item in require_list(value["units"], "units")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "citation_ids": list(self.citation_ids),
            "schema_version": self.schema_version,
            "units": [item.to_dict() for item in self.units],
        }


@dataclass(frozen=True)
class SourceCustodyV5:
    source_revision: str
    source_root_tree: str
    implementation_manifest_sha256: str
    source_tree_state: str

    def __post_init__(self) -> None:
        if _GIT_REVISION.fullmatch(self.source_revision) is None:
            raise EvidenceError("invalid_source_revision", self.source_revision)
        if _GIT_REVISION.fullmatch(self.source_root_tree) is None:
            raise EvidenceError("invalid_source_root_tree", self.source_root_tree)
        require_sha256(
            self.implementation_manifest_sha256,
            "implementation_manifest_sha256",
        )
        if self.source_tree_state != "clean":
            raise EvidenceError("source_tree_not_clean", self.source_tree_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceCustodyV5:
        value = strict_fields(
            data,
            {
                "implementation_manifest_sha256",
                "source_revision",
                "source_root_tree",
                "source_tree_state",
            },
            "SourceCustodyV5",
        )
        return cls(
            require_string(value["source_revision"], "source_revision"),
            require_string(value["source_root_tree"], "source_root_tree"),
            require_string(
                value["implementation_manifest_sha256"],
                "implementation_manifest_sha256",
            ),
            require_string(value["source_tree_state"], "source_tree_state"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "implementation_manifest_sha256": self.implementation_manifest_sha256,
            "source_revision": self.source_revision,
            "source_root_tree": self.source_root_tree,
            "source_tree_state": self.source_tree_state,
        }


@dataclass(frozen=True)
class ImplementationFileV5:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        require_sha256(self.sha256, "implementation file sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationFileV5:
        value = strict_fields(
            data,
            {"path", "sha256"},
            "ImplementationFileV5",
        )
        return cls(
            require_string(value["path"], "path"),
            require_string(value["sha256"], "sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ImplementationManifestV5:
    schema_version: str
    files: tuple[ImplementationFileV5, ...]
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != IMPLEMENTATION_MANIFEST_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        paths = tuple(item.path for item in self.files)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise EvidenceError(
                "invalid_implementation_manifest_files",
                "implementation paths must be nonempty, sorted, and unique",
            )
        if self.terminal_state != "closed":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationManifestV5:
        value = strict_fields(
            data,
            {"files", "schema_version", "terminal_state"},
            "ImplementationManifestV5",
        )
        return cls(
            require_string(value["schema_version"], "schema_version"),
            tuple(
                ImplementationFileV5.from_dict(require_object(item, "implementation file"))
                for item in require_list(value["files"], "files")
            ),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [item.to_dict() for item in self.files],
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class EligibilityRecordV5:
    schema_version: str
    request_sha256: str
    design_contract_sha256: str
    request_schema_sha256: str
    plan_schema_sha256: str
    publication_schema_sha256: str
    sentence_catalog_sha256: str
    renderer_sha256: str
    optimizer_sha256: str
    custody: SourceCustodyV5
    gate_result: str
    model_eligibility_status: str
    evaluated_before_model_actions: bool
    model_candidate_action_count: int
    reason_code: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != ELIGIBILITY_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        for name in (
            "request_sha256",
            "design_contract_sha256",
            "request_schema_sha256",
            "plan_schema_sha256",
            "publication_schema_sha256",
            "sentence_catalog_sha256",
            "renderer_sha256",
            "optimizer_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.gate_result != MODEL_GATE_RESULT_V5:
            raise EvidenceError("model_eligibility_drift", self.gate_result)
        if self.model_eligibility_status != MODEL_ELIGIBILITY_STATUS_V5:
            raise EvidenceError("model_eligibility_drift", self.model_eligibility_status)
        if not self.evaluated_before_model_actions or self.model_candidate_action_count != 0:
            raise EvidenceError("model_action_before_eligibility", self.request_sha256)
        if self.reason_code != "exact_deterministic_objectives_exhaust_current_value_endpoints":
            raise EvidenceError("model_eligibility_reason_drift", self.reason_code)
        if self.terminal_state != "eligibility_checked":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EligibilityRecordV5:
        value = strict_fields(
            data,
            {
                "custody",
                "design_contract_sha256",
                "evaluated_before_model_actions",
                "gate_result",
                "model_candidate_action_count",
                "model_eligibility_status",
                "optimizer_sha256",
                "plan_schema_sha256",
                "publication_schema_sha256",
                "reason_code",
                "renderer_sha256",
                "request_schema_sha256",
                "request_sha256",
                "schema_version",
                "sentence_catalog_sha256",
                "terminal_state",
            },
            "EligibilityRecordV5",
        )
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["design_contract_sha256"], "design_contract_sha256"),
            require_string(value["request_schema_sha256"], "request_schema_sha256"),
            require_string(value["plan_schema_sha256"], "plan_schema_sha256"),
            require_string(value["publication_schema_sha256"], "publication_schema_sha256"),
            require_string(value["sentence_catalog_sha256"], "sentence_catalog_sha256"),
            require_string(value["renderer_sha256"], "renderer_sha256"),
            require_string(value["optimizer_sha256"], "optimizer_sha256"),
            SourceCustodyV5.from_dict(require_object(value["custody"], "custody")),
            require_string(value["gate_result"], "gate_result"),
            require_string(value["model_eligibility_status"], "model_eligibility_status"),
            require_bool(
                value["evaluated_before_model_actions"],
                "evaluated_before_model_actions",
            ),
            require_int(value["model_candidate_action_count"], "model_candidate_action_count"),
            require_string(value["reason_code"], "reason_code"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "custody": self.custody.to_dict(),
            "design_contract_sha256": self.design_contract_sha256,
            "evaluated_before_model_actions": self.evaluated_before_model_actions,
            "gate_result": self.gate_result,
            "model_candidate_action_count": self.model_candidate_action_count,
            "model_eligibility_status": self.model_eligibility_status,
            "optimizer_sha256": self.optimizer_sha256,
            "plan_schema_sha256": self.plan_schema_sha256,
            "publication_schema_sha256": self.publication_schema_sha256,
            "reason_code": self.reason_code,
            "renderer_sha256": self.renderer_sha256,
            "request_schema_sha256": self.request_schema_sha256,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "sentence_catalog_sha256": self.sentence_catalog_sha256,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class OptimalityCertificateV5:
    schema_version: str
    request_sha256: str
    sentence_catalog_sha256: str
    renderer_sha256: str
    optimizer_sha256: str
    plan_sha256: str
    algorithm: str
    decision_key_version: str
    exhaustive: bool
    max_candidate_evaluations: int
    optional_subsets_examined: int
    caveat_assignments_examined: int
    topological_orders_examined: int
    feasible_candidates: int
    budget_rejections: int
    structural_rejections: int
    search_space_sha256: str
    winner_decision_key_sha256: str
    winner_optional_utility: int
    winner_emitted_utf8_bytes: int
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != OPTIMALITY_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        for name in (
            "request_sha256",
            "sentence_catalog_sha256",
            "renderer_sha256",
            "optimizer_sha256",
            "plan_sha256",
            "search_space_sha256",
            "winner_decision_key_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.algorithm != "bounded_complete_enumeration_v1":
            raise EvidenceError("unsupported_optimizer", self.algorithm)
        if self.decision_key_version != "canonical_decision_key_v1":
            raise EvidenceError("unsupported_decision_key", self.decision_key_version)
        if not self.exhaustive:
            raise EvidenceError("nonexhaustive_optimizer_forbidden", self.request_sha256)
        counts = (
            self.max_candidate_evaluations,
            self.optional_subsets_examined,
            self.caveat_assignments_examined,
            self.topological_orders_examined,
            self.feasible_candidates,
            self.budget_rejections,
            self.structural_rejections,
            self.winner_optional_utility,
            self.winner_emitted_utf8_bytes,
        )
        if any(value < 0 for value in counts) or self.max_candidate_evaluations <= 0:
            raise EvidenceError("invalid_optimality_count", self.request_sha256)
        if self.feasible_candidates <= 0 or self.winner_emitted_utf8_bytes <= 0:
            raise EvidenceError("missing_feasible_winner", self.request_sha256)
        if self.terminal_state != "proved":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimalityCertificateV5:
        fields = {
            "algorithm",
            "budget_rejections",
            "caveat_assignments_examined",
            "decision_key_version",
            "exhaustive",
            "feasible_candidates",
            "max_candidate_evaluations",
            "optimizer_sha256",
            "optional_subsets_examined",
            "plan_sha256",
            "renderer_sha256",
            "request_sha256",
            "schema_version",
            "search_space_sha256",
            "sentence_catalog_sha256",
            "structural_rejections",
            "terminal_state",
            "topological_orders_examined",
            "winner_decision_key_sha256",
            "winner_emitted_utf8_bytes",
            "winner_optional_utility",
        }
        value = strict_fields(data, fields, "OptimalityCertificateV5")
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["sentence_catalog_sha256"], "sentence_catalog_sha256"),
            require_string(value["renderer_sha256"], "renderer_sha256"),
            require_string(value["optimizer_sha256"], "optimizer_sha256"),
            require_string(value["plan_sha256"], "plan_sha256"),
            require_string(value["algorithm"], "algorithm"),
            require_string(value["decision_key_version"], "decision_key_version"),
            require_bool(value["exhaustive"], "exhaustive"),
            require_int(value["max_candidate_evaluations"], "max_candidate_evaluations"),
            require_int(value["optional_subsets_examined"], "optional_subsets_examined"),
            require_int(
                value["caveat_assignments_examined"],
                "caveat_assignments_examined",
            ),
            require_int(value["topological_orders_examined"], "topological_orders_examined"),
            require_int(value["feasible_candidates"], "feasible_candidates"),
            require_int(value["budget_rejections"], "budget_rejections"),
            require_int(value["structural_rejections"], "structural_rejections"),
            require_string(value["search_space_sha256"], "search_space_sha256"),
            require_string(
                value["winner_decision_key_sha256"],
                "winner_decision_key_sha256",
            ),
            require_int(value["winner_optional_utility"], "winner_optional_utility"),
            require_int(
                value["winner_emitted_utf8_bytes"],
                "winner_emitted_utf8_bytes",
            ),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "budget_rejections": self.budget_rejections,
            "caveat_assignments_examined": self.caveat_assignments_examined,
            "decision_key_version": self.decision_key_version,
            "exhaustive": self.exhaustive,
            "feasible_candidates": self.feasible_candidates,
            "max_candidate_evaluations": self.max_candidate_evaluations,
            "optimizer_sha256": self.optimizer_sha256,
            "optional_subsets_examined": self.optional_subsets_examined,
            "plan_sha256": self.plan_sha256,
            "renderer_sha256": self.renderer_sha256,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "search_space_sha256": self.search_space_sha256,
            "sentence_catalog_sha256": self.sentence_catalog_sha256,
            "structural_rejections": self.structural_rejections,
            "terminal_state": self.terminal_state,
            "topological_orders_examined": self.topological_orders_examined,
            "winner_decision_key_sha256": self.winner_decision_key_sha256,
            "winner_emitted_utf8_bytes": self.winner_emitted_utf8_bytes,
            "winner_optional_utility": self.winner_optional_utility,
        }


@dataclass(frozen=True)
class CoverageCountV5:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0 or self.numerator != self.denominator:
            raise EvidenceError(
                "incomplete_publication_coverage",
                f"{self.numerator}/{self.denominator}",
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageCountV5:
        value = strict_fields(data, {"denominator", "numerator"}, "CoverageCountV5")
        return cls(
            require_int(value["numerator"], "numerator"),
            require_int(value["denominator"], "denominator"),
        )

    def to_dict(self) -> dict[str, int]:
        return {"denominator": self.denominator, "numerator": self.numerator}


@dataclass(frozen=True)
class RenderedArtifactV5:
    schema_version: str
    request_sha256: str
    plan_sha256: str
    sentence_catalog_sha256: str
    renderer_sha256: str
    audience_profile: str
    ordered_unit_ids: tuple[str, ...]
    emitted_fact_ids: tuple[str, ...]
    emitted_caveat_ids: tuple[str, ...]
    emitted_citation_ids: tuple[str, ...]
    mandatory_fact_coverage: CoverageCountV5
    mandatory_caveat_coverage: CoverageCountV5
    citation_validity: CoverageCountV5
    prose: str
    output_sha256: str
    emitted_utf8_bytes: int
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != RENDERED_ARTIFACT_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        for name in (
            "request_sha256",
            "plan_sha256",
            "sentence_catalog_sha256",
            "renderer_sha256",
            "output_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.audience_profile != AUDIENCE_PROFILE_V5:
            raise EvidenceError("audience_profile_drift", self.audience_profile)
        for name, values in (
            ("ordered_unit_ids", self.ordered_unit_ids),
            ("emitted_fact_ids", self.emitted_fact_ids),
            ("emitted_caveat_ids", self.emitted_caveat_ids),
            ("emitted_citation_ids", self.emitted_citation_ids),
        ):
            if not values:
                raise EvidenceError("empty_rendered_identity", name)
            _unique(values, name)
        for values, name in (
            (self.emitted_fact_ids, "emitted_fact_ids"),
            (self.emitted_caveat_ids, "emitted_caveat_ids"),
            (self.emitted_citation_ids, "emitted_citation_ids"),
        ):
            if values != tuple(sorted(values)):
                raise EvidenceError("noncanonical_list_order", name)
        try:
            encoded = self.prose.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise EvidenceError("invalid_rendered_utf8", self.request_sha256) from exc
        if not encoded or len(encoded) != self.emitted_utf8_bytes:
            raise EvidenceError("rendered_byte_count_mismatch", self.request_sha256)
        from .canonical import sha256_bytes

        if sha256_bytes(encoded) != self.output_sha256:
            raise EvidenceError("rendered_output_digest_mismatch", self.request_sha256)
        if self.terminal_state != "verified":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderedArtifactV5:
        fields = {
            "audience_profile",
            "citation_validity",
            "emitted_caveat_ids",
            "emitted_citation_ids",
            "emitted_fact_ids",
            "emitted_utf8_bytes",
            "mandatory_caveat_coverage",
            "mandatory_fact_coverage",
            "ordered_unit_ids",
            "output_sha256",
            "plan_sha256",
            "prose",
            "renderer_sha256",
            "request_sha256",
            "schema_version",
            "sentence_catalog_sha256",
            "terminal_state",
        }
        value = strict_fields(data, fields, "RenderedArtifactV5")
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["plan_sha256"], "plan_sha256"),
            require_string(value["sentence_catalog_sha256"], "sentence_catalog_sha256"),
            require_string(value["renderer_sha256"], "renderer_sha256"),
            require_string(value["audience_profile"], "audience_profile"),
            _strings(value["ordered_unit_ids"], "ordered_unit_ids"),
            _strings(value["emitted_fact_ids"], "emitted_fact_ids"),
            _strings(value["emitted_caveat_ids"], "emitted_caveat_ids"),
            _strings(value["emitted_citation_ids"], "emitted_citation_ids"),
            CoverageCountV5.from_dict(
                require_object(value["mandatory_fact_coverage"], "mandatory_fact_coverage")
            ),
            CoverageCountV5.from_dict(
                require_object(
                    value["mandatory_caveat_coverage"],
                    "mandatory_caveat_coverage",
                )
            ),
            CoverageCountV5.from_dict(
                require_object(value["citation_validity"], "citation_validity")
            ),
            require_string(value["prose"], "prose"),
            require_string(value["output_sha256"], "output_sha256"),
            require_int(value["emitted_utf8_bytes"], "emitted_utf8_bytes"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audience_profile": self.audience_profile,
            "citation_validity": self.citation_validity.to_dict(),
            "emitted_caveat_ids": list(self.emitted_caveat_ids),
            "emitted_citation_ids": list(self.emitted_citation_ids),
            "emitted_fact_ids": list(self.emitted_fact_ids),
            "emitted_utf8_bytes": self.emitted_utf8_bytes,
            "mandatory_caveat_coverage": self.mandatory_caveat_coverage.to_dict(),
            "mandatory_fact_coverage": self.mandatory_fact_coverage.to_dict(),
            "ordered_unit_ids": list(self.ordered_unit_ids),
            "output_sha256": self.output_sha256,
            "plan_sha256": self.plan_sha256,
            "prose": self.prose,
            "renderer_sha256": self.renderer_sha256,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "sentence_catalog_sha256": self.sentence_catalog_sha256,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class PublicationRecordV5:
    schema_version: str
    run_id: str
    request_sha256: str
    audience_profile: str
    model_eligibility_status: str
    eligibility_record_sha256: str
    optimizer_proof_sha256: str
    baseline_plan_sha256: str
    baseline_output_sha256: str
    baseline_optional_utility: int
    baseline_emitted_utf8_bytes: int
    renderer_sha256: str
    mandatory_fact_coverage: CoverageCountV5
    mandatory_caveat_coverage: CoverageCountV5
    citation_validity: CoverageCountV5
    protocol_validity: str
    model_outcome: str
    model_input_sha256: None
    model_output_sha256: None
    model_plan_sha256: None
    published_variant: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != PUBLICATION_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        _identifier(self.run_id, "run_id")
        for name in (
            "request_sha256",
            "eligibility_record_sha256",
            "optimizer_proof_sha256",
            "baseline_plan_sha256",
            "baseline_output_sha256",
            "renderer_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.audience_profile != AUDIENCE_PROFILE_V5:
            raise EvidenceError("audience_profile_drift", self.audience_profile)
        if self.model_eligibility_status != MODEL_ELIGIBILITY_STATUS_V5:
            raise EvidenceError("model_eligibility_drift", self.model_eligibility_status)
        if self.baseline_optional_utility < 0 or self.baseline_emitted_utf8_bytes <= 0:
            raise EvidenceError("invalid_publication_objective", self.run_id)
        if self.protocol_validity != "valid":
            raise EvidenceError("invalid_protocol", self.protocol_validity)
        if self.model_outcome != "ineligible_not_requested":
            raise EvidenceError("unexpected_model_outcome", self.model_outcome)
        if (
            self.model_input_sha256 is not None
            or self.model_output_sha256 is not None
            or self.model_plan_sha256 is not None
        ):
            raise EvidenceError("model_artifact_forbidden", self.run_id)
        if self.published_variant != PLAN_MODE_V5:
            raise EvidenceError("invalid_published_variant", self.published_variant)
        if self.terminal_state != "closed":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublicationRecordV5:
        fields = {
            "audience_profile",
            "baseline_emitted_utf8_bytes",
            "baseline_optional_utility",
            "baseline_output_sha256",
            "baseline_plan_sha256",
            "citation_validity",
            "eligibility_record_sha256",
            "mandatory_caveat_coverage",
            "mandatory_fact_coverage",
            "model_eligibility_status",
            "model_input_sha256",
            "model_outcome",
            "model_output_sha256",
            "model_plan_sha256",
            "optimizer_proof_sha256",
            "protocol_validity",
            "published_variant",
            "renderer_sha256",
            "request_sha256",
            "run_id",
            "schema_version",
            "terminal_state",
        }
        value = strict_fields(data, fields, "PublicationRecordV5")
        for nullable in ("model_input_sha256", "model_output_sha256", "model_plan_sha256"):
            if value[nullable] is not None:
                raise EvidenceError("model_artifact_forbidden", nullable)
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["run_id"], "run_id"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["audience_profile"], "audience_profile"),
            require_string(value["model_eligibility_status"], "model_eligibility_status"),
            require_string(value["eligibility_record_sha256"], "eligibility_record_sha256"),
            require_string(value["optimizer_proof_sha256"], "optimizer_proof_sha256"),
            require_string(value["baseline_plan_sha256"], "baseline_plan_sha256"),
            require_string(value["baseline_output_sha256"], "baseline_output_sha256"),
            require_int(value["baseline_optional_utility"], "baseline_optional_utility"),
            require_int(
                value["baseline_emitted_utf8_bytes"],
                "baseline_emitted_utf8_bytes",
            ),
            require_string(value["renderer_sha256"], "renderer_sha256"),
            CoverageCountV5.from_dict(
                require_object(value["mandatory_fact_coverage"], "mandatory_fact_coverage")
            ),
            CoverageCountV5.from_dict(
                require_object(
                    value["mandatory_caveat_coverage"],
                    "mandatory_caveat_coverage",
                )
            ),
            CoverageCountV5.from_dict(
                require_object(value["citation_validity"], "citation_validity")
            ),
            require_string(value["protocol_validity"], "protocol_validity"),
            require_string(value["model_outcome"], "model_outcome"),
            None,
            None,
            None,
            require_string(value["published_variant"], "published_variant"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audience_profile": self.audience_profile,
            "baseline_emitted_utf8_bytes": self.baseline_emitted_utf8_bytes,
            "baseline_optional_utility": self.baseline_optional_utility,
            "baseline_output_sha256": self.baseline_output_sha256,
            "baseline_plan_sha256": self.baseline_plan_sha256,
            "citation_validity": self.citation_validity.to_dict(),
            "eligibility_record_sha256": self.eligibility_record_sha256,
            "mandatory_caveat_coverage": self.mandatory_caveat_coverage.to_dict(),
            "mandatory_fact_coverage": self.mandatory_fact_coverage.to_dict(),
            "model_eligibility_status": self.model_eligibility_status,
            "model_input_sha256": self.model_input_sha256,
            "model_outcome": self.model_outcome,
            "model_output_sha256": self.model_output_sha256,
            "model_plan_sha256": self.model_plan_sha256,
            "optimizer_proof_sha256": self.optimizer_proof_sha256,
            "protocol_validity": self.protocol_validity,
            "published_variant": self.published_variant,
            "renderer_sha256": self.renderer_sha256,
            "request_sha256": self.request_sha256,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class MeasurementStatusV5:
    schema_version: str
    request_sha256: str
    plan_sha256: str
    optimality_certificate_sha256: str
    deterministic_candidate_evaluations: int
    deterministic_feasible_candidates: int
    deterministic_planner_wall_clock_ms: str
    model_inference: str
    model_input_tokens: str
    model_output_tokens: str
    model_latency_ms: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != MEASUREMENT_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        for name in ("request_sha256", "plan_sha256", "optimality_certificate_sha256"):
            require_sha256(getattr(self, name), name)
        if (
            self.deterministic_candidate_evaluations <= 0
            or self.deterministic_feasible_candidates <= 0
        ):
            raise EvidenceError("invalid_planner_measurement", self.request_sha256)
        if self.deterministic_planner_wall_clock_ms != (
            "not_recorded_to_preserve_byte_for_byte_replay"
        ):
            raise EvidenceError(
                "invalid_planner_measurement",
                self.deterministic_planner_wall_clock_ms,
            )
        if any(
            value != NOT_APPLICABLE_NO_MODEL_V5
            for value in (
                self.model_inference,
                self.model_input_tokens,
                self.model_output_tokens,
                self.model_latency_ms,
            )
        ):
            raise EvidenceError("model_measurement_forbidden", self.request_sha256)
        if self.terminal_state != "completed":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementStatusV5:
        fields = {
            "deterministic_candidate_evaluations",
            "deterministic_feasible_candidates",
            "deterministic_planner_wall_clock_ms",
            "model_inference",
            "model_input_tokens",
            "model_latency_ms",
            "model_output_tokens",
            "optimality_certificate_sha256",
            "plan_sha256",
            "request_sha256",
            "schema_version",
            "terminal_state",
        }
        value = strict_fields(data, fields, "MeasurementStatusV5")
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["plan_sha256"], "plan_sha256"),
            require_string(
                value["optimality_certificate_sha256"],
                "optimality_certificate_sha256",
            ),
            require_int(
                value["deterministic_candidate_evaluations"],
                "deterministic_candidate_evaluations",
            ),
            require_int(
                value["deterministic_feasible_candidates"],
                "deterministic_feasible_candidates",
            ),
            require_string(
                value["deterministic_planner_wall_clock_ms"],
                "deterministic_planner_wall_clock_ms",
            ),
            require_string(value["model_inference"], "model_inference"),
            require_string(value["model_input_tokens"], "model_input_tokens"),
            require_string(value["model_output_tokens"], "model_output_tokens"),
            require_string(value["model_latency_ms"], "model_latency_ms"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic_candidate_evaluations": (self.deterministic_candidate_evaluations),
            "deterministic_feasible_candidates": self.deterministic_feasible_candidates,
            "deterministic_planner_wall_clock_ms": (self.deterministic_planner_wall_clock_ms),
            "model_inference": self.model_inference,
            "model_input_tokens": self.model_input_tokens,
            "model_latency_ms": self.model_latency_ms,
            "model_output_tokens": self.model_output_tokens,
            "optimality_certificate_sha256": self.optimality_certificate_sha256,
            "plan_sha256": self.plan_sha256,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class VerificationRecordV5:
    schema_version: str
    request_sha256: str
    eligibility_record_sha256: str
    optimality_certificate_sha256: str
    plan_sha256: str
    rendered_artifact_sha256: str
    publication_record_sha256: str
    exact_schema_gate: str
    identity_gate: str
    mandatory_coverage_gate: str
    citation_gate: str
    protocol_gate: str
    byte_budget_gate: str
    optimality_gate: str
    offline_replay_gate: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != VERIFICATION_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        for name in (
            "request_sha256",
            "eligibility_record_sha256",
            "optimality_certificate_sha256",
            "plan_sha256",
            "rendered_artifact_sha256",
            "publication_record_sha256",
        ):
            require_sha256(getattr(self, name), name)
        gates = (
            self.exact_schema_gate,
            self.identity_gate,
            self.mandatory_coverage_gate,
            self.citation_gate,
            self.protocol_gate,
            self.byte_budget_gate,
            self.optimality_gate,
            self.offline_replay_gate,
        )
        if any(value != "passed" for value in gates):
            raise EvidenceError("verification_gate_failed", self.request_sha256)
        if self.terminal_state != "publication_verified":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationRecordV5:
        fields = {
            "byte_budget_gate",
            "citation_gate",
            "eligibility_record_sha256",
            "exact_schema_gate",
            "identity_gate",
            "mandatory_coverage_gate",
            "offline_replay_gate",
            "optimality_certificate_sha256",
            "optimality_gate",
            "plan_sha256",
            "protocol_gate",
            "publication_record_sha256",
            "rendered_artifact_sha256",
            "request_sha256",
            "schema_version",
            "terminal_state",
        }
        value = strict_fields(data, fields, "VerificationRecordV5")
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["eligibility_record_sha256"], "eligibility_record_sha256"),
            require_string(
                value["optimality_certificate_sha256"],
                "optimality_certificate_sha256",
            ),
            require_string(value["plan_sha256"], "plan_sha256"),
            require_string(value["rendered_artifact_sha256"], "rendered_artifact_sha256"),
            require_string(value["publication_record_sha256"], "publication_record_sha256"),
            require_string(value["exact_schema_gate"], "exact_schema_gate"),
            require_string(value["identity_gate"], "identity_gate"),
            require_string(value["mandatory_coverage_gate"], "mandatory_coverage_gate"),
            require_string(value["citation_gate"], "citation_gate"),
            require_string(value["protocol_gate"], "protocol_gate"),
            require_string(value["byte_budget_gate"], "byte_budget_gate"),
            require_string(value["optimality_gate"], "optimality_gate"),
            require_string(value["offline_replay_gate"], "offline_replay_gate"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "byte_budget_gate": self.byte_budget_gate,
            "citation_gate": self.citation_gate,
            "eligibility_record_sha256": self.eligibility_record_sha256,
            "exact_schema_gate": self.exact_schema_gate,
            "identity_gate": self.identity_gate,
            "mandatory_coverage_gate": self.mandatory_coverage_gate,
            "offline_replay_gate": self.offline_replay_gate,
            "optimality_certificate_sha256": self.optimality_certificate_sha256,
            "optimality_gate": self.optimality_gate,
            "plan_sha256": self.plan_sha256,
            "protocol_gate": self.protocol_gate,
            "publication_record_sha256": self.publication_record_sha256,
            "rendered_artifact_sha256": self.rendered_artifact_sha256,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class PublicationFileV5:
    artifact_type: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.artifact_type not in _FILE_TYPES:
            raise EvidenceError("invalid_publication_artifact_type", self.artifact_type)
        safe_relative_path(self.path)
        require_sha256(self.sha256, "publication file sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublicationFileV5:
        value = strict_fields(
            data,
            {"artifact_type", "path", "sha256"},
            "PublicationFileV5",
        )
        return cls(
            require_string(value["artifact_type"], "artifact_type"),
            require_string(value["path"], "path"),
            require_string(value["sha256"], "sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PublicationIndexV5:
    schema_version: str
    run_id: str
    custody: SourceCustodyV5
    request_sha256: str
    eligibility_record_sha256: str
    optimality_certificate_sha256: str
    plan_sha256: str
    rendered_artifact_sha256: str
    publication_record_sha256: str
    measurement_status_sha256: str
    verification_record_sha256: str
    files: tuple[PublicationFileV5, ...]
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != INDEX_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        _identifier(self.run_id, "run_id")
        for name in (
            "request_sha256",
            "eligibility_record_sha256",
            "optimality_certificate_sha256",
            "plan_sha256",
            "rendered_artifact_sha256",
            "publication_record_sha256",
            "measurement_status_sha256",
            "verification_record_sha256",
        ):
            require_sha256(getattr(self, name), name)
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise EvidenceError("invalid_publication_index_files", self.run_id)
        if self.terminal_state != "publication_verified":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublicationIndexV5:
        fields = {
            "custody",
            "eligibility_record_sha256",
            "files",
            "measurement_status_sha256",
            "optimality_certificate_sha256",
            "plan_sha256",
            "publication_record_sha256",
            "rendered_artifact_sha256",
            "request_sha256",
            "run_id",
            "schema_version",
            "terminal_state",
            "verification_record_sha256",
        }
        value = strict_fields(data, fields, "PublicationIndexV5")
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["run_id"], "run_id"),
            SourceCustodyV5.from_dict(require_object(value["custody"], "custody")),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["eligibility_record_sha256"], "eligibility_record_sha256"),
            require_string(
                value["optimality_certificate_sha256"],
                "optimality_certificate_sha256",
            ),
            require_string(value["plan_sha256"], "plan_sha256"),
            require_string(value["rendered_artifact_sha256"], "rendered_artifact_sha256"),
            require_string(value["publication_record_sha256"], "publication_record_sha256"),
            require_string(value["measurement_status_sha256"], "measurement_status_sha256"),
            require_string(
                value["verification_record_sha256"],
                "verification_record_sha256",
            ),
            tuple(
                PublicationFileV5.from_dict(require_object(item, "publication file"))
                for item in require_list(value["files"], "files")
            ),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "custody": self.custody.to_dict(),
            "eligibility_record_sha256": self.eligibility_record_sha256,
            "files": [item.to_dict() for item in self.files],
            "measurement_status_sha256": self.measurement_status_sha256,
            "optimality_certificate_sha256": self.optimality_certificate_sha256,
            "plan_sha256": self.plan_sha256,
            "publication_record_sha256": self.publication_record_sha256,
            "rendered_artifact_sha256": self.rendered_artifact_sha256,
            "request_sha256": self.request_sha256,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
            "verification_record_sha256": self.verification_record_sha256,
        }


@dataclass(frozen=True)
class PublicationReceiptV5:
    schema_version: str
    run_id: str
    custody: SourceCustodyV5
    request_sha256: str
    eligibility_record_sha256: str
    optimality_certificate_sha256: str
    plan_sha256: str
    rendered_artifact_sha256: str
    publication_record_sha256: str
    measurement_status_sha256: str
    verification_record_sha256: str
    index_sha256: str
    closed_files: tuple[PublicationFileV5, ...]
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_V5:
            raise EvidenceError("unsupported_schema", self.schema_version)
        _identifier(self.run_id, "run_id")
        for name in (
            "request_sha256",
            "eligibility_record_sha256",
            "optimality_certificate_sha256",
            "plan_sha256",
            "rendered_artifact_sha256",
            "publication_record_sha256",
            "measurement_status_sha256",
            "verification_record_sha256",
            "index_sha256",
        ):
            require_sha256(getattr(self, name), name)
        paths = tuple(item.path for item in self.closed_files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise EvidenceError("invalid_publication_receipt_files", self.run_id)
        if "index.json" not in paths:
            raise EvidenceError("invalid_publication_receipt_files", "index.json missing")
        if self.terminal_state != "closed":
            raise EvidenceError("receipt_not_closed", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublicationReceiptV5:
        fields = {
            "closed_files",
            "custody",
            "eligibility_record_sha256",
            "index_sha256",
            "measurement_status_sha256",
            "optimality_certificate_sha256",
            "plan_sha256",
            "publication_record_sha256",
            "rendered_artifact_sha256",
            "request_sha256",
            "run_id",
            "schema_version",
            "terminal_state",
            "verification_record_sha256",
        }
        value = strict_fields(data, fields, "PublicationReceiptV5")
        return cls(
            require_string(value["schema_version"], "schema_version"),
            require_string(value["run_id"], "run_id"),
            SourceCustodyV5.from_dict(require_object(value["custody"], "custody")),
            require_string(value["request_sha256"], "request_sha256"),
            require_string(value["eligibility_record_sha256"], "eligibility_record_sha256"),
            require_string(
                value["optimality_certificate_sha256"],
                "optimality_certificate_sha256",
            ),
            require_string(value["plan_sha256"], "plan_sha256"),
            require_string(value["rendered_artifact_sha256"], "rendered_artifact_sha256"),
            require_string(value["publication_record_sha256"], "publication_record_sha256"),
            require_string(value["measurement_status_sha256"], "measurement_status_sha256"),
            require_string(
                value["verification_record_sha256"],
                "verification_record_sha256",
            ),
            require_string(value["index_sha256"], "index_sha256"),
            tuple(
                PublicationFileV5.from_dict(require_object(item, "closed file"))
                for item in require_list(value["closed_files"], "closed_files")
            ),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed_files": [item.to_dict() for item in self.closed_files],
            "custody": self.custody.to_dict(),
            "eligibility_record_sha256": self.eligibility_record_sha256,
            "index_sha256": self.index_sha256,
            "measurement_status_sha256": self.measurement_status_sha256,
            "optimality_certificate_sha256": self.optimality_certificate_sha256,
            "plan_sha256": self.plan_sha256,
            "publication_record_sha256": self.publication_record_sha256,
            "rendered_artifact_sha256": self.rendered_artifact_sha256,
            "request_sha256": self.request_sha256,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "terminal_state": self.terminal_state,
            "verification_record_sha256": self.verification_record_sha256,
        }
