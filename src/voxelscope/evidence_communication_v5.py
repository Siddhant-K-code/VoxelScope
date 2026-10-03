# SPDX-License-Identifier: Apache-2.0
"""Exact no-model v5 discourse planning, publication, and offline replay."""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .atomic import path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    is_link_like,
    load_json,
    require_sha256,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .evidence_communication_v5_records import (
    AUDIENCE_PROFILE_V5,
    CATALOG_SCHEMA_V5,
    ELIGIBILITY_SCHEMA_V5,
    INDEX_SCHEMA_V5,
    MEASUREMENT_SCHEMA_V5,
    MODEL_ELIGIBILITY_STATUS_V5,
    MODEL_GATE_RESULT_V5,
    NOT_APPLICABLE_NO_MODEL_V5,
    OPTIMALITY_SCHEMA_V5,
    PLAN_MODE_V5,
    PLAN_SCHEMA_V5,
    PUBLICATION_SCHEMA_V5,
    RECEIPT_SCHEMA_V5,
    RENDERED_ARTIFACT_SCHEMA_V5,
    REQUEST_SCHEMA_V5,
    VERIFICATION_SCHEMA_V5,
    CaveatAnchorDecisionV5,
    CaveatConstraintV5,
    CoverageCountV5,
    DiscourseBudgetV5,
    DiscoursePlanV5,
    DiscourseRequestV5,
    EligibilityRecordV5,
    MeasurementStatusV5,
    OptimalityCertificateV5,
    OptionalUnitV5,
    OrderingConstraintV5,
    PublicationFileV5,
    PublicationIndexV5,
    PublicationReceiptV5,
    PublicationRecordV5,
    RenderedArtifactV5,
    SentenceCatalogV5,
    SentenceUnitV5,
    SourceCustodyV5,
    VerificationRecordV5,
)
from .records import require_object

DESIGN_CONTRACT_SHA256_V5 = "7389b004f6a75ca00399573bf3ca45074f2941e4ac9772385d88f69a422a4fae"
REQUEST_SCHEMA_SHA256_V5 = "a347247bfaa818738f748de61633b47ccea640083df9aebde7f262fc54cfa038"
PLAN_SCHEMA_SHA256_V5 = "806d847af7dcf0484d57291a468d3e6141e557547c8bb293ae21be272d1d6c1a"
PUBLICATION_SCHEMA_SHA256_V5 = "c4e85b0e5e1c821f29431487782df7f1247a8028d3c619a2018fa57e21666dd3"

RENDERER_ID_V5 = "voxelscope.evidence-communication-v5.canonical-renderer.v1"
RENDERER_SPEC_V5 = {
    "citation_format": "space_then_square_bracketed_citation_ids_in_catalog_order",
    "line_format": "canonical_sentence_then_citations",
    "line_separator": "LF",
    "renderer_id": RENDERER_ID_V5,
    "trailing_line_separator": False,
    "unit_order": "discourse_plan.ordered_unit_ids",
}
RENDERER_SHA256_V5 = sha256_bytes(canonical_json_bytes(RENDERER_SPEC_V5))

OPTIMIZER_SPEC_V5 = {
    "algorithm": "bounded_complete_enumeration_v1",
    "decision_key": [
        "ordered_unit_ids",
        "selected_optional_unit_ids_sorted",
        "caveat_anchor_decisions_sorted",
    ],
    "feasibility": [
        "strict_request_catalog_identity",
        "mandatory_units_exactly_once",
        "optional_subset_cardinality",
        "ordering_constraints",
        "optional_anchor_precedes_optional",
        "caveat_anchor_placement",
        "mandatory_fact_caveat_and_citation_coverage",
        "rendered_utf8_byte_budget",
    ],
    "objective": [
        "maximize_optional_integer_utility",
        "minimize_rendered_utf8_bytes",
        "minimize_canonical_decision_key_bytes",
    ],
}
OPTIMIZER_SHA256_V5 = sha256_bytes(canonical_json_bytes(OPTIMIZER_SPEC_V5))

DEFAULT_MAX_CANDIDATE_EVALUATIONS_V5 = 250_000
HARD_MAX_CANDIDATE_EVALUATIONS_V5 = 1_000_000

_CORE_FILES = {
    "eligibility.json": "eligibility_record",
    "measurement-status.json": "measurement_status",
    "optimality-certificate.json": "optimality_certificate",
    "plan.json": "discourse_plan",
    "publication.json": "publication_record",
    "rendered-artifact.json": "rendered_artifact",
    "request.json": "discourse_request",
    "verification.json": "verification_record",
}
_STAGED_FILES = frozenset((*_CORE_FILES, "index.json"))
_CLOSED_FILES = frozenset((*_STAGED_FILES, "receipt.json"))


@dataclass(frozen=True)
class OptimizationResultV5:
    plan: DiscoursePlanV5
    certificate: OptimalityCertificateV5


@dataclass(frozen=True)
class PublicationBuildResultV5:
    request_sha256: str
    plan_sha256: str
    output_sha256: str
    optional_utility: int
    emitted_utf8_bytes: int
    receipt_sha256: str
    terminal_state: str


@dataclass(frozen=True)
class _RenderedCandidateV5:
    prose: str
    emitted_fact_ids: tuple[str, ...]
    emitted_caveat_ids: tuple[str, ...]
    emitted_citation_ids: tuple[str, ...]
    citation_reference_count: int

    @property
    def encoded(self) -> bytes:
        return self.prose.encode("utf-8")


@dataclass(frozen=True)
class _BuiltRecordsV5:
    request: DiscourseRequestV5
    eligibility: EligibilityRecordV5
    plan: DiscoursePlanV5
    certificate: OptimalityCertificateV5
    rendered: RenderedArtifactV5
    publication: PublicationRecordV5
    measurement: MeasurementStatusV5
    verification: VerificationRecordV5
    index: PublicationIndexV5


def _catalog_digest(catalog: SentenceCatalogV5) -> str:
    return sha256_bytes(canonical_json_bytes(catalog.to_dict()))


def _request_digest(request: DiscourseRequestV5) -> str:
    return sha256_bytes(canonical_json_bytes(request.to_dict()))


def _plan_digest(plan: DiscoursePlanV5) -> str:
    return sha256_bytes(canonical_json_bytes(plan.to_dict()))


def _rendered_unit_text(unit: SentenceUnitV5) -> str:
    citations = " ".join(f"[{citation_id}]" for citation_id in unit.citation_ids)
    return "\n".join(f"{sentence} {citations}" for sentence in unit.canonical_sentences)


def _rendered_unit_bytes(unit: SentenceUnitV5) -> int:
    return len(_rendered_unit_text(unit).encode("utf-8"))


def control_sentence_catalog_v5() -> SentenceCatalogV5:
    """Return the only shipped v5 catalog: deterministic contract/test evidence."""
    return SentenceCatalogV5(
        schema_version=CATALOG_SCHEMA_V5,
        catalog_id="deterministic-control-v1",
        citation_ids=("contract.design", "contract.publication", "contract.request"),
        units=(
            SentenceUnitV5(
                unit_id="caveat.evidence-status",
                unit_kind="caveat",
                canonical_sentences=(
                    "This output is contract and test evidence, not observed model evidence.",
                ),
                citation_ids=("contract.design",),
                fact_ids=(),
                caveat_ids=("caveat.not-observed-model-evidence",),
            ),
            SentenceUnitV5(
                unit_id="fact.code-ownership",
                unit_kind="fact",
                canonical_sentences=(
                    "Trusted code owns every sentence, citation, identifier, and utility value.",
                ),
                citation_ids=("contract.request",),
                fact_ids=("fact.code-owned-content",),
                caveat_ids=(),
            ),
            SentenceUnitV5(
                unit_id="fact.objective",
                unit_kind="fact",
                canonical_sentences=(
                    "The planner maximizes request-pinned optional utility before minimizing "
                    "emitted UTF-8 bytes.",
                ),
                citation_ids=("contract.design",),
                fact_ids=("fact.lexicographic-objective",),
                caveat_ids=(),
            ),
            SentenceUnitV5(
                unit_id="optional.enumeration",
                unit_kind="optional_context",
                canonical_sentences=(
                    "The optimizer enumerates the complete bounded feasible plan set.",
                ),
                citation_ids=("contract.design",),
                fact_ids=(),
                caveat_ids=(),
            ),
            SentenceUnitV5(
                unit_id="optional.replay",
                unit_kind="optional_context",
                canonical_sentences=(
                    "Offline replay reconstructs the same records without network access.",
                ),
                citation_ids=("contract.publication",),
                fact_ids=(),
                caveat_ids=(),
            ),
            SentenceUnitV5(
                unit_id="safety.non-clinical",
                unit_kind="safety",
                canonical_sentences=(
                    "The output does not support clinical use, diagnosis, prognosis, treatment, "
                    "or patient-specific claims.",
                ),
                citation_ids=("contract.design",),
                fact_ids=(),
                caveat_ids=("caveat.non-clinical-boundary",),
            ),
        ),
    )


def control_discourse_request_v5(
    catalog: SentenceCatalogV5 | None = None,
) -> DiscourseRequestV5:
    """Build the code-owned deterministic control request without free-form input."""
    selected_catalog = catalog or control_sentence_catalog_v5()
    units = {unit.unit_id: unit for unit in selected_catalog.units}
    optional_ids = ("optional.enumeration", "optional.replay")
    return DiscourseRequestV5(
        schema_version=REQUEST_SCHEMA_V5,
        case_id="deterministic-control",
        sentence_catalog_sha256=_catalog_digest(selected_catalog),
        study_declaration_sha256=DESIGN_CONTRACT_SHA256_V5,
        renderer_id=RENDERER_ID_V5,
        audience_profile=AUDIENCE_PROFILE_V5,
        mandatory_unit_ids=(
            "caveat.evidence-status",
            "fact.code-ownership",
            "fact.objective",
            "safety.non-clinical",
        ),
        mandatory_fact_ids=(
            "fact.code-owned-content",
            "fact.lexicographic-objective",
        ),
        mandatory_caveat_ids=(
            "caveat.non-clinical-boundary",
            "caveat.not-observed-model-evidence",
        ),
        ordering_constraints=(
            OrderingConstraintV5("fact.code-ownership", "fact.objective"),
            OrderingConstraintV5("fact.objective", "caveat.evidence-status"),
        ),
        caveat_constraints=(
            CaveatConstraintV5(
                "caveat.evidence-status",
                ("fact.objective",),
                ("after",),
            ),
            CaveatConstraintV5(
                "safety.non-clinical",
                ("caveat.evidence-status",),
                ("after",),
            ),
        ),
        optional_units=tuple(
            OptionalUnitV5(
                unit_id=unit_id,
                utility=4,
                estimated_max_bytes=_rendered_unit_bytes(units[unit_id]),
                allowed_anchor_ids=(
                    ("fact.objective",)
                    if unit_id == "optional.enumeration"
                    else ("caveat.evidence-status",)
                ),
            )
            for unit_id in optional_ids
        ),
        budget=DiscourseBudgetV5(
            max_output_bytes=1200,
            max_optional_units=2,
        ),
    )


def _trusted_catalog_for_request(request: DiscourseRequestV5) -> SentenceCatalogV5:
    if request.case_id != "deterministic-control":
        raise EvidenceError("unknown_code_owned_case", request.case_id)
    return control_sentence_catalog_v5()


def _validate_request_catalog_v5(
    request: DiscourseRequestV5,
    catalog: SentenceCatalogV5,
) -> None:
    if request.sentence_catalog_sha256 != _catalog_digest(catalog):
        raise EvidenceError("sentence_catalog_digest_mismatch", request.case_id)
    if request.study_declaration_sha256 != DESIGN_CONTRACT_SHA256_V5:
        raise EvidenceError("prospective_contract_digest_mismatch", request.case_id)
    if request.renderer_id != RENDERER_ID_V5:
        raise EvidenceError("renderer_identity_mismatch", request.renderer_id)
    units = {unit.unit_id: unit for unit in catalog.units}
    known_request_units = set(request.mandatory_unit_ids) | {
        item.unit_id for item in request.optional_units
    }
    if known_request_units != set(units):
        raise EvidenceError(
            "sentence_catalog_unit_set_mismatch",
            f"missing={sorted(set(units) - known_request_units)} "
            f"unknown={sorted(known_request_units - set(units))}",
        )
    mandatory_units = tuple(units[unit_id] for unit_id in request.mandatory_unit_ids)
    if any(unit.unit_kind == "optional_context" for unit in mandatory_units):
        raise EvidenceError("optional_mandatory_confusion", request.case_id)
    optional_by_id = {item.unit_id: item for item in request.optional_units}
    for unit_id, optional in optional_by_id.items():
        unit = units[unit_id]
        if unit.unit_kind != "optional_context":
            raise EvidenceError("optional_mandatory_confusion", unit_id)
        if optional.estimated_max_bytes < _rendered_unit_bytes(unit):
            raise EvidenceError("optional_byte_bound_understated", unit_id)
    facts = tuple(sorted(fact_id for unit in mandatory_units for fact_id in unit.fact_ids))
    caveats = tuple(sorted(caveat_id for unit in mandatory_units for caveat_id in unit.caveat_ids))
    if facts != request.mandatory_fact_ids:
        raise EvidenceError("mandatory_fact_set_mismatch", request.case_id)
    if caveats != request.mandatory_caveat_ids:
        raise EvidenceError("mandatory_caveat_set_mismatch", request.case_id)
    constrained = {item.caveat_unit_id for item in request.caveat_constraints}
    expected_constrained = {
        unit.unit_id for unit in mandatory_units if unit.unit_kind in {"caveat", "safety"}
    }
    if constrained != expected_constrained:
        raise EvidenceError("caveat_constraint_set_mismatch", request.case_id)


def canonical_decision_key_v5(plan: DiscoursePlanV5) -> bytes:
    """Return the contract-pinned decision key, excluding envelope fields."""
    return canonical_json_bytes(
        {
            "caveat_anchor_decisions": [item.to_dict() for item in plan.caveat_anchor_decisions],
            "ordered_unit_ids": list(plan.ordered_unit_ids),
            "selected_optional_unit_ids": list(plan.selected_optional_unit_ids),
        }
    )


def _validate_plan_v5(
    request: DiscourseRequestV5,
    plan: DiscoursePlanV5,
) -> None:
    request_sha256 = _request_digest(request)
    if plan.request_sha256 != request_sha256:
        raise EvidenceError("plan_request_digest_mismatch", request.case_id)
    optional_by_id = {item.unit_id: item for item in request.optional_units}
    selected = set(plan.selected_optional_unit_ids)
    if not selected.issubset(optional_by_id):
        raise EvidenceError("unknown_optional_unit", request.case_id)
    if len(selected) > request.budget.max_optional_units:
        raise EvidenceError("optional_unit_budget_exceeded", request.case_id)
    expected_units = set(request.mandatory_unit_ids) | selected
    if set(plan.ordered_unit_ids) != expected_units:
        raise EvidenceError("plan_unit_set_mismatch", request.case_id)
    positions = {unit_id: index for index, unit_id in enumerate(plan.ordered_unit_ids)}
    for ordering_constraint in request.ordering_constraints:
        if (
            ordering_constraint.before_unit_id in positions
            and ordering_constraint.after_unit_id in positions
            and positions[ordering_constraint.before_unit_id]
            >= positions[ordering_constraint.after_unit_id]
        ):
            raise EvidenceError("ordering_constraint_violation", request.case_id)
    for optional_id in plan.selected_optional_unit_ids:
        allowed = optional_by_id[optional_id].allowed_anchor_ids
        if not any(positions[anchor_id] < positions[optional_id] for anchor_id in allowed):
            raise EvidenceError("optional_anchor_violation", optional_id)
    constraints = {item.caveat_unit_id: item for item in request.caveat_constraints}
    decisions = {item.caveat_unit_id: item for item in plan.caveat_anchor_decisions}
    if len(decisions) != len(plan.caveat_anchor_decisions) or set(decisions) != set(constraints):
        raise EvidenceError("caveat_decision_set_mismatch", request.case_id)
    for caveat_id, decision in decisions.items():
        caveat_constraint = constraints[caveat_id]
        if (
            decision.anchor_id not in caveat_constraint.allowed_anchor_ids
            or decision.placement not in caveat_constraint.allowed_placements
        ):
            raise EvidenceError("invalid_caveat_decision", caveat_id)
        before = positions[caveat_id] < positions[decision.anchor_id]
        if before != (decision.placement == "before"):
            raise EvidenceError("caveat_placement_violation", caveat_id)


def _render_candidate_v5(
    request: DiscourseRequestV5,
    catalog: SentenceCatalogV5,
    plan: DiscoursePlanV5,
) -> _RenderedCandidateV5:
    _validate_request_catalog_v5(request, catalog)
    _validate_plan_v5(request, plan)
    units = {unit.unit_id: unit for unit in catalog.units}
    lines: list[str] = []
    fact_ids: set[str] = set()
    caveat_ids: set[str] = set()
    citation_ids: set[str] = set()
    citation_reference_count = 0
    for unit_id in plan.ordered_unit_ids:
        unit = units[unit_id]
        lines.extend(_rendered_unit_text(unit).splitlines())
        fact_ids.update(unit.fact_ids)
        caveat_ids.update(unit.caveat_ids)
        citation_ids.update(unit.citation_ids)
        citation_reference_count += len(unit.citation_ids) * len(unit.canonical_sentences)
    if tuple(sorted(fact_ids)) != request.mandatory_fact_ids:
        raise EvidenceError("mandatory_fact_coverage_failed", request.case_id)
    if tuple(sorted(caveat_ids)) != request.mandatory_caveat_ids:
        raise EvidenceError("mandatory_caveat_coverage_failed", request.case_id)
    if not citation_ids.issubset(catalog.citation_ids) or citation_reference_count <= 0:
        raise EvidenceError("citation_validity_failed", request.case_id)
    return _RenderedCandidateV5(
        prose="\n".join(lines),
        emitted_fact_ids=tuple(sorted(fact_ids)),
        emitted_caveat_ids=tuple(sorted(caveat_ids)),
        emitted_citation_ids=tuple(sorted(citation_ids)),
        citation_reference_count=citation_reference_count,
    )


def render_plan_v5(
    request: DiscourseRequestV5,
    catalog: SentenceCatalogV5,
    plan: DiscoursePlanV5,
) -> RenderedArtifactV5:
    """Render only canonical catalog text after every plan gate passes."""
    rendered = _render_candidate_v5(request, catalog, plan)
    encoded = rendered.encoded
    if len(encoded) > request.budget.max_output_bytes:
        raise EvidenceError("output_byte_budget_exceeded", request.case_id)
    request_sha256 = _request_digest(request)
    plan_sha256 = _plan_digest(plan)
    return RenderedArtifactV5(
        schema_version=RENDERED_ARTIFACT_SCHEMA_V5,
        request_sha256=request_sha256,
        plan_sha256=plan_sha256,
        sentence_catalog_sha256=_catalog_digest(catalog),
        renderer_sha256=RENDERER_SHA256_V5,
        audience_profile=request.audience_profile,
        ordered_unit_ids=plan.ordered_unit_ids,
        emitted_fact_ids=rendered.emitted_fact_ids,
        emitted_caveat_ids=rendered.emitted_caveat_ids,
        emitted_citation_ids=rendered.emitted_citation_ids,
        mandatory_fact_coverage=CoverageCountV5(
            len(rendered.emitted_fact_ids),
            len(request.mandatory_fact_ids),
        ),
        mandatory_caveat_coverage=CoverageCountV5(
            len(rendered.emitted_caveat_ids),
            len(request.mandatory_caveat_ids),
        ),
        citation_validity=CoverageCountV5(
            rendered.citation_reference_count,
            rendered.citation_reference_count,
        ),
        prose=rendered.prose,
        output_sha256=sha256_bytes(encoded),
        emitted_utf8_bytes=len(encoded),
        terminal_state="verified",
    )


def _topological_orders(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> Iterator[tuple[str, ...]]:
    successors: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for before, after in edges:
        if after not in successors[before]:
            successors[before].add(after)
            indegree[after] += 1

    def visit(
        ordered: tuple[str, ...],
        current_indegree: dict[str, int],
    ) -> Iterator[tuple[str, ...]]:
        if len(ordered) == len(nodes):
            yield ordered
            return
        ready = tuple(
            sorted(node for node in nodes if node not in ordered and current_indegree[node] == 0)
        )
        for node in ready:
            updated = dict(current_indegree)
            updated[node] = -1
            for successor in successors[node]:
                updated[successor] -= 1
            yield from visit((*ordered, node), updated)

    yield from visit((), indegree)


def _caveat_assignments(
    constraints: tuple[CaveatConstraintV5, ...],
) -> Iterator[tuple[CaveatAnchorDecisionV5, ...]]:
    choices = tuple(
        tuple(
            CaveatAnchorDecisionV5(constraint.caveat_unit_id, anchor, placement)
            for anchor in constraint.allowed_anchor_ids
            for placement in constraint.allowed_placements
        )
        for constraint in constraints
    )
    for selected in itertools.product(*choices):
        yield tuple(
            sorted(selected, key=lambda item: (item.caveat_unit_id, item.anchor_id, item.placement))
        )


def _selected_subsets(
    optional_ids: tuple[str, ...],
    maximum: int,
) -> Iterator[tuple[str, ...]]:
    for size in range(maximum + 1):
        yield from itertools.combinations(optional_ids, size)


def _combined_edges(
    request: DiscourseRequestV5,
    selected_units: set[str],
    decisions: tuple[CaveatAnchorDecisionV5, ...],
) -> tuple[tuple[str, str], ...]:
    edges = {
        (item.before_unit_id, item.after_unit_id)
        for item in request.ordering_constraints
        if item.before_unit_id in selected_units and item.after_unit_id in selected_units
    }
    for decision in decisions:
        if decision.placement == "before":
            edges.add((decision.caveat_unit_id, decision.anchor_id))
        else:
            edges.add((decision.anchor_id, decision.caveat_unit_id))
    return tuple(sorted(edges))


def _has_cycle(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> bool:
    return not any(True for _ in itertools.islice(_topological_orders(nodes, edges), 1))


def optimize_discourse_plan_v5(
    request: DiscourseRequestV5,
    catalog: SentenceCatalogV5,
    *,
    max_candidate_evaluations: int = DEFAULT_MAX_CANDIDATE_EVALUATIONS_V5,
) -> OptimizationResultV5:
    """Enumerate the complete bounded feasible set and return its exact optimum."""
    if not 1 <= max_candidate_evaluations <= HARD_MAX_CANDIDATE_EVALUATIONS_V5:
        raise EvidenceError(
            "invalid_optimizer_search_limit",
            str(max_candidate_evaluations),
        )
    _validate_request_catalog_v5(request, catalog)
    request_sha256 = _request_digest(request)
    optional_ids = tuple(item.unit_id for item in request.optional_units)
    caveat_choice_count = math.prod(
        len(item.allowed_anchor_ids) * len(item.allowed_placements)
        for item in request.caveat_constraints
    )
    subset_count = sum(
        math.comb(len(optional_ids), size) for size in range(request.budget.max_optional_units + 1)
    )
    if subset_count * caveat_choice_count > max_candidate_evaluations:
        raise EvidenceError(
            "optimizer_search_limit_exceeded",
            "subset and caveat configuration count exceeds the explicit limit",
        )
    optional_utility = {item.unit_id: item.utility for item in request.optional_units}
    optional_anchors = {item.unit_id: item.allowed_anchor_ids for item in request.optional_units}
    search_digest = hashlib.sha256()
    optional_subsets_examined = 0
    caveat_assignments_examined = 0
    topological_orders_examined = 0
    feasible_candidates = 0
    budget_rejections = 0
    structural_rejections = 0
    winner: tuple[tuple[int, int, bytes], DiscoursePlanV5] | None = None

    for subset in _selected_subsets(optional_ids, request.budget.max_optional_units):
        optional_subsets_examined += 1
        selected_units = set(request.mandatory_unit_ids) | set(subset)
        nodes = tuple(sorted(selected_units))
        utility = sum(optional_utility[unit_id] for unit_id in subset)
        for decisions in _caveat_assignments(request.caveat_constraints):
            caveat_assignments_examined += 1
            edges = _combined_edges(request, selected_units, decisions)
            if _has_cycle(nodes, edges):
                structural_rejections += 1
                search_digest.update(
                    canonical_json_bytes(
                        {
                            "caveat_anchor_decisions": [item.to_dict() for item in decisions],
                            "selected_optional_unit_ids": list(subset),
                            "status": "combined_ordering_cycle",
                        }
                    )
                )
                continue
            for ordered in _topological_orders(nodes, edges):
                topological_orders_examined += 1
                if topological_orders_examined > max_candidate_evaluations:
                    raise EvidenceError(
                        "optimizer_search_limit_exceeded",
                        "topological plan count exceeds the explicit limit",
                    )
                positions = {unit_id: index for index, unit_id in enumerate(ordered)}
                if any(
                    not any(
                        positions[anchor_id] < positions[optional_id]
                        for anchor_id in optional_anchors[optional_id]
                    )
                    for optional_id in subset
                ):
                    structural_rejections += 1
                    search_digest.update(
                        canonical_json_bytes(
                            {
                                "caveat_anchor_decisions": [item.to_dict() for item in decisions],
                                "ordered_unit_ids": list(ordered),
                                "selected_optional_unit_ids": list(subset),
                                "status": "optional_anchor_violation",
                            }
                        )
                    )
                    continue
                plan = DiscoursePlanV5(
                    schema_version=PLAN_SCHEMA_V5,
                    request_sha256=request_sha256,
                    plan_mode=PLAN_MODE_V5,
                    ordered_unit_ids=ordered,
                    selected_optional_unit_ids=subset,
                    caveat_anchor_decisions=decisions,
                )
                rendered = _render_candidate_v5(request, catalog, plan)
                emitted_bytes = len(rendered.encoded)
                if emitted_bytes > request.budget.max_output_bytes:
                    budget_rejections += 1
                    status = "byte_budget_rejected"
                else:
                    feasible_candidates += 1
                    status = "feasible"
                    decision_key = canonical_decision_key_v5(plan)
                    score = (-utility, emitted_bytes, decision_key)
                    if winner is None or score < winner[0]:
                        winner = (score, plan)
                search_digest.update(
                    canonical_json_bytes(
                        {
                            "caveat_anchor_decisions": [item.to_dict() for item in decisions],
                            "emitted_utf8_bytes": emitted_bytes,
                            "optional_utility": utility,
                            "ordered_unit_ids": list(ordered),
                            "selected_optional_unit_ids": list(subset),
                            "status": status,
                        }
                    )
                )
    if winner is None:
        raise EvidenceError("no_feasible_deterministic_plan", request.case_id)
    winning_plan = winner[1]
    plan_sha256 = _plan_digest(winning_plan)
    winner_rendered = _render_candidate_v5(request, catalog, winning_plan)
    winner_utility = sum(
        optional_utility[unit_id] for unit_id in winning_plan.selected_optional_unit_ids
    )
    certificate = OptimalityCertificateV5(
        schema_version=OPTIMALITY_SCHEMA_V5,
        request_sha256=request_sha256,
        sentence_catalog_sha256=_catalog_digest(catalog),
        renderer_sha256=RENDERER_SHA256_V5,
        optimizer_sha256=OPTIMIZER_SHA256_V5,
        plan_sha256=plan_sha256,
        algorithm="bounded_complete_enumeration_v1",
        decision_key_version="canonical_decision_key_v1",
        exhaustive=True,
        max_candidate_evaluations=max_candidate_evaluations,
        optional_subsets_examined=optional_subsets_examined,
        caveat_assignments_examined=caveat_assignments_examined,
        topological_orders_examined=topological_orders_examined,
        feasible_candidates=feasible_candidates,
        budget_rejections=budget_rejections,
        structural_rejections=structural_rejections,
        search_space_sha256=search_digest.hexdigest(),
        winner_decision_key_sha256=sha256_bytes(canonical_decision_key_v5(winning_plan)),
        winner_optional_utility=winner_utility,
        winner_emitted_utf8_bytes=len(winner_rendered.encoded),
        terminal_state="proved",
    )
    return OptimizationResultV5(winning_plan, certificate)


def build_eligibility_record_v5(
    request: DiscourseRequestV5,
    custody: SourceCustodyV5,
) -> EligibilityRecordV5:
    """Close the pre-inference gate; this module exposes no model action."""
    return EligibilityRecordV5(
        schema_version=ELIGIBILITY_SCHEMA_V5,
        request_sha256=_request_digest(request),
        design_contract_sha256=DESIGN_CONTRACT_SHA256_V5,
        request_schema_sha256=REQUEST_SCHEMA_SHA256_V5,
        plan_schema_sha256=PLAN_SCHEMA_SHA256_V5,
        publication_schema_sha256=PUBLICATION_SCHEMA_SHA256_V5,
        sentence_catalog_sha256=request.sentence_catalog_sha256,
        renderer_sha256=RENDERER_SHA256_V5,
        optimizer_sha256=OPTIMIZER_SHA256_V5,
        custody=custody,
        gate_result=MODEL_GATE_RESULT_V5,
        model_eligibility_status=MODEL_ELIGIBILITY_STATUS_V5,
        evaluated_before_model_actions=True,
        model_candidate_action_count=0,
        reason_code="exact_deterministic_objectives_exhaust_current_value_endpoints",
        terminal_state="eligibility_checked",
    )


def _publication_files(
    values: dict[str, tuple[str, dict[str, Any]]],
) -> tuple[PublicationFileV5, ...]:
    return tuple(
        PublicationFileV5(
            artifact_type=artifact_type,
            path=path,
            sha256=sha256_bytes(canonical_json_bytes(value)),
        )
        for path, (artifact_type, value) in sorted(values.items())
    )


def _build_records_v5(
    run_id: str,
    request: DiscourseRequestV5,
    catalog: SentenceCatalogV5,
    custody: SourceCustodyV5,
    eligibility: EligibilityRecordV5,
    optimization: OptimizationResultV5,
) -> _BuiltRecordsV5:
    request_sha256 = _request_digest(request)
    eligibility_sha256 = sha256_bytes(canonical_json_bytes(eligibility.to_dict()))
    plan = optimization.plan
    plan_sha256 = _plan_digest(plan)
    certificate = optimization.certificate
    certificate_sha256 = sha256_bytes(canonical_json_bytes(certificate.to_dict()))
    rendered = render_plan_v5(request, catalog, plan)
    rendered_sha256 = sha256_bytes(canonical_json_bytes(rendered.to_dict()))
    publication = PublicationRecordV5(
        schema_version=PUBLICATION_SCHEMA_V5,
        run_id=run_id,
        request_sha256=request_sha256,
        audience_profile=request.audience_profile,
        model_eligibility_status=MODEL_ELIGIBILITY_STATUS_V5,
        eligibility_record_sha256=eligibility_sha256,
        optimizer_proof_sha256=certificate_sha256,
        baseline_plan_sha256=plan_sha256,
        baseline_output_sha256=rendered.output_sha256,
        baseline_optional_utility=certificate.winner_optional_utility,
        baseline_emitted_utf8_bytes=rendered.emitted_utf8_bytes,
        renderer_sha256=RENDERER_SHA256_V5,
        mandatory_fact_coverage=rendered.mandatory_fact_coverage,
        mandatory_caveat_coverage=rendered.mandatory_caveat_coverage,
        citation_validity=rendered.citation_validity,
        protocol_validity="valid",
        model_outcome="ineligible_not_requested",
        model_input_sha256=None,
        model_output_sha256=None,
        model_plan_sha256=None,
        published_variant=PLAN_MODE_V5,
        terminal_state="closed",
    )
    publication_sha256 = sha256_bytes(canonical_json_bytes(publication.to_dict()))
    measurement = MeasurementStatusV5(
        schema_version=MEASUREMENT_SCHEMA_V5,
        request_sha256=request_sha256,
        plan_sha256=plan_sha256,
        optimality_certificate_sha256=certificate_sha256,
        deterministic_candidate_evaluations=certificate.topological_orders_examined,
        deterministic_feasible_candidates=certificate.feasible_candidates,
        deterministic_planner_wall_clock_ms=("not_recorded_to_preserve_byte_for_byte_replay"),
        model_inference=NOT_APPLICABLE_NO_MODEL_V5,
        model_input_tokens=NOT_APPLICABLE_NO_MODEL_V5,
        model_output_tokens=NOT_APPLICABLE_NO_MODEL_V5,
        model_latency_ms=NOT_APPLICABLE_NO_MODEL_V5,
        terminal_state="completed",
    )
    measurement_sha256 = sha256_bytes(canonical_json_bytes(measurement.to_dict()))
    verification = VerificationRecordV5(
        schema_version=VERIFICATION_SCHEMA_V5,
        request_sha256=request_sha256,
        eligibility_record_sha256=eligibility_sha256,
        optimality_certificate_sha256=certificate_sha256,
        plan_sha256=plan_sha256,
        rendered_artifact_sha256=rendered_sha256,
        publication_record_sha256=publication_sha256,
        exact_schema_gate="passed",
        identity_gate="passed",
        mandatory_coverage_gate="passed",
        citation_gate="passed",
        protocol_gate="passed",
        byte_budget_gate="passed",
        optimality_gate="passed",
        offline_replay_gate="passed",
        terminal_state="publication_verified",
    )
    verification_sha256 = sha256_bytes(canonical_json_bytes(verification.to_dict()))
    core_values: dict[str, tuple[str, dict[str, Any]]] = {
        "eligibility.json": ("eligibility_record", eligibility.to_dict()),
        "measurement-status.json": ("measurement_status", measurement.to_dict()),
        "optimality-certificate.json": (
            "optimality_certificate",
            certificate.to_dict(),
        ),
        "plan.json": ("discourse_plan", plan.to_dict()),
        "publication.json": ("publication_record", publication.to_dict()),
        "rendered-artifact.json": ("rendered_artifact", rendered.to_dict()),
        "request.json": ("discourse_request", request.to_dict()),
        "verification.json": ("verification_record", verification.to_dict()),
    }
    files = _publication_files(core_values)
    index = PublicationIndexV5(
        schema_version=INDEX_SCHEMA_V5,
        run_id=run_id,
        custody=custody,
        request_sha256=request_sha256,
        eligibility_record_sha256=eligibility_sha256,
        optimality_certificate_sha256=certificate_sha256,
        plan_sha256=plan_sha256,
        rendered_artifact_sha256=rendered_sha256,
        publication_record_sha256=publication_sha256,
        measurement_status_sha256=measurement_sha256,
        verification_record_sha256=verification_sha256,
        files=files,
        terminal_state="publication_verified",
    )
    return _BuiltRecordsV5(
        request=request,
        eligibility=eligibility,
        plan=plan,
        certificate=certificate,
        rendered=rendered,
        publication=publication,
        measurement=measurement,
        verification=verification,
        index=index,
    )


def _record_values(records: _BuiltRecordsV5) -> dict[str, dict[str, Any]]:
    return {
        "eligibility.json": records.eligibility.to_dict(),
        "index.json": records.index.to_dict(),
        "measurement-status.json": records.measurement.to_dict(),
        "optimality-certificate.json": records.certificate.to_dict(),
        "plan.json": records.plan.to_dict(),
        "publication.json": records.publication.to_dict(),
        "rendered-artifact.json": records.rendered.to_dict(),
        "request.json": records.request.to_dict(),
        "verification.json": records.verification.to_dict(),
    }


def _receipt_for_records(records: _BuiltRecordsV5) -> PublicationReceiptV5:
    values = _record_values(records)
    files = tuple(
        PublicationFileV5(
            artifact_type=("publication_index" if path == "index.json" else _CORE_FILES[path]),
            path=path,
            sha256=sha256_bytes(canonical_json_bytes(value)),
        )
        for path, value in sorted(values.items())
    )
    return PublicationReceiptV5(
        schema_version=RECEIPT_SCHEMA_V5,
        run_id=records.publication.run_id,
        custody=records.index.custody,
        request_sha256=records.index.request_sha256,
        eligibility_record_sha256=records.index.eligibility_record_sha256,
        optimality_certificate_sha256=(records.index.optimality_certificate_sha256),
        plan_sha256=records.index.plan_sha256,
        rendered_artifact_sha256=records.index.rendered_artifact_sha256,
        publication_record_sha256=records.index.publication_record_sha256,
        measurement_status_sha256=records.index.measurement_status_sha256,
        verification_record_sha256=records.index.verification_record_sha256,
        index_sha256=sha256_bytes(canonical_json_bytes(records.index.to_dict())),
        closed_files=files,
        terminal_state="closed",
    )


def _load_records_v5(bundle: Path) -> _BuiltRecordsV5:
    request = DiscourseRequestV5.from_dict(
        require_object(load_json(bundle / "request.json"), "discourse request")
    )
    return _BuiltRecordsV5(
        request=request,
        eligibility=EligibilityRecordV5.from_dict(
            require_object(load_json(bundle / "eligibility.json"), "eligibility record")
        ),
        plan=DiscoursePlanV5.from_dict(
            require_object(load_json(bundle / "plan.json"), "discourse plan")
        ),
        certificate=OptimalityCertificateV5.from_dict(
            require_object(
                load_json(bundle / "optimality-certificate.json"),
                "optimality certificate",
            )
        ),
        rendered=RenderedArtifactV5.from_dict(
            require_object(
                load_json(bundle / "rendered-artifact.json"),
                "rendered artifact",
            )
        ),
        publication=PublicationRecordV5.from_dict(
            require_object(load_json(bundle / "publication.json"), "publication record")
        ),
        measurement=MeasurementStatusV5.from_dict(
            require_object(
                load_json(bundle / "measurement-status.json"),
                "measurement status",
            )
        ),
        verification=VerificationRecordV5.from_dict(
            require_object(load_json(bundle / "verification.json"), "verification record")
        ),
        index=PublicationIndexV5.from_dict(
            require_object(load_json(bundle / "index.json"), "publication index")
        ),
    )


def _rebuild_records_v5(records: _BuiltRecordsV5) -> _BuiltRecordsV5:
    catalog = _trusted_catalog_for_request(records.request)
    expected_request = control_discourse_request_v5(catalog)
    if records.request != expected_request:
        raise EvidenceError("request_replay_mismatch", records.request.case_id)
    expected_eligibility = build_eligibility_record_v5(
        expected_request,
        records.index.custody,
    )
    optimization = optimize_discourse_plan_v5(
        expected_request,
        catalog,
        max_candidate_evaluations=records.certificate.max_candidate_evaluations,
    )
    return _build_records_v5(
        records.publication.run_id,
        expected_request,
        catalog,
        records.index.custody,
        expected_eligibility,
        optimization,
    )


def _validate_directory_entries(
    bundle: Path,
    expected: frozenset[str],
) -> None:
    if is_link_like(bundle) or not bundle.is_dir():
        raise EvidenceError("unsafe_publication_bundle", str(bundle))
    names: set[str] = set()
    for item in bundle.iterdir():
        if is_link_like(item) or not item.is_file():
            raise EvidenceError("symlink_forbidden", str(item))
        names.add(item.name)
    if names != expected:
        raise EvidenceError(
            "bundle_file_set_mismatch",
            f"missing={sorted(expected - names)} extra={sorted(names - expected)}",
        )


def _verify_staged_records_v5(stage: Path, expected_records: _BuiltRecordsV5) -> None:
    _validate_directory_entries(stage, _STAGED_FILES)
    stored = _load_records_v5(stage)
    rebuilt = _rebuild_records_v5(stored)
    if stored != rebuilt or stored != expected_records:
        raise EvidenceError("publication_replay_mismatch", stored.publication.run_id)
    values = _record_values(stored)
    for indexed in stored.index.files:
        value = values.get(indexed.path)
        if value is None or indexed.sha256 != sha256_bytes(canonical_json_bytes(value)):
            raise EvidenceError("publication_file_digest_mismatch", indexed.path)
        if sha256_file(stage / indexed.path) != indexed.sha256:
            raise EvidenceError("publication_file_digest_mismatch", indexed.path)
    if {item.path for item in stored.index.files} != set(_CORE_FILES):
        raise EvidenceError("publication_index_file_set_mismatch", stored.publication.run_id)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        ctypes_windows = cast(Any, ctypes)
        win_dll = ctypes_windows.WinDLL
        kernel32: Any = win_dll("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(path),
            0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            error_number = ctypes_windows.get_last_error()
            raise OSError(error_number, os.strerror(error_number), str(path))
        try:
            if not kernel32.FlushFileBuffers(handle):
                error_number = ctypes_windows.get_last_error()
                raise OSError(error_number, os.strerror(error_number), str(path))
        finally:
            kernel32.CloseHandle(handle)
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_link_like_parent(path: Path) -> None:
    if is_link_like(path):
        raise EvidenceError("symlink_forbidden", str(path))


def publish_control_bundle_v5(
    output: Path,
    run_id: str,
    custody: SourceCustodyV5,
    *,
    max_candidate_evaluations: int = DEFAULT_MAX_CANDIDATE_EVALUATIONS_V5,
) -> PublicationBuildResultV5:
    """Publish the code-owned control bundle with the immutable receipt last."""
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_like_parent(output.parent)
    catalog = control_sentence_catalog_v5()
    request = control_discourse_request_v5(catalog)
    eligibility = build_eligibility_record_v5(request, custody)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        write_json(stage / "request.json", request.to_dict())
        write_json(stage / "eligibility.json", eligibility.to_dict())
        _fsync_file(stage / "request.json")
        _fsync_file(stage / "eligibility.json")
        _fsync_directory(stage)

        optimization = optimize_discourse_plan_v5(
            request,
            catalog,
            max_candidate_evaluations=max_candidate_evaluations,
        )
        records = _build_records_v5(
            run_id,
            request,
            catalog,
            custody,
            eligibility,
            optimization,
        )
        for path, value in _record_values(records).items():
            write_json(stage / path, value)
        _verify_staged_records_v5(stage, records)
        for path in sorted(_STAGED_FILES):
            _fsync_file(stage / path)
        _fsync_directory(stage)

        receipt = _receipt_for_records(records)
        receipt_bytes = canonical_json_bytes(receipt.to_dict())
        rename_no_replace(stage, output)
        published = True
        receipt_path = output / "receipt.json"
        try:
            with receipt_path.open("xb") as stream:
                stream.write(receipt_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise EvidenceError("output_exists", str(receipt_path)) from exc
        _fsync_directory(output)
        replay_control_bundle_v5(output)
        return PublicationBuildResultV5(
            request_sha256=records.index.request_sha256,
            plan_sha256=records.index.plan_sha256,
            output_sha256=records.rendered.output_sha256,
            optional_utility=records.publication.baseline_optional_utility,
            emitted_utf8_bytes=records.rendered.emitted_utf8_bytes,
            receipt_sha256=sha256_bytes(receipt_bytes),
            terminal_state="closed",
        )
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def replay_control_bundle_v5(bundle: Path) -> PublicationBuildResultV5:
    """Reconstruct every v5 record byte-for-byte without network or a model."""
    _validate_directory_entries(bundle, _CLOSED_FILES)
    receipt = PublicationReceiptV5.from_dict(
        require_object(load_json(bundle / "receipt.json"), "publication receipt")
    )
    records = _load_records_v5(bundle)
    rebuilt = _rebuild_records_v5(records)
    if records != rebuilt:
        raise EvidenceError("publication_replay_mismatch", receipt.run_id)
    expected_receipt = _receipt_for_records(rebuilt)
    if receipt != expected_receipt:
        raise EvidenceError("publication_receipt_replay_mismatch", receipt.run_id)
    if sha256_file(bundle / "index.json") != receipt.index_sha256:
        raise EvidenceError("publication_index_digest_mismatch", receipt.run_id)
    expected_paths = {item.path for item in receipt.closed_files}
    if expected_paths != set(_STAGED_FILES):
        raise EvidenceError("publication_receipt_file_set_mismatch", receipt.run_id)
    for closed_file in receipt.closed_files:
        if sha256_file(bundle / closed_file.path) != closed_file.sha256:
            raise EvidenceError("publication_file_digest_mismatch", closed_file.path)
    return PublicationBuildResultV5(
        request_sha256=receipt.request_sha256,
        plan_sha256=receipt.plan_sha256,
        output_sha256=records.rendered.output_sha256,
        optional_utility=records.publication.baseline_optional_utility,
        emitted_utf8_bytes=records.rendered.emitted_utf8_bytes,
        receipt_sha256=sha256_file(bundle / "receipt.json"),
        terminal_state=receipt.terminal_state,
    )


def verify_contract_identities_v5(repository_root: Path) -> dict[str, str]:
    """Recompute the immutable merged design and schema digests."""
    contract_root = (
        repository_root / "research" / "evidence-communication-v5-discourse-planner-contract-v1"
    )
    expected = {
        "design_contract_sha256": (
            contract_root / "design-contract.json",
            DESIGN_CONTRACT_SHA256_V5,
        ),
        "discourse_request_schema_sha256": (
            contract_root / "discourse-request.schema.json",
            REQUEST_SCHEMA_SHA256_V5,
        ),
        "discourse_plan_schema_sha256": (
            contract_root / "discourse-plan.schema.json",
            PLAN_SCHEMA_SHA256_V5,
        ),
        "publication_record_schema_sha256": (
            contract_root / "publication-record.schema.json",
            PUBLICATION_SCHEMA_SHA256_V5,
        ),
    }
    result: dict[str, str] = {}
    for name, (path, digest) in expected.items():
        if is_link_like(path) or not path.is_file():
            raise EvidenceError("unsafe_contract_artifact", str(path))
        actual = sha256_file(path)
        require_sha256(actual, name)
        if actual != digest:
            raise EvidenceError("contract_identity_mismatch", name)
        result[name] = actual
    return result
