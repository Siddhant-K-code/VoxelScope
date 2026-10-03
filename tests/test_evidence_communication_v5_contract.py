# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

from voxelscope.canonical import load_json, sha256_file
from voxelscope.records import require_list, require_object, strict_fields

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "research/evidence-communication-v5-discourse-planner-contract-v1"
CONTRACT = CONTRACT_ROOT / "design-contract.json"
DOC = ROOT / "docs/evidence-communication-v5-discourse-planner-contract.md"


def _property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        names.update(properties)
    for value in schema.values():
        if isinstance(value, dict):
            names.update(_property_names(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    names.update(_property_names(item))
    return names


def test_v5_design_contract_is_canonical_strict_and_schema_pinned() -> None:
    contract = strict_fields(
        require_object(load_json(CONTRACT), "v5 design contract"),
        {
            "atomic_publication",
            "baseline",
            "claim_boundary",
            "contract_id",
            "custody",
            "decision_rule",
            "endpoints",
            "execution_status",
            "identity_pins_required",
            "model_interface",
            "non_comparable_endpoints",
            "research_question",
            "schema_artifacts",
            "schema_version",
            "state_transitions",
            "status",
            "stop_rules",
            "trust_boundaries",
        },
        "V5DesignContract",
    )
    assert (
        contract["schema_version"]
        == "voxelscope/evidence-communication-discourse-planner-design-contract/v1"
    )
    assert contract["status"] == "prospective_design_contract_pending_implementation_review"
    assert contract["research_question"] == (
        "Does a local model add measurable communication value beyond deterministic "
        "code once evidence claims and final prose are code-owned?"
    )

    artifacts = require_list(contract["schema_artifacts"], "schema_artifacts")
    for item in artifacts:
        artifact = strict_fields(
            require_object(item, "schema artifact"),
            {"path", "sha256"},
            "SchemaArtifact",
        )
        assert sha256_file(ROOT / artifact["path"]) == artifact["sha256"]
        load_json(ROOT / artifact["path"])


def test_model_plan_schema_has_only_bounded_enumerated_decisions() -> None:
    schema = require_object(
        load_json(CONTRACT_ROOT / "discourse-plan.schema.json"),
        "discourse plan schema",
    )
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "caveat_placements",
        "ordered_unit_ids",
        "request_id",
        "schema_version",
        "selected_audience_profile",
        "selected_optional_unit_ids",
        "terminal_state",
    }
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["selected_audience_profile"]["enum"] == [
        "researcher_concise",
        "researcher_detailed",
    ]
    assert not _property_names(schema) & {
        "citation",
        "clinical_language",
        "evidence_value",
        "final_sentence",
        "source_id",
        "text",
    }


def test_baseline_is_first_class_and_no_model_is_the_only_publication() -> None:
    contract = require_object(load_json(CONTRACT), "v5 design contract")
    baseline = strict_fields(
        require_object(contract["baseline"], "baseline"),
        {
            "algorithm_id",
            "audience_profile_rule",
            "caveat_placement_rule",
            "input_schema",
            "optional_selection_rule",
            "ordering_rule",
            "output_schema",
            "strength",
        },
        "BaselineContract",
    )
    model = require_object(contract["model_interface"], "model_interface")
    assert baseline["output_schema"] == model["output_schema"]
    assert baseline["strength"] == (
        "first_class_comparator_using_identical_inputs_plan_schema_verifier_renderer_and_budgets"
    )
    publication = require_object(
        load_json(CONTRACT_ROOT / "publication-record.schema.json"),
        "publication schema",
    )
    assert publication["properties"]["published_variant"] == {"const": "deterministic_baseline"}
    assert contract["decision_rule"]["no_value_conclusion"] == (
        "remove_the_local_model_from_this_discourse_planning_layer"
    )


def test_contract_covers_required_endpoints_and_preserves_no_go_boundaries() -> None:
    contract = require_object(load_json(CONTRACT), "v5 design contract")
    endpoint_names = {item["name"] for item in require_list(contract["endpoints"], "endpoints")}
    assert endpoint_names == {
        "budget_compliance",
        "citation_validity",
        "deterministic_baseline_parity_and_delta",
        "emitted_size_bytes",
        "mandatory_caveat_coverage",
        "mandatory_fact_coverage",
        "optional_unit_selection",
        "optional_unit_selection_utility",
        "planner_input_tokens",
        "planner_latency_ms",
        "planner_output_tokens",
        "protocol_validity",
        "semantic_plan_variance",
    }
    assert set(contract["claim_boundary"]) == {
        "biological_causality",
        "clinical_use",
        "diagnosis",
        "druggability",
        "model_superiority",
        "patient_specific_claims",
        "production_readiness",
        "prognosis",
        "protein_ranking",
        "real_data_acquisition",
        "therapeutic_target_designation",
        "treatment",
    }
    assert contract["execution_status"] == {
        "implementation_authorized": False,
        "observed_fixture_is_model_evidence": False,
        "prospective_study_execution_declared": False,
        "study_run_authorized": False,
    }
    assert "```mermaid" in DOC.read_text(encoding="utf-8")
