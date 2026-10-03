# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

from voxelscope.canonical import canonical_json_bytes, load_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "research/evidence-communication-v5-discourse-planner-contract-v1"
CONTRACT_PATH = CONTRACT_ROOT / "design-contract.json"
REQUEST_SCHEMA_PATH = CONTRACT_ROOT / "discourse-request.schema.json"
PLAN_SCHEMA_PATH = CONTRACT_ROOT / "discourse-plan.schema.json"
PUBLICATION_SCHEMA_PATH = CONTRACT_ROOT / "publication-record.schema.json"


def _load(path: Path) -> dict[str, Any]:
    value = load_json(path)
    assert isinstance(value, dict)
    return value


def test_v5_contract_artifacts_are_canonical_and_digest_pinned() -> None:
    expected = {
        CONTRACT_PATH: "7389b004f6a75ca00399573bf3ca45074f2941e4ac9772385d88f69a422a4fae",
        REQUEST_SCHEMA_PATH: ("a347247bfaa818738f748de61633b47ccea640083df9aebde7f262fc54cfa038"),
        PLAN_SCHEMA_PATH: ("806d847af7dcf0484d57291a468d3e6141e557547c8bb293ae21be272d1d6c1a"),
        PUBLICATION_SCHEMA_PATH: (
            "c4e85b0e5e1c821f29431487782df7f1247a8028d3c619a2018fa57e21666dd3"
        ),
    }
    for path, digest in expected.items():
        value = load_json(path)
        assert path.read_bytes() == canonical_json_bytes(value)
        assert sha256_file(path) == digest

    contract = _load(CONTRACT_PATH)
    assert contract["schema_identities"] == {
        "discourse_plan_schema_sha256": expected[PLAN_SCHEMA_PATH],
        "discourse_request_schema_sha256": expected[REQUEST_SCHEMA_PATH],
        "publication_record_schema_sha256": expected[PUBLICATION_SCHEMA_PATH],
    }


def test_v5_audience_is_caller_owned_and_not_a_plan_decision() -> None:
    request_schema = _load(REQUEST_SCHEMA_PATH)
    plan_schema = _load(PLAN_SCHEMA_PATH)

    assert request_schema["additionalProperties"] is False
    assert plan_schema["additionalProperties"] is False
    assert request_schema["properties"]["audience_profile"] == {"const": "research_technical"}
    assert "audience_profile" in request_schema["required"]
    assert "allowed_audience_profiles" not in request_schema["properties"]
    assert "default_audience_profile" not in request_schema["properties"]
    assert "selected_audience_profile" not in plan_schema["properties"]
    assert {"ordering_constraints", "caveat_constraints"} <= set(request_schema["required"])
    assert (
        request_schema["properties"]["ordering_constraints"]["items"]["additionalProperties"]
        is False
    )
    assert (
        request_schema["properties"]["caveat_constraints"]["items"]["additionalProperties"] is False
    )
    assert set(plan_schema["properties"]) == {
        "caveat_anchor_decisions",
        "ordered_unit_ids",
        "plan_mode",
        "request_sha256",
        "schema_version",
        "selected_optional_unit_ids",
    }


def test_v5_baseline_is_exact_with_frozen_lexicographic_objective() -> None:
    baseline = _load(CONTRACT_PATH)["deterministic_baseline"]

    assert baseline["status"] == "first_class_strong_baseline"
    assert baseline["same_inputs_and_renderer"] is True
    assert baseline["algorithm"].startswith(
        "enumerate the complete finite feasible discourse-plan set"
    )
    assert "Pareto-optimal" in baseline["optimality_claim"]
    assert baseline["objective"]["comparison"] == "lexicographic"
    assert baseline["objective"]["directions"] == [
        {
            "direction": "require",
            "objective": "all feasibility gates",
            "rank": 1,
        },
        {
            "direction": "maximize",
            "objective": ("sum of request-pinned integer utility for selected optional units"),
            "rank": 2,
        },
        {
            "direction": "minimize",
            "objective": "exact UTF-8 byte length of deterministic renderer output",
            "rank": 3,
        },
        {
            "direction": "minimize_lexicographically",
            "objective": "canonical decision key",
            "rank": 4,
        },
    ]
    assert baseline["objective"]["tie_break"] == {
        "canonical_decision_key_fields": [
            "ordered_unit_ids",
            "selected_optional_unit_ids sorted by unit_id",
            ("caveat_anchor_decisions sorted by caveat_unit_id then anchor_id then placement"),
        ],
        "encoding": "repository canonical JSON bytes",
        "excluded_fields": ["plan_mode", "request_sha256", "schema_version"],
    }


def test_v5_model_is_ineligible_before_inference_under_current_endpoints() -> None:
    contract = _load(CONTRACT_PATH)
    eligibility = contract["model_eligibility"]

    assert eligibility["current_eligible"] is False
    assert eligibility["gate_result"] == "model_inference_forbidden"
    assert eligibility["no_current_human_endpoint"] is True
    assert eligibility["evaluation_time"].startswith("before any model process")
    assert contract["candidate_model_protocol"]["status"] == ("ineligible_under_current_objectives")
    assert contract["decision_rule"]["current_conclusion"] == (
        "implement deterministic v5 first; prove exact deterministic dominance; "
        "stop before model inference unless eligibility is independently established"
    )
    assert "properly blinded human-evaluation protocol" in " ".join(
        eligibility["future_eligibility_requirements"]
    )

    endpoints = {item["id"]: item for item in contract["endpoints"]}
    assert endpoints["optional_unit_utility"]["role"] == ("exact_deterministic_objective")
    assert endpoints["emitted_utf8_bytes"]["direction"] == ("minimize_after_maximum_utility")
    for endpoint_id in ("planner_latency", "input_tokens", "output_tokens"):
        assert endpoints[endpoint_id]["role"] == "cost_only"
    assert endpoints["plan_variance"]["role"] == "model_behavior_diagnostic"


def test_v5_publication_record_isolates_ineligible_model_path() -> None:
    schema = _load(PUBLICATION_SCHEMA_PATH)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["published_variant"] == {"const": "deterministic_exact_baseline"}
    assert schema["properties"]["protocol_validity"] == {"const": "valid"}
    conditional = schema["allOf"][0]
    assert conditional["if"]["properties"]["model_eligibility_status"] == {
        "const": "ineligible_deterministic_dominance"
    }
    assert conditional["then"]["properties"] == {
        "model_input_sha256": {"type": "null"},
        "model_outcome": {"const": "ineligible_not_requested"},
        "model_output_sha256": {"type": "null"},
        "model_plan_sha256": {"type": "null"},
    }


def test_v5_contract_preserves_scope_and_v4_audit_basis() -> None:
    contract = _load(CONTRACT_PATH)
    assert contract["analysis_boundary"] == {
        "biomedical_data_acquisition_authorized": False,
        "full_v5_compiler_implemented": False,
        "human_evaluation_declared": False,
        "model_execution_authorized": False,
        "observed_v5_evidence_presented": False,
        "prospective_study_execution_declared": False,
        "status": "prospective_design_contract_only",
    }
    assert contract["v4_audit_basis"] == {
        "affirmative_clinical_process_statement": 4,
        "ambiguous_or_context_dependent": 92,
        "audit_sha256": ("a5735aad10d496950429282613cf15f04bb00bef7f773eda874979224c5f8573"),
        "explicit_negation_or_boundary_disclaimer": 12,
        "interpretation": (
            "post-hoc mechanical labels are not substantive unsafe-text truth and "
            "do not modify the frozen v4 result"
        ),
        "lexical_exclusions": 108,
        "negation_and_affirmative_feature_overlap": 0,
    }
    assert set(contract["no_go_boundaries"]) == {
        "biological causality",
        "clinical use",
        "diagnosis",
        "druggability",
        "patient-specific claims",
        "production-readiness claims",
        "prognosis",
        "ranking",
        "real-data acquisition",
        "superiority claims",
        "therapeutic-target designation",
        "treatment",
    }
