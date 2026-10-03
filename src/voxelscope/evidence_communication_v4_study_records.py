# SPDX-License-Identifier: Apache-2.0
"""Strict records for the prospective v4 local-model comparison study."""

from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    load_json,
    require_sha256,
    safe_relative_path,
    sha256_file,
    write_json,
)
from .evidence_benchmark_records import RunnerConfiguration
from .evidence_communication_records import ModelIdentity, PromptIdentity
from .records import (
    require_bool,
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

STUDY_DECLARATION_SCHEMA_V4 = "voxelscope/evidence-communication-local-model-study-declaration/v4"
STUDY_DECLARATION_RECEIPT_SCHEMA_V4 = (
    "voxelscope/evidence-communication-local-model-study-declaration-receipt/v4"
)
STUDY_EXECUTION_BINDING_SCHEMA_V4 = "voxelscope/evidence-communication-local-model-study-binding/v4"
STUDY_ATTEMPT_SCHEMA_V4 = "voxelscope/evidence-communication-local-model-study-attempt/v4"
STUDY_ID_V4 = "gbm-evidence-communication-qwen3-8b-q8-study-v4"
DECLARATION_PATH_V4 = (
    "research/gbm-evidence-communication-qwen3-8b-q8-study-v4/study-declaration.json"
)
DECLARATION_RECEIPT_PATH_V4 = (
    "research/gbm-evidence-communication-qwen3-8b-q8-study-v4/study-declaration-receipt.json"
)
BENCHMARK_OUTPUT_PATH_V4 = "research/gbm-evidence-communication-qwen3-8b-q8-study-v4/benchmark"
ATTEMPT_MARKER_PATH_V4 = "research/gbm-evidence-communication-qwen3-8b-q8-study-v4/attempt.json"

BASE_COMMIT = "c66eca8f072ba23d74b59455f6f4074bfb758b08"
BASE_TREE_SHA1 = "f8a68afc0bb15f06f39870d15bed47edeca0ac37"
V3_MERGE_COMMIT = "936549dcbf8c563cdef896bdb442a7629dcd260d"
V3_STUDY_TREE_SHA1 = "7e5fb406d7216868e1491e5ae9420b17c9242533"
MODEL_TAG = "qwen3:8b-q8_0"
MODEL_MANIFEST_SHA256 = "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130"
OLLAMA_RUNTIME_VERSION = "0.35.1"
ATLAS_SHA256 = "5a9f1a2078d951b3f10ff909b5c341e4071b5ed92050283c31859521b8c88647"
FIXTURE_SHA256 = "0db99c6ef56723df0cc18b4d692049cfb70aca196ee06601c72f99a2fb1ce0a8"

REQUIRED_SOURCE_ROLES = {
    "src/voxelscope/evidence_communication.py": ("shared_v3_planner_dependency",),
    "src/voxelscope/evidence_communication_records.py": ("shared_record_dependency",),
    "src/voxelscope/evidence_benchmark_records.py": ("shared_benchmark_records",),
    "src/voxelscope/evidence_communication_v4.py": (
        "compiler",
        "json_schema",
        "prompt",
        "runner",
        "verifier",
    ),
    "src/voxelscope/evidence_communication_v4_cli.py": ("cli",),
    "src/voxelscope/evidence_communication_v4_records.py": ("v4_records",),
    "src/voxelscope/evidence_communication_v4_study_records.py": (
        "declaration_schema",
        "execution_binding",
    ),
    "research/gbm-evidence-communication-benchmark-v4/benchmark-fixtures.json": ("fixture",),
    "research/gbm-evidence-atlas-v1/source-manifest.json": ("atlas_manifest",),
    "research/gbm-evidence-atlas-v1/evidence-items.json": ("atlas_evidence_items",),
    "research/gbm-evidence-atlas-v1/identifier-map.json": ("atlas_identifier_map",),
}

FROZEN_SIZE_PREFLIGHT = {
    "atlas_sha256": ATLAS_SHA256,
    "cases": [
        {
            "case_id": "supported-agreement",
            "prompt_utf8_bytes": 3831,
            "request_body_canonical_json_bytes": 5578,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "cross-layer-disagreement",
            "prompt_utf8_bytes": 3831,
            "request_body_canonical_json_bytes": 5578,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "missing-modality",
            "prompt_utf8_bytes": 3831,
            "request_body_canonical_json_bytes": 5578,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "unsupported-join",
            "prompt_utf8_bytes": 3834,
            "request_body_canonical_json_bytes": 5581,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "stale-evidence",
            "prompt_utf8_bytes": 3831,
            "request_body_canonical_json_bytes": 5578,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "restricted-evidence",
            "prompt_utf8_bytes": 3820,
            "request_body_canonical_json_bytes": 5567,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "not-measured-versus-negative",
            "prompt_utf8_bytes": 3830,
            "request_body_canonical_json_bytes": 5577,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "source-reported-target-evidence",
            "prompt_utf8_bytes": 3831,
            "request_body_canonical_json_bytes": 5578,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
        {
            "case_id": "prohibited-clinical-causal",
            "prompt_utf8_bytes": 3828,
            "request_body_canonical_json_bytes": 5575,
            "schema_canonical_json_bytes": 1492,
            "skeleton_count": 15,
        },
    ],
    "generation_requests": 0,
    "fixture_sha256": FIXTURE_SHA256,
    "measurement_method": (
        "len_utf8_for_prompt_and_len_voxelscope_canonical_json_bytes_for_"
        "schema_and_complete_nonstreaming_request_body"
    ),
    "observed_inference_counters": "unavailable_no_generation",
    "schema_version": "voxelscope/evidence-communication-local-model-size-preflight/v4",
    "status": "measured_without_generation",
    "token_estimation": (
        "not_computed_prompt_utf8_bytes_are_recorded_as_deterministic_inputs_"
        "but_are_not_token_counts"
    ),
}

FROZEN_REPOSITORY = {
    "base_commit": BASE_COMMIT,
    "base_tree_sha1": BASE_TREE_SHA1,
    "base_relation": "exact_main_commit_containing_merged_pr_19",
    "implementation_head": "e11ecb9c701641e53ff44340cf6058b128177fd1",
    "implementation_merge_base": "936549dcbf8c563cdef896bdb442a7629dcd260d",
    "implementation_relation": ("squash_source_tree_identical_to_merge_tree_not_commit_ancestor"),
    "implementation_tree_sha1": BASE_TREE_SHA1,
    "merged_pr": {
        "checks": [
            "quality:SUCCESS",
            "test (ubuntu-latest, 3.12):SUCCESS",
            "test (ubuntu-latest, 3.13):SUCCESS",
            "test (macos-latest, 3.13):SUCCESS",
            "test (windows-latest, 3.13):SUCCESS",
        ],
        "number": 19,
        "state": "MERGED",
    },
}

FROZEN_PROMPTS = (
    {
        "prompt_id": "voxelscope/evidence-communication-bounded-draft/v4",
        "sha256": "e18add50a76c973389aa654a2bc5bf72a807cc7490ed70ac63dbb551db119719",
    },
    {
        "prompt_id": "voxelscope/evidence-communication-bounded-draft/v4:clean",
        "sha256": "4d1309291cef682af2cda1c3963acfaca9703fd3c91fc65e967ba3edc5ead392",
    },
    {
        "prompt_id": (
            "voxelscope/evidence-communication-bounded-draft/v4:not_measured_as_negative"
        ),
        "sha256": "6cc1c9f7c2b48fd026a617ef0dc7b3448e5e537e625c81454bcab372975f9a24",
    },
    {
        "prompt_id": (
            "voxelscope/evidence-communication-bounded-draft/v4:prohibited_clinical_or_causal"
        ),
        "sha256": "31cba0c5e469ec126fe7847306f3c607ea0c43a84492636ab965070dfee5b0c4",
    },
    {
        "prompt_id": (
            "voxelscope/evidence-communication-bounded-draft/v4:restricted_without_source"
        ),
        "sha256": "2dc11e81e1ab22ccd9d1d4e32a32a2d1d814e8b144e1303df1dde8ee46f43238",
    },
    {
        "prompt_id": ("voxelscope/evidence-communication-bounded-draft/v4:unknown_source"),
        "sha256": "8f78426160c2f1f9b154c405ad3eb9750f28efefd53551ba2583c7af5b5c907e",
    },
)

FROZEN_CONTRACTS = {
    "benchmark_fixture_schema": "voxelscope/evidence-communication-benchmark-fixture/v4",
    "benchmark_index_schema": "voxelscope/evidence-communication-benchmark-index/v4",
    "benchmark_receipt_schema": "voxelscope/evidence-communication-benchmark-receipt/v4",
    "benchmark_schema": "voxelscope/evidence-communication-benchmark/v4",
    "draft_schema": "voxelscope/evidence-communication-draft-plan/v4",
    "format": "exact_json_schema_object",
    "prompts": list(FROZEN_PROMPTS),
    "request_schema": "voxelscope/evidence-communication-request/v4",
    "transformation_id": "voxelscope/evidence-communication-compiler/v4",
    "verifier_version": "voxelscope/evidence-communication-verifier/v4",
}

FROZEN_MODEL_RUNTIME = {
    "model_identity": {
        "adapter": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "model": MODEL_TAG,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "runtime": "ollama",
        "runtime_version": OLLAMA_RUNTIME_VERSION,
    },
    "preflight": {
        "generation_permitted": False,
        "identity_check_count": 2,
        "metadata_methods": ["GET /api/version", "GET /api/tags"],
    },
    "protocol": {
        "authentication": "forbidden",
        "done_required": True,
        "endpoint_locality": "localhost_only",
        "generation_method": "POST /api/generate",
        "manifest_tag_runtime_checks": (
            "at_start_and_immediately_before_and_after_every_generation"
        ),
        "network_outside_localhost": "forbidden",
        "redirects": "forbidden",
        "response_model_must_equal_tag": True,
        "seed": "unset",
        "seed_policy": (
            "intentionally_unset_to_measure_observed_local_runtime_semantic_"
            "variance_at_temperature_zero"
        ),
        "stream": False,
        "think_top_level": False,
    },
    "runner_configuration": {
        "context_window": 16384,
        "declared_environment": [
            {
                "name": "HOST_ARCHITECTURE",
                "provenance": "declared_unverified",
                "value": "arm64",
            },
            {
                "name": "HOST_CHIP",
                "provenance": "declared_unverified",
                "value": "Apple-M5-Pro",
            },
            {
                "name": "HOST_LOGICAL_CPU_CORES",
                "provenance": "declared_unverified",
                "value": "18",
            },
            {
                "name": "HOST_MACHINE_MODEL",
                "provenance": "declared_unverified",
                "value": "Mac17,8",
            },
            {
                "name": "HOST_MEMORY_BYTES",
                "provenance": "declared_unverified",
                "value": "25769803776",
            },
            {
                "name": "HOST_OS",
                "provenance": "declared_unverified",
                "value": "macOS-27.0.1-26A434",
            },
            {
                "name": "OLLAMA_FLASH_ATTENTION",
                "provenance": "declared_unverified",
                "value": "1",
            },
            {
                "name": "OLLAMA_KV_CACHE_TYPE",
                "provenance": "declared_unverified",
                "value": "q8_0",
            },
            {
                "name": "PYTHON_VERSION",
                "provenance": "declared_unverified",
                "value": "3.13.15",
            },
        ],
        "endpoint_locality": "localhost_only",
        "json_mode": "strict_json",
        "options": [{"name": "num_ctx", "value": 16384}],
        "temperature": 0.0,
        "thinking_enabled": False,
        "timeout_seconds": 300.0,
    },
}

FROZEN_DESIGN = {
    "attempt_count": 1,
    "case_count": 9,
    "case_source": "same_frozen_nine_conceptual_cases_as_v3",
    "fixture_sha256": FIXTURE_SHA256,
    "manual_repair": "forbidden",
    "no_extra_warmup": True,
    "no_retry": True,
    "no_selective_rerun": True,
    "real_biomedical_data": "forbidden",
    "repeat_count": 2,
    "repeat_policy": "fixture_defined",
    "synthetic_atlas_sha256": ATLAS_SHA256,
    "total_request_count": 18,
}

FROZEN_ENDPOINTS = (
    {
        "denominator": "expected skeleton count across all 18 requests",
        "kind": "primary",
        "name": "task_skeleton_coverage",
        "numerator": "returned skeleton IDs matched uniquely to expected skeleton IDs",
    },
    {
        "denominator": "returned skeleton entry count across all 18 requests",
        "kind": "primary",
        "name": "unknown_task_rate",
        "numerator": "returned entries whose skeleton ID is not expected",
    },
    {
        "denominator": "returned skeleton entry count across all 18 requests",
        "kind": "primary",
        "name": "duplicate_task_rate",
        "numerator": "returned entries duplicating an already returned skeleton ID",
    },
    {
        "denominator": "expected skeleton count across all 18 requests",
        "kind": "primary",
        "name": "omitted_task_rate",
        "numerator": "expected skeleton IDs absent from model output",
    },
    {
        "denominator": "18 frozen requests",
        "kind": "primary",
        "name": "invalid_model_output_rate",
        "numerator": "requests closed as digest-safe invalid model output",
    },
    {
        "denominator": "returned skeleton entry count across all 18 requests",
        "kind": "primary",
        "name": "unsupported_claim_rate",
        "numerator": "deterministic verifier exclusions",
    },
    {
        "denominator": "required fact-plan entries across all 18 requests",
        "kind": "primary",
        "name": "emitted_fact_coverage",
        "numerator": "required fact-plan entries represented in final verified prose",
    },
    {
        "denominator": "required caveat-plan entries across all 18 requests",
        "kind": "primary",
        "name": "emitted_caveat_coverage",
        "numerator": "required caveat-plan entries represented in final verified prose",
    },
    {
        "denominator": "required fact-plan entries across all 18 requests",
        "kind": "primary",
        "name": "verified_draft_fact_coverage",
        "numerator": "required fact-plan entries represented by verified expanded skeletons",
    },
    {
        "denominator": "required caveat-plan entries across all 18 requests",
        "kind": "primary",
        "name": "verified_draft_caveat_retention",
        "numerator": "required caveat-plan entries represented by verified expanded skeletons",
    },
    {
        "denominator": "all source citations attached to verified claims",
        "kind": "primary",
        "name": "evidence_citation_validity",
        "numerator": "verified-claim citations resolving to frozen atlas evidence",
    },
    {
        "denominator": "nine within-case repeat pairs",
        "kind": "primary",
        "name": "replay_semantic_variance",
        "numerator": "repeat pairs with unequal semantic artifact SHA-256",
    },
    {
        "denominator": "18 frozen requests",
        "kind": "primary",
        "name": "accepted_terminal_outcome_rate",
        "numerator": "runs whose terminal state is accepted",
    },
    {
        "denominator": "18 frozen requests",
        "kind": "primary",
        "name": "partially_excluded_terminal_outcome_rate",
        "numerator": "runs whose terminal state is partially_excluded",
    },
    {
        "denominator": "18 frozen requests",
        "kind": "primary",
        "name": "refused_terminal_outcome_rate",
        "numerator": "runs whose terminal state is refused",
    },
    {
        "denominator": "runner-recorded completed requests",
        "kind": "secondary",
        "name": "latency_ms",
        "numerator": (
            "per-request wall-clock milliseconds; summarized as count/minimum/mean/maximum"
        ),
    },
    {
        "denominator": "requests for which Ollama reports prompt_eval_count",
        "kind": "secondary",
        "name": "input_tokens",
        "numerator": "reported prompt_eval_count; unavailable values are not zero",
    },
    {
        "denominator": "requests for which Ollama reports eval_count",
        "kind": "secondary",
        "name": "output_tokens",
        "numerator": "reported eval_count; unavailable values are not zero",
    },
    {
        "denominator": "not applicable",
        "kind": "secondary",
        "name": "peak_memory_mb",
        "numerator": "unavailable because the declared runner does not attest peak memory",
    },
    {
        "denominator": "not applicable",
        "kind": "secondary",
        "name": "peak_metal_memory_mb",
        "numerator": "unavailable because the declared runner does not attest peak Metal memory",
    },
)

FROZEN_V3_FACTS = {
    "accepted_or_partially_excluded": 0,
    "evidence_citation_validity": {"denominator": 342, "numerator": 342, "value": 1.0},
    "invalid_outer_json_type_count": 4,
    "replay_semantic_variance": {"denominator": 9, "numerator": 0, "value": 0.0},
    "refused": 18,
    "request_count": 18,
    "status": "closed_immutable_historical_comparator",
}

FROZEN_COMPARATOR_MATRIX = (
    {
        "classification": "directly_comparable",
        "v3_field": "invalid_model_output_rate",
        "v4_field": "invalid_model_output_rate",
        "rule": "side_by_side_descriptive_only_same_request_denominator",
    },
    {
        "classification": "directly_comparable",
        "v3_field": "emitted_fact_coverage",
        "v4_field": "emitted_fact_coverage",
        "rule": "side_by_side_descriptive_only_same_fact_plan_denominator",
    },
    {
        "classification": "directly_comparable",
        "v3_field": "emitted_caveat_coverage",
        "v4_field": "emitted_caveat_coverage",
        "rule": "side_by_side_descriptive_only_same_caveat_plan_denominator",
    },
    {
        "classification": "directly_comparable",
        "v3_field": "replay_semantic_variance",
        "v4_field": "replay_semantic_variance",
        "rule": "side_by_side_descriptive_only_same_nine_repeat_pair_denominator",
    },
    {
        "classification": "transformed",
        "v3_field": "verified_draft_fact_coverage",
        "v4_field": "verified_draft_fact_coverage",
        "rule": "v4_uses_trusted_skeleton_expansion",
    },
    {
        "classification": "transformed",
        "v3_field": "verified_draft_caveat_retention",
        "v4_field": "verified_draft_caveat_retention",
        "rule": "v4_uses_trusted_skeleton_expansion_and_trusted_caveat_copy",
    },
    {
        "classification": "transformed",
        "v3_field": "verifier_counts",
        "v4_field": "accepted_partially_excluded_refused_terminal_outcomes",
        "rule": "interface_and_verifier_changed",
    },
    {
        "classification": "transformed",
        "v3_field": "latency_ms_input_tokens_output_tokens",
        "v4_field": "latency_ms_input_tokens_output_tokens",
        "rule": "same_units_but_prompt_schema_and_generated_contract_changed",
    },
    {
        "classification": "not_comparable",
        "v3_field": "none",
        "v4_field": "task_skeleton_coverage_unknown_duplicate_omitted_rates",
        "rule": "v3_has_no_skeleton_task_contract",
    },
    {
        "classification": "not_comparable",
        "v3_field": "unsupported_claim_rate",
        "v4_field": "unsupported_claim_rate",
        "rule": "denominator_and_model_owned_semantic_unit_changed",
    },
    {
        "classification": "not_comparable",
        "v3_field": "evidence_citation_validity",
        "v4_field": "evidence_citation_validity",
        "rule": "v4_citations_are_attached_by_trusted_expansion_not_model_output",
    },
    {
        "classification": "not_comparable",
        "v3_field": "peak_memory_mb_peak_metal_memory_mb",
        "v4_field": "peak_memory_mb_peak_metal_memory_mb",
        "rule": "both_remain_unavailable_not_zero",
    },
)

FROZEN_COMPARATOR = {
    "artifacts": [
        {
            "path": (
                "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/study-declaration.json"
            ),
            "sha256": "c4ee1f8669868410e86107e5e9ee0295e4870e746e90cf996d68729705149cdd",
        },
        {
            "path": (
                "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/result-summary.json"
            ),
            "sha256": "87a15a79eeec7a0ea988ff40d953d9f6d5f831b5568b20a8dc94782c107cc8c0",
        },
        {
            "path": (
                "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/failure-analysis.json"
            ),
            "sha256": "ec93745e5ca8b30d866ec64959ce06dc523832b2ac93156cecb2e36e48e5ac15",
        },
        {
            "path": (
                "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/benchmark/benchmark.json"
            ),
            "sha256": "66f26f41d8e7f4bf078e93d71aef577081a2fd5312e1cf1e8200d2d5c2fcf087",
        },
        {
            "path": (
                "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/benchmark/receipt.json"
            ),
            "sha256": "b454163e255cfe85d7412c8a90bcca31ce0840aa23675bb4b2e4ccf9174a61a6",
        },
    ],
    "facts": FROZEN_V3_FACTS,
    "merge_commit": V3_MERGE_COMMIT,
    "policy": [
        "historical_fixed_values_only",
        "no_v3_rerun_change_or_reinterpretation",
        "side_by_side_only_when_matrix_marks_semantically_comparable",
        "no_causal_model_quality_safety_generalization_significance_or_superiority_claim",
    ],
    "study_id": "gbm-evidence-communication-qwen3-8b-q8-study-v1",
    "study_tree_sha1": V3_STUDY_TREE_SHA1,
    "v3_to_v4_fields": list(FROZEN_COMPARATOR_MATRIX),
}

FROZEN_ENVIRONMENT = {
    "declared_unverified": [
        "host architecture arm64",
        "host chip Apple M5 Pro",
        "host logical CPU count 18",
        "host machine model Mac17,8",
        "host memory 25769803776 bytes",
        "host operating system macOS 27.0.1 build 26A434",
        "Python 3.13.15",
        "Ollama service process environment and accelerator state",
    ],
    "qualified_before_declaration_but_not_api_attested": [
        "OLLAMA_FLASH_ATTENTION=1",
        "OLLAMA_KV_CACHE_TYPE=q8_0",
    ],
    "measurement_policy": {
        "input_tokens": "available_only_when_runner_records_prompt_eval_count",
        "latency_ms": "runner_wall_clock_required_for_each_completed_generation",
        "output_tokens": "available_only_when_runner_records_eval_count",
        "peak_memory_mb": "unavailable",
        "peak_metal_memory_mb": "unavailable",
    },
    "spend": "none",
    "verified_by_preflight": [
        "Ollama runtime version",
        "unique local tag resolution",
        "full model manifest SHA-256",
        "localhost endpoint policy",
    ],
    "cloud_provider": "none",
}

FROZEN_EXECUTION = {
    "acceptance_rules": [
        "all_18_frozen_requests_are_consumed_in_fixture_order",
        "model_content_noncompliance_is_digest_safe_invalid_output_only",
        "complete_atomic_per_run_custody_and_exact_file_set_receipt_closure",
        "offline_replay_passes_before_result_publication",
    ],
    "attempt_marker_path": ATTEMPT_MARKER_PATH_V4,
    "authorization_boundary": (
        "independent_review_and_merge_of_this_declaration_then_one_explicit_operator_"
        "start_in_a_fresh_clean_session_bound_to_the_merged_declaration_sha256"
    ),
    "benchmark_output_path": BENCHMARK_OUTPUT_PATH_V4,
    "declaration_path": DECLARATION_PATH_V4,
    "future_execution_command": (
        "uv run python -m voxelscope.evidence_communication_v4_cli benchmark "
        "--atlas build/gbm-atlas/atlas.json --fixtures "
        "research/gbm-evidence-communication-benchmark-v4/benchmark-fixtures.json "
        "--output research/gbm-evidence-communication-qwen3-8b-q8-study-v4/benchmark "
        "--runner ollama --declaration "
        "research/gbm-evidence-communication-qwen3-8b-q8-study-v4/study-declaration.json "
        "--repository-root . --authorize-study "
        "gbm-evidence-communication-qwen3-8b-q8-study-v4"
    ),
    "no_clobber": True,
    "no_rerun_after_generation_consumed": True,
    "partial_execution_policy": (
        "if_any_generation_is_consumed_but_all_18_runs_do_not_close_and_replay_the_"
        "whole_tree_is_unpublished_the_attempt_marker_remains_terminal_and_no_rerun_"
        "or_partial_result_publication_is_permitted"
    ),
    "protocol_or_custody_failure_interpretation": (
        "invalidates_the_comparison_and_is_not_a_model_quality_observation"
    ),
    "stop_rules": [
        "declaration_commit_source_prompt_compiler_verifier_fixture_or_atlas_drift",
        "runtime_tag_manifest_endpoint_transport_redirect_or_network_policy_drift",
        "outer_protocol_done_response_model_think_or_json_schema_drift",
        "output_or_attempt_marker_already_exists",
        "custody_file_set_receipt_or_offline_replay_failure",
    ],
    "warmup": "none",
}

FROZEN_CLAIM_BOUNDARY = (
    "research_only_synthetic_evidence",
    "no_clinical_claim",
    "no_diagnosis_claim",
    "no_prognosis_claim",
    "no_treatment_claim",
    "no_patient_specific_claim",
    "no_protein_ranking_claim",
    "no_therapeutic_target_claim",
    "no_druggability_claim",
    "no_biological_causality_claim",
    "no_model_quality_claim",
    "no_production_safety_claim",
    "no_real_data_claim",
)


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _require_sha1(value: str, field: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise EvidenceError("invalid_git_sha1", f"{field} must be lowercase Git SHA-1")


def _require_exact(value: Any, expected: Any, code: str) -> None:
    if value != expected:
        raise EvidenceError(code, "declaration differs from the prospectively frozen value")


@dataclass(frozen=True)
class StudySourceIdentityV4:
    git_blob_sha1: str
    path: str
    roles: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        _require_sha1(self.git_blob_sha1, "git_blob_sha1")
        require_sha256(self.sha256, "source sha256")
        if not self.roles or self.roles != tuple(sorted(set(self.roles))):
            raise EvidenceError("invalid_source_roles", self.path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudySourceIdentityV4:
        value = strict_fields(
            data,
            {"git_blob_sha1", "path", "roles", "sha256"},
            "StudySourceIdentityV4",
        )
        return cls(
            require_string(value["git_blob_sha1"], "git_blob_sha1"),
            require_string(value["path"], "path"),
            tuple(
                require_string(role, "source role")
                for role in require_list(value["roles"], "roles")
            ),
            require_string(value["sha256"], "sha256"),
        )

    def verify(self, root: Path) -> None:
        path = ensure_no_symlink(root, safe_relative_path(self.path))
        if not path.is_file():
            raise EvidenceError("study_source_missing", self.path)
        if sha256_file(path) != self.sha256:
            raise EvidenceError("study_source_sha256_mismatch", self.path)
        if _git_blob_sha1(path.read_bytes()) != self.git_blob_sha1:
            raise EvidenceError("study_source_git_blob_mismatch", self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_blob_sha1": self.git_blob_sha1,
            "path": self.path,
            "roles": list(self.roles),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ProspectiveStudyDeclarationV4:
    claim_boundary: tuple[str, ...]
    comparator: dict[str, Any]
    contracts: dict[str, Any]
    declared_at: str
    design: dict[str, Any]
    endpoints: tuple[dict[str, Any], ...]
    environment: dict[str, Any]
    execution: dict[str, Any]
    model_runtime: dict[str, Any]
    repository: dict[str, Any]
    schema_version: str
    size_preflight: dict[str, Any]
    sources: tuple[StudySourceIdentityV4, ...]
    status: str
    study_id: str

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_DECLARATION_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.study_id != STUDY_ID_V4:
            raise EvidenceError("study_identity_mismatch", self.study_id)
        if self.status != "prospectively_frozen_pending_execution":
            raise EvidenceError("invalid_study_status", self.status)
        if self.declared_at != "2026-10-03T15:09:11Z":
            raise EvidenceError("declaration_time_drift", self.declared_at)
        paths = tuple(item.path for item in self.sources)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise EvidenceError("duplicate_or_unsorted_source_path", ",".join(paths))
        expected_paths = tuple(sorted(REQUIRED_SOURCE_ROLES))
        if paths != expected_paths:
            raise EvidenceError("study_source_set_mismatch", ",".join(paths))
        for source in self.sources:
            if source.roles != tuple(sorted(REQUIRED_SOURCE_ROLES[source.path])):
                raise EvidenceError("study_source_role_mismatch", source.path)
        _require_exact(self.repository, FROZEN_REPOSITORY, "repository_identity_drift")
        _require_exact(self.contracts, FROZEN_CONTRACTS, "contract_identity_drift")
        _require_exact(self.model_runtime, FROZEN_MODEL_RUNTIME, "model_runtime_drift")
        _require_exact(self.design, FROZEN_DESIGN, "study_design_drift")
        _require_exact(list(self.endpoints), list(FROZEN_ENDPOINTS), "endpoint_definition_drift")
        _require_exact(self.comparator, FROZEN_COMPARATOR, "v3_comparator_mutation")
        _require_exact(self.environment, FROZEN_ENVIRONMENT, "environment_declaration_drift")
        _require_exact(self.execution, FROZEN_EXECUTION, "execution_rule_drift")
        _require_exact(
            self.size_preflight,
            FROZEN_SIZE_PREFLIGHT,
            "size_preflight_drift",
        )
        _require_exact(
            self.claim_boundary,
            FROZEN_CLAIM_BOUNDARY,
            "claim_boundary_drift",
        )

    @property
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity.from_dict(
            require_object(self.model_runtime["model_identity"], "model_identity")
        )

    @property
    def runner_configuration(self) -> RunnerConfiguration:
        return RunnerConfiguration.from_dict(
            require_object(
                self.model_runtime["runner_configuration"],
                "runner_configuration",
            )
        )

    @property
    def prompt_identities(self) -> tuple[PromptIdentity, ...]:
        return tuple(
            PromptIdentity.from_dict(require_object(item, "prompt identity"))
            for item in require_list(self.contracts["prompts"], "prompts")
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProspectiveStudyDeclarationV4:
        value = strict_fields(
            data,
            {
                "claim_boundary",
                "comparator",
                "contracts",
                "declared_at",
                "design",
                "endpoints",
                "environment",
                "execution",
                "model_runtime",
                "repository",
                "schema_version",
                "size_preflight",
                "sources",
                "status",
                "study_id",
            },
            "ProspectiveStudyDeclarationV4",
        )
        endpoints = tuple(
            strict_fields(
                require_object(item, "endpoint"),
                {"denominator", "kind", "name", "numerator"},
                "EndpointDefinitionV4",
            )
            for item in require_list(value["endpoints"], "endpoints")
        )
        return cls(
            tuple(
                require_string(item, "claim boundary")
                for item in require_list(value["claim_boundary"], "claim_boundary")
            ),
            require_object(value["comparator"], "comparator"),
            require_object(value["contracts"], "contracts"),
            require_string(value["declared_at"], "declared_at"),
            require_object(value["design"], "design"),
            endpoints,
            require_object(value["environment"], "environment"),
            require_object(value["execution"], "execution"),
            require_object(value["model_runtime"], "model_runtime"),
            require_object(value["repository"], "repository"),
            require_string(value["schema_version"], "schema_version"),
            require_object(value["size_preflight"], "size_preflight"),
            tuple(
                StudySourceIdentityV4.from_dict(require_object(item, "source identity"))
                for item in require_list(value["sources"], "sources")
            ),
            require_string(value["status"], "status"),
            require_string(value["study_id"], "study_id"),
        )

    def verify_sources(self, root: Path) -> None:
        for source in self.sources:
            source.verify(root)
        for artifact in require_list(self.comparator["artifacts"], "comparator artifacts"):
            value = strict_fields(
                require_object(artifact, "comparator artifact"),
                {"path", "sha256"},
                "ComparatorArtifactV4",
            )
            path_string = require_string(value["path"], "comparator path")
            path = ensure_no_symlink(root, safe_relative_path(path_string))
            if not path.is_file():
                raise EvidenceError("v3_comparator_missing", path_string)
            if sha256_file(path) != require_string(value["sha256"], "comparator sha256"):
                raise EvidenceError("v3_comparator_mutation", path_string)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_boundary": list(self.claim_boundary),
            "comparator": self.comparator,
            "contracts": self.contracts,
            "declared_at": self.declared_at,
            "design": self.design,
            "endpoints": list(self.endpoints),
            "environment": self.environment,
            "execution": self.execution,
            "model_runtime": self.model_runtime,
            "repository": self.repository,
            "schema_version": self.schema_version,
            "size_preflight": self.size_preflight,
            "sources": [item.to_dict() for item in self.sources],
            "status": self.status,
            "study_id": self.study_id,
        }


def load_study_declaration_v4(path: Path, root: Path) -> ProspectiveStudyDeclarationV4:
    declaration = ProspectiveStudyDeclarationV4.from_dict(
        require_object(load_json(path), "prospective study declaration")
    )
    expected = ensure_no_symlink(root, safe_relative_path(DECLARATION_PATH_V4))
    if path.resolve() != expected.resolve():
        raise EvidenceError("study_declaration_path_mismatch", str(path))
    declaration.verify_sources(root)
    return declaration


def _prospective_study_declaration_value_v4(root: Path) -> dict[str, Any]:
    sources = []
    for path_string, roles in sorted(REQUIRED_SOURCE_ROLES.items()):
        path = ensure_no_symlink(root, safe_relative_path(path_string))
        if not path.is_file():
            raise EvidenceError("study_source_missing", path_string)
        data = path.read_bytes()
        sources.append(
            {
                "git_blob_sha1": _git_blob_sha1(data),
                "path": path_string,
                "roles": sorted(roles),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "claim_boundary": list(FROZEN_CLAIM_BOUNDARY),
        "comparator": copy.deepcopy(FROZEN_COMPARATOR),
        "contracts": copy.deepcopy(FROZEN_CONTRACTS),
        "declared_at": "2026-10-03T15:09:11Z",
        "design": copy.deepcopy(FROZEN_DESIGN),
        "endpoints": [copy.deepcopy(item) for item in FROZEN_ENDPOINTS],
        "environment": copy.deepcopy(FROZEN_ENVIRONMENT),
        "execution": copy.deepcopy(FROZEN_EXECUTION),
        "model_runtime": copy.deepcopy(FROZEN_MODEL_RUNTIME),
        "repository": copy.deepcopy(FROZEN_REPOSITORY),
        "schema_version": STUDY_DECLARATION_SCHEMA_V4,
        "size_preflight": copy.deepcopy(FROZEN_SIZE_PREFLIGHT),
        "sources": sources,
        "status": "prospectively_frozen_pending_execution",
        "study_id": STUDY_ID_V4,
    }


def freeze_study_declaration_v4(root: Path) -> StudyDeclarationReceiptV4:
    root = root.resolve()
    target = root / Path(DECLARATION_PATH_V4).parent
    if path_occupied(target):
        raise EvidenceError("output_exists", str(target))
    value = _prospective_study_declaration_value_v4(root)
    declaration = ProspectiveStudyDeclarationV4.from_dict(value)
    declaration.verify_sources(root)
    declaration_bytes = canonical_json_bytes(declaration.to_dict())
    receipt = StudyDeclarationReceiptV4(
        declaration_path=DECLARATION_PATH_V4,
        declaration_sha256=hashlib.sha256(declaration_bytes).hexdigest(),
        schema_version=STUDY_DECLARATION_RECEIPT_SCHEMA_V4,
        study_id=STUDY_ID_V4,
        terminal_state="prospectively_frozen",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        write_json(stage / "study-declaration.json", declaration.to_dict())
        write_json(stage / "study-declaration-receipt.json", receipt.to_dict())
        if sha256_file(stage / "study-declaration.json") != receipt.declaration_sha256:
            raise EvidenceError("study_declaration_digest_mismatch", str(stage))
        rename_no_replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return receipt


@dataclass(frozen=True)
class StudyDeclarationReceiptV4:
    declaration_path: str
    declaration_sha256: str
    schema_version: str
    study_id: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_DECLARATION_RECEIPT_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.study_id != STUDY_ID_V4:
            raise EvidenceError("study_identity_mismatch", self.study_id)
        if self.declaration_path != DECLARATION_PATH_V4:
            raise EvidenceError("study_declaration_path_mismatch", self.declaration_path)
        require_sha256(self.declaration_sha256, "declaration_sha256")
        if self.terminal_state != "prospectively_frozen":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudyDeclarationReceiptV4:
        value = strict_fields(
            data,
            {
                "declaration_path",
                "declaration_sha256",
                "schema_version",
                "study_id",
                "terminal_state",
            },
            "StudyDeclarationReceiptV4",
        )
        return cls(
            require_string(value["declaration_path"], "declaration_path"),
            require_string(value["declaration_sha256"], "declaration_sha256"),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["study_id"], "study_id"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "declaration_path": self.declaration_path,
            "declaration_sha256": self.declaration_sha256,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "terminal_state": self.terminal_state,
        }


def verify_study_declaration_receipt_v4(
    receipt_path: Path,
    declaration_path: Path,
) -> StudyDeclarationReceiptV4:
    receipt = StudyDeclarationReceiptV4.from_dict(
        require_object(load_json(receipt_path), "study declaration receipt")
    )
    if sha256_file(declaration_path) != receipt.declaration_sha256:
        raise EvidenceError("study_declaration_digest_mismatch", str(declaration_path))
    return receipt


@dataclass(frozen=True)
class StudyExecutionBindingV4:
    declaration_sha256: str
    schema_version: str
    study_id: str

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_EXECUTION_BINDING_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.study_id != STUDY_ID_V4:
            raise EvidenceError("study_identity_mismatch", self.study_id)
        require_sha256(self.declaration_sha256, "declaration_sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudyExecutionBindingV4:
        value = strict_fields(
            data,
            {"declaration_sha256", "schema_version", "study_id"},
            "StudyExecutionBindingV4",
        )
        return cls(
            require_string(value["declaration_sha256"], "declaration_sha256"),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["study_id"], "study_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "declaration_sha256": self.declaration_sha256,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
        }


@dataclass(frozen=True)
class StudyAttemptV4:
    benchmark_output_path: str
    declaration_sha256: str
    no_rerun: bool
    schema_version: str
    study_id: str
    terminal_state: str

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_ATTEMPT_SCHEMA_V4:
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.study_id != STUDY_ID_V4:
            raise EvidenceError("study_identity_mismatch", self.study_id)
        if self.benchmark_output_path != BENCHMARK_OUTPUT_PATH_V4:
            raise EvidenceError("study_output_path_mismatch", self.benchmark_output_path)
        require_sha256(self.declaration_sha256, "declaration_sha256")
        if not self.no_rerun:
            raise EvidenceError("study_rerun_forbidden", self.study_id)
        if self.terminal_state != "consumed_on_start":
            raise EvidenceError("invalid_terminal_state", self.terminal_state)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudyAttemptV4:
        value = strict_fields(
            data,
            {
                "benchmark_output_path",
                "declaration_sha256",
                "no_rerun",
                "schema_version",
                "study_id",
                "terminal_state",
            },
            "StudyAttemptV4",
        )
        return cls(
            require_string(value["benchmark_output_path"], "benchmark_output_path"),
            require_string(value["declaration_sha256"], "declaration_sha256"),
            require_bool(value["no_rerun"], "no_rerun"),
            require_string(value["schema_version"], "schema_version"),
            require_string(value["study_id"], "study_id"),
            require_string(value["terminal_state"], "terminal_state"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_output_path": self.benchmark_output_path,
            "declaration_sha256": self.declaration_sha256,
            "no_rerun": self.no_rerun,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "terminal_state": self.terminal_state,
        }


def validate_declaration_design_counts(declaration: ProspectiveStudyDeclarationV4) -> None:
    design = declaration.design
    for field, expected in (
        ("attempt_count", 1),
        ("case_count", 9),
        ("repeat_count", 2),
        ("total_request_count", 18),
    ):
        if require_int(design[field], field) != expected:
            raise EvidenceError("study_design_drift", field)
