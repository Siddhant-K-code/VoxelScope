# SPDX-License-Identifier: Apache-2.0
"""Deterministic post-hoc audit of the frozen observed v4 lexical exclusions."""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
from .evidence_benchmark_records import MeasurementValue
from .evidence_communication_v4 import replay_benchmark_v4
from .evidence_communication_v4_records import (
    BenchmarkIndexV4,
    BenchmarkReceiptV4,
    CommunicationReceiptV4,
    CommunicationRequestV4,
    DraftPlanEnvelopeV4,
    RunnerMeasurementV4,
    VerifiedCommunicationArtifactV4,
)
from .evidence_communication_v4_study_records import (
    load_study_declaration_v4,
    verify_study_declaration_receipt_v4,
)
from .gbm_atlas import assemble_gbm_evidence_atlas
from .records import (
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

AUDIT_SCHEMA_VERSION = "voxelscope/evidence-communication-v4-lexical-audit/v1"
AUDIT_ID = "gbm-evidence-communication-qwen3-8b-q8-v4-lexical-audit-v1"
CLASSIFIER_ID = "voxelscope/evidence-communication-v4-lexical-context-classifier/v1"
SOURCE_STUDY_RELATIVE_PATH = "research/gbm-evidence-communication-qwen3-8b-q8-study-v4"
AUDIT_RELATIVE_PATH = "research/evidence-communication-v4-lexical-audit-v1/audit.json"
ATLAS_MANIFEST_RELATIVE_PATH = "research/gbm-evidence-atlas-v1/source-manifest.json"
V4_FIXTURE_RELATIVE_PATH = (
    "research/gbm-evidence-communication-benchmark-v4/benchmark-fixtures.json"
)

SOURCE_STUDY_FILE_COUNT = 101
SOURCE_STUDY_MANIFEST_SHA256 = "1afe1d31613b0087b6d3d1ac9c8abbc279a58da07ca9b3dbda6845cca56aafd1"
SOURCE_STUDY_GIT_TREE_SHA1 = "328d8785b475dcd9dbf0b7a62dd4553542b335f2"
DECLARATION_SHA256 = "063b6055d430f14637d3dab05d1471cfc8e2fcf043b4030c633a956eff94a6c6"
BENCHMARK_REPORT_SHA256 = "86c1c86f62fda175dd121d84785bbb6ed5e11b216cb0383e9b7fcf0290fac409"
BENCHMARK_INDEX_SHA256 = "90f8dcd0002b52e4bdff89d00b0620b730d5ea7d5d5a137cdd3d7c1ff0ecadb3"
BENCHMARK_RECEIPT_SHA256 = "57347be8de8593a86e948a15c981890e747f684ac5a236b5a91305d712cae29d"

_VERIFIER_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "prohibited_diagnostic_or_prognostic_language",
        ("diagnos", "prognos", "survival", "patient classification"),
    ),
    (
        "prohibited_treatment_language",
        ("treatment", "therapy", "recommend", "guide care"),
    ),
    (
        "prohibited_causal_or_certainty_language",
        (" caus", "causal", "drives disease", "proves", "definitive", "guarantee"),
    ),
    (
        "prohibited_target_or_druggability_language",
        ("therapeutic target", "druggab"),
    ),
    (
        "prohibited_protein_ranking_language",
        ("ranked protein", "protein ranking", "best protein", "top protein"),
    ),
)

_CONTEXT_FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "explicit_no_definitive_diagnosis",
        ("no definitive diagnosis was established",),
    ),
    (
        "explicit_no_definitive_conclusion",
        ("no definitive conclusions were reached",),
    ),
    (
        "affirmative_treatment_monitoring",
        (
            "response to initial treatment was monitored",
            "response to treatment was monitored",
        ),
    ),
    (
        "affirmative_diagnostic_procedure",
        ("diagnostic procedures were conducted",),
    ),
    (
        "affirmative_diagnostic_therapeutic_decision_support",
        ("diagnostic and therapeutic decisions",),
    ),
    (
        "diagnostic_process_context",
        (
            "diagnostic evaluation",
            "diagnostic testing",
            "diagnostic steps",
            "diagnostic workup",
            "diagnostic procedures",
            "diagnostic considerations",
            "diagnostic possibilities",
            "differential diagnosis",
            "possible diagnoses",
        ),
    ),
    (
        "causal_question_or_hypothesis_context",
        (
            "underlying cause",
            "underlying causes",
            "potential causes",
            "underlying etiology",
            "possible explanations",
            "possible links",
            "potential links",
            "possible associations",
            "potential associations",
            "possible correlations",
            "potential correlations",
            "could be attributed",
        ),
    ),
    (
        "hedged_or_investigative_context",
        (
            "possible",
            "potential",
            "could",
            "further",
            "additional",
            "clarify",
            "explore",
            "refine",
            "considered",
            "proposed",
            "planned",
            "recommended",
            "suggested",
        ),
    ),
)

_NEGATION_FEATURES = frozenset(
    {
        "explicit_no_definitive_diagnosis",
        "explicit_no_definitive_conclusion",
    }
)
_AFFIRMATIVE_PROCESS_FEATURES = frozenset(
    {
        "affirmative_treatment_monitoring",
        "affirmative_diagnostic_procedure",
        "affirmative_diagnostic_therapeutic_decision_support",
    }
)
_CATEGORIES = (
    "affirmative_clinical_process_statement",
    "ambiguous_or_context_dependent",
    "explicit_negation_or_boundary_disclaimer",
)
_EXPECTED_CATEGORY_COUNTS = {
    "affirmative_clinical_process_statement": 4,
    "ambiguous_or_context_dependent": 92,
    "explicit_negation_or_boundary_disclaimer": 12,
}
_EXPECTED_REASON_COUNTS = {
    "prohibited_causal_or_certainty_language": 9,
    "prohibited_diagnostic_or_prognostic_language": 84,
    "prohibited_treatment_language": 15,
}


def _study_manifest(study_root: Path) -> tuple[list[dict[str, str]], str]:
    if is_link_like(study_root) or not study_root.is_dir():
        raise EvidenceError("unsafe_source_study", str(study_root))
    files: list[dict[str, str]] = []
    for path in sorted(study_root.rglob("*")):
        if is_link_like(path):
            raise EvidenceError("symlink_forbidden", str(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvidenceError("unsafe_source_study_entry", str(path))
        files.append(
            {
                "path": path.relative_to(study_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return files, sha256_bytes(canonical_json_bytes(files))


def verify_source_study_tree(repository_root: Path) -> dict[str, Any]:
    study_root = repository_root / SOURCE_STUDY_RELATIVE_PATH
    files, manifest_sha256 = _study_manifest(study_root)
    if len(files) != SOURCE_STUDY_FILE_COUNT or manifest_sha256 != SOURCE_STUDY_MANIFEST_SHA256:
        raise EvidenceError(
            "source_study_tree_drift",
            f"expected files={SOURCE_STUDY_FILE_COUNT} "
            f"manifest={SOURCE_STUDY_MANIFEST_SHA256}; "
            f"observed files={len(files)} manifest={manifest_sha256}",
        )
    identities = {
        "benchmark/index.json": BENCHMARK_INDEX_SHA256,
        "benchmark/receipt.json": BENCHMARK_RECEIPT_SHA256,
        "benchmark/benchmark.json": BENCHMARK_REPORT_SHA256,
        "study-declaration.json": DECLARATION_SHA256,
    }
    for relative_path, expected_sha256 in identities.items():
        actual_sha256 = sha256_file(study_root / relative_path)
        if actual_sha256 != expected_sha256:
            raise EvidenceError("source_study_digest_drift", relative_path)
    return {
        "file_count": len(files),
        "git_tree_sha1_at_main": SOURCE_STUDY_GIT_TREE_SHA1,
        "manifest_sha256": manifest_sha256,
        "path": SOURCE_STUDY_RELATIVE_PATH,
    }


def _verify_closed_benchmark(repository_root: Path) -> dict[str, Any]:
    study_root = repository_root / SOURCE_STUDY_RELATIVE_PATH
    declaration_path = study_root / "study-declaration.json"
    load_study_declaration_v4(declaration_path, repository_root)
    declaration_receipt = verify_study_declaration_receipt_v4(
        study_root / "study-declaration-receipt.json",
        declaration_path,
    )
    if declaration_receipt.declaration_sha256 != DECLARATION_SHA256:
        raise EvidenceError("study_declaration_digest_mismatch", str(declaration_path))

    atlas, _ = assemble_gbm_evidence_atlas(repository_root / ATLAS_MANIFEST_RELATIVE_PATH)
    with tempfile.TemporaryDirectory(prefix="voxelscope-v4-audit-") as temporary:
        atlas_path = Path(temporary) / "atlas.json"
        write_json(atlas_path, atlas)
        benchmark = replay_benchmark_v4(
            atlas_path,
            repository_root / V4_FIXTURE_RELATIVE_PATH,
            study_root / "benchmark",
            study_declaration_sha256=declaration_receipt.declaration_sha256,
        )
    return benchmark


def _matched_terms(text: str) -> tuple[list[str], list[str], str]:
    lowered = text.lower()
    all_matches: list[str] = []
    winning_matches: list[str] = []
    winning_reason = ""
    for reason, terms in _VERIFIER_TERMS:
        reason_matches = [term for term in terms if term in lowered]
        all_matches.extend(reason_matches)
        if not winning_reason and reason_matches:
            winning_reason = reason
            winning_matches = reason_matches
    if not winning_reason:
        raise EvidenceError("unmatched_lexical_exclusion", sha256_bytes(text.encode("utf-8")))
    return sorted(set(all_matches)), sorted(set(winning_matches)), winning_reason


def _context_features(text: str) -> list[str]:
    lowered = text.lower()
    return sorted(
        feature
        for feature, phrases in _CONTEXT_FEATURES
        if any(phrase in lowered for phrase in phrases)
    )


def _classify_text(text: str) -> tuple[str, str, list[str]]:
    features = _context_features(text)
    feature_set = frozenset(features)
    if feature_set & _NEGATION_FEATURES:
        return (
            "explicit_negation_or_boundary_disclaimer",
            "explicit_negation_of_diagnostic_or_definitive_conclusion",
            features,
        )
    if feature_set & _AFFIRMATIVE_PROCESS_FEATURES:
        return (
            "affirmative_clinical_process_statement",
            "affirmative_diagnostic_or_treatment_process_description",
            features,
        )
    if "hedged_or_investigative_context" in feature_set:
        return (
            "ambiguous_or_context_dependent",
            "hedged_investigative_or_context_dependent_usage",
            features,
        )
    raise EvidenceError(
        "unclassified_lexical_context",
        sha256_bytes(text.encode("utf-8")),
    )


def _available_measurement(
    value: MeasurementValue,
    name: str,
) -> int | float:
    if value.availability != "available" or value.value is None:
        raise EvidenceError("required_audit_measurement_unavailable", name)
    return value.value


def _measurement_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        raise EvidenceError("missing_audit_measurements", "measurement list is empty")
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "minimum": min(values),
    }


def _fraction(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise EvidenceError(
            "invalid_audit_fraction",
            f"numerator={numerator} denominator={denominator}",
        )
    return {
        "denominator": denominator,
        "numerator": numerator,
        "value": numerator / denominator,
    }


def _load_run_records(
    benchmark_root: Path,
    run_path: str,
) -> tuple[
    CommunicationRequestV4,
    DraftPlanEnvelopeV4,
    VerifiedCommunicationArtifactV4,
    CommunicationReceiptV4,
    RunnerMeasurementV4,
]:
    root = benchmark_root / run_path
    return (
        CommunicationRequestV4.from_dict(
            require_object(load_json(root / "request.json"), "communication request")
        ),
        DraftPlanEnvelopeV4.from_dict(require_object(load_json(root / "draft.json"), "draft plan")),
        VerifiedCommunicationArtifactV4.from_dict(
            require_object(load_json(root / "artifact.json"), "verified artifact")
        ),
        CommunicationReceiptV4.from_dict(
            require_object(load_json(root / "receipt.json"), "communication receipt")
        ),
        RunnerMeasurementV4.from_dict(
            require_object(load_json(root / "measurement.json"), "runner measurement")
        ),
    )


def build_audit_record(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    source_identity = verify_source_study_tree(root)
    replayed_benchmark = _verify_closed_benchmark(root)
    benchmark_root = root / SOURCE_STUDY_RELATIVE_PATH / "benchmark"
    index = BenchmarkIndexV4.from_dict(
        require_object(load_json(benchmark_root / "index.json"), "benchmark index")
    )
    receipt = BenchmarkReceiptV4.from_dict(
        require_object(load_json(benchmark_root / "receipt.json"), "benchmark receipt")
    )

    entries: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    all_term_counts: Counter[str] = Counter()
    winning_term_counts: Counter[str] = Counter()
    category_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    semantic_by_case: dict[str, list[str]] = defaultdict(list)
    latency_values: list[int | float] = []
    input_token_values: list[int | float] = []
    output_token_values: list[int | float] = []
    citation_count = 0
    valid_citation_count = 0
    invalid_model_outputs = 0
    run_count_with_exclusions = 0
    unique_excluded_skeleton_ids: set[str] = set()

    for indexed_run in index.runs:
        request, draft, artifact, communication_receipt, measurement = _load_run_records(
            benchmark_root,
            indexed_run.run_path,
        )
        draft_by_skeleton = {item.skeleton_id: item for item in draft.entries}
        skeleton_by_id = {item.skeleton_id: item for item in request.skeletons}
        if set(draft_by_skeleton) != set(skeleton_by_id):
            raise EvidenceError("audit_task_set_mismatch", indexed_run.run_path)

        if artifact.exclusions:
            run_count_with_exclusions += 1
        for exclusion in artifact.exclusions:
            draft_task = draft_by_skeleton[exclusion.skeleton_id]
            skeleton = skeleton_by_id[exclusion.skeleton_id]
            all_matches, winning_matches, winning_reason = _matched_terms(draft_task.draft_text)
            if winning_reason != exclusion.reason:
                raise EvidenceError(
                    "audit_verifier_reason_mismatch",
                    f"{indexed_run.run_path}:{exclusion.skeleton_id}",
                )
            category, category_reason, features = _classify_text(draft_task.draft_text)
            entry: dict[str, Any] = {
                "all_matched_terms": all_matches,
                "artifact_sha256": communication_receipt.artifact_sha256,
                "case_id": indexed_run.case_id,
                "category": category,
                "category_reason_code": category_reason,
                "claim_type": skeleton.claim_type,
                "context_features": features,
                "draft_sha256": communication_receipt.draft_sha256,
                "draft_text_code_points": len(draft_task.draft_text),
                "draft_text_sha256": sha256_bytes(draft_task.draft_text.encode("utf-8")),
                "draft_text_size_bytes": len(draft_task.draft_text.encode("utf-8")),
                "repeat_index": indexed_run.repeat_index,
                "request_id": request.request_id,
                "request_sha256": communication_receipt.request_sha256,
                "requirement_ids": list(skeleton.requirement_ids),
                "run_path": indexed_run.run_path,
                "skeleton_id": skeleton.skeleton_id,
                "state": skeleton.state,
                "verifier_reason": exclusion.reason,
                "winning_matched_terms": winning_matches,
            }
            entries.append(entry)
            reason_counts[exclusion.reason] += 1
            category_counts[category] += 1
            category_by_reason[category][exclusion.reason] += 1
            all_term_counts.update(all_matches)
            winning_term_counts.update(winning_matches)
            unique_excluded_skeleton_ids.add(exclusion.skeleton_id)

        for field in (
            "duplicate_count",
            "expected_count",
            "matched_count",
            "omitted_count",
            "returned_count",
            "unknown_count",
        ):
            task_counts[field] += getattr(artifact.task_outcome, field)
        terminal_bucket = {
            "accepted": "accepted",
            "accepted_with_exclusions": "partially_excluded",
            "refused": "refused",
        }[artifact.terminal_state]
        terminal_counts[terminal_bucket] += 1
        invalid_model_outputs += int(artifact.invalid_model_output is not None)
        coverage_counts["emitted_fact_numerator"] += artifact.emitted_fact_coverage.numerator
        coverage_counts["emitted_fact_denominator"] += artifact.emitted_fact_coverage.denominator
        coverage_counts["emitted_caveat_numerator"] += artifact.emitted_caveat_coverage.numerator
        coverage_counts["emitted_caveat_denominator"] += (
            artifact.emitted_caveat_coverage.denominator
        )
        coverage_counts["verified_fact_numerator"] += (
            artifact.verified_draft_fact_coverage.numerator
        )
        coverage_counts["verified_fact_denominator"] += (
            artifact.verified_draft_fact_coverage.denominator
        )
        coverage_counts["verified_caveat_numerator"] += (
            artifact.verified_draft_caveat_coverage.numerator
        )
        coverage_counts["verified_caveat_denominator"] += (
            artifact.verified_draft_caveat_coverage.denominator
        )

        request_skeletons = {item.skeleton_id: item for item in request.skeletons}
        for claim in artifact.verified_claims:
            citation_count += len(claim.source_ids)
            expected_sources = request_skeletons[claim.skeleton_id].source_ids
            if claim.source_ids == expected_sources:
                valid_citation_count += len(claim.source_ids)
        semantic_by_case[indexed_run.case_id].append(indexed_run.semantic_sha256)
        latency_values.append(_available_measurement(measurement.latency_ms, "latency_ms"))
        input_token_values.append(_available_measurement(measurement.input_tokens, "input_tokens"))
        output_token_values.append(
            _available_measurement(measurement.output_tokens, "output_tokens")
        )

    entries.sort(
        key=lambda item: (
            require_string(item["case_id"], "case_id"),
            require_int(item["repeat_index"], "repeat_index"),
            require_string(item["skeleton_id"], "skeleton_id"),
        )
    )
    pair_count = len(semantic_by_case)
    semantic_variance_count = sum(
        int(len(set(identities)) != 1) for identities in semantic_by_case.values()
    )
    terminal_total = sum(terminal_counts.values())
    observed_facts: dict[str, Any] = {
        "citation_validity": _fraction(valid_citation_count, citation_count),
        "coverage": {
            "emitted_caveat": _fraction(
                coverage_counts["emitted_caveat_numerator"],
                coverage_counts["emitted_caveat_denominator"],
            ),
            "emitted_fact": _fraction(
                coverage_counts["emitted_fact_numerator"],
                coverage_counts["emitted_fact_denominator"],
            ),
            "verified_draft_caveat": _fraction(
                coverage_counts["verified_caveat_numerator"],
                coverage_counts["verified_caveat_denominator"],
            ),
            "verified_draft_fact": _fraction(
                coverage_counts["verified_fact_numerator"],
                coverage_counts["verified_fact_denominator"],
            ),
        },
        "invalid_model_outputs": {
            "denominator": len(index.runs),
            "numerator": invalid_model_outputs,
        },
        "lexical_exclusions": {
            "reason_counts": dict(sorted(reason_counts.items())),
            "run_count_with_exclusions": run_count_with_exclusions,
            "total": len(entries),
            "unique_skeleton_ids": len(unique_excluded_skeleton_ids),
        },
        "measurements": {
            "input_tokens": _measurement_summary(input_token_values),
            "latency_ms": _measurement_summary(latency_values),
            "output_tokens": _measurement_summary(output_token_values),
        },
        "semantic_plan_variance": _fraction(semantic_variance_count, pair_count),
        "task_outcomes": {
            "duplicate": task_counts["duplicate_count"],
            "expected": task_counts["expected_count"],
            "matched": task_counts["matched_count"],
            "omitted": task_counts["omitted_count"],
            "returned": task_counts["returned_count"],
            "unknown": task_counts["unknown_count"],
        },
        "terminal_counts": {
            "accepted": terminal_counts["accepted"],
            "partially_excluded": terminal_counts["partially_excluded"],
            "refused": terminal_counts["refused"],
            "total": terminal_total,
        },
    }
    classification_summary: dict[str, Any] = {
        "all_matched_term_counts": dict(sorted(all_term_counts.items())),
        "category_by_verifier_reason": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(category_by_reason.items())
        },
        "category_counts": dict(sorted(category_counts.items())),
        "total_classified_entries": len(entries),
        "winning_matched_term_counts": dict(sorted(winning_term_counts.items())),
    }
    record: dict[str, Any] = {
        "analysis_boundary": {
            "classification_is_substantive_safety_truth": False,
            "generated_benchmark_modified": False,
            "model_inference_performed": False,
            "raw_draft_text_republished": False,
            "scope": "post_hoc_mechanical_classification_of_frozen_lexical_exclusions",
        },
        "audit_id": AUDIT_ID,
        "classification_contract": {
            "categories": list(_CATEGORIES),
            "classifier_id": CLASSIFIER_ID,
            "implementation_path": ("src/voxelscope/evidence_communication_v4_audit.py"),
            "implementation_sha256": sha256_file(Path(__file__)),
            "precedence": [
                "explicit_negation_or_boundary_disclaimer",
                "affirmative_clinical_process_statement",
                "ambiguous_or_context_dependent",
            ],
            "unit": "verifier_flagged_draft_entry",
        },
        "classification_summary": classification_summary,
        "custody": {
            "benchmark_closed_file_count": len(receipt.closed_files),
            "benchmark_replay": "verified_offline",
            "parsed_draft_count": len(index.runs),
            "run_count": len(index.runs),
            "source_tree_verified_before_analysis": True,
        },
        "entries": entries,
        "interpretation": {
            "benchmark_facts_are_observed_synthetic_results": True,
            "categories_are_post_hoc_mechanical_labels": True,
            "clinical_or_biomedical_claim": False,
            "model_quality_or_safety_claim": False,
            "reason": (
                "Lexical matches and deterministic context features do not establish "
                "whether draft text is substantively safe or unsafe."
            ),
            "v4_result_reinterpreted_or_overwritten": False,
        },
        "observed_benchmark_facts": observed_facts,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source": {
            "benchmark_index_sha256": sha256_file(benchmark_root / "index.json"),
            "benchmark_receipt_sha256": sha256_file(benchmark_root / "receipt.json"),
            "benchmark_report_sha256": sha256_file(benchmark_root / "benchmark.json"),
            "declaration_sha256": DECLARATION_SHA256,
            "replayed_benchmark_schema_version": replayed_benchmark["schema_version"],
            "study": source_identity,
        },
    }
    validate_audit_record(record)
    return record


def _strict_fraction(value: Any, name: str) -> None:
    fields = strict_fields(
        require_object(value, name),
        {"denominator", "numerator", "value"},
        name,
    )
    numerator = require_int(fields["numerator"], f"{name}.numerator")
    denominator = require_int(fields["denominator"], f"{name}.denominator")
    if fields != _fraction(numerator, denominator):
        raise EvidenceError("audit_fraction_disagreement", name)


def validate_audit_record(value: dict[str, Any]) -> None:
    record = strict_fields(
        value,
        {
            "analysis_boundary",
            "audit_id",
            "classification_contract",
            "classification_summary",
            "custody",
            "entries",
            "interpretation",
            "observed_benchmark_facts",
            "schema_version",
            "source",
        },
        "V4LexicalAudit",
    )
    if require_string(record["schema_version"], "schema_version") != AUDIT_SCHEMA_VERSION:
        raise EvidenceError("unsupported_schema", str(record["schema_version"]))
    if require_string(record["audit_id"], "audit_id") != AUDIT_ID:
        raise EvidenceError("unexpected_audit_id", str(record["audit_id"]))

    strict_fields(
        require_object(record["analysis_boundary"], "analysis_boundary"),
        {
            "classification_is_substantive_safety_truth",
            "generated_benchmark_modified",
            "model_inference_performed",
            "raw_draft_text_republished",
            "scope",
        },
        "analysis_boundary",
    )
    contract = strict_fields(
        require_object(record["classification_contract"], "classification_contract"),
        {
            "categories",
            "classifier_id",
            "implementation_path",
            "implementation_sha256",
            "precedence",
            "unit",
        },
        "classification_contract",
    )
    categories = tuple(
        require_string(item, "category")
        for item in require_list(contract["categories"], "categories")
    )
    if categories != _CATEGORIES or contract["classifier_id"] != CLASSIFIER_ID:
        raise EvidenceError("audit_classifier_contract_drift", AUDIT_ID)
    if contract["implementation_path"] != ("src/voxelscope/evidence_communication_v4_audit.py"):
        raise EvidenceError("audit_classifier_contract_drift", AUDIT_ID)
    require_sha256(require_string(contract["implementation_sha256"], "implementation_sha256"))

    entries = require_list(record["entries"], "entries")
    entry_keys = {
        "all_matched_terms",
        "artifact_sha256",
        "case_id",
        "category",
        "category_reason_code",
        "claim_type",
        "context_features",
        "draft_sha256",
        "draft_text_code_points",
        "draft_text_sha256",
        "draft_text_size_bytes",
        "repeat_index",
        "request_id",
        "request_sha256",
        "requirement_ids",
        "run_path",
        "skeleton_id",
        "state",
        "verifier_reason",
        "winning_matched_terms",
    }
    category_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    identities: list[tuple[str, int, str]] = []
    for item in entries:
        entry = strict_fields(require_object(item, "audit entry"), entry_keys, "AuditEntry")
        category = require_string(entry["category"], "category")
        if category not in _CATEGORIES:
            raise EvidenceError("unknown_audit_category", category)
        case_id = require_string(entry["case_id"], "case_id")
        repeat_index = require_int(entry["repeat_index"], "repeat_index")
        skeleton_id = require_string(entry["skeleton_id"], "skeleton_id")
        require_sha256(require_string(entry["draft_text_sha256"], "draft_text_sha256"))
        for digest_name in (
            "artifact_sha256",
            "draft_sha256",
            "request_sha256",
        ):
            require_sha256(require_string(entry[digest_name], digest_name))
        if require_int(entry["draft_text_code_points"], "draft_text_code_points") <= 0:
            raise EvidenceError("invalid_audit_text_size", skeleton_id)
        if require_int(entry["draft_text_size_bytes"], "draft_text_size_bytes") <= 0:
            raise EvidenceError("invalid_audit_text_size", skeleton_id)
        for list_name in (
            "all_matched_terms",
            "context_features",
            "winning_matched_terms",
        ):
            values = [
                require_string(value, f"{list_name} item")
                for value in require_list(entry[list_name], list_name)
            ]
            if not values or values != sorted(set(values)):
                raise EvidenceError("invalid_audit_entry_list", list_name)
        requirement_ids = [
            require_string(value, "requirement_ids item")
            for value in require_list(entry["requirement_ids"], "requirement_ids")
        ]
        if not requirement_ids or len(requirement_ids) != len(set(requirement_ids)):
            raise EvidenceError("invalid_audit_entry_list", "requirement_ids")
        category_counts[category] += 1
        reason_counts[require_string(entry["verifier_reason"], "verifier_reason")] += 1
        identities.append((case_id, repeat_index, skeleton_id))
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise EvidenceError("invalid_audit_entry_order", AUDIT_ID)

    summary = strict_fields(
        require_object(record["classification_summary"], "classification_summary"),
        {
            "all_matched_term_counts",
            "category_by_verifier_reason",
            "category_counts",
            "total_classified_entries",
            "winning_matched_term_counts",
        },
        "classification_summary",
    )
    if summary["category_counts"] != dict(sorted(category_counts.items())):
        raise EvidenceError("audit_category_count_disagreement", AUDIT_ID)
    if summary["category_counts"] != _EXPECTED_CATEGORY_COUNTS:
        raise EvidenceError("audit_expected_category_count_drift", AUDIT_ID)
    if summary["total_classified_entries"] != len(entries):
        raise EvidenceError("audit_entry_count_disagreement", AUDIT_ID)
    if dict(sorted(reason_counts.items())) != _EXPECTED_REASON_COUNTS:
        raise EvidenceError("audit_expected_reason_count_drift", AUDIT_ID)

    facts = strict_fields(
        require_object(record["observed_benchmark_facts"], "observed_benchmark_facts"),
        {
            "citation_validity",
            "coverage",
            "invalid_model_outputs",
            "lexical_exclusions",
            "measurements",
            "semantic_plan_variance",
            "task_outcomes",
            "terminal_counts",
        },
        "observed_benchmark_facts",
    )
    _strict_fraction(facts["citation_validity"], "citation_validity")
    _strict_fraction(facts["semantic_plan_variance"], "semantic_plan_variance")
    coverage = strict_fields(
        require_object(facts["coverage"], "coverage"),
        {
            "emitted_caveat",
            "emitted_fact",
            "verified_draft_caveat",
            "verified_draft_fact",
        },
        "coverage",
    )
    for name, fraction in coverage.items():
        _strict_fraction(fraction, name)
    exclusions = strict_fields(
        require_object(facts["lexical_exclusions"], "lexical_exclusions"),
        {"reason_counts", "run_count_with_exclusions", "total", "unique_skeleton_ids"},
        "lexical_exclusions",
    )
    if exclusions["reason_counts"] != _EXPECTED_REASON_COUNTS:
        raise EvidenceError("audit_observed_reason_count_disagreement", AUDIT_ID)
    if exclusions["total"] != len(entries):
        raise EvidenceError("audit_observed_entry_count_disagreement", AUDIT_ID)

    expected_observed = {
        "citation_validity": {"denominator": 192, "numerator": 192, "value": 1.0},
        "coverage": {
            "emitted_caveat": {
                "denominator": 148,
                "numerator": 60,
                "value": 60 / 148,
            },
            "emitted_fact": {
                "denominator": 158,
                "numerator": 42,
                "value": 42 / 158,
            },
            "verified_draft_caveat": {
                "denominator": 148,
                "numerator": 102,
                "value": 102 / 148,
            },
            "verified_draft_fact": {
                "denominator": 158,
                "numerator": 84,
                "value": 84 / 158,
            },
        },
        "invalid_model_outputs": {"denominator": 18, "numerator": 0},
        "semantic_plan_variance": {
            "denominator": 9,
            "numerator": 3,
            "value": 1 / 3,
        },
        "task_outcomes": {
            "duplicate": 0,
            "expected": 270,
            "matched": 270,
            "omitted": 0,
            "returned": 270,
            "unknown": 0,
        },
        "terminal_counts": {
            "accepted": 6,
            "partially_excluded": 0,
            "refused": 12,
            "total": 18,
        },
    }
    for name, expected in expected_observed.items():
        if facts[name] != expected:
            raise EvidenceError("audit_observed_count_disagreement", name)

    strict_fields(
        require_object(record["custody"], "custody"),
        {
            "benchmark_closed_file_count",
            "benchmark_replay",
            "parsed_draft_count",
            "run_count",
            "source_tree_verified_before_analysis",
        },
        "custody",
    )
    strict_fields(
        require_object(record["interpretation"], "interpretation"),
        {
            "benchmark_facts_are_observed_synthetic_results",
            "categories_are_post_hoc_mechanical_labels",
            "clinical_or_biomedical_claim",
            "model_quality_or_safety_claim",
            "reason",
            "v4_result_reinterpreted_or_overwritten",
        },
        "interpretation",
    )
    source = strict_fields(
        require_object(record["source"], "source"),
        {
            "benchmark_index_sha256",
            "benchmark_receipt_sha256",
            "benchmark_report_sha256",
            "declaration_sha256",
            "replayed_benchmark_schema_version",
            "study",
        },
        "source",
    )
    study = strict_fields(
        require_object(source["study"], "source study"),
        {"file_count", "git_tree_sha1_at_main", "manifest_sha256", "path"},
        "source study",
    )
    if (
        study["file_count"] != SOURCE_STUDY_FILE_COUNT
        or study["manifest_sha256"] != SOURCE_STUDY_MANIFEST_SHA256
        or study["git_tree_sha1_at_main"] != SOURCE_STUDY_GIT_TREE_SHA1
        or study["path"] != SOURCE_STUDY_RELATIVE_PATH
    ):
        raise EvidenceError("audit_source_identity_drift", AUDIT_ID)
    expected_source_digests = {
        "benchmark_index_sha256": BENCHMARK_INDEX_SHA256,
        "benchmark_receipt_sha256": BENCHMARK_RECEIPT_SHA256,
        "benchmark_report_sha256": BENCHMARK_REPORT_SHA256,
        "declaration_sha256": DECLARATION_SHA256,
    }
    for name, expected in expected_source_digests.items():
        if source[name] != expected:
            raise EvidenceError("audit_source_digest_drift", name)


def verify_committed_audit(
    repository_root: Path,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    root = repository_root.resolve()
    path = audit_path or root / AUDIT_RELATIVE_PATH
    if not path.is_absolute():
        path = root / path
    stored = require_object(load_json(path), "v4 lexical audit")
    validate_audit_record(stored)
    rebuilt = build_audit_record(root)
    if stored != rebuilt:
        raise EvidenceError("audit_recomputation_mismatch", str(path))
    return stored
