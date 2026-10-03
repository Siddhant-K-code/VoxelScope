# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from voxelscope.canonical import EvidenceError, load_json
from voxelscope.evidence_communication_v4_audit import (
    AUDIT_RELATIVE_PATH,
    build_audit_record,
    git_tree_identity,
    validate_audit_record,
    verify_committed_audit,
    verify_git_tree_custody,
)
from voxelscope.evidence_communication_v4_audit_cli import main as audit_cli_main

ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def test_committed_v4_lexical_audit_recomputes_exactly() -> None:
    audit = verify_committed_audit(ROOT)

    assert audit["classification_summary"]["category_counts"] == {
        "affirmative_clinical_process_statement": 4,
        "ambiguous_or_context_dependent": 92,
        "explicit_negation_or_boundary_disclaimer": 12,
    }
    assert audit["observed_benchmark_facts"]["lexical_exclusions"] == {
        "reason_counts": {
            "prohibited_causal_or_certainty_language": 9,
            "prohibited_diagnostic_or_prognostic_language": 84,
            "prohibited_treatment_language": 15,
        },
        "run_count_with_exclusions": 12,
        "total": 108,
        "unique_skeleton_ids": 23,
    }
    assert audit["observed_benchmark_facts"]["task_outcomes"] == {
        "duplicate": 0,
        "expected": 270,
        "matched": 270,
        "omitted": 0,
        "returned": 270,
        "unknown": 0,
    }
    assert audit["classification_summary"]["negation_and_affirmative_feature_overlap_count"] == 0
    assert all("draft_text" not in entry for entry in audit["entries"])


def test_audit_record_rejects_schema_and_count_disagreement() -> None:
    original = load_json(ROOT / AUDIT_RELATIVE_PATH)

    schema_drift = copy.deepcopy(original)
    schema_drift["schema_version"] = "different"
    with pytest.raises(EvidenceError) as schema_error:
        validate_audit_record(schema_drift)
    assert schema_error.value.code == "unsupported_schema"

    extra = copy.deepcopy(original)
    extra["unexpected"] = True
    with pytest.raises(EvidenceError) as extra_error:
        validate_audit_record(extra)
    assert extra_error.value.code == "invalid_record"

    count_drift = copy.deepcopy(original)
    count_drift["classification_summary"]["category_counts"]["ambiguous_or_context_dependent"] -= 1
    with pytest.raises(EvidenceError) as count_error:
        validate_audit_record(count_drift)
    assert count_error.value.code == "audit_category_count_disagreement"


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("analysis_boundary", "audit_analysis_boundary_drift"),
        ("custody", "audit_custody_assertion_drift"),
        ("interpretation", "audit_interpretation_drift"),
        ("precedence", "audit_classifier_contract_drift"),
        ("unit", "audit_classifier_contract_drift"),
        ("all_terms", "audit_all_term_count_disagreement"),
        ("category_reasons", "audit_category_reason_count_disagreement"),
        ("feature_overlap", "audit_context_feature_overlap"),
        ("source_status", "audit_source_identity_drift"),
        ("entry_reason_code", "invalid_audit_entry_classification"),
        ("entry_matched_term", "invalid_audit_entry_features"),
    ],
)
def test_audit_record_rejects_assertion_tampering(
    mutation: str,
    error_code: str,
) -> None:
    audit = copy.deepcopy(load_json(ROOT / AUDIT_RELATIVE_PATH))
    if mutation == "analysis_boundary":
        audit["analysis_boundary"]["model_inference_performed"] = True
    elif mutation == "custody":
        audit["custody"]["benchmark_replay"] = "not_verified"
    elif mutation == "interpretation":
        audit["interpretation"]["model_quality_or_safety_claim"] = True
    elif mutation == "precedence":
        audit["classification_contract"]["precedence"].reverse()
    elif mutation == "unit":
        audit["classification_contract"]["unit"] = "different"
    elif mutation == "all_terms":
        audit["classification_summary"]["all_matched_term_counts"]["diagnos"] -= 1
    elif mutation == "category_reasons":
        audit["classification_summary"]["category_by_verifier_reason"][
            "ambiguous_or_context_dependent"
        ]["prohibited_diagnostic_or_prognostic_language"] -= 1
    elif mutation == "feature_overlap":
        audit["classification_summary"]["negation_and_affirmative_feature_overlap_count"] = 1
    elif mutation == "source_status":
        audit["source"]["study"]["worktree_status"] = "clean"
    elif mutation == "entry_reason_code":
        audit["entries"][0]["category_reason_code"] = "different"
    elif mutation == "entry_matched_term":
        audit["entries"][0]["all_matched_terms"].append("unknown")
        audit["entries"][0]["all_matched_terms"].sort()
    else:
        raise AssertionError(mutation)

    with pytest.raises(EvidenceError) as caught:
        validate_audit_record(audit)
    assert caught.value.code == error_code


def test_git_custody_accepts_crlf_equivalence_and_rejects_real_drift(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    _git(tmp_path, "config", "user.name", "Audit Test")
    (tmp_path / ".gitattributes").write_bytes(b"*.txt text\n*.bin -text\n")
    study = tmp_path / "study"
    study.mkdir()
    text_path = study / "note.txt"
    binary_path = study / "payload.bin"
    text_path.write_bytes(b"line one\nline two\n")
    binary_path.write_bytes(b"\x00\x01\r\n\xff")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "fixture")
    identity = git_tree_identity(tmp_path, "study")
    expected = {
        "expected_tree_sha1": identity["git_tree_sha1"],
        "expected_file_count": identity["tracked_file_count"],
        "expected_blob_manifest_sha256": identity["git_blob_manifest_sha256"],
    }

    text_path.write_bytes(b"line one\r\nline two\r\n")
    assert verify_git_tree_custody(tmp_path, "study", **expected) == identity

    text_path.write_bytes(b"line one\r\nchanged\r\n")
    with pytest.raises(EvidenceError) as content_error:
        verify_git_tree_custody(tmp_path, "study", **expected)
    assert content_error.value.code == "source_study_worktree_drift"

    _git(tmp_path, "checkout", "--", "study/note.txt")
    extra = study / "extra.txt"
    extra.write_bytes(b"extra\n")
    with pytest.raises(EvidenceError) as extra_error:
        verify_git_tree_custody(tmp_path, "study", **expected)
    assert extra_error.value.code == "source_study_worktree_drift"

    extra.unlink()
    text_path.unlink()
    with pytest.raises(EvidenceError) as missing_error:
        verify_git_tree_custody(tmp_path, "study", **expected)
    assert missing_error.value.code == "source_study_worktree_drift"

    _git(tmp_path, "checkout", "--", "study/note.txt")
    binary_path.write_bytes(b"\x00\x01\r\n\xfe")
    with pytest.raises(EvidenceError) as binary_error:
        verify_git_tree_custody(tmp_path, "study", **expected)
    assert binary_error.value.code == "source_study_worktree_drift"


def test_git_custody_rejects_declared_tree_or_blob_manifest_drift(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    _git(tmp_path, "config", "user.name", "Audit Test")
    study = tmp_path / "study"
    study.mkdir()
    (study / "record.json").write_bytes(b'{"value":1}\n')
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "fixture")
    identity = git_tree_identity(tmp_path, "study")

    with pytest.raises(EvidenceError) as tree_error:
        verify_git_tree_custody(
            tmp_path,
            "study",
            expected_tree_sha1="0" * 40,
            expected_file_count=identity["tracked_file_count"],
            expected_blob_manifest_sha256=identity["git_blob_manifest_sha256"],
        )
    assert tree_error.value.code == "source_study_git_identity_drift"

    with pytest.raises(EvidenceError) as manifest_error:
        verify_git_tree_custody(
            tmp_path,
            "study",
            expected_tree_sha1=identity["git_tree_sha1"],
            expected_file_count=identity["tracked_file_count"],
            expected_blob_manifest_sha256="0" * 64,
        )
    assert manifest_error.value.code == "source_study_git_identity_drift"

    (study / "record.json").write_bytes(b'{"value":2}\n')
    _git(tmp_path, "add", "study/record.json")
    with pytest.raises(EvidenceError) as index_error:
        verify_git_tree_custody(
            tmp_path,
            "study",
            expected_tree_sha1=identity["git_tree_sha1"],
            expected_file_count=identity["tracked_file_count"],
            expected_blob_manifest_sha256=identity["git_blob_manifest_sha256"],
        )
    assert index_error.value.code == "source_study_index_drift"


def test_audit_cli_recomputes_without_model_or_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "audit.json"

    assert (
        audit_cli_main(
            [
                "recompute",
                "--repository-root",
                str(ROOT),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert load_json(output) == build_audit_record(ROOT)
    assert "entries=108" in capsys.readouterr().out
