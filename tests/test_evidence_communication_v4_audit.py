# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from voxelscope.canonical import EvidenceError, load_json, write_json
from voxelscope.evidence_communication_v4_audit import (
    AUDIT_RELATIVE_PATH,
    SOURCE_STUDY_RELATIVE_PATH,
    build_audit_record,
    validate_audit_record,
    verify_committed_audit,
    verify_source_study_tree,
)
from voxelscope.evidence_communication_v4_audit_cli import main as audit_cli_main

ROOT = Path(__file__).resolve().parents[1]


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


def test_audit_fails_closed_on_source_tree_missing_extra_and_digest_drift(
    tmp_path: Path,
) -> None:
    study = tmp_path / SOURCE_STUDY_RELATIVE_PATH
    shutil.copytree(ROOT / SOURCE_STUDY_RELATIVE_PATH, study)

    missing = study / "benchmark/runs/supported-agreement/repeat-0000/draft.json"
    missing.unlink()
    with pytest.raises(EvidenceError) as missing_error:
        verify_source_study_tree(tmp_path)
    assert missing_error.value.code == "source_study_tree_drift"

    shutil.copyfile(
        ROOT
        / SOURCE_STUDY_RELATIVE_PATH
        / "benchmark/runs/supported-agreement/repeat-0000/draft.json",
        missing,
    )
    extra = study / "unexpected.json"
    write_json(extra, {"unexpected": True})
    with pytest.raises(EvidenceError) as extra_error:
        verify_source_study_tree(tmp_path)
    assert extra_error.value.code == "source_study_tree_drift"

    extra.unlink()
    missing.write_bytes(missing.read_bytes() + b" ")
    with pytest.raises(EvidenceError) as digest_error:
        verify_source_study_tree(tmp_path)
    assert digest_error.value.code == "source_study_tree_drift"


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
