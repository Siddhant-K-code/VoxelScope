# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path

import pytest

from voxelscope.bundle import finalize_bundle, verify_bundle
from voxelscope.canonical import EvidenceError, load_json, sha256_file, write_json
from voxelscope.cli import main
from voxelscope.fixtures import build_fixture_bundle

GOLDEN_BUNDLE_SHA256 = "cc0f364fda43f4c29a446f8664fe458a3719092a9d2262d675aa76609882607e"


def test_bundle_is_reproducible_and_matches_golden_hash(tmp_path: Path) -> None:
    first = build_fixture_bundle(tmp_path / "first")
    second = build_fixture_bundle(tmp_path / "second")
    assert first == second == GOLDEN_BUNDLE_SHA256
    assert verify_bundle(tmp_path / "first") == first


def test_checksum_and_canonical_json_verification(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    index = load_json(root / "bundle.json")
    assert index["bundle_format"] == "voxelscope-closed-bundle-v1"
    assert sha256_file(root / "bundle.json") == GOLDEN_BUNDLE_SHA256


def test_reindexed_semantic_tamper_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    report_path = root / "reports" / "identical.json"
    report = load_json(report_path)
    report["changed_voxel_count"] = 999
    write_json(report_path, report)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "drift_report_mismatch"


def test_byte_tamper_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    target = root / "outputs" / "reference" / "probabilities.f32le"
    data = bytearray(target.read_bytes())
    data[0] ^= 1
    target.write_bytes(data)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "bundle_artifact_hash_mismatch"


def test_extra_and_missing_files_are_detected(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    build_fixture_bundle(extra)
    (extra / "unexpected.txt").write_text("unexpected", encoding="ascii")
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(extra)
    assert caught.value.code == "bundle_file_set_mismatch"
    missing = tmp_path / "missing"
    build_fixture_bundle(missing)
    (missing / "SUMMARY.txt").unlink()
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(missing)
    assert caught.value.code == "bundle_file_set_mismatch"


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks unavailable")
def test_symlinked_payload_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    target = root / "SUMMARY.txt"
    saved = tmp_path / "saved-summary.txt"
    target.replace(saved)
    try:
        target.symlink_to(saved)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "symlink_forbidden"


def test_all_cli_paths_work_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked)
    root = tmp_path / "bundle"
    assert main(["fixture", "build", "--output", str(root)]) == 0
    assert main(["verify", "--bundle", str(root)]) == 0
    windows = tmp_path / "windows"
    assert (
        main(
            [
                "windows",
                "build",
                "--manifest",
                str(root / "study-manifest.json"),
                "--output",
                str(windows),
            ]
        )
        == 0
    )
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "drift",
                "compare",
                "--reference",
                str(root / "outputs" / "reference" / "output.json"),
                "--candidate",
                str(root / "outputs" / "one-voxel-boundary" / "output.json"),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    assert report.is_file()
    assert main(["verify", "--bundle", str(root)]) == 0


def test_cli_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    assert main(["fixture", "build", "--output", str(root)]) == 0
    assert main(["fixture", "build", "--output", str(root)]) == 2


def test_fixture_privacy_scan(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    forbidden_extensions = {".dcm", ".nii", ".gz"}
    forbidden_keys = {"patient_id", "patient_name", "subject_id", "accession_number"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in forbidden_extensions
        data = path.read_bytes()
        assert not (len(data) >= 132 and data[128:132] == b"DICM")
        assert b"n+1\x00" not in data and b"ni1\x00" not in data
        if path.suffix == ".json":
            value = json.loads(data)
            stack = [value]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    assert forbidden_keys.isdisjoint(item)
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)


def test_documentation_has_no_em_dash() -> None:
    paths = [Path("README.md"), *Path("docs").glob("*.md")]
    assert all("\u2014" not in path.read_text(encoding="utf-8") for path in paths)


@pytest.mark.parametrize("root_name", ["bundle.json", "bundle.sha256"])
def test_symlinked_root_control_file_is_refused(tmp_path: Path, root_name: str) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    target = root / root_name
    saved = tmp_path / f"saved-{root_name}"
    target.replace(saved)
    try:
        target.symlink_to(saved)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "symlink_forbidden"


def test_bundle_artifact_unknown_field_is_refused_after_reseal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    index = load_json(root / "bundle.json")
    index["artifacts"][0]["unexpected"] = True
    write_json(root / "bundle.json", index)
    digest = sha256_file(root / "bundle.json")
    (root / "bundle.sha256").write_text(f"{digest}  bundle.json\n", encoding="ascii")
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "invalid_record"


def test_model_config_tamper_is_rejected_after_reseal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    config_path = root / "synthetic-model-config.json"
    config = load_json(config_path)
    config["input_channels"] = 5
    write_json(config_path, config)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "model_config_hash_mismatch"


def test_model_identity_unknown_field_is_refused_after_reseal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    identity_path = root / "model-identity.json"
    identity = load_json(identity_path)
    identity["unexpected"] = "not allowed"
    write_json(identity_path, identity)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "invalid_record"


@pytest.mark.parametrize("mutation", ["extra", "gpu"])
def test_inconsistent_receipt_is_refused_after_reseal(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    receipt_path = root / "run-receipt.json"
    receipt = load_json(receipt_path)
    if mutation == "extra":
        receipt["unexpected"] = True
    else:
        receipt["gpu_used"] = True
    write_json(receipt_path, receipt)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code in {"invalid_record", "invalid_receipt"}


def test_manifest_has_no_fabricated_timestamp(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    manifest = load_json(root / "study-manifest.json")
    assert "created_at" not in manifest


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("output", "planned_output_set_mismatch"),
        ("report", "planned_report_set_mismatch"),
        ("refusal", "planned_refusal_set_mismatch"),
    ],
)
def test_planned_evidence_cannot_be_removed_and_resealed(
    tmp_path: Path, kind: str, expected_code: str
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    if kind == "output":
        shutil.rmtree(root / "outputs" / "identical")
    elif kind == "report":
        (root / "reports" / "identical.json").unlink()
    else:
        (root / "refusals" / "invalid-nesting.json").unlink()
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == expected_code


def test_output_directory_is_bound_to_output_id(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    (root / "outputs" / "identical").rename(root / "outputs" / "renamed")
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "output_path_id_mismatch"


def test_report_filename_is_bound_to_report_id(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    (root / "reports" / "identical.json").rename(root / "reports" / "renamed.json")
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "planned_report_set_mismatch"


def test_bundle_digest_bytes_must_be_canonical(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    digest_path = root / "bundle.sha256"
    digest_path.write_bytes(digest_path.read_bytes().rstrip(b"\n") + b"  \n")
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "invalid_bundle_digest"


def test_artifact_media_type_is_bound_to_path(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    index_path = root / "bundle.json"
    index = load_json(index_path)
    index["artifacts"][0]["media_type"] = "application/json"
    write_json(index_path, index)
    digest = sha256_file(index_path)
    (root / "bundle.sha256").write_text(f"{digest}  bundle.json\n", encoding="ascii")
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "bundle_artifact_media_type_mismatch"


def test_output_spacing_must_be_positive_after_reseal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    output_path = root / "outputs" / "identical" / "output.json"
    output = load_json(output_path)
    output["spacing_mm"][0] = 0
    write_json(output_path, output)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "invalid_spacing"


def test_receipt_timing_path_must_be_safe_and_bound(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    receipt_path = root / "run-receipt.json"
    receipt = load_json(receipt_path)
    receipt["timings_path"] = "../outside.json"
    write_json(receipt_path, receipt)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "unsafe_path"


def test_refusal_evidence_must_bind_to_indexed_artifact_after_reseal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    request_path = root / "refusals" / "invalid-padding-request.json"
    request = load_json(request_path)
    request["mode"] = "changed-after-refusal"
    write_json(request_path, request)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "unbound_refusal_evidence"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("model_kind", "unknown", "invalid_model_kind"),
        ("license", "", "invalid_model_identity"),
    ],
)
def test_model_runtime_literals_are_enforced_after_reseal(
    tmp_path: Path, field: str, value: str, expected_code: str
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    path = root / "model-identity.json"
    identity = load_json(path)
    identity[field] = value
    write_json(path, identity)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == expected_code


def test_bundle_format_literal_is_enforced(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    path = root / "bundle.json"
    index = load_json(path)
    index["bundle_format"] = "unknown"
    write_json(path, index)
    digest = sha256_file(path)
    (root / "bundle.sha256").write_text(f"{digest}  bundle.json\n", encoding="ascii")
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "invalid_bundle_format"


@pytest.mark.parametrize(("field", "value"), [("unit", "ms"), ("clock", "")])
def test_timing_runtime_literals_are_enforced_after_reseal(
    tmp_path: Path, field: str, value: str
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    path = root / "stage-timings.json"
    timings = load_json(path)
    timings["records"][0][field] = value
    write_json(path, timings)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "invalid_timing"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("threshold_version", "unknown", "invalid_threshold"),
        ("connectivity", 6, "invalid_components"),
        ("output_id", "", "invalid_output_identity"),
    ],
)
def test_output_runtime_contracts_are_enforced_after_reseal(
    tmp_path: Path, field: str, value: object, expected_code: str
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    path = root / "outputs" / "identical" / "output.json"
    output = load_json(path)
    if field == "threshold_version":
        output["threshold"]["version"] = value
    elif field == "connectivity":
        output["components"]["TC"]["connectivity"] = value
    else:
        output["output_id"] = value
    write_json(path, output)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == expected_code


def test_synthetic_manifest_claim_escalation_is_refused_after_reseal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    path = root / "study-manifest.json"
    manifest = load_json(path)
    index = load_json(root / "bundle.json")
    manifest["claim_scope"] = "diagnostic_accuracy"
    manifest["lineage_status"] = "proven_subject_nonoverlap"
    manifest["lineage_evidence_sha256"] = [index["artifacts"][0]["sha256"]]
    manifest["diagnostic_accuracy_allowed"] = True
    write_json(path, manifest)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "synthetic_manifest_scope_mismatch"


@pytest.mark.parametrize(("field", "value"), [("run_id", "other"), ("command", "other")])
def test_synthetic_receipt_identity_is_exact_after_reseal(
    tmp_path: Path, field: str, value: str
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    path = root / "run-receipt.json"
    receipt = load_json(path)
    receipt[field] = value
    write_json(path, receipt)
    finalize_bundle(root)
    with pytest.raises(EvidenceError) as caught:
        verify_bundle(root)
    assert caught.value.code == "invalid_receipt"
