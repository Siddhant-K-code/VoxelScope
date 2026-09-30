# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from voxelscope.canonical import (
    EvidenceError,
    canonical_json_bytes,
    sha256_file,
    write_json,
)
from voxelscope.milestone7_cli import main
from voxelscope.milestone7_evidence import (
    INDEXED_PATHS,
    PAYLOAD_PATHS,
    build_milestone7_public_bundle,
    verify_milestone7_public_bundle,
)
from voxelscope.window_execution_contract import (
    LEGACY_SYNTHETIC_BUNDLE_SHA256,
    MILESTONE5_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_PUBLIC_BUNDLE_SHA256,
    TRUSTED_PUBLIC_BUNDLE_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/window-execution-plan-v1.json"
BUNDLE = ROOT / "research/milestone-7"


def _copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "bundle"
    shutil.copytree(BUNDLE, destination)
    return destination


def _reindex(root: Path) -> None:
    sums = "".join(
        f"{sha256_file(root / relative)}  {relative}\n" for relative in sorted(PAYLOAD_PATHS)
    )
    (root / "SHA256SUMS").write_bytes(sums.encode("ascii"))
    write_json(
        root / "bundle.json",
        {
            "artifacts": [
                {
                    "path": relative,
                    "sha256": sha256_file(root / relative),
                    "size_bytes": (root / relative).stat().st_size,
                }
                for relative in sorted(INDEXED_PATHS)
            ],
            "bundle_format": "voxelscope-closed-public-bundle-v1",
            "schema_version": "voxelscope/milestone-7-public-evidence/v1",
        },
    )
    digest = sha256_file(root / "bundle.json")
    (root / "bundle.sha256").write_bytes(f"{digest}  bundle.json\n".encode("ascii"))


def test_committed_milestone7_bundle_is_trusted() -> None:
    assert (
        verify_milestone7_public_bundle(BUNDLE, repository_root=ROOT)
        == TRUSTED_PUBLIC_BUNDLE_SHA256
    )


def test_bundle_rebuild_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    digest = build_milestone7_public_bundle(PLAN, output, repository_root=ROOT)
    assert digest == TRUSTED_PUBLIC_BUNDLE_SHA256
    assert {
        path.relative_to(output).as_posix(): path.read_bytes() for path in output.iterdir()
    } == {path.relative_to(BUNDLE).as_posix(): path.read_bytes() for path in BUNDLE.iterdir()}


def test_bundle_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(EvidenceError) as captured:
        build_milestone7_public_bundle(PLAN, output, repository_root=ROOT)
    assert captured.value.code == "output_exists"


def test_bundle_refuses_private_field_even_when_rehashed(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    summary_path = candidate / "public-summary-v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["real_window_count"] = 1
    write_json(summary_path, summary)
    _reindex(candidate)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone7_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "private_field_in_public_evidence"


def test_bundle_refuses_claim_expansion_even_when_rehashed(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    summary_path = candidate / "public-summary-v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["inference_authorized"] = True
    write_json(summary_path, summary)
    _reindex(candidate)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone7_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "public_summary_mismatch"


def test_bundle_refuses_source_manifest_tampering(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    manifest_path = candidate / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["milestone6_plan_sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    _reindex(candidate)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone7_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "source_manifest_mismatch"


def test_bundle_refuses_extra_file(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    (candidate / "extra.json").write_text("{}", encoding="ascii")
    with pytest.raises(EvidenceError) as captured:
        verify_milestone7_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "bundle_file_set_mismatch"


@pytest.mark.skipif(os.name == "nt", reason="symlink permissions vary on Windows")
def test_bundle_refuses_symlink(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    readme = candidate / "README.txt"
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="ascii")
    readme.unlink()
    readme.symlink_to(target)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone7_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "special_file_forbidden"


def test_public_bundle_contains_only_synthetic_counts_and_statuses() -> None:
    joined = b"\n".join(path.read_bytes() for path in BUNDLE.iterdir())
    for forbidden in (
        b"/Users/",
        b"one-volume-custody",
        b"preprocessing-v1",
        b"input-reference.f32le",
        b"preprocessing-report",
        b'"real_window_count":',
        b'"tensor_sha256":',
        b'"window_stream_sha256":',
    ):
        assert forbidden not in joined
    summary = json.loads((BUNDLE / "public-summary-v1.json").read_text(encoding="utf-8"))
    assert summary["synthetic_fixture_count"] == 8
    assert summary["synthetic_refusal_fixture_count"] == 3
    assert summary["real_execution_status"] == "not-run"


def test_cli_errors_redact_private_paths(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "/private/subject-secret"
    read_descriptor, write_descriptor = os.pipe()
    try:
        os.write(
            write_descriptor,
            canonical_json_bytes(
                {
                    "approve_plan_sha256": sha256_file(PLAN),
                    "approve_preprocessing_report_sha256": "a" * 64,
                    "approve_preprocessing_snapshot_sha256": "b" * 64,
                    "root": secret,
                }
            ),
        )
        os.close(write_descriptor)
        write_descriptor = -1
        result = main(
            [
                "private-verify",
                "--plan",
                str(PLAN),
                "--authorization-fd",
                str(read_descriptor),
            ]
        )
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)
    captured = capsys.readouterr()
    assert result in {1, 2}
    assert secret not in captured.err
    assert secret not in captured.out


def test_cli_verifies_milestone7_bundle(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["public-verify", "--bundle", str(BUNDLE)]) == 0
    captured = capsys.readouterr()
    assert TRUSTED_PUBLIC_BUNDLE_SHA256 in captured.out
    assert "private_data_accessed=false" in captured.out
    assert captured.err == ""


def test_historical_bundle_identities_are_preserved() -> None:
    assert sha256_file(ROOT / "research/milestone-5/bundle.json") == (
        MILESTONE5_PUBLIC_BUNDLE_SHA256
    )
    assert sha256_file(ROOT / "research/milestone-6/bundle.json") == (
        MILESTONE6_PUBLIC_BUNDLE_SHA256
    )
    assert LEGACY_SYNTHETIC_BUNDLE_SHA256 == (
        "c7f42adb2ec3d3f6ea51a7574e7dfc03e5198954ef6e25b6e41fe1d2517bd15c"
    )
