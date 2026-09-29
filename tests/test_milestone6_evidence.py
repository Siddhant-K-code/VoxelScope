# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from voxelscope.canonical import EvidenceError, sha256_file, write_json
from voxelscope.milestone6_cli import main
from voxelscope.milestone6_evidence import (
    EXPECTED_PUBLIC_BUNDLE_SHA256,
    INDEXED_PATHS,
    PAYLOAD_PATHS,
    build_milestone6_public_bundle,
    verify_milestone6_public_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/window-bridge-plan-v1.json"
REPORT = ROOT / "research/window-bridge-report-v1.json"
BUNDLE = ROOT / "research/milestone-6"


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
            "schema_version": "voxelscope/milestone-6-public-evidence/v1",
        },
    )
    digest = sha256_file(root / "bundle.json")
    (root / "bundle.sha256").write_bytes(f"{digest}  bundle.json\n".encode("ascii"))


def test_committed_milestone6_bundle_is_trusted() -> None:
    assert (
        verify_milestone6_public_bundle(BUNDLE, repository_root=ROOT)
        == EXPECTED_PUBLIC_BUNDLE_SHA256
    )


def test_bundle_rebuild_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    digest = build_milestone6_public_bundle(
        PLAN,
        REPORT,
        output,
        repository_root=ROOT,
    )
    assert digest == EXPECTED_PUBLIC_BUNDLE_SHA256
    assert {
        path.relative_to(output).as_posix(): path.read_bytes() for path in output.iterdir()
    } == {path.relative_to(BUNDLE).as_posix(): path.read_bytes() for path in BUNDLE.iterdir()}


def test_bundle_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(EvidenceError) as captured:
        build_milestone6_public_bundle(
            PLAN,
            REPORT,
            output,
            repository_root=ROOT,
        )
    assert captured.value.code == "output_exists"


def test_bundle_refuses_success_shaped_claim_expansion(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    summary_path = candidate / "public-summary-v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["inference_authorized"] = True
    write_json(summary_path, summary)
    _reindex(candidate)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone6_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "public_summary_mismatch"


def test_bundle_refuses_private_field_even_when_rehashed(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    summary_path = candidate / "public-summary-v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["input_spatial_shape"] = [1, 2, 3]
    write_json(summary_path, summary)
    _reindex(candidate)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone6_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "private_field_in_public_evidence"


def test_bundle_refuses_source_manifest_tampering_when_rehashed(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    manifest_path = candidate / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_sources"][0]["sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    _reindex(candidate)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone6_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "source_manifest_mismatch"


def test_bundle_refuses_extra_files(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    (candidate / "extra.json").write_text("{}", encoding="ascii")
    with pytest.raises(EvidenceError) as captured:
        verify_milestone6_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "bundle_file_set_mismatch"


@pytest.mark.skipif(os.name == "nt", reason="symlink permissions vary on Windows")
def test_bundle_refuses_symlinks(tmp_path: Path) -> None:
    candidate = _copy_bundle(tmp_path)
    readme = candidate / "README.txt"
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="ascii")
    readme.unlink()
    readme.symlink_to(target)
    with pytest.raises(EvidenceError) as captured:
        verify_milestone6_public_bundle(
            candidate,
            repository_root=ROOT,
            require_trusted_digest=False,
        )
    assert captured.value.code == "special_file_forbidden"


def test_cli_verifies_milestone6_bundle(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["public-verify", "--bundle", str(BUNDLE)]) == 0
    captured = capsys.readouterr()
    assert EXPECTED_PUBLIC_BUNDLE_SHA256 in captured.out
    assert "inference_authorized=false" in captured.out
    assert captured.err == ""


def test_bundle_contains_no_private_identifiers() -> None:
    joined = b"\n".join(path.read_bytes() for path in BUNDLE.iterdir())
    for forbidden in (
        b"/Users/",
        b"one-volume-custody",
        b"input-reference.f32le",
        b"input-independent.f32le",
        b"preprocessing-report",
        b"4a73ef8f",
        b"0584b51b",
        b"ee268fb3",
    ):
        assert forbidden not in joined


def test_bundle_machine_encodes_legacy_padded_reuse_refusal() -> None:
    summary = json.loads((BUNDLE / "public-summary-v1.json").read_text(encoding="utf-8"))
    assert summary["legacy_unpadded_enumeration_status"] == "go"
    assert summary["legacy_general_padded_reuse_status"] == "no-go"
    assert "legacy_positional_engine_status" not in summary
