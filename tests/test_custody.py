# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from voxelscope.canonical import EvidenceError, canonical_json_bytes, load_json
from voxelscope.custody import (
    download_artifact,
    load_acquisition_plan,
    load_source_registry,
    scan_public_tree,
    validate_download_url,
    verify_and_receipt,
    verify_public_contracts,
    verify_zip_archive,
)
from voxelscope.custody_records import (
    AcquisitionArtifact,
    AcquisitionPlan,
    ArchiveMember,
    ArchivePolicy,
    DigestIdentity,
)
from voxelscope.real_data_contract import verify_plan_contract

ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "research" / "source-registry-v1.json"
PLAN = ROOT / "research" / "acquisition-plan-v1.json"


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path / "custody"
    root.mkdir(mode=0o700)
    if os.name != "nt":
        root.chmod(0o700)
    return root


def _member(path: str, data: bytes, role: str = "test") -> ArchiveMember:
    return ArchiveMember(path, len(data), hashlib.sha256(data).hexdigest(), role)


def _policy(
    members: tuple[ArchiveMember, ...],
    *,
    license_paths: tuple[str, ...] = ("LICENSE",),
    max_files: int = 8,
    max_total: int = 1024,
    max_file: int = 1024,
    max_ratio: int = 100,
) -> ArchivePolicy:
    return ArchivePolicy(
        "zip",
        max_files,
        max_total,
        max_file,
        max_ratio,
        license_paths,
        members,
    )


def test_public_registry_and_plan_are_canonical_and_trusted() -> None:
    registry = load_source_registry(REGISTRY)
    plan = load_acquisition_plan(PLAN)
    verify_plan_contract(plan, registry)
    assert REGISTRY.read_bytes() == canonical_json_bytes(registry)
    assert PLAN.read_bytes() == canonical_json_bytes(plan)
    assert (
        verify_public_contracts(REGISTRY, PLAN) == hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    )


def test_source_registry_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    value = load_json(REGISTRY)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(EvidenceError) as caught:
        load_source_registry(path)
    assert caught.value.code == "noncanonical_json"


def test_redirect_origin_must_be_allowlisted() -> None:
    artifact = load_acquisition_plan(PLAN).artifacts[0]
    with pytest.raises(EvidenceError) as caught:
        validate_download_url(artifact, "https://example.invalid/model.zip")
    assert caught.value.code == "redirect_not_allowlisted"


def test_redirect_url_must_not_contain_credentials() -> None:
    artifact = load_acquisition_plan(PLAN).artifacts[0]
    with pytest.raises(EvidenceError) as caught:
        validate_download_url(
            artifact,
            "https://user:password@api.ngc.nvidia.com/model.zip",
        )
    assert caught.value.code == "unsafe_url"


def test_download_requires_explicit_network_opt_in(tmp_path: Path) -> None:
    artifact = load_acquisition_plan(PLAN).artifacts[0]
    with pytest.raises(EvidenceError) as caught:
        download_artifact(artifact, _private_root(tmp_path), allow_network=False)
    assert caught.value.code == "network_opt_in_required"


def test_download_refuses_blocked_artifact_before_network(tmp_path: Path) -> None:
    artifact = load_acquisition_plan(PLAN).artifacts[1]
    with pytest.raises(EvidenceError) as caught:
        download_artifact(artifact, _private_root(tmp_path), allow_network=True)
    assert caught.value.code == "acquisition_blocked"


def test_download_refuses_existing_destination_before_network(tmp_path: Path) -> None:
    artifact = load_acquisition_plan(PLAN).artifacts[0]
    root = _private_root(tmp_path)
    destination = root / artifact.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")
    with pytest.raises(EvidenceError) as caught:
        download_artifact(artifact, root, allow_network=True)
    assert caught.value.code == "output_exists"
    assert destination.read_bytes() == b"existing"


def test_zip_verifier_accepts_exact_closed_archive(tmp_path: Path) -> None:
    license_data = b"Apache License 2.0\n"
    config_data = b'{"version":"test"}\n'
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("LICENSE", license_data)
        archive.writestr("configs/metadata.json", config_data)
    result = verify_zip_archive(
        path,
        _policy(
            (
                _member("LICENSE", license_data, "license"),
                _member("configs/metadata.json", config_data, "metadata"),
            )
        ),
    )
    assert result["archive_file_count"] == 2


def test_zip_verifier_rejects_path_traversal(tmp_path: Path) -> None:
    path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape", b"x")
        archive.writestr("LICENSE", b"license")
    with pytest.raises(EvidenceError) as caught:
        verify_zip_archive(path, _policy((_member("LICENSE", b"license"),)))
    assert caught.value.code == "unsafe_path"


def test_zip_verifier_rejects_symlink(tmp_path: Path) -> None:
    path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, b"target")
        archive.writestr("LICENSE", b"license")
    policy = _policy((_member("link", b"target"), _member("LICENSE", b"license")))
    with pytest.raises(EvidenceError) as caught:
        verify_zip_archive(path, policy)
    assert caught.value.code == "archive_symlink_forbidden"


def test_zip_verifier_rejects_file_count_limit(tmp_path: Path) -> None:
    path = tmp_path / "count.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("LICENSE", b"license")
        archive.writestr("extra", b"x")
    policy = _policy((_member("LICENSE", b"license"), _member("extra", b"x")), max_files=1)
    with pytest.raises(EvidenceError) as caught:
        verify_zip_archive(path, policy)
    assert caught.value.code == "archive_file_limit"


def test_zip_verifier_rejects_total_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "size.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("LICENSE", b"license")
        archive.writestr("large", b"x" * 20)
    policy = _policy(
        (_member("LICENSE", b"license"), _member("large", b"x" * 20)),
        max_total=10,
    )
    with pytest.raises(EvidenceError) as caught:
        verify_zip_archive(path, policy)
    assert caught.value.code == "archive_total_size_limit"


def test_zip_verifier_rejects_compression_bomb_ratio(tmp_path: Path) -> None:
    payload = b"0" * 4096
    path = tmp_path / "ratio.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE", b"license")
        archive.writestr("compressed", payload)
    policy = _policy(
        (_member("LICENSE", b"license"), _member("compressed", payload)),
        max_total=8192,
        max_file=8192,
        max_ratio=2,
    )
    with pytest.raises(EvidenceError) as caught:
        verify_zip_archive(path, policy)
    assert caught.value.code == "archive_compression_ratio_limit"


def test_zip_verifier_requires_license_member(tmp_path: Path) -> None:
    path = tmp_path / "no-license.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("payload", b"x")
    policy = _policy((_member("LICENSE", b"license"), _member("payload", b"x")))
    with pytest.raises(EvidenceError) as caught:
        verify_zip_archive(path, policy)
    assert caught.value.code == "missing_license"


def test_private_receipt_is_owner_only_and_no_clobber(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    payload = b"Apache License 2.0\n"
    archive_relative = Path("artifacts/test.zip")
    archive_path = root / archive_relative
    archive_path.parent.mkdir(mode=0o700)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("LICENSE", payload)
    artifact = AcquisitionArtifact(
        artifact_id="test-bundle",
        source_id="test-source",
        role="test",
        source_url="https://example.invalid/test.zip",
        allowed_origins=("https://example.invalid",),
        destination=archive_relative.as_posix(),
        expected_size_bytes=archive_path.stat().st_size,
        expected_hashes=(
            DigestIdentity("sha256", hashlib.sha256(archive_path.read_bytes()).hexdigest()),
        ),
        status="ready",
        blocked_reason=None,
        archive=_policy((_member("LICENSE", payload, "license"),)),
    )
    plan = AcquisitionPlan(
        "voxelscope/acquisition-plan/v1",
        "test-plan",
        "test-registry",
        "disabled",
        "no-go",
        (artifact,),
    )
    verify_and_receipt(plan, artifact.artifact_id, root)
    receipt = root / "receipts" / "test-bundle.json"
    assert receipt.is_file()
    if os.name != "nt":
        assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    with pytest.raises(EvidenceError) as caught:
        verify_and_receipt(plan, artifact.artifact_id, root)
    assert caught.value.code == "output_exists"


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("weights.pt", b"not a real model", "private_artifact_in_repository"),
        (
            "notes.txt",
            ("/" + "Users" + "/alice/private/model.zip").encode(),
            "private_path_in_repository",
        ),
        (
            "subject.txt",
            ("BRATS" + "_001").encode(),
            "sensitive_content_in_repository",
        ),
    ],
)
def test_public_scan_rejects_private_material(
    tmp_path: Path, name: str, content: bytes, code: str
) -> None:
    (tmp_path / name).write_bytes(content)
    with pytest.raises(EvidenceError) as caught:
        scan_public_tree(tmp_path)
    assert caught.value.code == code


def test_public_scan_rejects_private_receipt(tmp_path: Path) -> None:
    receipt = {
        "artifact": {},
        "research_only": True,
        "schema_version": "voxelscope/custody-receipt/v1",
        "verified_at": "2026-09-27T00:00:00Z",
    }
    (tmp_path / "receipt.json").write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(EvidenceError) as caught:
        scan_public_tree(tmp_path)
    assert caught.value.code == "private_receipt_in_repository"
