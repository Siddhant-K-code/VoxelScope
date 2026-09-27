# SPDX-License-Identifier: Apache-2.0
"""Fail-closed acquisition and custody verification without archive extraction."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import IO, Any, Protocol, cast

from .atomic import link_file_no_replace, path_occupied
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    is_link_like,
    load_json,
    load_json_bytes,
    safe_relative_path,
    sha256_file,
)
from .custody_records import (
    AcquisitionArtifact,
    AcquisitionPlan,
    ArchivePolicy,
    DigestIdentity,
    SourceRegistry,
)
from .real_data_contract import verify_plan_contract, verify_registry_contract

_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_EXCLUDED_PUBLIC_DIRECTORIES = frozenset(
    {
        ".agent-traces",
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
)
_FORBIDDEN_PUBLIC_SUFFIXES = (
    ".ckpt",
    ".dcm",
    ".nii",
    ".nii.gz",
    ".onnx",
    ".pt",
    ".pth",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
)


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...

    def hexdigest(self) -> str: ...


class _Response(Protocol):
    status: int
    headers: Message

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def load_source_registry(path: Path) -> SourceRegistry:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "source registry must be a regular file")
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError("invalid_source_registry", "source registry must be an object")
    return SourceRegistry.from_dict(data)


def load_acquisition_plan(path: Path) -> AcquisitionPlan:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "acquisition plan must be a regular file")
    data = load_json(path)
    if not isinstance(data, dict):
        raise EvidenceError("invalid_acquisition_plan", "acquisition plan must be an object")
    return AcquisitionPlan.from_dict(data)


def verify_public_contracts(registry_path: Path, plan_path: Path | None = None) -> str:
    registry = load_source_registry(registry_path)
    verify_registry_contract(registry)
    if plan_path is not None:
        plan = load_acquisition_plan(plan_path)
        verify_plan_contract(plan, registry)
    return sha256_file(registry_path)


def render_plan(plan: AcquisitionPlan, registry: SourceRegistry) -> str:
    verify_plan_contract(plan, registry)
    lines = [
        f"Plan: {plan.plan_id}",
        f"Overall status: {plan.overall_status.upper()}",
        "Network default: disabled",
    ]
    for artifact in plan.artifacts:
        line = f"- {artifact.artifact_id}: {artifact.status.upper()}"
        if artifact.blocked_reason is not None:
            line += f" ({artifact.blocked_reason})"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _origin(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise EvidenceError("unsafe_url", "only credential-free HTTPS URLs are allowed")
    return f"{parsed.scheme}://{parsed.netloc}"


def validate_download_url(artifact: AcquisitionArtifact, url: str) -> None:
    if _origin(url) not in artifact.allowed_origins:
        raise EvidenceError("redirect_not_allowlisted", _origin(url))


def init_private_root(root: Path) -> None:
    if not root.is_absolute():
        raise EvidenceError("private_root_required", "custody root must be absolute")
    if path_occupied(root):
        raise EvidenceError("output_exists", str(root))
    if not root.parent.is_dir() or is_link_like(root.parent):
        raise EvidenceError("unsafe_path", "custody root parent must be a regular directory")
    root.mkdir(mode=0o700)
    if os.name != "nt":
        os.chmod(root, 0o700)


def require_private_root(root: Path) -> None:
    if not root.is_absolute() or is_link_like(root) or not root.is_dir():
        raise EvidenceError(
            "private_root_required", "custody root must be an absolute regular directory"
        )
    if os.name != "nt":
        metadata = root.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise EvidenceError("private_root_permissions", "custody root must be owner-only")


def _mkdir_private_chain(root: Path, relative_parent: PurePosixPath) -> Path:
    current = root
    for part in relative_parent.parts:
        current /= part
        if path_occupied(current):
            if is_link_like(current) or not current.is_dir():
                raise EvidenceError("unsafe_path", str(relative_parent))
        else:
            current.mkdir(mode=0o700)
        if os.name != "nt":
            os.chmod(current, 0o700)
    return current


def _hashers(digests: Iterable[DigestIdentity]) -> dict[str, _Digest]:
    result: dict[str, _Digest] = {}
    for digest in digests:
        if digest.algorithm not in {"md5", "sha1", "sha256"}:
            raise EvidenceError(
                "unsupported_download_digest",
                f"download verification does not accept {digest.algorithm}",
            )
        result[digest.algorithm] = hashlib.new(digest.algorithm, usedforsecurity=False)
    return result


def _open_allowlisted(artifact: AcquisitionArtifact) -> _Response:
    opener = urllib.request.build_opener(_NoRedirect)
    url = artifact.source_url
    for _ in range(4):
        validate_download_url(artifact, url)
        request = urllib.request.Request(url, headers={"User-Agent": "VoxelScope-custody/1"})
        try:
            return cast(_Response, opener.open(request, timeout=60))
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location")
            if exc.code not in _REDIRECT_CODES or location is None:
                raise EvidenceError("download_failed", f"HTTP {exc.code}") from exc
            url = urllib.parse.urljoin(url, location)
    raise EvidenceError("too_many_redirects", artifact.artifact_id)


def _check_hashes(artifact: AcquisitionArtifact, size: int, actual: dict[str, str]) -> None:
    if size != artifact.expected_size_bytes:
        raise EvidenceError(
            "artifact_size_mismatch",
            f"expected {artifact.expected_size_bytes}, observed {size}",
        )
    for digest in artifact.expected_hashes:
        observed = actual.get(digest.algorithm)
        if observed != digest.value:
            raise EvidenceError("artifact_hash_mismatch", digest.algorithm)


def download_artifact(artifact: AcquisitionArtifact, root: Path, *, allow_network: bool) -> Path:
    if not allow_network:
        raise EvidenceError(
            "network_opt_in_required", "pass --allow-network to acquire an artifact"
        )
    if artifact.status != "ready":
        raise EvidenceError("acquisition_blocked", artifact.blocked_reason or artifact.artifact_id)
    require_private_root(root)
    relative = safe_relative_path(artifact.destination)
    destination = ensure_no_symlink(root, relative)
    if path_occupied(destination):
        raise EvidenceError("output_exists", artifact.destination)
    parent = _mkdir_private_chain(root, PurePosixPath(*relative.parts[:-1]))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{relative.name}.", suffix=".part", dir=parent
    )
    os.chmod(temporary_name, 0o600)
    temporary = Path(temporary_name)
    hashers = _hashers(artifact.expected_hashes)
    size = 0
    response: _Response | None = None
    try:
        response = _open_allowlisted(artifact)
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != artifact.expected_size_bytes:
            raise EvidenceError("artifact_size_mismatch", "HTTP Content-Length differs")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > artifact.expected_size_bytes:
                    raise EvidenceError("artifact_size_mismatch", "download exceeded expected size")
                for hasher in hashers.values():
                    hasher.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        actual = {algorithm: hasher.hexdigest() for algorithm, hasher in hashers.items()}
        _check_hashes(artifact, size, actual)
        if artifact.archive is not None:
            try:
                verify_zip_archive(temporary, artifact.archive)
            except (RuntimeError, zipfile.BadZipFile) as exc:
                raise EvidenceError("invalid_archive", str(exc)) from exc
        link_file_no_replace(temporary, destination)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            response.close()
    return destination


def _validate_archive_path(value: str) -> str:
    if value.endswith("/"):
        value = value[:-1]
    safe_relative_path(value)
    return value


def verify_zip_archive(path: Path, policy: ArchivePolicy) -> dict[str, Any]:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "staged archive must be a regular file")
    if not zipfile.is_zipfile(path):
        raise EvidenceError("invalid_archive", "expected a ZIP archive")
    expected = {member.path: member for member in policy.exact_members}
    observed: dict[str, tuple[int, str]] = {}
    total_size = 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > policy.max_files:
            raise EvidenceError("archive_file_limit", str(len(entries)))
        for info in entries:
            member_path = _validate_archive_path(info.filename)
            if member_path in observed:
                raise EvidenceError("duplicate_archive_member", member_path)
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise EvidenceError("archive_symlink_forbidden", member_path)
            if info.flag_bits & 0x1:
                raise EvidenceError("encrypted_archive_forbidden", member_path)
            if info.is_dir():
                raise EvidenceError("unexpected_archive_member", member_path)
            if info.file_size > policy.max_file_size_bytes:
                raise EvidenceError("archive_file_size_limit", member_path)
            if info.file_size > 0 and (
                info.compress_size == 0
                or info.file_size > info.compress_size * policy.max_compression_ratio
            ):
                raise EvidenceError("archive_compression_ratio_limit", member_path)
            total_size += info.file_size
            if total_size > policy.max_total_uncompressed_bytes:
                raise EvidenceError("archive_total_size_limit", str(total_size))
            expected_member = expected.get(member_path)
            if expected_member is None:
                raise EvidenceError("unexpected_archive_member", member_path)
            digest = hashlib.sha256()
            observed_size = 0
            with archive.open(info, "r") as stream:
                while chunk := stream.read(1024 * 1024):
                    observed_size += len(chunk)
                    if observed_size > policy.max_file_size_bytes:
                        raise EvidenceError("archive_file_size_limit", member_path)
                    digest.update(chunk)
            if observed_size != info.file_size:
                raise EvidenceError("archive_member_size_mismatch", member_path)
            if (
                expected_member.size_bytes != observed_size
                or expected_member.sha256 != digest.hexdigest()
            ):
                raise EvidenceError("archive_member_identity_mismatch", member_path)
            observed[member_path] = (observed_size, digest.hexdigest())
    for license_path in policy.license_paths:
        if license_path not in observed:
            raise EvidenceError("missing_license", license_path)
    missing = set(expected) - set(observed)
    if missing:
        raise EvidenceError("missing_archive_member", sorted(missing)[0])
    return {
        "archive_file_count": len(observed),
        "archive_total_uncompressed_bytes": total_size,
        "license_paths": sorted(policy.license_paths),
    }


def verify_staged_artifact(artifact: AcquisitionArtifact, root: Path) -> dict[str, Any]:
    if artifact.status != "ready":
        raise EvidenceError("acquisition_blocked", artifact.blocked_reason or artifact.artifact_id)
    require_private_root(root)
    relative = safe_relative_path(artifact.destination)
    path = ensure_no_symlink(root, relative)
    if not path.is_file():
        raise EvidenceError("missing_staged_artifact", artifact.destination)
    hashers = _hashers(artifact.expected_hashes)
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > artifact.expected_size_bytes:
                raise EvidenceError("artifact_size_mismatch", artifact.artifact_id)
            for hasher in hashers.values():
                hasher.update(chunk)
    actual = {algorithm: hasher.hexdigest() for algorithm, hasher in hashers.items()}
    _check_hashes(artifact, size, actual)
    archive_result: dict[str, Any] = {}
    if artifact.archive is not None:
        try:
            archive_result = verify_zip_archive(path, artifact.archive)
        except (RuntimeError, zipfile.BadZipFile) as exc:
            raise EvidenceError("invalid_archive", str(exc)) from exc
    return {
        "artifact_id": artifact.artifact_id,
        "hashes": actual,
        "relative_path": artifact.destination,
        "size_bytes": size,
        **archive_result,
    }


def _write_private_receipt(
    root: Path,
    plan: AcquisitionPlan,
    artifact: AcquisitionArtifact,
    result: dict[str, Any],
) -> Path:
    receipts = _mkdir_private_chain(root, PurePosixPath("receipts"))
    destination = receipts / f"{artifact.artifact_id}.json"
    if path_occupied(destination):
        raise EvidenceError("output_exists", f"receipts/{artifact.artifact_id}.json")
    receipt = {
        "artifact": result,
        "plan_id": plan.plan_id,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        "registry_id": plan.registry_id,
        "research_only": True,
        "schema_version": "voxelscope/custody-receipt/v1",
        "source_id": artifact.source_id,
        "source_url": artifact.source_url,
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.artifact_id}.", suffix=".tmp", dir=receipts
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(canonical_json_bytes(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        link_file_no_replace(temporary, destination)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return destination


def verify_and_receipt(plan: AcquisitionPlan, artifact_id: str, root: Path) -> dict[str, Any]:
    artifact = next((item for item in plan.artifacts if item.artifact_id == artifact_id), None)
    if artifact is None:
        raise EvidenceError("unknown_artifact", artifact_id)
    result = verify_staged_artifact(artifact, root)
    _write_private_receipt(root, plan, artifact, result)
    return result


def _public_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        directory = stack.pop()
        if is_link_like(directory):
            raise EvidenceError("symlink_forbidden", str(directory.relative_to(root)))
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name in _EXCLUDED_PUBLIC_DIRECTORIES or entry.name.endswith(".egg-info"):
                    continue
                path = Path(entry.path)
                if entry.is_symlink() or is_link_like(path):
                    raise EvidenceError("symlink_forbidden", str(path.relative_to(root)))
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    yield path
                else:
                    raise EvidenceError("special_file_forbidden", str(path.relative_to(root)))


def scan_public_tree(root: Path) -> int:
    if is_link_like(root) or not root.is_dir():
        raise EvidenceError("unsafe_path", "public root must be a regular directory")
    private_path_patterns = (
        re.compile(rb"/" + rb"Users/[A-Za-z0-9._-]+/"),
        re.compile(rb"/" + rb"home/[A-Za-z0-9._-]+/"),
        re.compile(rb"[A-Za-z]:\\\\" + rb"Users\\\\"),
    )
    sensitive_patterns = (
        (re.compile(rb"AKIA[0-9A-Z]{16}"), "cloud credential"),
        (re.compile(rb"AWS_" + rb"SECRET_ACCESS_KEY"), "cloud credential"),
        (re.compile(rb"(?:NGC|NVIDIA)_" + rb"API_KEY"), "cloud credential"),
        (re.compile(rb"ghp_" + rb"[A-Za-z0-9]{20,}"), "cloud credential"),
        (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
        (re.compile(rb"(?:X-Amz-Credential|Signature)="), "signed private URL"),
        (re.compile(rb"BRATS_?[0-9]{3,}"), "subject identifier"),
    )
    count = 0
    for path in _public_files(root):
        relative = path.relative_to(root).as_posix()
        lower = relative.lower()
        if lower.endswith(_FORBIDDEN_PUBLIC_SUFFIXES):
            raise EvidenceError("private_artifact_in_repository", relative)
        count += 1
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        data = path.read_bytes()
        for pattern in private_path_patterns:
            if pattern.search(data):
                raise EvidenceError("private_path_in_repository", relative)
        for pattern, description in sensitive_patterns:
            if pattern.search(data):
                raise EvidenceError("sensitive_content_in_repository", f"{relative}: {description}")
        if path.suffix == ".json":
            try:
                value = load_json_bytes(data, require_canonical=False)
            except EvidenceError:
                continue
            if isinstance(value, dict) and value.get("schema_version") == (
                "voxelscope/custody-receipt/v1"
            ):
                raise EvidenceError("private_receipt_in_repository", relative)
    return count
