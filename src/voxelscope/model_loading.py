# SPDX-License-Identifier: Apache-2.0
"""One-shot offline extraction and CPU model-loading qualification."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .atomic import path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    is_link_like,
    load_json,
    load_json_bytes,
    require_sha256,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
)
from .custody import require_private_root
from .custody_records import AcquisitionArtifact, AcquisitionPlan, ArchiveMember
from .model_loading_contract import (
    ACQUISITION_PLAN_SHA256,
    MILESTONE7_PLAN_SHA256,
    MILESTONE7_PUBLIC_BUNDLE_SHA256,
    MODEL_ARCHIVE_SHA256,
    SOURCE_REGISTRY_SHA256,
    TRUSTED_PLAN_SHA256,
)
from .model_loading_records import (
    ARCHIVE_PATH,
    ATTEMPTS_NAME,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    EXTRACTION_PATHS,
    MANIFEST_PATH,
    RECEIPT_PATH,
    REPORT_PATH,
    SNAPSHOT_NAME,
    AttemptStage,
    ExtractionManifest,
    ExtractionMember,
    LoaderWorkerRequest,
    LoaderWorkerResult,
    ModelLoadingAttempt,
    ModelLoadingCompletion,
    ModelLoadingPlan,
    ModelLoadingRefusal,
    ModelLoadingReport,
    PrivateArtifact,
)
from .one_volume_custody import (
    _fsync_directory,
    _mkdir_private_chain,
    _write_private_json_no_clobber,
)
from .records import (
    require_object,
    strict_fields,
)

_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_WORKER_REQUEST_BYTES = 16 * 1024
_MAX_WORKER_RESULT_BYTES = 64 * 1024
_AUTHORIZATION_SCHEMA = "voxelscope/model-loading-authorization/v1"


@dataclass
class _Progress:
    last_completed_stage: AttemptStage = "attempt_marker"
    extracted_member_count: int = 0
    snapshot_published: bool = False
    model_instantiated: bool = False
    model_loaded: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_model_loading_plan(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[ModelLoadingPlan, str]:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "model-loading plan must be a regular file")
    digest = sha256_file(path)
    if TRUSTED_PLAN_SHA256 != "TO_BE_FINALIZED" and digest != TRUSTED_PLAN_SHA256:
        raise EvidenceError("model_loading_plan_digest_mismatch", "trusted digest differs")
    plan = ModelLoadingPlan.from_dict(require_object(load_json(path), "model-loading plan"))
    root = repository_root.resolve(strict=True)
    upstream = (
        ("research/source-registry-v1.json", SOURCE_REGISTRY_SHA256),
        ("research/acquisition-plan-v1.json", ACQUISITION_PLAN_SHA256),
        ("research/window-execution-plan-v1.json", MILESTONE7_PLAN_SHA256),
        ("research/milestone-7/bundle.json", MILESTONE7_PUBLIC_BUNDLE_SHA256),
        (plan.runtime.requirements_path, plan.runtime.requirements_sha256),
    )
    for relative, expected in upstream:
        source = ensure_no_symlink(root, safe_relative_path(relative))
        if not source.is_file() or sha256_file(source) != expected:
            raise EvidenceError("upstream_identity_mismatch", relative)
    for identity in plan.implementation_sources:
        source = ensure_no_symlink(root, safe_relative_path(identity.path))
        if not source.is_file() or sha256_file(source) != identity.sha256:
            raise EvidenceError("implementation_identity_mismatch", identity.path)
    return plan, digest


def _load_acquisition_artifact(repository_root: Path) -> AcquisitionArtifact:
    path = ensure_no_symlink(
        repository_root, safe_relative_path("research/acquisition-plan-v1.json")
    )
    if not path.is_file() or sha256_file(path) != ACQUISITION_PLAN_SHA256:
        raise EvidenceError("acquisition_plan_identity_mismatch", "trusted plan differs")
    plan = AcquisitionPlan.from_dict(require_object(load_json(path), "acquisition plan"))
    artifact = next(
        (item for item in plan.artifacts if item.artifact_id == "monai-brats-bundle-v0.5.2"),
        None,
    )
    if artifact is None or artifact.archive is None or artifact.status != "ready":
        raise EvidenceError("model_artifact_contract_missing", "official bundle is unavailable")
    return artifact


def _authorization(
    *,
    plan_sha256: str,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    approve_archive_sha256: str,
    approve_worker_python_sha256: str,
    approve_worker_runtime_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    if approve_plan_sha256 != plan_sha256:
        raise EvidenceError("wrong_model_loading_plan_approval", "approval differs")
    if approve_archive_sha256 != MODEL_ARCHIVE_SHA256:
        raise EvidenceError("wrong_model_archive_approval", "archive approval differs")
    for field, value in (
        ("approve_custody_receipt_sha256", approve_custody_receipt_sha256),
        ("approve_archive_sha256", approve_archive_sha256),
        ("approve_worker_python_sha256", approve_worker_python_sha256),
        ("approve_worker_runtime_sha256", approve_worker_runtime_sha256),
    ):
        require_sha256(value, field)
    authorization = {
        "accelerator_allowed": False,
        "approved_archive_sha256": approve_archive_sha256,
        "approved_custody_receipt_sha256": approve_custody_receipt_sha256,
        "approved_plan_sha256": approve_plan_sha256,
        "approved_worker_python_sha256": approve_worker_python_sha256,
        "approved_worker_runtime_sha256": approve_worker_runtime_sha256,
        "forward_allowed": False,
        "inference_authorized": False,
        "network_allowed": False,
        "schema_version": _AUTHORIZATION_SCHEMA,
    }
    authorization_sha256 = sha256_bytes(canonical_json_bytes(authorization))
    attempt_id = sha256_bytes(
        canonical_json_bytes(
            {
                "archive_sha256": approve_archive_sha256,
                "custody_receipt_sha256": approve_custody_receipt_sha256,
                "plan_sha256": approve_plan_sha256,
                "worker_python_sha256": approve_worker_python_sha256,
                "worker_runtime_sha256": approve_worker_runtime_sha256,
            }
        )
    )
    return authorization, authorization_sha256, attempt_id


def _attempt_paths(root: Path, attempt_id: str) -> tuple[Path, Path]:
    require_sha256(attempt_id, "attempt_id")
    attempts = ensure_no_symlink(root, safe_relative_path(ATTEMPTS_NAME))
    return attempts / f"{attempt_id}.started.json", attempts / f"{attempt_id}.terminal.json"


def _copy_immutable_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if is_link_like(source) or not source.is_file() or path_occupied(destination):
        raise EvidenceError("unsafe_path", "immutable source or destination")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(destination.parent, 0o700)
    source_path_before = source.stat()
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd = -1
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise EvidenceError("archive_identity_mismatch", "archive extent differs")
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        while chunk := os.read(source_fd, _COPY_CHUNK_BYTES):
            size += len(chunk)
            if size > expected_size:
                raise EvidenceError("archive_replacement", "archive extent changed")
            digest.update(chunk)
            pending = memoryview(chunk)
            while pending:
                written = os.write(destination_fd, pending)
                pending = pending[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        source_path_after = source.stat()
        stable_fd = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        stable_path = (
            source_path_before.st_dev,
            source_path_before.st_ino,
            source_path_before.st_size,
            source_path_before.st_mtime_ns,
            source_path_before.st_ctime_ns,
        ) == (
            source_path_after.st_dev,
            source_path_after.st_ino,
            source_path_after.st_size,
            source_path_after.st_mtime_ns,
            source_path_after.st_ctime_ns,
        )
        if (
            size != expected_size
            or digest.hexdigest() != expected_sha256
            or not stable_fd
            or not stable_path
        ):
            raise EvidenceError("archive_replacement", "archive changed during staging")
        if os.name != "nt":
            os.fchmod(destination_fd, 0o400)
    except BaseException:
        if destination_fd >= 0:
            os.close(destination_fd)
            destination_fd = -1
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)


def _zip_member_path(info: zipfile.ZipInfo) -> str:
    if info.is_dir() or info.filename.endswith("/"):
        raise EvidenceError("archive_special_member", info.filename)
    safe_relative_path(info.filename)
    if PurePosixPath(info.filename).as_posix() != info.filename:
        raise EvidenceError("unsafe_archive_path", info.filename)
    return info.filename


def extract_selected_archive(
    archive_path: Path,
    destination: Path,
    *,
    exact_members: tuple[ArchiveMember, ...],
    selected_members: tuple[ExtractionMember, ...],
    max_members: int,
    max_total_uncompressed_bytes: int,
    max_member_bytes: int,
    max_compression_ratio: int,
) -> tuple[PrivateArtifact, ...]:
    """Validate a closed ZIP and extract only the exact selected regular members."""
    if is_link_like(archive_path) or not archive_path.is_file():
        raise EvidenceError("unsafe_path", "archive must be a regular file")
    expected = {member.path: member for member in exact_members}
    selected = {member.path: member for member in selected_members}
    if not selected or not set(selected).issubset(expected):
        raise EvidenceError("unsafe_extraction_allowlist", "selection is not closed")
    if path_occupied(destination):
        raise EvidenceError("output_exists", str(destination))
    destination.mkdir(mode=0o700, parents=False)
    if os.name != "nt":
        os.chmod(destination, 0o700)
    observed: set[str] = set()
    folded: set[str] = set()
    extracted: dict[str, PrivateArtifact] = {}
    total_size = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            if len(entries) > max_members:
                raise EvidenceError("archive_file_limit", str(len(entries)))
            for info in entries:
                member_path = _zip_member_path(info)
                folded_path = member_path.casefold()
                if member_path in observed:
                    raise EvidenceError("duplicate_archive_member", member_path)
                if folded_path in folded:
                    raise EvidenceError("archive_case_collision", member_path)
                observed.add(member_path)
                folded.add(folded_path)
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG}:
                    raise EvidenceError("archive_special_member", member_path)
                if info.flag_bits & 0x1:
                    raise EvidenceError("encrypted_archive_forbidden", member_path)
                if info.file_size > max_member_bytes:
                    raise EvidenceError("archive_file_size_limit", member_path)
                if info.file_size > 0 and (
                    info.compress_size <= 0
                    or info.file_size > info.compress_size * max_compression_ratio
                ):
                    raise EvidenceError("archive_compression_ratio_limit", member_path)
                total_size += info.file_size
                if total_size > max_total_uncompressed_bytes:
                    raise EvidenceError("archive_total_size_limit", str(total_size))
                expected_member = expected.get(member_path)
                if expected_member is None:
                    raise EvidenceError("unexpected_archive_member", member_path)
                digest = hashlib.sha256()
                size = 0
                selected_member = selected.get(member_path)
                output_path: Path | None = None
                output_fd = -1
                if selected_member is not None:
                    relative = safe_relative_path(f"extracted/{member_path}")
                    output_path = destination / relative
                    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    if os.name != "nt":
                        current = destination
                        for part in relative.parts[:-1]:
                            current /= part
                            os.chmod(current, 0o700)
                    output_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with archive.open(info, "r") as source:
                        while chunk := source.read(_COPY_CHUNK_BYTES):
                            size += len(chunk)
                            if size > max_member_bytes or size > info.file_size:
                                raise EvidenceError("archive_member_size_mismatch", member_path)
                            digest.update(chunk)
                            if output_fd >= 0:
                                pending = memoryview(chunk)
                                while pending:
                                    written = os.write(output_fd, pending)
                                    pending = pending[written:]
                    if output_fd >= 0:
                        os.fsync(output_fd)
                finally:
                    if output_fd >= 0:
                        os.close(output_fd)
                observed_digest = digest.hexdigest()
                if (
                    size != info.file_size
                    or size != expected_member.size_bytes
                    or observed_digest != expected_member.sha256
                ):
                    raise EvidenceError("archive_member_identity_mismatch", member_path)
                if selected_member is not None:
                    if (
                        size != selected_member.size_bytes
                        or observed_digest != selected_member.sha256
                        or output_path is None
                    ):
                        raise EvidenceError("selected_member_identity_mismatch", member_path)
                    extracted[f"extracted/{member_path}"] = PrivateArtifact(
                        f"extracted/{member_path}", observed_digest, size
                    )
        if observed != set(expected):
            missing = sorted(set(expected) - observed)
            raise EvidenceError(
                "missing_archive_member", missing[0] if missing else "member set differs"
            )
        expected_output_paths = tuple(f"extracted/{member.path}" for member in selected_members)
        if set(extracted) != set(expected_output_paths):
            raise EvidenceError("extraction_file_set_mismatch", "selected members differ")
        return tuple(extracted[path] for path in expected_output_paths)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _verify_receipt(
    root: Path,
    *,
    approved_receipt_sha256: str,
    approved_archive_sha256: str,
) -> None:
    receipt_path = ensure_no_symlink(root, safe_relative_path(RECEIPT_PATH))
    if (
        not receipt_path.is_file()
        or is_link_like(receipt_path)
        or sha256_file(receipt_path) != approved_receipt_sha256
    ):
        raise EvidenceError("custody_receipt_identity_mismatch", "approved receipt differs")
    receipt = strict_fields(
        require_object(load_json(receipt_path), "custody receipt"),
        {
            "artifact",
            "plan_id",
            "plan_sha256",
            "registry_id",
            "research_only",
            "schema_version",
            "source_id",
            "source_url",
            "verified_at",
        },
        "custody receipt",
    )
    artifact = strict_fields(
        require_object(receipt["artifact"], "receipt artifact"),
        {
            "archive_file_count",
            "archive_total_uncompressed_bytes",
            "artifact_id",
            "hashes",
            "license_paths",
            "relative_path",
            "size_bytes",
        },
        "receipt artifact",
    )
    hashes = require_object(artifact["hashes"], "receipt hashes")
    if (
        receipt["schema_version"] != "voxelscope/custody-receipt/v1"
        or receipt["plan_id"] != "voxelscope-private-custody-v1"
        or receipt["plan_sha256"] != ACQUISITION_PLAN_SHA256
        or receipt["registry_id"] != "voxelscope-real-data-readiness-v1"
        or receipt["research_only"] is not True
        or receipt["source_id"] != "monai-brats-mri-segmentation-ngc-v0.5.2"
        or receipt["source_url"]
        != (
            "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/"
            "brats_mri_segmentation/versions/0.5.2/files/"
            "brats_mri_segmentation_v0.5.2.zip"
        )
        or type(receipt["verified_at"]) is not str
        or not receipt["verified_at"]
        or artifact["artifact_id"] != "monai-brats-bundle-v0.5.2"
        or artifact["relative_path"] != ARCHIVE_PATH
        or artifact["size_bytes"] != 35082630
        or artifact["archive_total_uncompressed_bytes"] != 37799553
        or artifact["license_paths"] != ["brats_mri_segmentation/LICENSE"]
        or hashes
        != {
            "sha1": "6b1dfef29d49c6f6a1d8bf9f65c84125ad37a6e9",
            "sha256": approved_archive_sha256,
        }
        or hashes.get("sha256") != approved_archive_sha256
        or artifact["archive_file_count"] != 13
    ):
        raise EvidenceError("custody_receipt_contract_mismatch", "receipt fields differ")


def _worker_environment() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "LANG": "C",
        "LC_ALL": "C",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "ROCR_VISIBLE_DEVICES": "",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def run_loader_worker(
    worker_python: Path,
    request: LoaderWorkerRequest,
    *,
    timeout_seconds: int,
    approved_python_sha256: str,
) -> LoaderWorkerResult:
    executable = worker_python.absolute()
    if not executable.is_file() or sha256_file(executable) != approved_python_sha256:
        raise EvidenceError("worker_python_identity_mismatch", "approved executable differs")
    worker_source = Path(__file__).with_name("model_loader_worker.py")
    if not worker_source.is_file() or sha256_file(worker_source) != request.worker_source_sha256:
        raise EvidenceError("worker_source_identity_mismatch", "worker source differs")
    payload = canonical_json_bytes(request)
    if len(payload) > _MAX_WORKER_REQUEST_BYTES:
        raise EvidenceError("worker_request_limit", "request is too large")
    if os.name == "nt":
        if request.operation != "synthetic-json":
            raise EvidenceError(
                "private_acl_unverified",
                "Windows private execution is unsupported",
            )
        from .model_loader_protocol import WorkerRequest
        from .model_loader_worker import execute_request

        worker_request = WorkerRequest.from_dict(request.to_dict())
        return LoaderWorkerResult.from_dict(execute_request(worker_request).to_dict())
    request_read, request_write = os.pipe()
    result_read, result_write = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [
                "voxelscope-model-loader",
                "-I",
                str(worker_source),
                "--request-fd",
                str(request_read),
                "--result-fd",
                str(result_write),
            ],
            executable=str(executable),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(request_read, result_write),
            cwd="/",
            env=_worker_environment(),
            start_new_session=True,
        )
        os.close(request_read)
        request_read = -1
        os.close(result_write)
        result_write = -1
        pending = memoryview(payload)
        try:
            while pending:
                written = os.write(request_write, pending)
                pending = pending[written:]
        except BrokenPipeError as exc:
            raise EvidenceError("loader_worker_crash", "worker closed the request") from exc
        os.close(request_write)
        request_write = -1
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise EvidenceError("loader_worker_timeout", "worker exceeded deadline") from exc
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(result_read, _MAX_WORKER_RESULT_BYTES + 1 - size):
            size += len(chunk)
            if size > _MAX_WORKER_RESULT_BYTES:
                raise EvidenceError("loader_worker_result_limit", "worker result is too large")
            chunks.append(chunk)
        if sha256_file(executable) != approved_python_sha256:
            raise EvidenceError("worker_python_replacement", "worker executable changed")
        if not chunks:
            raise EvidenceError("loader_worker_crash", f"worker exited {return_code}")
        result = LoaderWorkerResult.from_dict(
            require_object(load_json_bytes(b"".join(chunks)), "loader worker result")
        )
        if return_code == 0 and result.status != "go":
            raise EvidenceError("loader_worker_protocol_mismatch", "success exit refused")
        if return_code != 0 and result.status == "go":
            raise EvidenceError("loader_worker_protocol_mismatch", "failure exit claimed success")
        return result
    finally:
        for descriptor in (request_read, request_write, result_read, result_write):
            if descriptor >= 0:
                os.close(descriptor)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def _private_artifact(root: Path, relative: str) -> PrivateArtifact:
    path = ensure_no_symlink(root, safe_relative_path(relative))
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("missing_private_artifact", relative)
    return PrivateArtifact(relative, sha256_file(path), path.stat().st_size)


def _snapshot_identity(root: Path) -> str:
    artifacts = tuple(
        _private_artifact(root, relative)
        for relative in (*EXTRACTION_PATHS, MANIFEST_PATH, REPORT_PATH)
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "artifacts": [artifact.to_dict() for artifact in artifacts],
                "schema_version": "voxelscope/model-loading-snapshot-identity/v1",
                "snapshot": SNAPSHOT_NAME,
            }
        )
    )


def _write_terminal(root: Path, attempt_id: str, value: Mapping[str, Any]) -> None:
    relative = f"{ATTEMPTS_NAME}/{attempt_id}.terminal.json"
    try:
        _write_private_json_no_clobber(root, relative, dict(value))
    except BaseException:
        destination = ensure_no_symlink(root, safe_relative_path(relative))
        if (
            destination.is_file()
            and not is_link_like(destination)
            and destination.read_bytes() == canonical_json_bytes(value)
        ):
            return
        raise


def _record_refusal(
    root: Path,
    *,
    attempt_id: str,
    plan_sha256: str,
    authorization_sha256: str,
    attempt_record_sha256: str,
    receipt_sha256: str,
    archive_sha256: str,
    code: str,
    progress: _Progress,
) -> None:
    refusal = ModelLoadingRefusal(
        "voxelscope/model-loading-refusal/v1",
        "refused",
        code,
        progress.last_completed_stage,
        plan_sha256,
        authorization_sha256,
        attempt_record_sha256,
        receipt_sha256,
        archive_sha256,
        progress.extracted_member_count,
        progress.snapshot_published,
        progress.model_instantiated,
        progress.model_loaded,
        False,
        False,
        False,
        _now(),
    )
    _write_terminal(root, attempt_id, refusal.to_dict())


def execute_model_loading_qualification(
    plan_path: Path,
    root: Path,
    *,
    worker_python: Path,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    approve_archive_sha256: str,
    approve_worker_python_sha256: str,
    approve_worker_runtime_sha256: str,
    repository_root: Path,
) -> ModelLoadingReport:
    plan, plan_sha256 = load_model_loading_plan(plan_path, repository_root=repository_root)
    require_private_root(root, repository_root=repository_root)
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        approve_archive_sha256=approve_archive_sha256,
        approve_worker_python_sha256=approve_worker_python_sha256,
        approve_worker_runtime_sha256=approve_worker_runtime_sha256,
    )
    attempts = _mkdir_private_chain(root, PurePosixPath(ATTEMPTS_NAME))
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    snapshot = root / SNAPSHOT_NAME
    if path_occupied(started_path) or path_occupied(terminal_path):
        raise EvidenceError("model_loading_attempt_exists", "approval is consumed")
    if path_occupied(snapshot):
        raise EvidenceError("output_exists", SNAPSHOT_NAME)
    attempt = ModelLoadingAttempt(
        "voxelscope/model-loading-attempt/v1",
        plan_sha256,
        authorization_sha256,
        approve_custody_receipt_sha256,
        approve_archive_sha256,
        approve_worker_python_sha256,
        approve_worker_runtime_sha256,
        (
            "approved private model custody receipt",
            "approved official model archive",
            "approved isolated Python runtime executable",
            "approved closed PyTorch and MONAI runtime fingerprint",
        ),
        (
            "three allowlisted extracted model members",
            "one closed extraction manifest and model-loading report",
            "one terminal completion or refusal record",
        ),
        _now(),
    )
    attempt_record_sha256 = sha256_bytes(canonical_json_bytes(attempt))
    progress = _Progress()
    stage: Path | None = None
    marker_owned = False
    try:
        try:
            _write_private_json_no_clobber(
                root, f"{ATTEMPTS_NAME}/{attempt_id}.started.json", attempt.to_dict()
            )
            marker_owned = True
        except EvidenceError as exc:
            if exc.code == "output_exists":
                raise EvidenceError("model_loading_attempt_exists", "approval is consumed") from exc
            raise
        if started_path.parent != attempts or sha256_file(started_path) != attempt_record_sha256:
            raise EvidenceError("model_loading_attempt_mismatch", "attempt marker differs")
        _verify_receipt(
            root,
            approved_receipt_sha256=approve_custody_receipt_sha256,
            approved_archive_sha256=approve_archive_sha256,
        )
        progress.last_completed_stage = "receipt_verification"
        artifact = _load_acquisition_artifact(repository_root)
        archive_source = ensure_no_symlink(root, safe_relative_path(plan.archive_path))
        archive_metadata = archive_source.stat()
        if os.name != "nt" and (
            archive_metadata.st_uid != os.getuid()
            or stat.S_IMODE(archive_metadata.st_mode) != 0o600
        ):
            raise EvidenceError("private_file_permissions", "model archive")
        stage = Path(tempfile.mkdtemp(prefix=".model-loading-stage-", dir=root))
        if os.name != "nt":
            os.chmod(stage, 0o700)
        staged_archive = stage / ".archive.zip"
        _copy_immutable_file(
            archive_source,
            staged_archive,
            expected_sha256=plan.archive_sha256,
            expected_size=plan.archive_size_bytes,
        )
        progress.last_completed_stage = "archive_staging"
        if artifact.archive is None:
            raise EvidenceError("model_artifact_contract_missing", "archive policy missing")
        members = extract_selected_archive(
            staged_archive,
            stage / "selected",
            exact_members=artifact.archive.exact_members,
            selected_members=plan.extraction_members,
            max_members=plan.archive_max_members,
            max_total_uncompressed_bytes=plan.archive_max_total_uncompressed_bytes,
            max_member_bytes=plan.archive_max_member_bytes,
            max_compression_ratio=plan.archive_max_compression_ratio,
        )
        progress.last_completed_stage = "archive_verification"
        selected = stage / "selected"
        for item in selected.iterdir():
            item.rename(stage / item.name)
        selected.rmdir()
        staged_archive.unlink()
        progress.extracted_member_count = len(members)
        progress.last_completed_stage = "member_extraction"
        manifest = ExtractionManifest(
            "voxelscope/model-extraction-manifest/v1",
            "go",
            plan_sha256,
            approve_archive_sha256,
            approve_custody_receipt_sha256,
            members,
            3,
            False,
        )
        manifest_path = _write_private_json_no_clobber(stage, MANIFEST_PATH, manifest.to_dict())
        checkpoint = ensure_no_symlink(stage, safe_relative_path(CHECKPOINT_PATH))
        config = ensure_no_symlink(stage, safe_relative_path(CONFIG_PATH))
        request = LoaderWorkerRequest(
            "voxelscope/loader-worker-request/v1",
            "official-torch",
            plan_sha256,
            next(
                identity.sha256
                for identity in plan.implementation_sources
                if identity.path == "src/voxelscope/model_loader_worker.py"
            ),
            str(checkpoint),
            sha256_file(checkpoint),
            str(config),
            sha256_file(config),
            plan.architecture,
            plan.runtime,
            approve_worker_runtime_sha256,
            plan.loader_memory_limit_bytes,
        )
        progress.model_instantiated = True
        progress.model_loaded = True
        loader_result = run_loader_worker(
            worker_python,
            request,
            timeout_seconds=plan.loader_timeout_seconds,
            approved_python_sha256=approve_worker_python_sha256,
        )
        progress.model_instantiated = loader_result.model_instantiated
        progress.model_loaded = loader_result.state_loaded
        if loader_result.status != "go":
            raise EvidenceError(
                loader_result.error_code or "loader_worker_refused",
                "isolated loader refused the checkpoint",
            )
        progress.last_completed_stage = "loader_worker"
        manifest_artifact = PrivateArtifact(
            MANIFEST_PATH, sha256_file(manifest_path), manifest_path.stat().st_size
        )
        report = ModelLoadingReport(
            "voxelscope/model-loading-report/v1",
            "go",
            plan_sha256,
            authorization_sha256,
            attempt_record_sha256,
            approve_custody_receipt_sha256,
            approve_archive_sha256,
            manifest_artifact,
            loader_result,
            plan.runtime,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            _now(),
        )
        report_path = _write_private_json_no_clobber(stage, REPORT_PATH, report.to_dict())
        report_sha256 = sha256_file(report_path)
        progress.last_completed_stage = "report_publication"
        _fsync_directory(stage / "evidence")
        _fsync_directory(stage)
        rename_no_replace(stage, snapshot)
        progress.snapshot_published = True
        progress.last_completed_stage = "snapshot_publication"
        _fsync_directory(root)
        snapshot_sha256 = _snapshot_identity(snapshot)
        completion = ModelLoadingCompletion(
            "voxelscope/model-loading-completion/v1",
            "completed",
            plan_sha256,
            authorization_sha256,
            attempt_record_sha256,
            approve_custody_receipt_sha256,
            approve_archive_sha256,
            snapshot_sha256,
            report_sha256,
            SNAPSHOT_NAME,
            False,
            False,
            False,
            _now(),
        )
        _write_terminal(root, attempt_id, completion.to_dict())
        return report
    except BaseException as exc:
        code = exc.code if isinstance(exc, EvidenceError) else "model_loading_interrupted"
        if marker_owned:
            try:
                if path_occupied(terminal_path):
                    terminal = require_object(load_json(terminal_path), "model-loading terminal")
                    if terminal.get("status") != "completed":
                        ModelLoadingRefusal.from_dict(terminal)
                else:
                    _record_refusal(
                        root,
                        attempt_id=attempt_id,
                        plan_sha256=plan_sha256,
                        authorization_sha256=authorization_sha256,
                        attempt_record_sha256=attempt_record_sha256,
                        receipt_sha256=approve_custody_receipt_sha256,
                        archive_sha256=approve_archive_sha256,
                        code=code,
                        progress=progress,
                    )
            except BaseException as refusal_error:
                raise EvidenceError("refusal_write_failed", code) from refusal_error
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if isinstance(exc, EvidenceError):
            raise
        if isinstance(exc, KeyboardInterrupt):
            raise EvidenceError(
                "model_loading_interrupted", "approved attempt is consumed"
            ) from exc
        raise


def _verify_snapshot_file_set(snapshot: Path) -> None:
    expected = {*EXTRACTION_PATHS, MANIFEST_PATH, REPORT_PATH}
    observed: set[str] = set()
    for directory, subdirectories, filenames in os.walk(snapshot, followlinks=False):
        current = Path(directory)
        if is_link_like(current) or (
            os.name != "nt" and stat.S_IMODE(current.stat().st_mode) != 0o700
        ):
            raise EvidenceError("private_directory_permissions", "model-loading snapshot")
        for name in subdirectories:
            if is_link_like(current / name):
                raise EvidenceError("symlink_forbidden", "model-loading snapshot")
        for name in filenames:
            path = current / name
            if (
                is_link_like(path)
                or not path.is_file()
                or (os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600)
            ):
                raise EvidenceError("private_file_permissions", "model-loading snapshot")
            observed.add(path.relative_to(snapshot).as_posix())
    if observed != expected:
        raise EvidenceError("model_loading_snapshot_file_set_mismatch", "file set differs")


def verify_model_loading_qualification(
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    approve_archive_sha256: str,
    approve_worker_python_sha256: str,
    approve_worker_runtime_sha256: str,
    approve_snapshot_sha256: str,
    approve_loading_report_sha256: str,
    repository_root: Path,
) -> ModelLoadingReport:
    plan, plan_sha256 = load_model_loading_plan(plan_path, repository_root=repository_root)
    require_private_root(root, repository_root=repository_root)
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        approve_archive_sha256=approve_archive_sha256,
        approve_worker_python_sha256=approve_worker_python_sha256,
        approve_worker_runtime_sha256=approve_worker_runtime_sha256,
    )
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    attempt = ModelLoadingAttempt.from_dict(
        require_object(load_json(started_path), "model-loading attempt")
    )
    if (
        attempt.plan_sha256 != plan_sha256
        or attempt.authorization_sha256 != authorization_sha256
        or attempt.receipt_sha256 != approve_custody_receipt_sha256
        or attempt.archive_sha256 != approve_archive_sha256
        or attempt.worker_python_sha256 != approve_worker_python_sha256
        or attempt.worker_runtime_sha256 != approve_worker_runtime_sha256
        or sha256_bytes(canonical_json_bytes(authorization)) != authorization_sha256
    ):
        raise EvidenceError("model_loading_attempt_mismatch", "attempt fields differ")
    terminal = ModelLoadingCompletion.from_dict(
        require_object(load_json(terminal_path), "model-loading completion")
    )
    snapshot = ensure_no_symlink(root, safe_relative_path(SNAPSHOT_NAME))
    if is_link_like(snapshot) or not snapshot.is_dir():
        raise EvidenceError("missing_model_loading_snapshot", SNAPSHOT_NAME)
    _verify_snapshot_file_set(snapshot)
    snapshot_sha256 = _snapshot_identity(snapshot)
    report_path = ensure_no_symlink(snapshot, safe_relative_path(REPORT_PATH))
    report_sha256 = sha256_file(report_path)
    if (
        snapshot_sha256 != approve_snapshot_sha256
        or snapshot_sha256 != terminal.snapshot_sha256
        or report_sha256 != approve_loading_report_sha256
        or report_sha256 != terminal.report_sha256
    ):
        raise EvidenceError("model_loading_output_identity_mismatch", "approval differs")
    report = ModelLoadingReport.from_dict(
        require_object(load_json(report_path), "model-loading report")
    )
    manifest_path = ensure_no_symlink(snapshot, safe_relative_path(MANIFEST_PATH))
    manifest = ExtractionManifest.from_dict(
        require_object(load_json(manifest_path), "extraction manifest")
    )
    if (
        terminal.plan_sha256 != plan_sha256
        or terminal.authorization_sha256 != authorization_sha256
        or terminal.attempt_record_sha256 != sha256_file(started_path)
        or terminal.receipt_sha256 != approve_custody_receipt_sha256
        or terminal.archive_sha256 != approve_archive_sha256
        or terminal.snapshot != SNAPSHOT_NAME
        or report.plan_sha256 != plan_sha256
        or report.authorization_sha256 != authorization_sha256
        or report.attempt_record_sha256 != sha256_file(started_path)
        or report.receipt_sha256 != approve_custody_receipt_sha256
        or report.archive_sha256 != approve_archive_sha256
        or report.runtime != plan.runtime
        or report.loader_result.status != "go"
        or report.extraction_manifest.sha256 != sha256_file(manifest_path)
        or manifest.plan_sha256 != plan_sha256
        or manifest.receipt_sha256 != approve_custody_receipt_sha256
        or manifest.archive_sha256 != approve_archive_sha256
    ):
        raise EvidenceError("model_loading_report_mismatch", "private evidence differs")
    expected_members = tuple(
        (output_path, planned.size_bytes, planned.sha256)
        for output_path, planned in zip(EXTRACTION_PATHS, plan.extraction_members, strict=True)
    )
    observed_members = tuple(
        (member.path, member.size_bytes, member.sha256) for member in manifest.members
    )
    if observed_members != expected_members:
        raise EvidenceError("extraction_manifest_plan_mismatch", "members differ")
    for member in manifest.members:
        path = ensure_no_symlink(snapshot, safe_relative_path(member.path))
        if (
            not path.is_file()
            or path.stat().st_size != member.size_bytes
            or sha256_file(path) != member.sha256
        ):
            raise EvidenceError("extracted_member_identity_mismatch", member.path)
    return report


def verify_model_loading_refusal(
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    approve_archive_sha256: str,
    approve_worker_python_sha256: str,
    approve_worker_runtime_sha256: str,
    repository_root: Path,
) -> ModelLoadingRefusal:
    _, plan_sha256 = load_model_loading_plan(plan_path, repository_root=repository_root)
    require_private_root(root, repository_root=repository_root)
    _, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        approve_archive_sha256=approve_archive_sha256,
        approve_worker_python_sha256=approve_worker_python_sha256,
        approve_worker_runtime_sha256=approve_worker_runtime_sha256,
    )
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    attempt = ModelLoadingAttempt.from_dict(
        require_object(load_json(started_path), "model-loading attempt")
    )
    refusal = ModelLoadingRefusal.from_dict(
        require_object(load_json(terminal_path), "model-loading refusal")
    )
    if (
        attempt.plan_sha256 != plan_sha256
        or attempt.authorization_sha256 != authorization_sha256
        or attempt.receipt_sha256 != approve_custody_receipt_sha256
        or attempt.archive_sha256 != approve_archive_sha256
        or attempt.worker_python_sha256 != approve_worker_python_sha256
        or attempt.worker_runtime_sha256 != approve_worker_runtime_sha256
        or refusal.plan_sha256 != plan_sha256
        or refusal.authorization_sha256 != authorization_sha256
        or refusal.attempt_record_sha256 != sha256_file(started_path)
        or refusal.receipt_sha256 != approve_custody_receipt_sha256
        or refusal.archive_sha256 != approve_archive_sha256
    ):
        raise EvidenceError("model_loading_refusal_mismatch", "refusal fields differ")
    return refusal
