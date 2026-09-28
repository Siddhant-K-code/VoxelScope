# SPDX-License-Identifier: Apache-2.0
"""Fail-closed acquisition and structural validation for the trusted one-volume plan."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import struct
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import IO, Any, Protocol, cast

import nibabel as nib
import numpy as np

from .atomic import link_file_no_replace, path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    is_link_like,
    load_json,
    load_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
)
from .custody import require_private_root
from .one_volume import verify_one_volume_decision, verify_one_volume_plan
from .one_volume_contract import DECISION_SHA256, PLAN_SHA256
from .one_volume_custody_records import (
    AcquisitionRefusal,
    CustodyReceipt,
    NiftiStructure,
    PrivateArtifactResult,
    RefusalStage,
    StructuralReport,
)
from .one_volume_records import ArtifactRole, OneVolumeAcquisitionPlan, PlannedArtifact
from .records import require_list, require_object, require_string, strict_fields

SNAPSHOT_NAME = "one-volume-v1"
ATTEMPT_SCHEMA = "voxelscope/one-volume-acquisition-attempt/v1"
AUTHORIZATION_SCHEMA = "voxelscope/one-volume-acquisition-authorization/v1"
COMPLETION_SCHEMA = "voxelscope/one-volume-acquisition-completion/v1"
COMMAND_SCHEMA = "voxelscope/one-volume-custody-command/v1"
STRUCTURAL_REPORT_PATH = "evidence/structural-report.json"
RECEIPT_PATH = "evidence/custody-receipt.json"
MAX_NIFTI_DIMENSION = 512
MAX_NIFTI_VOXELS = 64_000_000
MAX_NIFTI_DATA_BYTES = 256 * 1024 * 1024
MAX_NIFTI_DECOMPRESSED_BYTES = MAX_NIFTI_DATA_BYTES + 1024
ALLOWED_SOURCE_LABELS = frozenset({0, 1, 2, 3})
TUMOR_SOURCE_LABELS = frozenset({1, 2, 3})
IMAGE_ROLES: tuple[ArtifactRole, ...] = (
    "image-t1c",
    "image-t1",
    "image-t2",
    "image-flair",
)
MEDICAL_ROLES: tuple[ArtifactRole, ...] = (*IMAGE_ROLES, "label")
_READ_CHUNK_BYTES = 1024 * 1024
_SUPPORTED_DATATYPES = {
    2: 1,
    4: 2,
    8: 4,
    16: 4,
    64: 8,
    256: 1,
    512: 2,
    768: 4,
    1024: 8,
    1280: 8,
}


class _Response(Protocol):
    status: int
    headers: Message

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...

    def geturl(self) -> str: ...


class _ArrayProxy(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    offset: int
    slope: float
    inter: float

    def __getitem__(self, key: object) -> Any: ...


class _NiftiHeader(Protocol):
    def get_zooms(self) -> tuple[float, ...]: ...

    def get_xyzt_units(self) -> tuple[str, str]: ...

    def get_qform(self, coded: bool = False) -> tuple[np.ndarray | None, int]: ...

    def get_sform(self, coded: bool = False) -> tuple[np.ndarray | None, int]: ...


OpenResponse = Callable[[str], _Response]


@dataclass
class _AcquisitionProgress:
    last_completed_stage: RefusalStage = "attempt_record"
    verified_artifact_count: int = 0
    validated_medical_file_count: int = 0
    geometry_comparison_count: int = 0


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_private_chain(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current /= part
        if path_occupied(current):
            if is_link_like(current) or not current.is_dir():
                raise EvidenceError("unsafe_path", relative.as_posix())
        else:
            current.mkdir(mode=0o700)
            if os.name != "nt":
                os.chmod(current, 0o700)
            _fsync_directory(current.parent)
        if os.name != "nt":
            metadata = current.stat()
            if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                raise EvidenceError("private_directory_permissions", relative.as_posix())
    return current


def _write_private_json_no_clobber(root: Path, relative: str, value: Any) -> Path:
    relative_path = safe_relative_path(relative)
    parent = _mkdir_private_chain(root, PurePosixPath(*relative_path.parts[:-1]))
    destination = ensure_no_symlink(root, relative_path)
    if path_occupied(destination):
        raise EvidenceError("output_exists", relative)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{relative_path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        link_file_no_replace(temporary, destination)
        _fsync_directory(parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _authorization(approve_plan_sha256: str) -> tuple[dict[str, Any], str, str]:
    if approve_plan_sha256 != PLAN_SHA256:
        raise EvidenceError("wrong_plan_approval", "approval does not match the trusted plan")
    approval_sha256 = sha256_bytes(approve_plan_sha256.encode("ascii"))
    authorization = {
        "approved_plan_sha256": approve_plan_sha256,
        "decision_sha256": DECISION_SHA256,
        "network_scope": "exact-eight-pinned-objects",
        "plan_sha256": PLAN_SHA256,
        "schema_version": AUTHORIZATION_SCHEMA,
    }
    return authorization, approval_sha256, sha256_bytes(canonical_json_bytes(authorization))


def _load_trusted_plan(
    decision_path: Path, plan_path: Path
) -> tuple[OneVolumeAcquisitionPlan, str, str]:
    decision, decision_sha256 = verify_one_volume_decision(decision_path)
    plan, plan_sha256 = verify_one_volume_plan(plan_path, decision)
    if decision_sha256 != DECISION_SHA256 or plan_sha256 != PLAN_SHA256:
        raise EvidenceError("one_volume_contract_mismatch", "trusted file hash")
    return plan, decision_sha256, plan_sha256


def _validate_exact_url(artifact: PlannedArtifact) -> str | None:
    parsed = urllib.parse.urlsplit(artifact.source_url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.hostname
    ):
        raise EvidenceError("unsafe_url", artifact.artifact_id)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if artifact.allowed_origins != (origin,):
        raise EvidenceError("unsafe_origin_policy", artifact.artifact_id)
    if parsed.hostname == "s3.amazonaws.com":
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        expected_version = artifact.immutable_id.removeprefix("s3-version-id:")
        if (
            not artifact.immutable_id.startswith("s3-version-id:")
            or query != [("versionId", expected_version)]
            or not parsed.path.startswith("/openneuro.org/ds007045/")
        ):
            raise EvidenceError("unsafe_version_pin", artifact.artifact_id)
        return expected_version
    if parsed.hostname == "raw.githubusercontent.com":
        parts = parsed.path.split("/")
        expected_blob = artifact.immutable_id.removeprefix("git-blob-sha1:")
        if (
            parsed.query
            or not artifact.immutable_id.startswith("git-blob-sha1:")
            or len(parts) < 5
            or parts[1:3] != ["OpenNeuroDatasets", "ds007045"]
            or parts[3] != "283007b96977b5b3285bd22cd67b7d97114b1eed"
            or len(expected_blob) != 40
        ):
            raise EvidenceError("unsafe_version_pin", artifact.artifact_id)
        return None
    raise EvidenceError("unsafe_origin_policy", artifact.artifact_id)


def _open_exact_url(url: str) -> _Response:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
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

    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "VoxelScope-one-volume-custody/1",
        },
    )
    try:
        return cast(_Response, opener.open(request, timeout=60))
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            raise EvidenceError("redirect_forbidden", "redirects are not authorized") from exc
        raise EvidenceError("download_failed", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise EvidenceError("download_failed", "network request failed") from exc


def _strong_etag(headers: Message, artifact_id: str) -> str:
    value = headers.get("ETag")
    if value is None or value.startswith("W/"):
        raise EvidenceError("artifact_etag_mismatch", artifact_id)
    if len(value) == 34 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise EvidenceError("artifact_etag_mismatch", artifact_id)
    return value


def _download_artifact(
    artifact: PlannedArtifact,
    stage: Path,
    *,
    open_response: OpenResponse,
) -> PrivateArtifactResult:
    expected_version = _validate_exact_url(artifact)
    expected_hashes = {item.algorithm: item.value for item in artifact.expected_hashes}
    if "sha256" not in expected_hashes:
        raise EvidenceError("missing_expected_hash", artifact.artifact_id)
    relative = safe_relative_path(artifact.destination)
    destination = ensure_no_symlink(stage, relative)
    if path_occupied(destination):
        raise EvidenceError("output_exists", artifact.artifact_id)
    parent = _mkdir_private_chain(stage, PurePosixPath(*relative.parts[:-1]))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{relative.name}.", suffix=".part", dir=parent
    )
    os.chmod(temporary_name, 0o600)
    temporary = Path(temporary_name)
    sha256 = hashlib.sha256()
    git_blob_sha1 = hashlib.sha1(usedforsecurity=False)
    git_blob_sha1.update(f"blob {artifact.expected_size_bytes}\0".encode("ascii"))
    size = 0
    response: _Response | None = None
    try:
        response = open_response(artifact.source_url)
        if response.status != 200 or response.geturl() != artifact.source_url:
            raise EvidenceError("unexpected_http_response", artifact.artifact_id)
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding not in {None, "", "identity"}:
            raise EvidenceError("unsupported_content_encoding", artifact.artifact_id)
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            if not content_length.isascii() or not content_length.isdigit():
                raise EvidenceError("invalid_content_length", artifact.artifact_id)
            if int(content_length) != artifact.expected_size_bytes:
                raise EvidenceError("artifact_size_mismatch", artifact.artifact_id)
        expected_etag = expected_hashes.get("etag")
        if (
            expected_etag is not None
            and _strong_etag(response.headers, artifact.artifact_id) != expected_etag
        ):
            raise EvidenceError("artifact_etag_mismatch", artifact.artifact_id)
        if expected_version is not None:
            observed_version = response.headers.get("x-amz-version-id")
            if observed_version != expected_version:
                raise EvidenceError("artifact_version_mismatch", artifact.artifact_id)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            while chunk := response.read(_READ_CHUNK_BYTES):
                size += len(chunk)
                if size > artifact.expected_size_bytes:
                    raise EvidenceError("artifact_size_mismatch", artifact.artifact_id)
                sha256.update(chunk)
                git_blob_sha1.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if size != artifact.expected_size_bytes:
            raise EvidenceError("artifact_size_mismatch", artifact.artifact_id)
        observed_sha256 = sha256.hexdigest()
        if observed_sha256 != expected_hashes["sha256"]:
            raise EvidenceError("artifact_hash_mismatch", "sha256")
        expected_git_blob = expected_hashes.get("git-blob-sha1")
        if expected_git_blob is not None and git_blob_sha1.hexdigest() != expected_git_blob:
            raise EvidenceError("artifact_hash_mismatch", "git-blob-sha1")
        link_file_no_replace(temporary, destination)
        os.chmod(destination, 0o600)
        _fsync_directory(parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            response.close()
    return PrivateArtifactResult(
        artifact.artifact_id,
        artifact.role,
        artifact.destination,
        size,
        observed_sha256,
        tuple(item.algorithm for item in artifact.expected_hashes),
        True,
    )


def _bounded_decompress(source: Path, destination: Path) -> int:
    if is_link_like(source) or not source.is_file():
        raise EvidenceError("unsafe_path", "NIfTI input must be a regular file")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    size = 0
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        with source.open("rb") as compressed:
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                try:
                    while compressed_chunk := compressed.read(_READ_CHUNK_BYTES):
                        if decoder.eof:
                            raise EvidenceError(
                                "gzip_trailing_member", "multiple gzip members are forbidden"
                            )
                        pending = compressed_chunk
                        while pending:
                            remaining = MAX_NIFTI_DECOMPRESSED_BYTES - size
                            expanded = decoder.decompress(pending, remaining + 1)
                            size += len(expanded)
                            if size > MAX_NIFTI_DECOMPRESSED_BYTES:
                                raise EvidenceError(
                                    "nifti_decompression_limit", "expanded file too large"
                                )
                            output.write(expanded)
                            if decoder.unused_data:
                                raise EvidenceError(
                                    "gzip_trailing_member",
                                    "multiple gzip members are forbidden",
                                )
                            pending = decoder.unconsumed_tail
                    if not decoder.eof:
                        raise EvidenceError("invalid_gzip", "compressed stream is incomplete")
                    expanded = decoder.flush()
                    size += len(expanded)
                    if size > MAX_NIFTI_DECOMPRESSED_BYTES:
                        raise EvidenceError("nifti_decompression_limit", "expanded file too large")
                    output.write(expanded)
                except zlib.error as exc:
                    raise EvidenceError("invalid_gzip", "malformed compressed NIfTI") from exc
                output.flush()
                os.fsync(output.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise
    return size


def _header_contract(path: Path) -> tuple[int, tuple[int, int, int], int, int, str]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(544)
    if len(prefix) < 352:
        raise EvidenceError("truncated_nifti", "header is incomplete")
    little_size = struct.unpack_from("<i", prefix, 0)[0]
    big_size = struct.unpack_from(">i", prefix, 0)[0]
    if little_size in {348, 540}:
        endian = "<"
        header_size = little_size
    elif big_size in {348, 540}:
        endian = ">"
        header_size = big_size
    else:
        raise EvidenceError("invalid_nifti_header", "unsupported header size")
    if len(prefix) < header_size + 4:
        raise EvidenceError("truncated_nifti", "extension marker is incomplete")
    if prefix[header_size : header_size + 4] != b"\0\0\0\0":
        raise EvidenceError("nifti_extensions_forbidden", "extensions are not authorized")
    if header_size == 348:
        version = 1
        if prefix[344:348] != b"n+1\0":
            raise EvidenceError("invalid_nifti_magic", "expected single-file NIfTI-1")
        dimensions = struct.unpack_from(f"{endian}8h", prefix, 40)
        datatype, bitpix = struct.unpack_from(f"{endian}2h", prefix, 70)
        pixdim = struct.unpack_from(f"{endian}8f", prefix, 76)
        raw_offset = struct.unpack_from(f"{endian}f", prefix, 108)[0]
        if not np.isfinite(raw_offset) or not float(raw_offset).is_integer():
            raise EvidenceError("invalid_nifti_offset", "non-integral data offset")
        data_offset = int(raw_offset)
        diagnose = cast(
            Callable[[bytes, str | None], str],
            nib.Nifti1Header.diagnose_binaryblock,
        )
        diagnostics = diagnose(prefix[:header_size], endian)
    else:
        version = 2
        if prefix[4:12] != b"n+2\0\r\n\x1a\n":
            raise EvidenceError("invalid_nifti_magic", "expected single-file NIfTI-2")
        datatype, bitpix = struct.unpack_from(f"{endian}2h", prefix, 12)
        dimensions = struct.unpack_from(f"{endian}8q", prefix, 16)
        pixdim = struct.unpack_from(f"{endian}8d", prefix, 104)
        data_offset = struct.unpack_from(f"{endian}q", prefix, 168)[0]
        diagnose = cast(
            Callable[[bytes, str | None], str],
            nib.Nifti2Header.diagnose_binaryblock,
        )
        diagnostics = diagnose(prefix[:header_size], endian)
    if diagnostics:
        raise EvidenceError("nifti_header_diagnostic", diagnostics)
    if dimensions[0] != 3 or any(item != 1 for item in dimensions[4:]):
        raise EvidenceError("invalid_nifti_shape", "exactly three dimensions are required")
    shape = cast(tuple[int, int, int], tuple(int(item) for item in dimensions[1:4]))
    if any(item <= 0 or item > MAX_NIFTI_DIMENSION for item in shape):
        raise EvidenceError("invalid_nifti_shape", "dimension is outside the fixed bound")
    voxel_count = 1
    for dimension in shape:
        voxel_count *= dimension
        if voxel_count > MAX_NIFTI_VOXELS:
            raise EvidenceError("nifti_voxel_limit", "voxel count exceeds the fixed bound")
    bytes_per_voxel = _SUPPORTED_DATATYPES.get(datatype)
    if bytes_per_voxel is None or bitpix != bytes_per_voxel * 8:
        raise EvidenceError("unsupported_nifti_dtype", str(datatype))
    data_bytes = voxel_count * bytes_per_voxel
    if data_bytes > MAX_NIFTI_DATA_BYTES:
        raise EvidenceError("nifti_data_limit", "data extent exceeds the fixed bound")
    if data_offset != header_size + 4:
        raise EvidenceError("invalid_nifti_offset", "nonstandard data offset is not authorized")
    if file_size != data_offset + data_bytes:
        raise EvidenceError("nifti_extent_mismatch", "truncated or trailing NIfTI payload")
    if any(not np.isfinite(value) or value <= 0 for value in pixdim[1:4]):
        raise EvidenceError("invalid_nifti_spacing", "spatial spacing must be finite and positive")
    if pixdim[0] not in {-1.0, 1.0}:
        raise EvidenceError("invalid_nifti_qfac", "qfac must be exactly -1 or 1")
    return version, shape, voxel_count, data_bytes, endian


def _matrix_record(value: np.ndarray | None, name: str) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise EvidenceError("invalid_affine", name)
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _inspect_nifti(
    artifact: PlannedArtifact,
    compressed_path: Path,
    inspection_root: Path,
) -> NiftiStructure:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.artifact_id}.", suffix=".nii", dir=inspection_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        expanded_size = _bounded_decompress(compressed_path, temporary)
        version, shape, voxel_count, data_bytes, endian = _header_contract(temporary)
        if expanded_size != temporary.stat().st_size:
            raise EvidenceError("nifti_extent_mismatch", artifact.artifact_id)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loaded = nib.load(str(temporary), mmap="r", keep_file_open=False)
        except Exception as exc:
            raise EvidenceError("invalid_nifti", "strict parser rejected the file") from exc
        if version == 1 and not isinstance(loaded, nib.Nifti1Image):
            raise EvidenceError("unsupported_nifti_version", artifact.artifact_id)
        if version == 2 and not isinstance(loaded, nib.Nifti2Image):
            raise EvidenceError("unsupported_nifti_version", artifact.artifact_id)
        image = cast(nib.Nifti1Image | nib.Nifti2Image, loaded)
        proxy = cast(_ArrayProxy, image.dataobj)
        proxy_shape = tuple(int(item) for item in proxy.shape)
        if proxy_shape != shape:
            raise EvidenceError("nifti_shape_mismatch", artifact.artifact_id)
        dtype = np.dtype(proxy.dtype)
        if dtype.kind not in {"i", "u", "f"} or dtype.itemsize > 8:
            raise EvidenceError("unsupported_nifti_dtype", dtype.str)
        if int(proxy.offset) != (348 if version == 1 else 540) + 4:
            raise EvidenceError("invalid_nifti_offset", artifact.artifact_id)
        slope = float(proxy.slope)
        intercept = float(proxy.inter)
        if not np.isfinite(slope) or slope == 0 or not np.isfinite(intercept):
            raise EvidenceError("invalid_nifti_scaling", artifact.artifact_id)
        header = cast(_NiftiHeader, image.header)
        spacing = tuple(float(item) for item in header.get_zooms()[:3])
        if len(spacing) != 3 or any(not np.isfinite(item) or item <= 0 for item in spacing):
            raise EvidenceError("invalid_nifti_spacing", artifact.artifact_id)
        spatial_unit, _ = header.get_xyzt_units()
        if spatial_unit != "mm":
            raise EvidenceError("invalid_nifti_units", artifact.artifact_id)
        affine_array = np.asarray(image.affine, dtype=np.float64)
        affine = _matrix_record(affine_array, "selected affine")
        if affine is None or abs(float(np.linalg.det(affine_array[:3, :3]))) <= 1e-12:
            raise EvidenceError("invalid_affine", "selected affine is singular")
        qform_value, qform_code = header.get_qform(coded=True)
        sform_value, sform_code = header.get_sform(coded=True)
        qform = _matrix_record(qform_value, "qform")
        sform = _matrix_record(sform_value, "sform")
        qform_code_int = int(qform_code)
        sform_code_int = int(sform_code)
        if qform_code_int == 0 and sform_code_int == 0:
            raise EvidenceError("missing_nifti_transform", artifact.artifact_id)
        aff2axcodes = cast(
            Callable[[np.ndarray], tuple[str | None, ...]],
            nib.orientations.aff2axcodes,
        )
        orientation_values = aff2axcodes(affine_array)
        if any(item is None for item in orientation_values):
            raise EvidenceError("invalid_nifti_orientation", artifact.artifact_id)
        orientation = tuple(cast(str, item) for item in orientation_values)
        finite_count = 0
        nonzero_count = 0
        mask_labels: set[int] = set()
        try:
            with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
                warnings.simplefilter("error")
                for index in range(shape[2]):
                    values = np.asarray(proxy[:, :, index], dtype=np.float64)
                    if values.shape != shape[:2] or not np.isfinite(values).all():
                        raise EvidenceError("nonfinite_voxel", artifact.artifact_id)
                    finite_count += values.size
                    if artifact.role == "label":
                        if not np.equal(values, np.floor(values)).all():
                            raise EvidenceError("noninteger_mask", artifact.artifact_id)
                        observed = {int(item) for item in np.unique(values)}
                        if not observed.issubset(ALLOWED_SOURCE_LABELS):
                            raise EvidenceError("unexpected_mask_label", artifact.artifact_id)
                        mask_labels.update(observed)
                    else:
                        nonzero_count += int(np.count_nonzero(values))
        except EvidenceError:
            raise
        except Exception as exc:
            raise EvidenceError("nifti_voxel_read_failed", artifact.artifact_id) from exc
        if finite_count != voxel_count:
            raise EvidenceError("nifti_voxel_count_mismatch", artifact.artifact_id)
        if artifact.role == "label":
            if 0 not in mask_labels or not mask_labels.intersection(TUMOR_SOURCE_LABELS):
                raise EvidenceError("missing_expected_mask_label", artifact.artifact_id)
        elif nonzero_count == 0:
            raise EvidenceError("empty_image", artifact.artifact_id)
        return NiftiStructure(
            artifact.artifact_id,
            artifact.role,
            cast(Any, version),
            shape,
            dtype.str,
            endian,
            voxel_count,
            data_bytes,
            spacing,
            spatial_unit,
            orientation,
            affine,
            qform_code_int,
            qform,
            sform_code_int,
            sform,
            slope,
            intercept,
            finite_count,
            None if artifact.role == "label" else nonzero_count,
            tuple(sorted(mask_labels)) if artifact.role == "label" else None,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _geometry_key(item: NiftiStructure) -> tuple[Any, ...]:
    return (
        item.shape,
        item.spacing,
        item.spatial_unit,
        item.orientation,
        item.affine,
        item.qform_code,
        item.qform,
        item.sform_code,
        item.sform,
    )


def _build_structural_report(
    artifact_root: Path,
    plan: OneVolumeAcquisitionPlan,
    *,
    decision_sha256: str,
    plan_sha256: str,
    authorization_sha256: str,
    inspection_root: Path,
    progress: _AcquisitionProgress | None = None,
) -> StructuralReport:
    by_role = {artifact.role: artifact for artifact in plan.artifacts}
    structures: list[NiftiStructure] = []
    for role in MEDICAL_ROLES:
        artifact = by_role[role]
        path = ensure_no_symlink(artifact_root, safe_relative_path(artifact.destination))
        structures.append(_inspect_nifti(artifact, path, inspection_root))
        if progress is not None:
            progress.validated_medical_file_count += 1
    image_dtypes = {item.dtype for item in structures if item.role in IMAGE_ROLES}
    if len(image_dtypes) != 1:
        raise EvidenceError("nifti_dtype_mismatch", "image modality dtypes differ")
    reference_geometry = _geometry_key(structures[0])
    for item in structures[1:]:
        if _geometry_key(item) != reference_geometry:
            raise EvidenceError("nifti_geometry_mismatch", "medical artifact geometry differs")
        if progress is not None:
            progress.geometry_comparison_count += 1
    return StructuralReport(
        "voxelscope/one-volume-structural-report/v1",
        "go",
        decision_sha256,
        plan_sha256,
        authorization_sha256,
        "exact",
        IMAGE_ROLES,
        tuple(sorted(ALLOWED_SOURCE_LABELS)),
        0,
        1,
        tuple(structures),
    )


def _verify_dataset_metadata(stage: Path, plan: OneVolumeAcquisitionPlan) -> None:
    artifact = next(item for item in plan.artifacts if item.role == "metadata-dataset-description")
    path = ensure_no_symlink(stage, safe_relative_path(artifact.destination))
    try:
        data = load_json_bytes(path.read_bytes(), require_canonical=False)
    except EvidenceError as exc:
        raise EvidenceError("invalid_dataset_metadata", "dataset description is invalid") from exc
    metadata = require_object(data, "dataset description")
    if (
        metadata.get("BIDSVersion") != "1.8.0"
        or metadata.get("License") != "CC0"
        or metadata.get("DatasetDOI") != "doi:10.18112/openneuro.ds007045.v2.0.1"
    ):
        raise EvidenceError("dataset_metadata_mismatch", "pinned dataset facts differ")


def _attempt_allowlist(plan: OneVolumeAcquisitionPlan) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": artifact.artifact_id,
            "destination": artifact.destination,
            "expected_hashes": [
                {"algorithm": digest.algorithm, "value": digest.value}
                for digest in artifact.expected_hashes
            ],
            "expected_size_bytes": artifact.expected_size_bytes,
            "immutable_id": artifact.immutable_id,
            "role": artifact.role,
            "source_url_sha256": sha256_bytes(artifact.source_url.encode("utf-8")),
        }
        for artifact in plan.artifacts
    ]


def _record_refusal(
    root: Path,
    *,
    decision_sha256: str,
    plan_sha256: str,
    approval_sha256: str,
    authorization_sha256: str,
    attempt_record_sha256: str,
    code: str,
    progress: _AcquisitionProgress,
) -> None:
    refusal = AcquisitionRefusal(
        "voxelscope/one-volume-acquisition-refusal/v1",
        "refused",
        code,
        progress.last_completed_stage,
        decision_sha256,
        plan_sha256,
        approval_sha256,
        authorization_sha256,
        attempt_record_sha256,
        progress.verified_artifact_count,
        progress.validated_medical_file_count,
        progress.geometry_comparison_count,
        False,
        False,
        _now(),
    )
    _write_private_json_no_clobber(
        root,
        f"attempts/{plan_sha256}.refusal.json",
        refusal.to_dict(),
    )


def acquire_trusted_one_volume(
    decision_path: Path,
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    allow_network: bool,
    repository_root: Path | None = None,
    open_response: OpenResponse | None = None,
) -> CustodyReceipt:
    if not allow_network:
        raise EvidenceError("network_opt_in_required", "explicit network opt-in is required")
    plan, decision_sha256, plan_sha256 = _load_trusted_plan(decision_path, plan_path)
    authorization, approval_sha256, authorization_sha256 = _authorization(approve_plan_sha256)
    require_private_root(root, repository_root=repository_root)
    snapshot = root / SNAPSHOT_NAME
    attempts = _mkdir_private_chain(root, PurePosixPath("attempts"))
    attempt_relative = f"attempts/{plan_sha256}.started.json"
    attempt_path = attempts / f"{plan_sha256}.started.json"
    if path_occupied(attempt_path):
        raise EvidenceError("acquisition_attempt_exists", "the approved attempt is consumed")
    if path_occupied(snapshot):
        raise EvidenceError("output_exists", SNAPSHOT_NAME)
    attempt = {
        "approval_sha256": approval_sha256,
        "artifact_allowlist": _attempt_allowlist(plan),
        "authorization": authorization,
        "authorization_sha256": authorization_sha256,
        "command_schema_version": COMMAND_SCHEMA,
        "decision_sha256": decision_sha256,
        "plan_sha256": plan_sha256,
        "schema_version": ATTEMPT_SCHEMA,
        "started_at": _now(),
    }
    try:
        _write_private_json_no_clobber(root, attempt_relative, attempt)
    except EvidenceError as exc:
        if exc.code == "output_exists":
            raise EvidenceError(
                "acquisition_attempt_exists", "the approved attempt is consumed"
            ) from exc
        raise
    attempt_record_sha256 = sha256_file(attempt_path)
    opener = open_response or _open_exact_url
    stage: Path | None = None
    progress = _AcquisitionProgress()
    try:
        stage = Path(tempfile.mkdtemp(prefix=".one-volume-stage-", dir=root))
        os.chmod(stage, 0o700)
        inspection = stage / ".inspection"
        inspection.mkdir(mode=0o700)
        artifact_result_list: list[PrivateArtifactResult] = []
        for artifact in plan.artifacts:
            artifact_result_list.append(_download_artifact(artifact, stage, open_response=opener))
            progress.verified_artifact_count += 1
            progress.last_completed_stage = "artifact_verification"
        artifact_results = tuple(artifact_result_list)
        progress.last_completed_stage = "content_identity"
        _verify_dataset_metadata(stage, plan)
        progress.last_completed_stage = "metadata_semantics"
        report = _build_structural_report(
            stage,
            plan,
            decision_sha256=decision_sha256,
            plan_sha256=plan_sha256,
            authorization_sha256=authorization_sha256,
            inspection_root=inspection,
            progress=progress,
        )
        progress.last_completed_stage = "structural_validation"
        inspection.rmdir()
        report_path = _write_private_json_no_clobber(
            stage, STRUCTURAL_REPORT_PATH, report.to_dict()
        )
        report_sha256 = sha256_file(report_path)
        receipt = CustodyReceipt(
            "voxelscope/one-volume-custody-receipt/v1",
            "verified",
            decision_sha256,
            plan_sha256,
            approval_sha256,
            authorization_sha256,
            attempt_record_sha256,
            tuple(item.artifact_id for item in plan.artifacts),
            artifact_results,
            STRUCTURAL_REPORT_PATH,
            report_sha256,
            False,
            False,
            _now(),
        )
        _write_private_json_no_clobber(stage, RECEIPT_PATH, receipt.to_dict())
        receipt_sha256 = sha256_file(stage / RECEIPT_PATH)
        _fsync_directory(stage / "evidence")
        _fsync_directory(stage)
        rename_no_replace(stage, snapshot)
        _fsync_directory(root)
        _write_private_json_no_clobber(
            root,
            f"attempts/{plan_sha256}.completed.json",
            {
                "attempt_record_sha256": attempt_record_sha256,
                "authorization_sha256": authorization_sha256,
                "completed_at": _now(),
                "custody_receipt_sha256": receipt_sha256,
                "plan_sha256": plan_sha256,
                "schema_version": COMPLETION_SCHEMA,
                "snapshot": SNAPSHOT_NAME,
                "status": "completed",
            },
        )
        return receipt
    except BaseException as exc:
        code = exc.code if isinstance(exc, EvidenceError) else "acquisition_interrupted"
        try:
            _record_refusal(
                root,
                decision_sha256=decision_sha256,
                plan_sha256=plan_sha256,
                approval_sha256=approval_sha256,
                authorization_sha256=authorization_sha256,
                attempt_record_sha256=attempt_record_sha256,
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
                "acquisition_interrupted", "the approved attempt is consumed"
            ) from exc
        raise


def _load_private_record(path: Path, name: str) -> dict[str, Any]:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", f"{name} must be a regular file")
    data = load_json(path)
    return require_object(data, name)


def _verify_attempt_record(
    data: dict[str, Any],
    plan: OneVolumeAcquisitionPlan,
    *,
    decision_sha256: str,
    plan_sha256: str,
    authorization_sha256: str,
    approval_sha256: str,
) -> None:
    data = strict_fields(
        data,
        {
            "approval_sha256",
            "artifact_allowlist",
            "authorization",
            "authorization_sha256",
            "command_schema_version",
            "decision_sha256",
            "plan_sha256",
            "schema_version",
            "started_at",
        },
        "acquisition attempt",
    )
    if (
        require_string(data["schema_version"], "schema_version") != ATTEMPT_SCHEMA
        or require_string(data["command_schema_version"], "command_schema_version")
        != COMMAND_SCHEMA
        or require_string(data["decision_sha256"], "decision_sha256") != decision_sha256
        or require_string(data["plan_sha256"], "plan_sha256") != plan_sha256
        or require_string(data["approval_sha256"], "approval_sha256") != approval_sha256
        or require_string(data["authorization_sha256"], "authorization_sha256")
        != authorization_sha256
        or data["authorization"] != _authorization(PLAN_SHA256)[0]
        or require_list(data["artifact_allowlist"], "artifact_allowlist")
        != _attempt_allowlist(plan)
    ):
        raise EvidenceError("attempt_record_mismatch", "attempt record does not match")
    require_string(data["started_at"], "started_at")


def verify_trusted_one_volume_refusal(
    decision_path: Path,
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    repository_root: Path | None = None,
) -> AcquisitionRefusal:
    plan, decision_sha256, plan_sha256 = _load_trusted_plan(decision_path, plan_path)
    _, approval_sha256, authorization_sha256 = _authorization(approve_plan_sha256)
    require_private_root(root, repository_root=repository_root)
    attempt_path = ensure_no_symlink(
        root, safe_relative_path(f"attempts/{plan_sha256}.started.json")
    )
    refusal_path = ensure_no_symlink(
        root, safe_relative_path(f"attempts/{plan_sha256}.refusal.json")
    )
    completion_path = root / "attempts" / f"{plan_sha256}.completed.json"
    if path_occupied(completion_path):
        raise EvidenceError("acquisition_completion_exists", "completion event exists")
    if (
        not attempt_path.is_file()
        or not refusal_path.is_file()
        or stat.S_IMODE(attempt_path.stat().st_mode) != 0o600
        or stat.S_IMODE(refusal_path.stat().st_mode) != 0o600
        or stat.S_IMODE(attempt_path.parent.stat().st_mode) != 0o700
    ):
        raise EvidenceError("missing_acquisition_refusal", "private refusal evidence required")
    _verify_attempt_record(
        _load_private_record(attempt_path, "acquisition attempt"),
        plan,
        decision_sha256=decision_sha256,
        plan_sha256=plan_sha256,
        authorization_sha256=authorization_sha256,
        approval_sha256=approval_sha256,
    )
    refusal = AcquisitionRefusal.from_dict(
        _load_private_record(refusal_path, "acquisition refusal")
    )
    if (
        refusal.decision_sha256 != decision_sha256
        or refusal.plan_sha256 != plan_sha256
        or refusal.approval_sha256 != approval_sha256
        or refusal.authorization_sha256 != authorization_sha256
        or refusal.attempt_record_sha256 != sha256_file(attempt_path)
    ):
        raise EvidenceError("acquisition_refusal_mismatch", "trusted identities differ")
    return refusal


def _verify_private_modes(root: Path, snapshot: Path) -> None:
    for directory, subdirectories, filenames in os.walk(snapshot, followlinks=False):
        current = Path(directory)
        if is_link_like(current) or stat.S_IMODE(current.stat().st_mode) != 0o700:
            raise EvidenceError("private_directory_permissions", "snapshot directory")
        for name in (*subdirectories, *filenames):
            path = current / name
            if is_link_like(path):
                raise EvidenceError("symlink_forbidden", "snapshot entry")
        for name in filenames:
            path = current / name
            if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
                raise EvidenceError("private_file_permissions", "snapshot file")
    if stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise EvidenceError("private_root_permissions", "custody root must be owner-only")


def _verify_snapshot_file_set(snapshot: Path, plan: OneVolumeAcquisitionPlan) -> None:
    expected = {
        *(artifact.destination for artifact in plan.artifacts),
        RECEIPT_PATH,
        STRUCTURAL_REPORT_PATH,
    }
    observed = {
        (Path(directory) / filename).relative_to(snapshot).as_posix()
        for directory, _, filenames in os.walk(snapshot, followlinks=False)
        for filename in filenames
    }
    if observed != expected:
        raise EvidenceError("custody_snapshot_file_set_mismatch", "snapshot file set differs")


def verify_trusted_one_volume(
    decision_path: Path,
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    repository_root: Path | None = None,
) -> CustodyReceipt:
    plan, decision_sha256, plan_sha256 = _load_trusted_plan(decision_path, plan_path)
    _, approval_sha256, authorization_sha256 = _authorization(approve_plan_sha256)
    require_private_root(root, repository_root=repository_root)
    snapshot = ensure_no_symlink(root, safe_relative_path(SNAPSHOT_NAME))
    if not snapshot.is_dir():
        raise EvidenceError("missing_custody_snapshot", SNAPSHOT_NAME)
    _verify_private_modes(root, snapshot)
    _verify_snapshot_file_set(snapshot, plan)
    attempt_path = ensure_no_symlink(
        root, safe_relative_path(f"attempts/{plan_sha256}.started.json")
    )
    if (
        not attempt_path.is_file()
        or stat.S_IMODE(attempt_path.stat().st_mode) != 0o600
        or stat.S_IMODE(attempt_path.parent.stat().st_mode) != 0o700
    ):
        raise EvidenceError("private_file_permissions", "acquisition attempt")
    attempt_data = _load_private_record(attempt_path, "acquisition attempt")
    _verify_attempt_record(
        attempt_data,
        plan,
        decision_sha256=decision_sha256,
        plan_sha256=plan_sha256,
        authorization_sha256=authorization_sha256,
        approval_sha256=approval_sha256,
    )
    refusal_path = root / "attempts" / f"{plan_sha256}.refusal.json"
    if path_occupied(refusal_path):
        raise EvidenceError("acquisition_refused", "a refusal terminal event exists")
    completion_path = ensure_no_symlink(
        root, safe_relative_path(f"attempts/{plan_sha256}.completed.json")
    )
    if not completion_path.is_file() or stat.S_IMODE(completion_path.stat().st_mode) != 0o600:
        raise EvidenceError("missing_acquisition_completion", "completion event required")
    receipt_path = ensure_no_symlink(snapshot, safe_relative_path(RECEIPT_PATH))
    receipt = CustodyReceipt.from_dict(_load_private_record(receipt_path, "custody receipt"))
    completion = strict_fields(
        _load_private_record(completion_path, "acquisition completion"),
        {
            "attempt_record_sha256",
            "authorization_sha256",
            "completed_at",
            "custody_receipt_sha256",
            "plan_sha256",
            "schema_version",
            "snapshot",
            "status",
        },
        "acquisition completion",
    )
    if (
        completion["schema_version"] != COMPLETION_SCHEMA
        or completion["status"] != "completed"
        or completion["snapshot"] != SNAPSHOT_NAME
        or completion["plan_sha256"] != plan_sha256
        or completion["authorization_sha256"] != authorization_sha256
        or completion["attempt_record_sha256"] != sha256_file(attempt_path)
        or completion["custody_receipt_sha256"] != sha256_file(receipt_path)
        or not isinstance(completion["completed_at"], str)
    ):
        raise EvidenceError("acquisition_completion_mismatch", "completion event differs")
    if (
        receipt.decision_sha256 != decision_sha256
        or receipt.plan_sha256 != plan_sha256
        or receipt.approval_sha256 != approval_sha256
        or receipt.authorization_sha256 != authorization_sha256
        or receipt.attempt_record_sha256 != sha256_file(attempt_path)
        or receipt.artifact_order != tuple(item.artifact_id for item in plan.artifacts)
    ):
        raise EvidenceError("custody_receipt_mismatch", "trusted identities differ")
    by_id = {item.artifact_id: item for item in receipt.artifacts}
    if set(by_id) != {item.artifact_id for item in plan.artifacts}:
        raise EvidenceError("custody_receipt_mismatch", "artifact set differs")
    for artifact in plan.artifacts:
        result = by_id[artifact.artifact_id]
        path = ensure_no_symlink(snapshot, safe_relative_path(result.relative_path))
        expected_hashes = {item.algorithm: item.value for item in artifact.expected_hashes}
        if (
            not path.is_file()
            or is_link_like(path)
            or result.role != artifact.role
            or result.relative_path != artifact.destination
            or result.size_bytes != artifact.expected_size_bytes
            or result.sha256 != expected_hashes["sha256"]
            or result.source_hashes_verified
            != tuple(item.algorithm for item in artifact.expected_hashes)
            or path.stat().st_size != artifact.expected_size_bytes
            or sha256_file(path) != expected_hashes["sha256"]
        ):
            raise EvidenceError("custody_artifact_mismatch", artifact.artifact_id)
        git_blob = expected_hashes.get("git-blob-sha1")
        if git_blob is not None:
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {artifact.expected_size_bytes}\0".encode("ascii"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(_READ_CHUNK_BYTES), b""):
                    digest.update(chunk)
            if digest.hexdigest() != git_blob:
                raise EvidenceError("custody_artifact_mismatch", artifact.artifact_id)
    report_path = ensure_no_symlink(snapshot, safe_relative_path(receipt.structural_report_path))
    if sha256_file(report_path) != receipt.structural_report_sha256:
        raise EvidenceError("structural_report_hash_mismatch", "private report changed")
    report = StructuralReport.from_dict(_load_private_record(report_path, "structural report"))
    if (
        report.decision_sha256 != decision_sha256
        or report.plan_sha256 != plan_sha256
        or report.authorization_sha256 != authorization_sha256
    ):
        raise EvidenceError("structural_report_mismatch", "trusted identities differ")
    inspection = Path(tempfile.mkdtemp(prefix=".one-volume-verify-", dir=root))
    os.chmod(inspection, 0o700)
    try:
        observed = _build_structural_report(
            snapshot,
            plan,
            decision_sha256=decision_sha256,
            plan_sha256=plan_sha256,
            authorization_sha256=authorization_sha256,
            inspection_root=inspection,
        )
    finally:
        shutil.rmtree(inspection)
    if canonical_json_bytes(observed.to_dict()) != canonical_json_bytes(report.to_dict()):
        raise EvidenceError("structural_report_mismatch", "observed structure differs")
    return receipt
