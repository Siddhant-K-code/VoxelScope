# SPDX-License-Identifier: Apache-2.0
"""One-shot private preprocessing for the trusted milestone 4 snapshot."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import nibabel as nib
import numpy as np

from .atomic import path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    is_link_like,
    load_json,
    require_sha256,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
)
from .custody import require_private_root
from .one_volume_contract import PLAN_SHA256 as CUSTODY_PLAN_SHA256
from .one_volume_custody import (
    RECEIPT_PATH as CUSTODY_RECEIPT_PATH,
)
from .one_volume_custody import (
    SNAPSHOT_NAME as CUSTODY_SNAPSHOT_NAME,
)
from .one_volume_custody import (
    _fsync_directory,
    _load_trusted_plan,
    _mkdir_private_chain,
    _write_private_json_no_clobber,
    verify_trusted_one_volume,
)
from .one_volume_custody_records import NiftiStructure, StructuralReport
from .one_volume_records import OneVolumeAcquisitionPlan, PlannedArtifact
from .preprocessing_records import (
    CHANNEL_ORDER,
    COMPARISON_ATOL,
    COMPARISON_RTOL,
    AdapterPlan,
    ChannelStatistics,
    ImplementationComparison,
    PreprocessingRefusal,
    PreprocessingReport,
    RefusalStage,
    RuntimeIdentity,
    TensorArtifact,
)
from .records import require_object, require_string, strict_fields

PREPROCESSING_SNAPSHOT_NAME = "preprocessing-v1"
PREPROCESSING_REPORT_PATH = "evidence/preprocessing-report.json"
REFERENCE_TENSOR_PATH = "tensors/input-reference.f32le"
INDEPENDENT_TENSOR_PATH = "tensors/input-independent.f32le"
ATTEMPT_SCHEMA = "voxelscope/preprocessing-attempt/v1"
AUTHORIZATION_SCHEMA = "voxelscope/preprocessing-authorization/v1"
COMPLETION_SCHEMA = "voxelscope/preprocessing-completion/v1"
_COMPARE_CHUNK_VALUES = 1024 * 1024


class _ArrayProxy(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    slope: float
    inter: float

    def __getitem__(self, key: object) -> Any: ...


@dataclass
class _ExecutionProgress:
    last_completed_stage: RefusalStage = "attempt_record"
    completed_channel_count: int = 0
    snapshot_published: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_adapter_plan(
    path: Path,
    *,
    repository_root: Path,
    verify_runtime: bool = True,
) -> tuple[AdapterPlan, str]:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "adapter plan must be a regular file")
    data = load_json(path)
    plan = AdapterPlan.from_dict(require_object(data, "adapter plan"))
    resolved_repository = repository_root.resolve(strict=True)
    if verify_runtime and _runtime_identity(resolved_repository) != plan.runtime_identity:
        raise EvidenceError("preprocessing_runtime_mismatch", "runtime identity differs")
    for identity in plan.implementation_sources:
        source = ensure_no_symlink(resolved_repository, safe_relative_path(identity.path))
        if not source.is_file() or sha256_file(source) != identity.sha256:
            raise EvidenceError("implementation_identity_mismatch", identity.path)
    return plan, sha256_file(path)


def _runtime_identity(repository_root: Path) -> RuntimeIdentity:
    lock = ensure_no_symlink(repository_root, safe_relative_path("uv.lock"))
    if not lock.is_file():
        raise EvidenceError("missing_runtime_lock", "uv.lock is required")
    return RuntimeIdentity(
        cast(Any, platform.python_version()),
        cast(Any, version("numpy")),
        cast(Any, version("nibabel")),
        cast(Any, sys.byteorder),
        sha256_file(lock),
    )


def _custody_receipt_path(root: Path) -> Path:
    snapshot = ensure_no_symlink(root, safe_relative_path(CUSTODY_SNAPSHOT_NAME))
    return ensure_no_symlink(snapshot, safe_relative_path(CUSTODY_RECEIPT_PATH))


def _authorization(
    *,
    plan_sha256: str,
    custody_receipt_sha256: str,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    runtime_identity: RuntimeIdentity,
) -> tuple[dict[str, Any], str, str]:
    if approve_plan_sha256 != plan_sha256:
        raise EvidenceError(
            "wrong_adapter_plan_approval",
            "approval does not match the adapter plan",
        )
    if approve_custody_receipt_sha256 != custody_receipt_sha256:
        raise EvidenceError(
            "wrong_custody_identity_approval",
            "approval does not match the custody receipt",
        )
    authorization = {
        "approved_adapter_plan_sha256": approve_plan_sha256,
        "approved_custody_receipt_sha256": approve_custody_receipt_sha256,
        "custody_plan_sha256": CUSTODY_PLAN_SHA256,
        "inference_authorized": False,
        "model_loading_allowed": False,
        "network_allowed": False,
        "runtime_identity": runtime_identity.to_dict(),
        "runtime_identity_sha256": sha256_bytes(canonical_json_bytes(runtime_identity.to_dict())),
        "schema_version": AUTHORIZATION_SCHEMA,
    }
    authorization_sha256 = sha256_bytes(canonical_json_bytes(authorization))
    attempt_id = sha256_bytes(
        canonical_json_bytes(
            {
                "adapter_plan_sha256": plan_sha256,
                "custody_receipt_sha256": custody_receipt_sha256,
            }
        )
    )
    return authorization, authorization_sha256, attempt_id


def _attempt_paths(root: Path, attempt_id: str) -> tuple[Path, Path]:
    require_sha256(attempt_id, "attempt_id")
    directory = ensure_no_symlink(root, safe_relative_path("preprocessing-attempts"))
    return (
        directory / f"{attempt_id}.started.json",
        directory / f"{attempt_id}.terminal.json",
    )


def _load_structural_report(root: Path, receipt_path: str) -> StructuralReport:
    snapshot = ensure_no_symlink(root, safe_relative_path(CUSTODY_SNAPSHOT_NAME))
    report_path = ensure_no_symlink(snapshot, safe_relative_path(receipt_path))
    if is_link_like(report_path) or not report_path.is_file():
        raise EvidenceError("unsafe_path", "structural report must be a regular file")
    return StructuralReport.from_dict(require_object(load_json(report_path), "structural report"))


def _geometry_identity(report: StructuralReport) -> str:
    image_structures = [item for item in report.artifacts if item.role in CHANNEL_ORDER]
    value = {
        "affine": image_structures[0].affine,
        "orientation": image_structures[0].orientation,
        "qform": image_structures[0].qform,
        "qform_code": image_structures[0].qform_code,
        "shape": image_structures[0].shape,
        "sform": image_structures[0].sform,
        "sform_code": image_structures[0].sform_code,
        "spacing": image_structures[0].spacing,
        "spatial_unit": image_structures[0].spatial_unit,
    }
    return sha256_bytes(canonical_json_bytes(value))


def _open_channel(
    path: Path,
    role: str,
    structure: NiftiStructure,
) -> tuple[nib.spatialimages.SpatialImage, _ArrayProxy]:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "input channel must be a regular file")
    try:
        image = nib.load(str(path), mmap="r", keep_file_open=False)
    except Exception as exc:
        raise EvidenceError("nifti_voxel_read_failed", role) from exc
    if not isinstance(image, nib.spatialimages.SpatialImage):
        raise EvidenceError("invalid_nifti", role)
    proxy = cast(_ArrayProxy, image.dataobj)
    observed_affine = np.asarray(image.affine, dtype=np.float64)
    expected_affine = np.asarray(structure.affine, dtype=np.float64)
    if (
        tuple(int(item) for item in proxy.shape) != structure.shape
        or np.dtype(proxy.dtype).str != structure.dtype
        or not np.array_equal(observed_affine, expected_affine)
        or float(proxy.slope) != structure.scaling_slope
        or float(proxy.inter) != structure.scaling_intercept
    ):
        raise EvidenceError("source_geometry_drift", role)
    return image, proxy


def normalize_reference(
    values: Any,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int, float, float]:
    """Normalize one channel with direct float32 NumPy reductions."""
    try:
        with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
            warnings.simplefilter("error")
            effective = np.array(
                np.asanyarray(values, order="C"),
                dtype=np.dtype("<f4"),
                order="C",
                copy=True,
            )
    except (ArithmeticError, RuntimeWarning, ValueError) as exc:
        raise EvidenceError("effective_dtype_overflow", "channel conversion failed") from exc
    if not bool(np.isfinite(effective).all()):
        raise EvidenceError("nonfinite_input", "channel contains nonfinite values")
    nonzero = effective != np.float32(0.0)
    count = int(np.count_nonzero(nonzero))
    if count == 0:
        raise EvidenceError("empty_channel", "channel has no nonzero voxels")
    selected = effective[nonzero]
    try:
        with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
            warnings.simplefilter("error")
            mean = np.mean(selected, dtype=np.dtype("<f4"))
            standard_deviation = np.std(selected, dtype=np.dtype("<f4"))
            if not np.isfinite(mean) or not np.isfinite(standard_deviation):
                raise EvidenceError("nonfinite_statistics", "channel statistics are nonfinite")
            if standard_deviation == np.float32(0.0):
                raise EvidenceError("zero_variance_channel", "channel variance is zero")
            normalized = np.zeros(effective.shape, dtype=np.dtype("<f4"), order="C")
            normalized[nonzero] = (selected - mean) / standard_deviation
    except EvidenceError:
        raise
    except (ArithmeticError, RuntimeWarning) as exc:
        raise EvidenceError("normalization_overflow", "channel normalization failed") from exc
    if not bool(np.isfinite(normalized).all()):
        raise EvidenceError("nonfinite_output", "normalized channel is nonfinite")
    if not bool(np.equal(normalized[~nonzero], np.float32(0.0)).all()):
        raise EvidenceError("background_changed", "zero background changed")
    return normalized, count, float(mean), float(standard_deviation)


def normalize_chunked_welford(
    values: Any,
    output: np.ndarray[Any, np.dtype[np.float32]],
    *,
    chunk_depth: int,
) -> tuple[int, float, float]:
    """Normalize one channel with chunked Welford combination."""
    shape = tuple(int(item) for item in values.shape)
    if len(shape) != 3 or output.shape != shape or chunk_depth <= 0:
        raise EvidenceError("invalid_channel_shape", "chunked implementation shape")
    count = 0
    mean = 0.0
    second_moment = 0.0
    try:
        for start in range(0, shape[2], chunk_depth):
            stop = min(start + chunk_depth, shape[2])
            with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
                warnings.simplefilter("error")
                chunk = np.array(
                    np.asanyarray(values[:, :, start:stop], order="C"),
                    dtype=np.dtype("<f4"),
                    order="C",
                    copy=True,
                )
            if not bool(np.isfinite(chunk).all()):
                raise EvidenceError("nonfinite_input", "channel contains nonfinite values")
            selected = chunk[chunk != np.float32(0.0)].astype(np.float64, copy=False)
            chunk_count = int(selected.size)
            if chunk_count == 0:
                continue
            chunk_mean = float(np.mean(selected, dtype=np.float64))
            centered = selected - chunk_mean
            chunk_second_moment = float(np.dot(centered, centered))
            if count == 0:
                count = chunk_count
                mean = chunk_mean
                second_moment = chunk_second_moment
            else:
                combined = count + chunk_count
                delta = chunk_mean - mean
                second_moment += chunk_second_moment + delta * delta * float(count) * float(
                    chunk_count
                ) / float(combined)
                mean += delta * float(chunk_count) / float(combined)
                count = combined
    except EvidenceError:
        raise
    except (ArithmeticError, RuntimeWarning, ValueError) as exc:
        raise EvidenceError("effective_dtype_overflow", "chunk conversion failed") from exc
    if count == 0:
        raise EvidenceError("empty_channel", "channel has no nonzero voxels")
    variance = second_moment / float(count)
    if not math_is_finite_nonnegative(variance):
        raise EvidenceError("nonfinite_statistics", "channel statistics are nonfinite")
    mean32 = np.float32(mean)
    standard_deviation32 = np.float32(np.sqrt(variance))
    if not np.isfinite(mean32) or not np.isfinite(standard_deviation32):
        raise EvidenceError("nonfinite_statistics", "channel statistics are nonfinite")
    if standard_deviation32 == np.float32(0.0):
        raise EvidenceError("zero_variance_channel", "channel variance is zero")
    try:
        for start in range(0, shape[2], chunk_depth):
            stop = min(start + chunk_depth, shape[2])
            with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
                warnings.simplefilter("error")
                chunk = np.array(
                    np.asanyarray(values[:, :, start:stop], order="C"),
                    dtype=np.dtype("<f4"),
                    order="C",
                    copy=True,
                )
                normalized = np.zeros(chunk.shape, dtype=np.dtype("<f4"), order="C")
                nonzero = chunk != np.float32(0.0)
                normalized[nonzero] = (chunk[nonzero] - mean32) / standard_deviation32
                output[:, :, start:stop] = normalized
    except (ArithmeticError, RuntimeWarning, ValueError) as exc:
        raise EvidenceError("normalization_overflow", "chunk normalization failed") from exc
    if not bool(np.isfinite(output).all()):
        raise EvidenceError("nonfinite_output", "normalized channel is nonfinite")
    return count, float(mean32), float(standard_deviation32)


def math_is_finite_nonnegative(value: float) -> bool:
    return bool(np.isfinite(value)) and value >= 0.0


def compare_implementations(
    reference: np.ndarray[Any, np.dtype[np.float32]],
    independent: np.ndarray[Any, np.dtype[np.float32]],
    *,
    atol: float = COMPARISON_ATOL,
    rtol: float = COMPARISON_RTOL,
) -> ImplementationComparison:
    if reference.shape != independent.shape or reference.dtype != np.dtype("<f4"):
        raise EvidenceError("implementation_shape_mismatch", "tensor shapes or dtypes differ")
    reference_flat = reference.reshape(-1)
    independent_flat = independent.reshape(-1)
    exact = True
    within = True
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for start in range(0, reference_flat.size, _COMPARE_CHUNK_VALUES):
        stop = min(start + _COMPARE_CHUNK_VALUES, reference_flat.size)
        left = np.asarray(reference_flat[start:stop], dtype=np.dtype("<f4"))
        right = np.asarray(independent_flat[start:stop], dtype=np.dtype("<f4"))
        if not bool(np.isfinite(left).all()) or not bool(np.isfinite(right).all()):
            raise EvidenceError("nonfinite_output", "implementation output is nonfinite")
        exact = exact and bool(np.array_equal(left, right))
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        if difference.size:
            maximum_absolute = max(maximum_absolute, float(np.max(difference)))
            denominator = np.maximum(np.abs(left.astype(np.float64)), atol)
            maximum_relative = max(
                maximum_relative,
                float(np.max(difference / denominator)),
            )
        within = within and bool(np.allclose(left, right, atol=atol, rtol=rtol))
    if not within:
        raise EvidenceError(
            "implementation_disagreement",
            "independent implementations exceed the prospective tolerance",
        )
    return ImplementationComparison(
        exact,
        True,
        atol,
        rtol,
        maximum_absolute,
        maximum_relative,
    )


def _create_tensor(path: Path, shape: tuple[int, ...]) -> np.memmap[Any, np.dtype[np.float32]]:
    if path_occupied(path):
        raise EvidenceError("output_exists", path.name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)
    size = int(np.prod(shape, dtype=np.int64)) * np.dtype("<f4").itemsize
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, size)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return np.memmap(path, dtype="<f4", mode="r+", shape=shape, order="C")


def _copy_verified_input(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if is_link_like(source) or not source.is_file() or path_occupied(destination):
        raise EvidenceError("unsafe_path", "verified channel copy path")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(destination.parent, 0o700)
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, source_flags)
    destination_descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise EvidenceError("custody_artifact_mismatch", "input channel extent")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        while chunk := os.read(source_descriptor, 1024 * 1024):
            size += len(chunk)
            if size > expected_size:
                raise EvidenceError("custody_artifact_mismatch", "input channel extent")
            digest.update(chunk)
            pending = memoryview(chunk)
            while pending:
                written = os.write(destination_descriptor, pending)
                pending = pending[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        if (
            size != expected_size
            or digest.hexdigest() != expected_sha256
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise EvidenceError("custody_artifact_mismatch", "input channel changed")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _write_terminal(root: Path, attempt_id: str, value: dict[str, Any]) -> None:
    relative = f"preprocessing-attempts/{attempt_id}.terminal.json"
    try:
        _write_private_json_no_clobber(root, relative, value)
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
    custody_receipt_sha256: str,
    authorization_sha256: str,
    attempt_record_sha256: str,
    code: str,
    progress: _ExecutionProgress,
) -> None:
    refusal = PreprocessingRefusal(
        "voxelscope/preprocessing-refusal/v1",
        "refused",
        code,
        progress.last_completed_stage,
        plan_sha256,
        custody_receipt_sha256,
        authorization_sha256,
        attempt_record_sha256,
        progress.completed_channel_count,
        progress.snapshot_published,
        False,
        False,
        False,
        _now(),
    )
    _write_terminal(root, attempt_id, refusal.to_dict())


def _source_maps(
    plan: OneVolumeAcquisitionPlan,
    report: StructuralReport,
) -> tuple[dict[str, PlannedArtifact], dict[str, NiftiStructure]]:
    artifacts: dict[str, PlannedArtifact] = {str(item.role): item for item in plan.artifacts}
    structures = {item.role: item for item in report.artifacts}
    if set(CHANNEL_ORDER) - artifacts.keys() or set(CHANNEL_ORDER) - structures.keys():
        raise EvidenceError("incomplete_channel_set", "trusted role set is incomplete")
    return artifacts, structures


def execute_preprocessing(
    decision_path: Path,
    custody_plan_path: Path,
    adapter_plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    repository_root: Path,
) -> PreprocessingReport:
    adapter_plan, plan_sha256 = load_adapter_plan(
        adapter_plan_path,
        repository_root=repository_root,
    )
    require_private_root(root, repository_root=repository_root)
    receipt_path = _custody_receipt_path(root)
    if (
        is_link_like(receipt_path)
        or not receipt_path.is_file()
        or (os.name != "nt" and stat.S_IMODE(receipt_path.stat().st_mode) != 0o600)
    ):
        raise EvidenceError("missing_custody_receipt", "trusted receipt is required")
    custody_receipt_sha256 = sha256_file(receipt_path)
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        custody_receipt_sha256=custody_receipt_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        runtime_identity=adapter_plan.runtime_identity,
    )
    attempts = _mkdir_private_chain(root, PurePosixPath("preprocessing-attempts"))
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    if path_occupied(started_path) or path_occupied(terminal_path):
        raise EvidenceError(
            "preprocessing_attempt_exists",
            "the approved preprocessing attempt is consumed",
        )
    snapshot = root / PREPROCESSING_SNAPSHOT_NAME
    if path_occupied(snapshot):
        raise EvidenceError("output_exists", PREPROCESSING_SNAPSHOT_NAME)
    attempt = {
        "adapter_plan_sha256": plan_sha256,
        "authorization": authorization,
        "authorization_sha256": authorization_sha256,
        "custody_receipt_sha256": custody_receipt_sha256,
        "declared_private_reads": [
            "trusted milestone 4 receipt, terminal records, structural report, and eight artifacts",
            "four image roles in T1c, T1, T2, FLAIR order",
        ],
        "declared_private_writes": [
            "four transient hash-verified image copies removed before publication",
            "one preprocessing snapshot with two tensors and one report",
            "one terminal completion or refusal record",
        ],
        "schema_version": ATTEMPT_SCHEMA,
        "started_at": _now(),
    }
    attempt_record_sha256 = sha256_bytes(canonical_json_bytes(attempt))
    progress = _ExecutionProgress()
    stage: Path | None = None
    marker_owned = False
    try:
        try:
            _write_private_json_no_clobber(
                root,
                f"preprocessing-attempts/{attempt_id}.started.json",
                attempt,
            )
            marker_owned = True
        except EvidenceError as exc:
            if exc.code == "output_exists":
                raise EvidenceError(
                    "preprocessing_attempt_exists",
                    "the approved preprocessing attempt is consumed",
                ) from exc
            raise
        except BaseException:
            marker_owned = (
                started_path.is_file()
                and not is_link_like(started_path)
                and sha256_file(started_path) == attempt_record_sha256
            )
            raise
        if started_path.parent != attempts or sha256_file(started_path) != attempt_record_sha256:
            raise EvidenceError("preprocessing_attempt_mismatch", "attempt marker differs")
        custody_receipt = verify_trusted_one_volume(
            decision_path,
            custody_plan_path,
            root,
            approve_plan_sha256=CUSTODY_PLAN_SHA256,
            repository_root=repository_root,
        )
        progress.last_completed_stage = "custody_verification"
        if sha256_file(receipt_path) != custody_receipt_sha256:
            raise EvidenceError("custody_identity_drift", "custody receipt changed")
        custody_plan, decision_sha256, observed_custody_plan_sha256 = _load_trusted_plan(
            decision_path,
            custody_plan_path,
        )
        if observed_custody_plan_sha256 != CUSTODY_PLAN_SHA256:
            raise EvidenceError("custody_plan_mismatch", "trusted plan changed")
        structural_report = _load_structural_report(
            root,
            custody_receipt.structural_report_path,
        )
        artifacts, structures = _source_maps(custody_plan, structural_report)
        input_shape = structures[CHANNEL_ORDER[0]].shape
        output_shape = (len(CHANNEL_ORDER), *input_shape)
        stage = Path(tempfile.mkdtemp(prefix=".preprocessing-stage-", dir=root))
        os.chmod(stage, 0o700)
        reference_path = stage / REFERENCE_TENSOR_PATH
        independent_path = stage / INDEPENDENT_TENSOR_PATH
        reference_tensor = _create_tensor(reference_path, output_shape)
        independent_tensor = _create_tensor(independent_path, output_shape)
        reference_statistics: list[ChannelStatistics] = []
        independent_statistics: list[ChannelStatistics] = []
        receipt_by_role = {item.role: item for item in custody_receipt.artifacts}
        custody_snapshot = ensure_no_symlink(root, safe_relative_path(CUSTODY_SNAPSHOT_NAME))
        for channel_index, role in enumerate(CHANNEL_ORDER):
            progress.last_completed_stage = "channel_loading"
            source_path = ensure_no_symlink(
                custody_snapshot,
                safe_relative_path(artifacts[role].destination),
            )
            staged_input = stage / ".verified-inputs" / f"channel-{channel_index}.nii.gz"
            _copy_verified_input(
                source_path,
                staged_input,
                expected_sha256=receipt_by_role[role].sha256,
                expected_size=receipt_by_role[role].size_bytes,
            )
            image, proxy = _open_channel(
                staged_input,
                role,
                structures[role],
            )
            raw_dtype = np.dtype(proxy.dtype).str
            progress.last_completed_stage = "reference_normalization"
            normalized, count, mean, standard_deviation = normalize_reference(proxy)
            if normalized.shape != input_shape:
                raise EvidenceError("source_geometry_drift", role)
            reference_tensor[channel_index] = normalized
            reference_statistics.append(
                ChannelStatistics(
                    role,
                    receipt_by_role[role].sha256,
                    raw_dtype,
                    "<f4",
                    count,
                    mean,
                    standard_deviation,
                )
            )
            del normalized
            progress.last_completed_stage = "independent_normalization"
            independent_count, independent_mean, independent_standard_deviation = (
                normalize_chunked_welford(
                    proxy,
                    independent_tensor[channel_index],
                    chunk_depth=adapter_plan.chunk_depth,
                )
            )
            independent_statistics.append(
                ChannelStatistics(
                    role,
                    receipt_by_role[role].sha256,
                    raw_dtype,
                    "<f4",
                    independent_count,
                    independent_mean,
                    independent_standard_deviation,
                )
            )
            image.uncache()
            progress.completed_channel_count += 1
        shutil.rmtree(stage / ".verified-inputs")
        _fsync_directory(stage)
        reference_tensor.flush()
        independent_tensor.flush()
        _fsync_directory(reference_path.parent)
        progress.last_completed_stage = "implementation_comparison"
        comparison = compare_implementations(
            reference_tensor,
            independent_tensor,
            atol=adapter_plan.comparison_atol,
            rtol=adapter_plan.comparison_rtol,
        )
        reference_artifact = TensorArtifact(
            REFERENCE_TENSOR_PATH,
            sha256_file(reference_path),
            reference_path.stat().st_size,
        )
        independent_artifact = TensorArtifact(
            INDEPENDENT_TENSOR_PATH,
            sha256_file(independent_path),
            independent_path.stat().st_size,
        )
        del reference_tensor
        del independent_tensor
        report = PreprocessingReport(
            "voxelscope/preprocessing-report/v1",
            "go",
            plan_sha256,
            decision_sha256,
            observed_custody_plan_sha256,
            custody_receipt_sha256,
            custody_receipt.structural_report_sha256,
            authorization_sha256,
            attempt_record_sha256,
            CHANNEL_ORDER,
            True,
            _geometry_identity(structural_report),
            True,
            input_shape,
            output_shape,
            "C,I,J,K",
            "<f4",
            "C",
            tuple(reference_statistics),
            tuple(independent_statistics),
            reference_artifact,
            independent_artifact,
            comparison,
            True,
            True,
            True,
            adapter_plan.runtime_identity,
            adapter_plan.implementation_sources,
            adapter_plan.monai_sources,
            False,
            False,
            False,
            _now(),
        )
        report_path = _write_private_json_no_clobber(
            stage,
            PREPROCESSING_REPORT_PATH,
            report.to_dict(),
        )
        report_sha256 = sha256_file(report_path)
        _fsync_directory(stage / "evidence")
        _fsync_directory(stage)
        progress.last_completed_stage = "snapshot_publication"
        rename_no_replace(stage, snapshot)
        progress.snapshot_published = True
        _fsync_directory(root)
        _write_terminal(
            root,
            attempt_id,
            {
                "adapter_plan_sha256": plan_sha256,
                "attempt_record_sha256": attempt_record_sha256,
                "authorization_sha256": authorization_sha256,
                "completed_at": _now(),
                "custody_receipt_sha256": custody_receipt_sha256,
                "preprocessing_report_sha256": report_sha256,
                "schema_version": COMPLETION_SCHEMA,
                "snapshot": PREPROCESSING_SNAPSHOT_NAME,
                "status": "completed",
            },
        )
        return report
    except BaseException as exc:
        code = exc.code if isinstance(exc, EvidenceError) else "preprocessing_interrupted"
        if marker_owned:
            try:
                if path_occupied(terminal_path):
                    terminal = _load_private_object(terminal_path, "preprocessing terminal")
                    if terminal.get("status") != "completed":
                        PreprocessingRefusal.from_dict(terminal)
                else:
                    _record_refusal(
                        root,
                        attempt_id=attempt_id,
                        plan_sha256=plan_sha256,
                        custody_receipt_sha256=custody_receipt_sha256,
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
                "preprocessing_interrupted",
                "the approved preprocessing attempt is consumed",
            ) from exc
        raise


def _load_private_object(path: Path, name: str) -> dict[str, Any]:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", f"{name} must be a regular file")
    return require_object(load_json(path), name)


def _verify_attempt(
    value: dict[str, Any],
    *,
    plan_sha256: str,
    custody_receipt_sha256: str,
    authorization: dict[str, Any],
    authorization_sha256: str,
) -> None:
    value = strict_fields(
        value,
        {
            "adapter_plan_sha256",
            "authorization",
            "authorization_sha256",
            "custody_receipt_sha256",
            "declared_private_reads",
            "declared_private_writes",
            "schema_version",
            "started_at",
        },
        "preprocessing attempt",
    )
    if (
        value["schema_version"] != ATTEMPT_SCHEMA
        or value["adapter_plan_sha256"] != plan_sha256
        or value["custody_receipt_sha256"] != custody_receipt_sha256
        or value["authorization"] != authorization
        or value["authorization_sha256"] != authorization_sha256
        or value["declared_private_reads"]
        != [
            "trusted milestone 4 receipt, terminal records, structural report, and eight artifacts",
            "four image roles in T1c, T1, T2, FLAIR order",
        ]
        or value["declared_private_writes"]
        != [
            "four transient hash-verified image copies removed before publication",
            "one preprocessing snapshot with two tensors and one report",
            "one terminal completion or refusal record",
        ]
    ):
        raise EvidenceError("preprocessing_attempt_mismatch", "attempt record differs")
    require_string(value["started_at"], "started_at")


def _verify_snapshot_modes(snapshot: Path) -> None:
    for directory, subdirectories, filenames in os.walk(snapshot, followlinks=False):
        current = Path(directory)
        if is_link_like(current) or (
            os.name != "nt" and stat.S_IMODE(current.stat().st_mode) != 0o700
        ):
            raise EvidenceError("private_directory_permissions", "preprocessing snapshot")
        for name in (*subdirectories, *filenames):
            if is_link_like(current / name):
                raise EvidenceError("symlink_forbidden", "preprocessing snapshot entry")
        for name in filenames:
            path = current / name
            if not path.is_file() or (
                os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600
            ):
                raise EvidenceError("private_file_permissions", "preprocessing snapshot")


def _verify_snapshot_file_set(snapshot: Path) -> None:
    expected = {
        PREPROCESSING_REPORT_PATH,
        REFERENCE_TENSOR_PATH,
        INDEPENDENT_TENSOR_PATH,
    }
    observed = {
        (Path(directory) / filename).relative_to(snapshot).as_posix()
        for directory, _, filenames in os.walk(snapshot, followlinks=False)
        for filename in filenames
    }
    if observed != expected:
        raise EvidenceError(
            "preprocessing_snapshot_file_set_mismatch",
            "preprocessing snapshot file set differs",
        )


def verify_preprocessing(
    decision_path: Path,
    custody_plan_path: Path,
    adapter_plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    repository_root: Path,
) -> PreprocessingReport:
    adapter_plan, plan_sha256 = load_adapter_plan(
        adapter_plan_path,
        repository_root=repository_root,
    )
    require_private_root(root, repository_root=repository_root)
    receipt_path = _custody_receipt_path(root)
    custody_receipt_sha256 = sha256_file(receipt_path)
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        custody_receipt_sha256=custody_receipt_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        runtime_identity=adapter_plan.runtime_identity,
    )
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    if not started_path.is_file() or not terminal_path.is_file():
        raise EvidenceError("missing_preprocessing_completion", "terminal completion required")
    _verify_attempt(
        _load_private_object(started_path, "preprocessing attempt"),
        plan_sha256=plan_sha256,
        custody_receipt_sha256=custody_receipt_sha256,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
    )
    attempt_record_sha256 = sha256_file(started_path)
    custody_receipt = verify_trusted_one_volume(
        decision_path,
        custody_plan_path,
        root,
        approve_plan_sha256=CUSTODY_PLAN_SHA256,
        repository_root=repository_root,
    )
    custody_plan, decision_sha256, observed_custody_plan_sha256 = _load_trusted_plan(
        decision_path,
        custody_plan_path,
    )
    structural_report = _load_structural_report(
        root,
        custody_receipt.structural_report_path,
    )
    snapshot = ensure_no_symlink(root, safe_relative_path(PREPROCESSING_SNAPSHOT_NAME))
    if not snapshot.is_dir():
        raise EvidenceError("missing_preprocessing_snapshot", PREPROCESSING_SNAPSHOT_NAME)
    _verify_snapshot_modes(snapshot)
    _verify_snapshot_file_set(snapshot)
    report_path = ensure_no_symlink(snapshot, safe_relative_path(PREPROCESSING_REPORT_PATH))
    report = PreprocessingReport.from_dict(
        _load_private_object(report_path, "preprocessing report")
    )
    terminal = _load_private_object(terminal_path, "preprocessing terminal")
    if terminal.get("status") == "refused":
        PreprocessingRefusal.from_dict(terminal)
        raise EvidenceError("preprocessing_refused", "a refusal terminal event exists")
    completion = strict_fields(
        terminal,
        {
            "adapter_plan_sha256",
            "attempt_record_sha256",
            "authorization_sha256",
            "completed_at",
            "custody_receipt_sha256",
            "preprocessing_report_sha256",
            "schema_version",
            "snapshot",
            "status",
        },
        "preprocessing completion",
    )
    if (
        completion["schema_version"] != COMPLETION_SCHEMA
        or completion["status"] != "completed"
        or completion["snapshot"] != PREPROCESSING_SNAPSHOT_NAME
        or completion["adapter_plan_sha256"] != plan_sha256
        or completion["custody_receipt_sha256"] != custody_receipt_sha256
        or completion["authorization_sha256"] != authorization_sha256
        or completion["attempt_record_sha256"] != attempt_record_sha256
        or completion["preprocessing_report_sha256"] != sha256_file(report_path)
    ):
        raise EvidenceError("preprocessing_completion_mismatch", "completion differs")
    require_string(completion["completed_at"], "completed_at")
    if (
        report.plan_sha256 != plan_sha256
        or report.decision_sha256 != decision_sha256
        or report.custody_plan_sha256 != observed_custody_plan_sha256
        or report.custody_receipt_sha256 != custody_receipt_sha256
        or report.custody_structural_report_sha256 != custody_receipt.structural_report_sha256
        or report.authorization_sha256 != authorization_sha256
        or report.attempt_record_sha256 != attempt_record_sha256
        or report.geometry_identity_sha256 != _geometry_identity(structural_report)
        or report.runtime_identity != adapter_plan.runtime_identity
        or report.implementation_sources != adapter_plan.implementation_sources
        or report.monai_sources != adapter_plan.monai_sources
    ):
        raise EvidenceError("preprocessing_report_mismatch", "trusted identities differ")
    expected_size = int(np.prod(report.output_shape, dtype=np.int64)) * 4
    artifacts = (report.reference_tensor, report.independent_tensor)
    tensor_paths: list[Path] = []
    for artifact in artifacts:
        path = ensure_no_symlink(snapshot, safe_relative_path(artifact.path))
        if (
            not path.is_file()
            or artifact.size_bytes != expected_size
            or path.stat().st_size != expected_size
            or sha256_file(path) != artifact.sha256
        ):
            raise EvidenceError("preprocessing_tensor_mismatch", artifact.path)
        tensor_paths.append(path)
    reference = np.memmap(
        tensor_paths[0],
        dtype="<f4",
        mode="r",
        shape=report.output_shape,
        order="C",
    )
    independent = np.memmap(
        tensor_paths[1],
        dtype="<f4",
        mode="r",
        shape=report.output_shape,
        order="C",
    )
    comparison = compare_implementations(
        reference,
        independent,
        atol=adapter_plan.comparison_atol,
        rtol=adapter_plan.comparison_rtol,
    )
    if comparison != report.implementation_comparison:
        raise EvidenceError("implementation_comparison_mismatch", "comparison changed")
    acquisition_artifacts, structures = _source_maps(custody_plan, structural_report)
    custody_snapshot = ensure_no_symlink(root, safe_relative_path(CUSTODY_SNAPSHOT_NAME))
    reference_stats = {item.role: item for item in report.reference_statistics}
    independent_stats = {item.role: item for item in report.independent_statistics}
    for channel_index, role in enumerate(CHANNEL_ORDER):
        source_path = ensure_no_symlink(
            custody_snapshot,
            safe_relative_path(acquisition_artifacts[role].destination),
        )
        image, proxy = _open_channel(
            source_path,
            role,
            structures[role],
        )
        if np.dtype(proxy.dtype).str != reference_stats[role].raw_dtype:
            raise EvidenceError("input_dtype_mismatch", role)
        observed_count = 0
        shape = structures[role].shape
        for start in range(0, shape[2], adapter_plan.chunk_depth):
            stop = min(start + adapter_plan.chunk_depth, shape[2])
            with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
                warnings.simplefilter("error")
                source = np.array(
                    np.asanyarray(proxy[:, :, start:stop], order="C"),
                    dtype=np.dtype("<f4"),
                    order="C",
                    copy=True,
                )
            if not bool(np.isfinite(source).all()):
                raise EvidenceError("nonfinite_input", role)
            nonzero = source != np.float32(0.0)
            observed_count += int(np.count_nonzero(nonzero))
            left = np.asarray(reference[channel_index, :, :, start:stop])
            right = np.asarray(independent[channel_index, :, :, start:stop])
            if not bool(np.equal(left[~nonzero], np.float32(0.0)).all()) or not bool(
                np.equal(right[~nonzero], np.float32(0.0)).all()
            ):
                raise EvidenceError("background_changed", role)
            expected_reference = np.zeros(source.shape, dtype=np.dtype("<f4"))
            expected_independent = np.zeros(source.shape, dtype=np.dtype("<f4"))
            expected_reference[nonzero] = (
                source[nonzero] - np.float32(reference_stats[role].mean)
            ) / np.float32(reference_stats[role].standard_deviation)
            expected_independent[nonzero] = (
                source[nonzero] - np.float32(independent_stats[role].mean)
            ) / np.float32(independent_stats[role].standard_deviation)
            if not bool(np.array_equal(left, expected_reference)) or not bool(
                np.array_equal(right, expected_independent)
            ):
                raise EvidenceError("normalization_verification_failed", role)
        if (
            observed_count != reference_stats[role].nonzero_voxel_count
            or observed_count != independent_stats[role].nonzero_voxel_count
        ):
            raise EvidenceError("nonzero_count_mismatch", role)
        image.uncache()
    del reference
    del independent
    return report


def verify_preprocessing_refusal(
    adapter_plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_custody_receipt_sha256: str,
    repository_root: Path,
) -> PreprocessingRefusal:
    adapter_plan, plan_sha256 = load_adapter_plan(
        adapter_plan_path,
        repository_root=repository_root,
    )
    require_private_root(root, repository_root=repository_root)
    custody_receipt_sha256 = sha256_file(_custody_receipt_path(root))
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        custody_receipt_sha256=custody_receipt_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_custody_receipt_sha256=approve_custody_receipt_sha256,
        runtime_identity=adapter_plan.runtime_identity,
    )
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    terminal = _load_private_object(terminal_path, "preprocessing terminal")
    if terminal.get("status") == "completed":
        raise EvidenceError("preprocessing_completion_exists", "completion event exists")
    _verify_attempt(
        _load_private_object(started_path, "preprocessing attempt"),
        plan_sha256=plan_sha256,
        custody_receipt_sha256=custody_receipt_sha256,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
    )
    refusal = PreprocessingRefusal.from_dict(terminal)
    if (
        refusal.plan_sha256 != plan_sha256
        or refusal.custody_receipt_sha256 != custody_receipt_sha256
        or refusal.authorization_sha256 != authorization_sha256
        or refusal.attempt_record_sha256 != sha256_file(started_path)
    ):
        raise EvidenceError("preprocessing_refusal_mismatch", "trusted identities differ")
    return refusal
