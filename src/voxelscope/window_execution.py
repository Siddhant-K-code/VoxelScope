# SPDX-License-Identifier: Apache-2.0
"""Offline private window materialization qualification."""

from __future__ import annotations

import hashlib
import itertools
import os
import platform
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, cast

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
from .one_volume_custody import (
    _fsync_directory,
    _mkdir_private_chain,
    _write_private_json_no_clobber,
)
from .preprocessing_records import AdapterPlan, PreprocessingReport
from .records import require_object, require_string, strict_fields
from .window_execution_contract import (
    MILESTONE5_ADAPTER_PLAN_SHA256,
    MILESTONE5_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_PLAN_SHA256,
    MILESTONE6_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_REPORT_SHA256,
    TRUSTED_PLAN_SHA256,
)
from .window_execution_records import (
    CHANNEL_ORDER,
    ROI_IJK,
    STRIDE_IJK,
    ExecutionStage,
    PrivateArtifact,
    RuntimeIdentity,
    WindowExecutionCompletion,
    WindowExecutionLedger,
    WindowExecutionPlan,
    WindowExecutionRefusal,
    WindowExecutionReport,
)

PREPROCESSING_SNAPSHOT_NAME = "preprocessing-v1"
PREPROCESSING_REPORT_PATH = "evidence/preprocessing-report.json"
REFERENCE_TENSOR_PATH = "tensors/input-reference.f32le"
INDEPENDENT_TENSOR_PATH = "tensors/input-independent.f32le"
WINDOW_SNAPSHOT_NAME = "window-qualification-v1"
WINDOW_REPORT_PATH = "evidence/window-execution-report.json"
WINDOW_LEDGER_PATH = "evidence/window-ledger.json"
ATTEMPT_SCHEMA = "voxelscope/window-execution-attempt/v1"
AUTHORIZATION_SCHEMA = "voxelscope/window-execution-authorization/v1"
_COPY_CHUNK_BYTES = 1024 * 1024
_SCAN_CHUNK_VALUES = 1024 * 1024


@dataclass(frozen=True)
class WindowGeometry:
    ordinal: int
    padded_start_ijk: tuple[int, int, int]
    padded_stop_ijk: tuple[int, int, int]
    source_start_ijk: tuple[int, int, int]
    source_stop_ijk: tuple[int, int, int]
    local_pad_before_ijk: tuple[int, int, int]
    local_pad_after_ijk: tuple[int, int, int]


@dataclass(frozen=True)
class _ExpectedLedgerGeometry:
    padded_shape_ijk: tuple[int, int, int]
    pad_before_ijk: tuple[int, int, int]
    pad_after_ijk: tuple[int, int, int]
    axis_start_counts: tuple[int, int, int]
    axis_starts_sha256: str
    window_count: int
    coordinate_sha256: str
    coverage_min: int
    coverage_max: int


@dataclass
class _ExecutionProgress:
    last_completed_stage: ExecutionStage = "attempt_marker"
    completed_window_count: int = 0
    snapshot_published: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _runtime_identity(repository_root: Path) -> RuntimeIdentity:
    lock = ensure_no_symlink(repository_root, safe_relative_path("uv.lock"))
    if not lock.is_file():
        raise EvidenceError("missing_runtime_lock", "uv.lock is required")
    return RuntimeIdentity(
        cast(Any, platform.python_version()),
        cast(Any, version("numpy")),
        cast(Any, sys.byteorder),
        sha256_file(lock),
    )


def load_window_execution_plan(
    path: Path,
    *,
    repository_root: Path,
    verify_runtime: bool = True,
) -> tuple[WindowExecutionPlan, str]:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "window execution plan must be a regular file")
    digest = sha256_file(path)
    if TRUSTED_PLAN_SHA256 != "TO_BE_FINALIZED" and digest != TRUSTED_PLAN_SHA256:
        raise EvidenceError("window_execution_plan_digest_mismatch", "trusted digest differs")
    plan = WindowExecutionPlan.from_dict(require_object(load_json(path), "window execution plan"))
    root = repository_root.resolve(strict=True)
    if verify_runtime and _runtime_identity(root) != plan.runtime_identity:
        raise EvidenceError("window_runtime_mismatch", "runtime identity differs")
    for identity in plan.implementation_sources:
        source = ensure_no_symlink(root, safe_relative_path(identity.path))
        if not source.is_file() or sha256_file(source) != identity.sha256:
            raise EvidenceError("implementation_identity_mismatch", identity.path)
    public_identities = (
        ("research/preprocessing-adapter-plan-v1.json", MILESTONE5_ADAPTER_PLAN_SHA256),
        ("research/milestone-5/bundle.json", MILESTONE5_PUBLIC_BUNDLE_SHA256),
        ("research/window-bridge-plan-v1.json", MILESTONE6_PLAN_SHA256),
        ("research/window-bridge-report-v1.json", MILESTONE6_REPORT_SHA256),
        ("research/milestone-6/bundle.json", MILESTONE6_PUBLIC_BUNDLE_SHA256),
    )
    for relative, expected in public_identities:
        source = ensure_no_symlink(root, safe_relative_path(relative))
        if not source.is_file() or sha256_file(source) != expected:
            raise EvidenceError("upstream_identity_mismatch", relative)
    return plan, digest


def symmetric_padding(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if any(item <= 0 for item in (*shape_ijk, *roi_ijk)):
        raise EvidenceError("invalid_shape", "shape and ROI must be positive")
    difference = tuple(
        max(roi - dimension, 0) for dimension, roi in zip(shape_ijk, roi_ijk, strict=True)
    )
    before = (difference[0] // 2, difference[1] // 2, difference[2] // 2)
    after = (
        difference[0] - before[0],
        difference[1] - before[1],
        difference[2] - before[2],
    )
    return before, after


def axis_starts(dimension: int, roi: int, stride: int) -> tuple[int, ...]:
    if dimension <= 0 or roi <= 0 or stride <= 0:
        raise EvidenceError("invalid_shape", "axis values must be positive")
    padded = max(dimension, roi)
    final = padded - roi
    if final == 0:
        return (0,)
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def enumerate_window_geometries(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
    stride_ijk: tuple[int, int, int],
) -> Iterator[WindowGeometry]:
    before, _ = symmetric_padding(shape_ijk, roi_ijk)
    starts = tuple(
        axis_starts(dimension, roi, stride)
        for dimension, roi, stride in zip(
            shape_ijk,
            roi_ijk,
            stride_ijk,
            strict=True,
        )
    )
    source_high = tuple(before[axis] + shape_ijk[axis] for axis in range(3))
    for ordinal, start in enumerate(itertools.product(*starts)):
        padded_start = (start[0], start[1], start[2])
        padded_stop = tuple(padded_start[axis] + roi_ijk[axis] for axis in range(3))
        source_start = tuple(max(padded_start[axis] - before[axis], 0) for axis in range(3))
        source_stop = tuple(
            min(padded_stop[axis] - before[axis], shape_ijk[axis]) for axis in range(3)
        )
        local_before = tuple(max(before[axis] - padded_start[axis], 0) for axis in range(3))
        local_after = tuple(max(padded_stop[axis] - source_high[axis], 0) for axis in range(3))
        yield WindowGeometry(
            ordinal,
            padded_start,
            cast(tuple[int, int, int], padded_stop),
            cast(tuple[int, int, int], source_start),
            cast(tuple[int, int, int], source_stop),
            cast(tuple[int, int, int], local_before),
            cast(tuple[int, int, int], local_after),
        )


def materialize_full_padded_tensor(
    source: np.ndarray[Any, np.dtype[np.float32]],
    roi_ijk: tuple[int, int, int],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if source.ndim != 4 or source.dtype != np.dtype("<f4"):
        raise EvidenceError("invalid_tensor_layout", "expected little-endian float32 C,I,J,K")
    shape = cast(tuple[int, int, int], tuple(int(item) for item in source.shape[1:]))
    before, after = symmetric_padding(shape, roi_ijk)
    padded = np.pad(
        source,
        (
            (0, 0),
            (before[0], after[0]),
            (before[1], after[1]),
            (before[2], after[2]),
        ),
        mode="constant",
        constant_values=0,
    )
    return np.asarray(padded, dtype="<f4", order="C")


def materialize_from_full_padding(
    padded: np.ndarray[Any, np.dtype[np.float32]],
    geometry: WindowGeometry,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    start = geometry.padded_start_ijk
    stop = geometry.padded_stop_ijk
    return np.ascontiguousarray(
        padded[
            :,
            start[0] : stop[0],
            start[1] : stop[1],
            start[2] : stop[2],
        ],
        dtype="<f4",
    )


def materialize_direct(
    source: np.ndarray[Any, np.dtype[np.float32]],
    geometry: WindowGeometry,
    roi_ijk: tuple[int, int, int],
    pad_before_ijk: tuple[int, int, int],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if source.ndim != 4 or source.dtype != np.dtype("<f4"):
        raise EvidenceError("invalid_tensor_layout", "expected little-endian float32 C,I,J,K")
    output = np.zeros((int(source.shape[0]), *roi_ijk), dtype="<f4", order="C")
    source_start = tuple(
        max(geometry.padded_start_ijk[axis] - pad_before_ijk[axis], 0) for axis in range(3)
    )
    source_stop = tuple(
        min(
            geometry.padded_stop_ijk[axis] - pad_before_ijk[axis],
            int(source.shape[axis + 1]),
        )
        for axis in range(3)
    )
    destination_start = tuple(
        max(pad_before_ijk[axis] - geometry.padded_start_ijk[axis], 0) for axis in range(3)
    )
    extent = tuple(source_stop[axis] - source_start[axis] for axis in range(3))
    destination_stop = tuple(destination_start[axis] + extent[axis] for axis in range(3))
    output[
        :,
        destination_start[0] : destination_stop[0],
        destination_start[1] : destination_stop[1],
        destination_start[2] : destination_stop[2],
    ] = source[
        :,
        source_start[0] : source_stop[0],
        source_start[1] : source_stop[1],
        source_start[2] : source_stop[2],
    ]
    return output


def _axis_coverage(
    dimension: int,
    starts: tuple[int, ...],
    roi: int,
    pad_before: int,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    coverage = np.zeros(dimension, dtype=np.int64)
    for start in starts:
        source_start = max(start - pad_before, 0)
        source_stop = min(start + roi - pad_before, dimension)
        coverage[source_start:source_stop] += 1
    if coverage.size == 0 or not bool((coverage > 0).all()):
        raise EvidenceError("incomplete_source_coverage", "source axis is not covered")
    return coverage


def _coordinate_record(geometry: WindowGeometry) -> dict[str, Any]:
    return {
        "local_pad_after_ijk": list(geometry.local_pad_after_ijk),
        "local_pad_before_ijk": list(geometry.local_pad_before_ijk),
        "ordinal": geometry.ordinal,
        "padded_start_ijk": list(geometry.padded_start_ijk),
        "padded_stop_ijk": list(geometry.padded_stop_ijk),
        "source_start_ijk": list(geometry.source_start_ijk),
        "source_stop_ijk": list(geometry.source_stop_ijk),
    }


def _expected_ledger_geometry(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
    stride_ijk: tuple[int, int, int],
) -> _ExpectedLedgerGeometry:
    before, after = symmetric_padding(shape_ijk, roi_ijk)
    padded_shape = cast(
        tuple[int, int, int],
        tuple(shape_ijk[axis] + before[axis] + after[axis] for axis in range(3)),
    )
    starts = cast(
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        tuple(axis_starts(shape_ijk[axis], roi_ijk[axis], stride_ijk[axis]) for axis in range(3)),
    )
    for axis in range(3):
        if starts[axis][-1] != padded_shape[axis] - roi_ijk[axis]:
            raise EvidenceError("final_anchor_mismatch", str(axis))
    coverages = tuple(
        _axis_coverage(
            shape_ijk[axis],
            starts[axis],
            roi_ijk[axis],
            before[axis],
        )
        for axis in range(3)
    )
    coordinate_digest = hashlib.sha256()
    count = 0
    for geometry in enumerate_window_geometries(shape_ijk, roi_ijk, stride_ijk):
        coordinate_digest.update(canonical_json_bytes(_coordinate_record(geometry)))
        count += 1
    expected_count = len(starts[0]) * len(starts[1]) * len(starts[2])
    if count != expected_count:
        raise EvidenceError("window_count_mismatch", str(count))
    return _ExpectedLedgerGeometry(
        padded_shape,
        before,
        after,
        (len(starts[0]), len(starts[1]), len(starts[2])),
        sha256_bytes(canonical_json_bytes([list(item) for item in starts])),
        count,
        coordinate_digest.hexdigest(),
        int(np.prod([int(item.min()) for item in coverages], dtype=np.int64)),
        int(np.prod([int(item.max()) for item in coverages], dtype=np.int64)),
    )


def qualify_windows(
    source: np.ndarray[Any, np.dtype[np.float32]],
    *,
    roi_ijk: tuple[int, int, int] = ROI_IJK,
    stride_ijk: tuple[int, int, int] = STRIDE_IJK,
) -> WindowExecutionLedger:
    if (
        source.ndim != 4
        or source.dtype != np.dtype("<f4")
        or int(source.shape[0]) != len(CHANNEL_ORDER)
        or any(int(item) <= 0 for item in source.shape)
    ):
        raise EvidenceError("invalid_tensor_layout", "expected four-channel float32 C,I,J,K")
    if not bool(np.isfinite(source).all()):
        raise EvidenceError("nonfinite_input", "preprocessing tensor contains nonfinite values")
    for channel in range(len(CHANNEL_ORDER)):
        if not bool(np.any(source[channel] != np.float32(0.0))):
            raise EvidenceError("zero_channel", CHANNEL_ORDER[channel])
    shape_ijk = cast(tuple[int, int, int], tuple(int(item) for item in source.shape[1:]))
    expected = _expected_ledger_geometry(shape_ijk, roi_ijk, stride_ijk)
    before = expected.pad_before_ijk
    padded = materialize_full_padded_tensor(source, roi_ijk)
    crop = padded[
        :,
        before[0] : before[0] + shape_ijk[0],
        before[1] : before[1] + shape_ijk[1],
        before[2] : before[2] + shape_ijk[2],
    ]
    if crop.tobytes(order="C") != source.tobytes(order="C"):
        raise EvidenceError("crop_geometry_mismatch", "symmetric crop differs")
    coordinate_digest = hashlib.sha256()
    stream_digest = hashlib.sha256()
    count = 0
    previous: WindowGeometry | None = None
    for geometry in enumerate_window_geometries(shape_ijk, roi_ijk, stride_ijk):
        if previous is not None and geometry.ordinal != previous.ordinal + 1:
            raise EvidenceError("window_ordinal_mismatch", str(geometry.ordinal))
        expected_source_start = tuple(
            max(geometry.padded_start_ijk[axis] - before[axis], 0) for axis in range(3)
        )
        expected_source_stop = tuple(
            min(
                geometry.padded_stop_ijk[axis] - before[axis],
                shape_ijk[axis],
            )
            for axis in range(3)
        )
        expected_before = tuple(
            max(before[axis] - geometry.padded_start_ijk[axis], 0) for axis in range(3)
        )
        expected_after = tuple(
            max(
                geometry.padded_stop_ijk[axis] - (before[axis] + shape_ijk[axis]),
                0,
            )
            for axis in range(3)
        )
        if (
            geometry.source_start_ijk != expected_source_start
            or geometry.source_stop_ijk != expected_source_stop
            or geometry.local_pad_before_ijk != expected_before
            or geometry.local_pad_after_ijk != expected_after
            or any(
                geometry.source_stop_ijk[axis] <= geometry.source_start_ijk[axis]
                for axis in range(3)
            )
        ):
            raise EvidenceError("window_intersection_mismatch", str(geometry.ordinal))
        full = materialize_from_full_padding(padded, geometry)
        direct = materialize_direct(source, geometry, roi_ijk, before)
        full_bytes = full.tobytes(order="C")
        direct_bytes = direct.tobytes(order="C")
        if full_bytes != direct_bytes:
            raise EvidenceError("window_materialization_disagreement", str(geometry.ordinal))
        if not bool(np.isfinite(full).all()):
            raise EvidenceError("nonfinite_window", str(geometry.ordinal))
        coordinate_digest.update(canonical_json_bytes(_coordinate_record(geometry)))
        stream_digest.update(geometry.ordinal.to_bytes(8, "little"))
        stream_digest.update(len(full_bytes).to_bytes(8, "little"))
        stream_digest.update(full_bytes)
        count += 1
        previous = geometry
    if count != expected.window_count:
        raise EvidenceError("window_count_mismatch", str(count))
    if coordinate_digest.hexdigest() != expected.coordinate_sha256:
        raise EvidenceError("coordinate_identity_mismatch", "window coordinates differ")
    return WindowExecutionLedger(
        "voxelscope/window-execution-ledger/v1",
        "go",
        tuple(int(item) for item in source.shape),
        expected.padded_shape_ijk,
        roi_ijk,
        expected.pad_before_ijk,
        expected.pad_after_ijk,
        expected.axis_start_counts,
        expected.axis_starts_sha256,
        count,
        expected.coordinate_sha256,
        stream_digest.hexdigest(),
        expected.coverage_min,
        expected.coverage_max,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    )


def preprocessing_snapshot_identity(
    report_sha256: str,
    report: PreprocessingReport,
) -> str:
    require_sha256(report_sha256, "preprocessing_report_sha256")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "independent_tensor": report.independent_tensor.to_dict()
                if hasattr(report.independent_tensor, "to_dict")
                else {
                    "path": report.independent_tensor.path,
                    "sha256": report.independent_tensor.sha256,
                    "size_bytes": report.independent_tensor.size_bytes,
                },
                "preprocessing_report_sha256": report_sha256,
                "reference_tensor": {
                    "path": report.reference_tensor.path,
                    "sha256": report.reference_tensor.sha256,
                    "size_bytes": report.reference_tensor.size_bytes,
                },
                "schema_version": "voxelscope/preprocessing-snapshot-identity/v1",
                "snapshot": PREPROCESSING_SNAPSHOT_NAME,
            }
        )
    )


def _authorization(
    *,
    plan_sha256: str,
    approve_plan_sha256: str,
    approve_preprocessing_report_sha256: str,
    approve_preprocessing_snapshot_sha256: str,
    runtime_identity: RuntimeIdentity,
) -> tuple[dict[str, Any], str, str]:
    if approve_plan_sha256 != plan_sha256:
        raise EvidenceError("wrong_window_plan_approval", "approval differs")
    require_sha256(
        approve_preprocessing_report_sha256,
        "approve_preprocessing_report_sha256",
    )
    require_sha256(
        approve_preprocessing_snapshot_sha256,
        "approve_preprocessing_snapshot_sha256",
    )
    authorization = {
        "approved_plan_sha256": approve_plan_sha256,
        "approved_preprocessing_report_sha256": approve_preprocessing_report_sha256,
        "approved_preprocessing_snapshot_sha256": approve_preprocessing_snapshot_sha256,
        "inference_authorized": False,
        "model_loading_allowed": False,
        "network_allowed": False,
        "runtime_identity": runtime_identity.to_dict(),
        "schema_version": AUTHORIZATION_SCHEMA,
    }
    authorization_sha256 = sha256_bytes(canonical_json_bytes(authorization))
    attempt_id = sha256_bytes(
        canonical_json_bytes(
            {
                "plan_sha256": plan_sha256,
                "preprocessing_report_sha256": approve_preprocessing_report_sha256,
                "preprocessing_snapshot_sha256": approve_preprocessing_snapshot_sha256,
            }
        )
    )
    return authorization, authorization_sha256, attempt_id


def _attempt_paths(root: Path, attempt_id: str) -> tuple[Path, Path]:
    require_sha256(attempt_id, "attempt_id")
    attempts = ensure_no_symlink(root, safe_relative_path("window-execution-attempts"))
    return (
        attempts / f"{attempt_id}.started.json",
        attempts / f"{attempt_id}.terminal.json",
    )


def _load_milestone5_plan(repository_root: Path) -> AdapterPlan:
    path = ensure_no_symlink(
        repository_root,
        safe_relative_path("research/preprocessing-adapter-plan-v1.json"),
    )
    if not path.is_file() or sha256_file(path) != MILESTONE5_ADAPTER_PLAN_SHA256:
        raise EvidenceError("preprocessing_plan_identity_mismatch", "trusted plan differs")
    return AdapterPlan.from_dict(require_object(load_json(path), "preprocessing plan"))


def _verify_private_snapshot_file_set(snapshot: Path) -> None:
    expected = {
        PREPROCESSING_REPORT_PATH,
        REFERENCE_TENSOR_PATH,
        INDEPENDENT_TENSOR_PATH,
    }
    observed: set[str] = set()
    for directory, subdirectories, filenames in os.walk(snapshot, followlinks=False):
        current = Path(directory)
        if is_link_like(current) or (
            os.name != "nt" and stat.S_IMODE(current.stat().st_mode) != 0o700
        ):
            raise EvidenceError("private_directory_permissions", "preprocessing snapshot")
        for name in subdirectories:
            if is_link_like(current / name):
                raise EvidenceError("symlink_forbidden", "preprocessing snapshot entry")
        for name in filenames:
            path = current / name
            if (
                is_link_like(path)
                or not path.is_file()
                or (os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600)
            ):
                raise EvidenceError("private_file_permissions", "preprocessing snapshot")
            observed.add(path.relative_to(snapshot).as_posix())
    if observed != expected:
        raise EvidenceError("preprocessing_snapshot_file_set_mismatch", "file set differs")


def _load_approved_preprocessing(
    root: Path,
    *,
    approved_report_sha256: str,
    approved_snapshot_sha256: str,
    repository_root: Path,
) -> tuple[PreprocessingReport, Path]:
    snapshot = ensure_no_symlink(root, safe_relative_path(PREPROCESSING_SNAPSHOT_NAME))
    if not snapshot.is_dir() or is_link_like(snapshot):
        raise EvidenceError("missing_preprocessing_snapshot", PREPROCESSING_SNAPSHOT_NAME)
    _verify_private_snapshot_file_set(snapshot)
    report_path = ensure_no_symlink(snapshot, safe_relative_path(PREPROCESSING_REPORT_PATH))
    if sha256_file(report_path) != approved_report_sha256:
        raise EvidenceError("preprocessing_report_identity_mismatch", "approval differs")
    report = PreprocessingReport.from_dict(
        require_object(load_json(report_path), "preprocessing report")
    )
    plan = _load_milestone5_plan(repository_root)
    if (
        report.plan_sha256 != MILESTONE5_ADAPTER_PLAN_SHA256
        or report.channel_order != CHANNEL_ORDER
        or not report.label_excluded
        or report.output_layout != "C,I,J,K"
        or report.output_dtype != "<f4"
        or report.output_order != "C"
        or report.output_shape != (len(CHANNEL_ORDER), *report.input_spatial_shape)
        or report.runtime_identity != plan.runtime_identity
        or report.implementation_sources != plan.implementation_sources
        or report.monai_sources != plan.monai_sources
        or not report.finite_output
        or report.model_loaded
        or report.inference_run
        or report.inference_authorized
    ):
        raise EvidenceError("preprocessing_report_mismatch", "trusted fields differ")
    if preprocessing_snapshot_identity(approved_report_sha256, report) != approved_snapshot_sha256:
        raise EvidenceError("preprocessing_snapshot_identity_mismatch", "approval differs")
    expected_size = int(np.prod(report.output_shape, dtype=np.int64)) * 4
    for artifact in (report.reference_tensor, report.independent_tensor):
        path = ensure_no_symlink(snapshot, safe_relative_path(artifact.path))
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or artifact.size_bytes != expected_size
            or sha256_file(path) != artifact.sha256
        ):
            raise EvidenceError("preprocessing_tensor_mismatch", artifact.path)
    return report, ensure_no_symlink(snapshot, safe_relative_path(report.reference_tensor.path))


def _copy_immutable_input(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if is_link_like(source) or not source.is_file() or path_occupied(destination):
        raise EvidenceError("unsafe_path", "staged input path")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(destination.parent, 0o700)
    path_before = source.stat()
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise EvidenceError("preprocessing_tensor_mismatch", "input extent")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        while chunk := os.read(source_descriptor, _COPY_CHUNK_BYTES):
            size += len(chunk)
            if size > expected_size:
                raise EvidenceError("source_replacement", "input extent changed")
            digest.update(chunk)
            pending = memoryview(chunk)
            while pending:
                written = os.write(destination_descriptor, pending)
                pending = pending[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        path_after = source.stat()
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
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_size,
            path_before.st_mtime_ns,
            path_before.st_ctime_ns,
        ) == (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            size != expected_size
            or digest.hexdigest() != expected_sha256
            or not stable_fd
            or not stable_path
        ):
            raise EvidenceError("source_replacement", "preprocessing tensor changed")
        if os.name != "nt":
            os.fchmod(destination_descriptor, 0o400)
    except BaseException:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
            destination_descriptor = -1
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _read_and_unlink_staged_input(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> bytes:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "staged input must be a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise EvidenceError("staged_input_identity_mismatch", "extent differs")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, _COPY_CHUNK_BYTES):
            size += len(chunk)
            if size > expected_size:
                raise EvidenceError("staged_input_identity_mismatch", "extent changed")
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            size != expected_size
            or digest.hexdigest() != expected_sha256
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise EvidenceError("staged_input_identity_mismatch", "staged input changed")
        path.unlink()
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _scan_staged_input(
    tensor: np.ndarray[Any, np.dtype[np.float32]],
) -> None:
    flattened = tensor.reshape(-1)
    for start in range(0, flattened.size, _SCAN_CHUNK_VALUES):
        stop = min(start + _SCAN_CHUNK_VALUES, flattened.size)
        if not bool(np.isfinite(flattened[start:stop]).all()):
            raise EvidenceError("nonfinite_input", "preprocessing tensor")
    for channel_index, role in enumerate(CHANNEL_ORDER):
        if not bool(np.any(tensor[channel_index] != np.float32(0.0))):
            raise EvidenceError("zero_channel", role)


def _write_terminal(root: Path, attempt_id: str, value: dict[str, Any]) -> None:
    relative = f"window-execution-attempts/{attempt_id}.terminal.json"
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
    preprocessing_report_sha256: str,
    preprocessing_snapshot_sha256: str,
    authorization_sha256: str,
    attempt_record_sha256: str,
    code: str,
    progress: _ExecutionProgress,
) -> None:
    refusal = WindowExecutionRefusal(
        "voxelscope/window-execution-refusal/v1",
        "refused",
        code,
        progress.last_completed_stage,
        plan_sha256,
        preprocessing_report_sha256,
        preprocessing_snapshot_sha256,
        authorization_sha256,
        attempt_record_sha256,
        progress.completed_window_count,
        progress.snapshot_published,
        False,
        False,
        False,
        _now(),
    )
    _write_terminal(root, attempt_id, refusal.to_dict())


def execute_window_qualification(
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_preprocessing_report_sha256: str,
    approve_preprocessing_snapshot_sha256: str,
    repository_root: Path,
) -> WindowExecutionReport:
    plan, plan_sha256 = load_window_execution_plan(
        plan_path,
        repository_root=repository_root,
    )
    require_private_root(root, repository_root=repository_root)
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_preprocessing_report_sha256=approve_preprocessing_report_sha256,
        approve_preprocessing_snapshot_sha256=approve_preprocessing_snapshot_sha256,
        runtime_identity=plan.runtime_identity,
    )
    attempts = _mkdir_private_chain(root, PurePosixPath("window-execution-attempts"))
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    if path_occupied(started_path) or path_occupied(terminal_path):
        raise EvidenceError("window_execution_attempt_exists", "approval is consumed")
    snapshot = root / WINDOW_SNAPSHOT_NAME
    if path_occupied(snapshot):
        raise EvidenceError("output_exists", WINDOW_SNAPSHOT_NAME)
    attempt = {
        "authorization": authorization,
        "authorization_sha256": authorization_sha256,
        "declared_private_reads": [
            "approved milestone 5 preprocessing report",
            "approved milestone 5 reference and independent tensor identities",
            "approved milestone 5 reference tensor bytes",
        ],
        "declared_private_writes": [
            "one transient immutable reference tensor copy removed before publication",
            "one bounded private window ledger and report",
            "one terminal completion or refusal record",
        ],
        "plan_sha256": plan_sha256,
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
                f"window-execution-attempts/{attempt_id}.started.json",
                attempt,
            )
            marker_owned = True
        except EvidenceError as exc:
            if exc.code == "output_exists":
                raise EvidenceError(
                    "window_execution_attempt_exists",
                    "approval is consumed",
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
            raise EvidenceError("window_execution_attempt_mismatch", "attempt marker differs")

        report, source_path = _load_approved_preprocessing(
            root,
            approved_report_sha256=approve_preprocessing_report_sha256,
            approved_snapshot_sha256=approve_preprocessing_snapshot_sha256,
            repository_root=repository_root,
        )
        progress.last_completed_stage = "preprocessing_verification"
        stage = Path(tempfile.mkdtemp(prefix=".window-stage-", dir=root))
        if os.name != "nt":
            os.chmod(stage, 0o700)
        staged_input = stage / ".input" / "reference.f32le"
        _copy_immutable_input(
            source_path,
            staged_input,
            expected_sha256=report.reference_tensor.sha256,
            expected_size=report.reference_tensor.size_bytes,
        )
        progress.last_completed_stage = "input_staging"
        staged_bytes = _read_and_unlink_staged_input(
            staged_input,
            expected_sha256=report.reference_tensor.sha256,
            expected_size=report.reference_tensor.size_bytes,
        )
        staged_input_sha256 = sha256_bytes(staged_bytes)
        if staged_input_sha256 != report.reference_tensor.sha256:
            raise EvidenceError("staged_input_identity_mismatch", "copy differs")
        tensor = np.frombuffer(
            staged_bytes,
            dtype="<f4",
        ).reshape(report.output_shape, order="C")
        if tensor.flags.writeable:
            raise EvidenceError("staged_input_mutable", "staged bytes are writable")
        shutil.rmtree(stage / ".input")
        _scan_staged_input(tensor)
        progress.last_completed_stage = "input_validation"
        ledger = qualify_windows(
            tensor,
            roi_ijk=plan.roi_ijk,
            stride_ijk=plan.stride_ijk,
        )
        progress.completed_window_count = ledger.window_count
        progress.last_completed_stage = "window_materialization"
        if sha256_bytes(staged_bytes) != report.reference_tensor.sha256:
            raise EvidenceError("staged_input_identity_mismatch", "input changed")
        del tensor
        del staged_bytes
        ledger_path = _write_private_json_no_clobber(
            stage,
            WINDOW_LEDGER_PATH,
            ledger.to_dict(),
        )
        ledger_artifact = PrivateArtifact(
            WINDOW_LEDGER_PATH,
            sha256_file(ledger_path),
            ledger_path.stat().st_size,
        )
        progress.last_completed_stage = "ledger_publication"
        execution_report = WindowExecutionReport(
            "voxelscope/window-execution-report/v1",
            "go",
            plan_sha256,
            report.plan_sha256,
            approve_preprocessing_report_sha256,
            approve_preprocessing_snapshot_sha256,
            report.reference_tensor.sha256,
            staged_input_sha256,
            authorization_sha256,
            attempt_record_sha256,
            report.output_shape,
            "C,I,J,K",
            "N,C,D,H,W",
            CHANNEL_ORDER,
            True,
            ledger_artifact,
            plan.runtime_identity,
            plan.implementation_sources,
            False,
            False,
            False,
            False,
            _now(),
        )
        execution_report_path = _write_private_json_no_clobber(
            stage,
            WINDOW_REPORT_PATH,
            execution_report.to_dict(),
        )
        report_sha256 = sha256_file(execution_report_path)
        _fsync_directory(stage / "evidence")
        _fsync_directory(stage)
        progress.last_completed_stage = "snapshot_publication"
        rename_no_replace(stage, snapshot)
        progress.snapshot_published = True
        _fsync_directory(root)
        completion = WindowExecutionCompletion(
            "voxelscope/window-execution-completion/v1",
            "completed",
            plan_sha256,
            approve_preprocessing_report_sha256,
            approve_preprocessing_snapshot_sha256,
            authorization_sha256,
            attempt_record_sha256,
            report_sha256,
            ledger_artifact.sha256,
            "window-qualification-v1",
            False,
            False,
            _now(),
        )
        _write_terminal(root, attempt_id, completion.to_dict())
        return execution_report
    except BaseException as exc:
        code = exc.code if isinstance(exc, EvidenceError) else "window_execution_interrupted"
        if marker_owned:
            try:
                if path_occupied(terminal_path):
                    terminal = _load_private_object(terminal_path, "window terminal")
                    if terminal.get("status") != "completed":
                        WindowExecutionRefusal.from_dict(terminal)
                else:
                    _record_refusal(
                        root,
                        attempt_id=attempt_id,
                        plan_sha256=plan_sha256,
                        preprocessing_report_sha256=approve_preprocessing_report_sha256,
                        preprocessing_snapshot_sha256=approve_preprocessing_snapshot_sha256,
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
                "window_execution_interrupted",
                "the approved attempt is consumed",
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
    authorization: dict[str, Any],
    authorization_sha256: str,
) -> None:
    value = strict_fields(
        value,
        {
            "authorization",
            "authorization_sha256",
            "declared_private_reads",
            "declared_private_writes",
            "plan_sha256",
            "schema_version",
            "started_at",
        },
        "window execution attempt",
    )
    if (
        value["schema_version"] != ATTEMPT_SCHEMA
        or value["plan_sha256"] != plan_sha256
        or value["authorization"] != authorization
        or value["authorization_sha256"] != authorization_sha256
        or value["declared_private_reads"]
        != [
            "approved milestone 5 preprocessing report",
            "approved milestone 5 reference and independent tensor identities",
            "approved milestone 5 reference tensor bytes",
        ]
        or value["declared_private_writes"]
        != [
            "one transient immutable reference tensor copy removed before publication",
            "one bounded private window ledger and report",
            "one terminal completion or refusal record",
        ]
    ):
        raise EvidenceError("window_execution_attempt_mismatch", "attempt differs")
    require_string(value["started_at"], "started_at")


def _verify_output_snapshot(snapshot: Path) -> None:
    expected = {WINDOW_LEDGER_PATH, WINDOW_REPORT_PATH}
    observed: set[str] = set()
    for directory, subdirectories, filenames in os.walk(snapshot, followlinks=False):
        current = Path(directory)
        if is_link_like(current) or (
            os.name != "nt" and stat.S_IMODE(current.stat().st_mode) != 0o700
        ):
            raise EvidenceError("private_directory_permissions", "window snapshot")
        for name in subdirectories:
            if is_link_like(current / name):
                raise EvidenceError("symlink_forbidden", "window snapshot")
        for name in filenames:
            path = current / name
            if (
                is_link_like(path)
                or not path.is_file()
                or (os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600)
            ):
                raise EvidenceError("private_file_permissions", "window snapshot")
            observed.add(path.relative_to(snapshot).as_posix())
    if observed != expected:
        raise EvidenceError("window_snapshot_file_set_mismatch", "file set differs")


def verify_window_qualification(
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_preprocessing_report_sha256: str,
    approve_preprocessing_snapshot_sha256: str,
    repository_root: Path,
) -> WindowExecutionReport:
    plan, plan_sha256 = load_window_execution_plan(
        plan_path,
        repository_root=repository_root,
    )
    require_private_root(root, repository_root=repository_root)
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_preprocessing_report_sha256=approve_preprocessing_report_sha256,
        approve_preprocessing_snapshot_sha256=approve_preprocessing_snapshot_sha256,
        runtime_identity=plan.runtime_identity,
    )
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    _verify_attempt(
        _load_private_object(started_path, "window attempt"),
        plan_sha256=plan_sha256,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
    )
    attempt_record_sha256 = sha256_file(started_path)
    terminal = _load_private_object(terminal_path, "window terminal")
    if terminal.get("status") == "refused":
        WindowExecutionRefusal.from_dict(terminal)
        raise EvidenceError("window_execution_refused", "refusal exists")
    completion = WindowExecutionCompletion.from_dict(terminal)
    snapshot = ensure_no_symlink(root, safe_relative_path(WINDOW_SNAPSHOT_NAME))
    if not snapshot.is_dir():
        raise EvidenceError("missing_window_snapshot", WINDOW_SNAPSHOT_NAME)
    _verify_output_snapshot(snapshot)
    ledger_path = ensure_no_symlink(snapshot, safe_relative_path(WINDOW_LEDGER_PATH))
    report_path = ensure_no_symlink(snapshot, safe_relative_path(WINDOW_REPORT_PATH))
    ledger = WindowExecutionLedger.from_dict(_load_private_object(ledger_path, "window ledger"))
    report = WindowExecutionReport.from_dict(_load_private_object(report_path, "window report"))
    if (
        completion.plan_sha256 != plan_sha256
        or completion.preprocessing_report_sha256 != approve_preprocessing_report_sha256
        or completion.preprocessing_snapshot_sha256 != approve_preprocessing_snapshot_sha256
        or completion.authorization_sha256 != authorization_sha256
        or completion.attempt_record_sha256 != attempt_record_sha256
        or completion.report_sha256 != sha256_file(report_path)
        or completion.ledger_sha256 != sha256_file(ledger_path)
        or report.plan_sha256 != plan_sha256
        or report.preprocessing_report_sha256 != approve_preprocessing_report_sha256
        or report.preprocessing_snapshot_sha256 != approve_preprocessing_snapshot_sha256
        or report.authorization_sha256 != authorization_sha256
        or report.attempt_record_sha256 != attempt_record_sha256
        or report.ledger.sha256 != sha256_file(ledger_path)
        or report.ledger.size_bytes != ledger_path.stat().st_size
        or report.input_shape != ledger.input_shape
        or report.runtime_identity != plan.runtime_identity
        or report.implementation_sources != plan.implementation_sources
    ):
        raise EvidenceError("window_execution_evidence_mismatch", "identities differ")
    expected_geometry = _expected_ledger_geometry(
        cast(tuple[int, int, int], report.input_shape[1:]),
        plan.roi_ijk,
        plan.stride_ijk,
    )
    if (
        ledger.roi_ijk != plan.roi_ijk
        or ledger.padded_shape_ijk != expected_geometry.padded_shape_ijk
        or ledger.pad_before_ijk != expected_geometry.pad_before_ijk
        or ledger.pad_after_ijk != expected_geometry.pad_after_ijk
        or ledger.axis_start_counts != expected_geometry.axis_start_counts
        or ledger.axis_starts_sha256 != expected_geometry.axis_starts_sha256
        or ledger.window_count != expected_geometry.window_count
        or ledger.coordinate_sha256 != expected_geometry.coordinate_sha256
        or ledger.coverage_min != expected_geometry.coverage_min
        or ledger.coverage_max != expected_geometry.coverage_max
    ):
        raise EvidenceError("window_geometry_evidence_mismatch", "plan geometry differs")
    preprocessing_report, source_path = _load_approved_preprocessing(
        root,
        approved_report_sha256=approve_preprocessing_report_sha256,
        approved_snapshot_sha256=approve_preprocessing_snapshot_sha256,
        repository_root=repository_root,
    )
    if (
        report.preprocessing_tensor_sha256 != preprocessing_report.reference_tensor.sha256
        or sha256_file(source_path) != report.preprocessing_tensor_sha256
    ):
        raise EvidenceError("preprocessing_tensor_mismatch", "source differs")
    return report


def verify_window_refusal(
    plan_path: Path,
    root: Path,
    *,
    approve_plan_sha256: str,
    approve_preprocessing_report_sha256: str,
    approve_preprocessing_snapshot_sha256: str,
    repository_root: Path,
) -> WindowExecutionRefusal:
    plan, plan_sha256 = load_window_execution_plan(
        plan_path,
        repository_root=repository_root,
    )
    require_private_root(root, repository_root=repository_root)
    authorization, authorization_sha256, attempt_id = _authorization(
        plan_sha256=plan_sha256,
        approve_plan_sha256=approve_plan_sha256,
        approve_preprocessing_report_sha256=approve_preprocessing_report_sha256,
        approve_preprocessing_snapshot_sha256=approve_preprocessing_snapshot_sha256,
        runtime_identity=plan.runtime_identity,
    )
    started_path, terminal_path = _attempt_paths(root, attempt_id)
    _verify_attempt(
        _load_private_object(started_path, "window attempt"),
        plan_sha256=plan_sha256,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
    )
    terminal = _load_private_object(terminal_path, "window terminal")
    if terminal.get("status") == "completed":
        raise EvidenceError("window_execution_completion_exists", "completion exists")
    refusal = WindowExecutionRefusal.from_dict(terminal)
    if (
        refusal.plan_sha256 != plan_sha256
        or refusal.preprocessing_report_sha256 != approve_preprocessing_report_sha256
        or refusal.preprocessing_snapshot_sha256 != approve_preprocessing_snapshot_sha256
        or refusal.authorization_sha256 != authorization_sha256
        or refusal.attempt_record_sha256 != sha256_file(started_path)
    ):
        raise EvidenceError("window_execution_refusal_mismatch", "identities differ")
    return refusal
