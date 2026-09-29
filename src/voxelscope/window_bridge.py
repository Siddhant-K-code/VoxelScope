# SPDX-License-Identifier: Apache-2.0
"""Offline synthetic proof for the preprocessing-to-window positional bridge."""

from __future__ import annotations

import hashlib
import itertools
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atomic import link_file_no_replace, path_occupied
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
    write_json,
)
from .fixtures import build_fixture_bundle
from .records import OverlapRatio, WindowConfig, require_object
from .synthetic_contract import synthetic_modalities, synthetic_window_config
from .window_bridge_contract import TRUSTED_PLAN_SHA256, TRUSTED_REPORT_SHA256
from .window_bridge_records import (
    LEGACY_SYNTHETIC_BUNDLE_SHA256,
    LEGACY_SYNTHETIC_COORDINATE_SHA256,
    LEGACY_SYNTHETIC_LEDGER_SHA256,
    PROOF_CASE_IDS,
    BridgeStage,
    WindowBridgePlan,
    WindowBridgeRefusal,
    WindowBridgeReport,
)
from .windows import build_window_evidence, identity_reconstruct


@dataclass(frozen=True)
class BridgeWindow:
    ordinal: int
    start_ijk: tuple[int, int, int]
    stop_ijk: tuple[int, int, int]
    source_stop_ijk: tuple[int, int, int]
    pad_before_ijk: tuple[int, int, int]
    pad_after_ijk: tuple[int, int, int]


@dataclass(frozen=True)
class OracleResult:
    starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    windows: tuple[BridgeWindow, ...]
    coordinate_sha256: str
    coverage_min: int
    coverage_max: int


@dataclass(frozen=True)
class ProofCase:
    case_id: str
    shape_ijk: tuple[int, int, int]
    roi_ijk: tuple[int, int, int]


@dataclass(frozen=True)
class AxisPadding:
    before: int
    after: int


@dataclass(frozen=True)
class SymmetricBridgeWindow:
    ordinal: int
    padded_start_ijk: tuple[int, int, int]
    padded_stop_ijk: tuple[int, int, int]
    source_start_ijk: tuple[int, int, int]
    source_stop_ijk: tuple[int, int, int]
    patch_pad_before_ijk: tuple[int, int, int]
    patch_pad_after_ijk: tuple[int, int, int]


PROOF_CASES = (
    ProofCase("asymmetric-components", (17, 19, 23), (8, 7, 6)),
    ProofCase("final-anchor", (18, 20, 22), (7, 8, 9)),
    ProofCase("small-volume-padding", (3, 4, 5), (6, 7, 8)),
    ProofCase("legacy-synthetic-v1", (7, 8, 9), (8, 5, 6)),
)


def symmetric_padding(dimension: int, roi: int) -> AxisPadding:
    """Return MONAI's low/high split for one positional spatial axis."""
    if dimension <= 0 or roi <= 0:
        raise EvidenceError("invalid_shape", f"{dimension},{roi}")
    difference = max(roi - dimension, 0)
    before = difference // 2
    return AxisPadding(before, difference - before)


def monai_scan_interval(
    dimension: int,
    roi: int,
    numerator: int,
    denominator: int,
) -> int:
    """Return MONAI 1.4.0's interval after minimum-ROI padding."""
    if dimension <= 0 or roi <= 0:
        raise EvidenceError("invalid_shape", f"{dimension},{roi}")
    if denominator <= 0 or not 0 <= numerator < denominator:
        raise EvidenceError("invalid_overlap", f"{numerator}/{denominator}")
    if dimension <= roi:
        return roi
    return max(1, (roi * (denominator - numerator)) // denominator)


def independent_axis_starts(
    dimension: int,
    roi: int,
    numerator: int,
    denominator: int,
) -> tuple[int, ...]:
    """Compute final-anchored starts without calling the engine helper."""
    if dimension <= 0 or roi <= 0:
        raise EvidenceError("invalid_shape", f"{dimension},{roi}")
    if denominator <= 0 or not 0 <= numerator < denominator:
        raise EvidenceError("invalid_overlap", f"{numerator}/{denominator}")
    interval = max(1, (roi * (denominator - numerator)) // denominator)
    final = max(dimension - roi, 0)
    count = (final + interval - 1) // interval + 1
    return tuple(min(index * interval, final) for index in range(count))


def monai_axis_starts(
    dimension: int,
    roi: int,
    numerator: int,
    denominator: int,
) -> tuple[int, ...]:
    """Mirror MONAI 1.4.0 dense patch starts for one padded spatial axis."""
    image_size = max(dimension, roi)
    interval = monai_scan_interval(dimension, roi, numerator, denominator)
    scan_count = (image_size - roi + interval - 1) // interval + 1
    return tuple(min(index * interval, image_size - roi) for index in range(scan_count))


def monai_symmetric_padding(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return MONAI low/high positional padding for source comparison."""
    padding = tuple(
        symmetric_padding(dimension, roi) for dimension, roi in zip(shape_ijk, roi_ijk, strict=True)
    )
    before = (padding[0].before, padding[1].before, padding[2].before)
    after = (padding[0].after, padding[1].after, padding[2].after)
    return before, after


def monai_pad_tuple(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
) -> tuple[int, ...]:
    """Return the last-axis-first pair order passed to torch functional pad."""
    before, after = monai_symmetric_padding(shape_ijk, roi_ijk)
    return (
        before[2],
        after[2],
        before[1],
        after[1],
        before[0],
        after[0],
    )


def enumerate_bridge_windows(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
    overlap_numerator: int,
    overlap_denominator: int,
) -> tuple[SymmetricBridgeWindow, ...]:
    """Enumerate MONAI-positioned windows with K changing fastest."""
    before, _ = monai_symmetric_padding(shape_ijk, roi_ijk)
    starts = tuple(
        monai_axis_starts(
            dimension,
            roi,
            overlap_numerator,
            overlap_denominator,
        )
        for dimension, roi in zip(shape_ijk, roi_ijk, strict=True)
    )
    windows: list[SymmetricBridgeWindow] = []
    for ordinal, start in enumerate(itertools.product(*starts)):
        start_ijk = (start[0], start[1], start[2])
        stop = (
            start[0] + roi_ijk[0],
            start[1] + roi_ijk[1],
            start[2] + roi_ijk[2],
        )
        source_start = (
            max(start[0] - before[0], 0),
            max(start[1] - before[1], 0),
            max(start[2] - before[2], 0),
        )
        source_stop = (
            min(stop[0] - before[0], shape_ijk[0]),
            min(stop[1] - before[1], shape_ijk[1]),
            min(stop[2] - before[2], shape_ijk[2]),
        )
        patch_before = (
            max(before[0] - start[0], 0),
            max(before[1] - start[1], 0),
            max(before[2] - start[2], 0),
        )
        source_high = (
            before[0] + shape_ijk[0],
            before[1] + shape_ijk[1],
            before[2] + shape_ijk[2],
        )
        patch_after = (
            max(stop[0] - source_high[0], 0),
            max(stop[1] - source_high[1], 0),
            max(stop[2] - source_high[2], 0),
        )
        windows.append(
            SymmetricBridgeWindow(
                ordinal,
                start_ijk,
                stop,
                source_start,
                source_stop,
                patch_before,
                patch_after,
            )
        )
    return tuple(windows)


def crop_slices(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Return slices that remove MONAI's symmetric spatial padding."""
    before, _ = monai_symmetric_padding(shape_ijk, roi_ijk)
    return (
        slice(before[0], before[0] + shape_ijk[0]),
        slice(before[1], before[1] + shape_ijk[1]),
        slice(before[2], before[2] + shape_ijk[2]),
    )


def apply_symmetric_padding(
    array: np.ndarray[Any, Any],
    roi_ijk: tuple[int, int, int],
) -> np.ndarray[Any, Any]:
    """Pad a channel-first tensor exactly as the positional bridge specifies."""
    if array.ndim != 4:
        raise EvidenceError("invalid_tensor_rank", str(array.ndim))
    shape_ijk = (int(array.shape[1]), int(array.shape[2]), int(array.shape[3]))
    before, after = monai_symmetric_padding(shape_ijk, roi_ijk)
    return np.pad(
        array,
        (
            (0, 0),
            (before[0], after[0]),
            (before[1], after[1]),
            (before[2], after[2]),
        ),
        mode="constant",
        constant_values=0,
    )


def independent_window_oracle(
    shape_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int],
    overlap: tuple[OverlapRatio, OverlapRatio, OverlapRatio],
) -> OracleResult:
    """Enumerate high-side-padded I/J/K windows with K changing fastest."""
    starts = tuple(
        independent_axis_starts(
            shape_ijk[axis],
            roi_ijk[axis],
            overlap[axis].numerator,
            overlap[axis].denominator,
        )
        for axis in range(3)
    )
    coverage = np.zeros(shape_ijk, dtype=np.uint32)
    windows: list[BridgeWindow] = []
    ordinal = 0
    for i_start in starts[0]:
        for j_start in starts[1]:
            for k_start in starts[2]:
                start = (i_start, j_start, k_start)
                stop = (
                    start[0] + roi_ijk[0],
                    start[1] + roi_ijk[1],
                    start[2] + roi_ijk[2],
                )
                source_stop = (
                    min(stop[0], shape_ijk[0]),
                    min(stop[1], shape_ijk[1]),
                    min(stop[2], shape_ijk[2]),
                )
                pad_after = (
                    stop[0] - source_stop[0],
                    stop[1] - source_stop[1],
                    stop[2] - source_stop[2],
                )
                source = (
                    slice(start[0], source_stop[0]),
                    slice(start[1], source_stop[1]),
                    slice(start[2], source_stop[2]),
                )
                coverage[source] += 1
                windows.append(
                    BridgeWindow(
                        ordinal,
                        start,
                        stop,
                        source_stop,
                        (0, 0, 0),
                        pad_after,
                    )
                )
                ordinal += 1
    if not bool((coverage > 0).all()):
        raise EvidenceError("incomplete_source_coverage", str(shape_ijk))
    coordinates = [
        {
            "end": list(item.stop_ijk),
            "index": item.ordinal,
            "pad_after": list(item.pad_after_ijk),
            "source_end": list(item.source_stop_ijk),
            "start": list(item.start_ijk),
        }
        for item in windows
    ]
    return OracleResult(
        starts,  # type: ignore[arg-type]
        tuple(windows),
        sha256_bytes(canonical_json_bytes(coordinates)),
        int(coverage.min()),
        int(coverage.max()),
    )


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def load_window_bridge_plan(
    path: Path,
    *,
    repository_root: Path,
) -> WindowBridgePlan:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "window bridge plan must be a regular file")
    plan_sha256 = sha256_file(path)
    if plan_sha256 != TRUSTED_PLAN_SHA256:
        raise EvidenceError("window_bridge_plan_digest_mismatch", "trusted digest differs")
    plan = WindowBridgePlan.from_dict(require_object(load_json(path), "window bridge plan"))
    root = repository_root.resolve(strict=True)
    for identity in plan.implementation_sources:
        source = ensure_no_symlink(root, safe_relative_path(identity.path))
        if (
            not source.is_file()
            or sha256_file(source) != identity.sha256
            or _git_blob_sha1(source) != identity.git_blob_sha1
        ):
            raise EvidenceError("implementation_identity_mismatch", identity.path)
    return plan


def load_window_bridge_report(path: Path) -> WindowBridgeReport:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_path", "window bridge report must be a regular file")
    if sha256_file(path) != TRUSTED_REPORT_SHA256:
        raise EvidenceError("window_bridge_report_digest_mismatch", "trusted digest differs")
    return WindowBridgeReport.from_dict(require_object(load_json(path), "window bridge report"))


def _fixture_values(shape_ijk: tuple[int, int, int]) -> np.ndarray[Any, Any]:
    i, j, k = np.indices(shape_ijk, dtype=np.float32)
    positional = i * np.float32(10000) + j * np.float32(100) + k
    return np.stack([positional + np.float32(channel * 1000000) for channel in range(4)]).astype(
        "<f4"
    )


def _engine_coordinates(report: Any) -> list[dict[str, Any]]:
    return [
        {
            "end": list(item.end),
            "index": item.index,
            "pad_after": list(item.pad_after),
            "source_end": list(item.source_end),
            "start": list(item.start),
        }
        for item in report.entries
    ]


def _verify_case(
    case: ProofCase,
    overlap: tuple[OverlapRatio, OverlapRatio, OverlapRatio],
) -> dict[str, Any]:
    config = WindowConfig(case.shape_ijk, case.roi_ijk, overlap, "constant", None)
    values = _fixture_values(case.shape_ijk)
    engine, weights = build_window_evidence(values, config)
    oracle = independent_window_oracle(case.shape_ijk, case.roi_ijk, overlap)
    engine_coordinates = _engine_coordinates(engine)
    oracle_coordinates = [
        {
            "end": list(item.stop_ijk),
            "index": item.ordinal,
            "pad_after": list(item.pad_after_ijk),
            "source_end": list(item.source_stop_ijk),
            "start": list(item.start_ijk),
        }
        for item in oracle.windows
    ]
    if canonical_json_bytes(engine_coordinates) != canonical_json_bytes(oracle_coordinates):
        raise EvidenceError("window_oracle_disagreement", case.case_id)
    if engine.coordinate_sha256 != oracle.coordinate_sha256:
        raise EvidenceError("coordinate_identity_disagreement", case.case_id)
    if (
        engine.coverage_min != oracle.coverage_min
        or engine.coverage_max != oracle.coverage_max
        or weights.shape != case.roi_ijk
        or not bool(np.equal(weights, 1.0).all())
    ):
        raise EvidenceError("window_geometry_disagreement", case.case_id)
    if not np.array_equal(identity_reconstruct(values, config), values):
        raise EvidenceError("output_crop_disagreement", case.case_id)
    for axis in range(3):
        if oracle.starts[axis][-1] != max(case.shape_ijk[axis] - case.roi_ijk[axis], 0):
            raise EvidenceError("final_anchor_disagreement", case.case_id)
    if len(oracle.starts[2]) > 1:
        first, second = oracle.windows[:2]
        if first.start_ijk[:2] != second.start_ijk[:2] or first.start_ijk[2] == second.start_ijk[2]:
            raise EvidenceError("traversal_order_disagreement", case.case_id)
    difference = (
        max(case.roi_ijk[0] - case.shape_ijk[0], 0),
        max(case.roi_ijk[1] - case.shape_ijk[1], 0),
        max(case.roi_ijk[2] - case.shape_ijk[2], 0),
    )
    expected_before = (
        difference[0] // 2,
        difference[1] // 2,
        difference[2] // 2,
    )
    expected_after = (
        difference[0] - expected_before[0],
        difference[1] - expected_before[1],
        difference[2] - expected_before[2],
    )
    padded_shape = (
        case.shape_ijk[0] + expected_before[0] + expected_after[0],
        case.shape_ijk[1] + expected_before[1] + expected_after[1],
        case.shape_ijk[2] + expected_before[2] + expected_after[2],
    )
    monai_starts = tuple(
        monai_axis_starts(
            case.shape_ijk[axis],
            case.roi_ijk[axis],
            overlap[axis].numerator,
            overlap[axis].denominator,
        )
        for axis in range(3)
    )
    if monai_starts != oracle.starts:
        raise EvidenceError("monai_scan_order_disagreement", case.case_id)
    before, after = monai_symmetric_padding(case.shape_ijk, case.roi_ijk)
    if before != expected_before or after != expected_after:
        raise EvidenceError("monai_padding_disagreement", case.case_id)
    padded = apply_symmetric_padding(values, case.roi_ijk)
    if padded.shape != (values.shape[0], *padded_shape):
        raise EvidenceError("monai_padding_geometry_disagreement", case.case_id)
    expected_crop = (
        slice(expected_before[0], expected_before[0] + case.shape_ijk[0]),
        slice(expected_before[1], expected_before[1] + case.shape_ijk[1]),
        slice(expected_before[2], expected_before[2] + case.shape_ijk[2]),
    )
    if crop_slices(case.shape_ijk, case.roi_ijk) != expected_crop or not np.array_equal(
        padded[(slice(None), *expected_crop)],
        values,
    ):
        raise EvidenceError("monai_crop_geometry_disagreement", case.case_id)

    symmetric_windows = enumerate_bridge_windows(
        case.shape_ijk,
        case.roi_ijk,
        overlap[0].numerator,
        overlap[0].denominator,
    )
    expected_starts = tuple(itertools.product(*monai_starts))
    if len(symmetric_windows) != len(expected_starts):
        raise EvidenceError("monai_window_count_disagreement", case.case_id)
    symmetric_coordinates: list[dict[str, Any]] = []
    symmetric_coverage = np.zeros(case.shape_ijk, dtype=np.uint32)
    for ordinal, (window, start) in enumerate(zip(symmetric_windows, expected_starts, strict=True)):
        stop = (
            start[0] + case.roi_ijk[0],
            start[1] + case.roi_ijk[1],
            start[2] + case.roi_ijk[2],
        )
        source_start = (
            max(start[0] - expected_before[0], 0),
            max(start[1] - expected_before[1], 0),
            max(start[2] - expected_before[2], 0),
        )
        source_stop = (
            min(stop[0] - expected_before[0], case.shape_ijk[0]),
            min(stop[1] - expected_before[1], case.shape_ijk[1]),
            min(stop[2] - expected_before[2], case.shape_ijk[2]),
        )
        patch_before = (
            max(expected_before[0] - start[0], 0),
            max(expected_before[1] - start[1], 0),
            max(expected_before[2] - start[2], 0),
        )
        source_high = (
            expected_before[0] + case.shape_ijk[0],
            expected_before[1] + case.shape_ijk[1],
            expected_before[2] + case.shape_ijk[2],
        )
        patch_after = (
            max(stop[0] - source_high[0], 0),
            max(stop[1] - source_high[1], 0),
            max(stop[2] - source_high[2], 0),
        )
        expected_window = SymmetricBridgeWindow(
            ordinal,
            start,
            stop,
            source_start,
            source_stop,
            patch_before,
            patch_after,
        )
        if window != expected_window:
            raise EvidenceError("monai_window_intersection_disagreement", case.case_id)
        symmetric_coverage[
            source_start[0] : source_stop[0],
            source_start[1] : source_stop[1],
            source_start[2] : source_stop[2],
        ] += 1
        symmetric_coordinates.append(
            {
                "ordinal": ordinal,
                "pad_after": list(patch_after),
                "pad_before": list(patch_before),
                "source_start": list(source_start),
                "source_stop": list(source_stop),
                "start": list(start),
                "stop": list(stop),
            }
        )
    if not bool((symmetric_coverage > 0).all()):
        raise EvidenceError("monai_source_coverage_disagreement", case.case_id)
    return {
        "case_id": case.case_id,
        "coordinate_sha256": oracle.coordinate_sha256,
        "coverage_max": oracle.coverage_max,
        "coverage_min": oracle.coverage_min,
        "monai_pad_tuple": list(monai_pad_tuple(case.shape_ijk, case.roi_ijk)),
        "oracle_starts": [list(item) for item in oracle.starts],
        "roi_ijk": list(case.roi_ijk),
        "shape_ijk": list(case.shape_ijk),
        "symmetric_coordinate_sha256": sha256_bytes(canonical_json_bytes(symmetric_coordinates)),
        "symmetric_coverage_max": int(symmetric_coverage.max()),
        "symmetric_coverage_min": int(symmetric_coverage.min()),
        "symmetric_pad_after": list(expected_after),
        "symmetric_pad_before": list(expected_before),
        "window_count": len(oracle.windows),
    }


def _verify_legacy_contract() -> None:
    legacy, _ = build_window_evidence(synthetic_modalities(), synthetic_window_config())
    if legacy.coordinate_sha256 != LEGACY_SYNTHETIC_COORDINATE_SHA256:
        raise EvidenceError("legacy_coordinate_drift", "synthetic v1 coordinates changed")
    temporary_root = Path(tempfile.mkdtemp(prefix=".voxelscope-m6-legacy-"))
    try:
        bundle = temporary_root / "bundle"
        bundle_sha256 = build_fixture_bundle(bundle)
        ledger_sha256 = sha256_file(bundle / "windows" / "window-ledger.json")
        if (
            bundle_sha256 != LEGACY_SYNTHETIC_BUNDLE_SHA256
            or ledger_sha256 != LEGACY_SYNTHETIC_LEDGER_SHA256
        ):
            raise EvidenceError("legacy_bundle_drift", "synthetic v1 identities changed")
    finally:
        shutil.rmtree(temporary_root)


def build_window_bridge_report(
    plan_path: Path,
    plan: WindowBridgePlan,
) -> WindowBridgeReport:
    if tuple(item.case_id for item in PROOF_CASES) != PROOF_CASE_IDS:
        raise EvidenceError("proof_case_drift", "proof case identities differ")
    overlap = (
        OverlapRatio(plan.overlap_numerator, plan.overlap_denominator),
        OverlapRatio(plan.overlap_numerator, plan.overlap_denominator),
        OverlapRatio(plan.overlap_numerator, plan.overlap_denominator),
    )
    proofs = [_verify_case(case, overlap) for case in PROOF_CASES]
    _verify_legacy_contract()
    report = WindowBridgeReport(
        schema_version="voxelscope/window-bridge-report/v1",
        status="go",
        plan_sha256=sha256_file(plan_path),
        proof_case_count=len(proofs),
        proof_sha256=sha256_bytes(canonical_json_bytes(proofs)),
        axis_mapping_status="go",
        roi_mapping_status="go",
        engine_oracle_status="go",
        monai_scan_order_status="go",
        voxelscope_padding_status="go",
        monai_padding_source_status="go",
        padding_policies_equivalent=False,
        crop_status="go",
        blend_axis_status="go",
        traversal_status="go",
        final_anchor_status="go",
        coverage_status="go",
        legacy_bundle_status="go",
        legacy_unpadded_enumeration_status="go",
        legacy_general_padded_reuse_status="no-go",
        private_data_accessed=False,
        private_shape_recorded=False,
        private_window_count_recorded=False,
        model_loaded=False,
        inference_run=False,
        inference_authorized=False,
    )
    return WindowBridgeReport.from_dict(report.to_dict())


def verify_window_bridge_report(
    plan_path: Path,
    report_path: Path,
    *,
    repository_root: Path,
) -> WindowBridgeReport:
    plan = load_window_bridge_plan(plan_path, repository_root=repository_root)
    expected = build_window_bridge_report(plan_path, plan)
    actual = load_window_bridge_report(report_path)
    if canonical_json_bytes(actual.to_dict()) != canonical_json_bytes(expected.to_dict()):
        raise EvidenceError("window_bridge_report_mismatch", "report content differs")
    return actual


def write_window_bridge_report(
    plan_path: Path,
    destination: Path,
    *,
    repository_root: Path,
) -> WindowBridgeReport:
    if path_occupied(destination):
        raise EvidenceError("output_exists", str(destination))
    plan = load_window_bridge_plan(plan_path, repository_root=repository_root)
    report = build_window_bridge_report(plan_path, plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_json(temporary, report.to_dict())
        link_file_no_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def window_bridge_refusal(
    error_code: str,
    stage: BridgeStage,
    plan_sha256: str,
) -> WindowBridgeRefusal:
    require_sha256(plan_sha256, "plan_sha256")
    refusal = WindowBridgeRefusal(
        "voxelscope/window-bridge-refusal/v1",
        "refused",
        sha256_bytes(
            canonical_json_bytes(
                {
                    "error_code": error_code,
                    "plan_sha256": plan_sha256,
                    "stage": stage,
                }
            )
        ),
        error_code,
        stage,
        plan_sha256,
        False,
        False,
        False,
        False,
        False,
    )
    return WindowBridgeRefusal.from_dict(refusal.to_dict())
