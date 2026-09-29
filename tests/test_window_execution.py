# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import voxelscope.window_execution as execution_module
from voxelscope.canonical import (
    EvidenceError,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from voxelscope.preprocessing_records import (
    AdapterPlan,
    ChannelStatistics,
    ImplementationComparison,
    PreprocessingReport,
    TensorArtifact,
)
from voxelscope.window_execution import (
    INDEPENDENT_TENSOR_PATH,
    PREPROCESSING_REPORT_PATH,
    PREPROCESSING_SNAPSHOT_NAME,
    REFERENCE_TENSOR_PATH,
    WINDOW_LEDGER_PATH,
    WINDOW_REPORT_PATH,
    WINDOW_SNAPSHOT_NAME,
    _copy_immutable_input,
    axis_starts,
    enumerate_window_geometries,
    execute_window_qualification,
    load_window_execution_plan,
    materialize_direct,
    materialize_from_full_padding,
    materialize_full_padded_tensor,
    preprocessing_snapshot_identity,
    qualify_windows,
    symmetric_padding,
    verify_window_qualification,
)
from voxelscope.window_execution_contract import (
    LEGACY_SYNTHETIC_BUNDLE_SHA256,
    MILESTONE6_PLAN_SHA256,
    MILESTONE6_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_REPORT_SHA256,
)
from voxelscope.window_execution_records import (
    CHANNEL_ORDER,
    WindowExecutionLedger,
    WindowExecutionPlan,
    WindowExecutionRefusal,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/window-execution-plan-v1.json"
M5_PLAN = ROOT / "research/preprocessing-adapter-plan-v1.json"


@pytest.fixture(autouse=True)
def _trusted_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = WindowExecutionPlan.from_dict(load_json(PLAN))
    monkeypatch.setattr(
        execution_module,
        "_runtime_identity",
        lambda repository_root: plan.runtime_identity,
    )


def _values(shape: tuple[int, int, int]) -> np.ndarray[Any, np.dtype[np.float32]]:
    i, j, k = np.indices(shape, dtype=np.float32)
    base = i * np.float32(10000) + j * np.float32(100) + k + np.float32(1)
    return np.stack([base + np.float32(index * 1000000) for index in range(4)]).astype("<f4")


def _private_file(path: Path, value: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(value)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)
        os.chmod(path, 0o600)


def _synthetic_preprocessing_root(
    tmp_path: Path,
) -> tuple[Path, str, str]:
    root = tmp_path / "private-root"
    root.mkdir(mode=0o700)
    if os.name != "nt":
        os.chmod(root, 0o700)
    snapshot = root / PREPROCESSING_SNAPSHOT_NAME
    values = _values((5, 6, 7))
    tensor_bytes = values.tobytes(order="C")
    reference_path = snapshot / REFERENCE_TENSOR_PATH
    independent_path = snapshot / INDEPENDENT_TENSOR_PATH
    _private_file(reference_path, tensor_bytes)
    _private_file(independent_path, tensor_bytes)
    adapter_plan = AdapterPlan.from_dict(load_json(M5_PLAN))
    statistics = tuple(
        ChannelStatistics(
            role,
            str(index + 1) * 64,
            "<f4",
            "<f4",
            int(np.prod(values.shape[1:])),
            float(index + 1),
            1.0,
        )
        for index, role in enumerate(CHANNEL_ORDER)
    )
    artifact_size = len(tensor_bytes)
    report = PreprocessingReport(
        "voxelscope/preprocessing-report/v1",
        "go",
        sha256_file(M5_PLAN),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        CHANNEL_ORDER,
        True,
        "1" * 64,
        True,
        values.shape[1:],
        values.shape,
        "C,I,J,K",
        "<f4",
        "C",
        statistics,
        statistics,
        TensorArtifact(REFERENCE_TENSOR_PATH, sha256_file(reference_path), artifact_size),
        TensorArtifact(
            INDEPENDENT_TENSOR_PATH,
            sha256_file(independent_path),
            artifact_size,
        ),
        ImplementationComparison(True, True, 0.00002, 0.000002, 0.0, 0.0),
        True,
        True,
        True,
        adapter_plan.runtime_identity,
        adapter_plan.implementation_sources,
        adapter_plan.monai_sources,
        False,
        False,
        False,
        "2026-09-29T00:00:00Z",
    )
    report_path = snapshot / PREPROCESSING_REPORT_PATH
    write_json(report_path, report.to_dict())
    if os.name != "nt":
        os.chmod(report_path, 0o600)
        os.chmod(report_path.parent, 0o700)
        os.chmod(snapshot, 0o700)
        os.chmod(snapshot / "tensors", 0o700)
    report_sha256 = sha256_file(report_path)
    return root, report_sha256, preprocessing_snapshot_identity(report_sha256, report)


def _small_qualifier(
    source: np.ndarray[Any, np.dtype[np.float32]],
    *,
    roi_ijk: tuple[int, int, int],
    stride_ijk: tuple[int, int, int],
) -> WindowExecutionLedger:
    shape_ijk = tuple(int(item) for item in source.shape[1:])
    expected = execution_module._expected_ledger_geometry(
        shape_ijk,
        roi_ijk,
        stride_ijk,
    )
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
        expected.window_count,
        expected.coordinate_sha256,
        sha256_bytes(b"synthetic streamed bytes"),
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


def _execute_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, str, str]:
    root, report_sha256, snapshot_sha256 = _synthetic_preprocessing_root(tmp_path)
    monkeypatch.setattr(execution_module, "qualify_windows", _small_qualifier)
    execute_window_qualification(
        PLAN,
        root,
        approve_plan_sha256=sha256_file(PLAN),
        approve_preprocessing_report_sha256=report_sha256,
        approve_preprocessing_snapshot_sha256=snapshot_sha256,
        repository_root=ROOT,
    )
    return root, report_sha256, snapshot_sha256


def test_plan_binds_frozen_milestone_identities() -> None:
    plan, digest = load_window_execution_plan(PLAN, repository_root=ROOT)
    assert digest == sha256_file(PLAN)
    assert plan.milestone6_plan_sha256 == MILESTONE6_PLAN_SHA256
    assert plan.milestone6_report_sha256 == MILESTONE6_REPORT_SHA256
    assert plan.milestone6_public_bundle_sha256 == MILESTONE6_PUBLIC_BUNDLE_SHA256
    assert plan.legacy_synthetic_bundle_sha256 == LEGACY_SYNTHETIC_BUNDLE_SHA256
    assert plan.roi_ijk == (240, 240, 160)
    assert plan.stride_ijk == (120, 120, 80)


@pytest.mark.parametrize(
    ("shape", "roi", "expected"),
    [
        ((3, 4, 5), (6, 7, 8), ((1, 1, 1), (2, 2, 2))),
        ((4, 5, 6), (8, 9, 10), ((2, 2, 2), (2, 2, 2))),
        ((6, 7, 8), (6, 7, 8), ((0, 0, 0), (0, 0, 0))),
    ],
)
def test_symmetric_padding_handles_odd_even_and_exact_roi(
    shape: tuple[int, int, int],
    roi: tuple[int, int, int],
    expected: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> None:
    assert symmetric_padding(shape, roi) == expected


def test_final_anchor_and_k_fastest_traversal() -> None:
    assert axis_starts(18, 7, 3) == (0, 3, 6, 9, 11)
    windows = tuple(enumerate_window_geometries((8, 9, 10), (4, 5, 4), (2, 2, 2)))
    assert windows[0].padded_start_ijk == (0, 0, 0)
    assert windows[1].padded_start_ijk == (0, 0, 2)
    assert windows[-1].padded_stop_ijk == (8, 9, 10)
    assert [item.ordinal for item in windows] == list(range(len(windows)))


def test_independent_paths_preserve_channel_order_and_bytes() -> None:
    values = _values((3, 4, 5))
    roi = (6, 7, 8)
    before, _ = symmetric_padding(values.shape[1:], roi)
    padded = materialize_full_padded_tensor(values, roi)
    geometry = next(enumerate_window_geometries(values.shape[1:], roi, (3, 3, 4)))
    full = materialize_from_full_padding(padded, geometry)
    direct = materialize_direct(values, geometry, roi, before)
    assert full.dtype == np.dtype("<f4")
    assert full.tobytes(order="C") == direct.tobytes(order="C")
    for channel in range(4):
        assert np.array_equal(full[channel], direct[channel])
        assert np.any(full[channel] == np.float32(channel * 1000000 + 1))


def test_asymmetric_non_cubic_qualification_is_closed() -> None:
    ledger = qualify_windows(
        _values((7, 9, 13)),
        roi_ijk=(5, 6, 4),
        stride_ijk=(2, 3, 2),
    )
    assert ledger.exact_window_bytes_equal
    assert ledger.complete_source_coverage
    assert ledger.crop_geometry_verified
    assert ledger.axis_start_counts[2] > 1
    assert ledger.window_count == int(np.prod(ledger.axis_start_counts))


def test_deliberate_path_disagreement_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = execution_module.materialize_direct

    def disagree(*args: Any, **kwargs: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
        result = original(*args, **kwargs)
        result[0, 0, 0, 0] += np.float32(1)
        return result

    monkeypatch.setattr(execution_module, "materialize_direct", disagree)
    with pytest.raises(EvidenceError) as captured:
        qualify_windows(
            _values((3, 4, 5)),
            roi_ijk=(6, 7, 8),
            stride_ijk=(3, 3, 4),
        )
    assert captured.value.code == "window_materialization_disagreement"


def test_corrupt_source_intersection_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = execution_module.enumerate_window_geometries

    def corrupted(*args: Any, **kwargs: Any) -> Any:
        for geometry in original(*args, **kwargs):
            if geometry.ordinal == 0:
                geometry = replace(
                    geometry,
                    source_start_ijk=(99, 99, 99),
                    source_stop_ijk=(100, 100, 100),
                )
            yield geometry

    monkeypatch.setattr(execution_module, "enumerate_window_geometries", corrupted)
    with pytest.raises(EvidenceError) as captured:
        qualify_windows(
            _values((3, 4, 5)),
            roi_ijk=(6, 7, 8),
            stride_ijk=(3, 3, 4),
        )
    assert captured.value.code == "window_intersection_mismatch"


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (np.empty((4, 0, 1, 1), dtype="<f4"), "invalid_tensor_layout"),
        (np.zeros((4, 2, 2, 2), dtype="<f4"), "zero_channel"),
        (np.full((4, 2, 2, 2), np.nan, dtype="<f4"), "nonfinite_input"),
        (np.full((4, 2, 2, 2), np.inf, dtype="<f4"), "nonfinite_input"),
    ],
)
def test_empty_zero_and_nonfinite_inputs_are_refused(
    values: np.ndarray[Any, np.dtype[np.float32]],
    code: str,
) -> None:
    with pytest.raises(EvidenceError) as captured:
        qualify_windows(values, roi_ijk=(2, 2, 2), stride_ijk=(1, 1, 1))
    assert captured.value.code == code


def test_plan_tampering_is_refused(tmp_path: Path) -> None:
    candidate = tmp_path / "plan.json"
    data = load_json(PLAN)
    data["roi_ijk"] = [160, 240, 240]
    write_json(candidate, data)
    with pytest.raises(EvidenceError) as captured:
        load_window_execution_plan(candidate, repository_root=ROOT)
    assert captured.value.code == "window_execution_plan_digest_mismatch"


def test_runtime_drift_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = WindowExecutionPlan.from_dict(load_json(PLAN))
    monkeypatch.setattr(
        execution_module,
        "_runtime_identity",
        lambda repository_root: plan.runtime_identity.__class__(
            "3.13.15",
            "2.5.3",
            "little",
            "0" * 64,
        ),
    )
    with pytest.raises(EvidenceError) as captured:
        load_window_execution_plan(PLAN, repository_root=ROOT)
    assert captured.value.code == "untrusted_window_runtime"


def test_implementation_drift_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = execution_module.sha256_file

    def changed(path: Path) -> str:
        if path.name == "window_execution.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(execution_module, "sha256_file", changed)
    with pytest.raises(EvidenceError) as captured:
        load_window_execution_plan(PLAN, repository_root=ROOT)
    assert captured.value.code == "implementation_identity_mismatch"


@pytest.mark.skipif(os.name == "nt", reason="symlink permissions vary on Windows")
def test_plan_symlink_is_refused(tmp_path: Path) -> None:
    candidate = tmp_path / "plan.json"
    candidate.symlink_to(PLAN)
    with pytest.raises(EvidenceError) as captured:
        load_window_execution_plan(candidate, repository_root=ROOT)
    assert captured.value.code == "unsafe_path"


def test_source_replacement_during_staging_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"a" * (1024 * 1024 + 1))
    expected = sha256_file(source)
    original_read = execution_module.os.read
    replaced = False

    def mutating_read(descriptor: int, amount: int) -> bytes:
        nonlocal replaced
        result = original_read(descriptor, amount)
        if result and not replaced:
            replaced = True
            source.write_bytes(b"b" * (1024 * 1024 + 1))
        return result

    monkeypatch.setattr(execution_module.os, "read", mutating_read)
    with pytest.raises(EvidenceError) as captured:
        _copy_immutable_input(
            source,
            destination,
            expected_sha256=expected,
            expected_size=source.stat().st_size,
        )
    assert captured.value.code == "source_replacement"
    assert not destination.exists()


@pytest.mark.skipif(os.name == "nt", reason="private ACL policy intentionally refuses Windows")
def test_synthetic_private_execution_is_bounded_and_verifiable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, report_sha256, snapshot_sha256 = _execute_synthetic(monkeypatch, tmp_path)
    verified = verify_window_qualification(
        PLAN,
        root,
        approve_plan_sha256=sha256_file(PLAN),
        approve_preprocessing_report_sha256=report_sha256,
        approve_preprocessing_snapshot_sha256=snapshot_sha256,
        repository_root=ROOT,
    )
    output = root / WINDOW_SNAPSHOT_NAME
    assert verified.status == "go"
    assert {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    } == {WINDOW_LEDGER_PATH, WINDOW_REPORT_PATH}
    assert not any(path.suffix == ".f32le" for path in output.rglob("*"))


@pytest.mark.skipif(os.name == "nt", reason="private ACL policy intentionally refuses Windows")
def test_staged_input_is_unlinked_and_read_only_during_qualification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, report_sha256, snapshot_sha256 = _synthetic_preprocessing_root(tmp_path)

    def inspect(
        source: np.ndarray[Any, np.dtype[np.float32]],
        *,
        roi_ijk: tuple[int, int, int],
        stride_ijk: tuple[int, int, int],
    ) -> WindowExecutionLedger:
        assert not source.flags.writeable
        with pytest.raises(ValueError):
            source[0, 0, 0, 0] = np.float32(0)
        assert not list(root.glob(".window-stage-*/.input/reference.f32le"))
        return _small_qualifier(
            source,
            roi_ijk=roi_ijk,
            stride_ijk=stride_ijk,
        )

    monkeypatch.setattr(execution_module, "qualify_windows", inspect)
    execute_window_qualification(
        PLAN,
        root,
        approve_plan_sha256=sha256_file(PLAN),
        approve_preprocessing_report_sha256=report_sha256,
        approve_preprocessing_snapshot_sha256=snapshot_sha256,
        repository_root=ROOT,
    )


@pytest.mark.skipif(os.name == "nt", reason="private ACL policy intentionally refuses Windows")
def test_verifier_refuses_authenticated_wrong_roi_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, report_sha256, snapshot_sha256 = _execute_synthetic(monkeypatch, tmp_path)
    output = root / WINDOW_SNAPSHOT_NAME
    ledger_path = output / WINDOW_LEDGER_PATH
    report_path = output / WINDOW_REPORT_PATH
    terminal_path = next((root / "window-execution-attempts").glob("*.terminal.json"))
    shape = (5, 6, 7)
    wrong = qualify_windows(
        _values(shape),
        roi_ijk=(7, 8, 9),
        stride_ijk=(3, 4, 4),
    )
    write_json(ledger_path, wrong.to_dict())
    report = load_json(report_path)
    report["ledger"] = {
        "path": WINDOW_LEDGER_PATH,
        "sha256": sha256_file(ledger_path),
        "size_bytes": ledger_path.stat().st_size,
    }
    write_json(report_path, report)
    completion = load_json(terminal_path)
    completion["ledger_sha256"] = sha256_file(ledger_path)
    completion["report_sha256"] = sha256_file(report_path)
    write_json(terminal_path, completion)
    with pytest.raises(EvidenceError) as captured:
        verify_window_qualification(
            PLAN,
            root,
            approve_plan_sha256=sha256_file(PLAN),
            approve_preprocessing_report_sha256=report_sha256,
            approve_preprocessing_snapshot_sha256=snapshot_sha256,
            repository_root=ROOT,
        )
    assert captured.value.code == "window_geometry_evidence_mismatch"


@pytest.mark.skipif(os.name == "nt", reason="private ACL policy intentionally refuses Windows")
def test_replay_and_overwrite_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, report_sha256, snapshot_sha256 = _execute_synthetic(monkeypatch, tmp_path)
    with pytest.raises(EvidenceError) as captured:
        execute_window_qualification(
            PLAN,
            root,
            approve_plan_sha256=sha256_file(PLAN),
            approve_preprocessing_report_sha256=report_sha256,
            approve_preprocessing_snapshot_sha256=snapshot_sha256,
            repository_root=ROOT,
        )
    assert captured.value.code == "window_execution_attempt_exists"
    assert (root / WINDOW_SNAPSHOT_NAME).is_dir()


@pytest.mark.skipif(os.name == "nt", reason="private ACL policy intentionally refuses Windows")
def test_output_tamper_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, report_sha256, snapshot_sha256 = _execute_synthetic(monkeypatch, tmp_path)
    ledger_path = root / WINDOW_SNAPSHOT_NAME / WINDOW_LEDGER_PATH
    ledger_path.write_bytes(ledger_path.read_bytes() + b" ")
    with pytest.raises(EvidenceError):
        verify_window_qualification(
            PLAN,
            root,
            approve_plan_sha256=sha256_file(PLAN),
            approve_preprocessing_report_sha256=report_sha256,
            approve_preprocessing_snapshot_sha256=snapshot_sha256,
            repository_root=ROOT,
        )


def test_terminal_fault_is_not_success_shaped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("private ACL policy intentionally refuses Windows")
    root, report_sha256, snapshot_sha256 = _synthetic_preprocessing_root(tmp_path)
    monkeypatch.setattr(execution_module, "qualify_windows", _small_qualifier)
    original = execution_module._write_terminal

    def fail_terminal(root: Path, attempt_id: str, value: dict[str, Any]) -> None:
        if value.get("status") == "completed":
            raise OSError("synthetic terminal failure")
        original(root, attempt_id, value)

    monkeypatch.setattr(execution_module, "_write_terminal", fail_terminal)
    with pytest.raises(OSError):
        execute_window_qualification(
            PLAN,
            root,
            approve_plan_sha256=sha256_file(PLAN),
            approve_preprocessing_report_sha256=report_sha256,
            approve_preprocessing_snapshot_sha256=snapshot_sha256,
            repository_root=ROOT,
        )
    terminals = list((root / "window-execution-attempts").glob("*.terminal.json"))
    assert len(terminals) == 1
    assert load_json(terminals[0])["status"] == "refused"


def test_windows_private_acl_policy_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr("voxelscope.custody._is_windows", lambda: True)
    with pytest.raises(EvidenceError) as captured:
        execution_module.require_private_root(root, repository_root=ROOT)
    assert captured.value.code == "private_acl_unverified"


@pytest.mark.parametrize(
    ("stage", "count", "published"),
    [
        ("attempt_marker", 1, False),
        ("input_validation", 0, True),
        ("window_materialization", 2, True),
    ],
)
def test_refusal_rejects_impossible_progress(
    stage: Any,
    count: int,
    published: bool,
) -> None:
    with pytest.raises(EvidenceError) as captured:
        WindowExecutionRefusal(
            "voxelscope/window-execution-refusal/v1",
            "refused",
            "synthetic_failure",
            stage,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "e" * 64,
            count,
            published,
            False,
            False,
            False,
            "2026-09-29T00:00:00Z",
        )
    assert captured.value.code == "invalid_refusal_progress"
