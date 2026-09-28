# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import stat
from email.message import Message
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

import voxelscope.custody as base_custody_module
import voxelscope.milestone5_evidence as milestone5_module
import voxelscope.one_volume_custody as custody_module
import voxelscope.preprocessing as preprocessing_module
from voxelscope.canonical import EvidenceError, load_json, sha256_file, write_json
from voxelscope.cli import main
from voxelscope.custody import init_private_root
from voxelscope.custody_records import DigestIdentity
from voxelscope.milestone5_evidence import (
    _build_public_bundle,
    _refusal_summary,
    _validated_public_refusal,
    verify_milestone5_public_bundle,
)
from voxelscope.one_volume_contract import DECISION_SHA256, PLAN_SHA256
from voxelscope.one_volume_custody import (
    RECEIPT_PATH,
    SNAPSHOT_NAME,
    STRUCTURAL_REPORT_PATH,
    acquire_trusted_one_volume,
)
from voxelscope.one_volume_custody_records import StructuralReport
from voxelscope.one_volume_records import (
    ArtifactRole,
    OneVolumeAcquisitionPlan,
    PlannedArtifact,
)
from voxelscope.preprocessing import (
    INDEPENDENT_TENSOR_PATH,
    PREPROCESSING_REPORT_PATH,
    PREPROCESSING_SNAPSHOT_NAME,
    REFERENCE_TENSOR_PATH,
    compare_implementations,
    execute_preprocessing,
    load_adapter_plan,
    normalize_chunked_welford,
    normalize_reference,
    verify_preprocessing,
)
from voxelscope.preprocessing_records import AdapterPlan, PreprocessingRefusal

ROOT = Path(__file__).parents[1]
DECISION = ROOT / "research" / "one-volume-source-decision-v1.json"
CUSTODY_PLAN = ROOT / "research" / "one-volume-acquisition-plan-v1.json"
ADAPTER_PLAN = ROOT / "research" / "preprocessing-adapter-plan-v1.json"
COMMIT = "283007b96977b5b3285bd22cd67b7d97114b1eed"
ROLES: tuple[ArtifactRole, ...] = (
    "image-t1c",
    "image-t1",
    "image-t2",
    "image-flair",
    "label",
    "metadata-dataset-description",
    "metadata-readme",
    "metadata-changes",
)


class FakeResponse:
    status = 200

    def __init__(self, body: bytes, url: str) -> None:
        self._stream = io.BytesIO(body)
        self._url = url
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))

    def read(self, amount: int = -1) -> bytes:
        return self._stream.read(amount)

    def close(self) -> None:
        self._stream.close()

    def geturl(self) -> str:
        return self._url


def _compressed_nifti(
    data: np.ndarray[Any, Any],
    *,
    affine: np.ndarray[Any, Any] | None = None,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> bytes:
    affine = np.eye(4, dtype=np.float64) if affine is None else affine
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    image.header.set_slope_inter(slope, intercept)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    stream = io.BytesIO()
    file_map = image.make_file_map()
    file_map["image"].fileobj = stream
    image.to_file_map(file_map)
    return gzip.compress(stream.getvalue(), mtime=0)


def _planned_artifact(
    payload: bytes,
    role: ArtifactRole,
    *,
    index: int,
    destination: str,
) -> PlannedArtifact:
    blob = hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {len(payload)}\0".encode("ascii"))
    blob.update(payload)
    return PlannedArtifact(
        f"synthetic-{role}-{index}",
        role,
        (
            "https://raw.githubusercontent.com/OpenNeuroDatasets/"
            f"ds007045/{COMMIT}/synthetic-{index}"
        ),
        f"git-blob-sha1:{blob.hexdigest()}",
        ("https://raw.githubusercontent.com",),
        destination,
        len(payload),
        (
            DigestIdentity("git-blob-sha1", blob.hexdigest()),
            DigestIdentity("sha256", hashlib.sha256(payload).hexdigest()),
        ),
        role in {"image-t1c", "image-t1", "image-t2", "image-flair", "label"},
        role in {"image-t1c", "image-t1", "image-t2", "image-flair", "label"},
    )


def _synthetic_plan(
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> tuple[OneVolumeAcquisitionPlan, dict[str, bytes], dict[str, np.ndarray[Any, Any]]]:
    base = np.arange(30, dtype=np.int16).reshape((2, 3, 5))
    values = {
        "image-t1c": (base + np.where(base % 2 == 0, 0, 3)).astype(np.int16),
        "image-t1": (np.flip(base, axis=0) + 4).astype(np.int16),
        "image-t2": np.where(base % 3 == 0, 0, base * 2 + 1).astype(np.int16),
        "image-flair": (np.roll(base, 1, axis=2) + 2).astype(np.int16),
    }
    values["image-t1c"][0, 0, 0] = 0
    label = np.zeros((2, 3, 5), dtype=np.uint8)
    label[1, 1, 1] = 1
    metadata = {
        "metadata-dataset-description": json.dumps(
            {
                "BIDSVersion": "1.8.0",
                "DatasetDOI": "doi:10.18112/openneuro.ds007045.v2.0.1",
                "License": "CC0",
            },
            sort_keys=True,
        ).encode(),
        "metadata-readme": b"synthetic metadata only\n",
        "metadata-changes": b"synthetic changes only\n",
    }
    destinations = {
        "image-t1c": "artifacts/openneuro/case-a/z-last.nii.gz",
        "image-t1": "artifacts/openneuro/case-a/a-first.nii.gz",
        "image-t2": "artifacts/openneuro/case-a/m-middle.nii.gz",
        "image-flair": "artifacts/openneuro/case-a/b-second.nii.gz",
        "label": "artifacts/openneuro/case-a/c-label.nii.gz",
        "metadata-dataset-description": "artifacts/openneuro/metadata/dataset.json",
        "metadata-readme": "artifacts/openneuro/metadata/readme.txt",
        "metadata-changes": "artifacts/openneuro/metadata/changes.txt",
    }
    payloads: dict[str, bytes] = {}
    artifacts: list[PlannedArtifact] = []
    for index, role in enumerate(ROLES):
        if role == "label":
            payload = _compressed_nifti(label)
        elif role.startswith("image-"):
            payload = _compressed_nifti(
                values[role],
                slope=slope,
                intercept=intercept,
            )
        else:
            payload = metadata[role]
        artifact = _planned_artifact(
            payload,
            role,
            index=index,
            destination=destinations[role],
        )
        payloads[artifact.source_url] = payload
        artifacts.append(artifact)
    plan = OneVolumeAcquisitionPlan(
        "voxelscope/one-volume-acquisition-plan/v1",
        "synthetic-plan",
        "synthetic-decision",
        "openneuro-ds007045-v2.0.1",
        "disabled",
        "blocked",
        "synthetic approval required",
        ("synthetic structural gate",),
        tuple(artifacts),
    )
    return plan, payloads, values


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path / "custody"
    init_private_root(root, repository_root=ROOT)
    return root


def _acquired_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> tuple[Path, OneVolumeAcquisitionPlan, dict[str, np.ndarray[Any, Any]]]:
    plan, payloads, values = _synthetic_plan(slope=slope, intercept=intercept)
    runtime_identity = AdapterPlan.from_dict(load_json(ADAPTER_PLAN)).runtime_identity

    def trusted(decision_path: Path, plan_path: Path) -> tuple[OneVolumeAcquisitionPlan, str, str]:
        return plan, DECISION_SHA256, PLAN_SHA256

    monkeypatch.setattr(custody_module, "_load_trusted_plan", trusted)
    monkeypatch.setattr(preprocessing_module, "_load_trusted_plan", trusted)
    monkeypatch.setattr(
        preprocessing_module,
        "_runtime_identity",
        lambda repository_root: runtime_identity,
    )
    root = _private_root(tmp_path)
    acquire_trusted_one_volume(
        DECISION,
        CUSTODY_PLAN,
        root,
        approve_plan_sha256=PLAN_SHA256,
        allow_network=True,
        repository_root=ROOT,
        open_response=lambda url: FakeResponse(payloads[url], url),
    )
    return root, plan, values


def _execute(
    root: Path,
) -> tuple[str, Any]:
    plan_sha256 = sha256_file(ADAPTER_PLAN)
    receipt_sha256 = sha256_file(root / SNAPSHOT_NAME / RECEIPT_PATH)
    report = execute_preprocessing(
        DECISION,
        CUSTODY_PLAN,
        ADAPTER_PLAN,
        root,
        approve_plan_sha256=plan_sha256,
        approve_custody_receipt_sha256=receipt_sha256,
        repository_root=ROOT,
    )
    return receipt_sha256, report


def _reindex_public_bundle(root: Path) -> None:
    sums = "".join(
        f"{sha256_file(root / relative)}  {relative}\n"
        for relative in sorted(milestone5_module.PAYLOAD_PATHS)
    )
    (root / "SHA256SUMS").write_text(sums, encoding="ascii")
    index = {
        "artifacts": [
            {
                "path": relative,
                "sha256": sha256_file(root / relative),
                "size_bytes": (root / relative).stat().st_size,
            }
            for relative in sorted(milestone5_module.INDEXED_PATHS)
        ],
        "bundle_format": milestone5_module.PUBLIC_BUNDLE_FORMAT,
        "schema_version": milestone5_module.PUBLIC_EVIDENCE_SCHEMA,
    }
    write_json(root / "bundle.json", index)
    digest = sha256_file(root / "bundle.json")
    (root / "bundle.sha256").write_text(f"{digest}  bundle.json\n", encoding="ascii")


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 2, 2), dtype=np.float32),
        np.ones((2, 2, 2), dtype=np.float32),
        np.array([[[0.0, np.nan], [1.0, 2.0]]], dtype=np.float32),
        np.array([[[0.0, np.inf], [1.0, 2.0]]], dtype=np.float32),
    ],
)
def test_both_implementations_refuse_empty_zero_variance_and_nonfinite(
    values: np.ndarray[Any, Any],
) -> None:
    expected = (
        "empty_channel"
        if not np.count_nonzero(values)
        else (
            "zero_variance_channel"
            if np.isfinite(values).all() and np.unique(values[values != 0]).size == 1
            else "nonfinite_input"
        )
    )
    with pytest.raises(EvidenceError) as direct:
        normalize_reference(values)
    assert direct.value.code == expected
    output = np.empty(values.shape, dtype=np.float32)
    with pytest.raises(EvidenceError) as chunked:
        normalize_chunked_welford(values, output, chunk_depth=2)
    assert chunked.value.code == expected


@pytest.mark.parametrize("dtype", [np.dtype(">i2"), np.dtype("<i4"), np.dtype(">f4")])
def test_dtype_and_endian_conversion_is_float32_and_preserves_zero(dtype: np.dtype[Any]) -> None:
    values = np.array([0, 1, 2, 4, 8], dtype=dtype).reshape((1, 1, 5))
    normalized, count, _, _ = normalize_reference(values)
    assert normalized.dtype == np.dtype("<f4")
    assert count == 4
    assert normalized[0, 0, 0] == 0
    independent = np.empty(values.shape, dtype=np.float32)
    normalize_chunked_welford(values, independent, chunk_depth=2)
    compare_implementations(normalized, independent)


def test_chunk_boundaries_and_scaled_values_agree() -> None:
    values = np.arange(68, dtype=np.float32).reshape((2, 2, 17))
    values[:, :, ::5] = 0
    direct, _, _, _ = normalize_reference(values * np.float32(2.5) + np.float32(3.0))
    independent = np.empty(values.shape, dtype=np.float32)
    normalize_chunked_welford(
        values * np.float32(2.5) + np.float32(3.0),
        independent,
        chunk_depth=8,
    )
    compare_implementations(direct, independent)


def test_float32_conversion_overflow_is_refused() -> None:
    values = np.array([0.0, np.finfo(np.float64).max], dtype=np.float64).reshape((1, 1, 2))
    with pytest.raises(EvidenceError) as direct:
        normalize_reference(values)
    assert direct.value.code == "effective_dtype_overflow"
    with pytest.raises(EvidenceError) as chunked:
        normalize_chunked_welford(
            values,
            np.empty(values.shape, dtype=np.float32),
            chunk_depth=1,
        )
    assert chunked.value.code == "effective_dtype_overflow"


def test_deliberate_implementation_disagreement_is_refused() -> None:
    reference = np.zeros((1, 2, 2, 2), dtype=np.float32)
    independent = reference.copy()
    independent[0, 0, 0, 0] = np.float32(0.01)
    with pytest.raises(EvidenceError) as caught:
        compare_implementations(reference, independent)
    assert caught.value.code == "implementation_disagreement"


def _refusal_data() -> dict[str, Any]:
    return {
        "attempt_record_sha256": "a" * 64,
        "authorization_sha256": "b" * 64,
        "completed_channel_count": 0,
        "custody_receipt_sha256": "c" * 64,
        "error_code": "empty_channel",
        "inference_authorized": False,
        "inference_run": False,
        "last_completed_stage": "reference_normalization",
        "model_loaded": False,
        "plan_sha256": "d" * 64,
        "refused_at": "2026-09-28T00:00:00Z",
        "schema_version": "voxelscope/preprocessing-refusal/v1",
        "snapshot_published": False,
        "status": "refused",
    }


@pytest.mark.parametrize(
    ("stage", "count", "snapshot"),
    [
        ("attempt_record", 1, False),
        ("custody_verification", 1, False),
        ("channel_loading", 4, False),
        ("reference_normalization", 4, False),
        ("independent_normalization", 0, True),
        ("implementation_comparison", 3, False),
        ("implementation_comparison", 4, True),
        ("snapshot_publication", 3, False),
    ],
)
def test_refusal_rejects_impossible_stage_progress(
    stage: str,
    count: int,
    snapshot: bool,
) -> None:
    data = _refusal_data()
    data.update(
        {
            "completed_channel_count": count,
            "last_completed_stage": stage,
            "snapshot_published": snapshot,
        }
    )
    with pytest.raises(EvidenceError) as caught:
        PreprocessingRefusal.from_dict(data)
    assert caught.value.code == "invalid_refusal_progress"


def test_public_refusal_rejects_unknown_and_gate_moving_codes() -> None:
    data = _refusal_data()
    data["error_code"] = "invented_error"
    unknown = PreprocessingRefusal.from_dict(data)
    with pytest.raises(EvidenceError) as unpublishable:
        _validated_public_refusal(unknown)
    assert unpublishable.value.code == "unpublishable_refusal"

    data = _refusal_data()
    data.update(
        {
            "completed_channel_count": 4,
            "last_completed_stage": "implementation_comparison",
        }
    )
    moved = PreprocessingRefusal.from_dict(data)
    with pytest.raises(EvidenceError) as mismatch:
        _validated_public_refusal(moved)
    assert mismatch.value.code == "refusal_progress_mismatch"

    data = _refusal_data()
    data.update(
        {
            "completed_channel_count": 4,
            "last_completed_stage": "independent_normalization",
        }
    )
    impossible_empty = PreprocessingRefusal.from_dict(data)
    with pytest.raises(EvidenceError) as empty_mismatch:
        _validated_public_refusal(impossible_empty)
    assert empty_mismatch.value.code == "refusal_progress_mismatch"

    data = _refusal_data()
    data.update(
        {
            "completed_channel_count": 4,
            "error_code": "output_exists",
            "last_completed_stage": "snapshot_publication",
            "snapshot_published": True,
        }
    )
    impossible_output = PreprocessingRefusal.from_dict(data)
    with pytest.raises(EvidenceError) as output_mismatch:
        _validated_public_refusal(impossible_output)
    assert output_mismatch.value.code == "refusal_progress_mismatch"


def test_public_refusal_statuses_follow_verified_progress() -> None:
    channel = PreprocessingRefusal.from_dict(_refusal_data())
    phase, count = _validated_public_refusal(channel)
    channel_summary = _refusal_summary(phase, count)
    assert channel_summary["preprocessing_adapter_status"] == "no-go"
    assert channel_summary["finite_output_status"] == "not-reached"
    assert channel_summary["private_output_status"] == "not-reached"
    assert channel_summary["terminal_evidence_status"] == "go"

    comparison_data = _refusal_data()
    comparison_data.update(
        {
            "completed_channel_count": 4,
            "error_code": "implementation_disagreement",
            "last_completed_stage": "implementation_comparison",
        }
    )
    comparison = PreprocessingRefusal.from_dict(comparison_data)
    phase, count = _validated_public_refusal(comparison)
    comparison_summary = _refusal_summary(phase, count)
    assert comparison_summary["preprocessing_adapter_status"] == "no-go"
    assert comparison_summary["finite_output_status"] == "go"
    assert comparison_summary["geometry_preserved_status"] == "go"
    assert comparison_summary["label_excluded_status"] == "go"

    publication_data = _refusal_data()
    publication_data.update(
        {
            "completed_channel_count": 4,
            "error_code": "preprocessing_interrupted",
            "last_completed_stage": "snapshot_publication",
            "snapshot_published": True,
        }
    )
    publication = PreprocessingRefusal.from_dict(publication_data)
    phase, count = _validated_public_refusal(publication)
    publication_summary = _refusal_summary(phase, count)
    assert publication_summary["preprocessing_adapter_status"] == "go"
    assert publication_summary["private_output_status"] == "no-go"
    assert publication_summary["terminal_evidence_status"] == "go"


def test_plan_rejects_label_inclusion_and_spatial_transform() -> None:
    data = load_json(ADAPTER_PLAN)
    data["channel_order"] = ["image-t1c", "image-t1", "image-t2", "label"]
    with pytest.raises(EvidenceError) as label:
        AdapterPlan.from_dict(data)
    assert label.value.code == "unsafe_preprocessing_contract"
    data = load_json(ADAPTER_PLAN)
    data["spatial_transforms"] = ["Orientationd"]
    with pytest.raises(EvidenceError) as spatial:
        AdapterPlan.from_dict(data)
    assert spatial.value.code == "unsafe_preprocessing_contract"
    data = load_json(ADAPTER_PLAN)
    data["output_layout"] = "C,Z,Y,X"
    with pytest.raises(EvidenceError) as axes:
        AdapterPlan.from_dict(data)
    assert axes.value.code == "unsafe_preprocessing_contract"


def test_plan_tampering_and_symlink_are_refused(tmp_path: Path) -> None:
    data = load_json(ADAPTER_PLAN)
    data["comparison_atol"] = 0.1
    tampered = tmp_path / "plan.json"
    write_json(tampered, data)
    with pytest.raises(EvidenceError) as policy:
        load_adapter_plan(tampered, repository_root=ROOT)
    assert policy.value.code == "unsafe_comparison_policy"
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(ADAPTER_PLAN)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(EvidenceError) as unsafe:
        load_adapter_plan(linked, repository_root=ROOT)
    assert unsafe.value.code == "unsafe_path"


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_geometry_drift_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _, source_values = _acquired_root(monkeypatch, tmp_path)
    report = StructuralReport.from_dict(load_json(root / SNAPSHOT_NAME / STRUCTURAL_REPORT_PATH))
    structure = next(item for item in report.artifacts if item.role == "image-t1c")
    drifted = tmp_path / "drifted.nii.gz"
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 2.0
    drifted.write_bytes(_compressed_nifti(source_values["image-t1c"], affine=affine))
    with pytest.raises(EvidenceError) as caught:
        preprocessing_module._open_channel(drifted, "image-t1c", structure)
    assert caught.value.code == "source_geometry_drift"


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_role_order_beats_filename_order_and_scaling_matches_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _, source_values = _acquired_root(
        monkeypatch,
        tmp_path,
        slope=2.0,
        intercept=3.0,
    )
    receipt_sha256, report = _execute(root)
    assert report.channel_order == (
        "image-t1c",
        "image-t1",
        "image-t2",
        "image-flair",
    )
    assert report.label_excluded
    assert report.output_layout == "C,I,J,K"
    assert report.input_spatial_shape == (2, 3, 5)
    assert report.output_shape == (4, 2, 3, 5)
    tensor = np.memmap(
        root / PREPROCESSING_SNAPSHOT_NAME / REFERENCE_TENSOR_PATH,
        dtype="<f4",
        mode="r",
        shape=report.output_shape,
    )
    for index, role in enumerate(report.channel_order):
        effective = source_values[role].astype(np.float32) * np.float32(2.0) + np.float32(3.0)
        expected, _, _, _ = normalize_reference(effective)
        assert np.array_equal(tensor[index], expected)
    del tensor
    verified = verify_preprocessing(
        DECISION,
        CUSTODY_PLAN,
        ADAPTER_PLAN,
        root,
        approve_plan_sha256=sha256_file(ADAPTER_PLAN),
        approve_custody_receipt_sha256=receipt_sha256,
        repository_root=ROOT,
    )
    assert verified == report
    snapshot = root / PREPROCESSING_SNAPSHOT_NAME
    assert {
        path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()
    } == {
        PREPROCESSING_REPORT_PATH,
        REFERENCE_TENSOR_PATH,
        INDEPENDENT_TENSOR_PATH,
    }
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 for path in snapshot.rglob("*") if path.is_file()
    )


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_attempt_marker_precedes_voxel_read_and_blocks_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _, _ = _acquired_root(monkeypatch, tmp_path)
    receipt_sha256 = sha256_file(root / SNAPSHOT_NAME / RECEIPT_PATH)
    calls = 0

    def fail_before_voxel_processing(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise EvidenceError("synthetic_custody_failure", "synthetic")

    monkeypatch.setattr(
        preprocessing_module,
        "verify_trusted_one_volume",
        fail_before_voxel_processing,
    )
    arguments = {
        "approve_plan_sha256": sha256_file(ADAPTER_PLAN),
        "approve_custody_receipt_sha256": receipt_sha256,
        "repository_root": ROOT,
    }
    with pytest.raises(EvidenceError) as first:
        execute_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            **arguments,
        )
    assert first.value.code == "synthetic_custody_failure"
    started = list((root / "preprocessing-attempts").glob("*.started.json"))
    terminal = list((root / "preprocessing-attempts").glob("*.terminal.json"))
    assert len(started) == len(terminal) == 1
    assert load_json(terminal[0])["status"] == "refused"
    with pytest.raises(EvidenceError) as replay:
        execute_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            **arguments,
        )
    assert replay.value.code == "preprocessing_attempt_exists"
    assert calls == 1


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_input_replacement_after_custody_verification_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, plan, _ = _acquired_root(monkeypatch, tmp_path)
    receipt_sha256 = sha256_file(root / SNAPSHOT_NAME / RECEIPT_PATH)
    real_verify = preprocessing_module.verify_trusted_one_volume

    def verify_then_replace(*args: object, **kwargs: object) -> Any:
        result = real_verify(*args, **kwargs)
        artifact = next(item for item in plan.artifacts if item.role == "image-t1c")
        path = root / SNAPSHOT_NAME / artifact.destination
        payload = bytearray(path.read_bytes())
        payload[-9] ^= 1
        path.write_bytes(payload)
        path.chmod(0o600)
        return result

    monkeypatch.setattr(
        preprocessing_module,
        "verify_trusted_one_volume",
        verify_then_replace,
    )
    with pytest.raises(EvidenceError) as caught:
        execute_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            approve_plan_sha256=sha256_file(ADAPTER_PLAN),
            approve_custody_receipt_sha256=receipt_sha256,
            repository_root=ROOT,
        )
    assert caught.value.code == "custody_artifact_mismatch"
    terminal = next((root / "preprocessing-attempts").glob("*.terminal.json"))
    assert load_json(terminal)["status"] == "refused"


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_started_marker_publish_failure_still_records_terminal_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _, _ = _acquired_root(monkeypatch, tmp_path)
    receipt_sha256 = sha256_file(root / SNAPSHOT_NAME / RECEIPT_PATH)
    real_write = preprocessing_module._write_private_json_no_clobber

    def publish_then_fail(private_root: Path, relative: str, value: object) -> Path:
        result = real_write(private_root, relative, value)
        if relative.endswith(".started.json"):
            raise OSError("synthetic marker fsync failure")
        return result

    monkeypatch.setattr(
        preprocessing_module,
        "_write_private_json_no_clobber",
        publish_then_fail,
    )
    with pytest.raises(OSError):
        execute_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            approve_plan_sha256=sha256_file(ADAPTER_PLAN),
            approve_custody_receipt_sha256=receipt_sha256,
            repository_root=ROOT,
        )
    terminal = next((root / "preprocessing-attempts").glob("*.terminal.json"))
    assert load_json(terminal)["status"] == "refused"


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_completion_publish_failure_never_creates_conflicting_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _, _ = _acquired_root(monkeypatch, tmp_path)
    receipt_sha256 = sha256_file(root / SNAPSHOT_NAME / RECEIPT_PATH)
    real_write = preprocessing_module._write_private_json_no_clobber

    def publish_then_fail(private_root: Path, relative: str, value: object) -> Path:
        result = real_write(private_root, relative, value)
        if relative.endswith(".terminal.json") and isinstance(value, dict):
            if value.get("status") == "completed":
                raise OSError("synthetic completion fsync failure")
        return result

    monkeypatch.setattr(
        preprocessing_module,
        "_write_private_json_no_clobber",
        publish_then_fail,
    )
    report = execute_preprocessing(
        DECISION,
        CUSTODY_PLAN,
        ADAPTER_PLAN,
        root,
        approve_plan_sha256=sha256_file(ADAPTER_PLAN),
        approve_custody_receipt_sha256=receipt_sha256,
        repository_root=ROOT,
    )
    assert report.status == "go"
    terminals = list((root / "preprocessing-attempts").glob("*.terminal.json"))
    assert len(terminals) == 1
    assert load_json(terminals[0])["status"] == "completed"


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_tensor_tampering_extra_file_and_output_overwrite_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _, _ = _acquired_root(monkeypatch, tmp_path)
    receipt_sha256, _ = _execute(root)
    arguments = {
        "approve_plan_sha256": sha256_file(ADAPTER_PLAN),
        "approve_custody_receipt_sha256": receipt_sha256,
        "repository_root": ROOT,
    }
    with pytest.raises(EvidenceError) as replay:
        execute_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            **arguments,
        )
    assert replay.value.code == "preprocessing_attempt_exists"
    extra = root / PREPROCESSING_SNAPSHOT_NAME / "extra.txt"
    extra.write_text("synthetic", encoding="ascii")
    extra.chmod(0o600)
    with pytest.raises(EvidenceError) as closed:
        verify_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            **arguments,
        )
    assert closed.value.code == "preprocessing_snapshot_file_set_mismatch"
    extra.unlink()
    tensor = root / PREPROCESSING_SNAPSHOT_NAME / REFERENCE_TENSOR_PATH
    original = tensor.read_bytes()
    tensor.write_bytes(b"x" + original[1:])
    tensor.chmod(0o600)
    with pytest.raises(EvidenceError) as tampered:
        verify_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            **arguments,
        )
    assert tampered.value.code == "preprocessing_tensor_mismatch"


def test_windows_fails_closed_before_preprocessing_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_identity = AdapterPlan.from_dict(load_json(ADAPTER_PLAN)).runtime_identity
    monkeypatch.setattr(
        preprocessing_module,
        "_runtime_identity",
        lambda repository_root: runtime_identity,
    )
    monkeypatch.setattr(base_custody_module, "_is_windows", lambda: True)
    root = tmp_path / "custody"
    with pytest.raises(EvidenceError) as caught:
        execute_preprocessing(
            DECISION,
            CUSTODY_PLAN,
            ADAPTER_PLAN,
            root,
            approve_plan_sha256=sha256_file(ADAPTER_PLAN),
            approve_custody_receipt_sha256="0" * 64,
            repository_root=ROOT,
        )
    assert caught.value.code == "private_acl_unverified"
    assert not root.exists()


def test_preprocessing_cli_redacts_private_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_identity = AdapterPlan.from_dict(load_json(ADAPTER_PLAN)).runtime_identity
    monkeypatch.setattr(
        preprocessing_module,
        "_runtime_identity",
        lambda repository_root: runtime_identity,
    )
    private_root = tmp_path / "private-custody"
    expected_code = 2 if os.name == "nt" else 1
    assert (
        main(
            [
                "preprocess",
                "execute",
                "--record",
                str(DECISION),
                "--custody-plan",
                str(CUSTODY_PLAN),
                "--adapter-plan",
                str(ADAPTER_PLAN),
                "--root",
                str(private_root),
                "--approve-plan-sha256",
                sha256_file(ADAPTER_PLAN),
                "--approve-custody-receipt-sha256",
                "0" * 64,
            ]
        )
        == expected_code
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    expected_error = "ERROR private_acl_unverified\n" if os.name == "nt" else "ERROR io_error\n"
    assert captured.err == expected_error
    assert str(tmp_path) not in captured.err


def test_non_little_endian_runtime_is_refused_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preprocessing_module.sys, "byteorder", "big")
    with pytest.raises(EvidenceError) as caught:
        load_adapter_plan(ADAPTER_PLAN, repository_root=ROOT)
    assert caught.value.code == "untrusted_preprocessing_runtime"


def test_public_bundle_is_sanitized_and_privacy_scan_rejects_private_fields(
    tmp_path: Path,
) -> None:
    plan, plan_sha256 = load_adapter_plan(
        ADAPTER_PLAN,
        repository_root=ROOT,
        verify_runtime=False,
    )
    sources = tuple((item.path, item.sha256) for item in plan.implementation_sources)
    output = tmp_path / "public"
    bundle_sha256 = _build_public_bundle(
        output,
        outcome="go",
        adapter_plan_sha256=plan_sha256,
        implementation_sources=sources,
    )
    assert verify_milestone5_public_bundle(output) == bundle_sha256
    refusal_output = tmp_path / "public-refusal"
    refusal_sha256 = _build_public_bundle(
        refusal_output,
        outcome="no-go",
        adapter_plan_sha256=plan_sha256,
        implementation_sources=sources,
        refusal_phase="comparison",
        completed_channel_count=4,
    )
    assert verify_milestone5_public_bundle(refusal_output) == refusal_sha256
    combined = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    for forbidden in (
        b"sub-",
        b"input_spatial_shape",
        b"custody_receipt_sha256",
        b"standard_deviation",
        b"input-reference.f32le",
    ):
        assert forbidden not in combined
    summary = load_json(output / "public-summary-v1.json")
    summary["private_output_status"] = "no-go"
    write_json(output / "public-summary-v1.json", summary)
    _reindex_public_bundle(output)
    with pytest.raises(EvidenceError) as tampered:
        verify_milestone5_public_bundle(output, require_trusted_digest=False)
    assert tampered.value.code == "invalid_public_summary"
