# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import stat
import struct
from email.message import Message
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

import voxelscope.milestone4_evidence as milestone4_module
import voxelscope.one_volume_custody as custody_module
from voxelscope.canonical import EvidenceError, load_json, sha256_file, write_json
from voxelscope.cli import main
from voxelscope.custody import init_private_root
from voxelscope.custody_records import DigestIdentity
from voxelscope.milestone4_evidence import (
    PAYLOAD_PATHS,
    _build_milestone4_public_bundle,
    build_milestone4_public_bundle,
    build_milestone4_refusal_public_bundle,
    verify_milestone4_public_bundle,
)
from voxelscope.one_volume_contract import DECISION_SHA256, PLAN_SHA256
from voxelscope.one_volume_custody import (
    RECEIPT_PATH,
    SNAPSHOT_NAME,
    STRUCTURAL_REPORT_PATH,
    _build_structural_report,
    _download_artifact,
    _inspect_nifti,
    acquire_trusted_one_volume,
    verify_trusted_one_volume,
    verify_trusted_one_volume_refusal,
)
from voxelscope.one_volume_custody_records import (
    AcquisitionRefusal,
    CustodyReceipt,
    StructuralReport,
)
from voxelscope.one_volume_records import (
    ArtifactRole,
    OneVolumeAcquisitionPlan,
    PlannedArtifact,
)

ROOT = Path(__file__).parents[1]
DECISION = ROOT / "research" / "one-volume-source-decision-v1.json"
PLAN = ROOT / "research" / "one-volume-acquisition-plan-v1.json"
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

    def __init__(
        self,
        body: bytes,
        url: str,
        *,
        etag: str | None = None,
        version: str | None = None,
        final_url: str | None = None,
    ) -> None:
        self._stream = io.BytesIO(body)
        self._url = final_url or url
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))
        if etag is not None:
            self.headers["ETag"] = f'"{etag}"'
        if version is not None:
            self.headers["x-amz-version-id"] = version

    def read(self, amount: int = -1) -> bytes:
        return self._stream.read(amount)

    def close(self) -> None:
        self._stream.close()

    def geturl(self) -> str:
        return self._url


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path / "custody"
    init_private_root(root, repository_root=ROOT)
    return root


def _raw_nifti(
    data: np.ndarray[Any, Any],
    *,
    affine: np.ndarray[Any, Any] | None = None,
    version: int = 1,
) -> bytes:
    affine = np.eye(4, dtype=np.float64) if affine is None else affine
    image_type = nib.Nifti1Image if version == 1 else nib.Nifti2Image
    image = image_type(data, affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    stream = io.BytesIO()
    file_map = image.make_file_map()
    file_map["image"].fileobj = stream
    image.to_file_map(file_map)
    return stream.getvalue()


def _compressed_nifti(
    data: np.ndarray[Any, Any],
    *,
    affine: np.ndarray[Any, Any] | None = None,
    version: int = 1,
) -> bytes:
    return gzip.compress(_raw_nifti(data, affine=affine, version=version), mtime=0)


def _planned_artifact(
    payload: bytes,
    role: ArtifactRole = "image-t1c",
    *,
    index: int = 0,
    destination: str | None = None,
) -> PlannedArtifact:
    blob = hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {len(payload)}\0".encode("ascii"))
    blob.update(payload)
    return PlannedArtifact(
        artifact_id=f"synthetic-{role}-{index}",
        role=role,
        source_url=(
            "https://raw.githubusercontent.com/OpenNeuroDatasets/"
            f"ds007045/{COMMIT}/synthetic-{index}"
        ),
        immutable_id=f"git-blob-sha1:{blob.hexdigest()}",
        allowed_origins=("https://raw.githubusercontent.com",),
        destination=destination or f"artifacts/synthetic/{role}-{index}.bin",
        expected_size_bytes=len(payload),
        expected_hashes=(
            DigestIdentity("git-blob-sha1", blob.hexdigest()),
            DigestIdentity("sha256", hashlib.sha256(payload).hexdigest()),
        ),
        medical_data=role in {"image-t1c", "image-t1", "image-t2", "image-flair", "label"},
        operator_approval_required=role
        in {"image-t1c", "image-t1", "image-t2", "image-flair", "label"},
    )


def _inspect_payload(
    tmp_path: Path,
    payload: bytes,
    *,
    role: ArtifactRole = "image-t1c",
) -> Any:
    source = tmp_path / "input.nii.gz"
    source.write_bytes(payload)
    source.chmod(0o600)
    work = tmp_path / "inspection"
    work.mkdir(mode=0o700)
    return _inspect_nifti(_planned_artifact(payload, role), source, work)


def _write_medical_set(
    root: Path,
    plan: OneVolumeAcquisitionPlan,
    *,
    affine_by_role: dict[str, np.ndarray[Any, Any]] | None = None,
    dtype_by_role: dict[str, np.dtype[Any]] | None = None,
    values_by_role: dict[str, np.ndarray[Any, Any]] | None = None,
) -> None:
    affine_by_role = affine_by_role or {}
    dtype_by_role = dtype_by_role or {}
    values_by_role = values_by_role or {}
    for artifact in plan.artifacts:
        if artifact.role not in {"image-t1c", "image-t1", "image-t2", "image-flair", "label"}:
            continue
        if artifact.role == "label":
            default = np.array(
                [
                    [[0, 0], [1, 1]],
                    [[0, 0], [1, 1]],
                ],
                dtype=np.uint8,
            )
        else:
            default = np.ones((2, 2, 2), dtype=np.float32)
        values = values_by_role.get(artifact.role, default)
        dtype = dtype_by_role.get(artifact.role)
        if dtype is not None:
            values = values.astype(dtype)
        payload = _compressed_nifti(
            values,
            affine=affine_by_role.get(artifact.role),
        )
        path = root / artifact.destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o600)


def _synthetic_plan() -> tuple[OneVolumeAcquisitionPlan, dict[str, bytes]]:
    image = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    mask = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=np.uint8)
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
    payloads: dict[str, bytes] = {}
    artifacts: list[PlannedArtifact] = []
    for index, role in enumerate(ROLES):
        if role == "label":
            payload = _compressed_nifti(mask)
            destination = "artifacts/openneuro/case-a/tumor-mask.nii.gz"
        elif role.startswith("image-"):
            payload = _compressed_nifti(image)
            destination = f"artifacts/openneuro/case-a/{role}.nii.gz"
        else:
            payload = metadata[role]
            destination = f"artifacts/openneuro/metadata/{role}.txt"
        artifact = _planned_artifact(payload, role, index=index, destination=destination)
        payloads[artifact.source_url] = payload
        artifacts.append(artifact)
    return (
        OneVolumeAcquisitionPlan(
            "voxelscope/one-volume-acquisition-plan/v1",
            "synthetic-plan",
            "synthetic-decision",
            "openneuro-ds007045-v2.0.1",
            "disabled",
            "blocked",
            "synthetic approval required",
            ("synthetic structural gate",),
            tuple(artifacts),
        ),
        payloads,
    )


def _real_plan(monkeypatch: pytest.MonkeyPatch) -> OneVolumeAcquisitionPlan:
    plan = custody_module._load_trusted_plan(DECISION, PLAN)[0]
    monkeypatch.setattr(
        custody_module,
        "_load_trusted_plan",
        lambda decision_path, plan_path: (plan, DECISION_SHA256, PLAN_SHA256),
    )
    return plan


def _refusal_data() -> dict[str, Any]:
    return {
        "approval_sha256": hashlib.sha256(PLAN_SHA256.encode("ascii")).hexdigest(),
        "attempt_record_sha256": "a" * 64,
        "authorization_sha256": "b" * 64,
        "decision_sha256": DECISION_SHA256,
        "error_code": "download_failed",
        "geometry_comparison_count": 0,
        "inference_authorized": False,
        "last_completed_stage": "attempt_record",
        "model_loaded": False,
        "plan_sha256": PLAN_SHA256,
        "refused_at": "2026-09-27T00:00:00Z",
        "schema_version": "voxelscope/one-volume-acquisition-refusal/v1",
        "status": "refused",
        "validated_medical_file_count": 0,
        "verified_artifact_count": 0,
    }


def test_identity_scaling_is_valid_and_streamed(tmp_path: Path) -> None:
    result = _inspect_payload(tmp_path, _compressed_nifti(np.ones((2, 3, 4), np.float32)))
    assert result.shape == (2, 3, 4)
    assert result.scaling_slope == 1.0
    assert result.scaling_intercept == 0.0
    assert result.nonzero_voxel_count == 24


def test_nifti2_single_file_is_supported(tmp_path: Path) -> None:
    result = _inspect_payload(
        tmp_path,
        _compressed_nifti(np.ones((2, 3, 4), np.float32), version=2),
    )
    assert result.nifti_version == 2


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"not gzip", "invalid_gzip"),
        (
            gzip.compress(_raw_nifti(np.ones((2, 2, 2), np.float32))[:-1], mtime=0),
            "nifti_extent_mismatch",
        ),
        (
            gzip.compress(_raw_nifti(np.ones((2, 2, 2), np.float32)) + b"x", mtime=0),
            "nifti_extent_mismatch",
        ),
        (gzip.compress(b"\0" * 352, mtime=0), "invalid_nifti_header"),
        (
            _compressed_nifti(np.ones((2, 2, 2), np.float32)) + gzip.compress(b"", mtime=0),
            "gzip_trailing_member",
        ),
        (
            _compressed_nifti(np.ones((513, 1, 1), np.float32)),
            "invalid_nifti_shape",
        ),
    ],
)
def test_nifti_validator_rejects_malformed_or_unbounded_inputs(
    tmp_path: Path, payload: bytes, code: str
) -> None:
    with pytest.raises(EvidenceError) as caught:
        _inspect_payload(tmp_path, payload)
    assert caught.value.code == code


def test_nifti_validator_rejects_extensions(tmp_path: Path) -> None:
    raw = bytearray(_raw_nifti(np.ones((2, 2, 2), np.float32)))
    raw[348] = 1
    with pytest.raises(EvidenceError) as caught:
        _inspect_payload(tmp_path, gzip.compress(raw, mtime=0))
    assert caught.value.code == "nifti_extensions_forbidden"


def test_nifti_validator_enforces_decompression_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(custody_module, "MAX_NIFTI_DECOMPRESSED_BYTES", 64)
    with pytest.raises(EvidenceError) as caught:
        _inspect_payload(tmp_path, _compressed_nifti(np.ones((2, 2, 2), np.float32)))
    assert caught.value.code == "nifti_decompression_limit"


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nifti_validator_rejects_nonfinite_voxels(tmp_path: Path, bad_value: float) -> None:
    values = np.ones((2, 2, 2), dtype=np.float32)
    values[0, 0, 0] = bad_value
    with pytest.raises(EvidenceError) as caught:
        _inspect_payload(tmp_path, _compressed_nifti(values))
    assert caught.value.code == "nonfinite_voxel"


def test_nifti_validator_rejects_nonfinite_affine_header(tmp_path: Path) -> None:
    raw = bytearray(_raw_nifti(np.ones((2, 2, 2), np.float32)))
    struct.pack_into("<f", raw, 280, float("nan"))
    with pytest.raises(EvidenceError) as caught:
        _inspect_payload(tmp_path, gzip.compress(raw, mtime=0))
    assert caught.value.code in {"invalid_affine", "nifti_header_diagnostic", "invalid_nifti"}


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (np.array([[[0, 5], [1, 1]]], dtype=np.uint8), "unexpected_mask_label"),
        (np.zeros((1, 2, 2), dtype=np.uint8), "missing_expected_mask_label"),
        (np.ones((1, 2, 2), dtype=np.uint8), "missing_expected_mask_label"),
        (
            np.array([[[0.0, 1.5], [1.0, 1.0]]], dtype=np.float32),
            "noninteger_mask",
        ),
    ],
)
def test_mask_contract_rejects_invalid_values(
    tmp_path: Path, values: np.ndarray[Any, Any], code: str
) -> None:
    with pytest.raises(EvidenceError) as caught:
        _inspect_payload(tmp_path, _compressed_nifti(values), role="label")
    assert caught.value.code == code


def test_mask_contract_allows_missing_individual_subregions(tmp_path: Path) -> None:
    values = np.array([[[0, 0], [2, 2]]], dtype=np.uint8)
    result = _inspect_payload(tmp_path, _compressed_nifti(values), role="label")
    assert result.mask_labels == (0, 2)


def test_nifti_validator_rejects_symlinked_input(tmp_path: Path) -> None:
    target = tmp_path / "target.nii.gz"
    target.write_bytes(_compressed_nifti(np.ones((2, 2, 2), np.float32)))
    source = tmp_path / "source.nii.gz"
    try:
        source.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    work = tmp_path / "inspection"
    work.mkdir()
    with pytest.raises(EvidenceError) as caught:
        _inspect_nifti(_planned_artifact(target.read_bytes()), source, work)
    assert caught.value.code == "unsafe_path"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("affine", "nifti_geometry_mismatch"),
        ("dtype", "nifti_dtype_mismatch"),
        ("nan", "nonfinite_voxel"),
    ],
)
def test_structural_report_rejects_cross_file_mismatch(
    tmp_path: Path, mutation: str, code: str
) -> None:
    plan, _ = _synthetic_plan()
    affine_by_role: dict[str, np.ndarray[Any, Any]] = {}
    dtype_by_role: dict[str, np.dtype[Any]] = {}
    values_by_role: dict[str, np.ndarray[Any, Any]] = {}
    if mutation == "affine":
        affine = np.eye(4, dtype=np.float64)
        affine[0, 3] = 0.000001
        affine_by_role["image-t1"] = affine
    elif mutation == "dtype":
        dtype_by_role["image-t1"] = np.dtype(np.float64)
    else:
        values = np.ones((2, 2, 2), dtype=np.float32)
        values[0, 0, 0] = np.nan
        values_by_role["image-t1"] = values
    _write_medical_set(
        tmp_path,
        plan,
        affine_by_role=affine_by_role,
        dtype_by_role=dtype_by_role,
        values_by_role=values_by_role,
    )
    inspection = tmp_path / "inspection"
    inspection.mkdir()
    with pytest.raises(EvidenceError) as caught:
        _build_structural_report(
            tmp_path,
            plan,
            decision_sha256=DECISION_SHA256,
            plan_sha256=PLAN_SHA256,
            authorization_sha256="a" * 64,
            inspection_root=inspection,
        )
    assert caught.value.code == code


def _s3_artifact(body: bytes) -> PlannedArtifact:
    version = "fixed-version"
    etag = hashlib.md5(body, usedforsecurity=False).hexdigest()
    return PlannedArtifact(
        "synthetic-s3",
        "image-t1c",
        f"https://s3.amazonaws.com/openneuro.org/ds007045/synthetic?versionId={version}",
        f"s3-version-id:{version}",
        ("https://s3.amazonaws.com",),
        "artifacts/synthetic.nii.gz",
        len(body),
        (
            DigestIdentity("etag", etag),
            DigestIdentity("sha256", hashlib.sha256(body).hexdigest()),
        ),
        True,
        True,
    )


@pytest.mark.parametrize("version", [None, "wrong-version"])
def test_transport_rejects_missing_or_wrong_s3_version(tmp_path: Path, version: str | None) -> None:
    body = b"synthetic"
    artifact = _s3_artifact(body)
    etag = next(item.value for item in artifact.expected_hashes if item.algorithm == "etag")
    response = FakeResponse(body, artifact.source_url, etag=etag, version=version)
    with pytest.raises(EvidenceError) as caught:
        _download_artifact(artifact, tmp_path, open_response=lambda url: response)
    assert caught.value.code == "artifact_version_mismatch"


@pytest.mark.parametrize("etag", [None, "0" * 32])
def test_transport_rejects_missing_or_wrong_etag(tmp_path: Path, etag: str | None) -> None:
    body = b"synthetic"
    artifact = _s3_artifact(body)
    response = FakeResponse(
        body,
        artifact.source_url,
        etag=etag,
        version="fixed-version",
    )
    with pytest.raises(EvidenceError) as caught:
        _download_artifact(artifact, tmp_path, open_response=lambda url: response)
    assert caught.value.code == "artifact_etag_mismatch"


def test_etag_match_never_qualifies_wrong_content(tmp_path: Path) -> None:
    expected = b"expected"
    artifact = _s3_artifact(expected)
    etag = next(item.value for item in artifact.expected_hashes if item.algorithm == "etag")
    wrong = b"wrong!!!"
    response = FakeResponse(
        wrong,
        artifact.source_url,
        etag=etag,
        version="fixed-version",
    )
    response.headers["Content-Length"] = str(len(expected))
    with pytest.raises(EvidenceError) as caught:
        _download_artifact(artifact, tmp_path, open_response=lambda url: response)
    assert caught.value.code == "artifact_hash_mismatch"


def test_transport_rejects_duplicate_version_query(tmp_path: Path) -> None:
    artifact = _s3_artifact(b"synthetic")
    object.__setattr__(
        artifact,
        "source_url",
        artifact.source_url + "&versionId=fixed-version",
    )
    with pytest.raises(EvidenceError) as caught:
        _download_artifact(
            artifact,
            tmp_path,
            open_response=lambda url: pytest.fail("opener must not be called"),
        )
    assert caught.value.code == "unsafe_version_pin"


def test_transport_rejects_changed_final_url(tmp_path: Path) -> None:
    body = b"synthetic"
    artifact = _s3_artifact(body)
    etag = next(item.value for item in artifact.expected_hashes if item.algorithm == "etag")
    response = FakeResponse(
        body,
        artifact.source_url,
        etag=etag,
        version="fixed-version",
        final_url="https://s3.amazonaws.com/openneuro.org/ds007045/other",
    )
    with pytest.raises(EvidenceError) as caught:
        _download_artifact(artifact, tmp_path, open_response=lambda url: response)
    assert caught.value.code == "unexpected_http_response"


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_wrong_plan_approval_refuses_before_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _real_plan(monkeypatch)
    root = _private_root(tmp_path)
    calls = 0

    def opener(url: str) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(b"", url)

    with pytest.raises(EvidenceError) as caught:
        acquire_trusted_one_volume(
            DECISION,
            PLAN,
            root,
            approve_plan_sha256="0" * 64,
            allow_network=True,
            repository_root=ROOT,
            open_response=opener,
        )
    assert caught.value.code == "wrong_plan_approval"
    assert calls == 0
    assert not (root / "attempts").exists()


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_one_volume_cli_redacts_private_path_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _private_root(tmp_path)
    assert (
        main(
            [
                "one-volume-custody",
                "acquire",
                "--record",
                str(DECISION),
                "--plan",
                str(PLAN),
                "--root",
                str(root),
                "--approve-plan-sha256",
                "0" * 64,
                "--allow-network",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR wrong_plan_approval\n"
    assert str(root) not in captured.err


def test_one_volume_cli_redacts_private_path_on_io_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_record = tmp_path / "private" / "missing-record.json"
    assert (
        main(
            [
                "one-volume-custody",
                "acquire",
                "--record",
                str(missing_record),
                "--plan",
                str(PLAN),
                "--root",
                str(tmp_path / "custody"),
                "--approve-plan-sha256",
                PLAN_SHA256,
                "--allow-network",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR unsafe_path\n"
    assert str(tmp_path) not in captured.err


@pytest.mark.skipif(os.name == "nt", reason="private custody fails closed on Windows")
def test_attempt_marker_consumes_approval_before_first_request_and_blocks_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = _real_plan(monkeypatch)
    root = _private_root(tmp_path)
    calls = 0

    def opener(url: str) -> FakeResponse:
        nonlocal calls
        calls += 1
        artifact = plan.artifacts[0]
        return FakeResponse(
            b"x" * artifact.expected_size_bytes,
            url,
            etag="0" * 32,
            version=artifact.immutable_id.removeprefix("s3-version-id:"),
        )

    arguments = {
        "approve_plan_sha256": PLAN_SHA256,
        "allow_network": True,
        "repository_root": ROOT,
        "open_response": opener,
    }
    with pytest.raises(EvidenceError) as caught:
        acquire_trusted_one_volume(DECISION, PLAN, root, **arguments)
    assert caught.value.code == "artifact_etag_mismatch"
    marker = root / "attempts" / f"{PLAN_SHA256}.started.json"
    refusal = root / "attempts" / f"{PLAN_SHA256}.refusal.json"
    assert marker.is_file() and refusal.is_file()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    refusal_record = verify_trusted_one_volume_refusal(
        DECISION,
        PLAN,
        root,
        approve_plan_sha256=PLAN_SHA256,
        repository_root=ROOT,
    )
    assert refusal_record.error_code == "artifact_etag_mismatch"
    assert refusal_record.verified_artifact_count == 0
    public = tmp_path / "public-refusal"
    build_milestone4_refusal_public_bundle(
        DECISION,
        PLAN,
        root,
        public,
        approve_plan_sha256=PLAN_SHA256,
        repository_root=ROOT,
    )
    verify_milestone4_public_bundle(public, require_trusted_digest=False)
    ledger = load_json(public / "stage-ledger.json")
    assert ledger["attempt_terminal_status"] == "refused"
    assert ledger["acquisition_status"] == "no-go"
    assert ledger["content_identity_status"] == "not_reached"
    assert ledger["structural_validation_status"] == "not_reached"
    with pytest.raises(EvidenceError) as replay:
        acquire_trusted_one_volume(DECISION, PLAN, root, **arguments)
    assert replay.value.code == "acquisition_attempt_exists"
    assert calls == 1


def test_refusal_schema_rejects_unknown_stage() -> None:
    data = _refusal_data()
    data["last_completed_stage"] = "arbitrary"
    with pytest.raises(EvidenceError) as caught:
        AcquisitionRefusal.from_dict(data)
    assert caught.value.code == "invalid_refusal_stage"


def test_refusal_schema_rejects_impossible_progress() -> None:
    data = _refusal_data()
    data["verified_artifact_count"] = 8
    with pytest.raises(EvidenceError) as caught:
        AcquisitionRefusal.from_dict(data)
    assert caught.value.code == "invalid_refusal_progress"


def test_public_refusal_rejects_unknown_error_code() -> None:
    data = _refusal_data()
    data["error_code"] = "invented_error"
    refusal = AcquisitionRefusal.from_dict(data)
    with pytest.raises(EvidenceError) as caught:
        milestone4_module._validated_public_refusal_gate(refusal)
    assert caught.value.code == "unpublishable_refusal"


def test_public_refusal_rejects_code_that_moves_terminal_gate() -> None:
    data = _refusal_data()
    data["error_code"] = "nifti_geometry_mismatch"
    refusal = AcquisitionRefusal.from_dict(data)
    with pytest.raises(EvidenceError) as caught:
        milestone4_module._validated_public_refusal_gate(refusal)
    assert caught.value.code == "refusal_progress_mismatch"


def test_public_refusal_accepts_dtype_failure_before_geometry() -> None:
    data = _refusal_data()
    data.update(
        {
            "error_code": "nifti_dtype_mismatch",
            "last_completed_stage": "metadata_semantics",
            "validated_medical_file_count": 5,
            "verified_artifact_count": 8,
        }
    )
    refusal = AcquisitionRefusal.from_dict(data)
    assert milestone4_module._validated_public_refusal_gate(refusal) == "nifti_structure"


@pytest.mark.parametrize("failed_gate", ["finite_values", "mask_domain"])
def test_public_gate_order_places_voxel_checks_before_geometry(failed_gate: str) -> None:
    gates = milestone4_module._gate_states("no-go", failed_gate)
    states = {item["gate"]: item["state"] for item in gates}
    assert states["nifti_structure"] == "passed"
    assert states["geometry"] == "not_reached"
    if failed_gate == "mask_domain":
        assert states["finite_values"] == "passed"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission assertions")
def test_synthetic_acquisition_publishes_closed_private_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, payloads = _synthetic_plan()
    monkeypatch.setattr(
        custody_module,
        "_load_trusted_plan",
        lambda decision_path, plan_path: (plan, DECISION_SHA256, PLAN_SHA256),
    )
    root = _private_root(tmp_path)

    def opener(url: str) -> FakeResponse:
        return FakeResponse(payloads[url], url)

    receipt = acquire_trusted_one_volume(
        DECISION,
        PLAN,
        root,
        approve_plan_sha256=PLAN_SHA256,
        allow_network=True,
        repository_root=ROOT,
        open_response=opener,
    )
    assert isinstance(receipt, CustodyReceipt)
    assert len(receipt.artifacts) == 8
    snapshot = root / SNAPSHOT_NAME
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o700
    assert stat.S_IMODE((snapshot / RECEIPT_PATH).stat().st_mode) == 0o600
    report = StructuralReport.from_dict(load_json(snapshot / STRUCTURAL_REPORT_PATH))
    assert report.status == "go"
    assert report.channel_order == (
        "image-t1c",
        "image-t1",
        "image-t2",
        "image-flair",
    )
    verified = verify_trusted_one_volume(
        DECISION,
        PLAN,
        root,
        approve_plan_sha256=PLAN_SHA256,
        repository_root=ROOT,
    )
    assert verified == receipt
    public = tmp_path / "public"
    build_milestone4_public_bundle(
        DECISION,
        PLAN,
        root,
        public,
        approve_plan_sha256=PLAN_SHA256,
        repository_root=ROOT,
    )
    verify_milestone4_public_bundle(public, require_trusted_digest=False)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission assertions")
def test_private_snapshot_rejects_extra_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, payloads = _synthetic_plan()
    monkeypatch.setattr(
        custody_module,
        "_load_trusted_plan",
        lambda decision_path, plan_path: (plan, DECISION_SHA256, PLAN_SHA256),
    )
    root = _private_root(tmp_path)

    def opener(url: str) -> FakeResponse:
        return FakeResponse(payloads[url], url)

    acquire_trusted_one_volume(
        DECISION,
        PLAN,
        root,
        approve_plan_sha256=PLAN_SHA256,
        allow_network=True,
        repository_root=ROOT,
        open_response=opener,
    )
    extra = root / SNAPSHOT_NAME / "unexpected.txt"
    extra.write_text("unexpected", encoding="ascii")
    extra.chmod(0o600)
    with pytest.raises(EvidenceError) as caught:
        verify_trusted_one_volume(
            DECISION,
            PLAN,
            root,
            approve_plan_sha256=PLAN_SHA256,
            repository_root=ROOT,
        )
    assert caught.value.code == "custody_snapshot_file_set_mismatch"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission assertions")
def test_snapshot_without_completion_is_never_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, payloads = _synthetic_plan()
    monkeypatch.setattr(
        custody_module,
        "_load_trusted_plan",
        lambda decision_path, plan_path: (plan, DECISION_SHA256, PLAN_SHA256),
    )
    root = _private_root(tmp_path)

    def opener(url: str) -> FakeResponse:
        return FakeResponse(payloads[url], url)

    real_publish = custody_module.rename_no_replace

    def publish_then_fail(source: Path, destination: Path) -> None:
        real_publish(source, destination)
        raise OSError("simulated directory fsync boundary")

    monkeypatch.setattr(custody_module, "rename_no_replace", publish_then_fail)
    with pytest.raises(OSError):
        acquire_trusted_one_volume(
            DECISION,
            PLAN,
            root,
            approve_plan_sha256=PLAN_SHA256,
            allow_network=True,
            repository_root=ROOT,
            open_response=opener,
        )
    assert (root / SNAPSHOT_NAME).is_dir()
    assert (root / "attempts" / f"{PLAN_SHA256}.refusal.json").is_file()
    assert not (root / "attempts" / f"{PLAN_SHA256}.completed.json").exists()
    with pytest.raises(EvidenceError) as caught:
        verify_trusted_one_volume(
            DECISION,
            PLAN,
            root,
            approve_plan_sha256=PLAN_SHA256,
            repository_root=ROOT,
        )
    assert caught.value.code == "acquisition_refused"


def test_windows_fails_closed_before_attempt_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _real_plan(monkeypatch)
    root = tmp_path / "custody"
    monkeypatch.setattr("voxelscope.custody._is_windows", lambda: True)
    with pytest.raises(EvidenceError) as caught:
        acquire_trusted_one_volume(
            DECISION,
            PLAN,
            root,
            approve_plan_sha256=PLAN_SHA256,
            allow_network=True,
            repository_root=ROOT,
            open_response=lambda url: pytest.fail("opener must not be called"),
        )
    assert caught.value.code == "private_acl_unverified"
    assert not root.exists()


def test_public_evidence_bundle_is_deterministic_closed_and_synthetic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hash = _build_milestone4_public_bundle(first, outcome="go")
    second_hash = _build_milestone4_public_bundle(second, outcome="go")
    assert first_hash == second_hash
    assert verify_milestone4_public_bundle(first, require_trusted_digest=False) == first_hash
    assert {path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()} == {
        *PAYLOAD_PATHS,
        "SHA256SUMS",
        "bundle.json",
        "bundle.sha256",
    }
    summary = load_json(first / "public-summary-v1.json")
    assert summary["overall_status"] == "go"
    assert summary["artifact_count"] == 8
    assert summary["medical_artifact_count"] == 5
    assert summary["metadata_artifact_count"] == 3
    assert summary["verified_acquired_artifact_count"] == 8
    assert summary["nifti_validated_medical_file_count"] == 5
    assert summary["geometry_comparison_count"] == 4
    assert summary["failed_structural_gate_count"] == 0
    assert summary["gates"][-1] == {
        "gate": "inference_authorization",
        "state": "not_authorized",
    }


def test_public_go_builder_requires_private_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise EvidenceError("private_verification_failed", "synthetic refusal")

    monkeypatch.setattr(milestone4_module, "verify_trusted_one_volume", refuse)
    output = tmp_path / "public"
    with pytest.raises(EvidenceError) as caught:
        build_milestone4_public_bundle(
            DECISION,
            PLAN,
            tmp_path / "private",
            output,
            approve_plan_sha256=PLAN_SHA256,
            repository_root=ROOT,
        )
    assert caught.value.code == "private_verification_failed"
    assert not output.exists()


def test_public_evidence_verifier_rejects_tampering(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _build_milestone4_public_bundle(bundle, outcome="go")
    (bundle / "README.txt").write_text("tampered", encoding="ascii")
    with pytest.raises(EvidenceError) as caught:
        verify_milestone4_public_bundle(bundle, require_trusted_digest=False)
    assert caught.value.code == "bundle_artifact_mismatch"


def test_public_evidence_verifier_rejects_progress_inflation(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _build_milestone4_public_bundle(
        bundle,
        outcome="no-go",
        first_failed_gate="geometry",
        verified_artifact_count=8,
        validated_medical_file_count=5,
        geometry_comparison_count=2,
    )
    summary = load_json(bundle / "public-summary-v1.json")
    summary["geometry_comparison_count"] = 4
    write_json(bundle / "public-summary-v1.json", summary)
    checksum_lines = [f"{sha256_file(bundle / path)}  {path}\n" for path in sorted(PAYLOAD_PATHS)]
    (bundle / "SHA256SUMS").write_text("".join(checksum_lines), encoding="ascii")
    index = load_json(bundle / "bundle.json")
    for indexed in index["artifacts"]:
        path = bundle / indexed["path"]
        indexed["sha256"] = sha256_file(path)
        indexed["size_bytes"] = path.stat().st_size
    write_json(bundle / "bundle.json", index)
    (bundle / "bundle.sha256").write_text(
        f"{sha256_file(bundle / 'bundle.json')}  bundle.json\n",
        encoding="ascii",
    )
    with pytest.raises(EvidenceError) as caught:
        verify_milestone4_public_bundle(bundle, require_trusted_digest=False)
    assert caught.value.code == "invalid_public_progress"


def test_public_verifier_rejects_self_consistent_fixture_substitution(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _build_milestone4_public_bundle(bundle, outcome="go")
    relative = "synthetic/malformed.synthetic-nifti-gzip"
    replacement = b"arbitrary private bytes"
    (bundle / relative).write_bytes(replacement)
    manifest = load_json(bundle / "synthetic-fixture-manifest.json")
    item = next(item for item in manifest["fixtures"] if item["path"] == relative)
    item["sha256"] = hashlib.sha256(replacement).hexdigest()
    item["size_bytes"] = len(replacement)
    write_json(bundle / "synthetic-fixture-manifest.json", manifest)
    checksum_lines = [f"{sha256_file(bundle / path)}  {path}\n" for path in sorted(PAYLOAD_PATHS)]
    (bundle / "SHA256SUMS").write_text("".join(checksum_lines), encoding="ascii")
    index = load_json(bundle / "bundle.json")
    for indexed in index["artifacts"]:
        path = bundle / indexed["path"]
        indexed["sha256"] = sha256_file(path)
        indexed["size_bytes"] = path.stat().st_size
    write_json(bundle / "bundle.json", index)
    (bundle / "bundle.sha256").write_text(
        f"{sha256_file(bundle / 'bundle.json')}  bundle.json\n",
        encoding="ascii",
    )
    with pytest.raises(EvidenceError) as caught:
        verify_milestone4_public_bundle(bundle, require_trusted_digest=False)
    assert caught.value.code == "invalid_synthetic_fixture"


def test_public_verifier_rejects_oversized_index(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _build_milestone4_public_bundle(bundle, outcome="go")
    with (bundle / "bundle.json").open("ab") as stream:
        stream.write(b" " * (2 * 1024 * 1024))
    with pytest.raises(EvidenceError) as caught:
        verify_milestone4_public_bundle(bundle, require_trusted_digest=False)
    assert caught.value.code == "public_file_limit"


def test_public_no_go_preserves_first_failure_and_downstream_state(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _build_milestone4_public_bundle(
        bundle,
        outcome="no-go",
        first_failed_gate="content_identity",
    )
    summary = load_json(bundle / "public-summary-v1.json")
    states = {item["gate"]: item["state"] for item in summary["gates"]}
    assert states["plan_approval"] == "passed"
    assert states["acquisition"] == "passed"
    assert states["content_identity"] == "failed"
    assert states["nifti_structure"] == "not_reached"
    assert states["inference_authorization"] == "not_authorized"
    assert summary["verified_acquired_artifact_count"] == 8
    assert summary["nifti_validated_medical_file_count"] == 0
    assert summary["geometry_comparison_count"] == 0
    assert summary["failed_structural_gate_count"] == 1
    ledger = load_json(bundle / "stage-ledger.json")
    assert ledger["attempt_terminal_status"] == "refused"
    assert ledger["acquisition_status"] == "go"
    assert ledger["content_identity_status"] == "no-go"
    assert ledger["structural_validation_status"] == "not_reached"
    verify_milestone4_public_bundle(bundle, require_trusted_digest=False)
