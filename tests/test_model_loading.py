# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import stat
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import voxelscope.model_loader_worker as worker_module
import voxelscope.model_loading as loading_module
from voxelscope.canonical import (
    EvidenceError,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from voxelscope.custody_records import ArchiveMember
from voxelscope.milestone8_cli import _read_descriptor_object
from voxelscope.model_loading import (
    _copy_immutable_file,
    _zip_member_path,
    execute_model_loading_qualification,
    extract_selected_archive,
    load_model_loading_plan,
    run_loader_worker,
    verify_model_loading_refusal,
)
from voxelscope.model_loading_contract import (
    LEGACY_SYNTHETIC_BUNDLE_SHA256,
    MILESTONE6_PLAN_SHA256,
    MILESTONE6_PUBLIC_BUNDLE_SHA256,
    MILESTONE6_REPORT_SHA256,
    MILESTONE7_PLAN_SHA256,
    MILESTONE7_PUBLIC_BUNDLE_SHA256,
    MODEL_ARCHIVE_SHA256,
)
from voxelscope.model_loading_records import (
    ArchitectureContract,
    ExtractionMember,
    LoaderWorkerRequest,
    LoaderWorkerResult,
    ModelLoadingRefusal,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/model-loading-plan-v1.json"


def _member(path: str, payload: bytes, role: str = "synthetic") -> ArchiveMember:
    return ArchiveMember(path, len(payload), sha256_bytes(payload), role)


def _selection(path: str, payload: bytes, role: str) -> ExtractionMember:
    return ExtractionMember(path, role, len(payload), sha256_bytes(payload))


def _write_archive(path: Path, members: list[tuple[str | zipfile.ZipInfo, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def _extract(
    tmp_path: Path,
    members: list[tuple[str | zipfile.ZipInfo, bytes]],
    *,
    exact: tuple[ArchiveMember, ...] | None = None,
    selected: tuple[ExtractionMember, ...] | None = None,
    max_members: int = 8,
    max_total: int = 4096,
    max_member: int = 2048,
    max_ratio: int = 100,
) -> tuple[Any, ...]:
    archive = tmp_path / "fixture.zip"
    _write_archive(archive, members)
    regular = [(name, payload) for name, payload in members if isinstance(name, str)]
    exact = exact or tuple(_member(name, payload) for name, payload in regular)
    selected = selected or (_selection(regular[0][0], regular[0][1], "license"),)
    return extract_selected_archive(
        archive,
        tmp_path / "output",
        exact_members=exact,
        selected_members=selected,
        max_members=max_members,
        max_total_uncompressed_bytes=max_total,
        max_member_bytes=max_member,
        max_compression_ratio=max_ratio,
    )


def _synthetic_request(tmp_path: Path) -> LoaderWorkerRequest:
    plan, plan_sha256 = load_model_loading_plan(PLAN, repository_root=ROOT)
    tensor = {
        "device": "cpu",
        "dtype": "float32",
        "finite": True,
        "layout": "strided",
        "shape": [2, 3],
    }
    checkpoint = tmp_path / "checkpoint.json"
    config = tmp_path / "config.json"
    write_json(
        checkpoint,
        {"format": "voxelscope-safe-tensor-fixture/v1", "state": {"weight": tensor}},
    )
    write_json(
        config,
        {"expected_state": {"weight": tensor}, "format": "voxelscope-safe-loader-config/v1"},
    )
    return LoaderWorkerRequest(
        "voxelscope/loader-worker-request/v1",
        "synthetic-json",
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
        "0" * 64,
        plan.loader_memory_limit_bytes,
    )


def _run_synthetic(request: LoaderWorkerRequest) -> LoaderWorkerResult:
    python = Path(sys.executable).absolute()
    return run_loader_worker(
        python,
        request,
        timeout_seconds=30,
        approved_python_sha256=sha256_file(python),
    )


def test_plan_binds_all_public_model_and_prior_gate_identities() -> None:
    plan, digest = load_model_loading_plan(PLAN, repository_root=ROOT)
    assert digest == sha256_file(PLAN)
    assert plan.archive_sha256 == MODEL_ARCHIVE_SHA256
    assert plan.milestone7_plan_sha256 == MILESTONE7_PLAN_SHA256
    assert plan.milestone7_public_bundle_sha256 == MILESTONE7_PUBLIC_BUNDLE_SHA256
    assert plan.milestone6_plan_sha256 == MILESTONE6_PLAN_SHA256
    assert plan.milestone6_report_sha256 == MILESTONE6_REPORT_SHA256
    assert plan.milestone6_public_bundle_sha256 == MILESTONE6_PUBLIC_BUNDLE_SHA256
    assert plan.legacy_synthetic_bundle_sha256 == LEGACY_SYNTHETIC_BUNDLE_SHA256
    assert plan.runtime.operating_system == "linux"
    assert plan.runtime.python_version == "3.12"
    assert plan.runtime.torch_version == "2.4.0"


def test_plan_tamper_is_refused(tmp_path: Path) -> None:
    candidate = tmp_path / "plan.json"
    data = load_json(PLAN)
    data["forward_allowed"] = True
    write_json(candidate, data)
    with pytest.raises(EvidenceError) as captured:
        load_model_loading_plan(candidate, repository_root=ROOT)
    assert captured.value.code == "model_loading_plan_digest_mismatch"


def test_runtime_contract_refuses_substitute_framework_version() -> None:
    data = load_json(PLAN)
    data["runtime"]["torch_version"] = "2.5.0"
    with pytest.raises(EvidenceError) as captured:
        loading_module.ModelLoadingPlan.from_dict(data)
    assert captured.value.code == "untrusted_model_runtime"


def test_implementation_drift_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    original = loading_module.sha256_file

    def changed(path: Path) -> str:
        if path.name == "model_loading.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(loading_module, "sha256_file", changed)
    with pytest.raises(EvidenceError) as captured:
        load_model_loading_plan(PLAN, repository_root=ROOT)
    assert captured.value.code == "implementation_identity_mismatch"


def test_safe_archive_extracts_only_allowlisted_member(tmp_path: Path) -> None:
    result = _extract(
        tmp_path,
        [("LICENSE", b"license"), ("config.json", b"{}"), ("checkpoint.bin", b"tensor")],
        selected=(_selection("checkpoint.bin", b"tensor", "pytorch-checkpoint"),),
    )
    assert tuple(item.path for item in result) == ("extracted/checkpoint.bin",)
    assert (tmp_path / "output/extracted/checkpoint.bin").read_bytes() == b"tensor"
    assert not (tmp_path / "output/extracted/config.json").exists()


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("../escape", "unsafe_path"),
        ("/absolute", "unsafe_path"),
        ("C:drive", "unsafe_path"),
    ],
)
def test_archive_path_escape_is_refused(tmp_path: Path, name: str, code: str) -> None:
    with pytest.raises(EvidenceError) as captured:
        _extract(
            tmp_path,
            [(name, b"x")],
            exact=(_member("safe", b"x"),),
            selected=(_selection("safe", b"x", "license"),),
        )
    assert captured.value.code == code
    assert not (tmp_path / "output").exists()


def test_archive_backslash_is_refused_before_platform_normalization() -> None:
    with pytest.raises(EvidenceError) as captured:
        _zip_member_path(zipfile.ZipInfo("a\\b"))
    assert captured.value.code == "unsafe_path"


def test_archive_duplicate_is_refused(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        with pytest.raises(EvidenceError) as captured:
            _extract(
                tmp_path,
                [("LICENSE", b"a"), ("LICENSE", b"a")],
                exact=(_member("LICENSE", b"a"),),
            )
    assert captured.value.code == "duplicate_archive_member"


def test_archive_case_collision_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError) as captured:
        _extract(
            tmp_path,
            [("LICENSE", b"a"), ("license", b"b")],
            exact=(_member("LICENSE", b"a"), _member("license", b"b")),
        )
    assert captured.value.code == "archive_case_collision"


@pytest.mark.parametrize(
    ("file_type", "code"),
    [
        (stat.S_IFLNK, "archive_special_member"),
        (stat.S_IFCHR, "archive_special_member"),
        (stat.S_IFBLK, "archive_special_member"),
        (stat.S_IFIFO, "archive_special_member"),
    ],
)
def test_archive_link_device_and_special_entries_are_refused(
    tmp_path: Path, file_type: int, code: str
) -> None:
    info = zipfile.ZipInfo("LICENSE")
    info.create_system = 3
    info.external_attr = (file_type | 0o600) << 16
    with pytest.raises(EvidenceError) as captured:
        _extract(
            tmp_path,
            [(info, b"target")],
            exact=(_member("LICENSE", b"target"),),
            selected=(_selection("LICENSE", b"target", "license"),),
        )
    assert captured.value.code == code


def test_encrypted_archive_flag_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    _write_archive(archive, [("LICENSE", b"license")])
    payload = bytearray(archive.read_bytes())
    local = payload.find(b"PK\x03\x04")
    central = payload.find(b"PK\x01\x02")
    payload[local + 6 : local + 8] = (1).to_bytes(2, "little")
    payload[central + 8 : central + 10] = (1).to_bytes(2, "little")
    archive.write_bytes(payload)
    with pytest.raises(EvidenceError) as captured:
        extract_selected_archive(
            archive,
            tmp_path / "output",
            exact_members=(_member("LICENSE", b"license"),),
            selected_members=(_selection("LICENSE", b"license", "license"),),
            max_members=2,
            max_total_uncompressed_bytes=100,
            max_member_bytes=100,
            max_compression_ratio=100,
        )
    assert captured.value.code == "encrypted_archive_forbidden"


@pytest.mark.parametrize(
    ("limits", "code"),
    [
        ({"max_members": 1}, "archive_file_limit"),
        ({"max_total": 5}, "archive_total_size_limit"),
        ({"max_member": 5}, "archive_file_size_limit"),
        ({"max_ratio": 1}, "archive_compression_ratio_limit"),
    ],
)
def test_archive_bomb_limits_are_refused(tmp_path: Path, limits: dict[str, int], code: str) -> None:
    kwargs = {
        "max_members": 8,
        "max_total": 4096,
        "max_member": 2048,
        "max_ratio": 100,
        **limits,
    }
    with pytest.raises(EvidenceError) as captured:
        _extract(tmp_path, [("LICENSE", b"a" * 1000), ("other", b"b")], **kwargs)
    assert captured.value.code == code


@pytest.mark.parametrize("drift", ["size", "hash"])
def test_archive_member_identity_drift_is_refused(tmp_path: Path, drift: str) -> None:
    expected = _member("LICENSE", b"license")
    if drift == "size":
        expected = replace(expected, size_bytes=expected.size_bytes + 1)
    else:
        expected = replace(expected, sha256="0" * 64)
    with pytest.raises(EvidenceError) as captured:
        _extract(tmp_path, [("LICENSE", b"license")], exact=(expected,))
    assert captured.value.code == "archive_member_identity_mismatch"


def test_archive_missing_and_unexpected_members_are_refused(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError) as captured:
        _extract(
            tmp_path,
            [("LICENSE", b"license")],
            exact=(_member("LICENSE", b"license"), _member("config", b"{}")),
        )
    assert captured.value.code == "missing_archive_member"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(EvidenceError) as captured:
        _extract(
            other,
            [("LICENSE", b"license"), ("extra", b"x")],
            exact=(_member("LICENSE", b"license"),),
        )
    assert captured.value.code == "unexpected_archive_member"


def test_archive_destination_no_clobber(tmp_path: Path) -> None:
    (tmp_path / "output").mkdir()
    with pytest.raises(EvidenceError) as captured:
        _extract(tmp_path, [("LICENSE", b"license")])
    assert captured.value.code == "output_exists"


def test_archive_replacement_during_staging_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "archive.zip"
    destination = tmp_path / "staged.zip"
    source.write_bytes(b"a" * (1024 * 1024 + 1))
    expected = sha256_file(source)
    original_read = loading_module.os.read
    replaced = False

    def mutating_read(descriptor: int, amount: int) -> bytes:
        nonlocal replaced
        result = original_read(descriptor, amount)
        if result and not replaced:
            replaced = True
            source.write_bytes(b"b" * (1024 * 1024 + 1))
        return result

    monkeypatch.setattr(loading_module.os, "read", mutating_read)
    with pytest.raises(EvidenceError) as captured:
        _copy_immutable_file(
            source,
            destination,
            expected_sha256=expected,
            expected_size=source.stat().st_size,
        )
    assert captured.value.code == "archive_replacement"
    assert not destination.exists()


def test_synthetic_loader_worker_accepts_only_closed_tensor_mapping(tmp_path: Path) -> None:
    result = _run_synthetic(_synthetic_request(tmp_path))
    assert result.status == "go"
    assert result.state_mapping_closed
    assert result.strict_keys_verified
    assert result.tensor_metadata_verified
    assert result.finite_parameters_verified
    assert result.cpu_only_verified
    assert result.model_instantiated
    assert result.state_loaded
    assert not result.forward_called
    assert not result.accelerator_used
    assert not result.network_used


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"format": "unsafe", "state": {}}, "unsafe_checkpoint_object"),
        (
            {"format": "voxelscope-safe-tensor-fixture/v1", "state": {"weight": "object"}},
            "invalid_json_type",
        ),
        (
            {
                "format": "voxelscope-safe-tensor-fixture/v1",
                "state": {
                    "unexpected": {
                        "device": "cpu",
                        "dtype": "float32",
                        "finite": True,
                        "layout": "strided",
                        "shape": [2, 3],
                    }
                },
            },
            "state_key_mismatch",
        ),
    ],
)
def test_synthetic_loader_refuses_unsafe_objects_and_keys(
    tmp_path: Path, mutation: dict[str, Any], code: str
) -> None:
    request = _synthetic_request(tmp_path)
    write_json(Path(request.checkpoint_path), mutation)
    request = replace(request, checkpoint_sha256=sha256_file(Path(request.checkpoint_path)))
    result = _run_synthetic(request)
    assert result.status == "refused"
    assert result.error_code == code
    assert not result.state_loaded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cuda"),
        ("dtype", "float64"),
        ("layout", "sparse"),
        ("shape", [3, 2]),
        ("finite", False),
    ],
)
def test_synthetic_loader_refuses_tensor_metadata_and_nonfinite_values(
    tmp_path: Path, field: str, value: Any
) -> None:
    request = _synthetic_request(tmp_path)
    checkpoint = load_json(Path(request.checkpoint_path))
    checkpoint["state"]["weight"][field] = value
    write_json(Path(request.checkpoint_path), checkpoint)
    request = replace(request, checkpoint_sha256=sha256_file(Path(request.checkpoint_path)))
    result = _run_synthetic(request)
    assert result.status == "refused"
    assert result.error_code in {"synthetic_tensor_mismatch", "tensor_metadata_mismatch"}
    assert not result.forward_called


def test_worker_refuses_checkpoint_and_config_replacement(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    result = _run_synthetic(replace(request, checkpoint_sha256="0" * 64))
    assert result.error_code == "checkpoint_identity_mismatch"
    result = _run_synthetic(replace(request, config_sha256="0" * 64))
    assert result.error_code == "config_identity_mismatch"


def test_worker_uses_verified_open_descriptor_after_path_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = _synthetic_request(tmp_path)
    original = worker_module._read_verified_file
    replaced = False

    def replace_after_open(
        path_value: str,
        expected_sha256: str,
        *,
        identity_code: str,
    ) -> tuple[Any, bytes]:
        nonlocal replaced
        result = original(path_value, expected_sha256, identity_code=identity_code)
        if not replaced and path_value == request.checkpoint_path:
            replaced = True
            Path(path_value).write_bytes(b'{"format":"unsafe"}\n')
        return result

    monkeypatch.setattr(worker_module, "_read_verified_file", replace_after_open)
    result = worker_module.execute_request(request)
    assert result.status == "go"
    assert Path(request.checkpoint_path).read_bytes() == b'{"format":"unsafe"}\n'


def test_runtime_distribution_identity_detects_file_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "site" / "torch"
    package.mkdir(parents=True)
    module_file = package / "__init__.py"
    module_file.write_bytes(b"trusted")

    class FakeFile:
        hash = None
        size = None

        def __str__(self) -> str:
            return "site/torch/__init__.py"

    class FakeDistribution:
        files = [FakeFile()]
        version = "2.4.0"
        metadata = {"Name": "torch"}

        def locate_file(self, item: object) -> Path:
            return tmp_path / str(item)

    monkeypatch.setattr(worker_module.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        worker_module.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(),
    )
    module = type("Module", (), {"__file__": str(module_file)})()
    first = worker_module._distribution_identity("torch", module)
    module_file.write_bytes(b"drift")
    second = worker_module._distribution_identity("torch", module)
    assert first != second


def test_worker_refusal_preserves_completed_progress() -> None:
    progress = worker_module._WorkerProgress(
        runtime_verified=True,
        architecture_verified=True,
        output_channels_verified=True,
        model_instantiated=True,
        weights_only_used=True,
    )
    result = progress.refusal("weights_only_load_refused")
    assert result.runtime_verified
    assert result.architecture_verified
    assert result.output_channels_verified
    assert result.model_instantiated
    assert result.weights_only_used
    assert not result.state_loaded


def test_worker_request_descriptor_bound_is_enforced(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    request = replace(request, checkpoint_path="x" * 20_000)
    with pytest.raises(EvidenceError) as captured:
        _run_synthetic(request)
    assert captured.value.code == "worker_request_limit"


def test_architecture_contract_refuses_mismatch() -> None:
    with pytest.raises(EvidenceError) as captured:
        ArchitectureContract(
            "monai.networks.nets.SegResNet",
            3,
            16,
            4,
            4,
            0.2,
            (1, 2, 2, 4),
            (1, 1, 1),
        )
    assert captured.value.code == "untrusted_architecture"


@pytest.mark.parametrize(
    "field", ["forward_called", "inference_run", "accelerator_used", "network_used"]
)
def test_loader_result_refuses_forbidden_action_claim(field: str) -> None:
    values: dict[str, Any] = {
        "schema_version": "voxelscope/loader-worker-result/v1",
        "status": "refused",
        "error_code": "synthetic",
        "runtime_verified": False,
        "weights_only_used": False,
        "state_mapping_closed": False,
        "strict_keys_verified": False,
        "tensor_metadata_verified": False,
        "finite_parameters_verified": False,
        "cpu_only_verified": False,
        "architecture_verified": False,
        "output_channels_verified": False,
        "parameter_tensor_count": 0,
        "buffer_tensor_count": 0,
        "parameter_element_count": 0,
        "model_instantiated": False,
        "state_loaded": False,
        "forward_called": False,
        "inference_run": False,
        "accelerator_used": False,
        "network_used": False,
    }
    values[field] = True
    with pytest.raises(EvidenceError) as captured:
        LoaderWorkerResult.from_dict(values)
    assert captured.value.code == "invalid_loader_result"


def test_worker_python_replacement_is_refused(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    with pytest.raises(EvidenceError) as captured:
        run_loader_worker(
            Path(sys.executable),
            request,
            timeout_seconds=30,
            approved_python_sha256="0" * 64,
        )
    assert captured.value.code == "worker_python_identity_mismatch"


class _FakeProcess:
    def __init__(self, *, timeout: bool) -> None:
        self.timeout = timeout
        self.killed = False
        self.returncode: int | None = None
        self.request_read = -1
        self.result_write = -1

    def wait(self, timeout: int | None = None) -> int:
        if self.timeout and not self.killed:
            raise loading_module.subprocess.TimeoutExpired("worker", timeout)
        if self.request_read >= 0:
            os.close(self.request_read)
            self.request_read = -1
        if self.result_write >= 0:
            os.close(self.result_write)
            self.result_write = -1
        self.returncode = -9 if self.killed else 1
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def poll(self) -> int | None:
        return self.returncode


@pytest.mark.parametrize(
    ("timeout", "code"),
    [(True, "loader_worker_timeout"), (False, "loader_worker_crash")],
)
@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited descriptors")
def test_worker_timeout_and_crash_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: bool, code: str
) -> None:
    request = _synthetic_request(tmp_path)
    fake = _FakeProcess(timeout=timeout)

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        assert kwargs["stdout"] is loading_module.subprocess.DEVNULL
        assert kwargs["stderr"] is loading_module.subprocess.DEVNULL
        assert kwargs["executable"] == str(Path(sys.executable).absolute())
        command = args[0]
        assert command[0] == "voxelscope-model-loader"
        fake.request_read = os.dup(int(command[command.index("--request-fd") + 1]))
        fake.result_write = os.dup(int(command[command.index("--result-fd") + 1]))
        return fake

    monkeypatch.setattr(loading_module.subprocess, "Popen", fake_popen)
    python = Path(sys.executable)
    with pytest.raises(EvidenceError) as captured:
        run_loader_worker(
            python,
            request,
            timeout_seconds=1,
            approved_python_sha256=sha256_file(python),
        )
    assert captured.value.code == code
    if timeout:
        assert fake.killed


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker permissions")
def test_attempt_marker_precedes_private_receipt_read_and_replay_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    receipt_sha = "a" * 64
    python_sha = "b" * 64

    def verify_after_marker(*args: Any, **kwargs: Any) -> None:
        markers = list((root / "model-loading-attempts").glob("*.started.json"))
        assert len(markers) == 1
        raise EvidenceError("synthetic_receipt_refusal", "stop before private read")

    monkeypatch.setattr(loading_module, "_verify_receipt", verify_after_marker)
    keywords = {
        "worker_python": Path(sys.executable),
        "approve_plan_sha256": sha256_file(PLAN),
        "approve_custody_receipt_sha256": receipt_sha,
        "approve_archive_sha256": MODEL_ARCHIVE_SHA256,
        "approve_worker_python_sha256": python_sha,
        "approve_worker_runtime_sha256": "c" * 64,
        "repository_root": ROOT,
    }
    with pytest.raises(EvidenceError) as captured:
        execute_model_loading_qualification(PLAN, root, **keywords)
    assert captured.value.code == "synthetic_receipt_refusal"
    terminals = list((root / "model-loading-attempts").glob("*.terminal.json"))
    assert len(terminals) == 1
    assert load_json(terminals[0])["status"] == "refused"
    verify_keywords = {key: value for key, value in keywords.items() if key != "worker_python"}
    verified = verify_model_loading_refusal(PLAN, root, **verify_keywords)
    assert verified.error_code == "synthetic_receipt_refusal"
    with pytest.raises(EvidenceError) as replay:
        execute_model_loading_qualification(PLAN, root, **keywords)
    assert replay.value.code == "model_loading_attempt_exists"
    terminal = load_json(terminals[0])
    terminal["plan_sha256"] = "d" * 64
    write_json(terminals[0], terminal)
    with pytest.raises(EvidenceError) as tampered:
        verify_model_loading_refusal(PLAN, root, **verify_keywords)
    assert tampered.value.code == "model_loading_refusal_mismatch"


def test_windows_private_execution_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr("voxelscope.custody._is_windows", lambda: True)
    with pytest.raises(EvidenceError) as captured:
        loading_module.require_private_root(root, repository_root=ROOT)
    assert captured.value.code == "private_acl_unverified"


def test_private_authorization_descriptor_bound_is_enforced(tmp_path: Path) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_bytes(b"x" * 8193)
    if os.name != "nt":
        os.chmod(authorization, 0o600)
    descriptor = os.open(authorization, os.O_RDONLY)
    try:
        with pytest.raises(EvidenceError) as captured:
            _read_descriptor_object(descriptor)
    finally:
        os.close(descriptor)
    assert captured.value.code == "private_authorization_limit"


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker permissions")
def test_output_no_clobber_precedes_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    (root / "model-loading-qualification-v1").mkdir()
    with pytest.raises(EvidenceError) as captured:
        execute_model_loading_qualification(
            PLAN,
            root,
            worker_python=Path(sys.executable),
            approve_plan_sha256=sha256_file(PLAN),
            approve_custody_receipt_sha256="a" * 64,
            approve_archive_sha256=MODEL_ARCHIVE_SHA256,
            approve_worker_python_sha256="b" * 64,
            approve_worker_runtime_sha256="c" * 64,
            repository_root=ROOT,
        )
    assert captured.value.code == "output_exists"
    assert not list((root / "model-loading-attempts").glob("*.started.json"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX marker permissions")
def test_terminal_fault_cannot_be_reported_as_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)

    def refuse_read(*args: Any, **kwargs: Any) -> None:
        raise EvidenceError("synthetic_receipt_refusal", "stop")

    def fail_terminal(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic terminal failure")

    monkeypatch.setattr(loading_module, "_verify_receipt", refuse_read)
    monkeypatch.setattr(loading_module, "_write_terminal", fail_terminal)
    with pytest.raises(EvidenceError) as captured:
        execute_model_loading_qualification(
            PLAN,
            root,
            worker_python=Path(sys.executable),
            approve_plan_sha256=sha256_file(PLAN),
            approve_custody_receipt_sha256="a" * 64,
            approve_archive_sha256=MODEL_ARCHIVE_SHA256,
            approve_worker_python_sha256="b" * 64,
            approve_worker_runtime_sha256="c" * 64,
            repository_root=ROOT,
        )
    assert captured.value.code == "refusal_write_failed"
    assert not list((root / "model-loading-attempts").glob("*.terminal.json"))


def test_loader_worker_source_has_no_forward_or_accelerator_invocation() -> None:
    source = (ROOT / "src/voxelscope/model_loader_worker.py").read_text(encoding="utf-8")
    assert ".forward(" not in source
    assert ".cuda(" not in source
    assert '.to("cuda"' not in source
    assert "weights_only=True" in source
    assert 'map_location="cpu"' in source


def test_refusal_record_rejects_loaded_without_instantiation() -> None:
    with pytest.raises(EvidenceError) as captured:
        ModelLoadingRefusal(
            "voxelscope/model-loading-refusal/v1",
            "refused",
            "synthetic",
            "loader_worker",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "e" * 64,
            3,
            False,
            False,
            True,
            False,
            False,
            False,
            "2026-09-30T00:00:00Z",
        )
    assert captured.value.code == "invalid_refusal_progress"
