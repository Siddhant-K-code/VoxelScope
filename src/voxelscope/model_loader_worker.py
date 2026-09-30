# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU-only loader worker. This module never calls a model forward method."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib
import importlib.metadata
import os
import stat
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_REPOSITORY_SOURCE_ROOT: Path | None = None
if __package__ in {None, ""}:
    _REPOSITORY_SOURCE_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPOSITORY_SOURCE_ROOT))
    __package__ = "voxelscope"

from .canonical import (  # noqa: E402
    EvidenceError,
    canonical_json_bytes,
    load_json_bytes,
    sha256_file,
)
from .model_loader_protocol import (  # noqa: E402
    WorkerRequest,
    WorkerResult,
)
from .records import require_list, require_object, strict_fields  # noqa: E402

if _REPOSITORY_SOURCE_ROOT is not None:
    sys.path.remove(str(_REPOSITORY_SOURCE_ROOT))

_MAX_REQUEST_BYTES = 16 * 1024
_MAX_RESULT_BYTES = 64 * 1024


@dataclass
class _WorkerProgress:
    runtime_verified: bool = False
    weights_only_used: bool = False
    state_mapping_closed: bool = False
    strict_keys_verified: bool = False
    tensor_metadata_verified: bool = False
    finite_parameters_verified: bool = False
    cpu_only_verified: bool = False
    architecture_verified: bool = False
    output_channels_verified: bool = False
    parameter_tensor_count: int = 0
    buffer_tensor_count: int = 0
    parameter_element_count: int = 0
    model_instantiated: bool = False
    state_loaded: bool = False

    def refusal(self, code: str) -> WorkerResult:
        return _refusal(code, **self.__dict__)


def _read_descriptor(descriptor: int, maximum: int) -> bytes:
    if descriptor < 0:
        raise EvidenceError("unsafe_worker_descriptor", "descriptor must be nonnegative")
    metadata = os.fstat(descriptor)
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISFIFO(metadata.st_mode)):
        raise EvidenceError("unsafe_worker_descriptor", "regular file or pipe required")
    chunks: list[bytes] = []
    size = 0
    while chunk := os.read(descriptor, maximum + 1 - size):
        size += len(chunk)
        if size > maximum:
            raise EvidenceError("worker_request_limit", "request is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _write_descriptor(descriptor: int, value: WorkerResult) -> None:
    payload = canonical_json_bytes(value)
    if len(payload) > _MAX_RESULT_BYTES:
        raise EvidenceError("worker_result_limit", "result is too large")
    pending = memoryview(payload)
    while pending:
        written = os.write(descriptor, pending)
        if written <= 0:
            raise OSError("result descriptor closed")
        pending = pending[written:]


def _install_network_refusal() -> None:
    def audit(event: str, _arguments: tuple[Any, ...]) -> None:
        if event.startswith("socket.") or event in {"urllib.Request", "http.client.connect"}:
            raise PermissionError("network access is forbidden")

    sys.addaudithook(audit)
    for key in (
        "ALL_PROXY",
        "FTP_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "ftp_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        os.environ.pop(key, None)


def _set_memory_limit(limit: int) -> None:
    if os.name == "nt":
        raise EvidenceError("private_acl_unverified", "Windows private execution is unsupported")
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError) as exc:
        raise EvidenceError("worker_resource_limit_unavailable", "resource limit failed") from exc


def _refusal(code: str, **progress: bool | int) -> WorkerResult:
    defaults: dict[str, bool | int] = {
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
    }
    defaults.update(progress)
    return WorkerResult(
        "voxelscope/loader-worker-result/v1",
        "refused",
        code,
        bool(defaults["runtime_verified"]),
        bool(defaults["weights_only_used"]),
        bool(defaults["state_mapping_closed"]),
        bool(defaults["strict_keys_verified"]),
        bool(defaults["tensor_metadata_verified"]),
        bool(defaults["finite_parameters_verified"]),
        bool(defaults["cpu_only_verified"]),
        bool(defaults["architecture_verified"]),
        bool(defaults["output_channels_verified"]),
        int(defaults["parameter_tensor_count"]),
        int(defaults["buffer_tensor_count"]),
        int(defaults["parameter_element_count"]),
        bool(defaults["model_instantiated"]),
        bool(defaults["state_loaded"]),
        False,
        False,
        False,
        False,
    )


def _closed_synthetic_state(value: Any) -> dict[str, dict[str, Any]]:
    state = require_object(value, "synthetic state")
    result: dict[str, dict[str, Any]] = {}
    for name, raw in state.items():
        if type(name) is not str or not name:
            raise EvidenceError("unsafe_state_key", "state key must be a nonempty string")
        item = strict_fields(
            require_object(raw, "synthetic tensor"),
            {"device", "dtype", "finite", "layout", "shape"},
            "synthetic tensor",
        )
        shape = tuple(item_value for item_value in require_list(item["shape"], "shape"))
        if (
            any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or item["device"] != "cpu"
            or item["dtype"] != "float32"
            or item["layout"] != "strided"
            or item["finite"] is not True
        ):
            raise EvidenceError("synthetic_tensor_mismatch", name)
        result[name] = item
    return result


def _read_verified_file(
    path_value: str,
    expected_sha256: str,
    *,
    identity_code: str,
) -> tuple[Any, bytes]:
    path = Path(path_value)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError("unsafe_worker_input", "input open failed") from exc
    stream = os.fdopen(descriptor, "rb", closefd=True)
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode):
        stream.close()
        raise EvidenceError("unsafe_worker_input", "regular input required")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        chunks.append(chunk)
    after = os.fstat(stream.fileno())
    stable = (
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
    if not stable:
        stream.close()
        raise EvidenceError("worker_input_replacement", "input changed while hashing")
    if digest.hexdigest() != expected_sha256:
        stream.close()
        raise EvidenceError(identity_code, "input digest differs")
    stream.seek(0)
    return stream, b"".join(chunks)


def _run_synthetic(
    request: WorkerRequest,
    checkpoint_bytes: bytes,
    config_bytes: bytes,
    progress: _WorkerProgress,
) -> WorkerResult:
    checkpoint = require_object(load_json_bytes(checkpoint_bytes), "synthetic checkpoint")
    checkpoint = strict_fields(checkpoint, {"format", "state"}, "synthetic checkpoint")
    if checkpoint["format"] != "voxelscope-safe-tensor-fixture/v1":
        raise EvidenceError("unsafe_checkpoint_object", "fixture format differs")
    config = strict_fields(
        require_object(load_json_bytes(config_bytes), "synthetic config"),
        {"expected_state", "format"},
        "synthetic config",
    )
    if config["format"] != "voxelscope-safe-loader-config/v1":
        raise EvidenceError("synthetic_config_mismatch", "fixture config differs")
    expected = _closed_synthetic_state(config["expected_state"])
    observed = _closed_synthetic_state(checkpoint["state"])
    if set(observed) != set(expected):
        raise EvidenceError("state_key_mismatch", "synthetic keys differ")
    if observed != expected:
        raise EvidenceError("tensor_metadata_mismatch", "synthetic tensor metadata differs")
    progress.runtime_verified = True
    progress.weights_only_used = True
    progress.state_mapping_closed = True
    progress.strict_keys_verified = True
    progress.tensor_metadata_verified = True
    progress.finite_parameters_verified = True
    progress.cpu_only_verified = True
    progress.architecture_verified = True
    progress.output_channels_verified = True
    progress.model_instantiated = True
    progress.state_loaded = True
    element_count = sum(
        _product(tuple(int(item) for item in tensor["shape"])) for tensor in observed.values()
    )
    return WorkerResult(
        "voxelscope/loader-worker-result/v1",
        "go",
        None,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        len(observed),
        0,
        element_count,
        True,
        True,
        False,
        False,
        False,
        False,
    )


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _verify_config(config_bytes: bytes, request: WorkerRequest) -> None:
    data = require_object(load_json_bytes(config_bytes), "inference config")
    network = require_object(data.get("network_def"), "network_def")
    expected = request.architecture
    target = network.get("_target_")
    allowed_targets = {
        "SegResNet",
        "monai.networks.nets.SegResNet",
        "$monai.networks.nets.SegResNet",
    }
    if (
        target not in allowed_targets
        or network.get("spatial_dims") != expected.spatial_dims
        or network.get("init_filters") != expected.init_filters
        or network.get("in_channels") != expected.in_channels
        or network.get("out_channels") != expected.out_channels
        or float(network.get("dropout_prob", -1.0)) != expected.dropout_prob
        or tuple(network.get("blocks_down", ())) != expected.blocks_down
        or tuple(network.get("blocks_up", ())) != expected.blocks_up
    ):
        raise EvidenceError("architecture_config_mismatch", "network_def differs")


def _closed_torch_mapping(value: Any, torch: Any) -> Mapping[str, Any]:
    if type(value) not in {dict, OrderedDict}:
        raise EvidenceError("unsafe_checkpoint_object", type(value).__name__)
    if set(value) != {"model"}:
        raise EvidenceError("checkpoint_container_mismatch", "expected exact model wrapper")
    state = value["model"]
    if type(state) not in {dict, OrderedDict} or not state:
        raise EvidenceError("unsafe_checkpoint_object", "model state must be a mapping")
    for key, tensor in state.items():
        if type(key) is not str or not key or type(tensor) is not torch.Tensor:
            raise EvidenceError("unsafe_checkpoint_object", "state must contain exact tensors")
    return cast(Mapping[str, Any], state)


def _distribution_identity(name: str, module: Any) -> str:
    distribution = importlib.metadata.distribution(name)
    environment_root = Path(sys.prefix).resolve(strict=True)
    origin = Path(module.__file__).resolve(strict=True)
    if not origin.is_relative_to(environment_root):
        raise EvidenceError("runtime_module_origin_mismatch", name)
    records: list[dict[str, Any]] = []
    files = distribution.files
    if not files:
        raise EvidenceError("runtime_manifest_missing", name)
    for item in sorted(files, key=lambda value: str(value)):
        path = Path(str(distribution.locate_file(item)))
        if path.is_symlink() or not path.is_file():
            raise EvidenceError("runtime_manifest_file_mismatch", str(item))
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(environment_root):
            raise EvidenceError("runtime_manifest_path_escape", str(item))
        digest = sha256_file(resolved)
        if item.hash is not None:
            if item.hash.mode != "sha256":
                raise EvidenceError("runtime_manifest_hash_algorithm", str(item))
            padding = "=" * (-len(item.hash.value) % 4)
            expected = base64.urlsafe_b64decode(item.hash.value + padding).hex()
            if digest != expected:
                raise EvidenceError("runtime_manifest_file_mismatch", str(item))
        if item.size is not None and resolved.stat().st_size != item.size:
            raise EvidenceError("runtime_manifest_file_mismatch", str(item))
        records.append({"path": str(item), "sha256": digest, "size_bytes": resolved.stat().st_size})
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "distribution": distribution.metadata["Name"],
                "files": records,
                "version": distribution.version,
            }
        )
    ).hexdigest()


def _runtime_is_exact(torch: Any, monai: Any, request: WorkerRequest) -> bool:
    torch_version = str(torch.__version__).split("+", 1)[0]
    torch_commit = str(getattr(torch.version, "git_version", ""))
    return (
        sys.platform.startswith("linux")
        and sys.version_info[:2] == (3, 12)
        and torch_version == request.runtime.torch_version
        and torch_commit == request.runtime.torch_commit
        and str(monai.__version__) == request.runtime.monai_version
        and hashlib.sha256(
            canonical_json_bytes(
                {
                    "monai": _distribution_identity("monai", monai),
                    "torch": _distribution_identity("torch", torch),
                }
            )
        ).hexdigest()
        == request.runtime_manifest_sha256
    )


def _run_official(
    request: WorkerRequest,
    checkpoint_stream: Any,
    config_bytes: bytes,
    progress: _WorkerProgress,
) -> WorkerResult:
    try:
        monai = importlib.import_module("monai")
        torch = importlib.import_module("torch")
        segresnet_constructor = importlib.import_module("monai.networks.nets").SegResNet
    except (ImportError, OSError) as exc:
        raise EvidenceError("qualification_runtime_unavailable", "pinned runtime missing") from exc
    if not _runtime_is_exact(torch, monai, request):
        raise EvidenceError("qualification_runtime_mismatch", "pinned runtime differs")
    progress.runtime_verified = True
    _verify_config(config_bytes, request)
    progress.architecture_verified = True
    progress.output_channels_verified = True
    contract = request.architecture
    model = segresnet_constructor(
        spatial_dims=contract.spatial_dims,
        init_filters=contract.init_filters,
        in_channels=contract.in_channels,
        out_channels=contract.out_channels,
        dropout_prob=contract.dropout_prob,
        blocks_down=contract.blocks_down,
        blocks_up=contract.blocks_up,
    )
    progress.model_instantiated = True
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise EvidenceError("accelerator_placement", "constructor did not remain on CPU")
    progress.cpu_only_verified = True
    try:
        progress.weights_only_used = True
        loaded = torch.load(
            checkpoint_stream,
            map_location="cpu",
            weights_only=True,
            mmap=False,
        )
    except Exception as exc:
        raise EvidenceError("weights_only_load_refused", type(exc).__name__) from exc
    state = _closed_torch_mapping(loaded, torch)
    progress.state_mapping_closed = True
    expected = model.state_dict()
    if set(state) != set(expected):
        raise EvidenceError("state_key_mismatch", "missing or unexpected state keys")
    progress.strict_keys_verified = True
    for key, expected_tensor in expected.items():
        tensor = state[key]
        if (
            tuple(tensor.shape) != tuple(expected_tensor.shape)
            or tensor.dtype != expected_tensor.dtype
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
        ):
            raise EvidenceError("tensor_metadata_mismatch", key)
        if not bool(torch.isfinite(tensor).all().item()):
            raise EvidenceError("nonfinite_parameter", key)
    progress.tensor_metadata_verified = True
    progress.finite_parameters_verified = True
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise EvidenceError("state_key_mismatch", "strict load returned incompatible keys")
    progress.state_loaded = True
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    all_tensors = (*parameters, *buffers)
    if any(tensor.device.type != "cpu" for tensor in all_tensors):
        raise EvidenceError("accelerator_placement", "loaded state is not CPU-only")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in all_tensors):
        raise EvidenceError("nonfinite_parameter", "loaded model contains nonfinite values")
    progress.parameter_tensor_count = len(parameters)
    progress.buffer_tensor_count = len(buffers)
    progress.parameter_element_count = sum(int(parameter.numel()) for parameter in parameters)
    result = WorkerResult(
        "voxelscope/loader-worker-result/v1",
        "go",
        None,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        len(parameters),
        len(buffers),
        sum(int(parameter.numel()) for parameter in parameters),
        True,
        True,
        False,
        False,
        False,
        False,
    )
    del state
    del loaded
    del expected
    del model
    gc.collect()
    return result


def execute_request(
    request: WorkerRequest,
    progress: _WorkerProgress | None = None,
) -> WorkerResult:
    progress = progress or _WorkerProgress()
    if sha256_file(Path(__file__)) != request.worker_source_sha256:
        raise EvidenceError("worker_source_identity_mismatch", "worker source differs")
    _install_network_refusal()
    checkpoint_stream, checkpoint_bytes = _read_verified_file(
        request.checkpoint_path,
        request.checkpoint_sha256,
        identity_code="checkpoint_identity_mismatch",
    )
    config_stream, config_bytes = _read_verified_file(
        request.config_path,
        request.config_sha256,
        identity_code="config_identity_mismatch",
    )
    try:
        if request.operation == "synthetic-json":
            return _run_synthetic(request, checkpoint_bytes, config_bytes, progress)
        _set_memory_limit(request.memory_limit_bytes)
        return _run_official(request, checkpoint_stream, config_bytes, progress)
    finally:
        checkpoint_stream.close()
        config_stream.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voxelscope.model_loader_worker")
    parser.add_argument("--request-fd", type=int, required=True)
    parser.add_argument("--result-fd", type=int, required=True)
    return parser


def _fatal_result(code: str) -> WorkerResult:
    return _refusal(code)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result: WorkerResult
    progress = _WorkerProgress()
    try:
        request = WorkerRequest.from_dict(
            require_object(
                load_json_bytes(_read_descriptor(args.request_fd, _MAX_REQUEST_BYTES)),
                "loader request",
            )
        )
        result = execute_request(request, progress)
    except EvidenceError as exc:
        result = progress.refusal(exc.code)
    except (ArithmeticError, KeyError, TypeError, ValueError):
        result = progress.refusal("malformed_worker_input")
    except BaseException:
        result = progress.refusal("loader_worker_interrupted")
    try:
        _write_descriptor(args.result_fd, result)
    except BaseException:
        return 1
    return 0 if result.status == "go" else 2


if __name__ == "__main__":
    raise SystemExit(main())
