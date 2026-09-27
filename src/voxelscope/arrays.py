# SPDX-License-Identifier: Apache-2.0
"""Canonical raw-array artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
)
from .records import ArrayArtifact

_ALLOWED = {"<f4", "<f8", "|u1"}


def canonical_array(value: npt.ArrayLike, dtype: str | None = None) -> np.ndarray[Any, Any]:
    array = np.asarray(value, dtype=np.dtype(dtype) if dtype else None)
    canonical_dtype = array.dtype.newbyteorder("<").str
    if canonical_dtype not in _ALLOWED:
        raise EvidenceError("unsupported_array_dtype", canonical_dtype)
    array = np.ascontiguousarray(array, dtype=np.dtype(canonical_dtype))
    if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
        raise EvidenceError("nonfinite_array", "arrays must be finite")
    return array


def array_content_sha256(value: npt.ArrayLike, dtype: str | None = None) -> str:
    array = canonical_array(value, dtype)
    descriptor = {"dtype": array.dtype.str, "order": "C", "shape": list(array.shape)}
    return sha256_bytes(
        b"voxelscope-array-v1\0"
        + canonical_json_bytes(descriptor)
        + b"\0"
        + array.tobytes(order="C")
    )


def write_array(path: Path, value: npt.ArrayLike, dtype: str | None = None) -> dict[str, Any]:
    array = canonical_array(value, dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes(order="C"))
    return {
        "content_sha256": array_content_sha256(array),
        "dtype": array.dtype.str,
        "file_sha256": sha256_file(path),
        "path": path.name,
        "shape": list(array.shape),
        "size_bytes": path.stat().st_size,
    }


def read_array(base: Path, identity: dict[str, Any] | ArrayArtifact) -> np.ndarray[Any, Any]:
    artifact = (
        identity if isinstance(identity, ArrayArtifact) else ArrayArtifact.from_dict(identity)
    )
    relative = safe_relative_path(artifact.path)
    path = ensure_no_symlink(base, relative)
    if not path.is_file():
        raise EvidenceError("missing_array", artifact.path)
    size = int(np.prod(artifact.shape, dtype=np.int64)) * np.dtype(artifact.dtype).itemsize
    if artifact.size_bytes != size or path.stat().st_size != size:
        raise EvidenceError("array_size_mismatch", artifact.path)
    if sha256_file(path) != artifact.file_sha256:
        raise EvidenceError("array_file_hash_mismatch", artifact.path)
    array = (
        np.frombuffer(path.read_bytes(), dtype=np.dtype(artifact.dtype))
        .reshape(artifact.shape)
        .copy()
    )
    if array_content_sha256(array) != artifact.content_sha256:
        raise EvidenceError("array_content_hash_mismatch", artifact.path)
    if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
        raise EvidenceError("nonfinite_array", artifact.path)
    return array
