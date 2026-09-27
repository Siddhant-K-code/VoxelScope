# SPDX-License-Identifier: Apache-2.0
"""Canonical raw-array artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .canonical import EvidenceError, canonical_json_bytes, sha256_bytes, sha256_file

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


def read_array(base: Path, identity: dict[str, Any]) -> np.ndarray[Any, Any]:
    required = {"content_sha256", "dtype", "file_sha256", "path", "shape", "size_bytes"}
    if set(identity) != required:
        raise EvidenceError("invalid_array_identity", "array identity fields differ")
    dtype = str(identity["dtype"])
    if dtype not in _ALLOWED:
        raise EvidenceError("unsupported_array_dtype", dtype)
    shape = tuple(int(item) for item in identity["shape"])
    if not shape or any(item <= 0 for item in shape):
        raise EvidenceError("invalid_array_shape", "shape values must be positive")
    relative = Path(str(identity["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceError("unsafe_path", "array path escapes descriptor directory")
    path = base / relative
    if path.is_symlink() or not path.is_file():
        raise EvidenceError("missing_array", str(relative))
    size = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if int(identity["size_bytes"]) != size or path.stat().st_size != size:
        raise EvidenceError("array_size_mismatch", str(relative))
    if sha256_file(path) != identity["file_sha256"]:
        raise EvidenceError("array_file_hash_mismatch", str(relative))
    array = np.frombuffer(path.read_bytes(), dtype=np.dtype(dtype)).reshape(shape).copy()
    if array_content_sha256(array) != identity["content_sha256"]:
        raise EvidenceError("array_content_hash_mismatch", str(relative))
    if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
        raise EvidenceError("nonfinite_array", str(relative))
    return array
