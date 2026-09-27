# SPDX-License-Identifier: Apache-2.0
"""Strict canonical JSON and checksum primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast


class EvidenceError(ValueError):
    """Evidence violates a declared contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise EvidenceError("duplicate_json_key", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> NoReturn:
    raise EvidenceError("nonfinite_json_number", f"nonfinite JSON number: {value}")


def primitive(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return primitive(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise EvidenceError("non_string_json_key", "JSON keys must be strings")
        return {key: primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError("nonfinite_json_number", "canonical JSON forbids NaN and infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise EvidenceError("unsupported_json_type", type(value).__name__)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            primitive(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def load_json_bytes(data: bytes, *, require_canonical: bool = True) -> Any:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("invalid_json", str(exc)) from exc
    if require_canonical and data != canonical_json_bytes(value):
        raise EvidenceError("noncanonical_json", "JSON is not in VoxelScope canonical form")
    return value


def load_json(path: Path, *, require_canonical: bool = True) -> Any:
    return load_json_bytes(path.read_bytes(), require_canonical=require_canonical)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, field: str = "sha256") -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EvidenceError("invalid_sha256", f"{field} is not lowercase SHA-256")


_DOS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def safe_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise EvidenceError("unsafe_path", f"unsafe path: {value!r}")
    path = PurePosixPath(value)
    unsafe_component = any(
        part in {"", ".", ".."}
        or ":" in part
        or part.endswith((".", " "))
        or any(ord(character) < 32 or ord(character) == 127 for character in part)
        or part.split(".", 1)[0].upper() in _DOS_RESERVED_BASENAMES
        for part in path.parts
    )
    if path.is_absolute() or path.as_posix() != value or unsafe_component:
        raise EvidenceError("unsafe_path", f"unsafe path: {value!r}")
    return path


def is_link_like(path: Path) -> bool:
    """Return true for symlinks, junctions, and other Windows reparse points."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    if os.name == "nt":
        try:
            attributes = cast(Any, path.lstat()).st_file_attributes
        except (FileNotFoundError, AttributeError):
            return False
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def ensure_no_symlink(root: Path, relative: PurePosixPath) -> Path:
    if is_link_like(root):
        raise EvidenceError("symlink_forbidden", f"root path is link-like: {root}")
    current = root
    for part in relative.parts:
        current = current / part
        if is_link_like(current):
            raise EvidenceError("symlink_forbidden", f"link-like path forbidden: {relative}")
    return current
