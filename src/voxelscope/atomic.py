# SPDX-License-Identifier: Apache-2.0
"""Atomic no-clobber publication primitives for supported platforms."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

from .canonical import EvidenceError

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


def path_occupied(path: Path) -> bool:
    """Return true for any directory entry, including a dangling symlink."""
    return os.path.lexists(path)


def _raise_rename_error(destination: Path) -> None:
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise EvidenceError("output_exists", str(destination))
    raise OSError(error_number, os.strerror(error_number), str(destination))


def link_file_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a regular file by linking without replacement."""
    if path_occupied(destination):
        raise EvidenceError("output_exists", str(destination))
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise EvidenceError("output_exists", str(destination)) from exc
    except OSError as exc:
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise EvidenceError("output_exists", str(destination)) from exc
        raise
    source.unlink()


def rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish without replacing a concurrently created destination."""
    if path_occupied(destination):
        raise EvidenceError("output_exists", str(destination))
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise EvidenceError("output_exists", str(destination)) from exc
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(source_bytes, destination_bytes, _RENAME_EXCL) != 0:
            _raise_rename_error(destination)
        return
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise EvidenceError(
                "atomic_publish_unsupported", "renameat2 is unavailable on this Linux runtime"
            ) from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if rename(_AT_FDCWD, source_bytes, _AT_FDCWD, destination_bytes, _RENAME_NOREPLACE) != 0:
            _raise_rename_error(destination)
        return
    raise EvidenceError(
        "atomic_publish_unsupported", f"no no-replace primitive for platform {sys.platform}"
    )
