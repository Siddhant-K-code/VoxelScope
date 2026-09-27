# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from voxelscope.arrays import array_content_sha256, read_array, write_array
from voxelscope.canonical import EvidenceError, canonical_json_bytes, load_json_bytes


def test_canonical_json_is_sorted_compact_and_newline_terminated() -> None:
    assert canonical_json_bytes({"z": 1, "a": [True, None]}) == b'{"a":[true,null],"z":1}\n'


@pytest.mark.parametrize(
    "payload,code",
    [
        (b'{"a": 1}\n', "noncanonical_json"),
        (b'{"a":1,"a":2}\n', "duplicate_json_key"),
        (b'{"a":NaN}\n', "nonfinite_json_number"),
    ],
)
def test_json_loader_rejects_ambiguous_evidence(payload: bytes, code: str) -> None:
    with pytest.raises(EvidenceError) as caught:
        load_json_bytes(payload)
    assert caught.value.code == code


def test_canonical_json_rejects_nonfinite() -> None:
    with pytest.raises(EvidenceError, match="forbids"):
        canonical_json_bytes({"value": math.inf})


def test_array_round_trip_and_content_hash(tmp_path: Path) -> None:
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    identity = write_array(tmp_path / "array.f32le", source, "<f4")
    restored = read_array(tmp_path, identity)
    assert np.array_equal(restored, source)
    assert identity["content_sha256"] == array_content_sha256(source, "<f4")


def test_array_tamper_is_detected(tmp_path: Path) -> None:
    identity = write_array(tmp_path / "array.f32le", np.ones((2, 2), dtype=np.float32), "<f4")
    path = tmp_path / "array.f32le"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(EvidenceError) as caught:
        read_array(tmp_path, identity)
    assert caught.value.code == "array_file_hash_mismatch"


def test_nonfinite_array_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError) as caught:
        write_array(tmp_path / "bad.f32le", np.array([np.nan], dtype=np.float32), "<f4")
    assert caught.value.code == "nonfinite_array"
