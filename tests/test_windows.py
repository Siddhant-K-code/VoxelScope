# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from voxelscope.canonical import EvidenceError
from voxelscope.records import OverlapRatio, WindowConfig
from voxelscope.windows import axis_starts, blend_map, build_window_evidence, identity_reconstruct


def config(
    shape: tuple[int, int, int] = (7, 8, 9),
    roi: tuple[int, int, int] = (8, 5, 6),
    mode: str = "constant",
) -> WindowConfig:
    return WindowConfig(
        shape,
        roi,
        (OverlapRatio(1, 2),) * 3,
        mode,  # type: ignore[arg-type]
        OverlapRatio(1, 8) if mode == "gaussian" else None,
    )


@st.composite
def geometries(draw: st.DrawFn) -> tuple[tuple[int, int, int], tuple[int, int, int], int, int]:
    shape = tuple(draw(st.integers(1, 12)) for _ in range(3))
    roi = tuple(draw(st.integers(1, 10)) for _ in range(3))
    denominator = draw(st.integers(1, 5))
    numerator = draw(st.integers(0, denominator - 1))
    return shape, roi, numerator, denominator  # type: ignore[return-value]


@given(geometries())
@settings(max_examples=60, deadline=None)
def test_arbitrary_geometry_has_complete_unique_lexicographic_coverage(
    geometry: tuple[tuple[int, int, int], tuple[int, int, int], int, int],
) -> None:
    shape, roi, numerator, denominator = geometry
    ratio = OverlapRatio(numerator, denominator)
    cfg = WindowConfig(shape, roi, (ratio, ratio, ratio), "constant", None)
    modalities = np.zeros((4, *shape), dtype=np.float32)
    evidence, _ = build_window_evidence(modalities, cfg)
    starts = [entry.start for entry in evidence.entries]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))
    assert [entry.index for entry in evidence.entries] == list(
        range(len(entries := evidence.entries))
    )
    coverage = np.zeros(shape, dtype=np.uint16)
    for entry in entries:
        slices = tuple(slice(entry.start[i], entry.source_end[i]) for i in range(3))
        coverage[slices] += 1
        assert tuple(entry.end[i] - entry.start[i] for i in range(3)) == roi
        assert entry.pad_before == (0, 0, 0)
    assert bool((coverage > 0).all())


def test_axis_contract_appends_end_anchor() -> None:
    assert axis_starts(10, 4, 1, 2) == (0, 2, 4, 6)
    assert axis_starts(11, 4, 1, 2) == (0, 2, 4, 6, 7)


def test_small_axis_uses_explicit_high_side_zero_padding() -> None:
    modalities = np.ones((4, 3, 4, 5), dtype=np.float32)
    evidence, _ = build_window_evidence(modalities, config((3, 4, 5), (5, 4, 6)))
    assert len(evidence.entries) == 1
    entry = evidence.entries[0]
    assert entry.start == (0, 0, 0)
    assert entry.end == (5, 4, 6)
    assert entry.source_end == (3, 4, 5)
    assert entry.pad_after == (2, 0, 1)


def test_gaussian_map_is_symmetric_positive_and_deterministic() -> None:
    cfg = config((8, 8, 8), (6, 5, 4), "gaussian")
    first = blend_map(cfg)
    second = blend_map(cfg)
    assert np.array_equal(first, second)
    assert float(first.max()) == 1.0
    assert bool((first > 0).all())
    assert np.allclose(first, first[::-1, ::-1, ::-1], rtol=1e-12, atol=0)
    modalities = np.zeros((4, 8, 8, 8), dtype=np.float32)
    evidence, _ = build_window_evidence(modalities, cfg)
    assert (
        evidence.weight_identity["array_content_sha256"]
        == "2d040f7f4a7e8cfc3da7396fb8eac1e30ab205c68d09f5bc35580150f3110f92"
    )
    assert (
        evidence.weight_identity["identity_sha256"]
        == "574832faeed1428bff7988b00492587aa6fb947cdc6f57eeeaff636d84e172a0"
    )


@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_identity_oracle_reconstructs_synthetic_volume(mode: str) -> None:
    rng = np.arange(4 * 7 * 8 * 9, dtype=np.float32).reshape(4, 7, 8, 9) / 100
    restored = identity_reconstruct(rng, config(mode=mode))
    assert np.allclose(restored, rng, rtol=1e-12, atol=1e-12)


def test_nonfinite_input_is_refused() -> None:
    volume = np.zeros((4, 7, 8, 9), dtype=np.float32)
    volume[0, 0, 0, 0] = np.nan
    with pytest.raises(EvidenceError) as caught:
        build_window_evidence(volume, config())
    assert caught.value.code == "nonfinite_array"


def test_unsupported_padding_is_refused() -> None:
    with pytest.raises(EvidenceError) as caught:
        replace(config(), padding_mode="symmetric")  # type: ignore[arg-type]
    assert caught.value.code == "unsupported_padding"
