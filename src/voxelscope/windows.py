# SPDX-License-Identifier: Apache-2.0
"""Deterministic sliding-window oracle for synthetic 3D arrays."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any

import numpy as np
import numpy.typing as npt

from .arrays import array_content_sha256, canonical_array
from .canonical import EvidenceError, canonical_json_bytes, sha256_bytes
from .records import WindowConfig, WindowLedgerEntry


@dataclass(frozen=True)
class WindowEvidence:
    config: WindowConfig
    entries: tuple[WindowLedgerEntry, ...]
    coordinate_sha256: str
    weight_identity: dict[str, Any]
    coverage_min: int
    coverage_max: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "coordinate_sha256": self.coordinate_sha256,
            "coverage_max": self.coverage_max,
            "coverage_min": self.coverage_min,
            "entries": [asdict(entry) for entry in self.entries],
            "schema_version": "voxelscope/v1",
            "weight_identity": self.weight_identity,
        }


def axis_starts(dimension: int, roi: int, numerator: int, denominator: int) -> tuple[int, ...]:
    if dimension <= 0 or roi <= 0:
        raise EvidenceError("invalid_geometry", "volume and ROI must be positive")
    if denominator <= 0 or not 0 <= numerator < denominator:
        raise EvidenceError("invalid_overlap", "overlap requires 0 <= p < q")
    stride = max(1, (roi * (denominator - numerator)) // denominator)
    final = max(dimension - roi, 0)
    starts = list(range(0, final + 1, stride)) or [0]
    if starts[-1] != final:
        starts.append(final)
    if len(starts) != len(set(starts)):
        raise EvidenceError("duplicate_window_coordinate", "duplicate axis start")
    return tuple(starts)


def blend_map(config: WindowConfig) -> np.ndarray[Any, Any]:
    if config.blend_mode == "constant":
        return np.ones(config.roi, dtype="<f8")
    if config.blend_mode != "gaussian" or config.sigma_scale is None:
        raise EvidenceError("unsupported_blend_mode", config.blend_mode)
    vectors: list[np.ndarray[Any, Any]] = []
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        two = Decimal(2)
        for length in config.roi:
            center = Decimal(length - 1) / two
            sigma = (
                Decimal(length)
                * Decimal(config.sigma_scale.numerator)
                / Decimal(config.sigma_scale.denominator)
            )
            if sigma <= 0:
                raise EvidenceError("invalid_sigma_scale", "sigma must be positive")
            values = [
                (-(((Decimal(index) - center) / sigma) ** 2) / two).exp() for index in range(length)
            ]
            maximum = max(values)
            vectors.append(np.asarray([float(value / maximum) for value in values], dtype="<f8"))
    weights = np.ascontiguousarray(
        np.multiply.outer(np.multiply.outer(vectors[0], vectors[1]), vectors[2]),
        dtype="<f8",
    )
    if not bool(np.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise EvidenceError("invalid_blend_weights", "weights must be finite and positive")
    if float(weights.max()) != 1.0:
        raise EvidenceError("invalid_blend_weights", "maximum must equal one")
    return weights


def weight_identity(config: WindowConfig, weights: np.ndarray[Any, Any]) -> dict[str, Any]:
    identity = {
        "array_content_sha256": array_content_sha256(weights, "<f8"),
        "axis_order": "zyx",
        "dtype": "<f8",
        "formula_version": "constant-v1" if config.blend_mode == "constant" else "gaussian-v1",
        "mode": config.blend_mode,
        "roi": list(config.roi),
        "sigma_scale": asdict(config.sigma_scale) if config.sigma_scale else None,
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    return identity


def extract_patch(
    modalities: np.ndarray[Any, Any],
    start: tuple[int, int, int],
    roi: tuple[int, int, int],
) -> tuple[np.ndarray[Any, Any], tuple[int, int, int], tuple[int, int, int]]:
    spatial = modalities.shape[1:]
    source_end = tuple(min(start[axis] + roi[axis], spatial[axis]) for axis in range(3))
    pad_after = tuple(start[axis] + roi[axis] - source_end[axis] for axis in range(3))
    patch = np.zeros((modalities.shape[0], *roi), dtype="<f4")
    source = tuple(slice(start[axis], source_end[axis]) for axis in range(3))
    target = tuple(slice(0, source_end[axis] - start[axis]) for axis in range(3))
    patch[(slice(None), *target)] = modalities[(slice(None), *source)]
    return patch, source_end, pad_after


def input_region_sha256(
    patch: np.ndarray[Any, Any],
    config: WindowConfig,
    start: tuple[int, int, int],
    source_end: tuple[int, int, int],
    pad_after: tuple[int, int, int],
) -> str:
    metadata = {
        "dtype": "<f4",
        "modality_order": ["T1", "T1c", "T2", "FLAIR"],
        "pad_after": list(pad_after),
        "pad_before": [0, 0, 0],
        "roi": list(config.roi),
        "source_end": list(source_end),
        "source_shape": list(config.volume_shape),
        "source_start": list(start),
    }
    return sha256_bytes(
        b"voxelscope-window-input-v1\0"
        + canonical_json_bytes(metadata)
        + b"\0"
        + patch.tobytes(order="C")
    )


def build_window_evidence(
    modalities: npt.ArrayLike, config: WindowConfig
) -> tuple[WindowEvidence, np.ndarray[Any, Any]]:
    volume = canonical_array(modalities, "<f4")
    if volume.ndim != 4 or volume.shape[0] != 4 or volume.shape[1:] != config.volume_shape:
        raise EvidenceError("invalid_volume_shape", "expected (4,Z,Y,X)")
    starts = [
        axis_starts(config.volume_shape[i], config.roi[i], ratio.numerator, ratio.denominator)
        for i, ratio in enumerate(config.overlap)
    ]
    coordinates = list(itertools.product(*starts))
    if len(coordinates) != len(set(coordinates)):
        raise EvidenceError("duplicate_window_coordinate", "coordinates are not unique")
    weights = blend_map(config)
    weight = weight_identity(config, weights)
    coverage = np.zeros(config.volume_shape, dtype=np.uint32)
    entries: list[WindowLedgerEntry] = []
    for index, raw_start in enumerate(coordinates):
        start = (int(raw_start[0]), int(raw_start[1]), int(raw_start[2]))
        end = (
            start[0] + config.roi[0],
            start[1] + config.roi[1],
            start[2] + config.roi[2],
        )
        patch, source_end, pad_after = extract_patch(volume, start, config.roi)
        source = tuple(slice(start[i], source_end[i]) for i in range(3))
        coverage[source] += 1
        entries.append(
            WindowLedgerEntry(
                index,
                start,
                end,
                start,
                source_end,
                (0, 0, 0),
                pad_after,
                config.roi,
                config.overlap,
                config.traversal_order,
                config.blend_mode,
                str(weight["identity_sha256"]),
                input_region_sha256(patch, config, start, source_end, pad_after),
            )
        )
    if not bool((coverage > 0).all()):
        raise EvidenceError("incomplete_window_coverage", "a source voxel is uncovered")
    coordinates_for_hash = [
        {
            "end": list(entry.end),
            "index": entry.index,
            "pad_after": list(entry.pad_after),
            "source_end": list(entry.source_end),
            "start": list(entry.start),
        }
        for entry in entries
    ]
    return (
        WindowEvidence(
            config,
            tuple(entries),
            sha256_bytes(canonical_json_bytes(coordinates_for_hash)),
            weight,
            int(coverage.min()),
            int(coverage.max()),
        ),
        weights,
    )


def identity_reconstruct(modalities: npt.ArrayLike, config: WindowConfig) -> np.ndarray[Any, Any]:
    volume = canonical_array(modalities, "<f4")
    evidence, weights = build_window_evidence(volume, config)
    accumulation = np.zeros(volume.shape, dtype="<f8")
    normalization = np.zeros(config.volume_shape, dtype="<f8")
    for entry in evidence.entries:
        patch, _, _ = extract_patch(volume, entry.start, config.roi)
        source_shape = tuple(entry.source_end[i] - entry.start[i] for i in range(3))
        local = tuple(slice(0, source_shape[i]) for i in range(3))
        source = tuple(slice(entry.start[i], entry.source_end[i]) for i in range(3))
        local_weights = weights[local]
        accumulation[(slice(None), *source)] += patch[(slice(None), *local)] * local_weights
        normalization[source] += local_weights
    if not bool((normalization > 0).all()):
        raise EvidenceError("incomplete_window_coverage", "normalization is zero")
    return accumulation / normalization[None, ...]
