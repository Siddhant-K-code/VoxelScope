# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import voxelscope.window_bridge as bridge_module
from voxelscope.canonical import EvidenceError, sha256_file, write_json
from voxelscope.milestone6_cli import main
from voxelscope.window_bridge import (
    TRUSTED_PLAN_SHA256,
    TRUSTED_REPORT_SHA256,
    apply_symmetric_padding,
    build_window_bridge_report,
    crop_slices,
    enumerate_bridge_windows,
    independent_axis_starts,
    load_window_bridge_plan,
    monai_axis_starts,
    monai_scan_interval,
    symmetric_padding,
    verify_window_bridge_report,
)
from voxelscope.window_bridge_records import WindowBridgePlan
from voxelscope.windows import axis_starts

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/window-bridge-plan-v1.json"
REPORT = ROOT / "research/window-bridge-report-v1.json"


def test_trusted_plan_and_report_verify_offline() -> None:
    verified = verify_window_bridge_report(PLAN, REPORT, repository_root=ROOT)
    assert verified.status == "go"
    assert verified.plan_sha256 == TRUSTED_PLAN_SHA256
    assert sha256_file(REPORT) == TRUSTED_REPORT_SHA256
    assert verified.padding_policies_equivalent is False
    assert verified.legacy_unpadded_enumeration_status == "go"
    assert verified.legacy_general_padded_reuse_status == "no-go"
    assert verified.inference_authorized is False


@pytest.mark.parametrize(
    ("source", "roi", "expected"),
    [
        (257, 240, (0, 17)),
        (501, 240, (0, 120, 240, 261)),
        (240, 240, (0,)),
        (155, 160, (0,)),
        (1, 160, (0,)),
    ],
)
def test_monai_axis_starts_match_independent_oracle(
    source: int,
    roi: int,
    expected: tuple[int, ...],
) -> None:
    assert monai_axis_starts(source, roi, 1, 2) == expected
    assert independent_axis_starts(source, roi, 1, 2) == expected


def test_scan_interval_uses_padded_single_window_rule() -> None:
    assert monai_scan_interval(155, 160, 1, 2) == 160
    assert monai_scan_interval(161, 160, 1, 2) == 80
    assert monai_scan_interval(10, 3, 2, 3) == 1


@pytest.mark.parametrize(
    ("source", "roi", "before", "after"),
    [
        (155, 160, 2, 3),
        (158, 160, 1, 1),
        (159, 160, 0, 1),
        (160, 160, 0, 0),
        (161, 160, 0, 0),
    ],
)
def test_symmetric_padding_matches_monai_split(
    source: int,
    roi: int,
    before: int,
    after: int,
) -> None:
    padding = symmetric_padding(source, roi)
    assert (padding.before, padding.after) == (before, after)


def test_traversal_is_lexicographic_ijk_with_k_fastest() -> None:
    source = (13, 15, 17)
    roi = (8, 9, 10)
    windows = enumerate_bridge_windows(source, roi, 1, 2)
    starts = tuple(
        monai_axis_starts(size, window, 1, 2) for size, window in zip(source, roi, strict=True)
    )
    assert tuple(item.padded_start_ijk for item in windows) == tuple(itertools.product(*starts))
    assert [item.padded_start_ijk for item in windows[:3]] == [
        (0, 0, 0),
        (0, 0, 5),
        (0, 0, 7),
    ]


def test_roi_components_map_positionally_without_transpose() -> None:
    source = (241, 242, 163)
    windows = enumerate_bridge_windows(source, (240, 240, 160), 1, 2)
    assert windows[0].padded_stop_ijk == (240, 240, 160)
    assert windows[-1].padded_start_ijk == (1, 2, 3)
    assert windows[-1].padded_stop_ijk == source


def test_symmetric_padding_and_crop_round_trip_asymmetric_tensor() -> None:
    array = np.arange(2 * 3 * 5 * 7, dtype=np.int32).reshape((2, 3, 5, 7))
    padded = apply_symmetric_padding(array, (8, 10, 12))
    assert padded.shape == (2, 8, 10, 12)
    assert np.array_equal(padded[(slice(None), *crop_slices((3, 5, 7), (8, 10, 12)))], array)
    assert np.count_nonzero(padded[:, :2, :, :]) == 0
    assert np.count_nonzero(padded[:, -3:, :, :]) == 0


def test_window_records_source_intersection_and_local_padding() -> None:
    (window,) = enumerate_bridge_windows((3, 5, 7), (8, 10, 12), 1, 2)
    assert window.source_start_ijk == (0, 0, 0)
    assert window.source_stop_ijk == (3, 5, 7)
    assert window.patch_pad_before_ijk == (2, 2, 2)
    assert window.patch_pad_after_ijk == (3, 3, 3)


def test_legacy_starts_agree_but_general_padding_does_not() -> None:
    source = (17, 19, 23)
    roi = (240, 240, 160)
    for size, window in zip(source, roi, strict=True):
        assert monai_axis_starts(size, window, 1, 2) == axis_starts(size, window, 1, 2)
    (bridge,) = enumerate_bridge_windows(source, roi, 1, 2)
    assert bridge.patch_pad_before_ijk != (0, 0, 0)
    assert bridge.patch_pad_after_ijk != tuple(
        window - size for size, window in zip(source, roi, strict=True)
    )


def test_historical_fixture_diff_one_still_has_high_side_only_padding() -> None:
    (bridge,) = enumerate_bridge_windows((7, 8, 9), (8, 8, 9), 1, 2)
    assert bridge.patch_pad_before_ijk == (0, 0, 0)
    assert bridge.patch_pad_after_ijk == (1, 0, 0)


@pytest.mark.parametrize(
    "helper_name",
    ["apply_symmetric_padding", "enumerate_bridge_windows"],
)
def test_report_go_depends_on_symmetric_bridge_helpers(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    def refuse(*args: object, **kwargs: object) -> object:
        raise EvidenceError("synthetic_symmetric_failure", helper_name)

    monkeypatch.setattr(bridge_module, helper_name, refuse)
    plan = load_window_bridge_plan(PLAN, repository_root=ROOT)
    with pytest.raises(EvidenceError) as captured:
        build_window_bridge_report(PLAN, plan)
    assert captured.value.code == "synthetic_symmetric_failure"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("axis_bindings", [], "unsafe_axis_mapping"),
        ("roi_ijk", [160, 240, 240], "unsafe_window_bridge_contract"),
        ("padding_policy", "right_zero", "unsafe_window_bridge_contract"),
        ("affine_axis_relabeling_allowed", True, "unsafe_window_bridge_policy"),
        ("inference_authorized", True, "unsafe_window_bridge_policy"),
        ("private_data_required", True, "unsafe_window_bridge_policy"),
    ],
)
def test_plan_refuses_axis_swaps_and_scope_expansion(
    tmp_path: Path,
    field: str,
    value: object,
    code: str,
) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload[field] = value
    candidate = tmp_path / "plan.json"
    write_json(candidate, payload)
    with pytest.raises(EvidenceError) as captured:
        WindowBridgePlan.from_dict(payload)
    assert captured.value.code == code


def test_plan_refuses_source_identity_drift(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["upstream_sources"][0]["sha256"] = "0" * 64
    candidate = tmp_path / "plan.json"
    write_json(candidate, payload)
    with pytest.raises(EvidenceError) as captured:
        load_window_bridge_plan(candidate, repository_root=ROOT)
    assert captured.value.code == "window_bridge_plan_digest_mismatch"


def test_plan_digest_is_approval_identity(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["observed_behavior"].append("tampered")
    candidate = tmp_path / "plan.json"
    write_json(candidate, payload)
    with pytest.raises(EvidenceError) as captured:
        load_window_bridge_plan(candidate, repository_root=ROOT)
    assert captured.value.code == "window_bridge_plan_digest_mismatch"


def test_report_rejects_success_shaped_tampering(tmp_path: Path) -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    payload["synthetic_case_count"] = 5
    candidate = tmp_path / "report.json"
    write_json(candidate, payload)
    with pytest.raises(EvidenceError) as captured:
        verify_window_bridge_report(PLAN, candidate, repository_root=ROOT)
    assert captured.value.code == "window_bridge_report_digest_mismatch"


def test_cli_reports_bounded_go_and_inference_no_go(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "window-bridge",
                "--plan",
                str(PLAN),
                "--report",
                str(REPORT),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "window_bridge_status=go" in captured.out
    assert "legacy_general_padded_reuse=no-go" in captured.out
    assert "private_data_accessed=false inference_authorized=false" in captured.out
    assert captured.err == ""


def test_public_records_contain_no_private_or_anatomical_observations() -> None:
    joined = PLAN.read_bytes() + REPORT.read_bytes()
    for forbidden in (
        b"/Users/",
        b"one-volume-custody",
        b"preprocessing-report",
        b"input-reference.f32le",
        b"input-independent.f32le",
        b'"affine":',
        b'"orientation":',
        b"4a73ef8f",
        b"0584b51b",
        b"ee268fb3",
    ):
        assert forbidden not in joined


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_shapes_refuse(value: int) -> None:
    with pytest.raises(EvidenceError) as captured:
        symmetric_padding(value, 8)
    assert captured.value.code == "invalid_shape"
    with pytest.raises(EvidenceError) as captured:
        monai_axis_starts(8, value, 1, 2)
    assert captured.value.code == "invalid_shape"


@pytest.mark.parametrize(("numerator", "denominator"), [(-1, 2), (1, 0), (1, 1), (2, 1)])
def test_invalid_overlap_refuses(numerator: int, denominator: int) -> None:
    with pytest.raises(EvidenceError) as captured:
        monai_scan_interval(16, 8, numerator, denominator)
    assert captured.value.code == "invalid_overlap"
