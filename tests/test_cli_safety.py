# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

import voxelscope.cli as cli_module
from voxelscope.canonical import EvidenceError, load_json, write_json
from voxelscope.cli import main
from voxelscope.fixtures import build_fixture_bundle


@pytest.mark.parametrize(
    "unsafe", ["../volume-identity.json", "/tmp/volume.json", "C:/outside.json", "a//b.json"]
)
def test_windows_build_rejects_unsafe_volume_identity_paths(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    manifest_path = root / "study-manifest.json"
    manifest = load_json(manifest_path)
    manifest["volume_identity_path"] = unsafe
    write_json(manifest_path, manifest)
    output = tmp_path / "windows"
    assert (
        main(["windows", "build", "--manifest", str(manifest_path), "--output", str(output)]) == 2
    )
    assert not output.exists()


def test_windows_build_rejects_symlinked_volume_identity(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    source = root / "volume-identity.json"
    outside = tmp_path / "outside-volume.json"
    source.replace(outside)
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    output = tmp_path / "windows"
    assert (
        main(
            [
                "windows",
                "build",
                "--manifest",
                str(root / "study-manifest.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_windows_build_rejects_unsafe_array_path(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    volume_path = root / "volume-identity.json"
    volume = load_json(volume_path)
    volume["modalities"]["T1"]["path"] = "arrays\\t1.f32le"
    write_json(volume_path, volume)
    output = tmp_path / "windows"
    assert (
        main(
            [
                "windows",
                "build",
                "--manifest",
                str(root / "study-manifest.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_windows_build_failure_leaves_no_output_or_temporary_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    output = tmp_path / "windows"

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(cli_module, "write_array", fail_write)
    assert (
        main(
            [
                "windows",
                "build",
                "--manifest",
                str(root / "study-manifest.json"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    assert not list(tmp_path.glob(".windows.tmp-*"))


def test_drift_compare_interruption_leaves_no_output_or_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    output = tmp_path / "report.json"

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli_module._drift_compare(
            root / "outputs" / "reference" / "output.json",
            root / "outputs" / "identical" / "output.json",
            output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".report.json.tmp-*"))


@pytest.mark.parametrize(
    ("field", "expected_code"),
    [
        ("volume_identity_sha256", "incompatible_volume_identity"),
        ("model_identity_sha256", "incompatible_model_identity"),
        ("window_ledger_sha256", "incompatible_window_ledger"),
        ("run_id", "incompatible_run_identity"),
    ],
)
def test_drift_compare_rejects_incompatible_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, expected_code: str
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    candidate_path = root / "outputs" / "identical" / "output.json"
    candidate = load_json(candidate_path)
    candidate[field] = "f" * 64 if field.endswith("sha256") else "different-run"
    write_json(candidate_path, candidate)
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "drift",
                "compare",
                "--reference",
                str(root / "outputs" / "reference" / "output.json"),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert expected_code in capsys.readouterr().err
    assert not output.exists()


def test_drift_compare_allows_different_arm_identity(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    candidate_path = root / "outputs" / "identical" / "output.json"
    candidate = load_json(candidate_path)
    candidate["arm_id"] = "candidate-arm"
    write_json(candidate_path, candidate)
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "drift",
                "compare",
                "--reference",
                str(root / "outputs" / "reference" / "output.json"),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()


@pytest.mark.parametrize("command_kind", ["windows", "drift"])
def test_cli_treats_dangling_destination_symlink_as_occupied(
    tmp_path: Path, command_kind: str
) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    output = tmp_path / ("windows" if command_kind == "windows" else "report.json")
    try:
        output.symlink_to(
            tmp_path / "missing-target", target_is_directory=command_kind == "windows"
        )
    except OSError:
        pytest.skip("symlink creation unavailable")
    if command_kind == "windows":
        arguments = [
            "windows",
            "build",
            "--manifest",
            str(root / "study-manifest.json"),
            "--output",
            str(output),
        ]
    else:
        arguments = [
            "drift",
            "compare",
            "--reference",
            str(root / "outputs" / "reference" / "output.json"),
            "--candidate",
            str(root / "outputs" / "identical" / "output.json"),
            "--output",
            str(output),
        ]
    assert main(arguments) == 2
    assert output.is_symlink()


def test_windows_publish_race_does_not_replace_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voxelscope.atomic import rename_no_replace as real_publish

    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    output = tmp_path / "windows"

    def race(source: Path, destination: Path) -> None:
        destination.mkdir()
        real_publish(source, destination)

    monkeypatch.setattr(cli_module, "rename_no_replace", race)
    assert (
        main(
            [
                "windows",
                "build",
                "--manifest",
                str(root / "study-manifest.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.is_dir()
    assert not any(output.iterdir())
    assert not list(tmp_path.glob(".windows.tmp-*"))


def test_drift_publish_race_does_not_replace_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voxelscope.atomic import link_file_no_replace as real_publish

    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    output = tmp_path / "report.json"

    def race(source: Path, destination: Path) -> None:
        destination.write_bytes(b"concurrent writer")
        real_publish(source, destination)

    monkeypatch.setattr(cli_module, "link_file_no_replace", race)
    assert (
        main(
            [
                "drift",
                "compare",
                "--reference",
                str(root / "outputs" / "reference" / "output.json"),
                "--candidate",
                str(root / "outputs" / "identical" / "output.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == b"concurrent writer"
    assert not list(tmp_path.glob(".report.json.tmp-*"))


def test_drift_compare_rejects_incompatible_shape(tmp_path: Path) -> None:
    import numpy as np

    from voxelscope.fixtures import _write_output

    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    candidate_path = root / "outputs" / "identical" / "output.json"
    candidate = load_json(candidate_path)
    _write_output(
        candidate_path.parent,
        "identical",
        np.zeros((3, 6, 8, 9), dtype=np.float32),
        volume_identity_sha256=candidate["volume_identity_sha256"],
        model_identity_sha256=candidate["model_identity_sha256"],
        window_ledger_sha256=candidate["window_ledger_sha256"],
    )
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "drift",
                "compare",
                "--reference",
                str(root / "outputs" / "reference" / "output.json"),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_drift_compare_rejects_incompatible_spacing(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    candidate_path = root / "outputs" / "identical" / "output.json"
    candidate = load_json(candidate_path)
    candidate["spacing_mm"] = [2.0, 1.5, 2.0]
    write_json(candidate_path, candidate)
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "drift",
                "compare",
                "--reference",
                str(root / "outputs" / "reference" / "output.json"),
                "--candidate",
                str(candidate_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_drift_compare_checks_threshold_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    from voxelscope.bundle import load_output

    root = tmp_path / "bundle"
    build_fixture_bundle(root)
    reference = load_output(root / "outputs" / "reference" / "output.json")
    candidate_identity, candidate_array = load_output(
        root / "outputs" / "identical" / "output.json"
    )
    candidate_identity = replace(candidate_identity)
    object.__setattr__(candidate_identity, "threshold", object())
    values = iter((reference, (candidate_identity, candidate_array)))
    monkeypatch.setattr(cli_module, "load_output", lambda path: next(values))
    output = tmp_path / "report.json"
    with pytest.raises(EvidenceError) as caught:
        cli_module._drift_compare(Path("reference"), Path("candidate"), output)
    assert caught.value.code == "incompatible_threshold"
    assert not output.exists()
