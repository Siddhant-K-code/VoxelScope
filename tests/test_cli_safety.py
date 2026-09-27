# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

import voxelscope.cli as cli_module
from voxelscope.canonical import load_json, write_json
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
