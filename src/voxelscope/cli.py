# SPDX-License-Identifier: Apache-2.0
"""Offline VoxelScope command-line interface."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .arrays import read_array, write_array
from .atomic import link_file_no_replace, path_occupied, rename_no_replace
from .bundle import load_output, verify_bundle
from .canonical import (
    EvidenceError,
    ensure_no_symlink,
    load_json,
    safe_relative_path,
    sha256_file,
    write_json,
)
from .custody import (
    download_artifact,
    init_private_root,
    load_acquisition_plan,
    load_source_registry,
    render_plan,
    scan_public_tree,
    verify_and_receipt,
    verify_public_contracts,
)
from .drift import compare
from .fixtures import build_fixture_bundle
from .real_data_contract import verify_plan_contract
from .records import StudyManifest, VolumeIdentity
from .windows import build_window_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxelscope", description="Offline evidence and custody tooling"
    )
    top = parser.add_subparsers(dest="command", required=True)

    fixture = top.add_parser("fixture", help="build deterministic synthetic fixtures")
    fixture_sub = fixture.add_subparsers(dest="fixture_command", required=True)
    fixture_build = fixture_sub.add_parser("build")
    fixture_build.add_argument("--output", type=Path, required=True)

    windows = top.add_parser("windows", help="build deterministic window evidence")
    windows_sub = windows.add_subparsers(dest="windows_command", required=True)
    windows_build = windows_sub.add_parser("build")
    windows_build.add_argument("--manifest", type=Path, required=True)
    windows_build.add_argument("--output", type=Path, required=True)

    verify = top.add_parser("verify", help="verify a closed evidence bundle")
    verify.add_argument("--bundle", type=Path, required=True)

    drift = top.add_parser("drift", help="audit output drift")
    drift_sub = drift.add_subparsers(dest="drift_command", required=True)
    drift_compare = drift_sub.add_parser("compare")
    drift_compare.add_argument("--reference", type=Path, required=True)
    drift_compare.add_argument("--candidate", type=Path, required=True)
    drift_compare.add_argument("--output", type=Path, required=True)

    source = top.add_parser("source", help="verify public source contracts")
    source_sub = source.add_subparsers(dest="source_command", required=True)
    source_verify = source_sub.add_parser("verify")
    source_verify.add_argument("--registry", type=Path, required=True)

    custody = top.add_parser("custody", help="plan and verify private artifact custody")
    custody_sub = custody.add_subparsers(dest="custody_command", required=True)
    custody_plan = custody_sub.add_parser("plan")
    custody_plan.add_argument("--registry", type=Path, required=True)
    custody_plan.add_argument("--plan", type=Path, required=True)
    custody_init = custody_sub.add_parser("init")
    custody_init.add_argument("--root", type=Path, required=True)
    custody_verify = custody_sub.add_parser("verify")
    custody_verify.add_argument("--registry", type=Path, required=True)
    custody_verify.add_argument("--plan", type=Path, required=True)
    custody_verify.add_argument("--artifact-id", required=True)
    custody_verify.add_argument("--root", type=Path, required=True)
    custody_acquire = custody_sub.add_parser("acquire")
    custody_acquire.add_argument("--registry", type=Path, required=True)
    custody_acquire.add_argument("--plan", type=Path, required=True)
    custody_acquire.add_argument("--artifact-id", required=True)
    custody_acquire.add_argument("--root", type=Path, required=True)
    custody_acquire.add_argument("--allow-network", action="store_true")
    custody_scan = custody_sub.add_parser("scan-public")
    custody_scan.add_argument("--root", type=Path, required=True)
    return parser


def _windows_build(manifest_path: Path, output: Path) -> str:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvidenceError("unsafe_path", "manifest must be a regular file")
    evidence_root = manifest_path.parent
    manifest_data = load_json(manifest_path)
    if not isinstance(manifest_data, dict):
        raise EvidenceError("invalid_study_manifest", "manifest must be an object")
    manifest = StudyManifest.from_dict(manifest_data)
    volume_path = ensure_no_symlink(
        evidence_root, safe_relative_path(manifest.volume_identity_path)
    )
    if not volume_path.is_file():
        raise EvidenceError("missing_volume_identity", manifest.volume_identity_path)
    volume_data = load_json(volume_path)
    if not isinstance(volume_data, dict):
        raise EvidenceError("invalid_volume_identity", "volume identity must be an object")
    volume = VolumeIdentity.from_dict(volume_data)
    modalities = np.stack(
        [read_array(volume_path.parent, volume.modalities[name]) for name in volume.modality_order]
    )
    evidence, weights = build_window_evidence(modalities, manifest.window_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        weight_artifact = write_array(temporary / "weight-map.f64le", weights, "<f8")
        ledger = evidence.to_dict()
        ledger["weight_artifact"] = weight_artifact
        write_json(temporary / "window-ledger.json", ledger)
        rename_no_replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return evidence.coordinate_sha256


def _drift_compare(reference_path: Path, candidate_path: Path, output: Path) -> str:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    reference_identity, reference = load_output(reference_path)
    candidate_identity, candidate = load_output(candidate_path)
    if reference.shape != candidate.shape:
        raise EvidenceError("incompatible_output_shape", "spatial shapes differ")
    if reference_identity.spacing_mm != candidate_identity.spacing_mm:
        raise EvidenceError("incompatible_spacing", "spacing differs")
    if reference_identity.threshold != candidate_identity.threshold:
        raise EvidenceError("incompatible_threshold", "threshold policies differ")
    if reference_identity.volume_identity_sha256 != candidate_identity.volume_identity_sha256:
        raise EvidenceError("incompatible_volume_identity", "volume identities differ")
    if reference_identity.model_identity_sha256 != candidate_identity.model_identity_sha256:
        raise EvidenceError("incompatible_model_identity", "model identities differ")
    if reference_identity.window_ledger_sha256 != candidate_identity.window_ledger_sha256:
        raise EvidenceError("incompatible_window_ledger", "window ledgers differ")
    if reference_identity.run_id != candidate_identity.run_id:
        raise EvidenceError("incompatible_run_identity", "run identities differ")
    report = compare(
        reference,
        candidate,
        spacing_mm=reference_identity.spacing_mm,
        reference_output_sha256=sha256_file(reference_path),
        candidate_output_sha256=sha256_file(candidate_path),
        report_id=f"{reference_identity.output_id}-vs-{candidate_identity.output_id}",
        threshold=reference_identity.threshold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_json(temporary, report)
        report_sha256 = sha256_file(temporary)
        link_file_no_replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report_sha256


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fixture" and args.fixture_command == "build":
            root_hash = build_fixture_bundle(args.output)
            print(f"bundle_sha256={root_hash}")
            print((args.output / "SUMMARY.txt").read_text(encoding="ascii"), end="")
        elif args.command == "windows" and args.windows_command == "build":
            print(f"coordinate_sha256={_windows_build(args.manifest, args.output)}")
        elif args.command == "verify":
            print(f"verified_bundle_sha256={verify_bundle(args.bundle)}")
        elif args.command == "drift" and args.drift_command == "compare":
            print(f"report_sha256={_drift_compare(args.reference, args.candidate, args.output)}")
        elif args.command == "source" and args.source_command == "verify":
            print(f"verified_registry_sha256={verify_public_contracts(args.registry)}")
        elif args.command == "custody" and args.custody_command == "plan":
            registry = load_source_registry(args.registry)
            plan = load_acquisition_plan(args.plan)
            print(render_plan(plan, registry), end="")
        elif args.command == "custody" and args.custody_command == "init":
            init_private_root(args.root)
            print("private_root_initialized=true")
        elif args.command == "custody" and args.custody_command in {"verify", "acquire"}:
            registry = load_source_registry(args.registry)
            plan = load_acquisition_plan(args.plan)
            verify_plan_contract(plan, registry)
            artifact = next(
                (item for item in plan.artifacts if item.artifact_id == args.artifact_id),
                None,
            )
            if artifact is None:
                raise EvidenceError("unknown_artifact", args.artifact_id)
            if args.custody_command == "acquire":
                download_artifact(artifact, args.root, allow_network=args.allow_network)
            result = verify_and_receipt(plan, args.artifact_id, args.root)
            print(f"custody_status=verified artifact_id={args.artifact_id}")
            print(f"file_count={result.get('archive_file_count', 1)}")
        elif args.command == "custody" and args.custody_command == "scan-public":
            print(f"public_files_scanned={scan_public_tree(args.root)}")
        else:
            raise EvidenceError("invalid_command", "unsupported command")
    except EvidenceError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 2
    except (TypeError, KeyError, ValueError, ArithmeticError) as exc:
        print(f"ERROR malformed_evidence: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR io_error: {exc}", file=sys.stderr)
        return 1
    return 0
