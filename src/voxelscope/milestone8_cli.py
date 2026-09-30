# SPDX-License-Identifier: Apache-2.0
"""Offline CLI for Milestone 8 model-loading qualification."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .canonical import EvidenceError, load_json_bytes
from .milestone8_evidence import (
    build_milestone8_public_bundle,
    qualify_synthetic_model_loading,
    verify_milestone8_public_bundle,
)
from .model_loading import (
    execute_model_loading_qualification,
    verify_model_loading_qualification,
    verify_model_loading_refusal,
)
from .records import require_object, require_string, strict_fields

_MAX_PRIVATE_AUTHORIZATION_BYTES = 8192


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _private_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization-fd", type=int, required=True)


def _read_descriptor_object(descriptor: int) -> dict[str, Any]:
    if descriptor < 0:
        raise EvidenceError("unsafe_authorization_fd", "descriptor must be nonnegative")
    metadata = os.fstat(descriptor)
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISFIFO(metadata.st_mode)):
        raise EvidenceError("unsafe_authorization_fd", "regular file or pipe required")
    if (
        os.name != "nt"
        and stat.S_ISREG(metadata.st_mode)
        and (metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077)
    ):
        raise EvidenceError("private_authorization_permissions", "owner-only input required")
    chunks: list[bytes] = []
    size = 0
    while chunk := os.read(descriptor, _MAX_PRIVATE_AUTHORIZATION_BYTES + 1 - size):
        size += len(chunk)
        if size > _MAX_PRIVATE_AUTHORIZATION_BYTES:
            raise EvidenceError("private_authorization_limit", "authorization is too large")
        chunks.append(chunk)
    return require_object(load_json_bytes(b"".join(chunks)), "private authorization")


def _read_execute_authorization(descriptor: int) -> tuple[Path, Path, dict[str, str]]:
    value = strict_fields(
        _read_descriptor_object(descriptor),
        {
            "approve_archive_sha256",
            "approve_custody_receipt_sha256",
            "approve_plan_sha256",
            "approve_worker_python_sha256",
            "approve_worker_runtime_sha256",
            "root",
            "worker_python",
        },
        "private execution authorization",
    )
    return (
        Path(require_string(value["root"], "root")),
        Path(require_string(value["worker_python"], "worker_python")),
        {
            "approve_archive_sha256": require_string(
                value["approve_archive_sha256"], "approve_archive_sha256"
            ),
            "approve_custody_receipt_sha256": require_string(
                value["approve_custody_receipt_sha256"],
                "approve_custody_receipt_sha256",
            ),
            "approve_plan_sha256": require_string(
                value["approve_plan_sha256"], "approve_plan_sha256"
            ),
            "approve_worker_python_sha256": require_string(
                value["approve_worker_python_sha256"],
                "approve_worker_python_sha256",
            ),
            "approve_worker_runtime_sha256": require_string(
                value["approve_worker_runtime_sha256"],
                "approve_worker_runtime_sha256",
            ),
        },
    )


def _read_verify_authorization(descriptor: int) -> tuple[Path, dict[str, str]]:
    value = strict_fields(
        _read_descriptor_object(descriptor),
        {
            "approve_archive_sha256",
            "approve_custody_receipt_sha256",
            "approve_loading_report_sha256",
            "approve_plan_sha256",
            "approve_snapshot_sha256",
            "approve_worker_python_sha256",
            "approve_worker_runtime_sha256",
            "root",
        },
        "private verification authorization",
    )
    return (
        Path(require_string(value["root"], "root")),
        {
            key: require_string(value[key], key)
            for key in (
                "approve_archive_sha256",
                "approve_custody_receipt_sha256",
                "approve_loading_report_sha256",
                "approve_plan_sha256",
                "approve_snapshot_sha256",
                "approve_worker_python_sha256",
                "approve_worker_runtime_sha256",
            )
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voxelscope.milestone8_cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-qualify")
    _private_arguments(commands.add_parser("private-execute"))
    _private_arguments(commands.add_parser("private-verify"))
    _private_arguments(commands.add_parser("private-refusal"))
    public_build = commands.add_parser("public-build")
    public_build.add_argument("--plan", type=Path, required=True)
    public_build.add_argument("--output", type=Path, required=True)
    public_verify = commands.add_parser("public-verify")
    public_verify.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = _repository_root()
    try:
        if args.command == "synthetic-qualify":
            result = qualify_synthetic_model_loading(
                repository_root / "research/model-loading-plan-v2.json",
                repository_root=repository_root,
            )
            print(
                "synthetic_model_loading_status=go "
                f"extracted_member_count={result['synthetic_extracted_member_count']}"
            )
            print("private_archive_accessed=false inference_authorized=false")
        elif args.command == "private-execute":
            private_root, worker_python, approvals = _read_execute_authorization(
                args.authorization_fd
            )
            execute_model_loading_qualification(
                args.plan,
                private_root,
                worker_python=worker_python,
                repository_root=repository_root,
                **approvals,
            )
            print("private_model_loading_status=go forward_called=false")
            print("inference_run=false inference_authorized=false accelerator_used=false")
        elif args.command == "private-verify":
            private_root, approvals = _read_verify_authorization(args.authorization_fd)
            verify_model_loading_qualification(
                args.plan,
                private_root,
                repository_root=repository_root,
                **approvals,
            )
            print("private_model_loading_verification=go forward_called=false")
            print("inference_run=false inference_authorized=false accelerator_used=false")
        elif args.command == "private-refusal":
            private_root, worker_python, approvals = _read_execute_authorization(
                args.authorization_fd
            )
            del worker_python
            refusal = verify_model_loading_refusal(
                args.plan,
                private_root,
                repository_root=repository_root,
                **approvals,
            )
            print(f"private_model_loading_status=refused error_code={refusal.error_code}")
            print("inference_run=false inference_authorized=false")
        elif args.command == "public-build":
            digest = build_milestone8_public_bundle(
                args.plan, args.output, repository_root=repository_root
            )
            print(f"public_evidence_status=go bundle_sha256={digest}")
            print("private_archive_accessed=false inference_authorized=false")
        elif args.command == "public-verify":
            digest = verify_milestone8_public_bundle(args.bundle, repository_root=repository_root)
            print(f"verified_milestone8_public_bundle_sha256={digest}")
            print("private_archive_accessed=false inference_authorized=false")
        else:
            raise EvidenceError("invalid_command", "unsupported command")
    except EvidenceError as exc:
        print(f"ERROR {exc.code}", file=sys.stderr)
        return 2
    except (ArithmeticError, KeyError, TypeError, ValueError):
        print("ERROR malformed_evidence", file=sys.stderr)
        return 2
    except OSError:
        print("ERROR io_error", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
