# SPDX-License-Identifier: Apache-2.0
"""Offline CLI for the public milestone 6 positional bridge."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .canonical import EvidenceError
from .milestone6_evidence import (
    build_milestone6_public_bundle,
    verify_milestone6_public_bundle,
)
from .window_bridge import verify_window_bridge_report


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voxelscope.milestone6_cli")
    commands = parser.add_subparsers(dest="command", required=True)

    bridge = commands.add_parser("window-bridge")
    bridge.add_argument("--plan", type=Path, required=True)
    bridge.add_argument("--report", type=Path, required=True)

    public_build = commands.add_parser("public-build")
    public_build.add_argument("--plan", type=Path, required=True)
    public_build.add_argument("--report", type=Path, required=True)
    public_build.add_argument("--output", type=Path, required=True)

    public_verify = commands.add_parser("public-verify")
    public_verify.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _repository_root()
    try:
        if args.command == "window-bridge":
            report = verify_window_bridge_report(
                args.plan,
                args.report,
                repository_root=root,
            )
            print(f"window_bridge_status={report.status} plan_sha256={report.plan_sha256}")
            print(
                "axis_mapping=go traversal=go symmetric_padding=go "
                "legacy_general_padded_reuse=no-go"
            )
            print("private_data_accessed=false inference_authorized=false")
        elif args.command == "public-build":
            digest = build_milestone6_public_bundle(
                args.plan,
                args.report,
                args.output,
                repository_root=root,
            )
            print(f"public_evidence_status=go bundle_sha256={digest}")
            print("legacy_general_padded_reuse=no-go inference_authorized=false")
        elif args.command == "public-verify":
            digest = verify_milestone6_public_bundle(
                args.bundle,
                repository_root=root,
            )
            print(f"verified_milestone6_public_bundle_sha256={digest}")
            print("inference_authorized=false")
        else:
            raise EvidenceError("invalid_command", "unsupported command")
    except EvidenceError as exc:
        print(f"ERROR {exc.code}", file=sys.stderr)
        return 2
    except (TypeError, KeyError, ValueError, ArithmeticError):
        print("ERROR malformed_evidence", file=sys.stderr)
        return 2
    except OSError:
        print("ERROR io_error", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
