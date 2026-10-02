# SPDX-License-Identifier: Apache-2.0
"""Offline CLI for the synthetic glioblastoma evidence atlas."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .canonical import EvidenceError
from .gbm_atlas import build_gbm_evidence_atlas


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voxelscope.gbm_atlas_cli")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command != "build":
            raise EvidenceError("invalid_command", "unsupported command")
        result = build_gbm_evidence_atlas(args.manifest, args.output)
        print(
            f"gbm_atlas_status=built atlas_sha256={result.atlas_sha256} "
            f"receipt_sha256={result.receipt_sha256}"
        )
        print("synthetic_only=true ranking_performed=false")
    except EvidenceError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 2
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR malformed_evidence: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR io_error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
