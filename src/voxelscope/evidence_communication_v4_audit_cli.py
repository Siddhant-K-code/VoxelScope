# SPDX-License-Identifier: Apache-2.0
"""CLI for the deterministic post-hoc v4 lexical audit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .canonical import EvidenceError, sha256_file, write_json
from .evidence_communication_v4_audit import (
    AUDIT_RELATIVE_PATH,
    build_audit_record,
    verify_committed_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voxelscope.evidence_communication_v4_audit_cli"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--repository-root", type=Path, default=Path("."))
    verify.add_argument("--audit", type=Path, default=Path(AUDIT_RELATIVE_PATH))
    recompute = commands.add_parser("recompute")
    recompute.add_argument("--repository-root", type=Path, default=Path("."))
    recompute.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repository_root = args.repository_root.resolve()
        if args.command == "verify":
            path = args.audit if args.audit.is_absolute() else repository_root / args.audit
            record = verify_committed_audit(repository_root, path)
            counts = record["classification_summary"]["category_counts"]
            print(
                f"v4_lexical_audit=verified audit_sha256={sha256_file(path)} "
                f"entries={record['classification_summary']['total_classified_entries']} "
                f"affirmative_process={counts['affirmative_clinical_process_statement']} "
                f"ambiguous={counts['ambiguous_or_context_dependent']} "
                f"negation_or_disclaimer="
                f"{counts['explicit_negation_or_boundary_disclaimer']}"
            )
        elif args.command == "recompute":
            path = args.output if args.output.is_absolute() else repository_root / args.output
            if path.exists():
                raise EvidenceError("output_exists", str(path))
            record = build_audit_record(repository_root)
            write_json(path, record)
            print(
                f"v4_lexical_audit=recomputed audit_sha256={sha256_file(path)} "
                f"entries={record['classification_summary']['total_classified_entries']}"
            )
        else:
            raise EvidenceError("invalid_command", "unsupported command")
    except EvidenceError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR filesystem_error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
