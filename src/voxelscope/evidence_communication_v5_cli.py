# SPDX-License-Identifier: Apache-2.0
"""Offline CLI for the deterministic no-model v5 discourse planner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .canonical import EvidenceError
from .evidence_communication_v5 import (
    publish_control_bundle_v5,
    replay_control_bundle_v5,
    verify_contract_identities_v5,
)
from .evidence_communication_v5_records import SourceCustodyV5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voxelscope.evidence_communication_v5_cli",
        description="Deterministic v5 discourse planning without model or network access",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("contract-verify")
    verify.add_argument("--repository-root", type=Path, default=Path("."))

    compile_command = commands.add_parser("fixture-compile")
    compile_command.add_argument("--repository-root", type=Path, default=Path("."))
    compile_command.add_argument("--run-id", required=True)
    compile_command.add_argument("--output", type=Path, required=True)

    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    return parser


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise EvidenceError(
            "source_custody_unavailable",
            completed.stderr.strip() or "git command failed",
        )
    return completed.stdout.strip()


def source_custody_from_repository_v5(repository_root: Path) -> SourceCustodyV5:
    root = repository_root.resolve(strict=True)
    revision = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise EvidenceError(
            "source_tree_not_clean",
            "fixture publication requires a clean index and worktree",
        )
    return SourceCustodyV5(revision, "clean")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "contract-verify":
            identities = verify_contract_identities_v5(args.repository_root.resolve())
            print(
                "v5_contract=verified "
                + " ".join(f"{name}={value}" for name, value in sorted(identities.items()))
            )
            print("model_eligibility=model_inference_forbidden model_actions=0 network_actions=0")
        elif args.command == "fixture-compile":
            root = args.repository_root.resolve(strict=True)
            verify_contract_identities_v5(root)
            result = publish_control_bundle_v5(
                args.output,
                args.run_id,
                source_custody_from_repository_v5(root),
            )
            print(
                f"v5_publication_status={result.terminal_state} "
                f"request_sha256={result.request_sha256} "
                f"plan_sha256={result.plan_sha256} "
                f"output_sha256={result.output_sha256} "
                f"optional_utility={result.optional_utility} "
                f"emitted_utf8_bytes={result.emitted_utf8_bytes} "
                f"receipt_sha256={result.receipt_sha256}"
            )
            print(
                "evidence_status=contract_test_evidence "
                "model_eligibility=model_inference_forbidden "
                "model_actions=0 network_actions=0"
            )
        elif args.command == "replay":
            result = replay_control_bundle_v5(args.bundle)
            print(
                f"v5_replay=verified v5_publication_status={result.terminal_state} "
                f"request_sha256={result.request_sha256} "
                f"plan_sha256={result.plan_sha256} "
                f"output_sha256={result.output_sha256} "
                f"optional_utility={result.optional_utility} "
                f"emitted_utf8_bytes={result.emitted_utf8_bytes} "
                f"receipt_sha256={result.receipt_sha256}"
            )
            print("model_actions=0 network_actions=0")
        else:
            raise EvidenceError("invalid_command", "unsupported command")
    except EvidenceError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR io_error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
