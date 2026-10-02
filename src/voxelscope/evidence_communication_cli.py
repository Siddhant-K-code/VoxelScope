# SPDX-License-Identifier: Apache-2.0
"""Offline CLI for evidence communication compilation and benchmarking."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .canonical import EvidenceError
from .evidence_communication import (
    ModelRunner,
    OllamaRunner,
    RecordedDraftRunner,
    compile_model_draft,
    compile_recorded_fixture,
    replay_communication,
    run_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voxelscope.evidence_communication_cli")
    commands = parser.add_subparsers(dest="command", required=True)

    fixture = commands.add_parser("fixture-compile")
    fixture.add_argument("--atlas", type=Path, required=True)
    fixture.add_argument("--protein", required=True)
    fixture.add_argument("--request-id", required=True)
    fixture.add_argument(
        "--profile",
        choices=[
            "clean",
            "not_measured_as_negative",
            "prohibited_clinical_or_causal",
            "restricted_without_source",
            "unknown_source",
        ],
        default="clean",
    )
    fixture.add_argument("--output", type=Path, required=True)

    compile_command = commands.add_parser("compile")
    compile_command.add_argument("--atlas", type=Path, required=True)
    compile_command.add_argument("--protein", required=True)
    compile_command.add_argument("--request-id", required=True)
    compile_command.add_argument("--draft", type=Path, required=True)
    compile_command.add_argument("--output", type=Path, required=True)

    replay = commands.add_parser("replay")
    replay.add_argument("--atlas", type=Path, required=True)
    replay.add_argument("--bundle", type=Path, required=True)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--atlas", type=Path, required=True)
    benchmark.add_argument("--fixtures", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--runner", choices=["recorded", "ollama"], default="recorded")
    benchmark.add_argument("--endpoint", default="http://127.0.0.1:11434")
    benchmark.add_argument("--model")
    benchmark.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fixture-compile":
            result = compile_recorded_fixture(
                args.atlas,
                args.protein,
                args.request_id,
                args.profile,
                args.output,
            )
            print(
                f"communication_status={result.terminal_state} "
                f"artifact_sha256={result.artifact_sha256} "
                f"receipt_sha256={result.receipt_sha256}"
            )
        elif args.command == "compile":
            result = compile_model_draft(
                args.atlas,
                args.protein,
                args.request_id,
                args.draft,
                args.output,
            )
            print(
                f"communication_status={result.terminal_state} "
                f"artifact_sha256={result.artifact_sha256} "
                f"receipt_sha256={result.receipt_sha256}"
            )
        elif args.command == "replay":
            result = replay_communication(args.atlas, args.bundle)
            print(
                f"communication_replay=verified "
                f"communication_status={result.terminal_state} "
                f"artifact_sha256={result.artifact_sha256} "
                f"receipt_sha256={result.receipt_sha256}"
            )
        elif args.command == "benchmark":
            runner: ModelRunner
            if args.runner == "recorded":
                runner = RecordedDraftRunner()
            else:
                if args.model is None:
                    raise EvidenceError(
                        "missing_local_model",
                        "--model is required for the Ollama runner",
                    )
                runner = OllamaRunner(args.endpoint, args.model, args.timeout_seconds)
            benchmark = run_benchmark(args.atlas, args.fixtures, args.output, runner)
            counts = benchmark["metrics"]["verifier_counts"]
            print(
                f"benchmark_status=completed runs={counts['total']} "
                f"accepted_or_partially_excluded="
                f"{counts['accepted_or_partially_excluded']} "
                f"refused={counts['refused']}"
            )
            print(f"synthetic_only=true local_model_downloaded=false runner={args.runner}")
        else:
            raise EvidenceError("invalid_command", "unsupported command")
    except EvidenceError as exc:
        print(f"ERROR {exc.code}: {exc}", file=sys.stderr)
        return 2
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR malformed_communication: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR io_error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
