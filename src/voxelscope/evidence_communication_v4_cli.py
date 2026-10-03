# SPDX-License-Identifier: Apache-2.0
"""CLI for the prospective v4 bounded evidence-communication compiler."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .canonical import EvidenceError, canonical_json_bytes
from .evidence_communication_v4 import (
    ModelRunnerV4,
    OllamaRunnerV4,
    RecordedDraftRunnerV4,
    compile_draft_plan_v4,
    compile_recorded_fixture_v4,
    preflight_study_v4,
    replay_benchmark_v4,
    replay_communication_v4,
    run_benchmark_v4,
    study_size_preflight_v4,
)
from .evidence_communication_v4_study_records import (
    DECLARATION_PATH_V4,
    DECLARATION_RECEIPT_PATH_V4,
    STUDY_ID_V4,
    ProspectiveStudyDeclarationV4,
    freeze_study_declaration_v4,
    load_study_declaration_v4,
    verify_study_declaration_receipt_v4,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voxelscope.evidence_communication_v4_cli")
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
    benchmark.add_argument("--endpoint")
    benchmark.add_argument("--model")
    benchmark.add_argument("--model-digest")
    benchmark.add_argument("--runtime-version")
    benchmark.add_argument("--thinking", choices=["disabled"])
    benchmark.add_argument("--timeout-seconds", type=float)
    benchmark.add_argument("--num-ctx", type=int)
    benchmark.add_argument("--declared-environment", action="append", default=[])
    benchmark.add_argument("--declaration", type=Path)
    benchmark.add_argument("--repository-root", type=Path, default=Path("."))
    benchmark.add_argument("--authorize-study")
    benchmark.add_argument("--authorize-declaration-sha256")

    benchmark_replay = commands.add_parser("benchmark-replay")
    benchmark_replay.add_argument("--atlas", type=Path, required=True)
    benchmark_replay.add_argument("--fixtures", type=Path, required=True)
    benchmark_replay.add_argument("--bundle", type=Path, required=True)
    benchmark_replay.add_argument("--declaration", type=Path)
    benchmark_replay.add_argument("--repository-root", type=Path, default=Path("."))

    declaration_verify = commands.add_parser("declaration-verify")
    declaration_verify.add_argument(
        "--declaration",
        type=Path,
        default=Path(DECLARATION_PATH_V4),
    )
    declaration_verify.add_argument("--repository-root", type=Path, default=Path("."))

    declaration_freeze = commands.add_parser("declaration-freeze")
    declaration_freeze.add_argument("--repository-root", type=Path, default=Path("."))

    preflight = commands.add_parser("preflight")
    preflight.add_argument(
        "--declaration",
        type=Path,
        default=Path(DECLARATION_PATH_V4),
    )
    preflight.add_argument("--repository-root", type=Path, default=Path("."))

    size_preflight = commands.add_parser("size-preflight")
    size_preflight.add_argument("--atlas", type=Path, required=True)
    size_preflight.add_argument("--fixtures", type=Path, required=True)
    size_preflight.add_argument(
        "--declaration",
        type=Path,
        default=Path(DECLARATION_PATH_V4),
    )
    size_preflight.add_argument("--repository-root", type=Path, default=Path("."))
    return parser


def _load_declared_study(
    repository_root: Path,
    declaration_path: Path,
) -> tuple[ProspectiveStudyDeclarationV4, str]:
    root = repository_root.resolve()
    path = declaration_path if declaration_path.is_absolute() else root / declaration_path
    declaration = load_study_declaration_v4(path, root)
    receipt = verify_study_declaration_receipt_v4(
        root / DECLARATION_RECEIPT_PATH_V4,
        path,
    )
    return declaration, receipt.declaration_sha256


def _declared_runner(
    declaration: ProspectiveStudyDeclarationV4,
    declaration_sha256: str,
) -> OllamaRunnerV4:
    identity = declaration.model_identity
    configuration = declaration.runner_configuration
    declared_environment = {item.name: item.value for item in configuration.declared_environment}
    return OllamaRunnerV4(
        identity.endpoint or "",
        identity.model,
        identity.model_manifest_sha256 or "",
        identity.runtime_version,
        configuration.thinking_enabled is True,
        configuration.timeout_seconds or 0.0,
        configuration.context_window,
        declared_environment,
        declaration_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fixture-compile":
            result = compile_recorded_fixture_v4(
                args.atlas,
                args.protein,
                args.request_id,
                args.profile,
                args.output,
            )
            print(
                f"communication_v4_status={result.terminal_state} "
                f"artifact_sha256={result.artifact_sha256} "
                f"receipt_sha256={result.receipt_sha256}"
            )
        elif args.command == "compile":
            result = compile_draft_plan_v4(
                args.atlas,
                args.protein,
                args.request_id,
                args.draft,
                args.output,
            )
            print(
                f"communication_v4_status={result.terminal_state} "
                f"artifact_sha256={result.artifact_sha256} "
                f"receipt_sha256={result.receipt_sha256}"
            )
        elif args.command == "replay":
            result = replay_communication_v4(args.atlas, args.bundle)
            print(
                f"communication_v4_replay=verified "
                f"communication_v4_status={result.terminal_state} "
                f"artifact_sha256={result.artifact_sha256} "
                f"receipt_sha256={result.receipt_sha256}"
            )
        elif args.command == "benchmark":
            runner: ModelRunnerV4
            declaration: ProspectiveStudyDeclarationV4 | None = None
            declaration_sha256: str | None = None
            if args.runner == "recorded":
                runner = RecordedDraftRunnerV4()
            else:
                if args.declaration is None:
                    raise EvidenceError(
                        "missing_study_declaration_binding",
                        "--declaration is required for the Ollama runner",
                    )
                if args.authorize_study != STUDY_ID_V4:
                    raise EvidenceError(
                        "study_execution_not_authorized",
                        f"--authorize-study must equal {STUDY_ID_V4}",
                    )
                if (
                    any(
                        value is not None
                        for value in (
                            args.endpoint,
                            args.model,
                            args.model_digest,
                            args.runtime_version,
                            args.thinking,
                            args.timeout_seconds,
                            args.num_ctx,
                        )
                    )
                    or args.declared_environment
                ):
                    raise EvidenceError(
                        "declaration_override_forbidden",
                        "Ollama identity and options come only from the declaration",
                    )
                declaration, declaration_sha256 = _load_declared_study(
                    args.repository_root,
                    args.declaration,
                )
                if args.authorize_declaration_sha256 != declaration_sha256:
                    raise EvidenceError(
                        "study_execution_not_authorized",
                        "--authorize-declaration-sha256 must equal the verified declaration digest",
                    )
                runner = _declared_runner(declaration, declaration_sha256)
            benchmark = run_benchmark_v4(
                args.atlas,
                args.fixtures,
                args.output,
                runner,
                study_declaration=declaration,
                study_declaration_sha256=declaration_sha256,
                repository_root=(
                    args.repository_root.resolve() if declaration is not None else None
                ),
            )
            counts = benchmark["metrics"]["verifier_counts"]
            print(
                f"benchmark_v4_status=completed runs={counts['total']} "
                f"accepted_or_partially_excluded="
                f"{counts['accepted_or_partially_excluded']} "
                f"refused={counts['refused']}"
            )
            print(f"synthetic_only=true local_model_downloaded=false runner={args.runner}")
        elif args.command == "benchmark-replay":
            declaration_sha256 = None
            if args.declaration is not None:
                _, declaration_sha256 = _load_declared_study(
                    args.repository_root,
                    args.declaration,
                )
            benchmark = replay_benchmark_v4(
                args.atlas,
                args.fixtures,
                args.bundle,
                study_declaration_sha256=declaration_sha256,
            )
            counts = benchmark["metrics"]["verifier_counts"]
            print(
                f"benchmark_v4_replay=verified runs={counts['total']} "
                f"accepted_or_partially_excluded="
                f"{counts['accepted_or_partially_excluded']} "
                f"refused={counts['refused']}"
            )
        elif args.command == "declaration-verify":
            declaration, declaration_sha256 = _load_declared_study(
                args.repository_root,
                args.declaration,
            )
            print(
                f"study_declaration=verified study_id={declaration.study_id} "
                f"declaration_sha256={declaration_sha256} "
                f"source_count={len(declaration.sources)}"
            )
        elif args.command == "declaration-freeze":
            receipt = freeze_study_declaration_v4(args.repository_root)
            print(
                f"study_declaration=frozen study_id={receipt.study_id} "
                f"declaration_sha256={receipt.declaration_sha256} "
                "generation_requests=0"
            )
        elif args.command == "preflight":
            declaration, declaration_sha256 = _load_declared_study(
                args.repository_root,
                args.declaration,
            )
            runner = _declared_runner(declaration, declaration_sha256)
            preflight_result = preflight_study_v4(runner)
            print(
                f"study_preflight={preflight_result['status']} "
                f"checks={preflight_result['checks']} "
                f"study_id={preflight_result['study_id']} "
                f"runtime={preflight_result['runtime']} "
                f"runtime_version={preflight_result['runtime_version']} "
                f"model={preflight_result['model']} "
                f"model_manifest_sha256={preflight_result['model_manifest_sha256']} "
                f"endpoint_locality={preflight_result['endpoint_locality']} "
                "metadata_requests=GET_/api/version,GET_/api/tags "
                f"generation_requests={preflight_result['generation_requests']}"
            )
        elif args.command == "size-preflight":
            declaration, declaration_sha256 = _load_declared_study(
                args.repository_root,
                args.declaration,
            )
            runner = _declared_runner(declaration, declaration_sha256)
            size_result = study_size_preflight_v4(
                args.atlas,
                args.fixtures,
                runner,
            )
            sys.stdout.buffer.write(canonical_json_bytes(size_result))
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
