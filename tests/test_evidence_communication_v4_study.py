# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from voxelscope import evidence_communication_v4 as v4_module
from voxelscope.canonical import (
    EvidenceError,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from voxelscope.evidence_communication_v4 import (
    OllamaRunnerV4,
    RecordedDraftRunnerV4,
    preflight_study_v4,
    run_benchmark_v4,
    study_size_preflight_v4,
)
from voxelscope.evidence_communication_v4_cli import main as v4_cli_main
from voxelscope.evidence_communication_v4_study_records import (
    ATTEMPT_MARKER_PATH_V4,
    BENCHMARK_OUTPUT_PATH_V4,
    DECLARATION_PATH_V4,
    DECLARATION_RECEIPT_PATH_V4,
    FROZEN_CLAIM_BOUNDARY,
    FROZEN_COMPARATOR,
    FROZEN_CONTRACTS,
    FROZEN_DESIGN,
    FROZEN_ENDPOINTS,
    FROZEN_ENVIRONMENT,
    FROZEN_EXECUTION,
    FROZEN_MODEL_RUNTIME,
    FROZEN_REPOSITORY,
    FROZEN_SIZE_PREFLIGHT,
    REQUIRED_SOURCE_ROLES,
    STUDY_DECLARATION_SCHEMA_V4,
    STUDY_ID_V4,
    ProspectiveStudyDeclarationV4,
    load_study_declaration_v4,
    verify_study_declaration_receipt_v4,
)
from voxelscope.gbm_atlas import assemble_gbm_evidence_atlas

ROOT = Path(__file__).resolve().parents[1]
ATLAS_MANIFEST = ROOT / "research/gbm-evidence-atlas-v1/source-manifest.json"
V4_FIXTURE = ROOT / "research/gbm-evidence-communication-benchmark-v4/benchmark-fixtures.json"


def _blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()


def _declaration_value() -> dict[str, Any]:
    sources = []
    for path_string, roles in sorted(REQUIRED_SOURCE_ROLES.items()):
        path = ROOT / path_string
        sources.append(
            {
                "git_blob_sha1": _blob_sha1(path),
                "path": path_string,
                "roles": sorted(roles),
                "sha256": sha256_file(path),
            }
        )
    return {
        "claim_boundary": list(FROZEN_CLAIM_BOUNDARY),
        "comparator": copy.deepcopy(FROZEN_COMPARATOR),
        "contracts": copy.deepcopy(FROZEN_CONTRACTS),
        "declared_at": "2026-10-03T15:09:11Z",
        "design": copy.deepcopy(FROZEN_DESIGN),
        "endpoints": [copy.deepcopy(item) for item in FROZEN_ENDPOINTS],
        "environment": copy.deepcopy(FROZEN_ENVIRONMENT),
        "execution": copy.deepcopy(FROZEN_EXECUTION),
        "model_runtime": copy.deepcopy(FROZEN_MODEL_RUNTIME),
        "repository": copy.deepcopy(FROZEN_REPOSITORY),
        "schema_version": STUDY_DECLARATION_SCHEMA_V4,
        "size_preflight": copy.deepcopy(FROZEN_SIZE_PREFLIGHT),
        "sources": sources,
        "status": "prospectively_frozen_pending_execution",
        "study_id": STUDY_ID_V4,
    }


def _mutate(value: dict[str, Any], path: tuple[str | int, ...], replacement: Any) -> None:
    target: Any = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement


def _mutation_paths(value: Any, prefix: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    if isinstance(value, dict):
        if not value:
            return [prefix]
        return [
            path for key, item in value.items() for path in _mutation_paths(item, (*prefix, key))
        ]
    if isinstance(value, list):
        if not value:
            return [prefix]
        return [
            path
            for index, item in enumerate(value)
            for path in _mutation_paths(item, (*prefix, index))
        ]
    return [prefix]


def _drifted_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + "-drift"
    if value is None:
        return "drift"
    if isinstance(value, list):
        return ["drift"]
    if isinstance(value, dict):
        return {"drift": True}
    raise AssertionError(f"unsupported mutation value: {value!r}")


def _materialize_study_root(path: Path) -> None:
    declaration = _declaration_value()
    source_paths = set(REQUIRED_SOURCE_ROLES)
    source_paths.update(item["path"] for item in declaration["comparator"]["artifacts"])
    source_paths.update({DECLARATION_PATH_V4, DECLARATION_RECEIPT_PATH_V4})
    for relative_path in sorted(source_paths):
        destination = path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_path, destination)


def test_exact_prospective_declaration_record_validates_sources() -> None:
    declaration = ProspectiveStudyDeclarationV4.from_dict(_declaration_value())

    declaration.verify_sources(ROOT)
    assert declaration.model_identity.model == "qwen3:8b-q8_0"
    assert declaration.model_identity.runtime_version == "0.35.1"
    assert declaration.runner_configuration.context_window == 16384
    assert declaration.runner_configuration.thinking_enabled is False
    assert len(declaration.endpoints) == 20


def test_comparator_matrix_conservatively_classifies_changed_contracts() -> None:
    matrix = {
        item["v4_field"]: item["classification"] for item in FROZEN_COMPARATOR["v3_to_v4_fields"]
    }

    assert matrix["emitted_fact_coverage"] == "directly_comparable"
    assert matrix["emitted_caveat_coverage"] == "directly_comparable"
    assert matrix["invalid_model_output_rate"] == "transformed"
    assert matrix["replay_semantic_variance"] == "transformed"
    assert matrix["verified_draft_fact_coverage"] == "transformed"
    assert matrix["verified_draft_caveat_retention"] == "transformed"
    assert matrix["accepted_partially_excluded_refused_terminal_outcomes"] == "transformed"
    assert matrix["latency_ms_input_tokens_output_tokens"] == "transformed"
    assert matrix["task_skeleton_coverage_unknown_duplicate_omitted_rates"] == "not_comparable"
    assert matrix["unsupported_claim_rate"] == "not_comparable"
    assert matrix["evidence_citation_validity"] == "not_comparable"
    assert matrix["peak_memory_mb_peak_metal_memory_mb"] == "not_comparable"


def test_committed_declaration_and_receipt_are_closed() -> None:
    declaration_path = ROOT / DECLARATION_PATH_V4
    declaration = load_study_declaration_v4(declaration_path, ROOT)
    receipt = verify_study_declaration_receipt_v4(
        ROOT / DECLARATION_RECEIPT_PATH_V4,
        declaration_path,
    )

    assert declaration.study_id == STUDY_ID_V4
    assert (
        receipt.declaration_sha256
        == "063b6055d430f14637d3dab05d1471cfc8e2fcf043b4030c633a956eff94a6c6"
    )


@pytest.mark.parametrize(
    ("path", "replacement", "error_code"),
    [
        (("repository", "base_commit"), "0" * 40, "repository_identity_drift"),
        (
            ("contracts", "prompts", 0, "sha256"),
            "0" * 64,
            "contract_identity_drift",
        ),
        (
            ("contracts", "transformation_id"),
            "different-compiler",
            "contract_identity_drift",
        ),
        (
            ("contracts", "verifier_version"),
            "different-verifier",
            "contract_identity_drift",
        ),
        (("design", "fixture_sha256"), "0" * 64, "study_design_drift"),
        (("design", "synthetic_atlas_sha256"), "0" * 64, "study_design_drift"),
        (("design", "total_request_count"), 17, "study_design_drift"),
        (
            ("model_runtime", "model_identity", "model_manifest_sha256"),
            "0" * 64,
            "model_runtime_drift",
        ),
        (
            ("model_runtime", "model_identity", "runtime_version"),
            "different",
            "model_runtime_drift",
        ),
        (
            ("model_runtime", "model_identity", "model"),
            "different",
            "model_runtime_drift",
        ),
        (
            ("model_runtime", "model_identity", "endpoint"),
            "http://localhost:11435",
            "model_runtime_drift",
        ),
        (
            ("model_runtime", "runner_configuration", "options", 0, "value"),
            8192,
            "model_runtime_drift",
        ),
        (
            ("model_runtime", "runner_configuration", "thinking_enabled"),
            True,
            "model_runtime_drift",
        ),
        (
            ("model_runtime", "protocol", "think_top_level"),
            True,
            "model_runtime_drift",
        ),
        (
            ("environment", "spend"),
            "unknown",
            "environment_declaration_drift",
        ),
        (
            ("endpoints", 0, "numerator"),
            "different numerator",
            "endpoint_definition_drift",
        ),
        (
            ("execution", "stop_rules", 0),
            "weakened stop rule",
            "execution_rule_drift",
        ),
        (
            ("comparator", "facts", "refused"),
            17,
            "v3_comparator_mutation",
        ),
    ],
)
def test_declaration_rejects_frozen_field_drift(
    path: tuple[str | int, ...],
    replacement: Any,
    error_code: str,
) -> None:
    value = _declaration_value()
    _mutate(value, path, replacement)

    with pytest.raises(EvidenceError) as caught:
        ProspectiveStudyDeclarationV4.from_dict(value)
    assert caught.value.code == error_code


def test_every_frozen_declaration_leaf_rejects_drift() -> None:
    original = _declaration_value()
    for path in _mutation_paths(original):
        changed = copy.deepcopy(original)
        current: Any = changed
        for component in path:
            current = current[component]
        _mutate(changed, path, _drifted_value(current))
        try:
            ProspectiveStudyDeclarationV4.from_dict(changed)
        except EvidenceError:
            continue
        pytest.fail(f"declaration mutation was accepted at {path!r}")


def test_declaration_rejects_unknown_fields_and_duplicate_paths() -> None:
    unknown = _declaration_value()
    unknown["unexpected"] = True
    with pytest.raises(EvidenceError) as caught:
        ProspectiveStudyDeclarationV4.from_dict(unknown)
    assert caught.value.code == "invalid_record"

    duplicate = _declaration_value()
    duplicate["sources"][1]["path"] = duplicate["sources"][0]["path"]
    with pytest.raises(EvidenceError) as caught:
        ProspectiveStudyDeclarationV4.from_dict(duplicate)
    assert caught.value.code == "duplicate_or_unsorted_source_path"

    unsafe = _declaration_value()
    unsafe["sources"][0]["path"] = "../outside.py"
    with pytest.raises(EvidenceError) as caught:
        ProspectiveStudyDeclarationV4.from_dict(unsafe)
    assert caught.value.code == "unsafe_path"


def test_declaration_rejects_source_digest_and_noncanonical_json(tmp_path: Path) -> None:
    value = _declaration_value()
    declaration = ProspectiveStudyDeclarationV4.from_dict(value)
    changed = list(declaration.sources)
    changed[0] = type(changed[0])(
        changed[0].git_blob_sha1,
        changed[0].path,
        changed[0].roles,
        "0" * 64,
    )
    drifted = ProspectiveStudyDeclarationV4(
        declaration.claim_boundary,
        declaration.comparator,
        declaration.contracts,
        declaration.declared_at,
        declaration.design,
        declaration.endpoints,
        declaration.environment,
        declaration.execution,
        declaration.model_runtime,
        declaration.repository,
        declaration.schema_version,
        declaration.size_preflight,
        tuple(changed),
        declaration.status,
        declaration.study_id,
    )
    with pytest.raises(EvidenceError) as caught:
        drifted.verify_sources(ROOT)
    assert caught.value.code == "study_source_sha256_mismatch"

    path = tmp_path / "noncanonical.json"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(EvidenceError) as caught:
        load_json(path)
    assert caught.value.code == "noncanonical_json"


def test_declaration_source_verification_rejects_symlink(tmp_path: Path) -> None:
    _materialize_study_root(tmp_path)
    relative_path = Path("src/voxelscope/atomic.py")
    candidate = tmp_path / relative_path
    candidate.unlink()
    try:
        candidate.symlink_to(ROOT / relative_path)
    except OSError:
        pytest.skip("symlink creation unavailable")

    declaration = ProspectiveStudyDeclarationV4.from_dict(_declaration_value())
    with pytest.raises(EvidenceError) as caught:
        declaration.verify_sources(tmp_path)
    assert caught.value.code == "symlink_forbidden"


def test_declaration_receipt_rejects_canonical_tampering(tmp_path: Path) -> None:
    declaration_path = tmp_path / "study-declaration.json"
    receipt_path = tmp_path / "study-declaration-receipt.json"
    shutil.copyfile(ROOT / DECLARATION_PATH_V4, declaration_path)
    shutil.copyfile(ROOT / DECLARATION_RECEIPT_PATH_V4, receipt_path)
    value = load_json(declaration_path)
    assert isinstance(value, dict)
    value["status"] = "tampered"
    write_json(declaration_path, value)

    with pytest.raises(EvidenceError) as caught:
        verify_study_declaration_receipt_v4(receipt_path, declaration_path)
    assert caught.value.code == "study_declaration_digest_mismatch"


class _BoundRunner:
    def __init__(
        self,
        declaration: ProspectiveStudyDeclarationV4,
        declaration_sha256: str,
        *,
        fail_validation: bool,
    ) -> None:
        self.identity = declaration.model_identity
        self.configuration = declaration.runner_configuration
        self.study_declaration_sha256 = declaration_sha256
        self.fail_validation = fail_validation
        self.validation_count = 0
        self.generation_count = 0

    def validate_identity(self) -> None:
        self.validation_count += 1
        if self.fail_validation:
            raise EvidenceError("runtime_identity_mismatch", "synthetic preflight failure")

    def run(self, *_: Any) -> Any:
        self.generation_count += 1
        raise AssertionError("generation must not be reached")


def test_target_model_requires_declaration_binding() -> None:
    with pytest.raises(EvidenceError) as caught:
        OllamaRunnerV4(
            "http://127.0.0.1:11434",
            "qwen3:8b-q8_0",
            "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130",
            "0.35.1",
            False,
            300.0,
            16384,
        )
    assert caught.value.code == "missing_study_declaration_binding"


def test_reserved_study_output_rejects_unbound_runner(tmp_path: Path) -> None:
    output = tmp_path / BENCHMARK_OUTPUT_PATH_V4

    with pytest.raises(EvidenceError) as caught:
        run_benchmark_v4(
            tmp_path / "atlas-not-read.json",
            V4_FIXTURE,
            output,
            RecordedDraftRunnerV4(),
        )

    assert caught.value.code == "missing_study_declaration_binding"
    assert not output.exists()
    assert not (tmp_path / ATTEMPT_MARKER_PATH_V4).exists()


def test_study_execution_rejects_unverified_declaration_digest(tmp_path: Path) -> None:
    _materialize_study_root(tmp_path)
    atlas, _ = assemble_gbm_evidence_atlas(ATLAS_MANIFEST)
    atlas_path = tmp_path / "atlas.json"
    write_json(atlas_path, atlas)
    declaration = ProspectiveStudyDeclarationV4.from_dict(_declaration_value())
    runner = _BoundRunner(declaration, "d" * 64, fail_validation=False)

    with pytest.raises(EvidenceError) as caught:
        run_benchmark_v4(
            atlas_path,
            V4_FIXTURE,
            tmp_path / BENCHMARK_OUTPUT_PATH_V4,
            runner,
            study_declaration=declaration,
            study_declaration_sha256="d" * 64,
            repository_root=tmp_path,
        )

    assert caught.value.code == "study_declaration_digest_mismatch"
    assert runner.generation_count == 0
    assert not (tmp_path / ATTEMPT_MARKER_PATH_V4).exists()


def test_cli_requires_explicit_declaration_digest_authorization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = v4_cli_main(
        [
            "benchmark",
            "--atlas",
            str(tmp_path / "atlas-not-read.json"),
            "--fixtures",
            str(V4_FIXTURE),
            "--output",
            str(tmp_path / BENCHMARK_OUTPUT_PATH_V4),
            "--runner",
            "ollama",
            "--declaration",
            DECLARATION_PATH_V4,
            "--repository-root",
            str(ROOT),
            "--authorize-study",
            STUDY_ID_V4,
            "--authorize-declaration-sha256",
            "0" * 64,
        ]
    )

    assert result == 2
    assert "ERROR study_execution_not_authorized" in capsys.readouterr().err
    assert not (tmp_path / BENCHMARK_OUTPUT_PATH_V4).exists()


def test_preflight_checks_identity_twice_without_generation() -> None:
    declaration = ProspectiveStudyDeclarationV4.from_dict(_declaration_value())
    runner = _BoundRunner(declaration, "d" * 64, fail_validation=False)

    result = preflight_study_v4(runner)  # type: ignore[arg-type]

    assert runner.validation_count == 2
    assert runner.generation_count == 0
    assert result["generation_requests"] == 0
    assert result["metadata_requests"] == [
        "GET /api/version",
        "GET /api/tags",
        "GET /api/version",
        "GET /api/tags",
    ]


def test_real_preflight_adapter_uses_only_four_read_only_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str, bytes | None]] = []

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    def respond(request: Any, _: float) -> Response:
        requests.append((request.get_method(), request.full_url, request.data))
        if request.full_url.endswith("/api/version"):
            return Response(canonical_json_bytes({"version": "0.35.1"}))
        if request.full_url.endswith("/api/tags"):
            return Response(
                canonical_json_bytes(
                    {
                        "models": [
                            {
                                "digest": (
                                    "e56358ca25dd14db6853a9f68a92d717"
                                    "aaa6f0a94250a72d1a0f3d86a9f30130"
                                ),
                                "model": "qwen3:8b-q8_0",
                                "name": "qwen3:8b-q8_0",
                            }
                        ]
                    }
                )
            )
        raise AssertionError(f"unexpected preflight URL: {request.full_url}")

    monkeypatch.setattr(v4_module, "_open_local_model_request", respond)
    runner = OllamaRunnerV4(
        "http://127.0.0.1:11434",
        "qwen3:8b-q8_0",
        "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130",
        "0.35.1",
        False,
        300.0,
        16384,
        study_declaration_sha256="d" * 64,
    )

    result = preflight_study_v4(runner)

    assert result["generation_requests"] == 0
    assert requests == [
        ("GET", "http://127.0.0.1:11434/api/version", None),
        ("GET", "http://127.0.0.1:11434/api/tags", None),
        ("GET", "http://127.0.0.1:11434/api/version", None),
        ("GET", "http://127.0.0.1:11434/api/tags", None),
    ]


def test_failed_bound_attempt_is_consumed_without_publication(tmp_path: Path) -> None:
    _materialize_study_root(tmp_path)
    atlas, _ = assemble_gbm_evidence_atlas(ATLAS_MANIFEST)
    atlas_path = tmp_path / "atlas.json"
    write_json(atlas_path, atlas)
    assert sha256_bytes(canonical_json_bytes(atlas)) == FROZEN_DESIGN["synthetic_atlas_sha256"]
    declaration = ProspectiveStudyDeclarationV4.from_dict(_declaration_value())
    declaration_sha256 = sha256_bytes(canonical_json_bytes(declaration.to_dict()))
    runner = _BoundRunner(declaration, declaration_sha256, fail_validation=True)
    output = tmp_path / BENCHMARK_OUTPUT_PATH_V4

    with pytest.raises(EvidenceError) as caught:
        run_benchmark_v4(
            atlas_path,
            V4_FIXTURE,
            output,
            runner,
            study_declaration=declaration,
            study_declaration_sha256=declaration_sha256,
            repository_root=tmp_path,
        )
    assert caught.value.code == "runtime_identity_mismatch"
    assert runner.generation_count == 0
    assert not output.exists()
    assert (tmp_path / ATTEMPT_MARKER_PATH_V4).is_file()

    with pytest.raises(EvidenceError) as repeated:
        run_benchmark_v4(
            atlas_path,
            V4_FIXTURE,
            output,
            runner,
            study_declaration=declaration,
            study_declaration_sha256=declaration_sha256,
            repository_root=tmp_path,
        )
    assert repeated.value.code == "study_attempt_already_consumed"


def test_size_preflight_measures_requests_without_generation(tmp_path: Path) -> None:
    atlas, _ = assemble_gbm_evidence_atlas(ATLAS_MANIFEST)
    atlas_path = tmp_path / "atlas.json"
    write_json(atlas_path, atlas)
    runner = OllamaRunnerV4(
        "http://127.0.0.1:11434",
        "qwen3:8b-q8_0",
        "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130",
        "0.35.1",
        False,
        300.0,
        16384,
        {
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
        },
        "d" * 64,
    )

    result = study_size_preflight_v4(atlas_path, V4_FIXTURE, runner)

    assert result == FROZEN_SIZE_PREFLIGHT
    assert result["generation_requests"] == 0
    assert result["observed_inference_counters"] == "unavailable_no_generation"
    assert len(result["cases"]) == 9
    assert all(item["skeleton_count"] == 15 for item in result["cases"])
    assert all(item["prompt_utf8_bytes"] > 0 for item in result["cases"])
    assert all(item["schema_canonical_json_bytes"] > 0 for item in result["cases"])
