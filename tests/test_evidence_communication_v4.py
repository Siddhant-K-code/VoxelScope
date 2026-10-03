# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import dataclasses
import json
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from voxelscope import evidence_communication_v4 as v4_module
from voxelscope.canonical import (
    EvidenceError,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    write_json,
)
from voxelscope.evidence_benchmark_records import RunnerConfiguration
from voxelscope.evidence_communication import replay_benchmark as replay_benchmark_v3
from voxelscope.evidence_communication_records import ModelIdentity
from voxelscope.evidence_communication_v4 import (
    DIRECTIONAL_SCOPE_TEXT,
    SOURCE_TARGET_BOUNDARY_TEXT,
    OllamaRunnerV4,
    RecordedDraftRunnerV4,
    RunnerResultV4,
    _profile_prompt_identity_v4,
    _recorded_output,
    _terminal_outcome_bucket_v4,
    compile_recorded_fixture_v4,
    derive_claim_skeletons,
    derive_communication_request_v4,
    draft_plan_json_schema,
    parse_model_draft_plan,
    replay_benchmark_v4,
    replay_communication_v4,
    run_benchmark_v4,
    verify_draft_plan_v4,
)
from voxelscope.evidence_communication_v4_records import (
    CommunicationRequestV4,
    DraftPlanEnvelopeV4,
    InvalidModelOutputV4,
)
from voxelscope.gbm_atlas import assemble_gbm_evidence_atlas

ROOT = Path(__file__).resolve().parents[1]
ATLAS_FIXTURE = ROOT / "research/gbm-evidence-atlas-v1/source-manifest.json"
V3_BENCHMARK_FIXTURE = (
    ROOT / "research/gbm-evidence-communication-benchmark-v3/benchmark-fixtures.json"
)
V3_OBSERVED_BUNDLE = ROOT / "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/benchmark"
V4_BENCHMARK_FIXTURE = (
    ROOT / "research/gbm-evidence-communication-benchmark-v4/benchmark-fixtures.json"
)
V4_OBSERVED_STUDY = ROOT / "research/gbm-evidence-communication-qwen3-8b-q8-study-v4"
V4_OBSERVED_BUNDLE = V4_OBSERVED_STUDY / "benchmark"
OLLAMA_DIGEST = "a" * 64
OLLAMA_VERSION = "0.35.1"


def _atlas() -> tuple[dict[str, Any], str]:
    atlas, _ = assemble_gbm_evidence_atlas(ATLAS_FIXTURE)
    return atlas, sha256_bytes(canonical_json_bytes(atlas))


def _atlas_path(tmp_path: Path) -> Path:
    atlas, _ = _atlas()
    path = tmp_path / "atlas.json"
    write_json(path, atlas)
    return path


def _request(
    protein_id: str = "P60484",
    profile: str = "clean",
) -> tuple[dict[str, Any], Any]:
    atlas, atlas_sha256 = _atlas()
    request = derive_communication_request_v4(
        atlas,
        atlas_sha256,
        protein_id,
        f"test-v4:{protein_id}:{profile}",
        _profile_prompt_identity_v4(profile),
    )
    return atlas, request


def _valid_draft(
    protein_id: str = "P60484",
    profile: str = "clean",
) -> tuple[dict[str, Any], Any, DraftPlanEnvelopeV4]:
    atlas, request = _request(protein_id, profile)
    result = RecordedDraftRunnerV4().run(request, "clean", 0)
    assert result.envelope is not None
    return atlas, request, result.envelope


class _FakeHttpResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _ollama_responder(
    generation: bytes,
    *,
    digest_by_validation: dict[int, str] | None = None,
    version_by_validation: dict[int, str] | None = None,
    tag_extra: dict[str, Any] | None = None,
    generated_requests: list[dict[str, Any]] | None = None,
    generation_error: Exception | None = None,
) -> Any:
    counts = {"tags": 0, "version": 0}

    def respond(
        request: urllib.request.Request,
        timeout_seconds: float | None = None,
    ) -> _FakeHttpResponse:
        del timeout_seconds
        if request.full_url.endswith("/api/version"):
            counts["version"] += 1
            version = (version_by_validation or {}).get(
                counts["version"],
                OLLAMA_VERSION,
            )
            return _FakeHttpResponse(canonical_json_bytes({"version": version}))
        if request.full_url.endswith("/api/tags"):
            counts["tags"] += 1
            digest = (digest_by_validation or {}).get(
                counts["tags"],
                OLLAMA_DIGEST,
            )
            return _FakeHttpResponse(
                canonical_json_bytes(
                    {
                        "models": [
                            {
                                "digest": digest,
                                "model": "installed-test-model",
                                "name": "installed-test-model",
                                **(tag_extra or {}),
                            }
                        ]
                    }
                )
            )
        if generation_error is not None:
            raise generation_error
        if generated_requests is not None:
            assert request.data is not None
            value = load_json_bytes_uncanonical(request.data)
            generated_requests.append(value)
        return _FakeHttpResponse(generation)

    return respond


def load_json_bytes_uncanonical(data: bytes) -> dict[str, Any]:
    from voxelscope.canonical import load_json_bytes

    value = load_json_bytes(data, require_canonical=False)
    assert isinstance(value, dict)
    return value


def _ollama_runner() -> OllamaRunnerV4:
    return OllamaRunnerV4(
        "http://127.0.0.1:11434",
        "installed-test-model",
        OLLAMA_DIGEST,
        OLLAMA_VERSION,
        False,
        context_window=4096,
        declared_environment={"OLLAMA_TEST": "1"},
    )


def _ollama_response(response: str, **extra: Any) -> bytes:
    return canonical_json_bytes(
        {
            "done": True,
            "eval_count": 3,
            "model": "installed-test-model",
            "prompt_eval_count": 5,
            "response": response,
            **extra,
        }
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _recorded_benchmark(tmp_path: Path, name: str = "benchmark") -> tuple[Path, Path]:
    atlas_path = _atlas_path(tmp_path)
    bundle = tmp_path / name
    run_benchmark_v4(
        atlas_path,
        V4_BENCHMARK_FIXTURE,
        bundle,
        RecordedDraftRunnerV4(),
    )
    return atlas_path, bundle


def _reclose_file(bundle: Path, relative_path: str) -> None:
    receipt = load_json(bundle / "receipt.json")
    assert isinstance(receipt, dict)
    digest = sha256_bytes((bundle / relative_path).read_bytes())
    for item in receipt["closed_files"]:
        assert isinstance(item, dict)
        if item["path"] == relative_path:
            item["sha256"] = digest
            break
    else:
        raise AssertionError(relative_path)
    if relative_path == "benchmark.json":
        receipt["benchmark_sha256"] = digest
    if relative_path == "index.json":
        receipt["index_sha256"] = digest
    write_json(bundle / "receipt.json", receipt)


def test_exact_skeleton_derivation_is_stable_and_covers_plan_once() -> None:
    _, request = _request()
    requirement_ids = [item.requirement_id for item in request.plan]
    covered = [
        requirement_id
        for skeleton in request.skeletons
        for requirement_id in skeleton.requirement_ids
    ]

    assert len(request.plan) == 17
    assert len(request.skeletons) == 15
    assert set(covered) == set(requirement_ids)
    assert len(covered) == len(set(covered))
    assert tuple(item.skeleton_id for item in request.skeletons) == (
        "skeleton:81cc782873e0d4a10a711ab071669885d4f54ca431f22eff99f3d0c74d450033",
        "skeleton:dc90487d656b4b7149ba86cb0d4d9ede89e0a0cc937d55a5aa76c9a102858561",
        "skeleton:54e41ec698f133e4461e039465a25d55803b775f69b4ebbba8ab4da924445952",
        "skeleton:b126d44c4f275a04e7c696f5c08db74e9a299f71409d071b018c78d30afb4d47",
        "skeleton:7670d745d02c430884c7c6e6129d0f03fdbcbfc1b99d3e74d37d508c7227a60f",
        "skeleton:2efbc8e1187483442b7aca02286b85ca0913b801969494b70b17645986796c38",
        "skeleton:2f42758146a3253f7059ccc6031bddf132ae0586eff6433fcd3ed935475543b2",
        "skeleton:1733782a14552f566f76d7008488300a0950f6415d304e13168ad3f4556c27bd",
        "skeleton:c68441023bca72709ed76a54fb4557d6c33f66578d2daba705f9d910b971823a",
        "skeleton:3f4ebd567fae378cd252b91488d916c28f88f95184788e12ab67f49cce7dd5d8",
        "skeleton:2e2f09041a4a03a4279cfb21976ed9aba0cdb7a1e7f85527068e21068a906434",
        "skeleton:a0111179de649e0b38004f2165bf6785b7668e42ce7ee9957de456bd5b61989c",
        "skeleton:72b6bd70cfa3e26772457a8fab67b5c203d20086326ef7cf718babf305dbfc10",
        "skeleton:a40d4d11aa2357c9b909498d61d54f0e0fb5351a17305bc87f69d10e6a03fcfa",
        "skeleton:afca9e94df21d226eef0fe1e80d8666db26760f08fdf6824f15bab67db9e1ca2",
    )
    _, repeated = _request()
    assert repeated.skeletons == request.skeletons


def test_paired_skeleton_bindings_are_explicit() -> None:
    _, request = _request()
    bindings = {item.requirement_ids for item in request.skeletons}

    assert (
        "fact:directional_assessment",
        "caveat:directional_scope",
    ) in bindings
    assert (
        "fact:evidence:synthetic-pten-target-evidence-1",
        "caveat:source_target_boundary",
    ) in bindings
    assert sum(len(item) > 1 for item in bindings) == 2


def test_grouped_skeletons_reject_conflicting_caveat_semantics() -> None:
    _, request = _request()
    plan = tuple(
        (
            dataclasses.replace(item, context="conflicting-context")
            if item.requirement_id == "caveat:directional_scope"
            else item
        )
        for item in request.plan
    )

    with pytest.raises(EvidenceError) as caught:
        derive_claim_skeletons(
            plan,
            request.canonical_gene_id,
            request.canonical_protein_id,
        )
    assert caught.value.code == "incompatible_skeleton_semantics"


def test_ollama_body_uses_exact_json_schema_object() -> None:
    _, request = _request()
    runner = _ollama_runner()
    body = runner.request_body(request, "clean")
    schema = body["format"]

    assert isinstance(schema, dict)
    assert schema == draft_plan_json_schema(request)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    entries = schema["properties"]["entries"]
    assert entries["minItems"] == entries["maxItems"] == len(request.skeletons)
    assert "prefixItems" not in entries
    assert entries["items"]["additionalProperties"] is False
    assert entries["items"]["properties"]["skeleton_id"]["enum"] == [
        item.skeleton_id for item in request.skeletons
    ]
    assert body["think"] is False
    assert body["options"]["temperature"] == 0.0
    assert "think" not in body["options"]
    prompt = body["prompt"]
    assert '"entries"' in prompt
    assert '"draft_text"' in prompt
    assert '"skeleton_id"' in prompt
    assert request.canonical_gene_id not in prompt
    assert request.canonical_protein_id not in prompt
    assert all(item.requirement_id not in prompt for item in request.plan)
    assert all(
        source_id not in prompt
        for skeleton in request.skeletons
        for source_id in skeleton.source_ids
    )


def test_trusted_caveat_copy_is_exact_and_raw_text_is_absent() -> None:
    atlas, request, draft = _valid_draft("P00533")
    raw = "A model-only sentence that must never become canonical prose."
    value = draft.to_dict()
    value["entries"][0]["draft_text"] = raw
    changed = DraftPlanEnvelopeV4.from_dict(value)

    artifact, _ = verify_draft_plan_v4(request, atlas, changed)

    assert artifact.terminal_state == "accepted"
    assert DIRECTIONAL_SCOPE_TEXT in artifact.prose
    assert SOURCE_TARGET_BOUNDARY_TEXT in artifact.prose
    assert raw not in canonical_json_bytes(artifact.to_dict()).decode("ascii")
    assert all("draft_text" not in item for item in artifact.to_dict()["verified_claims"])
    for text in (DIRECTIONAL_SCOPE_TEXT, SOURCE_TARGET_BOUNDARY_TEXT):
        lowered = text.lower()
        assert " caus" not in lowered
        assert "therapeutic target" not in lowered
        assert "druggab" not in lowered


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("omitted", "omitted_skeleton_id"),
        ("duplicate", "duplicate_skeleton_id"),
        ("unknown", "unknown_skeleton_id"),
        ("extra", "extra_skeleton_entry"),
        ("extra_property", "invalid_record"),
    ],
)
def test_task_set_failures_become_digest_safe_invalid_outcomes(
    mutation: str,
    error_code: str,
) -> None:
    _, request = _request()
    value = load_json_bytes_uncanonical(_recorded_output(request, "clean"))
    entries = value["entries"]
    assert isinstance(entries, list)
    if mutation == "omitted":
        entries.pop()
    elif mutation == "duplicate":
        entries[-1] = copy.deepcopy(entries[0])
    elif mutation == "unknown":
        entries[-1]["skeleton_id"] = "skeleton:" + "f" * 64
    elif mutation == "extra":
        entries.append(
            {
                "draft_text": "extra",
                "skeleton_id": "skeleton:" + "f" * 64,
            }
        )
    else:
        entries[0]["unexpected"] = True
    raw = canonical_json_bytes(value)

    envelope, invalid = parse_model_draft_plan(
        request,
        RecordedDraftRunnerV4().identity,
        raw,
    )

    assert envelope is None
    assert isinstance(invalid, InvalidModelOutputV4)
    assert invalid.error_code == error_code
    assert invalid.output_sha256 == sha256_bytes(raw)
    assert invalid.output_size_bytes == len(raw)
    assert "draft_text" not in canonical_json_bytes(invalid.to_dict()).decode("ascii")


@pytest.mark.parametrize(
    ("raw", "error_code"),
    [
        (b"{not-json", "invalid_json"),
        (b"[]", "invalid_json_type"),
        (b'{"entries":{}}', "invalid_json_type"),
        (b'{"entries":[],"extra":true}', "invalid_record"),
    ],
)
def test_malformed_model_shapes_are_invalid_outcomes(
    raw: bytes,
    error_code: str,
) -> None:
    _, request = _request()
    envelope, invalid = parse_model_draft_plan(
        request,
        RecordedDraftRunnerV4().identity,
        raw,
    )

    assert envelope is None
    assert invalid is not None
    assert invalid.error_code == error_code
    assert invalid.task_outcome.matched_count == 0
    assert invalid.task_outcome.omitted_count == len(request.skeletons)


class _InvalidDraftTextRunner:
    def __init__(self, draft_text: str) -> None:
        self._draft_text = draft_text
        self._identity = ModelIdentity(
            adapter="test-invalid-draft-text",
            endpoint=None,
            model="synthetic-test",
            model_manifest_sha256=None,
            runtime="pytest",
            runtime_version="recorded",
        )
        self._configuration = RunnerConfiguration(
            context_window=None,
            declared_environment=(),
            endpoint_locality="not_applicable",
            json_mode="recorded_fixture",
            options=(),
            temperature=None,
            thinking_enabled=None,
            timeout_seconds=None,
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def configuration(self) -> RunnerConfiguration:
        return self._configuration

    def validate_identity(self) -> None:
        return None

    def run(
        self,
        request: CommunicationRequestV4,
        profile: str,
        repeat_index: int,
    ) -> RunnerResultV4:
        del profile, repeat_index
        value = load_json_bytes_uncanonical(_recorded_output(request, "clean"))
        entries = value["entries"]
        assert isinstance(entries, list)
        first = entries[0]
        assert isinstance(first, dict)
        first["draft_text"] = self._draft_text
        envelope, invalid = parse_model_draft_plan(
            request,
            self.identity,
            canonical_json_bytes(value),
        )
        return RunnerResultV4(
            envelope=envelope,
            input_tokens=None,
            invalid_model_output=invalid,
            latency_ms=None,
            output_tokens=None,
            peak_memory_mb=None,
            peak_metal_memory_mb=None,
        )


@pytest.mark.parametrize("draft_text", ["", "x" * 501])
def test_draft_text_bounds_are_digest_safe_and_replayable(
    tmp_path: Path,
    draft_text: str,
) -> None:
    atlas_path = _atlas_path(tmp_path)
    output = tmp_path / f"invalid-length-{len(draft_text)}"
    benchmark = run_benchmark_v4(
        atlas_path,
        V4_BENCHMARK_FIXTURE,
        output,
        _InvalidDraftTextRunner(draft_text),
    )

    assert benchmark["metrics"]["invalid_model_output_rate"]["value"] == 1.0
    assert all(
        item["invalid_model_output"]["error_code"] == "invalid_draft_text_length"
        for item in benchmark["runs"]
    )
    assert (
        replay_benchmark_v4(
            atlas_path,
            V4_BENCHMARK_FIXTURE,
            output,
        )
        == benchmark
    )


@pytest.mark.parametrize(
    "field",
    [
        "request_id",
        "source_atlas_sha256",
        "source_card_sha256",
        "prompt_identity",
    ],
)
def test_draft_identity_drift_aborts_verification(field: str) -> None:
    atlas, request, draft = _valid_draft()
    value = draft.to_dict()
    if field == "prompt_identity":
        value[field] = {
            "prompt_id": "different",
            "sha256": "0" * 64,
        }
    elif field.endswith("sha256"):
        value[field] = "0" * 64
    else:
        value[field] = "different"
    changed = DraftPlanEnvelopeV4.from_dict(value)

    with pytest.raises(EvidenceError) as caught:
        verify_draft_plan_v4(request, atlas, changed)
    assert caught.value.code == "draft_source_identity_mismatch"


@pytest.mark.parametrize(
    ("drift_kind", "validation_index", "error_code", "generation_count"),
    [
        ("digest", 1, "model_manifest_digest_mismatch", 0),
        ("digest", 2, "model_manifest_digest_mismatch", 1),
        ("runtime", 1, "runtime_identity_mismatch", 0),
        ("runtime", 2, "runtime_identity_mismatch", 1),
    ],
)
def test_ollama_identity_is_checked_before_and_after_every_generation(
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
    validation_index: int,
    error_code: str,
    generation_count: int,
) -> None:
    _, request = _request()
    generations: list[dict[str, Any]] = []
    response = _ollama_response(_recorded_output(request, "clean").decode("ascii"))
    digest_drift = {validation_index: "b" * 64} if drift_kind == "digest" else None
    runtime_drift = {validation_index: "different-runtime"} if drift_kind == "runtime" else None
    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(
            response,
            digest_by_validation=digest_drift,
            version_by_validation=runtime_drift,
            generated_requests=generations,
        ),
    )

    with pytest.raises(EvidenceError) as caught:
        _ollama_runner().run(request, "clean", 0)
    assert caught.value.code == error_code
    assert len(generations) == generation_count


def test_ollama_accepts_official_capabilities_and_rejects_remote_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _ollama_response("{}")
    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(response, tag_extra={"capabilities": ["completion"]}),
    )
    _ollama_runner().validate_identity()

    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(
            response,
            tag_extra={
                "remote_host": "https://remote.invalid",
                "remote_model": "remote-model",
            },
        ),
    )
    with pytest.raises(EvidenceError) as caught:
        _ollama_runner().validate_identity()
    assert caught.value.code == "local_model_remote_execution_forbidden"


def test_outer_protocol_and_transport_drift_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request = _request()
    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(_ollama_response("{}", unexpected=True)),
    )
    with pytest.raises(EvidenceError) as protocol:
        _ollama_runner().run(request, "clean", 0)
    assert protocol.value.code == "local_model_protocol_failed"

    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(b"", generation_error=urllib.error.URLError("offline")),
    )
    with pytest.raises(EvidenceError) as transport:
        _ollama_runner().run(request, "clean", 0)
    assert transport.value.code == "local_model_request_failed"

    output = tmp_path / "aborted-publication"
    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(b"", generation_error=urllib.error.URLError("offline")),
    )
    with pytest.raises(EvidenceError) as benchmark_transport:
        run_benchmark_v4(
            _atlas_path(tmp_path),
            V4_BENCHMARK_FIXTURE,
            output,
            _ollama_runner(),
        )
    assert benchmark_transport.value.code == "local_model_request_failed"
    assert not output.exists()


def test_ollama_invalid_unicode_model_text_is_digest_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request = _request()
    response_text = '{"entries":[{"draft_text":"\ud800","skeleton_id":"irrelevant"}]}'
    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(_ollama_response(response_text)),
    )

    result = _ollama_runner().run(request, "clean", 0)

    assert result.envelope is None
    assert result.invalid_model_output is not None
    assert result.invalid_model_output.error_code == "invalid_draft_text_encoding"
    invalid_bytes = response_text.encode("utf-8", errors="surrogatepass")
    assert result.invalid_model_output.output_sha256 == sha256_bytes(invalid_bytes)
    assert result.invalid_model_output.output_size_bytes == len(invalid_bytes)
    assert "irrelevant" not in canonical_json_bytes(result.invalid_model_output.to_dict()).decode(
        "ascii"
    )


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_ollama_rejects_nonpositive_or_nonfinite_timeout(timeout_seconds: float) -> None:
    with pytest.raises(EvidenceError) as caught:
        OllamaRunnerV4(
            "http://127.0.0.1:11434",
            "installed-test-model",
            OLLAMA_DIGEST,
            OLLAMA_VERSION,
            False,
            timeout_seconds=timeout_seconds,
        )
    assert caught.value.code == "invalid_timeout"


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("missing_done", "local_model_protocol_failed"),
        ("false_done", "local_model_incomplete_response"),
        ("missing_model", "local_model_protocol_failed"),
        ("model_drift", "local_model_response_model_mismatch"),
        ("thinking_drift", "local_model_thinking_mode_mismatch"),
        ("remote_host", "local_model_remote_execution_forbidden"),
        ("remote_model", "local_model_remote_execution_forbidden"),
    ],
)
def test_ollama_completion_and_model_identity_drift_abort_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_code: str,
) -> None:
    _, request = _request()
    response_value = load_json_bytes_uncanonical(
        _ollama_response(_recorded_output(request, "clean").decode("ascii"))
    )
    if mutation == "missing_done":
        del response_value["done"]
    elif mutation == "false_done":
        response_value["done"] = False
    elif mutation == "missing_model":
        del response_value["model"]
    elif mutation == "model_drift":
        response_value["model"] = "different-model"
    elif mutation == "thinking_drift":
        response_value["thinking"] = "unexpected thinking content"
    else:
        response_value[mutation] = "unexpected-remote-value"
    monkeypatch.setattr(
        v4_module,
        "_open_local_model_request",
        _ollama_responder(canonical_json_bytes(response_value)),
    )
    output = tmp_path / mutation

    with pytest.raises(EvidenceError) as caught:
        run_benchmark_v4(
            _atlas_path(tmp_path),
            V4_BENCHMARK_FIXTURE,
            output,
            _ollama_runner(),
        )
    assert caught.value.code == error_code
    assert not output.exists()


def test_fixture_rejects_duplicate_expected_run_entries(tmp_path: Path) -> None:
    fixture = load_json(V4_BENCHMARK_FIXTURE)
    assert isinstance(fixture, dict)
    cases = fixture["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    expected_runs = first["expected_runs"]
    assert isinstance(expected_runs, list)
    expected_runs.append(copy.deepcopy(expected_runs[0]))
    path = tmp_path / "duplicate-expected-run.json"
    write_json(path, fixture)

    with pytest.raises(EvidenceError) as caught:
        v4_module._load_benchmark_cases_v4(path)
    assert caught.value.code == "expected_run_set_mismatch"


def test_recorded_compile_and_replay_are_byte_deterministic(tmp_path: Path) -> None:
    atlas_path = _atlas_path(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = compile_recorded_fixture_v4(
        atlas_path,
        "P60484",
        "deterministic-v4",
        "clean",
        first,
    )
    second_result = compile_recorded_fixture_v4(
        atlas_path,
        "P60484",
        "deterministic-v4",
        "clean",
        second,
    )

    assert first_result == second_result
    assert _tree_bytes(first) == _tree_bytes(second)
    assert replay_communication_v4(atlas_path, first) == first_result


@pytest.mark.parametrize(
    ("terminal_state", "bucket"),
    [
        ("accepted", "accepted"),
        ("accepted_with_exclusions", "partially_excluded"),
        ("refused", "refused"),
    ],
)
def test_terminal_outcome_bucket_uses_exact_artifact_states(
    terminal_state: str,
    bucket: str,
) -> None:
    assert _terminal_outcome_bucket_v4(terminal_state) == bucket


def test_recorded_benchmark_replay_and_exact_metrics(tmp_path: Path) -> None:
    atlas_path = _atlas_path(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    benchmark = run_benchmark_v4(
        atlas_path,
        V4_BENCHMARK_FIXTURE,
        first,
        RecordedDraftRunnerV4(),
    )
    repeated = run_benchmark_v4(
        atlas_path,
        V4_BENCHMARK_FIXTURE,
        second,
        RecordedDraftRunnerV4(),
    )

    assert repeated == benchmark
    assert _tree_bytes(first) == _tree_bytes(second)
    assert len(_tree_bytes(first)) == 93
    assert replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, first) == benchmark
    metrics = benchmark["metrics"]
    assert metrics["task_skeleton_coverage"] == {
        "availability": "available",
        "denominator": 270,
        "numerator": 266,
        "value": 266 / 270,
    }
    assert metrics["unknown_task_rate"] == {
        "availability": "available",
        "denominator": 270,
        "numerator": 2,
        "value": 2 / 270,
    }
    assert metrics["duplicate_task_rate"] == {
        "availability": "available",
        "denominator": 270,
        "numerator": 2,
        "value": 2 / 270,
    }
    assert metrics["omitted_task_rate"] == {
        "availability": "available",
        "denominator": 270,
        "numerator": 4,
        "value": 4 / 270,
    }
    assert metrics["invalid_model_output_rate"] == {
        "availability": "available",
        "denominator": 18,
        "numerator": 4,
        "value": 4 / 18,
    }
    assert metrics["accepted_terminal_outcome_rate"] == {
        "availability": "available",
        "denominator": 18,
        "numerator": 10,
        "value": 10 / 18,
    }
    assert metrics["partially_excluded_terminal_outcome_rate"] == {
        "availability": "available",
        "denominator": 18,
        "numerator": 0,
        "value": 0.0,
    }
    assert metrics["refused_terminal_outcome_rate"] == {
        "availability": "available",
        "denominator": 18,
        "numerator": 8,
        "value": 8 / 18,
    }
    assert metrics["evidence_citation_validity"]["value"] == 1.0
    assert metrics["replay_semantic_variance"]["value"] == 0.0
    assert metrics["verifier_counts"] == {
        "accepted": 10,
        "accepted_or_partially_excluded": 10,
        "partially_excluded": 0,
        "refused": 8,
        "total": 18,
    }
    for name in (
        "input_tokens",
        "latency_ms",
        "output_tokens",
        "peak_memory_mb",
        "peak_metal_memory_mb",
    ):
        assert metrics[name]["availability"] == "unavailable"
        assert 0 not in metrics[name].values()


@pytest.mark.parametrize(
    "relative_path",
    [
        "runs/supported-agreement/repeat-0000/request.json",
        "runs/supported-agreement/repeat-0000/draft.json",
        "runs/supported-agreement/repeat-0000/artifact.json",
        "runs/supported-agreement/repeat-0000/receipt.json",
        "runs/supported-agreement/repeat-0000/measurement.json",
        "benchmark.json",
        "index.json",
    ],
)
def test_replay_rejects_complete_tree_tampering(
    tmp_path: Path,
    relative_path: str,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    value = load_json(bundle / relative_path)
    assert isinstance(value, dict)
    value["tampered"] = True
    write_json(bundle / relative_path, value)
    _reclose_file(bundle, relative_path)

    with pytest.raises(EvidenceError):
        replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, bundle)


def test_replay_rejects_missing_extra_noncanonical_path_and_symlink(
    tmp_path: Path,
) -> None:
    atlas_path, missing = _recorded_benchmark(tmp_path, "missing")
    (missing / "runs/supported-agreement/repeat-0000/request.json").unlink()
    with pytest.raises(EvidenceError) as missing_error:
        replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, missing)
    assert missing_error.value.code == "bundle_file_set_mismatch"

    _, extra = _recorded_benchmark(tmp_path, "extra")
    write_json(extra / "unexpected.json", {"unexpected": True})
    with pytest.raises(EvidenceError) as extra_error:
        replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, extra)
    assert extra_error.value.code == "bundle_file_set_mismatch"

    _, noncanonical = _recorded_benchmark(tmp_path, "noncanonical")
    path = noncanonical / "runs/supported-agreement/repeat-0000/artifact.json"
    path.write_bytes(path.read_bytes().replace(b"{", b"{ ", 1))
    _reclose_file(noncanonical, path.relative_to(noncanonical).as_posix())
    with pytest.raises(EvidenceError) as noncanonical_error:
        replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, noncanonical)
    assert noncanonical_error.value.code == "noncanonical_json"

    _, traversal = _recorded_benchmark(tmp_path, "traversal")
    receipt = load_json(traversal / "receipt.json")
    receipt["closed_files"][0]["path"] = "../escape.json"
    write_json(traversal / "receipt.json", receipt)
    with pytest.raises(EvidenceError) as traversal_error:
        replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, traversal)
    assert traversal_error.value.code == "unsafe_path"

    _, symlink = _recorded_benchmark(tmp_path, "symlink")
    request_path = symlink / "runs/supported-agreement/repeat-0000/request.json"
    target = tmp_path / "request-copy.json"
    target.write_bytes(request_path.read_bytes())
    request_path.unlink()
    request_path.symlink_to(target)
    with pytest.raises(EvidenceError) as symlink_error:
        replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, symlink)
    assert symlink_error.value.code == "symlink_forbidden"


def test_replay_recomputes_aggregate_metrics(tmp_path: Path) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    benchmark = load_json(bundle / "benchmark.json")
    benchmark["metrics"]["verifier_counts"]["refused"] = 7
    write_json(bundle / "benchmark.json", benchmark)
    _reclose_file(bundle, "benchmark.json")

    with pytest.raises(EvidenceError) as caught:
        replay_benchmark_v4(atlas_path, V4_BENCHMARK_FIXTURE, bundle)
    assert caught.value.code == "benchmark_replay_mismatch"


def test_v3_observed_bundle_remains_replayable_byte_for_byte(tmp_path: Path) -> None:
    atlas_path = _atlas_path(tmp_path)
    before = _tree_bytes(V3_OBSERVED_BUNDLE)

    benchmark = replay_benchmark_v3(
        atlas_path,
        V3_BENCHMARK_FIXTURE,
        V3_OBSERVED_BUNDLE,
    )

    assert benchmark["schema_version"] == "voxelscope/evidence-communication-benchmark/v3"
    assert benchmark["metrics"]["verifier_counts"] == {
        "accepted_or_partially_excluded": 0,
        "refused": 18,
        "total": 18,
    }
    assert _tree_bytes(V3_OBSERVED_BUNDLE) == before


def test_v4_observed_bundle_remains_replayable_byte_for_byte(tmp_path: Path) -> None:
    atlas_path = _atlas_path(tmp_path)
    before = _tree_bytes(V4_OBSERVED_BUNDLE)
    study_binding = load_json(V4_OBSERVED_BUNDLE / "study-binding.json")

    benchmark = replay_benchmark_v4(
        atlas_path,
        V4_BENCHMARK_FIXTURE,
        V4_OBSERVED_BUNDLE,
        study_declaration_sha256=study_binding["declaration_sha256"],
    )

    assert len(before) == 94
    assert benchmark["schema_version"] == "voxelscope/evidence-communication-benchmark/v4"
    assert benchmark["metrics"]["verifier_counts"] == {
        "accepted": 6,
        "accepted_or_partially_excluded": 6,
        "partially_excluded": 0,
        "refused": 12,
        "total": 18,
    }
    assert _tree_bytes(V4_OBSERVED_BUNDLE) == before


def test_v4_post_hoc_analysis_recomputes_from_safe_artifacts() -> None:
    analysis = json.loads((V4_OBSERVED_STUDY / "post-hoc-analysis.json").read_text())
    result_summary = json.loads((V4_OBSERVED_STUDY / "result-summary.json").read_text())
    benchmark = load_json(V4_OBSERVED_BUNDLE / "benchmark.json")

    assert "analysis_performed_at" not in analysis
    assert analysis["observed_result_recorded_at"] == result_summary["result_recorded_at"]
    assert analysis["analysis_status"] == "post_observation_not_preregistered"
    assert analysis["analysis_boundary"]["model_owned_draft_text_accessed"] is False
    assert analysis["analysis_boundary"]["model_owned_draft_text_published"] is False
    assert "model_draft" not in analysis["analysis_boundary"]["used_artifact_types"]

    reason_counts: Counter[str] = Counter()
    claim_type_counts: Counter[str] = Counter()
    requirement_kind_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    unique_excluded_skeleton_ids: set[str] = set()
    excluded_sets_by_case: dict[str, list[set[str]]] = {}
    task_counts: Counter[str] = Counter()
    run_count_with_exclusions = 0
    invalid_model_outputs = 0

    for run in benchmark["runs"]:
        run_root = V4_OBSERVED_BUNDLE / run["run_path"]
        artifact = load_json(run_root / "artifact.json")
        request = load_json(run_root / "request.json")
        skeletons = {item["skeleton_id"]: item for item in request["skeletons"]}
        requirement_kinds = {item["requirement_id"]: item["kind"] for item in request["plan"]}
        excluded_skeleton_ids: set[str] = set()

        if artifact["exclusions"]:
            run_count_with_exclusions += 1
        for exclusion in artifact["exclusions"]:
            skeleton_id = exclusion["skeleton_id"]
            skeleton = skeletons[skeleton_id]
            excluded_skeleton_ids.add(skeleton_id)
            unique_excluded_skeleton_ids.add(skeleton_id)
            reason_counts[exclusion["reason"]] += 1
            claim_type_counts[skeleton["claim_type"]] += 1
            for requirement_id in skeleton["requirement_ids"]:
                requirement_kind_counts[requirement_kinds[requirement_id]] += 1

        excluded_sets_by_case.setdefault(run["case_id"], []).append(excluded_skeleton_ids)
        terminal_bucket = {
            "accepted": "accepted",
            "accepted_with_exclusions": "partially_excluded",
            "refused": "refused",
        }[artifact["terminal_state"]]
        terminal_counts[terminal_bucket] += 1
        invalid_model_outputs += int(artifact["invalid_model_output"] is not None)
        for field in (
            "expected_count",
            "matched_count",
            "unknown_count",
            "duplicate_count",
            "omitted_count",
        ):
            task_counts[field] += artifact["task_outcome"][field]

    expected_requirement_counts = {
        "caveat": requirement_kind_counts["caveat"],
        "fact": requirement_kind_counts["fact"],
        "total": requirement_kind_counts.total(),
    }
    expected_terminal_counts = {
        "accepted": terminal_counts["accepted"],
        "partially_excluded": terminal_counts["partially_excluded"],
        "refused": terminal_counts["refused"],
        "total": terminal_counts.total(),
    }
    expected_pair_differences = [
        {
            "case_id": case_id,
            "symmetric_difference_count": len(repeats[0] ^ repeats[1]),
        }
        for case_id, repeats in sorted(excluded_sets_by_case.items())
        if repeats[0] != repeats[1]
    ]

    assert analysis["exclusions"] == {
        "excluded_claim_type_counts": dict(sorted(claim_type_counts.items())),
        "excluded_requirement_binding_counts": expected_requirement_counts,
        "reason_counts": dict(sorted(reason_counts.items())),
        "run_count_with_exclusions": run_count_with_exclusions,
        "total_excluded_skeleton_entries": reason_counts.total(),
        "unique_excluded_skeleton_ids": len(unique_excluded_skeleton_ids),
    }
    assert analysis["repeat_pair_exclusion_set_differences"] == expected_pair_differences
    assert analysis["task_set_observations"] == {
        "duplicate_skeleton_entries": task_counts["duplicate_count"],
        "expected_skeleton_entries": task_counts["expected_count"],
        "invalid_model_outputs": invalid_model_outputs,
        "matched_skeleton_entries": task_counts["matched_count"],
        "omitted_skeleton_entries": task_counts["omitted_count"],
        "unknown_skeleton_entries": task_counts["unknown_count"],
    }
    assert analysis["terminal_counts"] == expected_terminal_counts
    for field in ("benchmark_index", "benchmark_receipt", "benchmark_report"):
        evidence = analysis["closed_evidence"][field]
        evidence_sha256 = sha256_bytes((V4_OBSERVED_STUDY / evidence["path"]).read_bytes())
        assert evidence_sha256 == evidence["sha256"]
