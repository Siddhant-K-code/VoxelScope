# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from voxelscope import evidence_communication as communication_module
from voxelscope.canonical import (
    EvidenceError,
    canonical_json_bytes,
    load_json,
    load_json_bytes,
    sha256_bytes,
    write_json,
)
from voxelscope.evidence_benchmark_records import RunnerConfiguration
from voxelscope.evidence_communication import (
    BOUNDARY_TEXT,
    OllamaRunner,
    RecordedDraftRunner,
    RunnerResult,
    compile_recorded_fixture,
    derive_communication_request,
    recorded_model_draft,
    replay_benchmark,
    replay_communication,
    run_benchmark,
    semantic_artifact_sha256,
    verify_model_draft,
)
from voxelscope.evidence_communication_cli import main
from voxelscope.evidence_communication_records import (
    CommunicationReceipt,
    CommunicationRequest,
    ModelDraftEnvelope,
    ModelIdentity,
    VerifiedCommunicationArtifact,
)
from voxelscope.gbm_atlas import assemble_gbm_evidence_atlas

ROOT = Path(__file__).resolve().parents[1]
ATLAS_FIXTURE = ROOT / "research/gbm-evidence-atlas-v1/source-manifest.json"
BENCHMARK_FIXTURE = (
    ROOT / "research/gbm-evidence-communication-benchmark-v3" / "benchmark-fixtures.json"
)
OLLAMA_MODEL_DIGEST = "a" * 64
OLLAMA_RUNTIME_VERSION = "0.35.1"


def _atlas() -> tuple[dict[str, Any], str]:
    atlas, _ = assemble_gbm_evidence_atlas(ATLAS_FIXTURE)
    return atlas, sha256_bytes(canonical_json_bytes(atlas))


def _atlas_path(tmp_path: Path) -> Path:
    atlas, _ = _atlas()
    path = tmp_path / "atlas.json"
    write_json(path, atlas)
    return path


def _draft(
    protein_id: str = "P60484",
    profile: str = "clean",
) -> tuple[dict[str, Any], Any, ModelDraftEnvelope]:
    atlas, atlas_sha256 = _atlas()
    request = derive_communication_request(atlas, atlas_sha256, protein_id, f"test:{protein_id}")
    return atlas, request, recorded_model_draft(request, atlas, profile, 0)


def _replace_claim(
    draft: ModelDraftEnvelope,
    target_claim_id: str,
    **updates: Any,
) -> ModelDraftEnvelope:
    value = draft.to_dict()
    claims = value["claims"]
    assert isinstance(claims, list)
    for claim in claims:
        assert isinstance(claim, dict)
        if claim["claim_id"] == target_claim_id:
            claim.update(updates)
            break
    else:
        raise AssertionError(f"claim not found: {target_claim_id}")
    return ModelDraftEnvelope.from_dict(value)


def _with_model_identity(
    draft: ModelDraftEnvelope,
    identity: ModelIdentity,
) -> ModelDraftEnvelope:
    return ModelDraftEnvelope.from_dict({**draft.to_dict(), "model_identity": identity.to_dict()})


class _AlternateClaimIdRunner:
    def __init__(self) -> None:
        self._identity = ModelIdentity(
            adapter="test-alternate-claim-id",
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
        request: CommunicationRequest,
        atlas: dict[str, Any],
        profile: str,
        repeat_index: int,
    ) -> RunnerResult:
        draft = recorded_model_draft(request, atlas, profile, repeat_index)
        if repeat_index == 1:
            draft = _replace_claim(draft, "claim:entity", claim_id="claim:entity:alternate")
        return RunnerResult(
            envelope=_with_model_identity(draft, self.identity),
            input_tokens=None,
            invalid_model_output=None,
            latency_ms=None,
            output_tokens=None,
            peak_memory_mb=None,
            peak_metal_memory_mb=None,
        )


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
    generate_response: bytes,
    *,
    observed_digest: str = OLLAMA_MODEL_DIGEST,
    observed_version: str = OLLAMA_RUNTIME_VERSION,
    generation_error: Exception | None = None,
    generated_requests: list[dict[str, Any]] | None = None,
) -> Any:
    def respond(
        request: urllib.request.Request,
        timeout_seconds: float | None = None,
    ) -> _FakeHttpResponse:
        del timeout_seconds
        if request.full_url.endswith("/api/version"):
            return _FakeHttpResponse(canonical_json_bytes({"version": observed_version}))
        if request.full_url.endswith("/api/tags"):
            return _FakeHttpResponse(
                canonical_json_bytes(
                    {
                        "models": [
                            {
                                "digest": observed_digest,
                                "model": "installed-test-model",
                                "name": "installed-test-model",
                            }
                        ]
                    }
                )
            )
        if generation_error is not None:
            raise generation_error
        if generated_requests is not None:
            assert request.data is not None
            request_value = load_json_bytes(request.data)
            assert isinstance(request_value, dict)
            generated_requests.append(request_value)
        return _FakeHttpResponse(generate_response)

    return respond


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _reclose_benchmark_file(bundle: Path, relative_path: str) -> None:
    receipt = load_json(bundle / "receipt.json")
    assert isinstance(receipt, dict)
    digest = sha256_bytes((bundle / relative_path).read_bytes())
    closed_files = receipt["closed_files"]
    assert isinstance(closed_files, list)
    for item in closed_files:
        assert isinstance(item, dict)
        if item["path"] == relative_path:
            item["sha256"] = digest
            break
    else:
        raise AssertionError(f"closed file not found: {relative_path}")
    if relative_path == "benchmark.json":
        receipt["benchmark_sha256"] = digest
    elif relative_path == "index.json":
        receipt["index_sha256"] = digest
    write_json(bundle / "receipt.json", receipt)


def _recorded_benchmark(tmp_path: Path, name: str = "benchmark") -> tuple[Path, Path]:
    atlas_path = _atlas_path(tmp_path)
    bundle = tmp_path / name
    run_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle, RecordedDraftRunner())
    return atlas_path, bundle


def test_recorded_compilation_and_replay_are_byte_deterministic(
    tmp_path: Path,
) -> None:
    atlas_path = _atlas_path(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = compile_recorded_fixture(atlas_path, "P60484", "deterministic", "clean", first)
    second_result = compile_recorded_fixture(atlas_path, "P60484", "deterministic", "clean", second)

    assert first_result == second_result
    for name in ("request.json", "draft.json", "artifact.json", "receipt.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert replay_communication(atlas_path, first) == first_result


def test_malformed_envelope_and_duplicate_source_ids_are_rejected() -> None:
    _, _, draft = _draft()
    malformed = draft.to_dict()
    malformed["unexpected"] = True

    with pytest.raises(EvidenceError) as malformed_error:
        ModelDraftEnvelope.from_dict(malformed)
    assert malformed_error.value.code == "invalid_record"

    duplicate = draft.to_dict()
    claims = duplicate["claims"]
    assert isinstance(claims, list)
    first_claim = claims[0]
    assert isinstance(first_claim, dict)
    source_ids = first_claim["source_ids"]
    assert isinstance(source_ids, list)
    first_claim["source_ids"] = [source_ids[0], source_ids[0]]

    with pytest.raises(EvidenceError) as duplicate_error:
        ModelDraftEnvelope.from_dict(duplicate)
    assert duplicate_error.value.code == "duplicate_record_value"


def test_unknown_evidence_id_is_visible_and_partially_excluded() -> None:
    atlas, request, draft = _draft(profile="unknown_source")

    artifact, receipt = verify_model_draft(request, atlas, draft)

    assert artifact.terminal_state == "accepted_with_exclusions"
    assert artifact.exclusions[0].reason == "unknown_source_id"
    assert receipt.exclusions == artifact.exclusions
    assert "unknown evidence item" not in artifact.prose.lower()


def test_plan_enumerates_exact_facts_and_caveats() -> None:
    _, request, _ = _draft()
    requirements = {item.requirement_id: item for item in request.plan}

    protein = requirements["fact:evidence:synthetic-pten-protein-1"]
    assert protein.modality == "protein_abundance"
    assert protein.direction == "down"
    assert protein.context == "Synthetic GBM cohort B"
    assert protein.value == -1.7
    assert protein.unit == "synthetic-relative-abundance"
    assert protein.comparison_key == ("Synthetic GBM cohort B|protein_abundance")
    assert requirements["fact:directional_assessment"].state == "agreement"
    assert requirements["caveat:not_measured:phosphorylation"].state == "not_measured"
    assert requirements["caveat:stale:synthetic-evidence-fixture-v1"].state == "stale"
    assert requirements["caveat:unsupported:synthetic-unsupported-spatial-1"].state == "unsupported"
    assert requirements["caveat:non_clinical_boundary"].description.endswith(BOUNDARY_TEXT)


def test_request_cannot_remove_a_required_caveat() -> None:
    atlas, request, draft = _draft()
    value = request.to_dict()
    plan = value["plan"]
    assert isinstance(plan, list)
    value["plan"] = [
        item
        for item in plan
        if isinstance(item, dict) and item["requirement_id"] != "caveat:non_clinical_boundary"
    ]
    changed = CommunicationRequest.from_dict(value)
    changed_draft = ModelDraftEnvelope.from_dict(
        {
            **draft.to_dict(),
            "prompt_identity": changed.prompt_identity.to_dict(),
        }
    )

    with pytest.raises(EvidenceError) as caught:
        verify_model_draft(changed, atlas, changed_draft)
    assert caught.value.code == "communication_plan_mismatch"


def test_card_digest_change_is_rejected() -> None:
    atlas, request, draft = _draft()
    changed = copy.deepcopy(atlas)
    cards = changed["cards"]
    assert isinstance(cards, list)
    pten = next(card for card in cards if card["canonical_protein_id"] == "P60484")
    pten["missing_modalities"] = []

    with pytest.raises(EvidenceError) as caught:
        verify_model_draft(request, changed, draft)
    assert caught.value.code == "source_card_digest_mismatch"


def test_cross_context_conflation_excludes_claim_and_refuses_output() -> None:
    atlas, request, draft = _draft()
    changed = _replace_claim(
        draft,
        "claim:evidence:synthetic-pten-mutation-1",
        context="Synthetic GBM cohort A",
        comparison_key="Synthetic GBM cohort A|mutation",
    )

    artifact, _ = verify_model_draft(request, atlas, changed)

    assert artifact.terminal_state == "refused"
    assert artifact.prose == ""
    assert any(item.reason == "evidence_field_mismatch" for item in artifact.exclusions)
    assert any(reason.startswith("missing_required_facts:") for reason in artifact.refusal_reasons)


def test_missing_required_caveat_refuses_communication() -> None:
    atlas, request, draft = _draft()
    value = draft.to_dict()
    claims = value["claims"]
    assert isinstance(claims, list)
    value["claims"] = [
        claim
        for claim in claims
        if isinstance(claim, dict) and claim["claim_id"] != "claim:boundary"
    ]
    changed = ModelDraftEnvelope.from_dict(value)

    artifact, _ = verify_model_draft(request, atlas, changed)

    assert artifact.terminal_state == "refused"
    assert artifact.sentences == ()
    assert any("caveat:non_clinical_boundary" in reason for reason in artifact.refusal_reasons)


@pytest.mark.parametrize(
    "text,reason",
    [
        (
            "This definitively diagnoses disease.",
            "prohibited_diagnostic_or_prognostic_language",
        ),
        ("This recommends treatment.", "prohibited_treatment_language"),
        (
            "This proves a causal mechanism.",
            "prohibited_causal_or_certainty_language",
        ),
        (
            "This is a therapeutic target and is druggable.",
            "prohibited_target_or_druggability_language",
        ),
        ("This is the top protein.", "prohibited_protein_ranking_language"),
    ],
)
def test_prohibited_certainty_and_clinical_claims_refuse(
    text: str,
    reason: str,
) -> None:
    atlas, request, draft = _draft()
    changed = _replace_claim(
        draft,
        "claim:evidence:synthetic-pten-mutation-1",
        draft_text=text,
    )

    artifact, _ = verify_model_draft(request, atlas, changed)

    assert artifact.terminal_state == "refused"
    assert reason in artifact.refusal_reasons
    assert text not in artifact.prose


def test_raw_model_prose_is_never_a_verified_claim() -> None:
    atlas, request, draft = _draft()
    raw_text = "This protein drives disease and should guide care."
    changed = _replace_claim(
        draft,
        "claim:evidence:synthetic-pten-mutation-1",
        draft_text=raw_text,
    )

    artifact, receipt = verify_model_draft(request, atlas, changed)

    assert artifact.terminal_state == "refused"
    assert raw_text in {item.draft_text for item in artifact.exclusions}
    assert raw_text not in artifact.prose
    assert all("draft_text" not in item for item in artifact.to_dict()["verified_claims"])
    assert all(not hasattr(item, "draft_text") for item in artifact.verified_claims)
    assert any(
        reason in artifact.refusal_reasons
        for reason in (
            "prohibited_causal_or_certainty_language",
            "prohibited_treatment_language",
        )
    )
    assert receipt.exclusions == artifact.exclusions


def test_benign_raw_model_prose_is_removed_before_acceptance() -> None:
    atlas, request, draft = _draft()
    raw_text = "A model-specific phrasing that is not trusted output."
    changed = _replace_claim(
        draft,
        "claim:evidence:synthetic-pten-mutation-1",
        draft_text=raw_text,
    )

    artifact, _ = verify_model_draft(request, atlas, changed)

    assert artifact.terminal_state == "accepted"
    assert raw_text not in canonical_json_bytes(artifact.to_dict()).decode("ascii")


def test_safety_copy_and_sentence_mapping_are_closed() -> None:
    atlas, request, draft = _draft()

    artifact, receipt = verify_model_draft(request, atlas, draft)

    assert artifact.terminal_state == "accepted"
    assert artifact.prose.endswith(BOUNDARY_TEXT)
    assert all(sentence.verified_claim_ids for sentence in artifact.sentences)
    assert all(sentence.source_ids for sentence in artifact.sentences)
    assert receipt.terminal_state == "closed"
    assert receipt.communication_terminal_state == "accepted"
    assert receipt.artifact_sha256 == sha256_bytes(canonical_json_bytes(artifact.to_dict()))


def test_not_measured_is_not_negative_evidence() -> None:
    atlas, request, draft = _draft(profile="not_measured_as_negative")

    artifact, _ = verify_model_draft(request, atlas, draft)

    assert artifact.terminal_state == "refused"
    assert "not_measured_conflated_with_negative_evidence" in artifact.refusal_reasons
    assert artifact.prose == ""


def test_source_target_evidence_does_not_become_voxelscope_target_claim() -> None:
    atlas, request, draft = _draft("P00533")

    artifact, _ = verify_model_draft(request, atlas, draft)

    assert artifact.terminal_state == "accepted"
    assert "source-reported evidence is not a VoxelScope target claim" in artifact.prose
    target_claims = [
        item for item in artifact.verified_claims if item.claim_type == "source_target_evidence"
    ]
    assert len(target_claims) == 1


def test_benchmark_metrics_are_exact_and_unavailable_is_not_zero(
    tmp_path: Path,
) -> None:
    atlas_path = _atlas_path(tmp_path)
    output = tmp_path / "benchmark"

    benchmark = run_benchmark(atlas_path, BENCHMARK_FIXTURE, output, RecordedDraftRunner())
    metrics = benchmark["metrics"]

    assert metrics["unsupported_claim_rate"] == {
        "availability": "available",
        "denominator": 278,
        "numerator": 8,
        "value": 8 / 278,
    }
    assert metrics["verified_draft_fact_coverage"] == {
        "availability": "available",
        "denominator": 158,
        "numerator": 158,
        "value": 1.0,
    }
    assert metrics["verified_draft_caveat_retention"] == {
        "availability": "available",
        "denominator": 148,
        "numerator": 148,
        "value": 1.0,
    }
    assert metrics["emitted_fact_coverage"] == {
        "availability": "available",
        "denominator": 158,
        "numerator": 122,
        "value": 122 / 158,
    }
    assert metrics["emitted_caveat_coverage"] == {
        "availability": "available",
        "denominator": 148,
        "numerator": 116,
        "value": 116 / 148,
    }
    assert metrics["evidence_citation_validity"] == {
        "availability": "available",
        "denominator": 338,
        "numerator": 336,
        "value": 336 / 338,
    }
    assert metrics["replay_semantic_variance"] == {
        "availability": "available",
        "denominator": 9,
        "numerator": 1,
        "value": 1 / 9,
    }
    assert metrics["invalid_model_output_rate"] == {
        "availability": "available",
        "denominator": 18,
        "numerator": 0,
        "value": 0.0,
    }
    assert metrics["verifier_counts"] == {
        "accepted_or_partially_excluded": 14,
        "refused": 4,
        "total": 18,
    }
    refused_runs = [item for item in benchmark["runs"] if item["terminal_state"] == "refused"]
    assert refused_runs
    assert all(
        item["emitted_fact_coverage"]["numerator"] == 0
        and item["emitted_caveat_coverage"]["numerator"] == 0
        and item["verified_draft_fact_coverage"]["value"] == 1.0
        and item["verified_draft_caveat_coverage"]["value"] == 1.0
        for item in refused_runs
    )
    for name in (
        "input_tokens",
        "latency_ms",
        "output_tokens",
        "peak_memory_mb",
        "peak_metal_memory_mb",
    ):
        assert metrics[name]["availability"] == "unavailable"
        assert 0 not in metrics[name].values()
    receipt = load_json(output / "receipt.json")
    assert receipt["terminal_state"] == "closed"
    assert receipt["benchmark_sha256"] == sha256_bytes((output / "benchmark.json").read_bytes())


def test_recorded_benchmark_tree_is_byte_deterministic_and_offline_replayable(
    tmp_path: Path,
) -> None:
    atlas_path = _atlas_path(tmp_path)
    first = tmp_path / "first-benchmark"
    second = tmp_path / "second-benchmark"

    first_result = run_benchmark(
        atlas_path,
        BENCHMARK_FIXTURE,
        first,
        RecordedDraftRunner(),
    )
    second_result = run_benchmark(
        atlas_path,
        BENCHMARK_FIXTURE,
        second,
        RecordedDraftRunner(),
    )

    assert first_result == second_result
    assert _tree_bytes(first) == _tree_bytes(second)
    assert len(_tree_bytes(first)) == 93
    assert replay_benchmark(atlas_path, BENCHMARK_FIXTURE, first) == first_result
    receipt = load_json(first / "receipt.json")
    assert len(receipt["closed_files"]) == 92
    assert {item["path"] for item in receipt["closed_files"]} == (
        set(_tree_bytes(first)) - {"receipt.json"}
    )
    sample = first / "runs/supported-agreement/repeat-0000"
    assert {item.name for item in sample.iterdir()} == {
        "artifact.json",
        "draft.json",
        "measurement.json",
        "receipt.json",
        "request.json",
    }


def test_benchmark_publication_refuses_clobber_without_changing_tree(
    tmp_path: Path,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    before = _tree_bytes(bundle)

    with pytest.raises(EvidenceError) as caught:
        run_benchmark(
            atlas_path,
            BENCHMARK_FIXTURE,
            bundle,
            RecordedDraftRunner(),
        )

    assert caught.value.code == "output_exists"
    assert _tree_bytes(bundle) == before


@pytest.mark.parametrize(
    "relative_path",
    [
        "runs/supported-agreement/repeat-0000/request.json",
        "runs/supported-agreement/repeat-0000/draft.json",
        "runs/supported-agreement/repeat-0000/artifact.json",
        "runs/supported-agreement/repeat-0000/receipt.json",
    ],
)
def test_benchmark_replay_rejects_modified_run_records(
    tmp_path: Path,
    relative_path: str,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    value = load_json(bundle / relative_path)
    assert isinstance(value, dict)
    if relative_path.endswith("draft.json"):
        claims = value["claims"]
        assert isinstance(claims, list)
        claim = claims[0]
        assert isinstance(claim, dict)
        claim["draft_text"] += " Modified."
    else:
        warnings = value["warnings"]
        assert isinstance(warnings, list)
        warnings.append("tampered")
    write_json(bundle / relative_path, value)
    _reclose_benchmark_file(bundle, relative_path)

    with pytest.raises(EvidenceError):
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)


@pytest.mark.parametrize("change", ["missing", "extra", "swapped"])
def test_benchmark_replay_rejects_missing_extra_and_swapped_run_files(
    tmp_path: Path,
    change: str,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    first = "runs/supported-agreement/repeat-0000/request.json"
    second = "runs/cross-layer-disagreement/repeat-0000/request.json"
    if change == "missing":
        (bundle / first).unlink()
    elif change == "extra":
        write_json(bundle / "unexpected.json", {"unexpected": True})
    else:
        first_bytes = (bundle / first).read_bytes()
        second_bytes = (bundle / second).read_bytes()
        (bundle / first).write_bytes(second_bytes)
        (bundle / second).write_bytes(first_bytes)
        _reclose_benchmark_file(bundle, first)
        _reclose_benchmark_file(bundle, second)

    with pytest.raises(EvidenceError):
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)


def test_benchmark_replay_rejects_semantic_digest_mismatch(
    tmp_path: Path,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    index = load_json(bundle / "index.json")
    assert isinstance(index, dict)
    runs = index["runs"]
    assert isinstance(runs, list)
    run = runs[0]
    assert isinstance(run, dict)
    run["semantic_sha256"] = "0" * 64
    write_json(bundle / "index.json", index)
    _reclose_benchmark_file(bundle, "index.json")

    with pytest.raises(EvidenceError) as caught:
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)
    assert caught.value.code == "semantic_digest_mismatch"


def test_benchmark_replay_recomputes_aggregate_metrics(
    tmp_path: Path,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    benchmark = load_json(bundle / "benchmark.json")
    assert isinstance(benchmark, dict)
    metrics = benchmark["metrics"]
    assert isinstance(metrics, dict)
    counts = metrics["verifier_counts"]
    assert isinstance(counts, dict)
    counts["refused"] = 3
    write_json(bundle / "benchmark.json", benchmark)
    _reclose_benchmark_file(bundle, "benchmark.json")

    with pytest.raises(EvidenceError) as caught:
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)
    assert caught.value.code == "benchmark_replay_mismatch"


def test_benchmark_replay_rejects_noncanonical_run_json(
    tmp_path: Path,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    relative_path = "runs/supported-agreement/repeat-0000/artifact.json"
    path = bundle / relative_path
    path.write_bytes(path.read_bytes().replace(b"{", b"{ ", 1))
    _reclose_benchmark_file(bundle, relative_path)

    with pytest.raises(EvidenceError) as caught:
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)
    assert caught.value.code == "noncanonical_json"


def test_benchmark_replay_rejects_thinking_mode_tampering(
    tmp_path: Path,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    relative_path = "runs/supported-agreement/repeat-0000/measurement.json"
    measurement = load_json(bundle / relative_path)
    assert isinstance(measurement, dict)
    configuration = measurement["runner_configuration"]
    assert isinstance(configuration, dict)
    assert configuration["thinking_enabled"] is None
    configuration["thinking_enabled"] = False
    write_json(bundle / relative_path, measurement)
    _reclose_benchmark_file(bundle, relative_path)

    with pytest.raises(EvidenceError) as caught:
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)
    assert caught.value.code == "invalid_thinking_mode"


def test_benchmark_replay_rejects_symlinks_and_path_traversal(
    tmp_path: Path,
) -> None:
    atlas_path, bundle = _recorded_benchmark(tmp_path)
    request_path = bundle / "runs/supported-agreement/repeat-0000/request.json"
    target = tmp_path / "request-copy.json"
    target.write_bytes(request_path.read_bytes())
    request_path.unlink()
    request_path.symlink_to(target)

    with pytest.raises(EvidenceError) as symlink:
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)
    assert symlink.value.code == "symlink_forbidden"

    receipt = load_json(bundle / "receipt.json")
    assert isinstance(receipt, dict)
    closed_files = receipt["closed_files"]
    assert isinstance(closed_files, list)
    item = closed_files[0]
    assert isinstance(item, dict)
    item["path"] = "../escape.json"
    write_json(bundle / "receipt.json", receipt)
    request_path.unlink()
    request_path.write_bytes(target.read_bytes())

    with pytest.raises(EvidenceError) as traversal:
        replay_benchmark(atlas_path, BENCHMARK_FIXTURE, bundle)
    assert traversal.value.code == "unsafe_path"


def test_benchmark_case_id_path_traversal_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    fixture = load_json(BENCHMARK_FIXTURE)
    assert isinstance(fixture, dict)
    cases = fixture["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    case["case_id"] = "../escape"
    fixture_path = tmp_path / "unsafe-fixture.json"
    write_json(fixture_path, fixture)
    output = tmp_path / "unsafe-output"

    with pytest.raises(EvidenceError) as caught:
        run_benchmark(
            _atlas_path(tmp_path),
            fixture_path,
            output,
            RecordedDraftRunner(),
        )
    assert caught.value.code == "unsafe_benchmark_case_id"
    assert not output.exists()


def test_semantic_variance_ignores_claim_ids_but_keeps_meaning(
    tmp_path: Path,
) -> None:
    atlas_path = _atlas_path(tmp_path)
    benchmark = run_benchmark(
        atlas_path,
        BENCHMARK_FIXTURE,
        tmp_path / "semantic",
        _AlternateClaimIdRunner(),
    )
    runs = benchmark["runs"]
    supported = [item for item in runs if item["case_id"] == "supported-agreement"]
    prohibited = [item for item in runs if item["case_id"] == "prohibited-clinical-causal"]

    assert supported[0]["draft_sha256"] != supported[1]["draft_sha256"]
    assert supported[0]["semantic_sha256"] == supported[1]["semantic_sha256"]
    assert prohibited[0]["semantic_sha256"] != prohibited[1]["semantic_sha256"]
    assert supported[0]["semantic_sha256"] != prohibited[0]["semantic_sha256"]
    assert benchmark["metrics"]["replay_semantic_variance"] == {
        "availability": "available",
        "denominator": 9,
        "numerator": 1,
        "value": 1 / 9,
    }

    atlas, request, draft = _draft()
    accepted, _ = verify_model_draft(request, atlas, draft)
    refused, _ = verify_model_draft(
        request,
        atlas,
        _replace_claim(
            draft,
            "claim:evidence:synthetic-pten-mutation-1",
            draft_text="This protein drives disease and should guide care.",
        ),
    )
    assert semantic_artifact_sha256(accepted) != semantic_artifact_sha256(refused)


def test_cli_compile_replay_and_benchmark(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    atlas_path = _atlas_path(tmp_path)
    bundle = tmp_path / "bundle"
    assert (
        main(
            [
                "fixture-compile",
                "--atlas",
                str(atlas_path),
                "--protein",
                "P60484",
                "--request-id",
                "cli:test",
                "--output",
                str(bundle),
            ]
        )
        == 0
    )
    assert "communication_status=accepted" in capsys.readouterr().out

    assert (
        main(
            [
                "replay",
                "--atlas",
                str(atlas_path),
                "--bundle",
                str(bundle),
            ]
        )
        == 0
    )
    assert "communication_replay=verified" in capsys.readouterr().out

    benchmark = tmp_path / "benchmark"
    assert (
        main(
            [
                "benchmark",
                "--atlas",
                str(atlas_path),
                "--fixtures",
                str(BENCHMARK_FIXTURE),
                "--output",
                str(benchmark),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "runs=18" in captured.out
    assert "local_model_downloaded=false" in captured.out
    assert captured.err == ""
    assert (
        main(
            [
                "benchmark-replay",
                "--atlas",
                str(atlas_path),
                "--fixtures",
                str(BENCHMARK_FIXTURE),
                "--bundle",
                str(benchmark),
            ]
        )
        == 0
    )
    assert "benchmark_replay=verified runs=18" in capsys.readouterr().out


def test_publication_refuses_clobber_without_changing_output(
    tmp_path: Path,
) -> None:
    atlas_path = _atlas_path(tmp_path)
    output = tmp_path / "communication"
    compile_recorded_fixture(atlas_path, "P60484", "no-clobber", "clean", output)
    before = {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()}

    with pytest.raises(EvidenceError) as caught:
        compile_recorded_fixture(atlas_path, "P60484", "no-clobber", "unknown_source", output)

    assert caught.value.code == "output_exists"
    assert before == {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()}


def test_receipt_and_artifact_strict_parsers_close_generated_records() -> None:
    atlas, request, draft = _draft()
    artifact, receipt = verify_model_draft(request, atlas, draft)

    assert VerifiedCommunicationArtifact.from_dict(artifact.to_dict()) == artifact
    assert CommunicationReceipt.from_dict(receipt.to_dict()) == receipt


def test_sentence_mapping_rejects_unknown_verified_claim() -> None:
    atlas, request, draft = _draft()
    artifact, _ = verify_model_draft(request, atlas, draft)
    value = artifact.to_dict()
    sentences = value["sentences"]
    assert isinstance(sentences, list)
    first = sentences[0]
    assert isinstance(first, dict)
    first["verified_claim_ids"] = ["verified:" + "0" * 64]

    with pytest.raises(EvidenceError) as caught:
        VerifiedCommunicationArtifact.from_dict(value)
    assert caught.value.code == "unknown_sentence_verified_claim"


@pytest.mark.parametrize(
    ("raw_model_output", "error_code"),
    [
        ("{not-json", "invalid_json"),
        ("{}", "invalid_record"),
    ],
)
def test_invalid_ollama_output_is_counted_and_benchmark_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_model_output: str,
    error_code: str,
) -> None:
    generated_requests: list[dict[str, Any]] = []
    response = canonical_json_bytes(
        {
            "eval_count": 3,
            "prompt_eval_count": 5,
            "response": raw_model_output,
        }
    )
    monkeypatch.setattr(
        communication_module,
        "_open_local_model_request",
        _ollama_responder(response, generated_requests=generated_requests),
    )
    atlas_path = _atlas_path(tmp_path)
    output = tmp_path / "malformed"

    benchmark = run_benchmark(
        atlas_path,
        BENCHMARK_FIXTURE,
        output,
        OllamaRunner(
            "http://127.0.0.1:11434",
            "installed-test-model",
            OLLAMA_MODEL_DIGEST,
            OLLAMA_RUNTIME_VERSION,
            False,
            context_window=4096,
            declared_environment={"OLLAMA_TEST_SETTING": "declared-value"},
        ),
    )

    assert benchmark["metrics"]["invalid_model_output_rate"] == {
        "availability": "available",
        "denominator": 18,
        "numerator": 18,
        "value": 1.0,
    }
    assert benchmark["metrics"]["verifier_counts"] == {
        "accepted_or_partially_excluded": 0,
        "refused": 18,
        "total": 18,
    }
    assert all(
        item["invalid_model_output"]
        == {
            "error_code": error_code,
            "output_sha256": sha256_bytes(raw_model_output.encode("utf-8")),
            "output_size_bytes": len(raw_model_output.encode("utf-8")),
        }
        and item["terminal_state"] == "refused"
        and item["draft_sha256"] is None
        for item in benchmark["runs"]
    )
    assert raw_model_output not in (output / "benchmark.json").read_text(encoding="ascii")
    receipt = load_json(output / "receipt.json")
    assert receipt["terminal_state"] == "closed"
    assert benchmark["runner_configuration"] == {
        "context_window": 4096,
        "declared_environment": [
            {
                "name": "OLLAMA_TEST_SETTING",
                "provenance": "declared_unverified",
                "value": "declared-value",
            }
        ],
        "endpoint_locality": "localhost_only",
        "json_mode": "strict_json",
        "options": [{"name": "num_ctx", "value": 4096}],
        "temperature": 0.0,
        "thinking_enabled": False,
        "timeout_seconds": 120.0,
    }
    assert benchmark["model_identity"]["runtime_version"] == "0.35.1"
    assert benchmark["model_identity"]["model_manifest_sha256"] == OLLAMA_MODEL_DIGEST
    assert len(generated_requests) == 18
    assert all(request["think"] is False for request in generated_requests)
    assert all("think" not in request["options"] for request in generated_requests)
    invalid_run = output / "runs/supported-agreement/repeat-0000"
    assert {item.name for item in invalid_run.iterdir()} == {
        "artifact.json",
        "invalid-output.json",
        "measurement.json",
        "receipt.json",
        "request.json",
    }
    invalid_record = load_json(invalid_run / "invalid-output.json")
    assert invalid_record["invalid_model_output"]["output_sha256"] == sha256_bytes(
        raw_model_output.encode("utf-8")
    )
    assert "raw_output" not in invalid_record
    monkeypatch.setattr(
        communication_module,
        "_open_local_model_request",
        lambda *args, **kwargs: pytest.fail("offline replay attempted network access"),
    )
    assert replay_benchmark(atlas_path, BENCHMARK_FIXTURE, output) == benchmark


@pytest.mark.parametrize(
    ("observed_digest", "observed_version", "error_code"),
    [
        ("b" * 64, OLLAMA_RUNTIME_VERSION, "model_manifest_digest_mismatch"),
        (OLLAMA_MODEL_DIGEST, "different-runtime", "runtime_identity_mismatch"),
    ],
)
def test_ollama_identity_drift_aborts_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_digest: str,
    observed_version: str,
    error_code: str,
) -> None:
    monkeypatch.setattr(
        communication_module,
        "_open_local_model_request",
        _ollama_responder(
            b"",
            observed_digest=observed_digest,
            observed_version=observed_version,
        ),
    )
    output = tmp_path / "identity-drift"

    with pytest.raises(EvidenceError) as caught:
        run_benchmark(
            _atlas_path(tmp_path),
            BENCHMARK_FIXTURE,
            output,
            OllamaRunner(
                "http://127.0.0.1:11434",
                "installed-test-model",
                OLLAMA_MODEL_DIGEST,
                OLLAMA_RUNTIME_VERSION,
                False,
            ),
        )
    assert caught.value.code == error_code
    assert not output.exists()


@pytest.mark.parametrize(
    ("drift_kind", "drift_validation", "error_code", "generated_count"),
    [
        ("digest", 3, "model_manifest_digest_mismatch", 1),
        ("runtime", 4, "runtime_identity_mismatch", 1),
    ],
)
def test_ollama_identity_is_rechecked_at_every_generation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
    drift_validation: int,
    error_code: str,
    generated_count: int,
) -> None:
    counts = {"generate": 0, "tags": 0, "version": 0}
    response = canonical_json_bytes(
        {
            "eval_count": 1,
            "prompt_eval_count": 1,
            "response": "{}",
        }
    )

    def respond(
        request: urllib.request.Request,
        timeout_seconds: float | None = None,
    ) -> _FakeHttpResponse:
        del timeout_seconds
        if request.full_url.endswith("/api/version"):
            counts["version"] += 1
            version = (
                "changed-runtime"
                if drift_kind == "runtime" and counts["version"] >= drift_validation
                else OLLAMA_RUNTIME_VERSION
            )
            return _FakeHttpResponse(canonical_json_bytes({"version": version}))
        if request.full_url.endswith("/api/tags"):
            counts["tags"] += 1
            digest = (
                "b" * 64
                if drift_kind == "digest" and counts["tags"] >= drift_validation
                else OLLAMA_MODEL_DIGEST
            )
            return _FakeHttpResponse(
                canonical_json_bytes(
                    {
                        "models": [
                            {
                                "digest": digest,
                                "model": "installed-test-model",
                                "name": "installed-test-model",
                            }
                        ]
                    }
                )
            )
        counts["generate"] += 1
        return _FakeHttpResponse(response)

    monkeypatch.setattr(
        communication_module,
        "_open_local_model_request",
        respond,
    )
    output = tmp_path / f"stateful-{drift_kind}"

    with pytest.raises(EvidenceError) as caught:
        run_benchmark(
            _atlas_path(tmp_path),
            BENCHMARK_FIXTURE,
            output,
            OllamaRunner(
                "http://127.0.0.1:11434",
                "installed-test-model",
                OLLAMA_MODEL_DIGEST,
                OLLAMA_RUNTIME_VERSION,
                False,
            ),
        )
    assert caught.value.code == error_code
    assert counts["generate"] == generated_count
    assert not output.exists()


def test_cli_refuses_tag_only_ollama_identity_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        communication_module,
        "_open_local_model_request",
        lambda *args, **kwargs: pytest.fail("tag-only identity attempted network access"),
    )
    output = tmp_path / "tag-only"

    assert (
        main(
            [
                "benchmark",
                "--atlas",
                str(_atlas_path(tmp_path)),
                "--fixtures",
                str(BENCHMARK_FIXTURE),
                "--output",
                str(output),
                "--runner",
                "ollama",
                "--model",
                "installed-test-model",
                "--runtime-version",
                OLLAMA_RUNTIME_VERSION,
                "--thinking",
                "disabled",
            ]
        )
        == 2
    )
    assert "missing_model_manifest_digest" in capsys.readouterr().err
    assert not output.exists()


def test_cli_refuses_omitted_ollama_thinking_mode_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        communication_module,
        "_open_local_model_request",
        lambda *args, **kwargs: pytest.fail("missing thinking mode attempted network access"),
    )
    output = tmp_path / "missing-thinking"

    assert (
        main(
            [
                "benchmark",
                "--atlas",
                str(_atlas_path(tmp_path)),
                "--fixtures",
                str(BENCHMARK_FIXTURE),
                "--output",
                str(output),
                "--runner",
                "ollama",
                "--model",
                "installed-test-model",
                "--model-digest",
                OLLAMA_MODEL_DIGEST,
                "--runtime-version",
                OLLAMA_RUNTIME_VERSION,
            ]
        )
        == 2
    )
    assert "missing_thinking_mode" in capsys.readouterr().err
    assert not output.exists()


def test_ollama_transport_failure_aborts_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        communication_module,
        "_open_local_model_request",
        _ollama_responder(
            b"",
            generation_error=urllib.error.URLError("offline"),
        ),
    )
    output = tmp_path / "transport-failure"

    with pytest.raises(EvidenceError) as caught:
        run_benchmark(
            _atlas_path(tmp_path),
            BENCHMARK_FIXTURE,
            output,
            OllamaRunner(
                "http://127.0.0.1:11434",
                "installed-test-model",
                OLLAMA_MODEL_DIGEST,
                OLLAMA_RUNTIME_VERSION,
                False,
            ),
        )
    assert caught.value.code == "local_model_request_failed"
    assert not output.exists()


def test_ollama_adapter_refuses_nonlocal_endpoint() -> None:
    with pytest.raises(EvidenceError) as caught:
        OllamaRunner(
            "https://example.com",
            "model",
            OLLAMA_MODEL_DIGEST,
            OLLAMA_RUNTIME_VERSION,
            False,
        )
    assert caught.value.code == "nonlocal_model_endpoint"
