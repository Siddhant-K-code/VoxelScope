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
    sha256_bytes,
    write_json,
)
from voxelscope.evidence_communication import (
    BOUNDARY_TEXT,
    OllamaRunner,
    RecordedDraftRunner,
    RunnerResult,
    compile_recorded_fixture,
    derive_communication_request,
    recorded_model_draft,
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
    ROOT / "research/gbm-evidence-communication-benchmark-v2" / "benchmark-fixtures.json"
)


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
            runtime="pytest",
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

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
    response = canonical_json_bytes(
        {
            "eval_count": 3,
            "prompt_eval_count": 5,
            "response": raw_model_output,
        }
    )
    monkeypatch.setattr(
        communication_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeHttpResponse(response),
    )
    atlas_path = _atlas_path(tmp_path)
    output = tmp_path / "malformed"

    benchmark = run_benchmark(
        atlas_path,
        BENCHMARK_FIXTURE,
        output,
        OllamaRunner("http://127.0.0.1:11434", "installed-test-model"),
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


def test_ollama_transport_failure_aborts_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_transport(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(
        communication_module.urllib.request,
        "urlopen",
        fail_transport,
    )
    output = tmp_path / "transport-failure"

    with pytest.raises(EvidenceError) as caught:
        run_benchmark(
            _atlas_path(tmp_path),
            BENCHMARK_FIXTURE,
            output,
            OllamaRunner("http://127.0.0.1:11434", "installed-test-model"),
        )
    assert caught.value.code == "local_model_request_failed"
    assert not output.exists()


def test_ollama_adapter_refuses_nonlocal_endpoint() -> None:
    with pytest.raises(EvidenceError) as caught:
        OllamaRunner("https://example.com", "model")
    assert caught.value.code == "nonlocal_model_endpoint"
