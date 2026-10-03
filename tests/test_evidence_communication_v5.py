# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import inspect
import itertools
import json
import random
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from voxelscope.canonical import (
    EvidenceError,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    write_json,
)
from voxelscope.evidence_communication_v5 import (
    DESIGN_CONTRACT_SHA256_V5,
    PLAN_SCHEMA_SHA256_V5,
    PUBLICATION_SCHEMA_SHA256_V5,
    RENDERER_ID_V5,
    REQUEST_SCHEMA_SHA256_V5,
    control_discourse_request_v5,
    control_sentence_catalog_v5,
    optimize_discourse_plan_v5,
    publish_control_bundle_v5,
    render_plan_v5,
    replay_control_bundle_v5,
    verify_contract_identities_v5,
)
from voxelscope.evidence_communication_v5_records import (
    AUDIENCE_PROFILE_V5,
    CATALOG_SCHEMA_V5,
    NOT_APPLICABLE_NO_MODEL_V5,
    REQUEST_SCHEMA_V5,
    CaveatConstraintV5,
    DiscourseBudgetV5,
    DiscoursePlanV5,
    DiscourseRequestV5,
    OptionalUnitV5,
    OrderingConstraintV5,
    SentenceCatalogV5,
    SentenceUnitV5,
    SourceCustodyV5,
)

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    "research/evidence-communication-v5-discourse-planner-contract-v1/design-contract.json",
    "research/evidence-communication-v5-discourse-planner-contract-v1/discourse-plan.schema.json",
    "research/evidence-communication-v5-discourse-planner-contract-v1/discourse-request.schema.json",
    "research/evidence-communication-v5-discourse-planner-contract-v1/publication-record.schema.json",
    "src/voxelscope/atomic.py",
    "src/voxelscope/canonical.py",
    "src/voxelscope/evidence_communication_v5.py",
    "src/voxelscope/evidence_communication_v5_cli.py",
    "src/voxelscope/evidence_communication_v5_records.py",
    "src/voxelscope/records.py",
)


@pytest.fixture(scope="module")
def custody_repository(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repository = tmp_path_factory.mktemp("v5-custody-repository")
    for relative in IMPLEMENTATION_PATHS:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    commands = (
        ("init", "--quiet"),
        ("config", "user.email", "tests@example.invalid"),
        ("config", "user.name", "VoxelScope tests"),
        ("add", "--all"),
        ("commit", "--quiet", "-m", "fixture"),
    )
    for command in commands:
        subprocess.run(
            ["git", "-C", str(repository), *command],
            check=True,
            capture_output=True,
            text=True,
        )
    return repository


def _unit_text(unit: SentenceUnitV5) -> str:
    citations = " ".join(f"[{citation_id}]" for citation_id in unit.citation_ids)
    return "\n".join(f"{sentence} {citations}" for sentence in unit.canonical_sentences)


def _catalog_digest(catalog: SentenceCatalogV5) -> str:
    return sha256_bytes(canonical_json_bytes(catalog.to_dict()))


def _run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _problem(
    optionals: tuple[tuple[str, int, str], ...],
    *,
    max_optional_units: int,
    max_output_bytes: int,
    caveat_anchors: tuple[str, ...] = ("fact.anchor",),
    caveat_placements: tuple[str, ...] = ("before", "after"),
    ordering: tuple[tuple[str, str], ...] = (),
) -> tuple[SentenceCatalogV5, DiscourseRequestV5]:
    units = (
        SentenceUnitV5(
            "caveat.scope",
            "caveat",
            ("This is bounded contract evidence.",),
            ("source.contract",),
            (),
            ("caveat.scope",),
        ),
        SentenceUnitV5(
            "fact.anchor",
            "fact",
            ("Trusted code owns this fact.",),
            ("source.contract",),
            ("fact.anchor",),
            (),
        ),
        *(
            SentenceUnitV5(
                unit_id,
                "optional_context",
                (sentence,),
                ("source.contract",),
                (),
                (),
            )
            for unit_id, _, sentence in optionals
        ),
    )
    catalog = SentenceCatalogV5(
        CATALOG_SCHEMA_V5,
        "test-catalog",
        ("source.contract",),
        tuple(sorted(units, key=lambda item: item.unit_id)),
    )
    by_id = {unit.unit_id: unit for unit in catalog.units}
    request = DiscourseRequestV5(
        schema_version=REQUEST_SCHEMA_V5,
        case_id="bounded-oracle-case",
        sentence_catalog_sha256=_catalog_digest(catalog),
        study_declaration_sha256=DESIGN_CONTRACT_SHA256_V5,
        renderer_id=RENDERER_ID_V5,
        audience_profile=AUDIENCE_PROFILE_V5,
        mandatory_unit_ids=("caveat.scope", "fact.anchor"),
        mandatory_fact_ids=("fact.anchor",),
        mandatory_caveat_ids=("caveat.scope",),
        ordering_constraints=tuple(
            OrderingConstraintV5(before, after) for before, after in sorted(ordering)
        ),
        caveat_constraints=(
            CaveatConstraintV5(
                "caveat.scope",
                tuple(sorted(caveat_anchors)),
                caveat_placements,
            ),
        ),
        optional_units=tuple(
            OptionalUnitV5(
                unit_id,
                utility,
                len(_unit_text(by_id[unit_id]).encode("utf-8")),
                ("fact.anchor",),
            )
            for unit_id, utility, _ in optionals
        ),
        budget=DiscourseBudgetV5(max_output_bytes, max_optional_units),
    )
    return catalog, request


def _oracle(
    catalog: SentenceCatalogV5,
    request: DiscourseRequestV5,
) -> dict[str, Any] | None:
    units = {unit.unit_id: unit for unit in catalog.units}
    optional = {item.unit_id: item for item in request.optional_units}
    best: tuple[tuple[int, int, bytes], dict[str, Any]] | None = None
    caveat_choices = tuple(
        tuple(
            {
                "anchor_id": anchor_id,
                "caveat_unit_id": constraint.caveat_unit_id,
                "placement": placement,
            }
            for anchor_id in constraint.allowed_anchor_ids
            for placement in constraint.allowed_placements
        )
        for constraint in request.caveat_constraints
    )
    for size in range(request.budget.max_optional_units + 1):
        for subset in itertools.combinations(tuple(optional), size):
            selected = set(request.mandatory_unit_ids) | set(subset)
            utility = sum(optional[unit_id].utility for unit_id in subset)
            for decisions in itertools.product(*caveat_choices):
                sorted_decisions = sorted(
                    decisions,
                    key=lambda item: (
                        item["caveat_unit_id"],
                        item["anchor_id"],
                        item["placement"],
                    ),
                )
                for ordered in itertools.permutations(sorted(selected)):
                    positions = {unit_id: index for index, unit_id in enumerate(ordered)}
                    if any(
                        positions[item.before_unit_id] >= positions[item.after_unit_id]
                        for item in request.ordering_constraints
                        if item.before_unit_id in positions and item.after_unit_id in positions
                    ):
                        continue
                    if any(
                        not any(
                            positions[anchor_id] < positions[unit_id]
                            for anchor_id in optional[unit_id].allowed_anchor_ids
                        )
                        for unit_id in subset
                    ):
                        continue
                    placement_valid = True
                    for decision in sorted_decisions:
                        is_before = (
                            positions[decision["caveat_unit_id"]] < positions[decision["anchor_id"]]
                        )
                        if is_before != (decision["placement"] == "before"):
                            placement_valid = False
                            break
                    if not placement_valid:
                        continue
                    prose = "\n".join(_unit_text(units[unit_id]) for unit_id in ordered)
                    emitted_bytes = len(prose.encode("utf-8"))
                    if emitted_bytes > request.budget.max_output_bytes:
                        continue
                    decision_key = {
                        "caveat_anchor_decisions": sorted_decisions,
                        "ordered_unit_ids": list(ordered),
                        "selected_optional_unit_ids": list(subset),
                    }
                    score = (
                        -utility,
                        emitted_bytes,
                        canonical_json_bytes(decision_key),
                    )
                    if best is None or score < best[0]:
                        best = (score, decision_key)
    return None if best is None else best[1]


def test_contract_and_schema_identities_recompute() -> None:
    assert verify_contract_identities_v5(ROOT) == {
        "design_contract_sha256": DESIGN_CONTRACT_SHA256_V5,
        "discourse_plan_schema_sha256": PLAN_SCHEMA_SHA256_V5,
        "discourse_request_schema_sha256": REQUEST_SCHEMA_SHA256_V5,
        "publication_record_schema_sha256": PUBLICATION_SCHEMA_SHA256_V5,
    }


def test_control_request_and_plan_have_only_merged_schema_fields() -> None:
    catalog = control_sentence_catalog_v5()
    request = control_discourse_request_v5(catalog)
    result = optimize_discourse_plan_v5(request, catalog)

    assert set(request.to_dict()) == {
        "audience_profile",
        "budget",
        "case_id",
        "caveat_constraints",
        "mandatory_caveat_ids",
        "mandatory_fact_ids",
        "mandatory_unit_ids",
        "optional_units",
        "ordering_constraints",
        "renderer_id",
        "schema_version",
        "sentence_catalog_sha256",
        "study_declaration_sha256",
    }
    assert set(result.plan.to_dict()) == {
        "caveat_anchor_decisions",
        "ordered_unit_ids",
        "plan_mode",
        "request_sha256",
        "schema_version",
        "selected_optional_unit_ids",
    }
    with pytest.raises(EvidenceError, match="extra=.*text"):
        DiscoursePlanV5.from_dict({**result.plan.to_dict(), "text": "forbidden"})


def test_optimizer_matches_independent_exhaustive_oracle_many_cases() -> None:
    generator = random.Random(20261003)
    for case_index in range(64):
        count = generator.randint(0, 3)
        optionals = tuple(
            (
                f"optional.{chr(ord('a') + index)}",
                generator.randint(0, 5),
                "x" * generator.randint(1, 24) + ".",
            )
            for index in range(count)
        )
        provisional_catalog, _ = _problem(
            optionals,
            max_optional_units=count,
            max_output_bytes=12000,
        )
        mandatory_bytes = len(
            (
                _unit_text(provisional_catalog.units[0])
                + "\n"
                + _unit_text(
                    next(
                        unit for unit in provisional_catalog.units if unit.unit_id == "fact.anchor"
                    )
                )
            ).encode("utf-8")
        )
        full_bytes = (
            sum(len(_unit_text(unit).encode("utf-8")) for unit in provisional_catalog.units)
            + len(provisional_catalog.units)
            - 1
        )
        maximum = generator.randint(0, count)
        budget = generator.randint(mandatory_bytes, full_bytes)
        ordering: tuple[tuple[str, str], ...] = ()
        if count and case_index % 3 == 0:
            ordering = (("fact.anchor", optionals[0][0]),)
        catalog, request = _problem(
            optionals,
            max_optional_units=maximum,
            max_output_bytes=budget,
            ordering=ordering,
        )
        expected = _oracle(catalog, request)
        assert expected is not None
        actual = optimize_discourse_plan_v5(
            request,
            catalog,
            max_candidate_evaluations=100_000,
        )
        assert {
            "caveat_anchor_decisions": [
                item.to_dict() for item in actual.plan.caveat_anchor_decisions
            ],
            "ordered_unit_ids": list(actual.plan.ordered_unit_ids),
            "selected_optional_unit_ids": list(actual.plan.selected_optional_unit_ids),
        } == expected


def test_optimizer_ties_utility_then_bytes_then_decision_key() -> None:
    catalog, request = _problem(
        (
            ("optional.a", 3, "This is much longer optional context."),
            ("optional.b", 3, "Short."),
        ),
        max_optional_units=1,
        max_output_bytes=12000,
    )
    result = optimize_discourse_plan_v5(request, catalog)
    assert result.plan.selected_optional_unit_ids == ("optional.b",)

    catalog, request = _problem(
        (
            ("optional.a", 3, "Equal A."),
            ("optional.b", 3, "Equal B."),
        ),
        max_optional_units=1,
        max_output_bytes=12000,
    )
    result = optimize_discourse_plan_v5(request, catalog)
    assert result.plan.selected_optional_unit_ids == ("optional.a",)
    assert result.plan.to_dict() == {
        **result.plan.to_dict(),
        "ordered_unit_ids": list(_oracle(catalog, request)["ordered_unit_ids"]),  # type: ignore[index]
    }


def test_optimizer_ties_order_and_caveat_anchor_by_canonical_key() -> None:
    units = (
        SentenceUnitV5(
            "caveat.scope",
            "caveat",
            ("Boundary.",),
            ("source.contract",),
            (),
            ("caveat.scope",),
        ),
        SentenceUnitV5(
            "fact.a",
            "fact",
            ("Fact A.",),
            ("source.contract",),
            ("fact.a",),
            (),
        ),
        SentenceUnitV5(
            "fact.b",
            "fact",
            ("Fact B.",),
            ("source.contract",),
            ("fact.b",),
            (),
        ),
    )
    catalog = SentenceCatalogV5(
        CATALOG_SCHEMA_V5,
        "anchor-tie",
        ("source.contract",),
        units,
    )
    request = DiscourseRequestV5(
        REQUEST_SCHEMA_V5,
        "anchor-tie",
        _catalog_digest(catalog),
        DESIGN_CONTRACT_SHA256_V5,
        RENDERER_ID_V5,
        AUDIENCE_PROFILE_V5,
        ("caveat.scope", "fact.a", "fact.b"),
        ("fact.a", "fact.b"),
        ("caveat.scope",),
        (OrderingConstraintV5("fact.b", "caveat.scope"),),
        (
            CaveatConstraintV5(
                "caveat.scope",
                ("fact.a", "fact.b"),
                ("after",),
            ),
        ),
        (),
        DiscourseBudgetV5(12000, 0),
    )
    result = optimize_discourse_plan_v5(request, catalog)
    assert result.plan.ordered_unit_ids == ("fact.a", "fact.b", "caveat.scope")
    assert result.plan.caveat_anchor_decisions[0].anchor_id == "fact.a"
    assert _oracle(catalog, request) == {
        "caveat_anchor_decisions": [
            {
                "anchor_id": "fact.a",
                "caveat_unit_id": "caveat.scope",
                "placement": "after",
            }
        ],
        "ordered_unit_ids": ["fact.a", "fact.b", "caveat.scope"],
        "selected_optional_unit_ids": [],
    }


def test_search_limit_fails_closed_instead_of_approximating() -> None:
    catalog, request = _problem(
        (
            ("optional.a", 1, "A."),
            ("optional.b", 2, "B."),
            ("optional.c", 3, "C."),
        ),
        max_optional_units=3,
        max_output_bytes=12000,
    )
    with pytest.raises(EvidenceError, match="explicit limit"):
        optimize_discourse_plan_v5(request, catalog, max_candidate_evaluations=4)


def test_infeasible_budget_stops_without_plan() -> None:
    catalog, request = _problem(
        (),
        max_optional_units=0,
        max_output_bytes=1,
    )
    with pytest.raises(EvidenceError) as error:
        optimize_discourse_plan_v5(request, catalog)
    assert error.value.code == "no_feasible_deterministic_plan"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda request: replace(
                request,
                ordering_constraints=(
                    OrderingConstraintV5("caveat.scope", "fact.anchor"),
                    OrderingConstraintV5("fact.anchor", "caveat.scope"),
                ),
            ),
            "ordering_cycle",
        ),
        (
            lambda request: replace(
                request,
                ordering_constraints=(OrderingConstraintV5("fact.anchor", "unknown.unit"),),
            ),
            "unknown_ordering_unit",
        ),
        (
            lambda request: replace(
                request,
                caveat_constraints=(
                    CaveatConstraintV5(
                        "caveat.scope",
                        ("unknown.anchor",),
                        ("after",),
                    ),
                ),
            ),
            "invalid_caveat_anchor",
        ),
        (
            lambda request: replace(request, audience_profile="general_public"),
            "audience_profile_drift",
        ),
    ],
)
def test_request_adversaries_are_rejected(
    mutation: Any,
    code: str,
) -> None:
    _, request = _problem((), max_optional_units=0, max_output_bytes=12000)
    with pytest.raises(EvidenceError) as error:
        mutation(request)
    assert error.value.code == code


def test_duplicate_and_noncanonical_lists_are_rejected() -> None:
    _, request = _problem((), max_optional_units=0, max_output_bytes=12000)
    with pytest.raises(EvidenceError) as duplicate:
        replace(request, mandatory_unit_ids=("fact.anchor", "fact.anchor"))
    assert duplicate.value.code == "duplicate_record_value"
    with pytest.raises(EvidenceError) as noncanonical:
        replace(request, mandatory_unit_ids=("fact.anchor", "caveat.scope"))
    assert noncanonical.value.code == "noncanonical_list_order"


def test_plan_rejects_mandatory_omission_and_bad_caveat_placement() -> None:
    catalog, request = _problem((), max_optional_units=0, max_output_bytes=12000)
    result = optimize_discourse_plan_v5(request, catalog)
    omitted = replace(result.plan, ordered_unit_ids=("fact.anchor",))
    with pytest.raises(EvidenceError) as omission:
        render_plan_v5(request, catalog, omitted)
    assert omission.value.code == "plan_unit_set_mismatch"
    decision = result.plan.caveat_anchor_decisions[0]
    bad_placement = "before" if decision.placement == "after" else "after"
    invalid = replace(
        result.plan,
        caveat_anchor_decisions=(replace(decision, placement=bad_placement),),
    )
    with pytest.raises(EvidenceError) as placement:
        render_plan_v5(request, catalog, invalid)
    assert placement.value.code == "caveat_placement_violation"


def test_control_publication_is_byte_identical_and_replayable(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = publish_control_bundle_v5(first, "control-run", custody_repository)
    second_result = publish_control_bundle_v5(second, "control-run", custody_repository)

    assert (
        first_result
        == second_result
        == replay_control_bundle_v5(
            first,
            custody_repository,
        )
    )
    assert first_result.optional_utility == 8
    assert first_result.emitted_utf8_bytes == 590
    assert first_result.request_sha256 == (
        "30bd85f9dbd3e2850ae90eaae2f5e643c4d2052eefcb128c8c7c4caafc51e03f"
    )
    assert first_result.plan_sha256 == (
        "2b444462bf0a84857dde60faad5dd559db0071ddc01b25e21abcc0fb1b48609e"
    )
    assert first_result.output_sha256 == (
        "c92770ff757b5ab46b3207a1965055695be2bde773964656747ec24e1f62b6ef"
    )
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    measurement = load_json(first / "measurement-status.json")
    assert measurement["model_inference"] == NOT_APPLICABLE_NO_MODEL_V5
    assert measurement["model_input_tokens"] == NOT_APPLICABLE_NO_MODEL_V5
    assert measurement["model_output_tokens"] == NOT_APPLICABLE_NO_MODEL_V5
    assert measurement["model_latency_ms"] == NOT_APPLICABLE_NO_MODEL_V5
    manifest = load_json(first / "implementation-manifest.json")
    assert tuple(item["path"] for item in manifest["files"]) == IMPLEMENTATION_PATHS
    index = load_json(first / "index.json")
    assert index["custody"]["source_revision"] == _run_git(custody_repository, "rev-parse", "HEAD")
    assert index["custody"]["source_root_tree"] == _run_git(
        custody_repository,
        "rev-parse",
        "HEAD^{tree}",
    )
    assert load_json(first / "sentence-catalog.json") == control_sentence_catalog_v5().to_dict()


def test_eligibility_is_persisted_before_optimizer_action(
    tmp_path: Path,
    custody_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxelscope.evidence_communication_v5 as implementation

    original = implementation.optimize_discourse_plan_v5
    output = tmp_path / "bundle"
    checked_before_publication = False

    def checked(*args: Any, **kwargs: Any) -> Any:
        nonlocal checked_before_publication
        stages = tuple(tmp_path.glob(".bundle.tmp-*"))
        if stages:
            assert len(stages) == 1
            assert (stages[0] / "eligibility.json").is_file()
            eligibility = load_json(stages[0] / "eligibility.json")
            assert eligibility["gate_result"] == "model_inference_forbidden"
            assert eligibility["model_candidate_action_count"] == 0
            checked_before_publication = True
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "optimize_discourse_plan_v5", checked)
    publish_control_bundle_v5(output, "eligibility-order", custody_repository)
    assert checked_before_publication


def test_receipt_is_absent_at_atomic_rename_and_written_last(
    tmp_path: Path,
    custody_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxelscope.evidence_communication_v5 as implementation

    original = implementation.rename_no_replace
    observations: list[bool] = []

    def checked(source: Path, destination: Path) -> None:
        observations.append((source / "receipt.json").exists())
        original(source, destination)

    monkeypatch.setattr(implementation, "rename_no_replace", checked)
    output = tmp_path / "bundle"
    publish_control_bundle_v5(output, "receipt-last", custody_repository)
    assert observations == [False]
    assert (output / "receipt.json").is_file()


def test_publication_collision_and_symlink_parent_fail_closed(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    output = tmp_path / "bundle"
    publish_control_bundle_v5(output, "collision", custody_repository)
    with pytest.raises(EvidenceError) as collision:
        publish_control_bundle_v5(output, "collision", custody_repository)
    assert collision.value.code == "output_exists"

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(EvidenceError) as symlink:
        publish_control_bundle_v5(
            linked_parent / "bundle",
            "symlink",
            custody_repository,
        )
    assert symlink.value.code == "symlink_forbidden"


def test_publication_api_accepts_only_repository_derived_custody(
    tmp_path: Path,
) -> None:
    parameters = tuple(inspect.signature(publish_control_bundle_v5).parameters)
    assert parameters == ("output", "run_id", "repository_root")
    invented = SourceCustodyV5("f" * 40, "e" * 40, "d" * 64, "clean")
    with pytest.raises(EvidenceError) as error:
        publish_control_bundle_v5(tmp_path / "bundle", "invented", invented)  # type: ignore[arg-type]
    assert error.value.code == "unsafe_repository_root"


def test_publication_rejects_untracked_source_entries(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    repository = tmp_path / "repository"
    shutil.copytree(custody_repository, repository)
    (repository / "untracked.txt").write_text("not committed\n")
    with pytest.raises(EvidenceError) as error:
        publish_control_bundle_v5(tmp_path / "bundle", "dirty", repository)
    assert error.value.code == "source_tree_not_clean"


def test_replay_rejects_source_revision_and_tree_drift(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    repository = tmp_path / "repository"
    shutil.copytree(custody_repository, repository)
    bundle = tmp_path / "bundle"
    publish_control_bundle_v5(bundle, "source-drift", repository)
    path = repository / "src/voxelscope/evidence_communication_v5_cli.py"
    path.write_text(path.read_text() + "\n")
    _run_git(repository, "add", "--all")
    _run_git(repository, "commit", "--quiet", "-m", "source drift")
    with pytest.raises(EvidenceError) as error:
        replay_control_bundle_v5(bundle, repository)
    assert error.value.code == "source_custody_mismatch"


def test_replay_rejects_coordinated_forged_custody(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    source = tmp_path / "source"
    publish_control_bundle_v5(source, "forged-custody", custody_repository)
    bundle = tmp_path / "forged"
    shutil.copytree(source, bundle)
    forged = {
        "implementation_manifest_sha256": "d" * 64,
        "source_revision": "f" * 40,
        "source_root_tree": "e" * 40,
        "source_tree_state": "clean",
    }
    for filename in ("eligibility.json", "index.json", "receipt.json"):
        value = load_json(bundle / filename)
        value["custody"] = forged
        write_json(bundle / filename, value)
    with pytest.raises(EvidenceError) as error:
        replay_control_bundle_v5(bundle, custody_repository)
    assert error.value.code == "source_custody_mismatch"


def test_replay_rejects_catalog_and_manifest_tamper(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    source = tmp_path / "source"
    publish_control_bundle_v5(source, "closed-inputs", custody_repository)
    for filename in ("sentence-catalog.json", "implementation-manifest.json"):
        bundle = tmp_path / filename
        shutil.copytree(source, bundle)
        value = load_json(bundle / filename)
        value["terminal_state"] = "tampered"
        write_json(bundle / filename, value)
        with pytest.raises(EvidenceError):
            replay_control_bundle_v5(bundle, custody_repository)


def test_atomic_collision_preserves_concurrent_destination(
    tmp_path: Path,
    custody_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxelscope.evidence_communication_v5 as implementation

    original = implementation.rename_no_replace
    output = tmp_path / "bundle"

    def collide(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_text("concurrent publisher\n")
        original(source, destination)

    monkeypatch.setattr(implementation, "rename_no_replace", collide)
    with pytest.raises(EvidenceError) as error:
        publish_control_bundle_v5(output, "race", custody_repository)
    assert error.value.code == "output_exists"
    assert (output / "owner.txt").read_text() == "concurrent publisher\n"
    assert not (output / "receipt.json").exists()


def test_receipt_failure_leaves_unclosed_bundle(
    tmp_path: Path,
    custody_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxelscope.evidence_communication_v5 as implementation

    output = tmp_path / "bundle"

    def fail_receipt(path: Path, receipt_bytes: bytes) -> None:
        del path, receipt_bytes
        raise OSError("simulated crash before receipt")

    monkeypatch.setattr(implementation, "_write_receipt_exclusive", fail_receipt)
    with pytest.raises(OSError, match="simulated crash"):
        publish_control_bundle_v5(output, "unclosed", custody_repository)
    assert output.is_dir()
    assert not (output / "receipt.json").exists()
    with pytest.raises(EvidenceError) as error:
        replay_control_bundle_v5(output, custody_repository)
    assert error.value.code == "bundle_file_set_mismatch"


def test_publication_fsync_order(
    tmp_path: Path,
    custody_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxelscope.evidence_communication_v5 as implementation

    output = tmp_path / "bundle"
    events: list[str] = []
    original_directory_fsync = implementation._fsync_directory
    original_rename = implementation.rename_no_replace
    original_receipt = implementation._write_receipt_exclusive

    def fsync_directory(path: Path) -> None:
        if path.name.startswith(".bundle.tmp-"):
            events.append("stage_fsync")
        elif path == output.parent:
            events.append("parent_fsync")
        elif path == output:
            events.append("output_fsync")
        original_directory_fsync(path)

    def rename(source: Path, destination: Path) -> None:
        events.append("rename")
        original_rename(source, destination)

    def write_receipt(path: Path, receipt_bytes: bytes) -> None:
        events.append("receipt_write_fsync")
        original_receipt(path, receipt_bytes)

    monkeypatch.setattr(implementation, "_fsync_directory", fsync_directory)
    monkeypatch.setattr(implementation, "rename_no_replace", rename)
    monkeypatch.setattr(implementation, "_write_receipt_exclusive", write_receipt)
    publish_control_bundle_v5(output, "fsync-order", custody_repository)
    final_stage_fsync = max(index for index, event in enumerate(events) if event == "stage_fsync")
    assert events[final_stage_fsync:] == [
        "stage_fsync",
        "rename",
        "parent_fsync",
        "receipt_write_fsync",
        "output_fsync",
    ]


def test_publication_rejects_link_like_existing_ancestor(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(EvidenceError) as error:
        publish_control_bundle_v5(
            linked / "missing-parent" / "bundle",
            "linked-ancestor",
            custody_repository,
        )
    assert error.value.code == "symlink_forbidden"


def test_custody_drift_is_rejected() -> None:
    with pytest.raises(EvidenceError) as state:
        SourceCustodyV5("f" * 40, "e" * 40, "d" * 64, "dirty")
    assert state.value.code == "source_tree_not_clean"
    with pytest.raises(EvidenceError) as revision:
        SourceCustodyV5("f" * 39, "e" * 40, "d" * 64, "clean")
    assert revision.value.code == "invalid_source_revision"


@pytest.mark.parametrize(
    ("filename", "field", "value"),
    [
        ("request.json", "audience_profile", "general_public"),
        ("eligibility.json", "terminal_state", "declared"),
        ("optimality-certificate.json", "terminal_state", "unproved"),
        ("plan.json", "request_sha256", "f" * 64),
        ("rendered-artifact.json", "output_sha256", "f" * 64),
        ("publication.json", "terminal_state", "published"),
        ("measurement-status.json", "model_input_tokens", "0"),
        ("verification.json", "optimality_gate", "failed"),
        ("index.json", "request_sha256", "f" * 64),
        ("receipt.json", "terminal_state", "published"),
    ],
)
def test_every_persisted_record_rejects_terminal_or_identity_tamper(
    tmp_path: Path,
    custody_repository: Path,
    filename: str,
    field: str,
    value: Any,
) -> None:
    source = tmp_path / "source"
    publish_control_bundle_v5(source, "tamper", custody_repository)
    bundle = tmp_path / f"tampered-{filename}"
    shutil.copytree(source, bundle)
    record = load_json(bundle / filename)
    record[field] = value
    write_json(bundle / filename, record)
    with pytest.raises(EvidenceError):
        replay_control_bundle_v5(bundle, custody_repository)


@pytest.mark.parametrize(
    "filename",
    [
        "eligibility.json",
        "implementation-manifest.json",
        "index.json",
        "measurement-status.json",
        "optimality-certificate.json",
        "plan.json",
        "publication.json",
        "receipt.json",
        "rendered-artifact.json",
        "request.json",
        "sentence-catalog.json",
        "verification.json",
    ],
)
def test_every_persisted_file_is_digest_or_canonicality_closed(
    tmp_path: Path,
    custody_repository: Path,
    filename: str,
) -> None:
    source = tmp_path / "source"
    if not source.exists():
        publish_control_bundle_v5(source, "digest-tamper", custody_repository)
    bundle = tmp_path / f"digest-{filename}"
    shutil.copytree(source, bundle)
    path = bundle / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(EvidenceError):
        replay_control_bundle_v5(bundle, custody_repository)


def test_bundle_rejects_missing_extra_and_symlink_entries(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    source = tmp_path / "source"
    publish_control_bundle_v5(source, "file-set", custody_repository)

    missing = tmp_path / "missing"
    shutil.copytree(source, missing)
    (missing / "plan.json").unlink()
    with pytest.raises(EvidenceError) as missing_error:
        replay_control_bundle_v5(missing, custody_repository)
    assert missing_error.value.code == "bundle_file_set_mismatch"

    extra = tmp_path / "extra"
    shutil.copytree(source, extra)
    (extra / "extra.json").write_text("{}")
    with pytest.raises(EvidenceError) as extra_error:
        replay_control_bundle_v5(extra, custody_repository)
    assert extra_error.value.code == "bundle_file_set_mismatch"

    linked = tmp_path / "linked"
    shutil.copytree(source, linked)
    (linked / "plan.json").unlink()
    (linked / "plan.json").symlink_to(source / "plan.json")
    with pytest.raises(EvidenceError) as linked_error:
        replay_control_bundle_v5(linked, custody_repository)
    assert linked_error.value.code == "symlink_forbidden"


def test_v5_implementation_has_no_model_or_network_runner_surface() -> None:
    paths = (
        ROOT / "src/voxelscope/evidence_communication_v5.py",
        ROOT / "src/voxelscope/evidence_communication_v5_records.py",
        ROOT / "src/voxelscope/evidence_communication_v5_cli.py",
    )
    forbidden_import_roots = {"http", "requests", "socket", "urllib"}
    forbidden_text = (
        "/api/generate",
        "Ollama",
        "ModelRunner",
        "prompt construction",
    )
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        assert imported_roots.isdisjoint(forbidden_import_roots)
        assert all(token not in source for token in forbidden_text)


def test_request_parser_rejects_extra_fields() -> None:
    request = control_discourse_request_v5()
    with pytest.raises(EvidenceError, match="extra=.*free_form_model_field"):
        DiscourseRequestV5.from_dict(
            {
                **request.to_dict(),
                "free_form_model_field": "forbidden",
            }
        )


def test_json_round_trip_remains_canonical(
    tmp_path: Path,
    custody_repository: Path,
) -> None:
    output = tmp_path / "bundle"
    publish_control_bundle_v5(output, "canonical", custody_repository)
    for path in output.iterdir():
        assert path.read_bytes() == canonical_json_bytes(json.loads(path.read_text()))
