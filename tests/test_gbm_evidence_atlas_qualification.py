# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from voxelscope.canonical import canonical_json_bytes, load_json

ROOT = Path(__file__).parents[1]
QUALIFICATION = ROOT / "research" / "gbm-evidence-atlas-qualification-v1"
REGISTRY = QUALIFICATION / "source-qualification-registry.json"
JOIN_MAP = QUALIFICATION / "join-map.json"
STUDY = QUALIFICATION / "first-study-preregistration.json"
PROTOCOL = ROOT / "docs" / "research" / "glioblastoma-evidence-atlas-source-qualification-v1.md"

REQUIRED_SOURCE_IDS = {
    "tcia-cptac-gbm-v16",
    "pdc000204",
    "pdc000205",
    "pdc000446",
    "pdc000448",
    "pdc000514",
    "pdc000515",
    "gdc-cptac-3-dr46",
    "gdc-tcga-gbm-dr46",
    "ivy-gap-live-2026-10-02",
    "hpa-v25.1",
    "depmap-26q1",
    "open-targets-26.09",
    "alphafold-db-v6",
    "glass-2022-05-31",
}
JOIN_LEVELS = {"patient", "sample", "gene", "protein", "phosphosite", "cohort"}
REQUIRED_OUTCOMES = {
    "modality_missingness",
    "join_coverage",
    "mutation_to_rna",
    "mutation_to_protein",
    "mutation_to_phosphorylation",
    "chain_concordance",
    "secondary_two_layer_discordance",
    "conflict_counts",
    "reproducibility",
}
RESULT_SHAPES = {
    "concordant",
    "null",
    "discordant",
    "secondary_two_layer_discordant",
    "mixed_or_indeterminate",
    "unavailable",
    "feasibility_refusal",
}


def _canonical(path: Path) -> dict[str, object]:
    value = load_json(path)
    assert path.read_bytes() == canonical_json_bytes(value)
    assert isinstance(value, dict)
    return value


def _assert_official_https(url: str) -> None:
    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.hostname
    assert parsed.username is None
    assert parsed.password is None
    assert "google." not in parsed.hostname
    assert "bing." not in parsed.hostname


def test_source_qualification_registry_is_canonical_and_complete() -> None:
    registry = _canonical(REGISTRY)
    assert registry["schema_version"] == "voxelscope/gbm-source-qualification/v1"
    assert registry["research_only"] is True
    sources = registry["sources"]
    assert isinstance(sources, list)
    assert {source["source_id"] for source in sources} == REQUIRED_SOURCE_IDS
    for source in sources:
        assert set(source) == {
            "access",
            "canonical_url",
            "evidence_urls",
            "identifiers",
            "join_keys",
            "license",
            "limitations",
            "modalities",
            "official_name",
            "release",
            "role",
            "scope",
            "source_id",
            "verification_status",
        }
        assert source["access"]["classification"] in {
            "open",
            "mixed",
            "partial",
            "operationally-blocked",
        }
        assert source["verification_status"] in {"verified", "partial", "blocked"}
        _assert_official_https(source["canonical_url"])
        for url in source["evidence_urls"]:
            _assert_official_https(url)
        assert source["limitations"]
        assert source["modalities"]


def test_registry_keeps_access_classes_and_pr_context_explicit() -> None:
    registry = _canonical(REGISTRY)
    context = registry["implementation_context"]
    implementation = context["synthetic_implementation"]
    assert implementation["state"] == "merged"
    assert implementation["merged"] is True
    assert implementation["commit"] == "88d074b9116d913ce720bd1a5e72a7e24933110f"
    product = context["product_contract"]
    assert product["state"] == "merged"
    assert product["merged"] is True
    assert product["commit"] == "4bd9d60b895981112af267f813e1a31c1aacf4fe"
    study_context = _canonical(STUDY)["implementation_context"]
    assert (
        study_context["synthetic_implementation_commit"]
        == "88d074b9116d913ce720bd1a5e72a7e24933110f"
    )
    assert study_context["product_contract_commit"] == "4bd9d60b895981112af267f813e1a31c1aacf4fe"
    by_id = {source["source_id"]: source for source in registry["sources"]}
    assert by_id["tcia-cptac-gbm-v16"]["access"]["classification"] == "mixed"
    assert by_id["gdc-cptac-3-dr46"]["access"]["classification"] == "mixed"
    assert by_id["glass-2022-05-31"]["verification_status"] == "blocked"
    assert by_id["depmap-26q1"]["verification_status"] == "blocked"
    assert by_id["pdc000204"]["license"]["status"] == "unverified"
    assert by_id["pdc000205"]["access"]["unverified_assets"]


def test_join_map_distinguishes_levels_and_references_qualified_sources() -> None:
    registry = _canonical(REGISTRY)
    join_map = _canonical(JOIN_MAP)
    source_ids = {source["source_id"] for source in registry["sources"]}
    assert join_map["schema_version"] == "voxelscope/gbm-join-map/v1"
    assert {item["level"] for item in join_map["join_levels"]} == JOIN_LEVELS
    joins = join_map["joins"]
    assert len({item["join_id"] for item in joins}) == len(joins)
    assert {item["join_level"] for item in joins} == JOIN_LEVELS
    for item in joins:
        assert item["left_source_id"] in source_ids
        assert item["right_source_id"] in source_ids
        assert item["status"] in {"conditional", "blocked", "deferred"}
        assert item["requirements"]
        if (
            item["join_level"] in {"patient", "sample"}
            and item["left_source_id"] != item["right_source_id"]
        ):
            requirements = " ".join([*item["keys"], *item["requirements"]]).lower()
            assert any(term in requirements for term in ("source", "relationship", "identity"))


def test_first_study_is_outcome_blind_and_has_no_observed_results() -> None:
    study = _canonical(STUDY)
    assert study["schema_version"] == "voxelscope/gbm-first-study/v1"
    assert study["status"] == "blocked"
    assert study["observed_results"] == []
    assert study["data_boundary"] == {
        "controlled_data_allowed": False,
        "individual_level_data_committed": False,
        "mri_included": False,
        "research_only": True,
    }
    selection = study["candidate_selection"]
    assert selection["outcome_blind"] is True
    assert selection["source_release"] == "26.09"
    assert selection["excluded_input_rows"] == [
        {
            "ensembl_gene_id": "ENSG00000147889",
            "reason": (
                "Open Targets returned more than one UniProt Swiss-Prot cross-reference, "
                "so the identity rule refused CDKN2A."
            ),
            "symbol": "CDKN2A",
        }
    ]


def test_starting_proteins_are_bounded_unique_and_deterministic() -> None:
    study = _canonical(STUDY)
    candidates = study["candidates"]
    assert 0 < len(candidates) <= 20
    assert len(candidates) == 19
    ensembl = [item["ensembl_gene_id"] for item in candidates]
    uniprot = [item["uniprot_accession"] for item in candidates]
    symbols = [item["symbol"] for item in candidates]
    assert ensembl == sorted(ensembl)
    assert len(ensembl) == len(set(ensembl))
    assert len(uniprot) == len(set(uniprot))
    assert len(symbols) == len(set(symbols))
    assert all(value.startswith("ENSG") and len(value) == 15 for value in ensembl)
    assert "CDKN2A" not in symbols


def test_study_freezes_outcomes_multiplicity_shapes_and_gates() -> None:
    study = _canonical(STUDY)
    assert {item["name"] for item in study["outcomes"]} == REQUIRED_OUTCOMES
    assert {item["label"] for item in study["result_shapes"]} == RESULT_SHAPES
    assert study["multiple_testing"]["method"] == "Benjamini-Hochberg"
    assert study["multiple_testing"]["q_threshold"] == 0.05
    gate_ids = {gate["gate_id"] for gate in study["gates"]}
    assert {
        "source-terms",
        "source-identity",
        "patient-join-coverage",
        "sample-join-coverage",
        "candidate-analyzability",
        "missingness",
        "identifier-integrity",
        "multiplicity",
        "primary-binomial-population",
        "reproducibility",
        "reporting-completeness",
        "privacy-and-copy",
    } == gate_ids
    assert study["study_level_blockers"]


def test_primary_binomial_population_and_states_are_consistent() -> None:
    study = _canonical(STUDY)
    analysis = study["analysis"]
    primary_rule = analysis["primary_binomial_rule"]
    population = "Frozen candidates with BH q <= 0.05 for RNA, protein, and selected phosphosite"
    assert primary_rule["population"] == population
    assert primary_rule["success"].startswith("concordant, defined as three significant layers")
    assert primary_rule["failure"].startswith("discordant, defined as three significant layers")
    assert "secondary_two_layer_discordant" in primary_rule["excluded_states"]

    primary_hypothesis = next(
        item
        for item in study["hypotheses"]
        if item["hypothesis_id"] == "primary-directional-concordance"
    )
    assert primary_hypothesis["population"] == population
    assert primary_hypothesis["success"] == primary_rule["success"]
    assert primary_hypothesis["failure"] == primary_rule["failure"]

    outcomes = {item["name"]: item for item in study["outcomes"]}
    assert outcomes["chain_concordance"]["denominator"] == population
    assert "exactly two" in outcomes["secondary_two_layer_discordance"]["denominator"]
    assert "no inclusion in the primary" in outcomes["secondary_two_layer_discordance"]["statistic"]

    states = analysis["candidate_state_rule"]
    assert states["discordant"].startswith(
        "RNA, protein, and selected phosphosite all have BH q <= 0.05"
    )
    assert states["secondary_two_layer_discordant"].startswith(
        "Exactly two layers have BH q <= 0.05"
    )
    shapes = {item["label"]: item["meaning"] for item in study["result_shapes"]}
    assert "primary exact-binomial success" in shapes["concordant"]
    assert "primary exact-binomial failure" in shapes["discordant"]
    assert (
        "excluded from the primary exact-binomial population"
        in shapes["secondary_two_layer_discordant"]
    )

    population_gate = next(
        item for item in study["gates"] if item["gate_id"] == "primary-binomial-population"
    )
    assert population_gate["measure"] == population
    assert "no p-value" in population_gate["failure_action"]


def test_public_qualification_prose_is_ascii_and_tracks_dependency_states() -> None:
    paths = (REGISTRY, JOIN_MAP, STUDY, PROTOCOL)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "\N{EM DASH}" not in text
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "Research evidence only. Not for diagnosis or treatment decisions." in protocol
    assert "merged at commit `88d074b9116d913ce720bd1a5e72a7e24933110f`" in protocol
    assert "merged at commit `4bd9d60b895981112af267f813e1a31c1aacf4fe`" in protocol
    assert "open and unmerged" not in protocol
    assert "wait for PR #13" not in protocol
