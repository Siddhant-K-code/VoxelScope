# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

from voxelscope.canonical import (
    canonical_json_bytes,
    load_json,
    require_sha256,
    sha256_bytes,
    sha256_file,
)

ROOT = Path(__file__).parents[1]
FREEZE = ROOT / "research" / "gbm-evidence-atlas-source-freeze-v1"
SOURCE_FREEZE = FREEZE / "metadata-source-freeze.json"
GDC_MANIFEST = FREEZE / "gdc-open-file-manifest.json"
ACCESS_DECISIONS = FREEZE / "access-decisions.json"
CROSSWALK_DECISION = FREEZE / "crosswalk-decision.json"
RECEIPT = FREEZE / "receipt.json"
MILESTONE = ROOT / "docs" / "research" / "glioblastoma-evidence-atlas-source-freeze-v1.md"

EXPECTED_PDC = {
    "PDC000204": {
        "aliquot_count": 111,
        "analytical_fraction": "Proteome",
        "case_count": 111,
        "study_name": "CPTAC GBM Discovery Study - Proteome",
        "version_uuid": "cfe9f4a2-1797-11ea-9bfa-0a42f3c845fe",
    },
    "PDC000205": {
        "aliquot_count": 111,
        "analytical_fraction": "Phosphoproteome",
        "case_count": 111,
        "study_name": "CPTAC GBM Discovery Study - Phosphoproteome",
        "version_uuid": "efc3143a-1797-11ea-9bfa-0a42f3c845fe",
    },
    "PDC000446": {
        "aliquot_count": 150,
        "analytical_fraction": "Proteome",
        "case_count": 118,
        "study_name": "CPTAC GBM Confirmatory Study - Proteome",
        "version_uuid": "b25c2cea-4a49-4bc3-9f87-3cf5ee026865",
    },
    "PDC000448": {
        "aliquot_count": 150,
        "analytical_fraction": "Phosphoproteome",
        "case_count": 118,
        "study_name": "CPTAC GBM Confirmatory Study - Phosphoproteome",
        "version_uuid": "e1f7dcb3-db7f-4f04-bc0f-b150af3888b9",
    },
    "PDC000514": {
        "aliquot_count": 216,
        "analytical_fraction": "Proteome",
        "case_count": 111,
        "study_name": "KNCC Glioblastoma Evolution - Proteome",
        "version_uuid": "524d5116-b6de-4e36-892a-e35dba7d0170",
    },
    "PDC000515": {
        "aliquot_count": 180,
        "analytical_fraction": "Phosphoproteome",
        "case_count": 91,
        "study_name": "KNCC Glioblastoma Evolution - Phosphoproteome",
        "version_uuid": "e5e0dd84-f982-46e3-b78a-5cb19eef31a8",
    },
}


def _canonical(path: Path) -> dict[str, Any]:
    value = load_json(path)
    assert isinstance(value, dict)
    assert path.read_bytes() == canonical_json_bytes(value)
    return value


def _assert_keys(value: dict[str, Any], expected: set[str]) -> None:
    assert set(value) == expected


def _assert_timestamp(value: str) -> None:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert value.endswith("Z")


def _assert_official_https(url: str) -> None:
    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.hostname
    assert parsed.username is None
    assert parsed.password is None
    assert not any(name in parsed.hostname for name in ("bing.", "google."))


def test_metadata_source_freeze_pins_required_pdc_versions_and_gdc_release() -> None:
    record = _canonical(SOURCE_FREEZE)
    _assert_keys(
        record,
        {
            "adapter_gate",
            "controlled_acquisition_performed",
            "credentials_used",
            "evidence_urls",
            "freeze_id",
            "gdc",
            "metadata_only",
            "patient_level_rows_acquired",
            "pdc_studies",
            "retrieved_at",
            "schema_version",
        },
    )
    assert record["schema_version"] == "voxelscope/gbm-metadata-source-freeze/v1"
    assert record["adapter_gate"] == "no-go"
    assert record["metadata_only"] is True
    assert record["controlled_acquisition_performed"] is False
    assert record["credentials_used"] is False
    assert record["patient_level_rows_acquired"] is False
    _assert_timestamp(record["retrieved_at"])
    for url in record["evidence_urls"]:
        _assert_official_https(url)

    studies = record["pdc_studies"]
    assert [item["stable_study_id"] for item in studies] == list(EXPECTED_PDC)
    version_uuids = [item["version_uuid"] for item in studies]
    assert len(version_uuids) == len(set(version_uuids)) == 6
    for item in studies:
        _assert_keys(
            item,
            {
                "access_decision_id",
                "aliquot_count",
                "analytical_fraction",
                "case_count",
                "experiment_type",
                "is_latest_version",
                "official_locator",
                "program_name",
                "project_name",
                "retrieved_at",
                "stable_study_id",
                "study_name",
                "version_number",
                "version_timestamp",
                "version_timestamp_status",
                "version_uuid",
            },
        )
        expected = EXPECTED_PDC[item["stable_study_id"]]
        assert item["version_uuid"] == expected["version_uuid"]
        assert item["study_name"] == expected["study_name"]
        assert item["analytical_fraction"] == expected["analytical_fraction"]
        assert item["case_count"] == expected["case_count"]
        assert item["aliquot_count"] == expected["aliquot_count"]
        assert item["version_uuid"] != item["stable_study_id"]
        assert str(UUID(item["version_uuid"])) == item["version_uuid"]
        assert item["version_number"] == "1"
        assert item["is_latest_version"] is True
        assert item["experiment_type"] == "TMT11"
        assert item["version_timestamp"] is None
        assert item["version_timestamp_status"].startswith("Unavailable.")
        assert item["official_locator"].endswith(item["stable_study_id"])
        _assert_official_https(item["official_locator"])
        _assert_timestamp(item["retrieved_at"])

    gdc = record["gdc"]
    _assert_keys(
        gdc,
        {
            "access_decision_id",
            "candidate_open_file_count",
            "case_counts_comparable",
            "case_counts_equivalent",
            "clinical_and_biospecimen_rows_acquired",
            "discovery_cohort_membership",
            "gliomas_case_count",
            "input_selection_status",
            "inventory_role",
            "manifest_path",
            "manifest_sha256",
            "pdc_discovery_catalog_case_count",
            "project_id",
            "release_date",
            "release_id",
            "retrieved_at",
        },
    )
    assert gdc["release_id"] == "Data Release 46.0"
    assert gdc["release_date"] == "2026-08-10"
    assert gdc["project_id"] == "CPTAC-3"
    assert gdc["gliomas_case_count"] == 211
    assert gdc["pdc_discovery_catalog_case_count"] == 111
    assert gdc["candidate_open_file_count"] == 498
    assert gdc["case_counts_comparable"] is False
    assert gdc["case_counts_equivalent"] is False
    assert gdc["discovery_cohort_membership"] == "unresolved"
    assert gdc["input_selection_status"] == ("blocked-pending-crosswalk-and-one-to-one-joins")
    assert gdc["inventory_role"] == "candidate-metadata-superset"
    assert gdc["clinical_and_biospecimen_rows_acquired"] is False
    assert gdc["manifest_path"] == GDC_MANIFEST.name
    assert gdc["manifest_sha256"] == sha256_file(GDC_MANIFEST)
    _assert_timestamp(gdc["retrieved_at"])


def test_gdc_manifest_is_candidate_open_workflow_metadata_superset() -> None:
    record = _canonical(GDC_MANIFEST)
    _assert_keys(
        record,
        {
            "evidence_urls",
            "files",
            "manifest_id",
            "metadata_boundary",
            "project",
            "query",
            "release",
            "retrieved_at",
            "schema_version",
            "scope",
        },
    )
    assert record["schema_version"] == "voxelscope/gbm-gdc-open-file-manifest/v1"
    assert record["manifest_id"] == (
        "gdc-cptac-3-gliomas-open-workflow-candidate-inventory-dr46-v1"
    )
    _assert_timestamp(record["retrieved_at"])
    for url in record["evidence_urls"]:
        _assert_official_https(url)

    assert record["metadata_boundary"] == {
        "case_fields_requested": False,
        "candidate_metadata_superset": True,
        "case_count_equivalence_authorized": False,
        "clinical_or_biospecimen_rows_acquired": False,
        "cohort_membership_resolved": False,
        "discovery_cohort_input_selection_authorized": False,
        "file_bytes_acquired": False,
        "file_metadata_only": True,
        "patient_or_sample_fields_requested": False,
    }
    project = record["project"]
    _assert_keys(
        project,
        {
            "dbgap_accession",
            "name",
            "project_id",
            "released",
            "state",
            "total_case_count",
            "total_file_count",
            "total_file_size",
        },
    )
    assert project["project_id"] == "CPTAC-3"
    assert project["dbgap_accession"] == "phs001287"
    assert project["total_case_count"] == 1866
    assert project["released"] is True

    query = record["query"]
    _assert_keys(query, {"endpoint", "fields", "filters", "size", "sort"})
    assert query["endpoint"] == "https://api.gdc.cancer.gov/files"
    assert query["sort"] == "file_id:asc"
    assert query["size"] == 1000
    assert set(query["fields"]) == {
        "access",
        "analysis.workflow_type",
        "data_format",
        "data_type",
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
    }
    assert not any(
        identifier in field
        for field in query["fields"]
        for identifier in ("case", "patient", "sample", "aliquot", "submitter")
    )

    files = record["files"]
    assert len(files) == 498
    assert [item["file_id"] for item in files] == sorted(item["file_id"] for item in files)
    assert len({item["file_id"] for item in files}) == 498
    assert len({item["file_name"] for item in files}) == 498
    for item in files:
        _assert_keys(
            item,
            {
                "access",
                "data_format",
                "data_type",
                "file_id",
                "file_name",
                "file_size",
                "md5",
                "workflow_type",
            },
        )
        assert item["access"] == "open"
        assert item["file_size"] > 0
        assert str(UUID(item["file_id"])) == item["file_id"]
        assert re.fullmatch(r"[0-9a-f]{32}", item["md5"])

    counts = Counter((item["data_type"], item["workflow_type"]) for item in files)
    assert counts == {
        ("Gene Expression Quantification", "STAR - Counts"): 229,
        (
            "Masked Somatic Mutation",
            "Aliquot Ensemble Somatic Variant Merging and Masking",
        ): 269,
    }
    sizes = Counter()
    for item in files:
        sizes[item["data_type"]] += item["file_size"]
    assert sizes == {
        "Gene Expression Quantification": 970115569,
        "Masked Somatic Mutation": 10680012,
    }

    release = record["release"]
    assert release["release_id"] == "Data Release 46.0"
    assert release["release_date"] == "2026-08-10"
    assert release["api_commit"] == "8f7c2a51ab0084b216ad1b62a3fae8b945439c53"
    assert len(release["release_manifest_locators"]) == 2
    for locator in release["release_manifest_locators"]:
        _assert_keys(locator, {"file_id", "file_name", "locator", "metadata_status"})
        assert str(UUID(locator["file_id"])) == locator["file_id"]
        _assert_official_https(locator["locator"])
        assert locator["metadata_status"].endswith("content was not downloaded.")

    scope = record["scope"]
    assert scope["returned_file_count"] == 498
    assert scope["access"] == "open"
    assert scope["disease_type"] == "Gliomas"
    assert scope["case_count_query"]["size"] == 0
    assert scope["case_count_query"]["rows_returned"] == 0
    assert scope["case_count_query"]["total_case_count"] == 211
    assert scope["candidate_inventory_only"] is True
    assert scope["inventory_role"] == ("candidate-gdc-metadata-superset-for-eligible-workflows")
    assert scope["discovery_cohort_membership"] == (
        "unresolved-pending-official-target-study-crosswalk"
    )
    assert scope["selection_status"] == "blocked"
    assert "one-to-one join checks" in scope["selection_gate"]
    assert scope["cohort_case_counts"] == {
        "comparison_authorized": False,
        "equivalent": False,
        "gdc_cptac_3_gliomas": 211,
        "pdc_discovery_catalog": 111,
    }


def test_access_decisions_refuse_pdc_and_qualify_only_gdc_metadata() -> None:
    record = _canonical(ACCESS_DECISIONS)
    _assert_keys(
        record,
        {
            "adapter_gate",
            "controlled_acquisition_performed",
            "credentials_used",
            "decisions",
            "evidence_urls",
            "freeze_id",
            "metadata_only",
            "patient_level_rows_acquired",
            "refusals",
            "retrieved_at",
            "schema_version",
        },
    )
    assert record["schema_version"] == "voxelscope/gbm-access-decisions/v1"
    assert record["adapter_gate"] == "no-go"
    assert record["controlled_acquisition_performed"] is False
    assert record["credentials_used"] is False
    assert record["patient_level_rows_acquired"] is False
    assert record["metadata_only"] is True
    assert len(record["refusals"]) == 3
    _assert_timestamp(record["retrieved_at"])

    decisions = {item["decision_id"]: item for item in record["decisions"]}
    assert set(decisions) == {
        *(f"{study_id.lower()}-access" for study_id in EXPECTED_PDC),
        "gdc-cptac-3-dr46-candidate-metadata-openness",
    }
    for study_id in EXPECTED_PDC:
        item = decisions[f"{study_id.lower()}-access"]
        _assert_keys(
            item,
            {
                "accept_dua_required",
                "credential_or_application_status",
                "decision",
                "decision_id",
                "evidence_urls",
                "file_byte_access",
                "file_bytes_acquired",
                "metadata_access",
                "next_gate",
                "probe",
                "provider",
                "scope",
                "terms",
                "terms_status",
            },
        )
        assert item["scope"] == study_id
        assert item["decision"] == "refused"
        assert item["metadata_access"] == "open-without-credential"
        assert item["file_byte_access"] == "bounded-unverified"
        assert item["file_bytes_acquired"] is False
        assert item["accept_dua_required"] is True
        assert item["terms_status"] == "refused-current-live-text-not-captured"
        assert item["probe"]["access"] == "Open"
        assert item["probe"]["downloadable"] == "Yes"
        assert str(UUID(item["probe"]["file_id"])) == item["probe"]["file_id"]
        assert "No file byte request" in item["probe"]["method"]
        assert item["terms"]["live_route_status"].startswith("HTTP 200")
        for url in item["evidence_urls"]:
            _assert_official_https(url)

    gdc = decisions["gdc-cptac-3-dr46-candidate-metadata-openness"]
    _assert_keys(
        gdc,
        {
            "accept_dua_required",
            "cohort_inclusion",
            "credential_or_application_status",
            "decision",
            "decision_id",
            "evidence_urls",
            "file_byte_access",
            "file_bytes_acquired",
            "input_selection",
            "metadata_access",
            "next_gate",
            "probe",
            "provider",
            "scope",
            "terms",
            "terms_status",
        },
    )
    assert gdc["decision"] == "metadata-openness-qualified"
    assert gdc["terms_status"] == "verified"
    assert gdc["metadata_access"] == "open-without-credential"
    assert gdc["file_byte_access"] == "not-exercised"
    assert gdc["file_bytes_acquired"] is False
    assert gdc["accept_dua_required"] is False
    assert gdc["cohort_inclusion"] == "not-qualified"
    assert gdc["input_selection"] == ("blocked-pending-crosswalk-and-one-to-one-joins")
    assert gdc["probe"]["file_count"] == 498
    assert gdc["probe"]["inventory_role"] == "candidate-metadata-superset"
    assert gdc["probe"]["source_hash_field"] == "md5sum"
    assert "cohort inclusion is unresolved" in gdc["scope"].lower()


def test_crosswalk_decision_blocks_execution_at_every_required_level() -> None:
    record = _canonical(CROSSWALK_DECISION)
    _assert_keys(
        record,
        {
            "adapter_gate",
            "decision",
            "decision_id",
            "evidence_urls",
            "identifier_levels",
            "matching_c3_labels_authorize_join",
            "metadata_only",
            "next_gate",
            "patient_or_sample_rows_acquired",
            "refusal_reasons",
            "retrieved_at",
            "schema_version",
            "target_study_ids",
        },
    )
    assert record["schema_version"] == "voxelscope/gbm-crosswalk-decision/v1"
    assert record["decision"] == "refused"
    assert record["adapter_gate"] == "no-go"
    assert record["matching_c3_labels_authorize_join"] is False
    assert record["patient_or_sample_rows_acquired"] is False
    assert record["target_study_ids"] == list(EXPECTED_PDC)
    assert len(record["refusal_reasons"]) == 3
    _assert_timestamp(record["retrieved_at"])

    levels = {item["level"]: item for item in record["identifier_levels"]}
    assert set(levels) == {"patient", "sample", "aliquot"}
    for item in levels.values():
        _assert_keys(
            item,
            {
                "execution_status",
                "gdc_identifier",
                "level",
                "pdc_identifier",
                "source_support",
            },
        )
    assert levels["patient"]["execution_status"] == "blocked"
    assert "externalReferences" in levels["patient"]["source_support"]
    assert levels["sample"]["execution_status"] == "blocked"
    assert "gdc_sample_id" in levels["sample"]["gdc_identifier"]
    assert levels["aliquot"]["execution_status"] == "unsupported"
    assert levels["aliquot"]["gdc_identifier"] is None
    assert "no GDC aliquot identifier" in " ".join(record["refusal_reasons"])


def test_freeze_receipt_closes_digests_and_replays_byte_identically() -> None:
    record = _canonical(RECEIPT)
    _assert_keys(
        record,
        {
            "adapter_gate",
            "artifact_digests",
            "controlled_acquisition_performed",
            "credentials_used",
            "freeze_id",
            "generated_at",
            "metadata_only",
            "output_digest",
            "patient_level_rows_acquired",
            "schema_version",
            "terminal_state",
            "transformation_version",
            "warnings",
        },
    )
    assert record["schema_version"] == ("voxelscope/gbm-metadata-source-freeze-receipt/v1")
    assert record["transformation_version"] == ("voxelscope/gbm-metadata-source-freeze/v1")
    assert record["terminal_state"] == "metadata-freeze-complete-adapter-refused"
    assert record["adapter_gate"] == "no-go"
    assert record["metadata_only"] is True
    assert record["controlled_acquisition_performed"] is False
    assert record["credentials_used"] is False
    assert record["patient_level_rows_acquired"] is False
    assert any(
        warning.startswith("The 498 GDC file records are a candidate metadata superset.")
        for warning in record["warnings"]
    )
    _assert_timestamp(record["generated_at"])

    expected_paths = [
        "access-decisions.json",
        "crosswalk-decision.json",
        "gdc-open-file-manifest.json",
        "metadata-source-freeze.json",
    ]
    assert [item["path"] for item in record["artifact_digests"]] == expected_paths
    for item in record["artifact_digests"]:
        _assert_keys(item, {"path", "sha256"})
        require_sha256(item["sha256"])
        assert item["sha256"] == sha256_file(FREEZE / item["path"])
    require_sha256(record["output_digest"])
    assert record["output_digest"] == sha256_bytes(canonical_json_bytes(record["artifact_digests"]))

    for path in (
        SOURCE_FREEZE,
        GDC_MANIFEST,
        ACCESS_DECISIONS,
        CROSSWALK_DECISION,
        RECEIPT,
    ):
        value = load_json(path)
        assert path.read_bytes() == canonical_json_bytes(value)


def test_public_freeze_has_no_sensitive_acquisition_or_disallowed_copy() -> None:
    paths = (
        SOURCE_FREEZE,
        GDC_MANIFEST,
        ACCESS_DECISIONS,
        CROSSWALK_DECISION,
        RECEIPT,
        MILESTONE,
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "\N{EM DASH}" not in combined
    assert "/Users/" not in combined
    assert "X-Auth-Token" not in combined
    assert "Authorization: Bearer" not in combined
    assert "ghp_" not in combined
    assert "therapeutic target" not in combined.lower()

    milestone = MILESTONE.read_text(encoding="utf-8")
    assert "Research evidence only. Not for diagnosis or treatment decisions." in milestone
    assert "real-data adapter gate remains **NO-GO**" in milestone
    assert "No patient, sample, or aliquot join may execute" in milestone
    assert "This inventory is not a selected analysis input" in milestone
    assert "must not be compared with or treated as equivalent" in milestone
    assert "This qualifies metadata openness only." in milestone
