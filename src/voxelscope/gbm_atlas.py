# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic glioblastoma protein evidence cards."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .atomic import path_occupied, rename_no_replace
from .canonical import (
    EvidenceError,
    canonical_json_bytes,
    ensure_no_symlink,
    is_link_like,
    load_json,
    require_sha256,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .records import (
    require_bool,
    require_int,
    require_list,
    require_number,
    require_object,
    require_string,
    strict_fields,
)

SOURCE_MANIFEST_SCHEMA = "voxelscope/gbm-evidence-source-manifest/v1"
IDENTIFIER_MAP_SCHEMA = "voxelscope/gbm-identifier-map/v1"
EVIDENCE_ITEMS_SCHEMA = "voxelscope/gbm-evidence-items/v1"
ATLAS_SCHEMA = "voxelscope/gbm-protein-evidence-atlas/v1"
RECEIPT_SCHEMA = "voxelscope/gbm-evidence-receipt/v1"
TRANSFORMATION_VERSION = "voxelscope/gbm-evidence-transform/v1"

MODALITIES = (
    "mutation",
    "rna_abundance",
    "protein_abundance",
    "phosphorylation",
    "spatial_expression",
    "dependency",
    "normal_brain_expression",
    "target_evidence",
    "alphafold_structure",
)
_EFFECT_DIRECTIONS = frozenset({"up", "down", "neutral"})
_FORBIDDEN_OUTPUT_TERMS = (
    b"clinical",
    b"diagnos",
    b"drug",
    b"patient",
    b"prognos",
    b"recommend",
    b"survival",
    b"therapy",
    b"treatment",
)


def _parse_date(value: Any, field: str) -> date:
    text = require_string(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceError("invalid_date", field) from exc
    if parsed.isoformat() != text:
        raise EvidenceError("invalid_date", field)
    return parsed


@dataclass(frozen=True)
class RawIdentifier:
    namespace: str
    value: str

    def __post_init__(self) -> None:
        if (
            self.namespace != self.namespace.lower()
            or self.namespace.strip() != self.namespace
            or self.value.strip() != self.value
        ):
            raise EvidenceError("invalid_identifier", f"{self.namespace}:{self.value}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawIdentifier:
        value = strict_fields(data, {"namespace", "value"}, "RawIdentifier")
        return cls(
            require_string(value["namespace"], "namespace"),
            require_string(value["value"], "value"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"namespace": self.namespace, "value": self.value}


@dataclass(frozen=True)
class IdentifierProvenance:
    record_id: str
    source_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentifierProvenance:
        value = strict_fields(data, {"record_id", "source_id"}, "IdentifierProvenance")
        return cls(
            require_string(value["record_id"], "record_id"),
            require_string(value["source_id"], "source_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"record_id": self.record_id, "source_id": self.source_id}


@dataclass(frozen=True)
class IdentifierMapping:
    canonical_gene_id: str
    canonical_protein_id: str
    identifiers: tuple[RawIdentifier, ...]
    provenance: IdentifierProvenance

    def __post_init__(self) -> None:
        if not self.canonical_gene_id.startswith("ENSG") or not self.canonical_protein_id:
            raise EvidenceError("invalid_canonical_identifier", self.canonical_gene_id)
        keys = [(item.namespace, item.value) for item in self.identifiers]
        if not keys or len(keys) != len(set(keys)):
            raise EvidenceError("invalid_identifier_mapping", self.canonical_gene_id)
        required = {
            ("ensembl-gene", self.canonical_gene_id),
            ("uniprot", self.canonical_protein_id),
        }
        if not required.issubset(keys):
            raise EvidenceError("incomplete_identifier_mapping", self.canonical_gene_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentifierMapping:
        value = strict_fields(
            data,
            {
                "canonical_gene_id",
                "canonical_protein_id",
                "identifiers",
                "provenance",
            },
            "IdentifierMapping",
        )
        return cls(
            require_string(value["canonical_gene_id"], "canonical_gene_id"),
            require_string(value["canonical_protein_id"], "canonical_protein_id"),
            tuple(
                RawIdentifier.from_dict(require_object(item, "identifier"))
                for item in require_list(value["identifiers"], "identifiers")
            ),
            IdentifierProvenance.from_dict(require_object(value["provenance"], "provenance")),
        )


@dataclass(frozen=True)
class EvidenceItem:
    context: str
    effect_direction: str
    evidence_id: str
    measurement_unit: str
    measurement_value: float
    modality: str
    observed_on: date
    raw_identifier: RawIdentifier
    source_id: str
    synthetic: bool

    def __post_init__(self) -> None:
        if self.effect_direction not in _EFFECT_DIRECTIONS:
            raise EvidenceError("invalid_effect_direction", self.effect_direction)
        if self.modality not in MODALITIES:
            raise EvidenceError("unsupported_evidence_modality", self.modality)
        if not self.synthetic:
            raise EvidenceError("nonsynthetic_evidence_forbidden", self.evidence_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceItem:
        value = strict_fields(
            data,
            {
                "context",
                "effect_direction",
                "evidence_id",
                "measurement",
                "modality",
                "observed_on",
                "raw_identifier",
                "source_id",
                "synthetic",
            },
            "EvidenceItem",
        )
        measurement = strict_fields(
            require_object(value["measurement"], "measurement"),
            {"unit", "value"},
            "Measurement",
        )
        return cls(
            require_string(value["context"], "context"),
            require_string(value["effect_direction"], "effect_direction"),
            require_string(value["evidence_id"], "evidence_id"),
            require_string(measurement["unit"], "measurement unit"),
            require_number(measurement["value"], "measurement value"),
            require_string(value["modality"], "modality"),
            _parse_date(value["observed_on"], "observed_on"),
            RawIdentifier.from_dict(require_object(value["raw_identifier"], "raw_identifier")),
            require_string(value["source_id"], "source_id"),
            require_bool(value["synthetic"], "synthetic"),
        )

    def to_card_dict(self, mapping: IdentifierMapping) -> dict[str, Any]:
        return {
            "context": self.context,
            "effect_direction": self.effect_direction,
            "evidence_id": self.evidence_id,
            "identifier_provenance": mapping.provenance.to_dict(),
            "measurement": {
                "unit": self.measurement_unit,
                "value": self.measurement_value,
            },
            "modality": self.modality,
            "normalized_from": self.raw_identifier.to_dict(),
            "observed_on": self.observed_on.isoformat(),
            "source_id": self.source_id,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class SourceArtifact:
    artifact_path: str
    released_on: date
    sha256: str
    source_id: str
    source_uri: str
    stale_after_days: int
    synthetic: bool

    def __post_init__(self) -> None:
        safe_relative_path(self.artifact_path)
        require_sha256(self.sha256)
        if self.stale_after_days <= 0:
            raise EvidenceError("invalid_freshness_window", self.source_id)
        if not self.synthetic:
            raise EvidenceError("nonsynthetic_source_forbidden", self.source_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceArtifact:
        value = strict_fields(
            data,
            {
                "artifact_path",
                "released_on",
                "sha256",
                "source_id",
                "source_uri",
                "stale_after_days",
                "synthetic",
            },
            "SourceArtifact",
        )
        return cls(
            require_string(value["artifact_path"], "artifact_path"),
            _parse_date(value["released_on"], "released_on"),
            require_string(value["sha256"], "sha256"),
            require_string(value["source_id"], "source_id"),
            require_string(value["source_uri"], "source_uri"),
            require_int(value["stale_after_days"], "stale_after_days"),
            require_bool(value["synthetic"], "synthetic"),
        )


@dataclass(frozen=True)
class SourceManifest:
    atlas_id: str
    freshness_assessed_on: date
    sources: tuple[SourceArtifact, ...]
    synthetic: bool

    def __post_init__(self) -> None:
        source_ids = [item.source_id for item in self.sources]
        paths = [item.artifact_path for item in self.sources]
        if not self.synthetic:
            raise EvidenceError("nonsynthetic_manifest_forbidden", self.atlas_id)
        if len(self.sources) != 2 or len(source_ids) != len(set(source_ids)):
            raise EvidenceError("invalid_source_manifest", "source IDs must be unique")
        if len(paths) != len(set(paths)):
            raise EvidenceError("invalid_source_manifest", "artifact paths must be unique")
        if any(item.released_on > self.freshness_assessed_on for item in self.sources):
            raise EvidenceError("invalid_source_freshness", "source release is in the future")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceManifest:
        value = strict_fields(
            data,
            {
                "atlas_id",
                "freshness_assessed_on",
                "schema_version",
                "sources",
                "synthetic",
            },
            "SourceManifest",
        )
        schema = require_string(value["schema_version"], "schema_version")
        if schema != SOURCE_MANIFEST_SCHEMA:
            raise EvidenceError("unsupported_schema", schema)
        return cls(
            require_string(value["atlas_id"], "atlas_id"),
            _parse_date(value["freshness_assessed_on"], "freshness_assessed_on"),
            tuple(
                SourceArtifact.from_dict(require_object(item, "source"))
                for item in require_list(value["sources"], "sources")
            ),
            require_bool(value["synthetic"], "synthetic"),
        )


@dataclass(frozen=True)
class AtlasBuildResult:
    atlas_sha256: str
    receipt_sha256: str


def _load_manifest(path: Path) -> SourceManifest:
    if is_link_like(path) or not path.is_file():
        raise EvidenceError("unsafe_manifest", "manifest must be a regular file")
    return SourceManifest.from_dict(require_object(load_json(path), "source manifest"))


def _load_source_records(
    manifest_path: Path,
    manifest: SourceManifest,
) -> tuple[list[IdentifierMapping], list[EvidenceItem], list[dict[str, str]]]:
    root = manifest_path.parent
    loaded: dict[str, dict[str, Any]] = {}
    digests = [{"path": manifest_path.name, "sha256": sha256_file(manifest_path)}]
    source_ids = {item.source_id for item in manifest.sources}
    for source in manifest.sources:
        path = ensure_no_symlink(root, safe_relative_path(source.artifact_path))
        if not path.is_file() or sha256_file(path) != source.sha256:
            raise EvidenceError("source_digest_mismatch", source.artifact_path)
        record = require_object(load_json(path), source.artifact_path)
        schema = require_string(record.get("schema_version"), "schema_version")
        if schema in loaded:
            raise EvidenceError("duplicate_source_schema", schema)
        loaded[schema] = record
        digests.append({"path": source.artifact_path, "sha256": source.sha256})
    if set(loaded) != {IDENTIFIER_MAP_SCHEMA, EVIDENCE_ITEMS_SCHEMA}:
        raise EvidenceError("source_schema_set_mismatch", "required source schemas differ")

    identifier_record = strict_fields(
        loaded[IDENTIFIER_MAP_SCHEMA],
        {"mappings", "schema_version", "synthetic"},
        "IdentifierMap",
    )
    if (
        identifier_record["schema_version"] != IDENTIFIER_MAP_SCHEMA
        or require_bool(identifier_record["synthetic"], "synthetic") is not True
    ):
        raise EvidenceError("invalid_identifier_map", "identifier map identity differs")
    mappings = [
        IdentifierMapping.from_dict(require_object(item, "mapping"))
        for item in require_list(identifier_record["mappings"], "mappings")
    ]
    if not mappings:
        raise EvidenceError("invalid_identifier_map", "at least one mapping is required")
    canonical_pairs = [(item.canonical_gene_id, item.canonical_protein_id) for item in mappings]
    if len(canonical_pairs) != len(set(canonical_pairs)):
        raise EvidenceError("invalid_identifier_map", "canonical pairs must be unique")
    if any(item.provenance.source_id not in source_ids for item in mappings):
        raise EvidenceError("unknown_provenance_source", "identifier provenance is unbound")

    evidence_record = strict_fields(
        loaded[EVIDENCE_ITEMS_SCHEMA],
        {"items", "schema_version", "synthetic"},
        "EvidenceItems",
    )
    if (
        evidence_record["schema_version"] != EVIDENCE_ITEMS_SCHEMA
        or require_bool(evidence_record["synthetic"], "synthetic") is not True
    ):
        raise EvidenceError("invalid_evidence_items", "evidence identity differs")
    items = [
        EvidenceItem.from_dict(require_object(item, "evidence item"))
        for item in require_list(evidence_record["items"], "items")
    ]
    evidence_ids = [item.evidence_id for item in items]
    if not items or len(evidence_ids) != len(set(evidence_ids)):
        raise EvidenceError("invalid_evidence_items", "evidence IDs must be unique")
    if any(item.source_id not in source_ids for item in items):
        raise EvidenceError("unknown_evidence_source", "evidence source is unbound")
    if any(item.observed_on > manifest.freshness_assessed_on for item in items):
        raise EvidenceError("invalid_evidence_date", "evidence observation is in the future")
    return mappings, items, sorted(digests, key=lambda item: item["path"])


def _identifier_index(
    mappings: list[IdentifierMapping],
) -> dict[tuple[str, str], IdentifierMapping]:
    candidates: dict[tuple[str, str], list[IdentifierMapping]] = {}
    for mapping in mappings:
        for identifier in mapping.identifiers:
            candidates.setdefault((identifier.namespace, identifier.value), []).append(mapping)
    ambiguous = sorted(key for key, values in candidates.items() if len(values) != 1)
    if ambiguous:
        namespace, value = ambiguous[0]
        raise EvidenceError("ambiguous_identifier_mapping", f"{namespace}:{value}")
    return {key: values[0] for key, values in candidates.items()}


def _directional_assessment(items: list[EvidenceItem]) -> dict[str, Any]:
    directional = {
        direction: sorted(item.evidence_id for item in items if item.effect_direction == direction)
        for direction in ("down", "up")
    }
    present = [direction for direction, evidence_ids in directional.items() if evidence_ids]
    groups = [
        {"direction": direction, "evidence_ids": evidence_ids}
        for direction, evidence_ids in directional.items()
        if len(evidence_ids) >= 2
    ]
    if len(present) > 1:
        state = "disagreement"
    elif sum(len(evidence_ids) for evidence_ids in directional.values()) >= 2:
        state = "agreement"
    else:
        state = "insufficient"
    return {
        "agreement_groups": groups,
        "conflicting_directions": present if state == "disagreement" else [],
        "state": state,
    }


def _warning(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def assemble_gbm_evidence_atlas(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    mappings, items, source_digests = _load_source_records(manifest_path, manifest)
    index = _identifier_index(mappings)
    joined: dict[tuple[str, str], list[EvidenceItem]] = {
        (mapping.canonical_gene_id, mapping.canonical_protein_id): [] for mapping in mappings
    }
    exclusions: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    for item in sorted(items, key=lambda item: item.evidence_id):
        mapping = index.get((item.raw_identifier.namespace, item.raw_identifier.value))
        if mapping is None:
            exclusions.append(
                {
                    "evidence_id": item.evidence_id,
                    "raw_identifier": item.raw_identifier.to_dict(),
                    "reason": "unsupported_identifier_join",
                }
            )
            warnings.append(
                _warning(
                    "unsupported_join",
                    f"{item.evidence_id} has no unique identifier mapping.",
                )
            )
            continue
        joined[(mapping.canonical_gene_id, mapping.canonical_protein_id)].append(item)

    cards: list[dict[str, Any]] = []
    for mapping in sorted(
        mappings,
        key=lambda item: (item.canonical_protein_id, item.canonical_gene_id),
    ):
        card_items = sorted(
            joined[(mapping.canonical_gene_id, mapping.canonical_protein_id)],
            key=lambda item: (MODALITIES.index(item.modality), item.evidence_id),
        )
        present = tuple(
            modality
            for modality in MODALITIES
            if any(item.modality == modality for item in card_items)
        )
        missing = [modality for modality in MODALITIES if modality not in present]
        assessment = _directional_assessment(card_items)
        cards.append(
            {
                "canonical_gene_id": mapping.canonical_gene_id,
                "canonical_protein_id": mapping.canonical_protein_id,
                "directional_assessment": assessment,
                "evidence": [item.to_card_dict(mapping) for item in card_items],
                "missing_modalities": missing,
                "modalities_present": list(present),
            }
        )
        if missing:
            warnings.append(
                _warning(
                    "missing_evidence",
                    f"{mapping.canonical_protein_id} is missing {', '.join(missing)}.",
                )
            )
        if assessment["state"] == "disagreement":
            warnings.append(
                _warning(
                    "contradictory_evidence",
                    f"{mapping.canonical_protein_id} has both up and down directional evidence.",
                )
            )

    freshness: list[dict[str, Any]] = []
    for source in sorted(manifest.sources, key=lambda item: item.source_id):
        age_days = (manifest.freshness_assessed_on - source.released_on).days
        status = "current" if age_days <= source.stale_after_days else "stale"
        freshness.append(
            {
                "age_days": age_days,
                "assessed_on": manifest.freshness_assessed_on.isoformat(),
                "released_on": source.released_on.isoformat(),
                "source_id": source.source_id,
                "status": status,
            }
        )
        if status == "stale":
            warnings.append(
                _warning(
                    "stale_source",
                    f"{source.source_id} is {age_days} days old at the manifest assessment date.",
                )
            )

    warnings.sort(key=lambda item: (item["code"], item["detail"]))
    exclusions.sort(key=lambda item: item["evidence_id"])
    atlas = {
        "atlas_id": manifest.atlas_id,
        "cards": cards,
        "ranking_performed": False,
        "schema_version": ATLAS_SCHEMA,
        "source_freshness": freshness,
        "synthetic_only": True,
        "unsupported_joins": exclusions,
    }
    output_digest = sha256_bytes(canonical_json_bytes(atlas))
    receipt = {
        "atlas_id": manifest.atlas_id,
        "exclusions": exclusions,
        "output_digest": output_digest,
        "schema_version": RECEIPT_SCHEMA,
        "source_digests": source_digests,
        "transformation_version": TRANSFORMATION_VERSION,
        "warnings": warnings,
    }
    _assert_safe_output(atlas, receipt)
    return atlas, receipt


def _assert_safe_output(atlas: dict[str, Any], receipt: dict[str, Any]) -> None:
    output = canonical_json_bytes({"atlas": atlas, "receipt": receipt}).lower()
    for term in _FORBIDDEN_OUTPUT_TERMS:
        if term in output:
            raise EvidenceError("prohibited_output_language", term.decode("ascii"))


def build_gbm_evidence_atlas(manifest_path: Path, output: Path) -> AtlasBuildResult:
    if path_occupied(output):
        raise EvidenceError("output_exists", str(output))
    atlas, receipt = assemble_gbm_evidence_atlas(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(stage / "atlas.json", atlas)
        write_json(stage / "receipt.json", receipt)
        if sha256_file(stage / "atlas.json") != receipt["output_digest"]:
            raise EvidenceError("output_digest_mismatch", "atlas digest differs")
        result = AtlasBuildResult(
            atlas_sha256=receipt["output_digest"],
            receipt_sha256=sha256_file(stage / "receipt.json"),
        )
        rename_no_replace(stage, output)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)
