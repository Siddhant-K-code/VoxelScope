# SPDX-License-Identifier: Apache-2.0
"""Strict records for public sources and private artifact custody."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from .canonical import EvidenceError, require_sha256, safe_relative_path
from .records import (
    require_int,
    require_list,
    require_object,
    require_string,
    strict_fields,
)

VerificationStatus = Literal["verified", "partial", "blocked"]
AcquisitionStatus = Literal["ready", "blocked"]
DigestAlgorithm = Literal["md5", "sha1", "sha256", "etag", "git-blob-sha1"]


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(require_string(item, f"{name} item") for item in require_list(value, name))


def require_https_url(value: Any, name: str, *, origin_only: bool = False) -> str:
    url = require_string(value, name)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise EvidenceError("unsafe_url", f"{name} must be a credential-free HTTPS URL")
    if origin_only and (parsed.path not in {"", "/"} or parsed.query):
        raise EvidenceError("unsafe_url", f"{name} must be an HTTPS origin")
    return url.rstrip("/") if origin_only else url


def _verification_status(value: Any) -> VerificationStatus:
    status = require_string(value, "verification_status")
    if status not in {"verified", "partial", "blocked"}:
        raise EvidenceError("invalid_verification_status", status)
    return cast(VerificationStatus, status)


def _acquisition_status(value: Any) -> AcquisitionStatus:
    status = require_string(value, "status")
    if status not in {"ready", "blocked"}:
        raise EvidenceError("invalid_acquisition_status", status)
    return cast(AcquisitionStatus, status)


@dataclass(frozen=True)
class DigestIdentity:
    algorithm: DigestAlgorithm
    value: str

    def __post_init__(self) -> None:
        lengths = {
            "md5": 32,
            "sha1": 40,
            "sha256": 64,
            "etag": None,
            "git-blob-sha1": 40,
        }
        expected = lengths.get(self.algorithm)
        if self.algorithm not in lengths:
            raise EvidenceError("unsupported_digest", self.algorithm)
        if not self.value or any(character not in "0123456789abcdef-" for character in self.value):
            raise EvidenceError("invalid_digest", self.algorithm)
        if expected is not None and len(self.value) != expected:
            raise EvidenceError("invalid_digest", self.algorithm)
        if self.algorithm == "sha256":
            require_sha256(self.value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DigestIdentity:
        data = strict_fields(data, {"algorithm", "value"}, "DigestIdentity")
        algorithm = require_string(data["algorithm"], "algorithm")
        if algorithm not in {"md5", "sha1", "sha256", "etag", "git-blob-sha1"}:
            raise EvidenceError("unsupported_digest", algorithm)
        return cls(cast(DigestAlgorithm, algorithm), require_string(data["value"], "value"))


def _digests(value: Any, name: str) -> tuple[DigestIdentity, ...]:
    digests = tuple(
        DigestIdentity.from_dict(require_object(item, f"{name} item"))
        for item in require_list(value, name)
    )
    algorithms = [digest.algorithm for digest in digests]
    if len(algorithms) != len(set(algorithms)):
        raise EvidenceError("duplicate_digest", name)
    return digests


@dataclass(frozen=True)
class SourceArtifact:
    path: str
    identity_scope: str
    size_bytes: int | None
    hashes: tuple[DigestIdentity, ...]
    verification_status: VerificationStatus
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        if self.size_bytes is not None and self.size_bytes < 0:
            raise EvidenceError("invalid_size", self.path)
        if self.verification_status not in {"verified", "partial", "blocked"}:
            raise EvidenceError("invalid_verification_status", self.verification_status)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceArtifact:
        data = strict_fields(
            data,
            {
                "assumptions",
                "evidence",
                "hashes",
                "identity_scope",
                "path",
                "size_bytes",
                "verification_status",
            },
            "SourceArtifact",
        )
        size_value = data["size_bytes"]
        size = require_int(size_value, "size_bytes") if size_value is not None else None
        return cls(
            path=require_string(data["path"], "path"),
            identity_scope=require_string(data["identity_scope"], "identity_scope"),
            size_bytes=size,
            hashes=_digests(data["hashes"], "hashes"),
            verification_status=_verification_status(data["verification_status"]),
            evidence=_strings(data["evidence"], "evidence"),
            assumptions=_strings(data["assumptions"], "assumptions"),
        )


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_kind: str
    canonical_url: str
    immutable_id: str | None
    expected_license: str
    retrieved_on: str
    verification_status: VerificationStatus
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    artifacts: tuple[SourceArtifact, ...]

    def __post_init__(self) -> None:
        require_https_url(self.canonical_url, "canonical_url")
        if self.verification_status not in {"verified", "partial", "blocked"}:
            raise EvidenceError("invalid_verification_status", self.verification_status)
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise EvidenceError("duplicate_source_artifact", self.source_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRecord:
        data = strict_fields(
            data,
            {
                "artifacts",
                "assumptions",
                "canonical_url",
                "evidence",
                "expected_license",
                "immutable_id",
                "retrieved_on",
                "source_id",
                "source_kind",
                "verification_status",
            },
            "SourceRecord",
        )
        immutable_value = data["immutable_id"]
        immutable_id = (
            require_string(immutable_value, "immutable_id") if immutable_value is not None else None
        )
        return cls(
            source_id=require_string(data["source_id"], "source_id"),
            source_kind=require_string(data["source_kind"], "source_kind"),
            canonical_url=require_https_url(data["canonical_url"], "canonical_url"),
            immutable_id=immutable_id,
            expected_license=require_string(data["expected_license"], "expected_license"),
            retrieved_on=require_string(data["retrieved_on"], "retrieved_on"),
            verification_status=_verification_status(data["verification_status"]),
            evidence=_strings(data["evidence"], "evidence"),
            assumptions=_strings(data["assumptions"], "assumptions"),
            artifacts=tuple(
                SourceArtifact.from_dict(require_object(item, "artifact"))
                for item in require_list(data["artifacts"], "artifacts")
            ),
        )


@dataclass(frozen=True)
class SourceRegistry:
    schema_version: Literal["voxelscope/source-registry/v1"]
    registry_id: str
    retrieved_on: str
    overall_status: Literal["no-go"]
    sources: tuple[SourceRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/source-registry/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.overall_status != "no-go":
            raise EvidenceError("unsafe_registry_status", self.overall_status)
        identifiers = [source.source_id for source in self.sources]
        if len(identifiers) != len(set(identifiers)):
            raise EvidenceError("duplicate_source", self.registry_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRegistry:
        data = strict_fields(
            data,
            {"overall_status", "registry_id", "retrieved_on", "schema_version", "sources"},
            "SourceRegistry",
        )
        schema_version = require_string(data["schema_version"], "schema_version")
        if schema_version != "voxelscope/source-registry/v1":
            raise EvidenceError("unsupported_schema", schema_version)
        overall_status = require_string(data["overall_status"], "overall_status")
        if overall_status != "no-go":
            raise EvidenceError("unsafe_registry_status", overall_status)
        return cls(
            "voxelscope/source-registry/v1",
            require_string(data["registry_id"], "registry_id"),
            require_string(data["retrieved_on"], "retrieved_on"),
            "no-go",
            tuple(
                SourceRecord.from_dict(require_object(item, "source"))
                for item in require_list(data["sources"], "sources")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    size_bytes: int
    sha256: str
    role: str

    def __post_init__(self) -> None:
        safe_relative_path(self.path)
        if self.size_bytes < 0:
            raise EvidenceError("invalid_size", self.path)
        require_sha256(self.sha256)
        if not self.role:
            raise EvidenceError("invalid_record", "archive member role is empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchiveMember:
        data = strict_fields(data, {"path", "role", "sha256", "size_bytes"}, "ArchiveMember")
        return cls(
            require_string(data["path"], "path"),
            require_int(data["size_bytes"], "size_bytes"),
            require_string(data["sha256"], "sha256"),
            require_string(data["role"], "role"),
        )


@dataclass(frozen=True)
class ArchivePolicy:
    format: Literal["zip"]
    max_files: int
    max_total_uncompressed_bytes: int
    max_file_size_bytes: int
    max_compression_ratio: int
    license_paths: tuple[str, ...]
    exact_members: tuple[ArchiveMember, ...]

    def __post_init__(self) -> None:
        if self.format != "zip":
            raise EvidenceError("unsupported_archive", self.format)
        if (
            min(
                self.max_files,
                self.max_total_uncompressed_bytes,
                self.max_file_size_bytes,
                self.max_compression_ratio,
            )
            <= 0
        ):
            raise EvidenceError("invalid_archive_limit", "archive limits must be positive")
        for path in self.license_paths:
            safe_relative_path(path)
        paths = [member.path for member in self.exact_members]
        if len(paths) != len(set(paths)):
            raise EvidenceError("duplicate_archive_member", "exact_members")
        if not self.license_paths or not set(self.license_paths).issubset(paths):
            raise EvidenceError("missing_license_contract", "license paths must be exact members")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchivePolicy:
        data = strict_fields(
            data,
            {
                "exact_members",
                "format",
                "license_paths",
                "max_compression_ratio",
                "max_file_size_bytes",
                "max_files",
                "max_total_uncompressed_bytes",
            },
            "ArchivePolicy",
        )
        archive_format = require_string(data["format"], "format")
        if archive_format != "zip":
            raise EvidenceError("unsupported_archive", archive_format)
        return cls(
            "zip",
            require_int(data["max_files"], "max_files"),
            require_int(data["max_total_uncompressed_bytes"], "max_total_uncompressed_bytes"),
            require_int(data["max_file_size_bytes"], "max_file_size_bytes"),
            require_int(data["max_compression_ratio"], "max_compression_ratio"),
            _strings(data["license_paths"], "license_paths"),
            tuple(
                ArchiveMember.from_dict(require_object(item, "exact member"))
                for item in require_list(data["exact_members"], "exact_members")
            ),
        )


@dataclass(frozen=True)
class AcquisitionArtifact:
    artifact_id: str
    source_id: str
    role: str
    source_url: str
    allowed_origins: tuple[str, ...]
    destination: str
    expected_size_bytes: int
    expected_hashes: tuple[DigestIdentity, ...]
    status: AcquisitionStatus
    blocked_reason: str | None
    archive: ArchivePolicy | None

    def __post_init__(self) -> None:
        require_https_url(self.source_url, "source_url")
        source_origin = f"{urlsplit(self.source_url).scheme}://{urlsplit(self.source_url).netloc}"
        if not self.allowed_origins or source_origin not in self.allowed_origins:
            raise EvidenceError("source_not_allowlisted", self.source_url)
        for origin in self.allowed_origins:
            require_https_url(origin, "allowed_origin", origin_only=True)
        safe_relative_path(self.destination)
        if self.expected_size_bytes <= 0:
            raise EvidenceError("invalid_size", self.artifact_id)
        if self.status not in {"ready", "blocked"}:
            raise EvidenceError("invalid_acquisition_status", self.status)
        if (self.status == "blocked") != (self.blocked_reason is not None):
            raise EvidenceError(
                "invalid_acquisition_status", "blocked artifacts require a reason only"
            )
        if self.status == "ready" and not self.expected_hashes:
            raise EvidenceError("missing_expected_hash", self.artifact_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcquisitionArtifact:
        data = strict_fields(
            data,
            {
                "allowed_origins",
                "archive",
                "artifact_id",
                "blocked_reason",
                "destination",
                "expected_hashes",
                "expected_size_bytes",
                "role",
                "source_id",
                "source_url",
                "status",
            },
            "AcquisitionArtifact",
        )
        status = _acquisition_status(data["status"])
        blocked_value = data["blocked_reason"]
        blocked_reason = (
            require_string(blocked_value, "blocked_reason") if blocked_value is not None else None
        )
        archive_value = data["archive"]
        archive = (
            ArchivePolicy.from_dict(require_object(archive_value, "archive"))
            if archive_value is not None
            else None
        )
        return cls(
            artifact_id=require_string(data["artifact_id"], "artifact_id"),
            source_id=require_string(data["source_id"], "source_id"),
            role=require_string(data["role"], "role"),
            source_url=require_https_url(data["source_url"], "source_url"),
            allowed_origins=tuple(
                require_https_url(item, "allowed_origin", origin_only=True)
                for item in require_list(data["allowed_origins"], "allowed_origins")
            ),
            destination=require_string(data["destination"], "destination"),
            expected_size_bytes=require_int(data["expected_size_bytes"], "expected_size_bytes"),
            expected_hashes=_digests(data["expected_hashes"], "expected_hashes"),
            status=status,
            blocked_reason=blocked_reason,
            archive=archive,
        )


@dataclass(frozen=True)
class AcquisitionPlan:
    schema_version: Literal["voxelscope/acquisition-plan/v1"]
    plan_id: str
    registry_id: str
    network_default: Literal["disabled"]
    overall_status: Literal["no-go"]
    artifacts: tuple[AcquisitionArtifact, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "voxelscope/acquisition-plan/v1":
            raise EvidenceError("unsupported_schema", self.schema_version)
        if self.network_default != "disabled":
            raise EvidenceError("unsafe_network_default", self.network_default)
        if self.overall_status != "no-go":
            raise EvidenceError("unsafe_plan_status", self.overall_status)
        identifiers = [artifact.artifact_id for artifact in self.artifacts]
        destinations = [artifact.destination for artifact in self.artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise EvidenceError("duplicate_acquisition_artifact", self.plan_id)
        if len(destinations) != len(set(destinations)):
            raise EvidenceError("duplicate_destination", self.plan_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcquisitionPlan:
        data = strict_fields(
            data,
            {
                "artifacts",
                "network_default",
                "overall_status",
                "plan_id",
                "registry_id",
                "schema_version",
            },
            "AcquisitionPlan",
        )
        schema_version = require_string(data["schema_version"], "schema_version")
        network_default = require_string(data["network_default"], "network_default")
        overall_status = require_string(data["overall_status"], "overall_status")
        if schema_version != "voxelscope/acquisition-plan/v1":
            raise EvidenceError("unsupported_schema", schema_version)
        if network_default != "disabled":
            raise EvidenceError("unsafe_network_default", network_default)
        if overall_status != "no-go":
            raise EvidenceError("unsafe_plan_status", overall_status)
        return cls(
            "voxelscope/acquisition-plan/v1",
            require_string(data["plan_id"], "plan_id"),
            require_string(data["registry_id"], "registry_id"),
            "disabled",
            "no-go",
            tuple(
                AcquisitionArtifact.from_dict(require_object(item, "artifact"))
                for item in require_list(data["artifacts"], "artifacts")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
