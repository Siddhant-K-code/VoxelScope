# SPDX-License-Identifier: Apache-2.0
"""Trusted public pins for the milestone 2 custody contract."""

from __future__ import annotations

from .canonical import EvidenceError, canonical_json_bytes, sha256_bytes
from .custody_records import AcquisitionPlan, SourceRegistry

REGISTRY_ID = "voxelscope-real-data-readiness-v1"
PLAN_ID = "voxelscope-private-custody-v1"
REGISTRY_SHA256 = "8bc54b665a63f22555b5369d346a90e03bc32cb07e99c75af28e728ab8384e1f"
PLAN_SHA256 = "7e35c6432bcd67d9f88e70e2eb9153785468573a08058c49c6e228deae6fb2cc"

MONAI_SOURCE_ID = "monai-brats-mri-segmentation-ngc-v0.5.2"
MONAI_VERSION = "0.5.2"
MONAI_MODEL_ZOO_COMMIT = "5370ce6ea1dd132856b9c92e2fa125548594835d"
MONAI_MODEL_INFO_URL = (
    "https://raw.githubusercontent.com/Project-MONAI/model-zoo/"
    "5370ce6ea1dd132856b9c92e2fa125548594835d/models/model_info.json"
)
MONAI_MODEL_INFO_BLOB_SHA1 = "a2da19efdb0b7f3afc12589d503100704218851a"
MONAI_MODEL_INFO_SHA256 = "c25ae88807635399df5671322088ce4bbb59408506adb9c81120eb323d59511d"
MONAI_ARCHIVE_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/"
    "brats_mri_segmentation/versions/0.5.2/files/"
    "brats_mri_segmentation_v0.5.2.zip"
)
MONAI_ARCHIVE_SIZE = 35_082_630
MONAI_ARCHIVE_SHA1 = "6b1dfef29d49c6f6a1d8bf9f65c84125ad37a6e9"
MONAI_ARCHIVE_SHA256 = "6f7d21de3f0ce28deb332cb777ba30cc9a68f5bedaa18071c4fa7d2f7a147267"

MSD_SOURCE_ID = "msd-task01-brain-tumour-aws-v1"
MSD_REGISTRY_COMMIT = "54b063ff060ab2359ee598d2510d4c4b26729a93"
MSD_ARCHIVE_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MSD_ARCHIVE_VERSION_ID = "GGb3au0oJ_TTvN7jw0HE17RKvzVTFj0R"
MSD_ARCHIVE_SIZE = 7_608_266_240


def verify_registry_contract(registry: SourceRegistry) -> None:
    if sha256_bytes(canonical_json_bytes(registry)) != REGISTRY_SHA256:
        raise EvidenceError("registry_contract_mismatch", "canonical registry hash")
    if registry.registry_id != REGISTRY_ID:
        raise EvidenceError("registry_contract_mismatch", "registry_id")
    by_id = {source.source_id: source for source in registry.sources}
    if set(by_id) != {
        MONAI_SOURCE_ID,
        "monai-model-zoo-v0.5.2",
        "monai-msd-catalog-pin",
        MSD_SOURCE_ID,
        "medical-segmentation-decathlon-task01",
    }:
        raise EvidenceError("registry_contract_mismatch", "source set")
    model = by_id[MONAI_SOURCE_ID]
    if (
        model.canonical_url != MONAI_ARCHIVE_URL
        or model.immutable_id != f"ngc-version:{MONAI_VERSION}"
        or model.expected_license != "Apache-2.0"
        or model.verification_status != "verified"
    ):
        raise EvidenceError("registry_contract_mismatch", MONAI_SOURCE_ID)
    archive = next(
        (
            artifact
            for artifact in model.artifacts
            if artifact.path == "brats_mri_segmentation_v0.5.2.zip"
        ),
        None,
    )
    if archive is None or archive.size_bytes != MONAI_ARCHIVE_SIZE:
        raise EvidenceError("registry_contract_mismatch", "MONAI archive identity")
    hashes = {digest.algorithm: digest.value for digest in archive.hashes}
    if hashes.get("sha1") != MONAI_ARCHIVE_SHA1 or hashes.get("sha256") != MONAI_ARCHIVE_SHA256:
        raise EvidenceError("registry_contract_mismatch", "MONAI archive hashes")
    model_zoo = by_id["monai-model-zoo-v0.5.2"]
    if (
        model_zoo.immutable_id != f"git-commit:{MONAI_MODEL_ZOO_COMMIT}"
        or model_zoo.canonical_url
        != f"https://github.com/Project-MONAI/model-zoo/commit/{MONAI_MODEL_ZOO_COMMIT}"
    ):
        raise EvidenceError("registry_contract_mismatch", "MONAI model-zoo revision")
    model_info = next(
        (artifact for artifact in model_zoo.artifacts if artifact.path == "models/model_info.json"),
        None,
    )
    if model_info is None or model_info.size_bytes != 117_477:
        raise EvidenceError("registry_contract_mismatch", "MONAI model index identity")
    model_info_hashes = {digest.algorithm: digest.value for digest in model_info.hashes}
    if (
        model_info_hashes.get("git-blob-sha1") != MONAI_MODEL_INFO_BLOB_SHA1
        or model_info_hashes.get("sha256") != MONAI_MODEL_INFO_SHA256
        or MONAI_MODEL_INFO_URL not in model_zoo.evidence
    ):
        raise EvidenceError("registry_contract_mismatch", "MONAI model index hashes")
    dataset = by_id[MSD_SOURCE_ID]
    if (
        dataset.canonical_url != MSD_ARCHIVE_URL
        or dataset.immutable_id != f"s3-version-id:{MSD_ARCHIVE_VERSION_ID}"
        or dataset.expected_license != "CC-BY-SA-4.0"
        or dataset.verification_status != "blocked"
    ):
        raise EvidenceError("registry_contract_mismatch", MSD_SOURCE_ID)


def verify_plan_contract(plan: AcquisitionPlan, registry: SourceRegistry) -> None:
    verify_registry_contract(registry)
    if sha256_bytes(canonical_json_bytes(plan)) != PLAN_SHA256:
        raise EvidenceError("plan_contract_mismatch", "canonical plan hash")
    if plan.plan_id != PLAN_ID or plan.registry_id != registry.registry_id:
        raise EvidenceError("plan_contract_mismatch", "plan identity")
    by_id = {artifact.artifact_id: artifact for artifact in plan.artifacts}
    if set(by_id) != {"monai-brats-bundle-v0.5.2", "msd-task01-single-volume"}:
        raise EvidenceError("plan_contract_mismatch", "artifact set")
    model = by_id["monai-brats-bundle-v0.5.2"]
    model_hashes = {digest.algorithm: digest.value for digest in model.expected_hashes}
    if (
        model.source_id != MONAI_SOURCE_ID
        or model.source_url != MONAI_ARCHIVE_URL
        or model.expected_size_bytes != MONAI_ARCHIVE_SIZE
        or model_hashes.get("sha1") != MONAI_ARCHIVE_SHA1
        or model_hashes.get("sha256") != MONAI_ARCHIVE_SHA256
        or model.status != "ready"
        or model.archive is None
    ):
        raise EvidenceError("plan_contract_mismatch", model.artifact_id)
    dataset = by_id["msd-task01-single-volume"]
    if (
        dataset.source_id != MSD_SOURCE_ID
        or dataset.source_url != MSD_ARCHIVE_URL
        or dataset.expected_size_bytes != MSD_ARCHIVE_SIZE
        or dataset.status != "blocked"
        or dataset.blocked_reason is None
    ):
        raise EvidenceError("plan_contract_mismatch", dataset.artifact_id)
