# Real-data readiness v1

VoxelScope remains **NO-GO** for a real-data feasibility run. This milestone prepares source and custody contracts only. It does not load a model, inspect voxel arrays, run inference, provision a host, authorize GPU spend, or support a clinical claim.

## Pinned sources

The public source registry is [`research/source-registry-v1.json`](../research/source-registry-v1.json). It separates observed evidence from assumptions and is verified as canonical JSON by the offline CLI.

### Model

- Bundle: MONAI `brats_mri_segmentation` version `0.5.2`.
- Official NGC archive: `https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/brats_mri_segmentation/versions/0.5.2/files/brats_mri_segmentation_v0.5.2.zip`.
- Official model-zoo revision: `ebdbeb6e1d374ec1e17f4f7f86406ecf19f943c6`.
- License: Apache-2.0.
- Archive size: 35,082,630 bytes.
- Published archive SHA-1: `6b1dfef29d49c6f6a1d8bf9f65c84125ad37a6e9`.
- Independently observed archive SHA-256: `6f7d21de3f0ce28deb332cb777ba30cc9a68f5bedaa18071c4fa7d2f7a147267`.

The official metadata defines a four-channel float32 input and a three-channel float32 output. Model input channel order is T1c, T1, T2, and FLAIR at 1 mm isotropic resolution. Output channels are TC, WT, and ET. The model card states an actual training input of 224 x 224 x 144 and at least 16 GB GPU memory for training. The pinned inference configuration instead declares a 240 x 240 x 160 sliding-window ROI with overlap 0.5. That difference must be resolved prospectively before execution.

The inference preprocessing loads aligned images and performs channel-wise intensity normalization over nonzero voxels. Postprocessing applies sigmoid, restores the prediction to source geometry, thresholds at 0.5, and encodes ET as 4, TC as 1, WT as 2, and background as 0. The bundle documents ONNX-TensorRT conversion support. Torch-TensorRT is described as under development in the pinned model card.

The model card includes validation scores. They are not VoxelScope results, are not reproduced here, and do not authorize a diagnostic accuracy claim.

### Data

- Dataset: Medical Segmentation Decathlon Task01_BrainTumour.
- Official registry revision: `54b063ff060ab2359ee598d2510d4c4b26729a93`.
- Official bucket object: `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar`.
- S3 object version: `GGb3au0oJ_TTvN7jw0HE17RKvzVTFj0R`.
- Object size: 7,608,266,240 bytes.
- License: CC BY-SA 4.0.

The official Medical Segmentation Decathlon site describes Task01 as multimodal MRI from BraTS 2016 and 2017. The expected modalities are FLAIR, T1w, T1gd, and T2w. The expected label mapping is background 0, edema 1, non-enhancing or necrotic tumor 2, and enhancing tumor 3.

The expected archive structure is `Task01_BrainTumour/dataset.json`, `imagesTr` with 484 four-channel training images, `labelsTr` with matching training labels, and `imagesTs` with 266 four-channel test images. This structure and its member names are assumptions until the official descriptor and selected files can be acquired and verified directly.

The official AWS bucket currently lists Task01 only as the full tar object. It does not expose individual training images, labels, or `dataset.json` as directly addressable objects. VoxelScope therefore did not acquire the 7.6 GB archive, did not select a subject, and did not extract any member. The expected descriptor member is recorded as `Task01_BrainTumour/dataset.json`, but its bytes, file identity, and candidate volume references remain unverified.

## Subject overlap assessment

The model was trained on BraTS 2018. Task01 derives from BraTS 2016 and 2017. Earlier BraTS subjects may be included in later releases. No exact subject identity comparison has been completed, so overlap is unresolved and must be assumed possible.

This blocks diagnostic accuracy metrics and claims. A future one-volume run may evaluate only output preservation against its own pinned PyTorch FP32 reference unless non-overlap is proven before execution.

## GO and NO-GO gates

| Gate | Current state | Evidence required for GO |
|---|---|---|
| License | NO-GO | Owner review of Apache-2.0 model terms, bundled data terms, and CC BY-SA 4.0 dataset terms. |
| Source identity | NO-GO | Reviewer approval of the pinned registry and the blocked Task01 descriptor identity. |
| Model weights | NO-GO | Private receipt review confirming the pinned archive and all member identities without model loading. |
| Data identity | NO-GO | Direct official acquisition of exactly one training image, its label, and `dataset.json`, followed by private format and hash verification. |
| Subject overlap | NO-GO | Exact, independently reviewed non-overlap proof. Without it, output preservation is the only allowed measurement scope. |
| Preprocessing equivalence | NO-GO | Approved channel reorder and label handling, normalization equivalence, spatial geometry checks, and resolution of the 224 x 224 x 144 training input versus 240 x 240 x 160 inference ROI. |
| Runtime | NO-GO | Pinned PyTorch, MONAI, CUDA, driver, and future TensorRT versions with an offline environment receipt. |
| Output contract | NO-GO | Approved FP32 reference record, same-runtime repeat rule, exact TC, WT, and ET channel handling, and a separately approved candidate threshold. |
| Host | NO-GO | An explicitly approved host with at least 24 GB GPU memory and no automatic retry. |
| Budget | NO-GO | Explicit operator approval for a total cap of USD 20 or less. |
| Privacy | NO-GO | Owner-private custody root, private receipts, no public identifiers or paths, and repository scan approval. |

No individual GO item overrides another NO-GO item. A run requires every gate to be reviewed and changed to GO in a new versioned readiness record.

## Current custody result

The pinned MONAI archive was acquired into owner-only session custody and verified without extraction or model loading. Public-safe model hashes and bundle-member identities are in the source registry. The repository contains no model weights, medical images, subject identifiers, credentials, private paths, or private receipt content.

Task01 custody is blocked because exact single-file acquisition from the official bucket is unavailable. This is the controlling NO-GO blocker.

## Next operator action

Obtain an official, directly addressable source for exactly one Task01 training image, its matching label, and the exact `dataset.json`, with immutable sizes and content hashes. Do not download the full Task01 tar as a workaround. After those three objects are privately staged and reviewed, create a new readiness revision that resolves data identity and preprocessing equivalence while keeping subject overlap unresolved unless exact non-overlap evidence exists.
