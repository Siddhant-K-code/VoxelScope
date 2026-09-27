# Future execution runbook

## Status

This is a non-executable planning document. It does not authorize medical-data acquisition, file inspection, model loading, inference, GPU or MPS execution, cloud provisioning, or spending.

Milestone 3 establishes:

- The official MONAI `brats_mri_segmentation` 0.5.2 model identity is pinned.
- The model archive is privately acquired and hash-verified without extraction or loading.
- OpenNeuro `ds007045` v2.0.1 source identity and one-volume selection are GO.
- The OpenNeuro acquisition plan remains network-disabled and blocked.
- Real-data inference remains NO-GO.

## Model custody state

The model source, version, archive size, published SHA-1, observed SHA-256, closed member list, Apache-2.0 notice, configuration identities, and checkpoint identities are pinned by milestone 2.

Future work must not reacquire, extract, deserialize, or load the model until the execution protocol explicitly permits that action. Before use, an operator must review the existing private receipt and independently confirm that the custody root and artifact remain unchanged.

## Dataset acquisition gate

The canonical plan is [`research/one-volume-acquisition-plan-v1.json`](../research/one-volume-acquisition-plan-v1.json). It pins four N4-only MRI objects in model order T1c, T1, T2, and FLAIR, one spatially aligned mask, and three dataset metadata files.

The selected case is fixed by an outcome-blind rule: the lexicographically first snapshot subject directory with the complete required file set, determined from tree metadata before any medical-byte, label-content, lesion-property, or model-output inspection.

The plan is blocked and has no acquisition command. To advance it:

1. Obtain explicit operator approval for the exact canonical plan hash.
2. Add or approve a private-custody acquisition mechanism that requires an owner-only root outside the repository, exact URL and origin allowlists, no redirect outside those origins, no retry, and no overwrite.
3. Acquire only the eight pinned objects. Do not acquire the full dataset or use range extraction from a bulk archive.
4. Verify every expected byte size and SHA-256 before opening any file.
5. Write owner-only receipts without publishing private paths, signed URLs, medical bytes, or private receipt content.
6. Stop after custody verification until the structural and preprocessing gates are separately approved.

The source record uses a public deidentified BIDS locator only to bind the five medical objects. No participant crosswalk, direct identity, or clinical metadata may enter public evidence.

## NIfTI structural gate

After private acquisition and before model loading:

1. Inspect only NIfTI headers and bounded structural properties.
2. Verify regular files, expected compression, dtype, finite values, dimensions, affine matrices, spacing, orientation, and exact geometry across all four modalities and the mask.
3. Confirm that the mask is integer-valued and contains only source values 0, 1, 2, and 3.
4. Record all observations in private hash-bound evidence.
5. Stop on any mismatch, unexpected value, malformed header, decompression limit, or evidence-write failure.

No voxel rendering, model execution, or diagnostic assessment is authorized by this gate.

## Preprocessing and label gate

The selected source derivative is MNI152-registered, skull-stripped, and N4-corrected. The pre-z-scored derivative is not selected because the pinned MONAI inference configuration performs channel-wise normalization over nonzero voxels.

Before inference, a reviewed adapter must define:

1. Exact loading and stacking in T1c, T1, T2, and FLAIR order.
2. Whether source geometry is accepted unchanged or transformed, with a hash-bound proof.
3. Exact reproduction of the bundle's nonzero channel-wise normalization.
4. Output inversion, sigmoid, threshold 0.5, and nested TC, WT, and ET channel handling.
5. Any comparison-only mapping between source-mask labels and model output labels.

The source mask uses 0 for background, 1 for necrotic core, 2 for non-enhancing or edematous tumor, and 3 for contrast-enhancing tumor. The model output encoding uses 0, 1, 2, and 4 after collapsing nested channels with ET priority. A numeric source 3 to model-output 4 mapping has been identified but is not authorized. It does not prove full semantic equivalence, especially for source value 2.

The label remains custody-only unless a later protocol proves semantic equivalence and authorizes its use. Output-preservation comparisons between runtime executions do not require a reference-label accuracy metric.

## Subject overlap gate

The model was trained on BraTS 2018. The selected OpenNeuro cohort is described as a separate collection acquired from 2018 through 2025, but the selected case acquisition date and the model's exact training roster are not available in the public contract.

Treat overlap as unresolved. Unless independently resolved or explicitly bounded in an approved amendment, report only output preservation against the same pinned FP32 reference. Do not report reference-label accuracy, diagnostic validity, clinical safety, or patient-specific performance.

## Local M5 Pro role

The local Apple M5 Pro environment is limited to fixture generation, schema development, verifier execution, report preparation, test execution, and approved evidence inspection. A later amendment may permit an isolated MPS compatibility probe. Such a probe is not a reference arm or performance result.

Local work must not silently alter affine orientation, modality order, threshold policy, padding, or array dtype. No medical data, private paths, credentials, or private receipts belong in the repository, test cache, logs, screenshots, or issue attachments.

## First real cloud pass

The first real pass requires a single GPU with at least 24 GB of memory. Before provisioning:

1. Approve exact provider, region, instance type, image digest, storage, network policy, and hourly cost.
2. Confirm the projected total is at most USD 20.
3. Disable automatic retry, autoscaling, restart, and fallback provisioning.
4. Prepare a stop command and cost observation independent of the run process.
5. Upload only approved hash-verified artifacts through the approved private mechanism.
6. Run the FP32 reference qualification before any accelerated arm.
7. Stop immediately on a refusal gate, nondeterministic reference, evidence error, or projected budget breach.
8. Download and verify the closed evidence bundle before destroying the instance and storage.

No cloud resource is provisioned by this repository.

## Arm order

The future order is PyTorch FP32 reference and exact repeat, PyTorch AMP, TensorRT FP32, then TensorRT FP16. Each arm uses the same pinned input, preprocessing, ledger, padding, blend weights, threshold, and output writer. A failed arm is recorded and is not silently retried or replaced.

## Remaining approvals

Real execution remains blocked on:

- Explicit approval and private execution of the OpenNeuro acquisition plan.
- Private NIfTI structural and cross-file geometry verification.
- Preprocessing adapter and equivalence review.
- Source-mask and model-output semantic mapping review.
- Subject-overlap resolution or an approved output-preservation-only bound.
- Runtime image, library, CUDA, driver, and backend identities.
- Host selection, budget authorization, and independent protocol review.
- An explicit operator start action.
