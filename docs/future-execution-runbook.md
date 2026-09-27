# Future execution runbook

## Status

This is a non-executable planning document. PR 1 does not authorize or perform model acquisition, dataset acquisition, medical-data handling, GPU execution, MPS execution, cloud provisioning, or spending.

## Acquisition gates

The exact acquisition commands remain placeholders until an independent reviewer pins all identities and terms.

### Model placeholder

1. Confirm the official MONAI `brats_mri_segmentation` bundle source and immutable version.
2. Record the source commit or release, Apache-2.0 notice, configuration hash, weights hash, network hash, preprocessing contract, and documented input shape.
3. Review the bundle for remote code, dynamic dependencies, and license obligations.
4. Replace `[MODEL_ACQUISITION_COMMAND_PENDING_REVIEW]` only through a reviewed protocol amendment.
5. Acquire locally into a quarantined staging directory, verify hashes before loading, and copy only approved artifacts into custody.

### Dataset placeholder

1. Confirm the Medical Segmentation Decathlon Task01 BrainTumour source, immutable object identity, and CC BY-SA 4.0 obligations.
2. Document the modality mapping from FLAIR, T1w, T1gd, and T2w to the model contract.
3. Resolve whether any candidate subject overlaps BraTS 2018 training data.
4. Replace `[DATASET_ACQUISITION_COMMAND_PENDING_REVIEW]` only through a reviewed protocol amendment.
5. Stage exactly one approved volume locally, scan it for identifiers, and hash it before any transformation.

A mutable URL, filename, or dataset title is not an identity. No acquisition proceeds while lineage is unresolved if diagnostic accuracy is proposed.

## Local M5 Pro role

The local Apple M5 Pro environment is intended for fixture generation, schema development, verifier execution, report rendering preparation, test execution, and evidence inspection. A later amendment may permit an isolated MPS compatibility probe. Such a probe is not a reference arm, not a performance result, and not part of PR 1.

Local work must not silently alter affine orientation, modality order, threshold policy, padding, or array dtype. No patient data or secrets belong in the repository, test cache, logs, screenshots, or issue attachments.

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

Real execution remains blocked on model identity, dataset identity, license and terms review, subject-lineage evidence, one-volume selection, preprocessing equivalence, runtime image identity, cloud budget authorization, and independent protocol review.
