# One-volume feasibility protocol v1

## Status and purpose

This is a prospective, zero-inference protocol. It does not authorize execution. It defines how one future VoxelScope run may test numerical output preservation against a pinned PyTorch FP32 reference.

The protocol is research infrastructure only. It does not assess diagnosis, treatment, clinical safety, radiologist equivalence, patient-specific validity, or diagnostic accuracy. No clinical meaning may be inferred from a model name, channel, label, or mask.

## Preconditions

Execution remains prohibited until every gate in [`real-data-readiness-v1.md`](real-data-readiness-v1.md) has an approved GO record. Required approvals include:

1. Private custody receipts for the pinned model bundle, exactly one training image, its matching label, and `dataset.json`.
2. An approved modality conversion from the dataset descriptor order to the model order T1c, T1, T2, and FLAIR.
3. An approved explanation of spatial geometry, nonzero channel-wise normalization, output inversion, thresholding, and label encoding.
4. A pinned software and driver environment.
5. A single approved host with at least 24 GB GPU memory.
6. A total authorized spend of USD 20 or less.
7. An explicit operator start action. There is no automatic retry.

The label is retained for custody identity only. It must not be used to compute diagnostic accuracy unless exact subject non-overlap is proved prospectively.

## Fixed study scope

- Use exactly one privately held Task01 training volume.
- Use the pinned MONAI `brats_mri_segmentation` 0.5.2 checkpoint.
- Preserve all source, model, environment, preprocessing, window, output, and run identities in private evidence.
- Publish no medical image, subject identifier, source file name, private path, or private receipt.
- Publish a medical-data hash only after an owner determines that it cannot disclose or enable linkage to a subject identity. Otherwise publish status and counts only.
- Do not provision or change cloud resources from the run command.

## Arm order

1. Produce the PyTorch FP32 reference first.
2. Repeat the PyTorch FP32 arm in the same runtime before any candidate.
3. Require the repeat to satisfy the separately approved deterministic repeat contract.
4. Stop if the reference or repeat fails, differs unexpectedly, or lacks complete evidence.
5. Consider future PyTorch AMP or ONNX-TensorRT arms only after the reference and repeat qualify.
6. Execute each authorized arm once. Do not retry automatically after interruption, error, timeout, or incomplete evidence.

PyTorch AMP and TensorRT are future candidate arms, not part of this milestone. Torch-TensorRT is not an approved arm.

## Output contract

The reference contract binds:

- The exact model archive and checkpoint identity.
- The exact private volume identity and modality order.
- Source affine, spacing, shape, and orientation evidence.
- Preprocessing parameters and normalized input evidence.
- Sliding-window ROI, overlap, traversal, padding, and blend identities.
- Three probability channels in TC, WT, and ET order.
- Sigmoid activation and threshold 0.5.
- The restored output geometry and encoded mask.
- Runtime, device, precision, library, driver, and environment identities.

The same-runtime repeat is a qualification check before candidate execution. Candidate comparison must use VoxelScope output-preservation records and must remain descriptive.

## Acceptance rule

There is no approved real-data drift pass threshold in version 1. No AMP or TensorRT result can be labeled pass, equivalent, safe, or acceptable under this protocol.

A future threshold must be preregistered in a new protocol version before candidate execution. It must define probability, threshold-flip, mask, connected-component, and physical boundary criteria. It cannot be selected after observing candidate output.

## Stop conditions

Stop before inference if any required artifact, identity, license review, preprocessing decision, host pin, privacy review, or budget approval is missing.

Stop during the future run after any reference failure, repeat mismatch, evidence-write failure, unexpected resource use, or cost risk. Preserve failure evidence privately. Do not retry automatically.

## Reporting boundary

Allowed reporting is limited to whether a candidate preserved output relative to the pinned FP32 reference under the preregistered contract. If subject overlap remains unresolved, no comparison to the held label and no diagnostic accuracy metric is allowed.
