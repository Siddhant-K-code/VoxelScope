# One-volume feasibility protocol v2

## Status and purpose

This is the active prospective, zero-inference protocol. It supersedes unexecuted protocol v1 because milestone 3 selected a different data source. Milestone 4 completed the separately approved private acquisition and structural gates. This protocol still does not authorize model loading or inference.

The protocol defines how one future VoxelScope run may test numerical output preservation against a pinned PyTorch FP32 reference. It does not assess diagnosis, treatment, clinical safety, radiologist equivalence, patient-specific validity, or diagnostic accuracy. No clinical meaning may be inferred from a model name, channel, label, or mask.

## Preconditions

Execution remains prohibited until every gate below has an approved GO record:

1. Private custody receipts for the pinned model bundle, four selected OpenNeuro N4-only MRI objects, aligned mask, `dataset_description.json`, `README.txt`, and `CHANGES`.
2. Verified NIfTI shape, affine, dtype, finite values, and cross-file geometry, plus an approved binding to model order T1c, T1, T2, and FLAIR.
3. An approved preprocessing adapter covering source MNI152 registration, skull stripping, N4 correction, bundle nonzero channel-wise normalization, output inversion, thresholding, and label encoding.
4. A pinned software and driver environment.
5. A single approved host with at least 24 GB GPU memory.
6. A total authorized spend of USD 20 or less.
7. An explicit operator start action. There is no automatic retry.

The aligned label is retained for custody identity only. The source value 3 to model-output value 4 numeric mapping is not authorized by this protocol. Full source-label to nested TC, WT, and ET semantic equivalence remains unresolved. The label must not be used for diagnostic accuracy unless semantics and exact subject non-overlap are proved prospectively.

Source-path GO in [`one-volume-source-decision-v1.md`](one-volume-source-decision-v1.md) did not satisfy an execution precondition by itself. Milestone 4 consumed one explicit approval for the canonical plan, acquired its exact objects privately, and recorded structural GO. The sanitized public result is [`research/milestone-4`](../research/milestone-4/). Private custody does not authorize model loading.

The case selection is outcome-blind: choose the lexicographically first snapshot subject directory with all four required N4-only modalities and an aligned mask, using only OpenNeuro tree metadata before any medical-byte, label-content, lesion-property, or model-output inspection.

## Fixed study scope

- Use exactly the selected OpenNeuro `ds007045` v2.0.1 case pinned by [`research/one-volume-source-decision-v1.json`](../research/one-volume-source-decision-v1.json) and [`research/one-volume-acquisition-plan-v1.json`](../research/one-volume-acquisition-plan-v1.json).
- Use the pinned MONAI `brats_mri_segmentation` 0.5.2 checkpoint.
- Preserve all source, model, environment, preprocessing, window, output, and run identities in private evidence.
- Canonical source records may publish only repository-published deidentified locators, exact immutable source paths, sizes, version IDs, ETags, and hashes needed to bind official public artifacts.
- Publish no participant crosswalk, direct identity, clinical metadata, medical bytes, private custody path, signed URL, or private receipt.
- Keep locally derived private custody hashes and receipt content private unless a separate privacy review approves publication. A hash identical to an already published official source identity may be referenced as that public identity, but its private receipt and custody path remain private.
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
- Source affine, spacing, shape, orientation, and finite-value evidence.
- Preprocessing parameters and normalized input evidence.
- Sliding-window ROI, overlap, traversal, padding, and blend identities.
- Three probability channels in TC, WT, and ET order.
- Sigmoid activation and threshold 0.5.
- The restored output geometry and encoded mask.
- Runtime, device, precision, library, driver, and environment identities.

The same-runtime repeat is a qualification check before candidate execution. Candidate comparison must use VoxelScope output-preservation records and remain descriptive.

## Label boundary

The selected source mask records:

- Value 0: background or non-tumor tissue.
- Value 1: necrotic core.
- Value 2: non-enhancing or edematous tumor.
- Value 3: contrast-enhancing tumor.

The pinned MONAI output encoding records:

- Value 0: background.
- Value 1: tumor core excluding enhancing tumor after priority encoding.
- Value 2: whole tumor excluding tumor core after priority encoding.
- Value 4: enhancing tumor.

These are distinct semantic domains. A future numeric 3 to 4 mapping would address ET encoding only. It would not prove full equivalence for values 1 and 2, and it is not authorized by this protocol.

## Acceptance rule

There is no approved real-data drift pass threshold in version 2. No AMP or TensorRT result can be labeled pass, equivalent, safe, or acceptable under this protocol.

A future threshold must be preregistered in a new protocol version before candidate execution. It must define probability, threshold-flip, mask, connected-component, and physical boundary criteria. It cannot be selected after observing candidate output.

## Stop conditions

The approved acquisition attempt is complete and must not be repeated.

Stop before inference if any required artifact, identity, NIfTI structural check, license review, preprocessing decision, semantic decision, overlap bound, host pin, privacy review, or budget approval is missing.

Stop during the future run after any reference failure, repeat mismatch, evidence-write failure, unexpected resource use, or cost risk. Preserve failure evidence privately. Do not retry automatically.

## Reporting boundary

Allowed reporting is limited to whether a candidate preserved output relative to the pinned FP32 reference under the preregistered contract. While subject overlap or label semantics remain unresolved, no comparison to the held label and no diagnostic accuracy metric is allowed.
