# Future execution runbook

## Status

This is a non-executable planning document. It does not authorize medical-data acquisition, file inspection, model loading, inference, GPU or MPS execution, cloud provisioning, or spending.

Milestones 3 through 8 establish:

- The official MONAI `brats_mri_segmentation` 0.5.2 model identity is pinned.
- The model archive is privately acquired and hash-verified without extraction or loading.
- OpenNeuro `ds007045` v2.0.1 source identity and one-volume selection are GO.
- The exact OpenNeuro plan was acquired once into owner-private custody.
- All eight content identities and the five-file bounded NIfTI structural gate are GO.
- A sanitized closed public bundle records only gate states, counts, synthetic fixtures, and non-claims.
- The neutral preprocessing positions `(I,J,K)` bind unchanged to model spatial positions `(D,H,W)`.
- The ROI, overlap, scan interval, final-anchor, and last-axis-fastest traversal contracts are pinned to immutable MONAI, model-zoo, and PyTorch source identities.
- MONAI symmetric padding and crop are distinct from the legacy VoxelScope high-side-only padded path.
- Two independent private window materialization paths are implemented with exact streamed byte comparison and bounded terminal evidence.
- Milestone 7 public evidence is synthetic-only; one separately approved private adapter execution and fresh-descriptor verification completed GO without persisting windows.
- A one-shot, allowlist-only model archive extraction and CPU loading qualification gate is implemented.
- The model-loading worker is bound to Linux, Python 3.12, PyTorch 2.4.0, MONAI 1.4.0, an explicitly approved closed runtime-distribution fingerprint, a reviewed SegResNet constructor, strict weights-only state loading, and no forward call.
- Milestone 8 public evidence is synthetic-only; the private model archive has not been accessed or extracted by this milestone.
- Real-data inference remains NO-GO.

## Model custody state

The model source, version, archive size, published SHA-1, observed SHA-256, closed member list, Apache-2.0 notice, configuration identities, and checkpoint identities are pinned by milestone 2.

Future work must not reacquire, extract, deserialize, or load the model until the execution protocol explicitly permits that action. Before use, an operator must review the existing private receipt and independently confirm that the custody root and artifact remain unchanged.

## Dataset custody state

The canonical plan is [`research/one-volume-acquisition-plan-v1.json`](../research/one-volume-acquisition-plan-v1.json). It pins four N4-only MRI objects in model order T1c, T1, T2, and FLAIR, one spatially aligned mask, and three dataset metadata files.

The selected case is fixed by an outcome-blind rule: the lexicographically first snapshot subject directory with the complete required file set, determined from tree metadata before any medical-byte, label-content, lesion-property, or model-output inspection.

The exact plan was approved and executed once through a no-retry mechanism. The attempt marker, terminal completion event, eight artifacts, custody receipt, and structural report remain owner-private. Every expected size and source-published SHA-256 matched. Do not reacquire or copy the files into the repository.

The source record uses a public deidentified BIDS locator only to bind the five medical objects. No participant crosswalk, direct identity, or clinical metadata may enter public evidence.

## NIfTI structural gate

Milestone 4 completed these checks before model loading:

1. Inspect only NIfTI headers and bounded structural properties.
2. Verify regular files, expected compression, dtype, finite values, dimensions, affine matrices, spacing, orientation, and exact geometry across all four modalities and the mask.
3. Confirm that the mask is integer-valued and contains only source values 0, 1, 2, and 3.
4. Record all observations in private hash-bound evidence.
5. Stop on any mismatch, unexpected value, malformed header, decompression limit, or evidence-write failure.

The resulting status is structural GO. This means only that the exact private files met the frozen mechanical predicates. It does not establish anatomical correctness, reference-label accuracy, preprocessing equivalence, or clinical validity.

No voxel rendering, model execution, or diagnostic assessment is authorized by this gate.

## Preprocessing and label gate

The selected source derivative is MNI152-registered, skull-stripped, and N4-corrected. The pre-z-scored derivative is not selected because the pinned MONAI inference configuration performs channel-wise normalization over nonzero voxels.

Milestone 5 completed one offline execution of a reviewed adapter that:

1. Loaded and stacked T1c, T1, T2, and FLAIR in role-defined order.
2. Preserved the unchanged source voxel-index axes and bound them to the private geometry identity.
3. Reproduced the bundle's nonzero channel-wise normalization through two independent implementations.
4. Excluded the label and refused every model-loading or inference path.

The resulting status is preprocessing GO. This status covers only role-driven channel loading, scaled float32 conversion, nonzero channel-wise normalization, independent implementation agreement, finite outputs, geometry preservation, label exclusion, private output publication, and terminal evidence. It does not resolve the mapping between unchanged Nibabel `(I,J,K)` voxel-index axes and VoxelScope synthetic `(Z,Y,X)` traversal.

Output inversion, sigmoid, threshold 0.5, nested TC, WT, and ET handling, and every comparison-only source-label mapping remain later gates.

The source mask uses 0 for background, 1 for necrotic core, 2 for non-enhancing or edematous tumor, and 3 for contrast-enhancing tumor. The model output encoding uses 0, 1, 2, and 4 after collapsing nested channels with ET priority. A numeric source 3 to model-output 4 mapping has been identified but is not authorized. It does not prove full semantic equivalence, especially for source value 2.

The label remains custody-only unless a later protocol proves semantic equivalence and authorizes its use. Output-preservation comparisons between runtime executions do not require a reference-label accuracy metric.

## Positional window bridge

Milestone 6 used public source identities and asymmetric synthetic fixtures only. It did not read a private tensor or record a private shape, affine, orientation, voxel value, tensor hash, or real window count.

The bridge proves:

1. Preprocessing positions `I`, `J`, and `K` map directly to model positions `D`, `H`, and `W`.
2. ROI components `(240,240,160)` retain that order.
3. Half-overlap intervals, final anchors, coverage, constant blending, and `K`-fastest traversal agree with an independent oracle.
4. MONAI pads undersized axes symmetrically and crops accumulated output back to the original positional extent.
5. Historical synthetic `Z,Y,X` names remain synthetic-only labels and the legacy bundle remains hash-identical.

The legacy VoxelScope engine pads only the high side. Its unpadded enumeration is positionally compatible, but its general padded extraction is not MONAI-equivalent and is **NO-GO** for real inference. A later private adapter must use the milestone 6 symmetric contract and must preserve the neutral axis names in evidence.

The canonical public identities are:

- plan SHA-256 `e50352f949a7d87470adf186f729861c770a99e4bc2b4fdb49eae490b101d048`;
- report SHA-256 `381083f12182689057fac91c804117e6cd23c7dd6ebbcae0a64f5ff9b07d645f`;
- closed public bundle SHA-256 `76895b2af591364156ac021ad482d2ed72fd96ad62f33042a4545238c587caa4`.

## Private window execution adapter

Milestone 7 implemented and completed one explicitly approved private qualification. The adapter:

1. Requires one explicit approval bound to the exact public adapter plan and exact private Milestone 5 report and snapshot identities.
2. Publishes a durable attempt marker before reading the approved private preprocessing report or tensor.
3. Verifies the closed Milestone 5 snapshot, stages the reference tensor through a no-clobber immutable copy, and refuses replacement races.
4. Materializes every window independently by full symmetric pad plus slice and by direct source intersection plus local zero padding.
5. Requires identical little-endian float32 bytes in `K`-fastest order and verifies finite values, anchors, starts, padded extent, local padding, intersections, coverage, crop geometry, channel order, and label exclusion.
6. Persists only a bounded private ledger, report, and one terminal completion or refusal event. Window tensors are never retained.

The public plan is [`research/window-execution-plan-v1.json`](../research/window-execution-plan-v1.json), SHA-256 `a9a6d123028269e534b48fd1bfe073972e2d4e4c338cbacde98aa6eb8cb35a52`. The synthetic-only closed public bundle is [`research/milestone-7`](../research/milestone-7), SHA-256 `3625597508b326497e31688c73ed1deff481fb198a36923f2fb44c921239b2f6`.

Private execution remains unauthorized. A later operator action must supply the privately held Milestone 5 report and snapshot identities through the CLI's bounded owner-private authorization file descriptor, never through command-line arguments. Those identities and the private root path must never be copied into committed records, shell history, command transcripts, process arguments, screenshots, issues, or pull requests.

## Private model-loading qualification gate

Milestone 8 implements, but does not privately execute, the model archive extraction and loading gate. Runtime preflight superseded v1 because the official CPU wheel reports its actual source commit rather than the later release tag commit. The corrected public plan is [`research/model-loading-plan-v2.json`](../research/model-loading-plan-v2.json), its reviewed build contract is [`research/model-loading-runtime-build-v1.json`](../research/model-loading-runtime-build-v1.json), and its synthetic-only public evidence is [`research/milestone-8-v2`](../research/milestone-8-v2). The v1 plan and bundle remain immutable historical refusal evidence.

The executor requires one new owner-private authorization bound to the final plan digest, existing private custody receipt digest, public official archive digest, exact isolated Python executable digest, and a separately reviewed closed PyTorch/MONAI runtime-distribution fingerprint. The private root, receipt identity, runtime path, and runtime fingerprint enter through a bounded inherited descriptor, never through command-line arguments or environment variables. The subprocess uses a fixed nonsensitive process argument name while selecting the approved executable separately. The executor writes a durable attempt marker before reading the approved receipt or archive and consumes the authorization on any terminal outcome.

The archive path is immutable-staged and checked against the exact 35,082,630-byte official archive identity. Every archive entry is verified against the already-public closed 13-member manifest. Absolute paths, traversal, duplicates, case collisions, encryption, directories, links, devices, special files, excessive counts, sizes, totals, or compression ratios are refused. Only the Apache license, pinned inference configuration, and `model.pt` checkpoint are extracted into an owner-private staging tree and atomically published.

The worker is a separate isolated subprocess with inherited request and result descriptors, sanitized standard streams and environment, a 120-second timeout, a 2 GiB Linux address-space limit, CPU-only accelerator visibility, and a Python audit-hook network refusal. It parses the inference JSON without dynamic evaluation and constructs only the reviewed `monai.networks.nets.SegResNet` call with fixed arguments. It calls only `torch.load(..., weights_only=True, map_location="cpu")`, accepts exactly a single `model` mapping of plain tensors, compares every key, shape, dtype, layout, and device against the constructed architecture, performs strict state loading, checks all parameters and buffers for finite CPU placement, records bounded aggregate counts privately, and exits. It never calls `forward`, creates model input, compiles, benchmarks, or moves the model to an accelerator.

Ordinary Python 3.12/3.13 CI does not install PyTorch or MONAI. It tests the archive and descriptor protocols with synthetic ZIPs and a safe JSON tensor fixture. Real compatibility with the private official checkpoint is deliberately unknown until the separately authorized run. Any unsupported weights-only encoding, wrapper, runtime mismatch, state mismatch, or resource failure produces a truthful private NO-GO.

Before a future private run:

1. Independently review the final plan and record its exact SHA-256.
2. Create a Linux Python 3.12 environment from `model-loading-requirements-py312.txt`; do not substitute framework versions.
3. Confirm the environment is offline and compute the exact Python executable SHA-256.
4. Review the existing private model custody receipt and official archive identity without copying either path or receipt identity into public artifacts.
5. Supply the private root, receipt digest, archive digest, plan digest, runtime executable path, and executable digest in one owner-only regular file or pipe attached as the authorization descriptor.
6. Run `python -m voxelscope.milestone8_cli private-execute --plan research/model-loading-plan-v2.json --authorization-fd <FD>` exactly once.
7. Treat either a private completion or refusal as terminal. Do not retry under the same authorization.
8. Keep the extraction manifest, loading report, state observations, and all resulting identities private.

This action, when separately approved, authorizes extraction and stop-after-load qualification only. It does not authorize a medical tensor, model input, forward call, inference, GPU/MPS, network, cloud, spend, or publication of private observations.

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

- A separate explicit authorization bound to the final Milestone 8 plan, reviewed private model receipt/archive identities, exact interpreter identity, and reviewed runtime-distribution fingerprint.
- Source-mask and model-output semantic mapping review.
- Subject-overlap resolution or an approved output-preservation-only bound.
- Runtime image, library, CUDA, driver, and backend identities.
- Host selection, budget authorization, and independent protocol review.
- An explicit operator start action.
