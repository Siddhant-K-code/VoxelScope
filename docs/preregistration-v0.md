# Prospective feasibility protocol v0

## Purpose

This protocol governs the first future one-volume feasibility study for VoxelScope. The question is whether accelerated 3D sliding-window inference can preserve the output of a pinned PyTorch FP32 reference. This is a research software feasibility study, not a diagnostic-accuracy, clinical-safety, radiologist-equivalence, or patient-validity study.

PR 1 executes only independent synthetic unit tests. It does not execute the one-volume study.

Milestone 3 supersedes the Task01 source preference in this historical preregistration with the selected OpenNeuro path. It does not authorize acquisition or execution. The active source-specific protocol is [`one-volume-feasibility-protocol-v2.md`](one-volume-feasibility-protocol-v2.md).

## Study boundaries

### Independent unit boundary

Unit evidence uses analytic NumPy arrays with no medical metadata. It establishes canonical serialization, hash custody, window scheduling, blend identity, nested-region validation, drift calculations, refusal behavior, and bundle verification. Passing unit tests does not imply that any model backend is equivalent.

### One-volume feasibility boundary

A later amendment may authorize exactly one licensed, locally staged volume after model identity, dataset identity, terms, and lineage restrictions are independently pinned. That run will compare four arms on the same input and declared preprocessing:

1. PyTorch FP32 reference.
2. PyTorch AMP.
3. TensorRT FP32.
4. TensorRT FP16.

No arm is executed by PR 1. No additional volume may be added until the feasibility receipt is complete and reviewed.

## Input, model, and runtime custody

Every input modality must have a canonical content hash, file hash, shape, spacing, and modality position in the fixed order `T1`, `T1c`, `T2`, `FLAIR`. The affine must have its own digest. Labels are optional and must have a digest when present.

The future model identity must pin the official MONAI `brats_mri_segmentation` bundle version, source, Apache-2.0 notice, configuration hash, weights hash, network definition, preprocessing configuration, and expected `(224, 224, 144)` model input. A name or mutable URL is not sufficient.

Each runtime must record operating system, architecture, Python, NumPy, PyTorch, MONAI, CUDA, cuDNN, TensorRT, driver, GPU model, deterministic settings, precision controls, engine hash, and relevant build configuration in a content-hashed provenance record. Usernames, hostnames, cloud account identifiers, and secrets are excluded.

Custody fails if any required artifact is missing, mutable, unhashed, noncanonical, unexpectedly changed, or linked outside the evidence root.

## Window contract

Arrays use `(C, Z, Y, X)` order. Coordinates are zero-based and half-open `[start, end)`. Traversal is lexicographic `Z, Y, X`, with `X` changing fastest.

Overlap is an exact rational `p/q` relative to ROI extent. Per-axis stride is:

```text
max(1, floor(roi * (q - p) / q))
```

The final start is `max(volume - roi, 0)` and is appended if not already present. Only explicit high-side zero padding is allowed. Low-side, symmetric, reflected, nonzero, implicit, or runtime-selected padding is refused. Every source voxel must be covered. Window indices, coordinates, padded ROI bytes, input-region digests, blend identity, and traversal must match the pinned ledger exactly.

Constant weights are float64 ones. Gaussian weights use the versioned formula and rational sigma scale in the manifest. Weight-map identity is exact. The analytical unit-test tolerance of `1e-12` is not an output-drift acceptance threshold.

## Timing stages

Each arm records these non-overlapping stages in integer nanoseconds from a named monotonic clock:

1. `load`: local artifact reads and checksum validation.
2. `normalize`: declared modality preprocessing.
3. `window_enumeration`: schedule, padding, and ledger construction.
4. `h2d`: host-to-device transfer.
5. `inference`: model or engine execution.
6. `blending`: weighted accumulation and normalization.
7. `postprocess`: fixed thresholding and mask construction.
8. `write`: evidence serialization and checksum finalization.

Unavailable or unexecuted stages have `value: null`, `available: false`, and an explicit reason. PR 1 binds all eight records to the indexed `timing-provenance.json` file by path and hash. Values are never estimated or replaced by zero.

## Exactness and drift metrics

All regions use the inclusive fixed threshold `probability >= 0.5`. Outputs must satisfy `ET subset TC subset WT`.

Same-runtime repeats under an identical pinned environment are expected to produce byte-identical canonical probability and mask hashes. A mismatch is recorded as `reference_nondeterministic` and disqualifies the reference. Repeats are not averaged.

Every output identity binds the exact volume identity, model identity, window ledger, run, and arm hashes or identifiers. A valid but substituted output is refused before comparison.

Cross-runtime comparisons report:

- Probability byte equality.
- Changed probability element count and rate.
- Mean and maximum absolute probability error.
- Spatial voxels changed in any region.
- Per-region changed voxel count and rate.
- Positive-to-negative and negative-to-positive threshold flips.
- Dice for TC, WT, and ET.
- Candidate-minus-reference volume change in voxels and cubic millimetres.
- 26-connected component counts, sizes, additions, and removals.
- Surface Dice at a recorded metric tolerance.
- Symmetric HD95 in millimetres.
- Maximum symmetric boundary displacement in millimetres.

Surface voxels are foreground voxels removed by one 6-connected erosion. Distances are between surface-voxel centres using physical spacing. Both-empty binary Dice is `1.0`. One-empty binary Dice is `0.0`. If either surface is empty, surface Dice, HD95, and maximum boundary displacement are `null` and unavailable with a specific reason. Relative volume change is `null` when reference volume is zero.

PR 1 does not define pass thresholds for real segmentation drift. Such thresholds remain `null` until a blinded pilot supports a prospective amendment.

## Stopping and refusal gates

Stop before inference or refuse the affected result if any of these occurs:

- Model, data, license, terms, or runtime identity is incomplete.
- Subject non-overlap is unproven and a diagnostic claim is requested.
- An input contains PHI, secrets, unexpected metadata, nonfinite values, shape disagreement, or modality disagreement.
- The affine, spacing, channel order, threshold policy, preprocessing, ROI, overlap, padding, traversal, blend map, engine, or ledger differs from the manifest.
- Window indices or coordinates are missing, duplicated, reordered, out of bounds without declared high-side padding, or leave a source gap.
- ET is not a subset of TC or TC is not a subset of WT.
- A checksum, canonical encoding, closed bundle file set, or same-runtime repeat fails.
- Evidence is missing, malformed, unreadable, noncanonical, or internally inconsistent.
- The projected exploratory cost would exceed USD 20.

A refusal produces an explicit failure state. It never produces a success receipt, a default zero metric, or a partial result presented as complete.

## Budget and retry rule

The first real exploratory run has a hard total cloud GPU cap of USD 20. It requires a GPU with at least 24 GB of memory. There is no automatic retry, automatic instance replacement, automatic scale-up, or unattended spend. Any failed attempt stops the study and requires review of its receipt before a new authorization.

## Publication and lineage rules

The intended public data source is Medical Segmentation Decathlon Task01 BrainTumour under CC BY-SA 4.0. It derives from BraTS 2016 and 2017. The intended model was pretrained on BraTS 2018. Subject overlap is therefore considered unresolved until independently proven otherwise.

Without hash-bound proof of subject non-overlap, publications may report only output-preservation evidence against the pinned FP32 reference. They must not report diagnostic accuracy, clinical safety, clinical equivalence, radiologist equivalence, or patient-specific validity. Negative results, refusals, unavailable metrics, protocol deviations, and all tested arms must be reported. Selective omission of failing regions, components, or arms is prohibited.

## Malformed or missing evidence

The verifier rejects missing files, extra files, symlinks, unsafe paths, changed sizes, changed hashes, duplicate JSON keys, noncanonical JSON, unknown schema fields, unsupported versions, nonfinite arrays, invalid masks, inconsistent receipts, and any manifest attempt to rewrite the independently pinned PR 1 evidence plan. Missing metrics remain unavailable with reasons. They are not imputed. A malformed report cannot support a claim.

## Confirmatory expansion

Expansion beyond one volume requires all feasibility gates to pass, independent review of the complete evidence bundle, a frozen analysis script, a prospective sample-size rationale, pinned eligibility criteria, a lineage determination, declared multiplicity handling, and a protocol amendment committed before additional outputs are inspected. Confirmatory work remains research-only.
