# VoxelScope

VoxelScope is research infrastructure for evidence-preserving GPU inference in 3D brain tumor MRI segmentation.

Its thesis is precise:

> Can we accelerate 3D sliding-window brain tumor MRI segmentation without moving tumor boundaries, dropping small regions, or changing the final mask?

VoxelScope is research-only software. It does not provide diagnosis, treatment, patient advice, clinical safety claims, radiologist equivalence, or patient-specific validity.

## Current status

Milestone 4 acquired the eight objects in the approved OpenNeuro `ds007045` v2.0.1 plan into owner-private custody through one no-retry attempt. All expected sizes and source-published hashes matched. The five medical files passed the preregistered bounded NIfTI structure, finite-value, mask-domain, and exact cross-file geometry gates. Milestone 5 then consumed one explicit offline approval for the exact preprocessing adapter. Four image channels were processed in T1c, T1, T2, and FLAIR order by two independent implementations. The private outputs passed the frozen tolerance, finite-output, geometry-preservation, label-exclusion, closed-snapshot, and replay checks.

Milestone 6 now proves the public positional bridge from preprocessing `(C,I,J,K)` to model input `(N,C,D,H,W)`: `I -> D`, `J -> H`, and `K -> W`, with no transpose, reorientation, resampling, affine relabeling, or anatomical-direction claim. The pinned ROI is `(240,240,160)`, overlap is one half, traversal is lexicographic `I,J,K` with `K` changing fastest, and final anchors and coverage agree with an independent oracle across asymmetric synthetic cases. MONAI's symmetric padding and crop behavior are pinned separately from VoxelScope's legacy high-side-only padding.

The project is still **NO-GO** for real-data inference. The legacy engine's unpadded positional enumeration is GO, but its general padded extraction is not equivalent to MONAI and must not be reused for real inference. Source-label versus model-output semantics, model-training overlap, runtime image, host and budget authorization, and the explicit inference start gate also remain unresolved. The official MONAI `brats_mri_segmentation` 0.5.2 archive remains verified without extraction or model loading. The project has not loaded a model, run inference, used a GPU, provisioned cloud resources, authorized spend, or measured diagnostic accuracy.

The current package provides:

- Strict canonical JSON and raw-array content hashes.
- Typed study, volume, model, window, timing, output, drift, receipt, and refusal records.
- Deterministic sliding-window enumeration in `(Z, Y, X)` order.
- Explicit high-side zero padding for volumes smaller than the ROI.
- Constant and Gaussian blend-map identities.
- Synthetic probability and nested `TC`, `WT`, and `ET` masks.
- Dice, connected-component, threshold-flip, surface Dice, HD95, and maximum boundary-distance auditing.
- Complete evidence bundles with a closed file allowlist and tamper detection.
- A canonical official source registry and acquisition plan.
- Strict typed custody records with an exact network and redirect allowlist.
- Owner-private, no-clobber model acquisition and receipt generation.
- Archive path, symlink, encryption, file-count, size, compression-ratio, closed-member, and license checks without extraction.
- A repository scan for model weights, medical images, subject identifiers, credentials, private paths, and private receipts.
- A role-driven, no-spatial-transform preprocessing adapter with a one-shot authorization bound to the adapter plan and custody receipt.
- Direct float32 NumPy normalization and an independent chunked Welford implementation with a frozen comparison tolerance.
- A source-pinned positional bridge from neutral `I,J,K` axes to model `D,H,W` positions.
- Independent final-anchor, traversal, coverage, symmetric-padding, crop, and legacy-bundle checks over asymmetric synthetic fixtures.
- A closed milestone 6 public evidence bundle that records the legacy general padded path as NO-GO.

## Architecture

```text
src/voxelscope/
  canonical.py   Canonical JSON, path safety, and SHA-256
  arrays.py      Canonical little-endian array artifacts
  records.py     Versioned typed evidence contracts
  synthetic_contract.py  Trusted PR 1 fixture identities and evidence plan
  windows.py     Sliding-window ledger, padding, blend map, and coverage oracle
  drift.py       Nested-region validation and boundary-drift metrics
  bundle.py      Closed bundle indexing and semantic verification
  fixtures.py    Deterministic synthetic evidence generation
  custody_records.py  Versioned source and acquisition records
  real_data_contract.py  Trusted milestone 2 source and plan pins
  one_volume_records.py  Typed milestone 3 source-decision evidence
  one_volume_contract.py  Trusted milestone 3 decision pin
  one_volume.py  Offline decision verification and rendering
  one_volume_custody_records.py  Private acquisition and structural records
  one_volume_custody.py  One-shot acquisition and bounded NIfTI validation
  milestone4_evidence.py  Sanitized closed public evidence
  preprocessing_records.py  Typed adapter plan, report, and refusal records
  preprocessing.py  One-shot dual-implementation private preprocessing
  milestone5_evidence.py  Sanitized preprocessing evidence
  window_bridge_records.py  Strict positional bridge plan, report, and refusal records
  window_bridge.py  Independent window oracle and MONAI positional padding proof
  window_bridge_contract.py  Trusted milestone 6 plan and report identities
  milestone6_evidence.py  Sanitized positional bridge evidence
  custody.py      Private acquisition, archive verification, and public scans
  cli.py         Offline command-line interface
```

Historical synthetic evidence arrays use `(C, Z, Y, X)` layout. Those letters are synthetic coordinate names, not anatomical directions. Milestone 6 binds the unchanged preprocessing positions `(I,J,K)` to window positions `(S0,S1,S2)` and model positions `(D,H,W)` without renaming them anatomically. Windows use zero-based half-open coordinates `[start, end)`, exact rational overlap, and last-axis-fastest traversal. The historical engine still uses explicit high-side zero padding; MONAI 1.4.0 uses symmetric constant-zero padding for undersized inputs. The policies are not generally equivalent.

## Quickstart

Prerequisites are Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Dependency installation may access the configured package index. All VoxelScope commands are offline after dependencies are present.

```bash
uv sync --frozen
make test
make lint
make typecheck
make demo
make custody-validate
make build
```

The CLI commands are:

```bash
voxelscope fixture build --output build/evidence
voxelscope windows build --manifest build/evidence/study-manifest.json --output build/windows
voxelscope verify --bundle build/evidence
voxelscope drift compare \
  --reference build/evidence/outputs/reference/output.json \
  --candidate build/evidence/outputs/one-voxel-boundary/output.json \
  --output build/boundary-report.json
voxelscope source verify --registry research/source-registry-v1.json
voxelscope source one-volume \
  --record research/one-volume-source-decision-v1.json \
  --plan research/one-volume-acquisition-plan-v1.json
voxelscope custody plan \
  --registry research/source-registry-v1.json \
  --plan research/acquisition-plan-v1.json
voxelscope milestone4-public verify --bundle research/milestone-4
voxelscope milestone5-public verify --bundle research/milestone-5
python -m voxelscope.milestone6_cli window-bridge \
  --plan research/window-bridge-plan-v1.json \
  --report research/window-bridge-report-v1.json
python -m voxelscope.milestone6_cli public-verify --bundle research/milestone-6
voxelscope preprocess execute --help
voxelscope preprocess verify --help
voxelscope milestone5-public verify --help
voxelscope custody scan-public --root .
```

`fixture build`, `windows build`, and `drift compare` refuse to overwrite an existing destination.
All custody commands are offline unless `custody acquire` receives `--allow-network`. Acquisition is restricted to the exact source and redirect origins in the canonical plan, requires known size and content hashes, refuses overwrite, never extracts archives, and writes receipts only under an owner-private root outside the repository. Private custody commands currently fail closed on Windows because restrictive ACL verification is not implemented.

## Evidence model

A generated bundle contains synthetic modalities, volume and model identities, a deterministic window ledger, output identities, scenario drift reports, explicit refusal records, unavailable timing records with hash-bound provenance, a run receipt, and a readable summary. Every output identity binds the exact volume identity, model identity, window ledger, run, and arm. `bundle.json` indexes every payload file. `bundle.sha256` binds that index. The study manifest records the expected output IDs, report comparisons, and refusal IDs, while the verifier independently pins the complete PR 1 plan and required artifact paths in trusted code. Verification rejects missing, extra, changed, noncanonical, path-traversing, or symlinked evidence.

Same-runtime deterministic repeats are expected to have byte-identical canonical probability and mask hashes. Cross-runtime outputs may differ, so VoxelScope records exact probability changes, threshold flips, mask changes, region Dice, volume change, component changes, and physical boundary distances. PR 1 sets no acceptance threshold for real segmentation drift.

## Data lineage rule

The pinned MONAI model was trained on BraTS 2018. The selected OpenNeuro cohort reports acquisitions from 2018 through 2025 and is described as a separate collection, but neither the selected case acquisition date nor the model's exact training roster is available in the public contract. Subject overlap is therefore unresolved. VoxelScope fails closed: no diagnostic-accuracy claim is allowed unless non-overlap is independently proven and hash-bound. Without that proof, only output preservation against a pinned FP32 reference may be reported.

## Explicit non-claims

VoxelScope does not currently claim that:

- Any accelerated backend preserves a real medical segmentation.
- Any output is diagnostically accurate.
- The privately acquired OpenNeuro artifacts are approved for model loading or inference.
- Any model, dataset, or output has clinical approval.
- Any real subject is independent from model training data.
- Any numerical drift threshold is clinically acceptable.
- Any backend is safe for clinical or patient-specific use.

See [the positional window bridge record](docs/milestone-6-window-bridge-readiness-v1.md), [the preprocessing adapter contract](docs/preprocessing-adapter-v1.md), [real-data readiness record](docs/real-data-readiness-v1.md), [active one-volume feasibility protocol](docs/one-volume-feasibility-protocol-v2.md), [historical protocol v1](docs/one-volume-feasibility-protocol-v1.md), and [private custody workflow](docs/private-custody-workflow-v1.md) before proposing real-data work.

## Next gate

Prospectively define and independently review a private window-execution adapter that consumes the approved preprocessing snapshot, applies the pinned symmetric positional padding and crop contract, and emits a closed private window ledger without loading the model. It must not route real data through the legacy high-side padded path. Model-training overlap and source-label versus model-output semantics remain unresolved. No model load, inference, GPU provisioning, or spend is authorized.

## License

VoxelScope is licensed under Apache-2.0. Future model and dataset artifacts retain their own licenses and terms and are not distributed by this repository.
