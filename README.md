# VoxelScope

VoxelScope is research infrastructure for evidence-preserving GPU inference in 3D brain tumor MRI segmentation.

Its thesis is precise:

> Can we accelerate 3D sliding-window brain tumor MRI segmentation without moving tumor boundaries, dropping small regions, or changing the final mask?

VoxelScope is research-only software. It does not provide diagnosis, treatment, patient advice, clinical safety claims, radiologist equivalence, or patient-specific validity.

## Current status

Milestone 4 acquired the eight objects in the approved OpenNeuro `ds007045` v2.0.1 plan into owner-private custody through one no-retry attempt. All expected sizes and source-published hashes matched. The five medical files passed the preregistered bounded NIfTI structure, finite-value, mask-domain, and exact cross-file geometry gates. The official MONAI `brats_mri_segmentation` 0.5.2 archive remains separately verified without extraction or model loading.

The project is still **NO-GO** for real-data inference. Structural GO makes the selected input eligible only for the next reviewed stage. Preprocessing adapter equivalence, source-label versus model-output semantics, model-training overlap, runtime identity, host and budget authorization, and the explicit inference start gate remain unresolved. The project has not loaded a model, run inference, used a GPU, provisioned cloud resources, authorized spend, or measured diagnostic accuracy.

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
  custody.py      Private acquisition, archive verification, and public scans
  cli.py         Offline command-line interface
```

Arrays use `(C, Z, Y, X)` layout. Windows use zero-based half-open coordinates `[start, end)`. Traversal is lexicographic `Z, Y, X`, with `X` changing fastest. Overlap is stored as an exact rational relative to ROI size. The only permitted padding is explicit zero padding on the high side of an axis.

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

See [the real-data readiness record](docs/real-data-readiness-v1.md), [active one-volume feasibility protocol](docs/one-volume-feasibility-protocol-v2.md), [historical protocol v1](docs/one-volume-feasibility-protocol-v1.md), and [private custody workflow](docs/private-custody-workflow-v1.md) before proposing real-data work.

## Next gate

Specify and independently review the exact preprocessing adapter without loading the model. It must bind channel stacking, source geometry handling, nonzero channel-wise normalization, output inversion, and the separation between source labels `0/1/2/3` and model output labels `0/1/2/4`. Model-training overlap must remain unresolved or be explicitly bounded. No inference, GPU provisioning, or spend is authorized.

## License

VoxelScope is licensed under Apache-2.0. Future model and dataset artifacts retain their own licenses and terms and are not distributed by this repository.
