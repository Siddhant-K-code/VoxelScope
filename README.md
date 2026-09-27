# VoxelScope

VoxelScope is research infrastructure for evidence-preserving GPU inference in 3D brain tumor MRI segmentation.

Its thesis is precise:

> Can we accelerate 3D sliding-window brain tumor MRI segmentation without moving tumor boundaries, dropping small regions, or changing the final mask?

VoxelScope is research-only software. It does not provide diagnosis, treatment, patient advice, clinical safety claims, radiologist equivalence, or patient-specific validity.

## Current status

PR 1 is an offline synthetic foundation. It defines the evidence contracts needed before a one-volume MONAI BraTS feasibility run. It does not download medical data or model weights, execute MONAI or TensorRT, use a GPU, provision cloud resources, or measure diagnostic accuracy.

The current package provides:

- Strict canonical JSON and raw-array content hashes.
- Typed study, volume, model, window, timing, output, drift, receipt, and refusal records.
- Deterministic sliding-window enumeration in `(Z, Y, X)` order.
- Explicit high-side zero padding for volumes smaller than the ROI.
- Constant and Gaussian blend-map identities.
- Synthetic probability and nested `TC`, `WT`, and `ET` masks.
- Dice, connected-component, threshold-flip, surface Dice, HD95, and maximum boundary-distance auditing.
- Complete evidence bundles with a closed file allowlist and tamper detection.

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
```

`fixture build`, `windows build`, and `drift compare` refuse to overwrite an existing destination.

## Evidence model

A generated bundle contains synthetic modalities, volume and model identities, a deterministic window ledger, output identities, scenario drift reports, explicit refusal records, unavailable timing records with hash-bound provenance, a run receipt, and a readable summary. Every output identity binds the exact volume identity, model identity, window ledger, run, and arm. `bundle.json` indexes every payload file. `bundle.sha256` binds that index. The study manifest records the expected output IDs, report comparisons, and refusal IDs, while the verifier independently pins the complete PR 1 plan and required artifact paths in trusted code. Verification rejects missing, extra, changed, noncanonical, path-traversing, or symlinked evidence.

Same-runtime deterministic repeats are expected to have byte-identical canonical probability and mask hashes. Cross-runtime outputs may differ, so VoxelScope records exact probability changes, threshold flips, mask changes, region Dice, volume change, component changes, and physical boundary distances. PR 1 sets no acceptance threshold for real segmentation drift.

## Data lineage rule

The planned public dataset derives from earlier BraTS releases, while the planned MONAI model was trained on BraTS 2018. Subject overlap is therefore unresolved by default. VoxelScope fails closed: no diagnostic-accuracy claim is allowed unless subject non-overlap is independently proven and hash-bound. Without that proof, only output preservation against a pinned FP32 reference may be reported.

## Explicit non-claims

VoxelScope does not currently claim that:

- Any accelerated backend preserves a real medical segmentation.
- Any output is diagnostically accurate.
- Any model or dataset identity has been approved or acquired.
- Any real subject is independent from model training data.
- Any numerical drift threshold is clinically acceptable.
- Any backend is safe for clinical or patient-specific use.

See [the preregistration](docs/preregistration-v0.md) and [future execution runbook](docs/future-execution-runbook.md) before proposing real-data work.

## License

VoxelScope is licensed under Apache-2.0. Future model and dataset artifacts retain their own licenses and terms and are not distributed by this repository.
