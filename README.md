# VoxelScope

VoxelScope is research infrastructure for evidence-preserving GPU inference in 3D brain tumor MRI segmentation.

Its thesis is precise:

> Can we accelerate 3D sliding-window brain tumor MRI segmentation without moving tumor boundaries, dropping small regions, or changing the final mask?

VoxelScope is research-only software. It does not provide diagnosis, treatment, patient advice, clinical safety claims, radiologist equivalence, or patient-specific validity.

## Current status

Milestone 4 acquired the eight objects in the approved OpenNeuro `ds007045` v2.0.1 plan into owner-private custody through one no-retry attempt. All expected sizes and source-published hashes matched. The five medical files passed the preregistered bounded NIfTI structure, finite-value, mask-domain, and exact cross-file geometry gates. Milestone 5 then consumed one explicit offline approval for the exact preprocessing adapter. Four image channels were processed in T1c, T1, T2, and FLAIR order by two independent implementations. The private outputs passed the frozen tolerance, finite-output, geometry-preservation, label-exclusion, closed-snapshot, and replay checks.

Milestone 6 now proves the public positional bridge from preprocessing `(C,I,J,K)` to model input `(N,C,D,H,W)`: `I -> D`, `J -> H`, and `K -> W`, with no transpose, reorientation, resampling, affine relabeling, or anatomical-direction claim. The pinned ROI is `(240,240,160)`, overlap is one half, traversal is lexicographic `I,J,K` with `K` changing fastest, and final anchors and coverage agree with an independent oracle across asymmetric synthetic cases. MONAI's symmetric padding and crop behavior are pinned separately from VoxelScope's legacy high-side-only padding.

Milestone 7 then completed one explicitly approved offline private window qualification. It placed a one-shot marker before the private preprocessing read, bound the exact private Milestone 5 report and snapshot identities only inside private authorization and terminal evidence, staged the reference tensor immutably, and compared every window byte-for-byte through two independent paths: full symmetric padding followed by slicing, and direct source intersection with local zero padding. The result and a fresh-descriptor verification were GO; no windows were persisted. The committed public evidence remains synthetic-only, and no private identity, geometry, shape, count, hash, or path is published.

Milestone 8 adds a prospective private model archive extraction and CPU loading gate. Runtime preflight proved the original plan incorrectly treated the PyTorch `v2.4.0` tag commit as the official CPU wheel's source commit, so that plan is superseded and mechanically unexecutable. The corrected v2 plan preserves the tag commit and separately binds the wheel source commit, exact Linux arm64/Python 3.12.14 runtime-build contract, official direct wheel identities, reviewed SegResNet constructor, and prior gate identities. The one-shot executor still requires a separately approved interpreter and closed runtime-distribution fingerprint.

The project is still **NO-GO** for private model qualification and real-data inference. Milestone 8 implementation did not access or extract the private archive, deserialize its checkpoint, instantiate or load the official model, run a forward pass, allocate medical-shaped input, use an accelerator or network, provision cloud resources, or authorize spend. A later owner action must separately approve the final exact Milestone 8 plan, private receipt/archive identities, exact interpreter, and reviewed closed PyTorch/MONAI runtime fingerprint through an inherited descriptor. Source-label versus model-output semantics, model-training overlap, inference runtime/host/budget authorization, and the explicit inference start gate also remain unresolved.

The current package provides:

- Strict canonical JSON and raw-array content hashes.
- Deterministic synthetic glioblastoma protein evidence cards with replay receipts.
- A synthetic evidence communication compiler with typed model drafts, independent
  claim verification, canonical prose, refusal receipts, and an offline local-model
  benchmark.
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
- A one-shot private window adapter with immutable staging and a single completion-or-refusal terminal path.
- Independent full-padding and direct-intersection materializers with exact streamed byte comparison.
- A bounded private report/ledger design that persists no windows.
- A closed milestone 7 public evidence bundle containing synthetic fixture statuses and counts only.
- A one-shot, allowlist-only private model extraction and CPU loading qualification gate.
- A fixed reviewed SegResNet constructor with strict weights-only state validation and no forward path.
- An isolated Linux/Python 3.12/PyTorch 2.4.0/MONAI 1.4.0 qualification requirements artifact.
- A closed milestone 8 public evidence bundle containing synthetic protocol results and explicit non-actions only.

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
  gbm_atlas.py   Synthetic protein evidence cards and replay receipts
  gbm_atlas_cli.py  Separate offline synthetic atlas command-line interface
  evidence_communication_records.py  Strict communication and receipt schemas
  evidence_communication.py  Deterministic claim verifier, compiler, and benchmark
  evidence_communication_cli.py  Separate offline communication command-line interface
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
  window_execution_records.py  Strict private adapter plan, report, ledger, completion, and refusal records
  window_execution.py  One-shot dual-path private window materialization qualification
  window_execution_contract.py  Trusted milestone 5, 6, 7, and legacy identities
  milestone7_evidence.py  Synthetic-only closed public implementation evidence
  milestone7_cli.py  Separate offline milestone 7 command-line interface
  model_loading_records.py  Strict Milestone 8 plan, worker, private, and terminal records
  model_loader_protocol.py  Self-contained typed inherited-FD worker protocol
  model_loading.py  One-shot allowlist extraction and isolated CPU loader orchestration
  model_loader_worker.py  Weights-only fixed-constructor CPU loader subprocess
  milestone8_evidence.py  Synthetic-only closed model-loading evidence
  milestone8_cli.py  Separate offline Milestone 8 command-line interface
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
python -m voxelscope.gbm_atlas_cli build \
  --manifest research/gbm-evidence-atlas-v1/source-manifest.json \
  --output build/gbm-atlas
python -m voxelscope.evidence_communication_cli benchmark \
  --atlas build/gbm-atlas/atlas.json \
  --fixtures research/gbm-evidence-communication-benchmark-v2/benchmark-fixtures.json \
  --runner recorded \
  --output build/evidence-communication-benchmark
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
python -m voxelscope.milestone7_cli synthetic-qualify
python -m voxelscope.milestone7_cli public-verify --bundle research/milestone-7
python -m voxelscope.milestone7_cli private-execute --help
python -m voxelscope.milestone7_cli private-verify --help
python -m voxelscope.milestone8_cli synthetic-qualify
python -m voxelscope.milestone8_cli public-verify --bundle research/milestone-8-v2
python -m voxelscope.milestone8_cli private-execute --help
python -m voxelscope.milestone8_cli private-verify --help
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

## Synthetic glioblastoma evidence atlas

The bundled atlas fixture is synthetic and contains no imaging or individual-level records. The command validates a versioned source manifest, normalizes gene and protein identifiers with source-bound provenance, and writes canonical `atlas.json` and `receipt.json` files. Cards expose agreement, disagreement, missing modalities, unsupported joins, and source freshness. They do not rank proteins.

The implementation fits the existing [architecture](#architecture) and canonical evidence model. Repeating the command with unchanged inputs produces byte-identical output.

```bash
uv run python -m voxelscope.gbm_atlas_cli build \
  --manifest research/gbm-evidence-atlas-v1/source-manifest.json \
  --output build/gbm-atlas
```

Sample command output:

```text
gbm_atlas_status=built atlas_sha256=5a9f1a2078d951b3f10ff909b5c341e4071b5ed92050283c31859521b8c88647 receipt_sha256=abd5bb0407ff5df345489b110a87be8e0c83cc2aeb3a21853f355eed4ee69ae0
synthetic_only=true ranking_performed=false
```

The receipt records every source digest, `voxelscope/gbm-evidence-transform/v1`, exclusions, warnings, and the canonical atlas digest. Ambiguous identifier mappings stop the build. Unmapped identifiers remain explicit exclusions.

## Evidence communication compiler

The synthetic evidence communication compiler lets a local model propose only typed,
source-cited claims. Deterministic code derives the required facts and caveats,
verifies every field against the source card, excludes unsupported claims, compiles
canonical researcher-facing prose, and closes a receipt. The model cannot authorize
its own claims or supply accepted final prose. Trusted verified claims contain only
canonical typed fields and deterministic identities, never raw model text.

The committed benchmark runs offline without a model installation or network. It
includes accepted, partially excluded, and refused recorded drafts. A localhost-only
Ollama adapter can evaluate an already installed model without adding a mandatory
dependency or downloading model files. Envelope noncompliance is retained as a
digest-safe refused result instead of aborting the remaining runs.

The frozen v2 recorded result emits `122 / 158` required facts and `116 / 148`
required caveats. Its independently verified drafts cover `158 / 158` facts and
`148 / 148` caveats. Invalid model output is `0 / 18`, and semantic variance is
`1 / 9`.

See the
[evidence communication compiler and study protocol](docs/evidence-communication-compiler-v2.md)
for trust boundaries, exact commands, metric denominators, local-runner setup, and
measurement limitations.

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

See [the corrected model-loading readiness record](docs/milestone-8-model-loading-readiness-v2.md), [the positional window bridge record](docs/milestone-6-window-bridge-readiness-v1.md), [the preprocessing adapter contract](docs/preprocessing-adapter-v1.md), [real-data readiness record](docs/real-data-readiness-v1.md), [active one-volume feasibility protocol](docs/one-volume-feasibility-protocol-v2.md), [historical protocol v1](docs/one-volume-feasibility-protocol-v1.md), and [private custody workflow](docs/private-custody-workflow-v1.md) before proposing real-data work.

The [Glioblastoma Evidence Atlas product contract](docs/glioblastoma-evidence-atlas-v0.md) extends the executable synthetic slice merged in PR #12 at commit `88d074b9116d913ce720bd1a5e72a7e24933110f`. Neither the merged slice nor the future phases implement production source ingestion, authorize current private assets for a new use, or change an existing NO-GO gate.

The proposed Glioblastoma Evidence Atlas has a separate [open-data source qualification and first-study preregistration](docs/research/glioblastoma-evidence-atlas-source-qualification-v1.md). It qualifies molecular sources without authorizing controlled MRI, downloading patient data, or changing the existing inference milestones.

## Next gate

Obtain a separate explicit owner approval bound to the final Milestone 8 plan SHA-256 and reviewed private model custody receipt/archive identities, then run the private CPU-only model-loading qualification exactly once. That approval authorizes only the closed extraction and stop-after-load qualification; it does not authorize model input, forward, inference, GPU/MPS, network, cloud, or spend. Model-training overlap and source-label versus model-output semantics remain unresolved.

## License

VoxelScope is licensed under Apache-2.0. Future model and dataset artifacts retain their own licenses and terms and are not distributed by this repository.
