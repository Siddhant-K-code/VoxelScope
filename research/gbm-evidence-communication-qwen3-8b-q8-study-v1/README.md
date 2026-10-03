# Qwen3 8B Q8_0 evidence communication study v1

## Prospective status

This is the prospectively frozen protocol for the first observed local-model
Evidence Communication Compiler study. At declaration time, no `/api/generate`
request had been issued, all result fields were **pending/unobserved**, and the
benchmark output directory did not exist.

The study is synthetic and non-clinical. It uses no real biomedical, patient, or MRI
data and cannot support diagnosis, treatment, biological, clinical, causal,
protein-ranking, therapeutic-target, or druggability claims.

## Frozen identity

| Field | Declaration |
|---|---|
| Declaration time | `2026-10-03T08:56:41Z` |
| Base/compiler commit | `0b83e57ac97a2fe5a8d133947caebb79df5c5d5f` |
| Base tree | `204a8a0da7ac6c781f8961c96b0fbd86ed137e2c` |
| Atlas manifest SHA-256 | `edce8ad2a0748d1b95cf6c58ad9cce7c155e7d66fbb5d69985d8ed387fc3ba3f` |
| Evidence items SHA-256 | `9f31d4d2899672e4aec3a803867ea0c50dd29bb172a69036e16ee3470e48e2bd` |
| Identifier map SHA-256 | `2d6d32635f480529e3924ee76bd4d3b8971ef487ef1f21537c1851891399394a` |
| Benchmark fixture SHA-256 | `f461dc85f91a6bb5605de692641e66d89d688a61ec0cce37e16620820caee599` |
| Endpoint/runtime | `http://127.0.0.1:11434`; Ollama `0.35.1` |
| Model identity | `qwen3:8b-q8_0`; manifest `e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130` |
| Observed local metadata | `8,851,089,538` bytes; Qwen3; `8.2B`; `Q8_0`; context limit `40,960`; embedding length `4,096` |
| Compiler controls | temperature `0.0`; strict JSON; thinking disabled with top-level `think=false` |
| Request controls | context `16,384`; timeout `300` seconds; seed intentionally unset |
| Run plan | fixture-defined two repeats over nine cases; 18 requests; no warmup; no selective rerun |

The seed is intentionally unset so the two repeats measure observed local-runtime
semantic variance at temperature zero. The largest exact benchmark prompt was
`11,071` UTF-8 bytes and the largest recorded 16-claim envelope was `9,205` bytes,
both measured without inference. These byte sizes are the prospective rationale for
the `16,384` context setting; they are not model token measurements.

The declared host is a MacBook Pro with Apple M5 Pro, 18 cores (6 performance and
12 efficiency), 24 GB unified memory, and macOS 27.0.1 build 26A434. The runner will
receive:

```text
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
HOST_MACHINE=MacBook-Pro
HOST_CHIP=Apple-M5-Pro
HOST_CPU_CORES=18
HOST_UNIFIED_MEMORY_GB=24
HOST_OS=macOS-27.0.1-26A434
```

The Ollama service process environment was inspected before session creation and
contained exactly the two declared `OLLAMA_*` settings. The Ollama API does not attest
process environment, so the runner records every setting as `declared_unverified`.

## Exact procedure

The declaration commit must be committed, pushed, and confirmed on the remote branch
before the first `/api/generate` request. Build the atlas offline:

```bash
uv run python -m voxelscope.gbm_atlas_cli build --manifest research/gbm-evidence-atlas-v1/source-manifest.json --output build/gbm-atlas
```

Run the benchmark exactly once:

```bash
uv run python -m voxelscope.evidence_communication_cli benchmark --atlas build/gbm-atlas/atlas.json --fixtures research/gbm-evidence-communication-benchmark-v3/benchmark-fixtures.json --runner ollama --endpoint http://127.0.0.1:11434 --model qwen3:8b-q8_0 --model-digest e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130 --runtime-version 0.35.1 --thinking disabled --timeout-seconds 300 --num-ctx 16384 --declared-environment OLLAMA_FLASH_ATTENTION=1 --declared-environment OLLAMA_KV_CACHE_TYPE=q8_0 --declared-environment HOST_MACHINE=MacBook-Pro --declared-environment HOST_CHIP=Apple-M5-Pro --declared-environment HOST_CPU_CORES=18 --declared-environment HOST_UNIFIED_MEMORY_GB=24 --declared-environment HOST_OS=macOS-27.0.1-26A434 --output research/gbm-evidence-communication-qwen3-8b-q8-study-v1/benchmark
```

Immediately replay the published bundle offline:

```bash
uv run python -m voxelscope.evidence_communication_cli benchmark-replay --atlas build/gbm-atlas/atlas.json --fixtures research/gbm-evidence-communication-benchmark-v3/benchmark-fixtures.json --bundle research/gbm-evidence-communication-qwen3-8b-q8-study-v1/benchmark
```

The atlas, fixture, compiler, verifier, thresholds, schemas, and synthetic atlas are
not changed by this study. The complete generated v3 benchmark tree is preserved
unchanged after publication.

## Stop and retry policy

- Stop before inference if the declaration commit cannot be pushed and verified on
  the remote branch.
- If transport, outer protocol, timeout, runtime identity, tag resolution, or model
  manifest identity fails or drifts, abort without retry and report the incident.
- Do not selectively rerun failed, refused, or malformed cases and do not alter this
  declaration after observing output.
- Strict JSON or envelope noncompliance is a measured `invalid_model_output` refusal,
  not a selective-rerun opportunity.
- Do not download a model during the study.

## Result record

| Field | Prospective state |
|---|---|
| Benchmark execution | `pending/unobserved` |
| Offline replay | `pending/unobserved` |
| Per-run terminal states | `pending/unobserved` |
| Compiler-defined metrics | `pending/unobserved` |
| Latency and token summaries | `pending/unobserved` |
| Peak memory and Metal memory | `pending/unobserved` |
| Benchmark report SHA-256 | `pending/unobserved` |
| Benchmark receipt SHA-256 | `pending/unobserved` |
| Protocol incident | `pending/unobserved` |

Observed results may be appended to this document only after the unchanged declaration
commit is remotely verifiable. Negative, refused, malformed, and unavailable outcomes
must be reported without reinterpretation.

## Observed result

Declaration commit
`ee7642b3b96c0e9a81afca42827953e3ceb56257` was committed at
`2026-10-03T14:27:42+05:30`, pushed, and confirmed as the exact remote branch head
before the first inference request. The frozen command then completed exactly once,
published all 18 runs, and reported `local_model_downloaded=false`. The immediate
offline replay verified all 18 runs and the closed receipt. No transport, outer
protocol, timeout, runtime-identity, model-identity, or receipt-closure incident
occurred.

All 18 runs were refused. Four runs preserved digest-safe `invalid_json_type`
outcomes; the other 14 parsed envelopes were deterministically refused. No failed,
refused, or malformed case was rerun.

### Compiler-defined metrics

| Metric | Observed result |
|---|---:|
| Unsupported-claim rate | `56 / 238 = 0.23529411764705882` |
| Emitted fact coverage | `0 / 158 = 0.0` |
| Emitted caveat coverage | `0 / 148 = 0.0` |
| Verified-draft fact coverage | `102 / 158 = 0.6455696202531646` |
| Verified-draft caveat retention | `80 / 148 = 0.5405405405405406` |
| Evidence-citation validity | `342 / 342 = 1.0` |
| Replay semantic variance | `0 / 9 = 0.0` |
| Invalid-model-output rate | `4 / 18 = 0.2222222222222222` |
| Accepted or partially excluded | `0 / 18` |
| Refused | `18 / 18` |

No threshold was reinterpreted and no failure was rewritten. Refused runs contribute
zero emitted coverage under the compiler's pre-existing metric definition.

### Per-run states

| Case | Repeat | State | Outcome record | Semantic SHA-256 |
|---|---:|---|---|---|
| `supported-agreement` | 0 | `refused` | parsed envelope | `b6c844239b712324d179d4f3b07efaaebdb8956ea8748db17565f651a4ccb27a` |
| `supported-agreement` | 1 | `refused` | parsed envelope | `b6c844239b712324d179d4f3b07efaaebdb8956ea8748db17565f651a4ccb27a` |
| `cross-layer-disagreement` | 0 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |
| `cross-layer-disagreement` | 1 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |
| `missing-modality` | 0 | `refused` | `invalid_json_type` | `fc547a7657c2950480468461c33bf6f634f51126800983c31a3b1e3bef0cb72c` |
| `missing-modality` | 1 | `refused` | `invalid_json_type` | `fc547a7657c2950480468461c33bf6f634f51126800983c31a3b1e3bef0cb72c` |
| `unsupported-join` | 0 | `refused` | parsed envelope | `b6c844239b712324d179d4f3b07efaaebdb8956ea8748db17565f651a4ccb27a` |
| `unsupported-join` | 1 | `refused` | parsed envelope | `b6c844239b712324d179d4f3b07efaaebdb8956ea8748db17565f651a4ccb27a` |
| `stale-evidence` | 0 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |
| `stale-evidence` | 1 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |
| `restricted-evidence` | 0 | `refused` | `invalid_json_type` | `fc547a7657c2950480468461c33bf6f634f51126800983c31a3b1e3bef0cb72c` |
| `restricted-evidence` | 1 | `refused` | `invalid_json_type` | `fc547a7657c2950480468461c33bf6f634f51126800983c31a3b1e3bef0cb72c` |
| `not-measured-versus-negative` | 0 | `refused` | parsed envelope | `b6c844239b712324d179d4f3b07efaaebdb8956ea8748db17565f651a4ccb27a` |
| `not-measured-versus-negative` | 1 | `refused` | parsed envelope | `b6c844239b712324d179d4f3b07efaaebdb8956ea8748db17565f651a4ccb27a` |
| `source-reported-target-evidence` | 0 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |
| `source-reported-target-evidence` | 1 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |
| `prohibited-clinical-causal` | 0 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |
| `prohibited-clinical-causal` | 1 | `refused` | parsed envelope | `e3398c4d60f392f82b574a9c4cbc8c318126b3cc2f3ab45359be3e1cc14bd47f` |

Every parsed-envelope run was refused for the exact combination of missing required
directional and source-target requirements plus prohibited causal/certainty and
target/druggability language. The four invalid outputs were:

| Case | Repeat | Error | Output bytes | Output SHA-256 |
|---|---:|---|---:|---|
| `missing-modality` | 0 | `invalid_json_type` | 9,629 | `7b1032d646567d5ee81e0814c00a5848b87a331637ea209c88feb6b2503cd1b6` |
| `missing-modality` | 1 | `invalid_json_type` | 9,629 | `7b1032d646567d5ee81e0814c00a5848b87a331637ea209c88feb6b2503cd1b6` |
| `restricted-evidence` | 0 | `invalid_json_type` | 9,773 | `e768e6ff2b7d7b5a82ef48acc28b2ab18ea7d75a81aa3b0cfee291933d19d93a` |
| `restricted-evidence` | 1 | `invalid_json_type` | 9,773 | `e768e6ff2b7d7b5a82ef48acc28b2ab18ea7d75a81aa3b0cfee291933d19d93a` |

### Measurements

| Measurement | Count | Minimum | Mean | Maximum |
|---|---:|---:|---:|---:|
| Latency (ms) | 18 | `96784.58245799993` | `103526.30805777799` | `113156.43612499116` |
| Input tokens | 18 | `2800` | `2911.5555555555557` | `3069` |
| Output tokens | 18 | `2802` | `2918.1111111111113` | `2978` |

Peak process memory and peak Metal memory are unavailable with reason
`runner_did_not_report_peak_memory` and
`runner_did_not_report_peak_metal_memory`, respectively. Unavailable values are not
reported as zero.

### Closed custody

| Artifact | SHA-256 |
|---|---|
| `benchmark/benchmark.json` | `66f26f41d8e7f4bf078e93d71aef577081a2fd5312e1cf1e8200d2d5c2fcf087` |
| `benchmark/index.json` | `6fb9bd4d228b2caa139724441b2b6778772c263389f927ffaaf09bd9b03e532b` |
| `benchmark/receipt.json` | `b454163e255cfe85d7412c8a90bcca31ce0840aa23675bb4b2e4ccf9174a61a6` |

The receipt terminal state is `closed`; it closes the report, index, and 90 per-run
files (92 files total, excluding the receipt itself). The published bundle contains
93 files: 14 parsed drafts, four digest-safe invalid-output records, and the complete
requests, verified artifacts, communication receipts, and runner measurements.

The observed result does not change any safety gate. This remains synthetic,
non-clinical research evidence only, and every clinical, diagnosis, treatment,
protein-ranking, therapeutic-target, and druggability boundary remains **NO-GO**.
