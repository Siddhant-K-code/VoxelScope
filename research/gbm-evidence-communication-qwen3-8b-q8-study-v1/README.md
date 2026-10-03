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
