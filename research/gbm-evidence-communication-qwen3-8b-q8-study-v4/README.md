# Qwen3 8B Q8_0 evidence communication study v4

## Status

This is the single observed execution of the prospectively frozen v4 local-model
Evidence Communication Compiler study. The declaration was merged in PR #20 at
commit `2bddf3355a88078a1346031a4209c818aa714c8a` before execution and remains
unchanged at SHA-256
`063b6055d430f14637d3dab05d1471cfc8e2fcf043b4030c633a956eff94a6c6`.

The immutable attempt was consumed once. Exactly 18 generation requests were made:
nine frozen synthetic cases with two fixture-defined repeats, no warmup, no retry,
no selective rerun, and no manual repair. The model was already installed; no model
or runtime was downloaded, pulled, created, removed, or changed.

This study is synthetic and non-clinical. It uses no real biomedical, patient, MRI,
credential, token, or private-path data and cannot support diagnosis, prognosis,
treatment, biological causality, protein ranking, therapeutic-target designation,
druggability, model-quality, safety, generalization, significance, superiority,
clinical utility, production readiness, or biomedical-validity claims.

## Frozen identity and execution

| Field | Observed execution |
|---|---|
| Study ID | `gbm-evidence-communication-qwen3-8b-q8-study-v4` |
| Declaration SHA-256 | `063b6055d430f14637d3dab05d1471cfc8e2fcf043b4030c633a956eff94a6c6` |
| Merged declaration commit | `2bddf3355a88078a1346031a4209c818aa714c8a` |
| Declared implementation commit/tree | `c66eca8f072ba23d74b59455f6f4074bfb758b08` / `f8a68afc0bb15f06f39870d15bed47edeca0ac37` |
| Model | `qwen3:8b-q8_0` |
| Full manifest SHA-256 | `e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130` |
| Runtime | Ollama `0.35.1` |
| Endpoint | redirect-free, unauthenticated `http://127.0.0.1:11434` |
| Request controls | exact JSON Schema; `think=false`; temperature `0`; `num_ctx=16384`; timeout `300` seconds; no seed |
| Attempt/request count | `1` attempt; `18` generation requests |
| Warmup/retry count | `0` / `0` |
| Preflight | two identity captures matched; `generation_requests=0` |
| Execution identity checks | one study-start check plus checks immediately before and after every request; all matched |
| Offline replay | verified after stopping the Ollama service |

The deterministic size preflight also matched the declaration: 15 skeletons per
request, a 1,492-byte canonical schema, prompts from 3,820 to 3,834 UTF-8 bytes,
and canonical request bodies from 5,567 to 5,581 bytes.

## Frozen endpoint results

These are descriptive compiler-defined observations, not estimates of model quality
or safety.

| Primary endpoint | Result |
|---|---:|
| Task/skeleton coverage | `270 / 270 = 1.0` |
| Unknown-task rate | `0 / 270 = 0.0` |
| Duplicate-task rate | `0 / 270 = 0.0` |
| Omitted-task rate | `0 / 270 = 0.0` |
| Invalid-model-output rate | `0 / 18 = 0.0` |
| Unsupported-claim rate | `108 / 270 = 0.4` |
| Emitted fact coverage | `42 / 158 = 0.26582278481012656` |
| Emitted caveat coverage | `60 / 148 = 0.40540540540540543` |
| Verified-draft fact coverage | `84 / 158 = 0.5316455696202531` |
| Verified-draft caveat retention | `102 / 148 = 0.6891891891891891` |
| Evidence-citation validity | `192 / 192 = 1.0` |
| Replay semantic variance | `3 / 9 = 0.3333333333333333` |
| Accepted terminal outcomes | `6 / 18 = 0.3333333333333333` |
| Partially excluded terminal outcomes | `0 / 18 = 0.0` |
| Refused terminal outcomes | `12 / 18 = 0.6666666666666666` |

| Secondary endpoint | Observed result |
|---|---:|
| Latency (ms) | `18` recorded; min `41521.08416600095`; mean `46526.308546279324`; max `50496.15258400445` |
| Input tokens | `18` recorded; min `1417`; mean `1424.3333333333333`; max `1432` |
| Output tokens | `18` recorded; min `1260`; mean `1389.111111111111`; max `1515` |
| Peak memory | unavailable: `runner_did_not_report_peak_memory` |
| Peak Metal memory | unavailable: `runner_did_not_report_peak_metal_memory` |

Unavailable memory values are not zero.

## Per-run terminal outcomes

All 18 responses satisfied the bounded outer model-output contract. Terminal
refusals below are deterministic compiler outcomes; they are not equivalent to
hallucination, unsafe behavior, or model failure.

| Case | Repeat 0 | Repeat 1 | Pair semantic identity |
|---|---|---|---|
| `supported-agreement` | `refused` | `refused` | different |
| `cross-layer-disagreement` | `refused` | `refused` | different |
| `missing-modality` | `refused` | `refused` | equal |
| `unsupported-join` | `accepted` | `accepted` | equal |
| `stale-evidence` | `refused` | `refused` | equal |
| `restricted-evidence` | `accepted` | `accepted` | equal |
| `not-measured-versus-negative` | `accepted` | `accepted` | equal |
| `source-reported-target-evidence` | `refused` | `refused` | equal |
| `prohibited-clinical-causal` | `refused` | `refused` | different |

## Historical v3 comparison

Only final emitted fact and caveat coverage retain matching definitions and
denominators. Their side-by-side values are descriptive only:

| Directly comparable field | Immutable v3 | Observed v4 |
|---|---:|---:|
| Emitted fact coverage | `0 / 158` | `42 / 158` |
| Emitted caveat coverage | `0 / 148` | `60 / 148` |

Invalid-output rate, repeat semantic variance, verified-draft coverage, terminal
outcomes, latency, and token counts are transformed fields because the prompt,
interface, schema, trusted expansion, or semantic projection changed. They must not
be presented as direct improvement. Skeleton metrics have no v3 counterpart.
Unsupported-claim rate and citation validity changed denominator or ownership and
are not comparable. Peak-memory fields remain unavailable in both studies and are
not comparable.

## Closed custody

The benchmark tree contains 94 files: the report, index, declaration binding,
receipt, and five files for each of 18 runs. The receipt closes 93 files, excluding
itself.

| Artifact | SHA-256 |
|---|---|
| `attempt.json` | `577ae65e77a4016361f9a709f21219acdd4e4612911635f338df6c3bf9d5abf4` |
| `benchmark/benchmark.json` | `86c1c86f62fda175dd121d84785bbb6ed5e11b216cb0383e9b7fcf0290fac409` |
| `benchmark/index.json` | `90f8dcd0002b52e4bdff89d00b0620b730d5ea7d5d5a137cdd3d7c1ff0ecadb3` |
| `benchmark/receipt.json` | `57347be8de8593a86e948a15c981890e747f684ac5a236b5a91305d712cae29d` |
| `benchmark/study-binding.json` | `77c20a37cd327a6e5f8f269e51cf3b9cfc50666623c3c2214decd36c6c0658de` |

The immediate offline replay reconstructed and verified the exact file set,
canonical JSON, requests, skeletons, trusted expansion, verifier outcomes, semantic
digests, aggregate metrics, index, declaration binding, and receipt. The attempt is
terminal and this study must never be rerun.

## Post-hoc analysis

After the unchanged observed result was frozen in commit
`4cc05da4fee06113fd2288b20b677aa43df2d637`, a separate bounded analysis was
derived from verified artifacts and closed metadata only. It is recorded in
[`post-hoc-analysis.md`](post-hoc-analysis.md) and
[`post-hoc-analysis.json`](post-hoc-analysis.json).

The analysis did not read, reproduce, or publish raw draft text. It found 108
deterministic lexical exclusions across the 12 refused runs: 84 diagnostic or
prognostic-language exclusions, 15 treatment-language exclusions, and nine causal
or certainty-language exclusions. All 18 outputs had complete, unique, ordered task
sets, so no refusal arose from missing, duplicate, unknown, or malformed model
output. Three repeat pairs had unequal verified semantic projections because their
excluded skeleton sets differed.

These post-hoc counts do not classify the underlying untrusted text as substantively
unsafe, do not alter any observed artifact or compiler-defined metric, and do not
support a model-quality, safety, or improvement claim.
