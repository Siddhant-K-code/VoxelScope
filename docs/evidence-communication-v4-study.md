# v4 local-Qwen3 comparison study

## Frozen declaration and observed status

The prospective declaration is
[`study-declaration.json`](../research/gbm-evidence-communication-qwen3-8b-q8-study-v4/study-declaration.json),
SHA-256
`063b6055d430f14637d3dab05d1471cfc8e2fcf043b4030c633a956eff94a6c6`.
Its no-clobber receipt is
[`study-declaration-receipt.json`](../research/gbm-evidence-communication-qwen3-8b-q8-study-v4/study-declaration-receipt.json).
This declaration was frozen prospectively and remains unchanged. After PR #20
merged at commit `2bddf3355a88078a1346031a4209c818aa714c8a`, one fresh
declaration-bound execution consumed the immutable attempt and completed all 18
requests. No model or runtime was downloaded, pulled, created, removed, or changed.
The bounded observed result is recorded in the
[study README](../research/gbm-evidence-communication-qwen3-8b-q8-study-v4/README.md)
and
[`result-summary.json`](../research/gbm-evidence-communication-qwen3-8b-q8-study-v4/result-summary.json).

The declaration is based on merged PR #19 commit
`c66eca8f072ba23d74b59455f6f4074bfb758b08`, tree
`f8a68afc0bb15f06f39870d15bed47edeca0ac37`. PR #19 implementation head
`e11ecb9c701641e53ff44340cf6058b128177fd1` has the identical tree; the merge was a
squash commit, so the implementation head is not a commit ancestor of the merge.
Their merge base is the immutable v3 merge commit
`936549dcbf8c563cdef896bdb442a7629dcd260d`. All five PR #19 checks were successful
before this declaration was frozen.

## Model and transport identity

- tag: `qwen3:8b-q8_0`;
- full manifest SHA-256:
  `e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130`;
- Ollama: exactly `0.35.1`;
- endpoint: unauthenticated, redirect-free
  `http://127.0.0.1:11434`, with no network outside localhost;
- top-level `think=false`, non-streaming exact JSON Schema object format,
  temperature `0`, `num_ctx=16384`, timeout `300` seconds, and no seed;
- declared service settings: `OLLAMA_FLASH_ATTENTION=1` and
  `OLLAMA_KV_CACHE_TYPE=q8_0`.

The API verifies the runtime, unique local tag resolution, and full manifest at
study start and immediately before and after every generation. The service process
environment, host fields, and accelerator state remain declared-unverified because
the Ollama metadata API does not attest them.

The sanitized preflight performs exactly two pairs of `GET /api/version` and
`GET /api/tags`. On 2026-10-03 it matched all frozen identities and reported
`generation_requests=0`. It did not call `/api/generate`.

## Design and endpoints

The study uses only the frozen synthetic atlas SHA-256
`5a9f1a2078d951b3f10ff909b5c341e4071b5ed92050283c31859521b8c88647`
and v4 fixture SHA-256
`0db99c6ef56723df0cc18b4d692049cfb70aca196ee06601c72f99a2fb1ce0a8`.
It preserves the same nine conceptual cases with two fixture-defined repeats:
18 requests in one attempt, no warmup, no retry, no selective rerun, and no manual
repair.

Primary endpoints are frozen with exact numerators and denominators in the
declaration:

- task/skeleton coverage; unknown-, duplicate-, and omitted-task rates;
- invalid-model-output and unsupported-claim rates;
- emitted fact and caveat coverage;
- verified-draft fact coverage and caveat retention;
- evidence-citation validity and within-case replay semantic variance;
- separate accepted, partially-excluded, and refused terminal-outcome rates, where
  partially excluded maps exactly to artifact terminal state
  `accepted_with_exclusions`.

Secondary endpoints are request latency and input/output token counts when recorded.
Peak memory and peak Metal memory remain unavailable, never zero. The deterministic
size preflight measured, without generation, 15 skeletons per request, prompt sizes
from 3,820 to 3,834 UTF-8 bytes, a 1,492-byte canonical schema, and complete
canonical request bodies from 5,567 to 5,581 bytes. These are byte measurements and
token-estimation inputs, not observed token counters or token estimates.

## Comparator policy

The v3 study under
`research/gbm-evidence-communication-qwen3-8b-q8-study-v1/` is immutable historical
input only. Its frozen values remain 18 requests, 18 refusals, four invalid outer
JSON types, `342/342` citation validity, and `0/9` semantic variance. The declaration
pins its study tree, declaration, result summary, failure analysis, benchmark report,
and receipt identities.

Only emitted fact/caveat coverage is directly side-by-side comparable, and only
descriptively, because the required-plan denominators and final-prose meaning are
unchanged. Invalid-output rate and within-case semantic variance are transformed:
their request/pair denominators match, but the generated contract and semantic
artifact projection changed. Draft coverage, terminal outcomes, latency, and tokens
are also transformed because the interface or prompt changed. Skeleton task metrics
have no v3 field; unsupported-claim and citation-validity denominators or ownership
changed and are not comparable. No result may support a causal, model-quality,
safety, generalization, significance, or superiority claim.

## Stop, custody, and authorization rules

The output path and immutable attempt marker are fixed. Execution consumes the
attempt marker before initial identity validation. Any code, source, prompt,
compiler, verifier, fixture, atlas, model, runtime, tag, manifest, endpoint,
transport, redirect, protocol, `done`, response-model, thinking, schema, custody,
file-set, receipt, or replay drift aborts whole-tree publication.

Bounded malformed model content becomes only the declared digest-safe invalid-output
outcome and is never repaired or retried. A transport, protocol, custody, or replay
failure invalidates the comparison; it is not a model-quality observation. If any
generation is consumed but all 18 runs do not close and replay, the partial tree is
unpublished, the attempt marker remains terminal, and neither rerun nor partial
result publication is permitted. Successful publication is atomic, no-clobber,
receipt-closed, and must pass immediate offline replay.

The single authorized execution used the exact declaration-bound command:

```bash
uv run python -m voxelscope.evidence_communication_v4_cli benchmark \
  --atlas build/gbm-atlas/atlas.json \
  --fixtures research/gbm-evidence-communication-benchmark-v4/benchmark-fixtures.json \
  --output research/gbm-evidence-communication-qwen3-8b-q8-study-v4/benchmark \
  --runner ollama \
  --declaration research/gbm-evidence-communication-qwen3-8b-q8-study-v4/study-declaration.json \
  --repository-root . \
  --authorize-study gbm-evidence-communication-qwen3-8b-q8-study-v4 \
  --authorize-declaration-sha256 063b6055d430f14637d3dab05d1471cfc8e2fcf043b4030c633a956eff94a6c6
```

The attempt is consumed. Do not run this command again.

## Observed result

The frozen execution completed one attempt with exactly 18 generation requests,
zero warmup requests, zero retries, zero selective reruns, and zero manual repairs.
The immediate replay verified all 18 runs after the Ollama service was stopped.

- accepted: `6 / 18`;
- partially excluded: `0 / 18`;
- refused: `12 / 18`;
- task/skeleton coverage: `270 / 270`;
- invalid model output: `0 / 18`;
- emitted fact coverage: `42 / 158`;
- emitted caveat coverage: `60 / 148`;
- replay semantic variance: `3 / 9`.

Only emitted fact and caveat coverage are directly side-by-side comparable with v3,
and only descriptively. Every transformed and not-comparable field remains
classified exactly as frozen in the declaration. The result does not establish
causality, model quality, safety, generalization, statistical significance,
superiority, clinical utility, production readiness, or biomedical validity.

A separate
[post-hoc analysis](../research/gbm-evidence-communication-qwen3-8b-q8-study-v4/post-hoc-analysis.md)
was created only after the unchanged observed tree was committed. It uses verified
artifacts and closed metadata without reading or publishing raw draft text, does not
alter any observed outcome, and remains subject to the same claim boundary.

## Claim boundary

Research-only synthetic evidence. Clinical use, diagnosis, prognosis, treatment,
patient-specific claims, protein ranking, therapeutic-target designation,
druggability, biological causality, model-quality claims, production-safety claims,
real-data access, and biomedical-data acquisition remain **NO-GO**.
