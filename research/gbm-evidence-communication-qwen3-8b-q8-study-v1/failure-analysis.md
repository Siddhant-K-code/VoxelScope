# Post-hoc failure analysis

## Status

This analysis was performed at `2026-10-03T09:37:18Z`, after the study had completed
and the benchmark receipt had closed. It was not preregistered and is not a new study
result.

> **Headline:** Highly repeatable, correctly cited output failed closed because of a
> systematic construction/verifier mismatch; `4 / 18` runs additionally violated
> the outer envelope type.

The analysis does not modify or reinterpret the frozen declaration, generated
benchmark tree, compiler-defined metrics, thresholds, or terminal states. The study
ran once and must never be rerun, retrofitted, or reinterpreted.

## Closed evidence

| Artifact | SHA-256 |
|---|---|
| `benchmark/benchmark.json` | `66f26f41d8e7f4bf078e93d71aef577081a2fd5312e1cf1e8200d2d5c2fcf087` |
| `benchmark/index.json` | `6fb9bd4d228b2caa139724441b2b6778772c263389f927ffaaf09bd9b03e532b` |
| `benchmark/receipt.json` | `b454163e255cfe85d7412c8a90bcca31ce0840aa23675bb4b2e4ccf9174a61a6` |

The receipt remains closed. No file under `benchmark/` and no byte of
`study-declaration.json` was changed for this analysis.

## Observed failure shape

Four runs preserved digest-safe `invalid_json_type` records. The other 14 runs
produced parsed envelopes with an identical count shape:

| Per parsed run | Count |
|---|---:|
| Proposed claims | 17 |
| Verified claims | 13 |
| Excluded claims | 4 |

Across the 14 parsed runs, `56 / 238` proposed claims were excluded:

| Exclusion reason | Count |
|---|---:|
| `requirement_binding_mismatch` | 28 |
| `prohibited_causal_or_certainty_language` | 14 |
| `prohibited_target_or_druggability_language` | 14 |

Citation validity was `342 / 342 = 1.0`. All 18 terminal states remain `refused`;
that does not mean all 18 outputs hallucinated, and it is not evidence of
nondeterminism.

## Construction/verifier conflicts

### Separate plan entries versus combined verifier bindings

Each committed request exposes `fact:directional_assessment` and
`caveat:directional_scope` as separate plan entries. It likewise exposes the
source-target evidence fact and `caveat:source_target_boundary` as separate plan
entries. The runner prompt says:

> Produce one typed claim for every plan requirement. A single exact claim may bind
> multiple requirement IDs when the plan uses the same claim type and source IDs.

The model followed the literal separate-entry construction and emitted one typed
claim for each requirement. The pinned verifier's
`_expected_requirements_for_claim` instead requires:

- one `directional_assessment` claim bound to both
  `fact:directional_assessment` and `caveat:directional_scope`; and
- one `source_target_evidence` claim bound to both its evidence fact and
  `caveat:source_target_boundary`.

The separately bound model claims therefore produced exactly two
`requirement_binding_mismatch` exclusions per parsed run. This is a compiler/model
interface conflict, not evidence that the model omitted the two plan entries.

### Safe negations versus substring prohibition

The plan itself asks for the safe caveats:

- “Agreement or disagreement does not establish truth, causality, clinical utility,
  or therapeutic relevance.”
- “Source-reported target evidence is not a VoxelScope target, ranking, or
  druggability claim.”

The model emitted those negations. The pinned `_prohibited_reason` implementation
checks lower-cased substrings and rejects `" caus"` and `"druggab"` without negation
semantics. It therefore classified the requested safety caveats as
`prohibited_causal_or_certainty_language` and
`prohibited_target_or_druggability_language`, one each per parsed run.

These are verifier/interface conflicts. The observed sentences negate causal and
druggability claims; they are not unsafe positive claims by the model.

## Repeatability

All nine within-case repeat pairs were byte-identical at the model-outcome level.
For parsed pairs, the committed `draft.json` SHA-256 values match exactly. For
invalid pairs, the raw-output SHA-256 and byte length match exactly.

| Case | Outcome custody identity | Pair identical |
|---|---|---|
| `cross-layer-disagreement` | draft `3099b3250fcdd9805c66b062fb922770f20b241a6e4dceb88290e3535941c8f3` | yes |
| `missing-modality` | raw output `7b1032d646567d5ee81e0814c00a5848b87a331637ea209c88feb6b2503cd1b6`, 9,629 bytes | yes |
| `not-measured-versus-negative` | draft `3decbc43414cedf8a9dbeb9b33d7553d4420af256c9f1407f19dfc07453ae059` | yes |
| `prohibited-clinical-causal` | draft `4a3337892012fc24fec12c763ef8cfb6da478439b4fd995ee1412dd77d7f737a` | yes |
| `restricted-evidence` | raw output `e768e6ff2b7d7b5a82ef48acc28b2ab18ea7d75a81aa3b0cfee291933d19d93a`, 9,773 bytes | yes |
| `source-reported-target-evidence` | draft `4734274240696eb9a94f8ed8893bc990d10fe29c9f427abcfccee6c6d41f5c45` | yes |
| `stale-evidence` | draft `1b3fc7d685a0e694cab1a4acf1355ffcdd87476486c774f9afc9d49a9084e0b7` | yes |
| `supported-agreement` | draft `180c239009820a8dc58b53fb53797b9e54bf191fe3aaacc7c598f1d51a629f2a` | yes |
| `unsupported-join` | draft `d8702d16abbb672641e1b48bc9cf88fa5a5ababb3670ac05ed56dd34bb5d0064` | yes |

This is stronger than the compiler-defined semantic variance result of
`0 / 9 = 0.0`: the observed failure was perfectly repeatable.

The four invalid outputs cannot be interpreted further. The custody contract
intentionally preserves only their `invalid_json_type` error, raw-output digest, and
byte length; it does not preserve raw model text.

## Proposed next intervention — not executed

A separate future study could test this bounded interface change:

1. Require explicit Ollama JSON Schema **object** output rather than generic JSON
   mode.
2. Provide deterministic pre-grouped claim skeletons and binding IDs so the model
   cannot split a multi-requirement verifier unit.
3. Generate canonical safe caveat prose in trusted code rather than asking the model
   to repeat text that a lexical safety filter rejects.
4. Freeze a new prospective declaration and run it with the same pinned
   `qwen3:8b-q8_0` manifest
   `e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130`
   and Ollama `0.35.1`.

This proposal was not implemented or executed. It does not authorize a rerun of this
study, alter the existing result, or change any clinical, ranking,
therapeutic-target, or druggability **NO-GO** boundary.
