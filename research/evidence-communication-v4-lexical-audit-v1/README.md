# Observed v4 lexical-exclusion audit

## Status

This directory is a standalone, reproducible **post-hoc mechanical audit** of the
108 lexical exclusions in the immutable observed v4 study. It is not part of that
frozen study, does not alter its benchmark facts or terminal outcomes, and is not a
clinical, biomedical, model-quality, or model-safety evaluation.

The source study remains
`research/gbm-evidence-communication-qwen3-8b-q8-study-v4/`, whose current `main`
Git tree is `328d8785b475dcd9dbf0b7a62dd4553542b335f2`. The audit independently
resolves and verifies that actual tree, checks its 101 canonical Git
`{path, mode, blob}` entries against manifest SHA-256
`da50890729426b979e36efff26d5a515152a9d2f90851960bdb74b16608f2f4c`,
requires a clean index, and hashes each checked-out file through Git's
path-specific normalization before analysis. This accepts LF/CRLF-equivalent text
checkouts while still refusing changed text, any changed binary byte, symlinks,
and missing, extra, staged, or untracked files. The declaration, benchmark report,
index, receipt, every receipt-closed run file, and a complete offline v4 replay
are then verified before draft text is read.

## Method

No model or subjective reviewer classified these entries. The deterministic
classifier:

1. verifies the frozen study tree and known top-level digests;
2. rebuilds the committed synthetic atlas in a temporary directory;
3. replays all 18 v4 runs offline through the existing compiler/verifier;
4. joins each artifact exclusion to the exact receipt-closed draft, request,
   skeleton, and run;
5. reproduces the v4 first-match lexical label and records both winning and
   overlapping matched terms;
6. applies three ordered context rules:
   - explicit negation of a definitive diagnosis or conclusion;
   - an exact affirmative diagnostic/treatment **process** description;
   - otherwise, a hedged, investigative, or context-dependent use.

The canonical [`audit.json`](audit.json) stores no raw draft prose. Each of its 108
entry records stores the draft-text SHA-256 and size, run/case/request/skeleton
identity, source record digests, claim context, verifier label, matched terms,
deterministic context features, category, and reason code. Its SHA-256 is
`a5735aad10d496950429282613cf15f04bb00bef7f773eda874979224c5f8573`.
The audit implementation identity is its Git-normalized blob identity rather than
an operating-system-dependent checkout-byte digest.

## Recomputed findings

The observed benchmark facts remain distinct from the post-hoc categories:

| Observed v4 benchmark fact | Recomputed result |
|---|---:|
| Returned/matched tasks | `270 / 270` |
| Unknown / duplicate / omitted tasks | `0 / 0 / 0` |
| Invalid model outputs | `0 / 18` |
| Lexical exclusions | `108 / 270` across `12` runs |
| Verifier labels | `84` diagnostic/prognostic; `15` treatment; `9` causal/certainty |
| Emitted fact coverage | `42 / 158` |
| Emitted caveat coverage | `60 / 148` |
| Verified-draft fact coverage | `84 / 158` |
| Verified-draft caveat retention | `102 / 148` |
| Citation validity | `192 / 192` |
| Semantic variance | `3 / 9` repeat pairs |
| Terminal outcomes | `6` accepted; `0` partially excluded; `12` refused |
| Mean latency | `46526.308546279324 ms` |
| Mean input / output tokens | `1424.3333333333333 / 1389.111111111111` |

| Post-hoc mechanical category | Count |
|---|---:|
| `explicit_negation_or_boundary_disclaimer` | `12` |
| `affirmative_clinical_process_statement` | `4` |
| `ambiguous_or_context_dependent` | `92` |

No entry had both the explicit-negation and affirmative-process context features.
The overlapping raw substring features occurred more often than the winning v4
labels because the original verifier stops at the first reason: `diagnos` matched
84 entries, `recommend` 32, ` caus` 33, `definitive` 12, and `treatment` 3.
These counts explain gate behavior; they do not determine substantive safety truth.

## Reproduce

The verifier requires no Ollama process, model files, network, inference, or
biomedical data:

```bash
uv run python -m voxelscope.evidence_communication_v4_audit_cli verify \
  --repository-root .
```

To produce an independently comparable canonical record at a new path:

```bash
uv run python -m voxelscope.evidence_communication_v4_audit_cli recompute \
  --repository-root . \
  --output build/v4-lexical-audit.json
```

Both commands fail closed on Git tree/blob/index or worktree drift, missing or
extra study files, symlinks, receipt/digest or replay failure, schema drift,
changed canonical assertions, missing or extra audit fields, unclassified
contexts, and any expected count disagreement.
