# Post-hoc analysis of the observed v4 study

## Status and evidence boundary

This analysis was created only after the unchanged observed result was frozen in
commit `4cc05da4fee06113fd2288b20b677aa43df2d637`.
The machine-readable `observed_result_recorded_at` value is the frozen result
timestamp, not an analysis timestamp; commit ordering establishes the post-hoc
sequence.

It uses only `benchmark.json`, verified `artifact.json` records, closed
`request.json` skeleton metadata, the index, and the receipt. It does not read,
reproduce, or publish model-owned `draft_text`. It does not alter the declaration,
attempt marker, generated benchmark tree, terminal outcomes, or compiler-defined
metrics.

The frozen observed identities are:

- study tree SHA-1:
  `5a062f9076e3ea41e881abac3682fdea089313c9`;
- benchmark tree SHA-1:
  `2322817495bdff1a3df3dff8dd1f7f0487f7e8ad`;
- benchmark report SHA-256:
  `86c1c86f62fda175dd121d84785bbb6ed5e11b216cb0383e9b7fcf0290fac409`;
- benchmark index SHA-256:
  `90f8dcd0002b52e4bdff89d00b0620b730d5ea7d5d5a137cdd3d7c1ff0ecadb3`;
- benchmark receipt SHA-256:
  `57347be8de8593a86e948a15c981890e747f684ac5a236b5a91305d712cae29d`.

## Bounded diagnosis

All 18 model responses satisfied the exact bounded task-set contract:

- `270 / 270` expected skeleton IDs matched;
- zero unknown, duplicate, or omitted skeletons;
- zero digest-safe invalid model outputs.

The deterministic verifier excluded 108 of 270 returned skeleton entries across the
12 refused runs:

| Exclusion reason | Count |
|---|---:|
| `prohibited_diagnostic_or_prognostic_language` | `84` |
| `prohibited_treatment_language` | `15` |
| `prohibited_causal_or_certainty_language` | `9` |

The exclusions affected 23 distinct skeleton IDs. Their frozen claim-type
distribution was:

| Claim type | Excluded entries |
|---|---:|
| `evidence_observation` | `56` |
| `availability` | `29` |
| `directional_assessment` | `6` |
| `entity_identity` | `6` |
| `source_target_evidence` | `6` |
| `non_clinical_boundary` | `5` |

Grouped skeletons can bind both a fact and a caveat, so the 108 excluded skeletons
removed 120 required-plan bindings: 74 fact bindings and 46 caveat bindings. Every
refused run therefore closed with missing required fact or caveat coverage after
deterministic exclusion. The six accepted runs had no exclusions.

## Repeat-pair differences

Six repeat pairs had equal semantic artifact identities. Three differed:

| Case | Excluded-skeleton symmetric difference |
|---|---:|
| `supported-agreement` | `8` |
| `cross-layer-disagreement` | `3` |
| `prohibited-clinical-causal` | `1` |

This accounts for the frozen compiler-defined semantic variance result of `3 / 9`.
It is an artifact-projection observation under the v4 verifier, not a general
nondeterminism or model-quality estimate.

## Interpretation limits

The verified artifacts disclose exclusion categories and skeleton identities, not
the raw text that triggered them. This analysis therefore does not determine whether
an exclusion reflects unsafe content, conservative lexical matching, phrasing
variation, or another text-level cause. It makes no causal, clinical, biomedical,
model-quality, safety, generalization, significance, superiority, or production
claim and proposes no rerun.
