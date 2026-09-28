# Milestone 4 structural readiness v1

## Result

The one approved OpenNeuro acquisition attempt completed. All eight pinned artifacts matched their expected sizes and source-published content identities. The five medical files passed the frozen bounded NIfTI structure, finite-value, mask-domain, and exact cross-file geometry predicates.

The result is **structural GO** and **inference NO-GO**.

Structural GO means only that the exact private files are mechanically admissible for the next reviewed research stage. It does not establish anatomical correctness, segmentation accuracy, diagnostic validity, clinical safety, patient-specific validity, preprocessing equivalence, or source-label equivalence.

## Public evidence

The closed sanitized bundle is [`research/milestone-4`](../research/milestone-4/).

- Public bundle root SHA-256: `358ab0cf2d9c7f11d9a60d310406845d0c5bb721cf3e091ff6821997d8d94a5e`.
- Verified artifacts: 8.
- Validated medical files: 5.
- Exact geometry comparisons: 4.
- Failed admission gates: 0.
- Inference authorized: false.

The bundle includes deterministic synthetic positive and negative NIfTI fixtures so the validator behavior can be reproduced without medical data. It does not include medical bytes, private paths, receipt content, real geometry values, locally derived private hashes, clinical metadata, or participant crosswalks.

## Private evidence

Owner-private evidence binds:

- the decision and plan identities;
- the one-shot approval and durable attempt record;
- all eight acquired artifacts;
- source size, version, ETag, and SHA-256 verification;
- bounded NIfTI records for four modalities and one mask;
- exact cross-file geometry;
- the custody receipt, structural report, and terminal completion record.

Private evidence is not committed or copied into public artifacts.

## Remaining blockers

Before model loading or inference:

1. Define and independently review the preprocessing adapter.
2. Bind channel stacking in T1c, T1, T2, and FLAIR order.
3. Review source N4 correction against bundle nonzero channel-wise normalization.
4. Keep source labels `0/1/2/3` distinct from model output labels `0/1/2/4`; no remap is authorized.
5. Resolve or explicitly bound model-training overlap.
6. Pin the runtime, host, GPU, driver, budget, and stop controls.
7. Require a new explicit operator start action.

No model was extracted or loaded. No inference, GPU use, cloud provisioning, or spend occurred.
