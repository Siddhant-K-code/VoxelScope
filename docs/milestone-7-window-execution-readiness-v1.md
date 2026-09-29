# Milestone 7 private window execution readiness v1

## Decision

The private window-execution adapter is **IMPLEMENTATION-READY**. Private execution and real-data inference remain **NO-GO**.

This milestone adds the offline adapter, strict records, synthetic tests, and sanitized public evidence needed for a later explicitly approved qualification of the exact Milestone 5 preprocessing snapshot. No private custody root, preprocessing report, tensor, medical value, geometry, affine, orientation, tensor identity, or real window count was read during this milestone.

## Bound public identities

- Milestone 5 adapter plan SHA-256: `74d0214092b37502911b334fa11380cfd077aa10277425e532e6e861377f519f`
- Milestone 5 public bundle SHA-256: `08c0e0fe5871582b19fbad0887625e121c8371187449a4a7abbe7b1d97a524cf`
- Milestone 6 plan SHA-256: `e50352f949a7d87470adf186f729861c770a99e4bc2b4fdb49eae490b101d048`
- Milestone 6 report SHA-256: `381083f12182689057fac91c804117e6cd23c7dd6ebbcae0a64f5ff9b07d645f`
- Milestone 6 public bundle SHA-256: `76895b2af591364156ac021ad482d2ed72fd96ad62f33042a4545238c587caa4`
- Legacy synthetic bundle SHA-256: `c7f42adb2ec3d3f6ea51a7574e7dfc03e5198954ef6e25b6e41fe1d2517bd15c`
- Milestone 7 adapter plan SHA-256: `a9a6d123028269e534b48fd1bfe073972e2d4e4c338cbacde98aa6eb8cb35a52`
- Milestone 7 public bundle SHA-256: `3625597508b326497e31688c73ed1deff481fb198a36923f2fb44c921239b2f6`

Private Milestone 5 report, snapshot, receipt, tensor, and approval identities are deliberately absent from every committed artifact. A later operator supplies the private root and approved report and snapshot digests through a bounded owner-private authorization file descriptor, not command-line arguments. They remain inside owner-private authorization, attempt, report, and terminal records.

## Execution contract

The adapter consumes one little-endian float32 tensor in unchanged `(C,I,J,K)` layout with the role order T1c, T1, T2, and FLAIR. It maps positions unchanged to `(N,C,D,H,W)` as `I -> D`, `J -> H`, and `K -> W`. It permits no transpose, reorientation, resampling, affine relabeling, or anatomical-direction claim.

The frozen window contract is:

- ROI `(240,240,160)`;
- overlap `1/2` and stride `(120,120,80)`;
- symmetric floor-before, ceil-after constant-zero padding;
- lexicographic `I,J,K` traversal with `K` changing fastest;
- final anchors ending at the padded extent;
- constant blending and crop to original positional extents;
- source label excluded.

The legacy high-side-only padded engine is not called and remains **NO-GO** for general private window materialization.

## Independent materialization

Every window is produced twice:

1. Materialize the complete symmetrically padded channel-first tensor, then slice by the approved padded coordinates.
2. Allocate one zeroed window, compute its direct source intersection independently, and place source values at the derived local offset.

The adapter requires byte-identical C-order float32 windows from both paths. It streams one pair at a time into private coordinate and byte-stream digests and does not retain window files.

The closed private ledger verifies final anchors, per-axis start identities, padded extent, local padding placement, source intersections, finite values, complete source coverage, crop geometry, channel order, and label exclusion.

## One-shot and terminal behavior

The attempt identity binds the public Milestone 7 plan approval and the private approved Milestone 5 report and snapshot identities. The private CLI reads those values and the private root from a bounded regular owner-only descriptor or pipe so they do not enter process arguments. The adapter writes the attempt marker before any private report or tensor read. Replay, overwrite, symlink or special-file paths, unsafe permissions, source replacement, plan/runtime/implementation drift, nonfinite input, zero channels, materialization disagreement, evidence tamper, and publication faults all fail closed.

Only one terminal path is used per attempt. A successful attempt records a typed completion. Any owned failed attempt records a typed refusal. A completion publication fault cannot be reported as success.

Windows private ACL enforcement remains fail-closed until restrictive ACL behavior is independently implemented and verified.

## Synthetic evidence

Eight success fixtures cover asymmetric non-cubic geometry, odd and even symmetric padding, exact ROI geometry, undersized axes, final anchors, `K`-fastest traversal, and channel/role preservation. Three refusal fixtures cover empty extent, zero channels, and nonfinite values. Additional tests cover deliberate path disagreement, source replacement, plan/runtime/implementation drift, symlinks, replay, output tamper, terminal fault injection, Windows ACL refusal, path-redacted CLI errors, closed-bundle tampering, deterministic rebuilding, and preservation of historical bundle identities.

The committed Milestone 7 bundle contains only synthetic statuses and counts plus public source identities. It excludes real shape, real window count, affine, orientation, dtype observations, statistics, tensor or window hashes, subject locators, receipt hashes, approval identities, and private paths.

## Next gate

The next gate is a separately reviewed and explicitly approved offline run of this adapter against the exact private Milestone 5 snapshot. That authorization does not authorize model extraction, model loading, inference, GPU or MPS use, cloud provisioning, network access, or spend.

Real inference remains blocked on source-label versus model-output semantic review, subject-overlap resolution or an output-preservation-only bound, runtime and host identities, budget approval, and an explicit inference start action.
