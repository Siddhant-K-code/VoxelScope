# Milestone 6 positional window bridge readiness v1

## Decision

The public positional bridge is **GO**. Real-data inference remains **NO-GO**.

This milestone proves how the unchanged preprocessing tensor positions `(C,I,J,K)` map into the pinned model input positions `(N,C,D,H,W)`. It does not assign anatomical directions, read private tensors, load a model, run inference, use a GPU, provision cloud resources, or authorize spend.

## Bound identities

- MONAI version: `1.4.0`
- MONAI commit: `46a5272196a6c2590ca2589029eed8e4d56ff008`
- PyTorch version: `2.4.0`
- PyTorch commit: `d990dada86a8ad94882b5c23e859b88c0c255bda`
- Model-zoo commit: `5370ce6ea1dd132856b9c92e2fa125548594835d`
- Model configuration SHA-256: `22dead767f9bfcd6e3e7cbf5353ef1641e52f37acd9a20cc346361c1e603cf9e`
- Milestone 5 public bundle SHA-256: `08c0e0fe5871582b19fbad0887625e121c8371187449a4a7abbe7b1d97a524cf`
- Legacy synthetic bundle SHA-256: `c7f42adb2ec3d3f6ea51a7574e7dfc03e5198954ef6e25b6e41fe1d2517bd15c`

The plan additionally pins exact Git blob, SHA-256, size, path, commit, release, and immutable raw URL identities for the MONAI inference helpers, MONAI data utilities, `SlidingWindowInferer`, the model inference configuration, and PyTorch padding implementation. It also pins the local bridge, record, and legacy window implementation identities.

## Proven positional contract

| Preprocessing | Window | Model | ROI |
|---|---|---|---:|
| `I` at spatial position 0 | `S0` | `D` at tensor position 2 | 240 |
| `J` at spatial position 1 | `S1` | `H` at tensor position 3 | 240 |
| `K` at spatial position 2 | `S2` | `W` at tensor position 4 | 160 |

The mapping permits no transpose, reorientation, resampling, or affine-derived axis relabeling. `D`, `H`, and `W` are model tensor positions, not anatomical direction claims.

The verified window contract uses:

- zero-based half-open coordinates;
- exact overlap `1/2`;
- stride `(120,120,80)`;
- lexicographic `I,J,K` traversal with `K` changing fastest;
- final anchors that end the final window at the padded extent;
- constant blending and sliding-window batch size one;
- symmetric constant-zero padding for MONAI inputs;
- crop back to the original positional spatial extent.

Four non-cubic synthetic cases cover asymmetric components, a non-stride-aligned final anchor, an undersized volume, and the historical synthetic fixture. The implementation is checked against an independent oracle for coordinates, coverage, final anchors, traversal, padding geometry, crop geometry, and identity reconstruction.

## Legacy boundary

The historical VoxelScope engine uses synthetic `(Z,Y,X)` names and high-side-only zero padding. Its unpadded coordinate enumeration is positionally compatible with the pinned MONAI traversal. Its general padded extraction is not equivalent to MONAI's symmetric padding.

The legacy synthetic bundle remains valid because its only padded difference is one voxel, which produces zero low-side padding and one high-side padding under both paths. That narrow preservation result does not authorize general legacy padded reuse.

## Public evidence

- Plan: [`research/window-bridge-plan-v1.json`](../research/window-bridge-plan-v1.json)
  - SHA-256 `e50352f949a7d87470adf186f729861c770a99e4bc2b4fdb49eae490b101d048`
- Report: [`research/window-bridge-report-v1.json`](../research/window-bridge-report-v1.json)
  - SHA-256 `381083f12182689057fac91c804117e6cd23c7dd6ebbcae0a64f5ff9b07d645f`
- Closed bundle: [`research/milestone-6`](../research/milestone-6)
  - SHA-256 `76895b2af591364156ac021ad482d2ed72fd96ad62f33042a4545238c587caa4`

The public bundle contains no private path, source subject locator, medical value, tensor hash, real shape, affine, orientation, dtype observation, spacing, zoom, or real window count.

## Remaining blockers

1. Build and independently review a private window-execution adapter that uses the symmetric positional contract and emits a closed private ledger without loading the model.
2. Resolve source-label versus model-output semantic equivalence, or keep the label excluded.
3. Resolve model-training overlap or approve an output-preservation-only bound.
4. Pin the runtime image, host, GPU, CUDA, driver, and library identities.
5. Approve provider, region, instance, network policy, budget, and explicit start action.

Milestone 6 is an admission result, not a segmentation result. Inference authorization remains false.
