# Milestone 5 preprocessing readiness v1

## Result

One explicit offline approval executed the adapter plan with SHA-256 `74d0214092b37502911b334fa11380cfd077aa10277425e532e6e861377f519f` against the existing verified private custody.

The result is **preprocessing GO** and **inference NO-GO**.

The execution:

- used exactly four image roles in T1c, T1, T2, and FLAIR order;
- excluded the label from both tensors;
- preserved unchanged Nibabel voxel-index axes as `(C,I,J,K)`;
- applied scaled float32 loading and nonzero channel-wise population normalization;
- compared a direct float32 NumPy implementation with an independent chunked Welford implementation;
- remained within the preregistered tolerance;
- required finite outputs and exact zero-background preservation;
- published a closed owner-private snapshot and one completion record;
- did not load a model, run inference, use a GPU or MPS device, access a network, provision cloud resources, or spend money.

## Public evidence

The sanitized closed bundle is [`research/milestone-5`](../research/milestone-5/).

- Public bundle root SHA-256: `08c0e0fe5871582b19fbad0887625e121c8371187449a4a7abbe7b1d97a524cf`.
- Input channels: 4.
- Independent implementations: 2.
- Preprocessing adapter: GO.
- Finite output: GO.
- Geometry preservation: GO.
- Label excluded: GO.
- Private output: GO.
- Terminal evidence: GO.
- Inference authorized: false.

The public bundle does not include medical values, private evidence identities, private paths, tensors, tensor hashes, real geometry, normalization statistics, or the source subject locator.

## Remaining blockers

Before model loading or inference:

1. Define the mapping from private `(I,J,K)` voxel-index axes to VoxelScope synthetic `(Z,Y,X)` traversal.
2. Bind window coordinates, ROI extents, padding sides, and output inversion to that mapping.
3. Keep source labels `0/1/2/3` separate from model output labels `0/1/2/4`; no label comparison is authorized.
4. Resolve or explicitly bound model-training overlap.
5. Pin runtime image, model-loading mechanism, host, GPU, driver, budget, upload path, teardown, and stop controls.
6. Require a separate explicit operator action.

Preprocessing GO makes the private tensors eligible only for the next reviewed mapping stage. It is not an inference or clinical result.
