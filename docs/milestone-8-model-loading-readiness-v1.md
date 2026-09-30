# Milestone 8 model-loading readiness v1

## Decision

The prospective archive extraction and CPU model-loading gate is **IMPLEMENTATION-READY**. Private extraction and real model loading remain **NO-GO** until a separate owner approval is bound to the final plan and reviewed private custody identities. Inference remains **NO-GO**.

Milestone 8 used repository evidence and public synthetic fixtures only. It did not access the private custody root, read a private receipt, open or extract the private official archive, deserialize the official checkpoint, instantiate or load the official model, allocate model input, call `forward`, use GPU/MPS, access a network, provision cloud resources, or authorize spend.

## Publicly trusted inputs

The repository already publicly pins:

- official MONAI `brats_mri_segmentation` bundle version 0.5.2;
- archive SHA-256 `6f7d21de3f0ce28deb332cb777ba30cc9a68f5bedaa18071c4fa7d2f7a147267` and size 35,082,630 bytes;
- checkpoint SHA-256 `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae`;
- inference config SHA-256 `22dead767f9bfcd6e3e7cbf5353ef1641e52f37acd9a20cc346361c1e603cf9e`;
- Apache-2.0 license member SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`;
- model-zoo commit `5370ce6ea1dd132856b9c92e2fa125548594835d`;
- MONAI commit `46a5272196a6c2590ca2589029eed8e4d56ff008`;
- PyTorch 2.4 source commit `d990dada86a8ad94882b5c23e859b88c0c255bda`;
- Milestone 7 plan SHA-256 `a9a6d123028269e534b48fd1bfe073972e2d4e4c338cbacde98aa6eb8cb35a52`;
- Milestone 7 public bundle SHA-256 `3625597508b326497e31688c73ed1deff481fb198a36923f2fb44c921239b2f6`.

The private custody root, receipt digest, filesystem path, and any future extraction snapshot, loading report, state key, tensor shape, tensor count, or observed result remain owner-private. They are absent from public evidence.

The final Milestone 8 public identities are:

- plan SHA-256 `7e4180c7f262d44a1ec18b67fd58773b0627beb9794182c15f7d679dec528243`;
- synthetic-only closed public bundle SHA-256 `92499bc9693ec59d165123e18b81fb0bb3697f6a0c8f5c81c14e11076442bf4d`.

## Extraction contract

The executor validates the entire closed 13-member ZIP manifest against already-public names, sizes, and hashes, but extracts only:

1. `brats_mri_segmentation/LICENSE`;
2. `brats_mri_segmentation/configs/inference.json`;
3. `brats_mri_segmentation/models/model.pt`.

It rejects path traversal, absolute or non-normalized names, duplicate or case-colliding names, directories, symbolic or hard-link representations, devices, special files, encryption, excessive member count, size, total expansion, compression ratio, identity drift, source replacement, and output overwrite. The archive is copied through a stable descriptor into owner-private immutable staging. Publication is atomic and no-clobber.

## Loading contract

The private worker runtime is exactly Linux, Python 3.12, PyTorch 2.4.0 at the pinned source commit, MONAI 1.4.0 at the pinned source commit, and CPU. A future authorization must bind both the interpreter and a separately reviewed closed runtime-distribution fingerprint. The worker verifies every file recorded by the installed PyTorch and MONAI distributions, validates module origins under the isolated runtime, and removes repository source paths before importing either framework. Its requirements artifact is separate from normal Python 3.13 validation, so ordinary CI does not install multi-hundred-megabyte frameworks or silently test a substitute version.

The worker:

- receives a bounded canonical request and writes a bounded canonical result through inherited descriptors;
- receives no private value through argv or environment;
- uses isolated Python, sanitized streams/environment, disabled accelerator visibility, one CPU thread, a 120-second deadline, a 2 GiB Linux address-space limit, and a network-refusing audit hook;
- parses the pinned JSON only as data and never evaluates targets, expressions, plugins, or checkpoint-provided code;
- constructs only reviewed `SegResNet(spatial_dims=3, init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2, blocks_down=(1,2,2,4), blocks_up=(1,1,1))`;
- uses `torch.load(..., weights_only=True, map_location="cpu", mmap=False)`;
- accepts only an exact top-level `model` mapping containing plain tensors;
- requires exact keys and tensor shapes, dtypes, strided layouts, and CPU devices against the constructed architecture;
- performs strict state loading and bounded finite-value inspection of every parameter and buffer;
- confirms the three-output-channel constructor contract and CPU-only placement;
- destroys references and exits without a forward call.

If the official checkpoint encoding is not accepted by PyTorch's safe weights-only loader, does not have the exact wrapper, or differs from the reviewed architecture, the correct private result is **NO-GO**. Unsafe pickle fallback and broad safe-global allowlisting are forbidden.

## One-shot private evidence

The deterministic attempt identity binds the public plan, private custody receipt, official archive, isolated Python executable, and reviewed runtime-distribution identities. The durable start marker precedes any approved private receipt or archive read. Replay and output overwrite are refused.

Exactly one terminal completion or refusal is written. Completion binds a closed extraction manifest, private loading report, and closed snapshot identity. Refusal records the last completed stage, extracted-member count, whether construction or state loading occurred, and whether publication occurred. Both paths state that forward, inference, and inference authorization are false.

Windows private execution fails closed because restrictive ACL enforcement is not implemented. The official qualification runtime is Linux.

## Synthetic evidence boundary

CI exercises positive and adversarial archive handling, descriptor bounds, subprocess isolation, timeout/crash handling, safe synthetic tensor mapping, key/shape/dtype/layout/device/finite checks, record closure, plan and implementation drift, replay, terminal behavior, public evidence determinism, privacy scans, and historical bundle preservation.

The synthetic checkpoint is canonical JSON tensor metadata. It is not a PyTorch checkpoint and does not claim real checkpoint compatibility. The committed public bundle states that extraction was not executed against the private archive; the official model was not deserialized, instantiated, or loaded; inference/forward, accelerator, network, cloud, and spend are false; a real private result is absent; and inference authorization is false.

## Next gate

After merge, the next action requires a new explicit owner approval bound to the final exact Milestone 8 plan SHA-256, reviewed private custody receipt/archive identities, exact isolated interpreter, and reviewed closed PyTorch/MONAI runtime fingerprint. That approval is limited to one offline CPU extraction-and-loading qualification and stops after introspection and cleanup. It does not authorize any model input or inference action.
