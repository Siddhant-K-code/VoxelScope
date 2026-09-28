# Preprocessing adapter v1

## Status

The milestone 5 adapter contract and implementation are complete on synthetic fixtures. Private execution is not authorized by this document. The adapter plan SHA-256 is `74d0214092b37502911b334fa11380cfd077aa10277425e532e6e861377f519f`.

The project remains **NO-GO** for inference. The adapter does not extract or load model weights, invoke a model, use a GPU or MPS device, access a network, provision cloud resources, or authorize spend.

The private execution approval prompt was unavailable on 2026-09-28. No preprocessing attempt marker was written, no private voxel array was processed, and no milestone 5 public result bundle was generated. This change is implementation-only and makes no real preprocessing GO or NO-GO claim.

## Pinned official sources

MONAI tag `1.4.0` is an annotated tag object `416ede8a845eff9294c7656f110c763dec1410fa` that resolves to commit `46a5272196a6c2590ca2589029eed8e4d56ff008`.

| Source | Git blob SHA-1 | File SHA-256 |
|---|---|---|
| [`monai/transforms/io/dictionary.py`](https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/transforms/io/dictionary.py) | `be1e78db8a4e7ead93e6f646a848f5bda51ed39a` | `eb833d294b0acb3c17a290dc6ab5d3ec9288bfdac73fc01ef9f611bc8d3206e3` |
| [`monai/transforms/io/array.py`](https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/transforms/io/array.py) | `4e71870fc9a290a38e22da6e21e2a54184bd24c2` | `cf868ec4ae38221ced37e2f31b6b913980f083fbb50ba81ca71f591fe661a150` |
| [`monai/data/image_reader.py`](https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/data/image_reader.py) | `b4ae562911cc1b852e9b231cfdb9c7a9b489b7f9` | `5c391855640dfde1e8308d6214558df08088d31f40c4a49cf1b1926af1ec91fb` |
| [`monai/transforms/intensity/dictionary.py`](https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/transforms/intensity/dictionary.py) | `5dbac485fe89a1d8a99cfc2e461d9ed873e0c332` | `45765e406f5d3ef76f8e88965a536461e738343581c585e37dcb787f2c2fd487` |
| [`monai/transforms/intensity/array.py`](https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/transforms/intensity/array.py) | `20000c52c40ba100a4caa0b9a9e8d45450b4022a` | `4a91b7bbd2cfb90c4393bee512f3c2e3eb47d6d88eb5a7030df7324327488642` |

The pinned model-zoo commit remains `5370ce6ea1dd132856b9c92e2fa125548594835d`. Its inference configuration Git blob is `a6d5f8878616ce469f3d885a5fb3a0dc2581ce8e`, and the verified NGC archive member SHA-256 is `22dead767f9bfcd6e3e7cbf5353ef1641e52f37acd9a20cc346361c1e603cf9e`.

## Observed source behavior

The following behavior is directly observed in the pinned official source:

1. `LoadImaged` delegates each dictionary value to `LoadImage`. Its default `dtype` is NumPy float32.
2. A list-valued image entry is passed to the reader in caller-provided order. `NibabelReader.read` iterates that sequence without sorting.
3. `NibabelReader` defaults `as_closest_canonical` to false. It does not reorient the image unless that option is explicitly true.
4. `NibabelReader` obtains data through `np.asanyarray(img.dataobj, order="C")`. Nibabel proxy scaling therefore occurs before `LoadImage` converts the stacked result to float32.
5. For channel-less images, `_stack_images` uses `np.stack(image_list, axis=0)`. The first image supplies metadata after compatible affine and spatial shape checks.
6. `NormalizeIntensityd` delegates to `NormalizeIntensity`. The array is converted to a tensor, the first axis is treated channel-wise, nonzero values are selected, and float32 mean and population standard deviation are used.
7. MONAI leaves an empty nonzero selection unchanged and replaces a zero divisor with one.
8. The pinned model inference configuration performs only `LoadImaged` and `NormalizeIntensityd(nonzero=true, channel_wise=true)` before inference. It contains no orientation or spacing transform.

The adapter intentionally tightens item 7. An empty channel or zero-variance nonzero selection is a refusal rather than a silent unchanged channel or divisor substitution.

## Declared assumptions

The adapter names the unchanged three Nibabel voxel-index axes `I,J,K`. The private tensor layout is `(C,I,J,K)`. The adapter does not transpose, reorient, resample, crop, or pad those axes. The source affine and complete milestone 4 geometry record remain bound to the output report. Mapping `(I,J,K)` into VoxelScope's synthetic `(Z,Y,X)` naming and traversal contract is a later explicit adapter gate and remains unresolved.

PyTorch and NumPy float32 reduction order is not assumed to be byte-identical. No exact-byte claim against a MONAI runtime is made. Exact equality between the two VoxelScope implementations is recorded, but the prospective acceptance rule is `rtol=0.000002` and `atol=0.00002`. This rule cannot be relaxed after private execution.

MONAI and PyTorch are not added as development or runtime dependencies. Pulling them only for conformance would add a large platform-sensitive dependency surface. The adapter instead uses the pinned official source as its semantic oracle, the existing pinned Nibabel dependency for scaled loading, and two independent NumPy control flows for verification.

Private execution is additionally pinned to Python 3.13.15, NumPy 2.5.3, Nibabel 5.4.2, little-endian byte order, and `uv.lock` SHA-256 `5bbe48a51f4a649af5370a87ace2e87e2a51f564f2633412b2ff94f1a7b2a71f`. The runtime identity is verified before the one-shot marker is consumed and is bound into the authorization and private report.

## Frozen adapter contract

The canonical plan is [`research/preprocessing-adapter-plan-v1.json`](../research/preprocessing-adapter-plan-v1.json).

1. Verify the exact milestone 4 custody snapshot, receipt, terminal record, content identities, and structural report before preprocessing.
2. Consume a durable one-shot preprocessing attempt marker before milestone 4 verification reads private voxel arrays.
3. Bind the authorization to both the exact adapter plan hash and exact private custody receipt hash.
4. Select image artifacts by role in T1c, T1, T2, and FLAIR order. Never derive channel order from filenames or directory traversal.
5. Exclude the label from both tensors. The source label values `0/1/2/3` remain distinct from model output values `0/1/2/4`.
6. Load scaled Nibabel proxy values in C order and convert the effective array to little-endian float32.
7. Require finite effective input. For each channel, calculate mean and population standard deviation over nonzero voxels only.
8. Refuse empty and zero-variance channels. Preserve every zero input voxel as exactly zero and require finite output.
9. Write owner-private, no-clobber C-order float32 tensors with layout `(C,I,J,K)`.
10. Bind the report to the decision, custody plan, custody receipt, structural report, authorization, attempt, role order, input identities, implementation identities, MONAI source identities, geometry identity, private shapes and dtypes, normalization statistics, output identities, comparison metrics, and no-model/no-inference states.
11. Persist exactly one terminal completion or refusal record. Replay is forbidden.

## Independent implementations and memory bounds

The reference implementation loads and converts one complete channel, uses direct NumPy float32 mean and standard deviation reductions, normalizes into a separate float32 channel, then releases it. It never holds all four source channels in memory.

The independent implementation scans depth chunks of eight source planes. It combines per-chunk float64 count, mean, and second moment with Welford equations, casts the final mean and population standard deviation to float32, then performs a separate chunked normalization pass. It does not call the reference normalization helper.

Both output tensors are disk-backed memory maps. Comparison scans at most 1,048,576 float32 values from each tensor at a time. Under the milestone 4 bounds, the reference arm can hold the raw or scaled full channel, one float32 effective channel, its nonzero selection, and one normalized channel. The independent arm holds only one eight-plane float32 chunk and its selected values, in addition to disk-backed outputs.

## Exact execution command

The command is intentionally unusable until the creator supplies the existing owner-private custody root and explicitly approves both hashes:

```bash
uv run voxelscope preprocess execute \
  --record research/one-volume-source-decision-v1.json \
  --custody-plan research/one-volume-acquisition-plan-v1.json \
  --adapter-plan research/preprocessing-adapter-plan-v1.json \
  --root <existing-owner-private-custody-root> \
  --approve-plan-sha256 74d0214092b37502911b334fa11380cfd077aa10277425e532e6e861377f519f \
  --approve-custody-receipt-sha256 <reviewed-private-custody-receipt-sha256>
```

The command performs no network request. It reads the trusted milestone 4 private receipt, terminal records, structural report, eight custody artifacts during trusted verification, and then only the four image artifacts for adapter execution. Each image is copied through a no-follow descriptor into the private staging directory while its size and SHA-256 are reverified. Both implementations consume the staged immutable copy, which is deleted before publication. The command writes one started marker, one completion or refusal terminal record, and one closed owner-private preprocessing snapshot containing the reference tensor, independent tensor, and private report.

The output must be verified with `voxelscope preprocess verify` using the same arguments before sanitized public evidence is generated. Private paths, receipt hashes, geometry, statistics, nonzero counts, tensor bytes, tensor hashes, and source subject locators must never enter public evidence.

Sanitized refusal evidence is not a generic NO-GO. The private refusal record enforces stage-specific channel-count and snapshot-publication invariants. Public refusal generation accepts only a closed mapping of known error codes to valid stages, derives a sanitized refusal phase and completed-channel count, and refuses unknown or stage-inconsistent codes. Public summaries separately report adapter, finite-output, geometry-preservation, label-exclusion, private-output, and terminal-evidence status. Private identities and exact error codes remain private.

The verifier prospectively pins the deterministic sanitized GO template and every allowed sanitized refusal-phase and channel-count template. These template identities do not report a real outcome. No milestone 5 public bundle exists until private completion or refusal evidence is explicitly authorized, produced, and verified.
